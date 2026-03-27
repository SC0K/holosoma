"""
Augment retargeted OmniRetarget robot/object motion pairs.

Two modes:
1) Fast transform mode (metadata-level rigid transforms).
2) Interaction-mesh mode (default): re-runs InteractionMeshRetargeter by
   synthesizing demo-joint targets from source robot link trajectories,
   then applying object augmentation from the original pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import (
    build_retargeter_kwargs_from_config,
    create_task_constants,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.utils import (
    augment_object_poses,
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    load_object_data,
)


@dataclass
class AugmentRetargetedConfig:
    """Config for augmenting retargeted motions."""

    omniretarget_root: Path | None = None
    """Path to OmniRetarget root containing 'robot/' and 'object/' folders."""

    motion_name: str | None = None
    """Motion stem name, e.g. 'sub3_largebox_003_original'."""

    robot_npz: Path | None = None
    """Path to robot motion .npz (alternative to omniretarget_root+motion_name)."""

    object_npz: Path | None = None
    """Path to object motion .npz (alternative to omniretarget_root+motion_name)."""

    output_dir: Path = Path("augmented_omniretarget")
    """Output directory for generated variants."""

    robot: str = "g1"
    """Robot type used to build retargeter constants."""

    object_name: str = "largebox"
    """Object name used for default mesh/URDF resolution."""

    object_mesh_file: Path | None = None
    """Optional object mesh override (OBJ)."""

    object_urdf_file: Path | None = None
    """Optional object URDF override."""

    data_format: str = "smplh"
    """Demo-joint schema used by retargeter (default matches object_interaction)."""

    use_interaction_mesh_retarget: bool = True
    """If True, run robot->robot augmentation through InteractionMeshRetargeter."""

    include_original_copy: bool = True
    """If True, also writes an 'original' copy to output."""

    motion_translation_magnitude: float = 0.3
    """Translation magnitude (meters) for trajectory augmentation variants."""

    motion_rotation_degrees: tuple[float, ...] = (45.0, -45.0)
    """Initial yaw rotations (degrees) for trajectory augmentation variants."""

    object_size_scales: tuple[float, ...] = (0.75)
    """Object size multipliers for size augmentation variants."""

    movement_epsilon: float = 1e-3
    """Threshold for detecting first moving frame from object position deltas."""

    export_qpos_for_vis: bool = True
    """If True, also export merged qpos .npz for viser_player."""

    variant_limit: int | None = None
    """Optional limit on number of generated variants (for quick tests)."""


def _resolve_motion_paths(cfg: AugmentRetargetedConfig) -> tuple[Path, Path]:
    if cfg.robot_npz is not None and cfg.object_npz is not None:
        return cfg.robot_npz, cfg.object_npz
    if cfg.omniretarget_root is not None and cfg.motion_name is not None:
        robot_npz = cfg.omniretarget_root / "robot" / f"{cfg.motion_name}.npz"
        object_npz = cfg.omniretarget_root / "object" / f"{cfg.motion_name}.npz"
        return robot_npz, object_npz
    raise ValueError(
        "Provide either (--robot-npz and --object-npz) or (--omniretarget-root and --motion-name)."
    )


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return _normalize_quat_wxyz(out)


def _yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    return np.array([np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=np.float32)


def _rotz(yaw_rad: float) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _first_moving_frame(pos_xyz: np.ndarray, eps: float) -> int:
    if pos_xyz.shape[0] <= 1:
        return 0
    delta = np.linalg.norm(np.diff(pos_xyz, axis=0), axis=1)
    idx = np.where(delta > eps)[0]
    return int(idx[0] + 1) if idx.size > 0 else 0


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _save_npz_dict(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)


def _apply_world_transform_after_idx(
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
    pivot_xyz: np.ndarray,
    start_idx: int,
    rot_mat: np.ndarray,
    rot_quat_wxyz: np.ndarray,
    translation_xyz: np.ndarray,
    radial_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pos_new = pos_xyz.copy()
    quat_new = quat_wxyz.copy()
    if start_idx >= pos_xyz.shape[0]:
        return pos_new, quat_new

    rel = pos_xyz[start_idx:] - pivot_xyz[None, :]
    rel = radial_scale * rel
    pos_new[start_idx:] = (rel @ rot_mat.T) + pivot_xyz[None, :] + translation_xyz[None, :]
    quat_new[start_idx:] = _quat_mul_wxyz(np.broadcast_to(rot_quat_wxyz, quat_wxyz[start_idx:].shape), quat_wxyz[start_idx:])
    return pos_new, quat_new


def _build_qpos_for_vis(robot_vals: dict[str, np.ndarray], obj_pos: np.ndarray, obj_quat: np.ndarray) -> np.ndarray:
    dof_positions = robot_vals["dof_positions"]
    base_pos = robot_vals["body_positions"][:, 0, :]
    base_quat = robot_vals["body_rotations"][:, 0, :]
    t = min(dof_positions.shape[0], base_pos.shape[0], base_quat.shape[0], obj_pos.shape[0], obj_quat.shape[0])
    return np.concatenate(
        [base_pos[:t], base_quat[:t], dof_positions[:t], obj_pos[:t], obj_quat[:t]],
        axis=1,
    ).astype(np.float32)


def _variant_transforms(cfg: AugmentRetargetedConfig) -> Iterable[tuple[str, np.ndarray, float, float]]:
    mag = cfg.motion_translation_magnitude
    # yield ("trans_fwd", np.array([mag, 0.0, 0.0], dtype=np.float32), 0.0, 1.0)
    yield ("trans_left", np.array([0.0, mag, 0.0], dtype=np.float32), 0.0, 1.0)
    # yield ("trans_right", np.array([0.0, -mag, 0.0], dtype=np.float32), 0.0, 1.0)

    # for i, deg in enumerate(cfg.motion_rotation_degrees):
    #     yield (f"rot_{i}", np.zeros(3, dtype=np.float32), float(np.deg2rad(deg)), 1.0)

    if type(cfg.object_size_scales) == tuple:
        for s in cfg.object_size_scales:
            yield (f"size_{s:.2f}".replace(".", "p"), np.zeros(3, dtype=np.float32), 0.0, float(s))
    else:
        yield ("size", np.zeros(3, dtype=np.float32), 0.0, float(cfg.object_size_scales))



def _resolve_pkg_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return str(path_obj)
    pkg_root = Path(__file__).resolve().parents[1]
    return str(pkg_root / path_obj)


def _resolve_body_index(link_name: str, body_name_to_idx: dict[str, int]) -> int:
    if link_name in body_name_to_idx:
        return body_name_to_idx[link_name]

    alias = {
        "pelvis_contour_link": ["pelvis"],
        "left_ankle_intermediate_1_link": ["left_ankle_pitch_link", "left_ankle_roll_link"],
        "right_ankle_intermediate_1_link": ["right_ankle_pitch_link", "right_ankle_roll_link"],
        "left_ankle_roll_sphere_5_link": ["left_ankle_roll_link"],
        "right_ankle_roll_sphere_5_link": ["right_ankle_roll_link"],
        "left_rubber_hand_link": ["left_wrist_yaw_link", "left_wrist_roll_link"],
        "right_rubber_hand_link": ["right_wrist_yaw_link", "right_wrist_roll_link"],
        "left_sphere_hand_link": ["left_wrist_yaw_link", "left_wrist_roll_link"],
        "right_sphere_hand_link": ["right_wrist_yaw_link", "right_wrist_roll_link"],
    }
    for candidate in alias.get(link_name, []):
        if candidate in body_name_to_idx:
            return body_name_to_idx[candidate]

    raise KeyError(f"Could not map link '{link_name}' to source robot body_names.")


def _synthesize_demo_joints_from_robot(
    body_positions: np.ndarray,
    body_names: np.ndarray,
    demo_joints: list[str],
    joints_mapping: dict[str, str],
) -> np.ndarray:
    t = body_positions.shape[0]
    demo = np.zeros((t, len(demo_joints), 3), dtype=np.float32)
    body_name_to_idx = {str(name): i for i, name in enumerate(body_names.tolist())}

    for demo_joint_name, link_name in joints_mapping.items():
        if demo_joint_name not in demo_joints:
            continue
        di = demo_joints.index(demo_joint_name)
        bi = _resolve_body_index(link_name, body_name_to_idx)
        demo[:, di, :] = body_positions[:, bi, :]

    # Fill unresolved joints with pelvis/root if available (keeps shape stable).
    root_idx = demo_joints.index("Pelvis") if "Pelvis" in demo_joints else 0
    unresolved = np.where(np.linalg.norm(demo, axis=2).sum(axis=0) < 1e-12)[0]
    for j in unresolved:
        demo[:, j, :] = demo[:, root_idx, :]

    return demo


def _to_mujoco_object_pose_order(object_pose_quat_pos: np.ndarray) -> np.ndarray:
    # [qw, qx, qy, qz, x, y, z] -> [x, y, z, qw, qx, qy, qz]
    return object_pose_quat_pos[:, [4, 5, 6, 0, 1, 2, 3]].astype(np.float32)


def _build_robot_q_nominal(robot_vals: dict[str, np.ndarray], dof: int, nq: int, t: int) -> np.ndarray:
    q = np.zeros((t, nq), dtype=np.float32)
    base_pos = np.asarray(robot_vals["body_positions"], dtype=np.float32)[:t, 0, :]
    base_quat = _normalize_quat_wxyz(np.asarray(robot_vals["body_rotations"], dtype=np.float32)[:t, 0, :])
    dof_pos = np.asarray(robot_vals["dof_positions"], dtype=np.float32)[:t, :dof]
    q[:, :3] = base_pos
    q[:, 3:7] = base_quat
    q[:, 7 : 7 + dof] = dof_pos
    return q


def _write_split_outputs(
    out_robot_path: Path,
    out_object_path: Path,
    out_qpos_path: Path,
    src_robot_vals: dict[str, np.ndarray],
    src_object_vals: dict[str, np.ndarray],
    qpos: np.ndarray,
    object_scale: float,
) -> None:
    robot_out = {k: np.array(v, copy=True) for k, v in src_robot_vals.items()}
    object_out = {k: np.array(v, copy=True) for k, v in src_object_vals.items()}

    t = qpos.shape[0]
    dof = src_robot_vals["dof_positions"].shape[1]

    robot_out["dof_positions"] = qpos[:, 7 : 7 + dof].astype(np.float32)
    if "body_positions" in robot_out and robot_out["body_positions"].ndim == 3:
        bp = np.asarray(robot_out["body_positions"], dtype=np.float32)
        bp[:t, 0, :] = qpos[:, :3]
        robot_out["body_positions"] = bp
    if "body_rotations" in robot_out and robot_out["body_rotations"].ndim == 3:
        br = _normalize_quat_wxyz(np.asarray(robot_out["body_rotations"], dtype=np.float32))
        br[:t, 0, :] = qpos[:, 3:7]
        robot_out["body_rotations"] = br

    if qpos.shape[1] >= 7 + dof + 7:
        obj_pos = qpos[:, -7:-4].astype(np.float32)
        obj_quat = _normalize_quat_wxyz(qpos[:, -4:].astype(np.float32))
    else:
        obj_pos = np.asarray(object_out["object_position_xyz"], dtype=np.float32)[:t]
        obj_quat = _normalize_quat_wxyz(np.asarray(object_out["object_quat_wxyz"], dtype=np.float32)[:t])

    object_out["object_position_xyz"] = obj_pos
    object_out["object_quat_wxyz"] = obj_quat
    object_out["object_pose"] = np.concatenate([obj_quat, obj_pos], axis=1).astype(np.float32)
    object_out["object_scale"] = np.array([object_scale], dtype=np.float32)

    _save_npz_dict(out_robot_path, robot_out)
    _save_npz_dict(out_object_path, object_out)
    np.savez(out_qpos_path, qpos=qpos.astype(np.float32), fps=src_robot_vals.get("fps", np.array(30.0)))


def _run_interaction_mesh_variant(
    cfg: AugmentRetargetedConfig,
    retargeter: InteractionMeshRetargeter,
    source_robot_vals: dict[str, np.ndarray],
    source_object_vals: dict[str, np.ndarray],
    human_joints_synth: np.ndarray,
    toe_names: list[str],
    object_pose_quat_pos: np.ndarray,
    object_local_pts_base: np.ndarray,
    object_local_pts_demo_base: np.ndarray,
    local_translation: np.ndarray,
    rotation_initial: float,
    object_size_scale: float,
    output_qpos_path: Path,
) -> np.ndarray:
    t = min(
        human_joints_synth.shape[0],
        object_pose_quat_pos.shape[0],
        source_robot_vals["dof_positions"].shape[0],
    )
    human = human_joints_synth[:t]
    object_pose = object_pose_quat_pos[:t]

    move_idx = extract_object_first_moving_frame(object_pose[:, -3:])
    root_idx = retargeter.demo_joints.index("Pelvis") if "Pelvis" in retargeter.demo_joints else 0
    human_initial_root = human[0, root_idx, :]

    object_pose_aug = augment_object_poses(
        object_poses=object_pose,
        object_moving_frame_idx=move_idx,
        human_initial_root=human_initial_root,
        local_translation=local_translation,
        rotation_initial=rotation_initial,
    )

    object_pose_mj = _to_mujoco_object_pose_order(object_pose)
    object_pose_aug_mj = _to_mujoco_object_pose_order(object_pose_aug)

    q_nominal = _build_robot_q_nominal(source_robot_vals, retargeter.task_constants.ROBOT_DOF, retargeter.nq, t)
    q_init = q_nominal[0]

    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(human, retargeter.demo_joints, toe_names)
    if len(foot_sticking_sequences) > 0 and len(toe_names) >= 2:
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    object_local_pts = (object_local_pts_base * object_size_scale).astype(np.float32)
    object_local_pts_demo = (object_local_pts_demo_base).astype(np.float32)

    qpos, _, _, _ = retargeter.retarget_motion(
        human_joint_motions=human,
        object_poses=object_pose_mj,
        object_poses_augmented=object_pose_aug_mj,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        original=False,
        dest_res_path=str(output_qpos_path),
        fps=int(source_robot_vals.get("fps", np.array(30.0))),
    )

    return qpos


def _run_fast_transform_variant(
    source_robot_vals: dict[str, np.ndarray],
    source_object_vals: dict[str, np.ndarray],
    local_translation: np.ndarray,
    rotation_initial: float,
    object_size_scale: float,
    movement_epsilon: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    robot_new = {k: np.array(v, copy=True) for k, v in source_robot_vals.items()}
    object_new = {k: np.array(v, copy=True) for k, v in source_object_vals.items()}

    obj_pos = np.asarray(object_new["object_position_xyz"], dtype=np.float32)
    obj_quat = _normalize_quat_wxyz(np.asarray(object_new["object_quat_wxyz"], dtype=np.float32))
    move_idx = _first_moving_frame(obj_pos, movement_epsilon)
    pivot = obj_pos[move_idx].copy() if obj_pos.shape[0] > 0 else np.zeros(3, dtype=np.float32)

    yaw_q = _yaw_quat_wxyz(rotation_initial)
    rot_m = _rotz(rotation_initial)

    obj_pos_new, obj_quat_new = _apply_world_transform_after_idx(
        obj_pos,
        obj_quat,
        pivot,
        move_idx,
        rot_m,
        yaw_q,
        local_translation,
        radial_scale=object_size_scale,
    )

    object_new["object_position_xyz"] = obj_pos_new.astype(np.float32)
    object_new["object_quat_wxyz"] = obj_quat_new.astype(np.float32)
    object_new["object_pose"] = np.concatenate([object_new["object_quat_wxyz"], object_new["object_position_xyz"]], axis=1)
    object_new["object_scale"] = np.array([object_size_scale], dtype=np.float32)

    qpos = _build_qpos_for_vis(robot_new, object_new["object_position_xyz"], object_new["object_quat_wxyz"])
    return robot_new, object_new, qpos


def main(cfg: AugmentRetargetedConfig) -> None:
    robot_npz, object_npz = _resolve_motion_paths(cfg)
    if not robot_npz.exists():
        raise FileNotFoundError(f"robot npz not found: {robot_npz}")
    if not object_npz.exists():
        raise FileNotFoundError(f"object npz not found: {object_npz}")

    source_robot_vals = _load_npz_dict(robot_npz)
    source_object_vals = _load_npz_dict(object_npz)

    required_robot = {"dof_positions", "body_positions", "body_rotations", "body_names"}
    if not required_robot.issubset(source_robot_vals):
        missing = sorted(required_robot - set(source_robot_vals.keys()))
        raise KeyError(f"robot_npz missing keys: {missing}")

    if "object_quat_wxyz" not in source_object_vals or "object_position_xyz" not in source_object_vals:
        raise KeyError("object_npz must contain object_quat_wxyz and object_position_xyz")

    object_pose_quat_pos = np.concatenate(
        [
            _normalize_quat_wxyz(np.asarray(source_object_vals["object_quat_wxyz"], dtype=np.float32)),
            np.asarray(source_object_vals["object_position_xyz"], dtype=np.float32),
        ],
        axis=1,
    )

    base_name = robot_npz.stem
    robot_out_dir = cfg.output_dir / "robot"
    object_out_dir = cfg.output_dir / "object"
    qpos_out_dir = cfg.output_dir / "qpos"
    qpos_out_dir.mkdir(parents=True, exist_ok=True)

    retargeter = None
    toe_names: list[str] = []
    human_joints_synth = None
    object_local_pts = None
    object_local_pts_demo = None

    if cfg.use_interaction_mesh_retarget:
        task_cfg = TaskConfig(object_name=cfg.object_name)
        robot_cfg = RobotConfig(robot_type=cfg.robot)
        motion_cfg = MotionDataConfig(data_format=cfg.data_format, robot_type=cfg.robot)
        constants = create_task_constants(
            robot_config=robot_cfg,
            motion_data_config=motion_cfg,
            task_config=task_cfg,
            task_type="object_interaction",
        )

        if cfg.object_mesh_file is not None:
            constants.OBJECT_MESH_FILE = str(cfg.object_mesh_file)
        if cfg.object_urdf_file is not None:
            constants.OBJECT_URDF_FILE = str(cfg.object_urdf_file)

        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE is not set; provide --object-mesh-file or valid --object-name")
        if constants.OBJECT_URDF_FILE is None:
            raise ValueError("OBJECT_URDF_FILE is not set; provide --object-urdf-file or valid --object-name")

        retargeter_cfg = RetargeterConfig()
        retargeter_kwargs = build_retargeter_kwargs_from_config(
            retargeter_cfg,
            constants,
            constants.OBJECT_URDF_FILE,
            "object_interaction",
        )
        retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

        mesh_path = _resolve_pkg_path(constants.OBJECT_MESH_FILE)
        if mesh_path is None:
            raise ValueError("Failed to resolve object mesh path")
        object_local_pts, object_local_pts_demo = load_object_data(mesh_path, smpl_scale=1.0, sample_count=100)

        human_joints_synth = _synthesize_demo_joints_from_robot(
            body_positions=np.asarray(source_robot_vals["body_positions"], dtype=np.float32),
            body_names=np.asarray(source_robot_vals["body_names"]),
            demo_joints=motion_cfg.resolved_demo_joints,
            joints_mapping=motion_cfg.resolved_joints_mapping,
        )
        toe_names = motion_cfg.toe_names

    if cfg.include_original_copy:
        _save_npz_dict(robot_out_dir / f"{base_name}.npz", source_robot_vals)
        _save_npz_dict(object_out_dir / f"{base_name}.npz", source_object_vals)
        if cfg.export_qpos_for_vis:
            qpos = _build_qpos_for_vis(
                source_robot_vals,
                np.asarray(source_object_vals["object_position_xyz"], dtype=np.float32),
                _normalize_quat_wxyz(np.asarray(source_object_vals["object_quat_wxyz"], dtype=np.float32)),
            )
            np.savez(qpos_out_dir / f"{base_name}_vis_qpos.npz", qpos=qpos, fps=source_robot_vals.get("fps", np.array(30.0)))

    variants = list(_variant_transforms(cfg))
    if cfg.variant_limit is not None:
        variants = variants[: cfg.variant_limit]

    for tag, local_translation, rotation_initial, size_scale in variants:
        name = f"{base_name}_{tag}"
        qpos_path = qpos_out_dir / f"{name}_vis_qpos.npz"

        if cfg.use_interaction_mesh_retarget:
            assert retargeter is not None
            assert human_joints_synth is not None
            assert object_local_pts is not None
            assert object_local_pts_demo is not None

            print(f"[run] interaction-mesh variant: {name}")
            qpos = _run_interaction_mesh_variant(
                cfg=cfg,
                retargeter=retargeter,
                source_robot_vals=source_robot_vals,
                source_object_vals=source_object_vals,
                human_joints_synth=human_joints_synth,
                toe_names=toe_names,
                object_pose_quat_pos=object_pose_quat_pos,
                object_local_pts_base=object_local_pts,
                object_local_pts_demo_base=object_local_pts_demo,
                local_translation=local_translation,
                rotation_initial=rotation_initial,
                object_size_scale=size_scale,
                output_qpos_path=qpos_path,
            )
            _write_split_outputs(
                out_robot_path=robot_out_dir / f"{name}.npz",
                out_object_path=object_out_dir / f"{name}.npz",
                out_qpos_path=qpos_path,
                src_robot_vals=source_robot_vals,
                src_object_vals=source_object_vals,
                qpos=qpos,
                object_scale=size_scale,
            )
            print(f"[ok] wrote variant: {name}")
            continue

        robot_new, object_new, qpos = _run_fast_transform_variant(
            source_robot_vals=source_robot_vals,
            source_object_vals=source_object_vals,
            local_translation=local_translation,
            rotation_initial=rotation_initial,
            object_size_scale=size_scale,
            movement_epsilon=cfg.movement_epsilon,
        )

        _save_npz_dict(robot_out_dir / f"{name}.npz", robot_new)
        _save_npz_dict(object_out_dir / f"{name}.npz", object_new)
        if cfg.export_qpos_for_vis:
            np.savez(qpos_path, qpos=qpos, fps=source_robot_vals.get("fps", np.array(30.0)))

        print(f"[ok] wrote variant: {name}")

    print(f"Done. Output: {cfg.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(AugmentRetargetedConfig))
