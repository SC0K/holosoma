#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


LAFAN_BONES = (
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToeBase"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("Spine2", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Spine2", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
)


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def load_source_motion(
    npz_path: str,
) -> tuple[np.ndarray, list[str], np.ndarray | None, float]:
    """Load Z-up joints and optional object poses from a converted LAFAN NPZ."""
    with np.load(npz_path, allow_pickle=True) as data:
        if "global_joint_positions" not in data or "joint_names" not in data:
            raise KeyError(f"{npz_path} must contain global_joint_positions and joint_names")
        joints = np.asarray(data["global_joint_positions"], dtype=np.float32)
        joint_names = [str(name) for name in data["joint_names"].tolist()]
        object_poses = (
            np.asarray(data["object_poses"], dtype=np.float32) if "object_poses" in data else None
        )
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0
    if joints.ndim != 3 or joints.shape[2] != 3 or joints.shape[1] != len(joint_names):
        raise ValueError(
            f"Invalid source skeleton shape {joints.shape} for {len(joint_names)} joint names"
        )
    if len(joints) == 0 or not np.isfinite(joints).all():
        raise ValueError(f"Source skeleton is empty or contains non-finite values: {npz_path}")
    if object_poses is not None:
        if object_poses.shape != (len(joints), 7) or not np.isfinite(object_poses).all():
            raise ValueError(
                f"Invalid object_poses shape {object_poses.shape}; expected ({len(joints)}, 7)"
            )
    return joints, joint_names, object_poses, fps


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0) + 1e-12
    q1 /= np.linalg.norm(q1) + 1e-12
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / sin_theta


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    fps: int | None = None,
):
    """
    qpos layout (MuJoCo order):
      [0:3]   robot base position (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end-4] (optional) object position (xyz)
      [end-4:end]   (optional) object quat (wxyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer()
    actual_fps = fps if fps is not None else config.fps

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=config.show_object_axes)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_path = Path(config.robot_urdf).expanduser().resolve()
    robot_urdf_y = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    vo = None
    if config.object_urdf:
        object_urdf_path = Path(config.object_urdf).expanduser().resolve()
        object_urdf_y = yourdfpy.URDF.load(str(object_urdf_path), load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    skeleton_points = None
    skeleton_bones = None
    source_joints = None
    source_object_poses = None
    source_object = None
    source_fps = float(actual_fps)
    bone_indices: np.ndarray | None = None
    if config.source_motion_npz:
        source_path = Path(config.source_motion_npz).expanduser().resolve()
        source_joints, source_names, source_object_poses, source_fps = load_source_motion(
            str(source_path)
        )
        name_to_index = {name: index for index, name in enumerate(source_names)}
        available_bones = [
            (name_to_index[parent], name_to_index[child])
            for parent, child in LAFAN_BONES
            if parent in name_to_index and child in name_to_index
        ]
        if not available_bones:
            raise ValueError(f"No recognized LAFAN bones found in {source_path}")
        bone_indices = np.asarray(available_bones, dtype=np.int32)
        offset = np.asarray(config.source_skeleton_offset, dtype=np.float32)
        initial_joints = source_joints[0] + offset
        skeleton_points = server.scene.add_point_cloud(
            "/source_skeleton/joints",
            points=initial_joints,
            colors=(0, 220, 255),
            point_size=config.source_skeleton_point_size,
            point_shape="circle",
            visible=config.show_source_skeleton,
        )
        skeleton_bones = server.scene.add_line_segments(
            "/source_skeleton/bones",
            points=np.stack(
                [initial_joints[bone_indices[:, 0]], initial_joints[bone_indices[:, 1]]], axis=1
            ),
            colors=(0, 220, 255),
            line_width=config.source_skeleton_line_width,
            visible=config.show_source_skeleton,
        )
        if config.source_object_mesh:
            if source_object_poses is None:
                raise KeyError(f"{source_path} has no object_poses for the original object")
            source_object_path = Path(config.source_object_mesh).expanduser().resolve()
            source_mesh = trimesh.load(source_object_path, force="mesh", process=False)
            if not isinstance(source_mesh, trimesh.Trimesh) or source_mesh.is_empty:
                raise ValueError(f"Could not load source object mesh: {source_object_path}")
            source_object = server.scene.add_mesh_simple(
                "/source_object/bucket_stl",
                vertices=np.asarray(source_mesh.vertices, dtype=np.float32),
                faces=np.asarray(source_mesh.faces, dtype=np.uint32),
                color=(255, 145, 45),
                opacity=0.55,
                side="double",
                position=source_object_poses[0, 4:7] + offset,
                wxyz=source_object_poses[0, :4],
                visible=config.show_source_object,
            )

    # A tiny grid
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    # Set initial mesh visibility
    vr.show_visual = config.show_meshes
    if vo is not None:
        vo.show_visual = config.show_meshes

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=config.show_meshes)
        show_source_cb = None
        if skeleton_points is not None:
            show_source_cb = server.gui.add_checkbox(
                "Show original skeleton", initial_value=config.show_source_skeleton
            )
        show_source_object_cb = None
        if source_object is not None:
            show_source_object_cb = server.gui.add_checkbox(
                "Show original bucket", initial_value=config.show_source_object
            )

    @show_meshes_cb.on_update
    def _(_):
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    if show_source_cb is not None:

        @show_source_cb.on_update
        def _(_):
            visible = bool(show_source_cb.value)
            skeleton_points.visible = visible
            skeleton_bones.visible = visible

    if show_source_object_cb is not None:

        @show_source_object_cb.on_update
        def _(_):
            source_object.visible = bool(show_source_object_cb.value)

    def _update_source_skeleton(k0: int, _k1: int, u: float) -> None:
        if source_joints is None or bone_indices is None:
            return
        # Map the retargeted playback time to the source clip's frame rate.
        source_frame = ((float(k0) + float(u)) * source_fps / float(actual_fps)) % len(source_joints)
        s0 = int(np.floor(source_frame))
        s1 = (s0 + 1) % len(source_joints) if config.loop else min(s0 + 1, len(source_joints) - 1)
        alpha = float(source_frame - s0)
        joints = (1.0 - alpha) * source_joints[s0] + alpha * source_joints[s1]
        joints = joints + np.asarray(config.source_skeleton_offset, dtype=np.float32)
        skeleton_points.points = joints
        skeleton_bones.points = np.stack(
            [joints[bone_indices[:, 0]], joints[bone_indices[:, 1]]], axis=1
        )
        if source_object is not None and source_object_poses is not None:
            pose0 = source_object_poses[s0]
            pose1 = source_object_poses[s1]
            source_object.position = (
                (1.0 - alpha) * pose0[4:7]
                + alpha * pose1[4:7]
                + np.asarray(config.source_skeleton_offset, dtype=np.float32)
            )
            source_object.wxyz = _slerp(pose0[:4], pose1[:4], alpha)

    # ---------- Use reusable motion control sliders from viser_utils ----------
    create_motion_control_sliders(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        viser_object=vo if config.assume_object_in_qpos else None,
        object_base_frame=object_root if config.assume_object_in_qpos else None,
        contains_object_in_qpos=config.assume_object_in_qpos,
        initial_fps=actual_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
        frame_update_callback=_update_source_skeleton if source_joints is not None else None,
    )
    n_frames = int(qpos.shape[0])
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if (config.object_urdf and config.assume_object_in_qpos) else 'no'} | "
        f"source_skeleton={'yes' if source_joints is not None else 'no'} | "
        f"source_object={'yes' if source_object is not None else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
    qpos, fps = load_npz(cfg.qpos_npz)
    make_player(
        config=cfg,
        qpos=qpos,
        fps=fps,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
