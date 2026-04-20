from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco  # type: ignore[import-not-found]
import numpy as np

default_box_size = np.array([0.3, 0.3, 0.3], dtype=np.float64)
OBJECT_X_ROTATE_DEG_CLOCKWISE = 0.0
OBJECT_Y_ROTATE_DEG_CLOCKWISE = 0.0
OBJECT_Z_ROTATE_DEG_CLOCKWISE = 0.0


def _parse_vec3(text: str) -> np.ndarray:
    vals = [float(x) for x in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 comma-separated values, got: {text}")
    return np.asarray(vals, dtype=np.float64)


def _parse_body_names(text: str) -> list[str]:
    names = [s.strip() for s in text.split(",") if s.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Expected at least one body name")
    return names


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_wxyz_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_axis_clockwise_deg(axis: str, deg: float) -> np.ndarray:
    """Clockwise about +axis means negative angle by right-hand rule."""
    rad = -np.deg2rad(float(deg))
    c = np.cos(rad / 2.0)
    s = np.sin(rad / 2.0)
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=np.float64)
    raise ValueError(f"Unsupported axis: {axis}")


def _apply_local_box_rotation_offsets(
    q_wxyz: np.ndarray,
    rotate_x_deg_clockwise: float,
    rotate_y_deg_clockwise: float,
    rotate_z_deg_clockwise: float,
) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).copy()
    if rotate_x_deg_clockwise != 0.0:
        q = _quat_wxyz_multiply(q, _quat_axis_clockwise_deg("x", rotate_x_deg_clockwise))
    if rotate_y_deg_clockwise != 0.0:
        q = _quat_wxyz_multiply(q, _quat_axis_clockwise_deg("y", rotate_y_deg_clockwise))
    if rotate_z_deg_clockwise != 0.0:
        q = _quat_wxyz_multiply(q, _quat_axis_clockwise_deg("z", rotate_z_deg_clockwise))
    return q / max(np.linalg.norm(q), 1e-12)


def _yaw_only_quat_from_wxyz(q: np.ndarray) -> np.ndarray:
    """Extract world yaw (about z) from MuJoCo wxyz quaternion."""
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    vq = np.array([0.0, *v], dtype=np.float64)
    qr = _quat_wxyz_multiply(_quat_wxyz_multiply(q, vq), _quat_wxyz_conj(q))
    vx, vy = float(qr[1]), float(qr[2])
    yaw = np.arctan2(vy, vx)
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


@dataclass
class BoxFrame:
    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)

    @property
    def half_extents(self) -> np.ndarray:
        return 0.5 * self.size

    @property
    def rot(self) -> np.ndarray:
        return _quat_wxyz_to_rotmat(self.quat_wxyz)

    def world_to_local(self, pts_w: np.ndarray) -> np.ndarray:
        return (pts_w - self.center) @ self.rot

    def local_to_world(self, pts_l: np.ndarray) -> np.ndarray:
        return pts_l @ self.rot.T + self.center


def infer_source_box_from_ee(ee_world: np.ndarray, dst_box: BoxFrame) -> BoxFrame:
    """Heuristic source box estimate from EE positions.

    Assumptions:
    - Two-hand grasp around the box (side grasp).
    - Source box orientation is approximately the same as the target box orientation.
    - Source box size is fixed to default_box_size for this dataset.
    """
    center = ee_world.mean(axis=0)
    quat = dst_box.quat_wxyz.copy()
    return BoxFrame(center=center, size=default_box_size.copy(), quat_wxyz=quat)


def infer_scaled_targets(src_box: BoxFrame, dst_box: BoxFrame, ee_world: np.ndarray) -> np.ndarray:
    """Scale EE targets by preserving normalized local coordinates in the source box frame."""
    src_half = np.maximum(src_box.half_extents, 1e-6)
    local = src_box.world_to_local(ee_world)
    normalized = local / src_half
    dst_local = normalized * dst_box.half_extents
    return dst_box.local_to_world(dst_local)


def map_point_by_box_corner_reference(src_box: BoxFrame, dst_box: BoxFrame, point_world: np.ndarray) -> np.ndarray:
    """Map a world point from source-box frame to target-box frame via normalized box coordinates.

    This is equivalent to expressing the point relative to source box corners
    ([-1, 1] range per axis), then reconstructing it in the target box.
    """
    src_half = np.maximum(src_box.half_extents, 1e-6)
    local = src_box.world_to_local(point_world[None, :])[0]
    normalized = local / src_half
    dst_local = normalized * dst_box.half_extents
    return dst_box.local_to_world(dst_local[None, :])[0]


def _get_body_pos(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def solve_multi_ee_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_init: np.ndarray,
    body_ids: Sequence[int],
    target_positions: np.ndarray,
    fixed_body_ids: Sequence[int] | None = None,
    fixed_body_targets: np.ndarray | None = None,
    fixed_body_mask: np.ndarray | None = None,
    fixed_body_weight: float = 6.0,
    max_iters: int = 80,
    pos_tol: float = 1e-4,
    damping: float = 1e-3,
    step_scale: float = 0.7,
    regularization: float = 2e-4,
) -> tuple[np.ndarray, float]:
    """Damped least-squares IK using MuJoCo body-position Jacobians."""
    q = q_init.astype(np.float64, copy=True)
    nv = model.nv
    active = np.arange(nv, dtype=np.int32)

    # Freeze base z and base orientation updates to prevent lifting/tilting the whole robot.
    if nv >= 6:
        active = active[~np.isin(active, np.array([2, 3, 4, 5], dtype=np.int32))]

    for _ in range(max_iters):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        pos_errs = []
        jac_rows = []
        for i, body_id in enumerate(body_ids):
            cur = _get_body_pos(data, body_id)
            err = target_positions[i] - cur
            pos_errs.append(err)

            jacp = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacBody(model, data, jacp, None, body_id)
            jac_rows.append(jacp[:, active])

        if fixed_body_ids is not None and fixed_body_targets is not None and len(fixed_body_ids) > 0:
            for i, body_id in enumerate(fixed_body_ids):
                cur = _get_body_pos(data, body_id)
                err = fixed_body_targets[i] - cur
                mask = np.ones(3, dtype=np.float64) if fixed_body_mask is None else fixed_body_mask[i].astype(np.float64)
                rows = np.where(mask > 0.5)[0]
                if rows.size == 0:
                    continue
                pos_errs.append(fixed_body_weight * err[rows])
                jacp = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacBody(model, data, jacp, None, body_id)
                jac_rows.append(fixed_body_weight * jacp[rows][:, active])

        e = np.concatenate(pos_errs, axis=0)
        err_norm = float(np.linalg.norm(e))
        if err_norm < pos_tol:
            return q, err_norm

        J = np.vstack(jac_rows)
        A = J.T @ J + (damping + regularization) * np.eye(J.shape[1], dtype=np.float64)
        b = J.T @ e
        dq_active = np.linalg.solve(A, b)
        dqvel = np.zeros(nv, dtype=np.float64)
        dqvel[active] = step_scale * dq_active

        mujoco.mj_integratePos(model, q, dqvel, 1.0)
        mujoco.mj_normalizeQuat(model, q)

    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    residual = []
    for i, body_id in enumerate(body_ids):
        residual.append(target_positions[i] - _get_body_pos(data, body_id))
    return q, float(np.linalg.norm(np.concatenate(residual, axis=0)))


def _default_robot_xml(robot: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    if robot.lower() == "g1":
        return root / "models/g1/g1_29dof.xml"
    if robot.lower() == "t1":
        return root / "models/t1/t1_23dof.xml"
    raise ValueError(f"Unsupported robot preset: {robot}")


def _default_robot_urdf(robot: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    if robot.lower() == "g1":
        return root / "models/g1/g1_29dof.urdf"
    if robot.lower() == "t1":
        return root / "models/t1/t1_23dof.urdf"
    raise ValueError(f"Unsupported robot preset: {robot}")


def _body_names_from_model(model: mujoco.MjModel) -> list[str]:
    names = []
    for body_id in range(1, model.nbody):
        nm = model.body(body_id).name
        if nm:
            names.append(nm)
    return names


def _pick_existing_default_ee(model: mujoco.MjModel) -> list[str]:
    candidates = [
        ("left_hand_palm_link", "right_hand_palm_link"),
        ("left_rubber_hand_link", "right_rubber_hand_link"),
        ("left_sphere_hand", "right_sphere_hand"),
        ("left_wrist_roll_link", "right_wrist_roll_link"),
    ]
    model_names = set(_body_names_from_model(model))
    for pair in candidates:
        if all(n in model_names for n in pair):
            return list(pair)
    raise ValueError(
        "Could not infer end-effector bodies. Pass --ee-bodies explicitly. "
        f"Available bodies include: {sorted(list(model_names))[:20]} ..."
    )


def _pick_existing_default_feet(model: mujoco.MjModel) -> list[str] | None:
    candidates = [
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        ("left_foot_link", "right_foot_link"),
        ("left_ankle_pitch_link", "right_ankle_pitch_link"),
    ]
    model_names = set(_body_names_from_model(model))
    for pair in candidates:
        if all(n in model_names for n in pair):
            return list(pair)
    return None


def update_npz_kinematics(data_dict: dict[str, np.ndarray], model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    qpos_prev = data_dict.get("qpos")
    if qpos_prev is None:
        data_dict["qpos"] = q.astype(np.float32)
    else:
        qpos_prev = np.asarray(qpos_prev)
        qpos_dtype = np.float64 if qpos_prev.dtype == np.float64 else np.float32
        if qpos_prev.ndim == 2:
            data_dict["qpos"] = q[None, :].astype(qpos_dtype)
        else:
            data_dict["qpos"] = q.astype(qpos_dtype)

    if "dof_positions" in data_dict:
        dof_prev = np.asarray(data_dict["dof_positions"])
        n = dof_prev.shape[-1]
        dof_new = q[7 : 7 + n].astype(dof_prev.dtype)
        if dof_prev.ndim == 2:
            data_dict["dof_positions"] = dof_new[None, :]
        else:
            data_dict["dof_positions"] = dof_new

    if "body_positions" in data_dict and "body_names" in data_dict:
        body_names = [str(x) for x in data_dict["body_names"]]
        out_pos = np.asarray(data_dict["body_positions"]).copy()
        out_rot = data_dict.get("body_rotations", None)
        for i, name in enumerate(body_names):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            if out_pos.ndim == 3:
                out_pos[0, i] = data.xpos[bid]
            else:
                out_pos[i] = data.xpos[bid]
            if out_rot is not None:
                if out_rot.ndim == 3:
                    out_rot[0, i] = data.xquat[bid]
                else:
                    out_rot[i] = data.xquat[bid]
        data_dict["body_positions"] = out_pos.astype(data_dict["body_positions"].dtype)
        if out_rot is not None:
            data_dict["body_rotations"] = out_rot.astype(data_dict["body_rotations"].dtype)


def _read_first_frame(arr: np.ndarray, value_ndim: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == value_ndim + 1:
        return arr[0]
    return arr


def _extract_object_pose_from_payload(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    pos = None
    quat = None
    if "object_position_xyz" in payload:
        pos = np.asarray(payload["object_position_xyz"], dtype=np.float64).reshape(-1)
    elif "object_pose" in payload:
        pose = np.asarray(payload["object_pose"], dtype=np.float64).reshape(-1)
        if pose.size >= 3:
            pos = pose[:3]
        if pose.size >= 7:
            quat = pose[3:7]

    if "object_quat_wxyz" in payload:
        quat = np.asarray(payload["object_quat_wxyz"], dtype=np.float64).reshape(-1)
    elif "object_orientation" in payload:
        quat = np.asarray(payload["object_orientation"], dtype=np.float64).reshape(-1)

    if pos is None or quat is None or pos.size < 3 or quat.size < 4:
        return None

    quat = quat[:4]
    qn = np.linalg.norm(quat)
    if qn < 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat = quat / qn
    return pos[:3], quat


def _build_qpos_from_keyframe_payload(payload: dict[str, np.ndarray], model: mujoco.MjModel) -> np.ndarray:
    q0 = np.zeros(model.nq, dtype=np.float64)
    if model.nq >= 7:
        q0[3] = 1.0

    if "body_positions" in payload and "body_rotations" in payload and "body_names" in payload:
        body_positions = _read_first_frame(payload["body_positions"], value_ndim=2)
        body_rotations = _read_first_frame(payload["body_rotations"], value_ndim=2)
        body_names = [str(x) for x in payload["body_names"]]
        root_idx = None
        for candidate in ("pelvis", "base_link", "torso_link"):
            if candidate in body_names:
                root_idx = body_names.index(candidate)
                break
        if root_idx is None and body_names:
            root_idx = 0
        if root_idx is not None:
            q0[0:3] = np.asarray(body_positions[root_idx], dtype=np.float64)
            q0[3:7] = np.asarray(body_rotations[root_idx], dtype=np.float64)

    if "dof_positions" in payload:
        dof_positions = _read_first_frame(payload["dof_positions"], value_ndim=1).astype(np.float64, copy=False).reshape(-1)
        dof_names = [str(x) for x in payload.get("dof_names", [])]
        if dof_names and len(dof_names) == len(dof_positions):
            for name, val in zip(dof_names, dof_positions):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    continue
                qadr = int(model.jnt_qposadr[jid])
                q0[qadr] = float(val)
        else:
            n = min(len(dof_positions), max(model.nq - 7, 0))
            q0[7 : 7 + n] = dof_positions[:n]

    qn = np.linalg.norm(q0[3:7]) if model.nq >= 7 else 1.0
    if model.nq >= 7:
        if qn < 1e-12:
            q0[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            q0[3:7] /= qn
    return q0


def process_file(
    input_file: Path,
    output_file: Path,
    model: mujoco.MjModel,
    ee_bodies: list[str],
    foot_bodies: list[str] | None,
    src_box: BoxFrame | None,
    dst_box: BoxFrame,
    match_box_relative_base: bool = True,
    match_box_relative_base_z: bool = False,
    ground_z: float = 0.0,
    align_box_with_robot_yaw: bool = False,
    apply_box_rotation: bool = True,
    box_rotate_x_deg_clockwise: float = OBJECT_X_ROTATE_DEG_CLOCKWISE,
    box_rotate_y_deg_clockwise: float = OBJECT_Y_ROTATE_DEG_CLOCKWISE,
    box_rotate_z_deg_clockwise: float = OBJECT_Z_ROTATE_DEG_CLOCKWISE,
    retarget: bool = True,
    debug: bool = False,
) -> dict[str, np.ndarray]:
    with np.load(input_file, allow_pickle=True) as npz:
        payload = {k: npz[k] for k in npz.files}

    if "qpos" in payload:
        qpos = np.asarray(payload["qpos"])
        if qpos.ndim == 1:
            q0 = qpos.astype(np.float64)
        elif qpos.ndim == 2 and qpos.shape[0] >= 1:
            q0 = qpos[0].astype(np.float64)
        else:
            raise ValueError(f"Expected qpos shape (D,) or (T, D), got {qpos.shape}")
    else:
        q0 = _build_qpos_from_keyframe_payload(payload, model)

    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)

    body_ids = []
    for name in ee_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in MuJoCo model")
        body_ids.append(bid)

    foot_ids: list[int] = []
    foot_targets: np.ndarray | None = None
    foot_mask: np.ndarray | None = None
    if foot_bodies:
        for name in foot_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                foot_ids.append(bid)
        if foot_ids:
            foot_targets = np.vstack([_get_body_pos(data, bid) for bid in foot_ids])
            foot_targets[:, 2] = float(ground_z)
            foot_mask = np.zeros((len(foot_ids), 3), dtype=np.float64)
            foot_mask[:, 2] = 1.0

    ee_world = np.vstack([_get_body_pos(data, bid) for bid in body_ids])
    robot_yaw_quat = _yaw_only_quat_from_wxyz(q0[3:7])
    object_pose = _extract_object_pose_from_payload(payload)
    if object_pose is not None:
        object_pos, object_quat = object_pose
        if apply_box_rotation:
            # Apply in box local/root frame (intrinsic rotations): q_new = q_obj * q_offset.
            object_quat = _apply_local_box_rotation_offsets(
                object_quat,
                rotate_x_deg_clockwise=box_rotate_x_deg_clockwise,
                rotate_y_deg_clockwise=box_rotate_y_deg_clockwise,
                rotate_z_deg_clockwise=box_rotate_z_deg_clockwise,
            )
        src_box_used = BoxFrame(
            center=object_pos.copy(),
            size=src_box.size.copy() if src_box is not None else default_box_size.copy(),
            quat_wxyz=object_quat.copy(),
        )
        dst_box_used = BoxFrame(
            center=object_pos.copy(),
            size=dst_box.size.copy(),
            quat_wxyz=object_quat.copy(),
        )
        if apply_box_rotation:
            if "object_quat_wxyz" in payload:
                oq = np.asarray(payload["object_quat_wxyz"])
                payload["object_quat_wxyz"] = object_quat.astype(oq.dtype)
            if "object_orientation" in payload:
                oo = np.asarray(payload["object_orientation"])
                payload["object_orientation"] = object_quat.astype(oo.dtype)
    else:
        src_box_used = src_box if src_box is not None else infer_source_box_from_ee(ee_world, dst_box)
        dst_box_used = dst_box
        if dst_box_used.center is None:
            dst_box_used = BoxFrame(center=src_box_used.center.copy(), size=dst_box.size.copy(), quat_wxyz=dst_box.quat_wxyz.copy())

    if align_box_with_robot_yaw:
        src_box_used = BoxFrame(
            center=src_box_used.center.copy(),
            size=src_box_used.size.copy(),
            quat_wxyz=robot_yaw_quat.copy(),
        )
    if align_box_with_robot_yaw:
        dst_box_used = BoxFrame(
            center=dst_box_used.center.copy(),
            size=dst_box_used.size.copy(),
            quat_wxyz=robot_yaw_quat.copy(),
        )

    if not retarget:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_file, **payload)
        if debug:
            print(f"[{input_file.name}]")
            print("  mode: overwrite-rotation-only (skip IK and robot/body updates)")
            print(f"  wrote: {output_file}")
        return {
            "q_before": q0.copy(),
            "q_after": q0.copy(),
            "ee_before": ee_world.copy(),
            "ee_after": ee_world.copy(),
            "ee_targets": ee_world.copy(),
            "src_box_center_used": src_box_used.center.copy(),
            "src_box_size_used": src_box_used.size.copy(),
            "src_box_quat_used": src_box_used.quat_wxyz.copy(),
            "dst_box_center_used": dst_box_used.center.copy(),
            "dst_box_quat_used": dst_box_used.quat_wxyz.copy(),
        }

    targets = infer_scaled_targets(src_box_used, dst_box_used, ee_world)
    q_init = q0.copy()
    base_before = q0[0:3].copy()
    if match_box_relative_base:
        mapped = map_point_by_box_corner_reference(src_box_used, dst_box_used, base_before)
        q_init[0:2] = mapped[0:2]
        if match_box_relative_base_z:
            q_init[2] = mapped[2]

    q_new, residual = solve_multi_ee_ik(
        model,
        data,
        q_init,
        body_ids,
        targets,
        fixed_body_ids=foot_ids,
        fixed_body_targets=foot_targets,
        fixed_body_mask=foot_mask,
    )
    data.qpos[:] = q_new
    mujoco.mj_forward(model, data)
    ee_after = np.vstack([_get_body_pos(data, bid) for bid in body_ids])
    update_npz_kinematics(payload, model, data, q_new)

    if "cost" in payload and np.asarray(payload["cost"]).shape == ():
        payload["cost"] = np.asarray(float(residual), dtype=payload["cost"].dtype)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **payload)

    if debug:
        print(f"[{input_file.name}]")
        if src_box is None:
            print(f"  inferred src box center: {src_box_used.center}")
            print(f"  inferred src box size:   {src_box_used.size}")
        if dst_box.center is None:
            print(f"  inferred dst box center: {dst_box_used.center}")
        if align_box_with_robot_yaw:
            print(f"  aligned box yaw to robot (quat wxyz): {robot_yaw_quat}")
        if match_box_relative_base:
            print(f"  base shift (corner-referenced): {base_before} -> {q_init[0:3]}")
        if foot_ids:
            print(f"  grounded feet: {len(foot_ids)} bodies, ground_z={ground_z:.3f}, xy_sliding=True")
        for name, cur, tgt in zip(ee_bodies, ee_world, targets):
            print(f"  {name}: {cur} -> {tgt}")
        print(f"  residual: {residual:.6e}")
        print(f"  wrote: {output_file}")

    return {
        "q_before": q0,
        "q_after": q_new.copy(),
        "ee_before": ee_world,
        "ee_after": ee_after,
        "ee_targets": targets,
        "src_box_center_used": src_box_used.center.copy(),
        "src_box_size_used": src_box_used.size.copy(),
        "src_box_quat_used": src_box_used.quat_wxyz.copy(),
        "dst_box_center_used": dst_box_used.center.copy(),
        "dst_box_quat_used": dst_box_used.quat_wxyz.copy(),
    }


def _visualize_before_after(
    robot_urdf: Path,
    src_box: BoxFrame,
    dst_box: BoxFrame,
    q_before: np.ndarray,
    q_after: np.ndarray,
    ee_before: np.ndarray,
    ee_after: np.ndarray,
    ee_targets: np.ndarray,
) -> None:
    try:
        import viser  # type: ignore[import-not-found]
        import trimesh  # type: ignore[import-not-found]
        import yourdfpy  # type: ignore[import-untyped]
        from viser.extras import ViserUrdf  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Visualization requires `viser`, `yourdfpy`, and `trimesh` in the retargeting environment.") from exc

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=4, height=4, position=(0.0, 0.0, 0.0))
    server.scene.add_label("/labels/before", "before", position=(-0.8, 0.0, 1.2))
    server.scene.add_label("/labels/after", "after", position=(0.8, 0.0, 1.2))

    urdf = yourdfpy.URDF.load(str(robot_urdf), load_meshes=True, build_scene_graph=True)
    before_root = server.scene.add_frame("/before", show_axes=False)
    after_root = server.scene.add_frame("/after", show_axes=False)
    before_root.position = np.array([-0.8, 0.0, 0.0], dtype=np.float64)
    after_root.position = np.array([0.8, 0.0, 0.0], dtype=np.float64)
    vr_before = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/before")
    vr_after = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/after")
    robot_dof = len(vr_before.get_actuated_joint_limits())

    def _apply(vr: ViserUrdf, root, q: np.ndarray):
        vr.update_cfg(q[7 : 7 + robot_dof])
        root.position = root.position * 0 + root.position + q[0:3]  # keep side-by-side offset plus base translation
        root.wxyz = q[3:7]

    # Apply poses
    vr_before.update_cfg(q_before[7 : 7 + robot_dof])
    before_root.position = np.array([-0.8, 0.0, 0.0], dtype=np.float64) + q_before[0:3]
    before_root.wxyz = q_before[3:7]
    vr_after.update_cfg(q_after[7 : 7 + robot_dof])
    after_root.position = np.array([0.8, 0.0, 0.0], dtype=np.float64) + q_after[0:3]
    after_root.wxyz = q_after[3:7]

    def _add_box_mesh(
        name: str,
        box: BoxFrame,
        side_offset: np.ndarray,
        color: tuple[int, int, int],
        opacity: float,
        center_override: np.ndarray | None = None,
    ) -> None:
        mesh = trimesh.creation.box(extents=box.size)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        box_center = box.center if center_override is None else np.asarray(center_override, dtype=np.float64)
        verts = (verts @ box.rot.T + box_center + side_offset).astype(np.float32)
        server.scene.add_mesh_simple(
            name,
            vertices=verts,
            faces=faces,
            position=(0.0, 0.0, 0.0),
            color=color,
            opacity=opacity,
        )

    src_box_viz_center = ee_before.mean(axis=0) if ee_before.shape[0] >= 2 else src_box.center
    dst_box_viz_center = ee_targets.mean(axis=0) if ee_targets.shape[0] >= 2 else dst_box.center
    _add_box_mesh(
        "/box/source",
        src_box,
        np.array([-0.8, 0.0, 0.0], dtype=np.float64),
        (240, 180, 80),
        0.25,
        center_override=src_box_viz_center,
    )
    _add_box_mesh(
        "/box/target",
        dst_box,
        np.array([0.8, 0.0, 0.0], dtype=np.float64),
        (80, 180, 240),
        0.25,
        center_override=dst_box_viz_center,
    )
    server.scene.add_frame(
        "/box/source_axis",
        position=tuple(src_box_viz_center + np.array([-0.8, 0.0, 0.0], dtype=np.float64)),
        wxyz=tuple(src_box.quat_wxyz),
        axes_length=0.18,
        axes_radius=0.01,
    )
    server.scene.add_frame(
        "/box/target_axis",
        position=tuple(dst_box_viz_center + np.array([0.8, 0.0, 0.0], dtype=np.float64)),
        wxyz=tuple(dst_box.quat_wxyz),
        axes_length=0.18,
        axes_radius=0.01,
    )

    # Mark EE and target points
    for i, p in enumerate(ee_before):
        server.scene.add_icosphere(
            f"/pts/before_{i}",
            radius=0.02,
            color=(200, 80, 80),
            position=tuple(p + np.array([-0.8, 0, 0])),
        )
    for i, p in enumerate(ee_after):
        server.scene.add_icosphere(
            f"/pts/after_{i}",
            radius=0.02,
            color=(80, 200, 80),
            position=tuple(p + np.array([0.8, 0, 0])),
        )
    for i, p in enumerate(ee_targets):
        server.scene.add_icosphere(
            f"/pts/target_{i}",
            radius=0.018,
            color=(80, 160, 255),
            position=tuple(p + np.array([0.8, 0, 0])),
        )

    print("Open the Viser URL shown above to inspect before/after retargeting. Press Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retarget robot keyframe npz files for different box sizes by adjusting end-effector positions."
    )
    p.add_argument("--input", type=Path, help="Single input npz file.")
    p.add_argument("--input-dir", type=Path, help="Directory of npz keyframe files.")
    p.add_argument("--output-dir", type=Path, help="Output directory for updated npz files.")
    p.add_argument(
        "--overwrite-input",
        action="store_true",
        help="Overwrite input npz files in place instead of writing to --output-dir.",
    )
    p.add_argument("--robot-xml", type=Path, help="MuJoCo XML model path. Defaults to g1 preset XML.")
    p.add_argument("--robot", default="g1", choices=["g1", "t1"], help="Robot preset used when --robot-xml is omitted.")
    p.add_argument(
        "--ee-bodies",
        type=_parse_body_names,
        help="Comma-separated body names for end effectors. Default auto-detects common hand bodies.",
    )
    p.add_argument("--src-box-center", type=_parse_vec3, help="Source box center (x,y,z). Optional with --infer-src-box.")
    p.add_argument("--src-box-size", type=_parse_vec3, help="Source box size (sx,sy,sz). Optional with --infer-src-box.")
    p.add_argument("--dst-box-center", type=_parse_vec3, help="Target box center (x,y,z). Defaults to source center.")
    p.add_argument("--dst-box-size", type=_parse_vec3, required=True, help="Target box size (sx,sy,sz).")
    p.add_argument(
        "--src-box-quat-wxyz",
        type=lambda s: np.asarray([float(x) for x in s.split(",")], dtype=np.float64),
        default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        help="Source box orientation quaternion w,x,y,z (default identity).",
    )
    p.add_argument(
        "--dst-box-quat-wxyz",
        type=lambda s: np.asarray([float(x) for x in s.split(",")], dtype=np.float64),
        default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        help="Target box orientation quaternion w,x,y,z (default identity).",
    )
    p.add_argument("--debug", action="store_true", help="Print detailed per-file diagnostics.")
    p.add_argument(
        "--infer-src-box",
        action="store_true",
        help="Infer source box center/size from current EE positions (heuristic side-grasp assumption).",
    )
    p.add_argument(
        "--align-box-with-robot-yaw",
        action="store_true",
        help="Override source/target box yaw to match robot base yaw so the robot directly faces the box.",
    )
    p.add_argument(
        "--no-match-box-relative-base",
        action="store_true",
        help="Disable matching robot base position relative to source/target box corners.",
    )
    p.add_argument(
        "--match-box-relative-base-z",
        action="store_true",
        help="Also match base z to box-relative mapping (default keeps original base height).",
    )
    p.add_argument(
        "--foot-bodies",
        type=_parse_body_names,
        help="Comma-separated feet body names used for ground constraints (default auto-detects).",
    )
    p.add_argument(
        "--ground-z",
        type=float,
        default=0.0,
        help="Ground plane height for feet constraints (default: 0.0).",
    )
    p.add_argument("--visualize", action="store_true", help="Open a Viser viewer for before/after comparison.")
    p.add_argument(
        "--box-rotate-x-deg-clockwise",
        type=float,
        default=OBJECT_X_ROTATE_DEG_CLOCKWISE,
        help="Local box-frame clockwise rotation offset around x (deg) applied to object orientation.",
    )
    p.add_argument(
        "--box-rotate-y-deg-clockwise",
        type=float,
        default=OBJECT_Y_ROTATE_DEG_CLOCKWISE,
        help="Local box-frame clockwise rotation offset around y (deg) applied to object orientation.",
    )
    p.add_argument(
        "--box-rotate-z-deg-clockwise",
        type=float,
        default=OBJECT_Z_ROTATE_DEG_CLOCKWISE,
        help="Local box-frame clockwise rotation offset around z (deg) applied to object orientation.",
    )
    p.add_argument(
        "--visualize-file",
        type=str,
        help="For --input-dir runs, visualize this filename (default: first processed file).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.input and not args.input_dir:
        raise ValueError("Provide --input or --input-dir")
    if args.overwrite_input and args.output_dir is None:
        raise ValueError("When using --overwrite-input, also provide --output-dir for retargeted outputs.")
    if not args.overwrite_input and args.output_dir is None:
        raise ValueError("Provide --output-dir, or use --overwrite-input.")

    robot_xml = args.robot_xml or _default_robot_xml(args.robot)
    robot_urdf = _default_robot_urdf(args.robot)
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    ee_bodies = args.ee_bodies or _pick_existing_default_ee(model)
    foot_bodies = args.foot_bodies or _pick_existing_default_feet(model)

    src_box = None
    if args.src_box_center is not None:
        src_box = BoxFrame(
            center=args.src_box_center,
            size=args.src_box_size if args.src_box_size is not None else default_box_size.copy(),
            quat_wxyz=np.asarray(args.src_box_quat_wxyz, dtype=np.float64),
        )
    elif args.infer_src_box:
        src_box = None
    dst_box = BoxFrame(
        center=args.dst_box_center if args.dst_box_center is not None else args.src_box_center,
        size=args.dst_box_size,
        quat_wxyz=np.asarray(args.dst_box_quat_wxyz, dtype=np.float64),
    )

    inputs: list[Path] = []
    if args.input:
        inputs.append(args.input)
    if args.input_dir:
        inputs.extend(sorted(args.input_dir.glob("*.npz")))
    if not inputs:
        raise ValueError("No input files found")

    viz_payload: dict[str, np.ndarray] | None = None
    viz_name = args.visualize_file
    for in_file in inputs:
        if args.overwrite_input:
            # 1) Overwrite original input with box-rotation-only payload.
            process_file(
                input_file=in_file,
                output_file=in_file,
                model=model,
                ee_bodies=ee_bodies,
                foot_bodies=foot_bodies,
                src_box=src_box,
                dst_box=dst_box,
                match_box_relative_base=not args.no_match_box_relative_base,
                match_box_relative_base_z=args.match_box_relative_base_z,
                ground_z=args.ground_z,
                align_box_with_robot_yaw=args.align_box_with_robot_yaw,
                apply_box_rotation=True,
                box_rotate_x_deg_clockwise=args.box_rotate_x_deg_clockwise,
                box_rotate_y_deg_clockwise=args.box_rotate_y_deg_clockwise,
                box_rotate_z_deg_clockwise=args.box_rotate_z_deg_clockwise,
                retarget=False,
                debug=args.debug,
            )
            # 2) Save full retargeted result to output-dir from rotated input.
            ret = process_file(
                input_file=in_file,
                output_file=args.output_dir / in_file.name,
                model=model,
                ee_bodies=ee_bodies,
                foot_bodies=foot_bodies,
                src_box=src_box,
                dst_box=dst_box,
                match_box_relative_base=not args.no_match_box_relative_base,
                match_box_relative_base_z=args.match_box_relative_base_z,
                ground_z=args.ground_z,
                align_box_with_robot_yaw=args.align_box_with_robot_yaw,
                apply_box_rotation=False,
                box_rotate_x_deg_clockwise=args.box_rotate_x_deg_clockwise,
                box_rotate_y_deg_clockwise=args.box_rotate_y_deg_clockwise,
                box_rotate_z_deg_clockwise=args.box_rotate_z_deg_clockwise,
                retarget=True,
                debug=args.debug,
            )
        else:
            ret = process_file(
                input_file=in_file,
                output_file=args.output_dir / in_file.name,
                model=model,
                ee_bodies=ee_bodies,
                foot_bodies=foot_bodies,
                src_box=src_box,
                dst_box=dst_box,
                match_box_relative_base=not args.no_match_box_relative_base,
                match_box_relative_base_z=args.match_box_relative_base_z,
                ground_z=args.ground_z,
                align_box_with_robot_yaw=args.align_box_with_robot_yaw,
                apply_box_rotation=True,
                box_rotate_x_deg_clockwise=args.box_rotate_x_deg_clockwise,
                box_rotate_y_deg_clockwise=args.box_rotate_y_deg_clockwise,
                box_rotate_z_deg_clockwise=args.box_rotate_z_deg_clockwise,
                retarget=True,
                debug=args.debug,
            )
        if args.visualize and viz_payload is None and (viz_name is None or in_file.name == viz_name):
            viz_payload = ret

    if args.visualize:
        if viz_payload is None:
            raise ValueError(f"--visualize-file '{viz_name}' was not found in processed inputs")
        src_box_for_viz = src_box
        if src_box_for_viz is None:
            src_box_for_viz = BoxFrame(
                center=viz_payload["src_box_center_used"],
                size=viz_payload["src_box_size_used"],
                quat_wxyz=viz_payload["src_box_quat_used"],
            )
        dst_box_for_viz = BoxFrame(
            center=viz_payload["dst_box_center_used"],
            size=dst_box.size,
            quat_wxyz=viz_payload["dst_box_quat_used"],
        )
        _visualize_before_after(
            robot_urdf=robot_urdf,
            src_box=src_box_for_viz,
            dst_box=dst_box_for_viz,
            q_before=viz_payload["q_before"],
            q_after=viz_payload["q_after"],
            ee_before=viz_payload["ee_before"],
            ee_after=viz_payload["ee_after"],
            ee_targets=viz_payload["ee_targets"],
        )


if __name__ == "__main__":
    main()
