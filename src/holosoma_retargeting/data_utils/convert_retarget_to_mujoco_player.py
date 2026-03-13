from __future__ import annotations

import argparse
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np


def _default_robot_xml() -> Path:
    return Path(__file__).resolve().parents[1] / "models/g1/g1_29dof.xml"


def _collect_actuated_joint_and_body_names(model: mujoco.MjModel) -> tuple[list[str], list[str], list[int]]:
    """Return actuated joint names and corresponding body names in qpos order."""
    dof_joint_ids: list[int] = []
    dof_names: list[str] = []
    body_ids: list[int] = []
    body_names: list[str] = []

    # Root body: body attached to the free joint (fallback to pelvis/body-1).
    root_bid = 1 if model.nbody > 1 else 0
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            root_bid = int(model.jnt_bodyid[j])
            break
    body_ids.append(root_bid)
    body_names.append(model.body(root_bid).name)

    for j in range(model.njnt):
        jt = model.jnt_type[j]
        if jt not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        jname = model.joint(j).name
        if not jname:
            continue
        dof_joint_ids.append(j)
        dof_names.append(jname)
        bid = int(model.jnt_bodyid[j])
        body_ids.append(bid)
        body_names.append(model.body(bid).name)

    return dof_names, body_names, body_ids


def _compute_qvel_sequence(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    """Compute qvel from qpos with proper quaternion handling."""
    T = qpos.shape[0]
    qvel = np.zeros((T, model.nv), dtype=np.float64)
    dt = 1.0 / max(float(fps), 1e-8)
    for t in range(1, T):
        mujoco.mj_differentiatePos(model, qvel[t], dt, qpos[t - 1], qpos[t])
    if T > 1:
        qvel[0] = qvel[1]
    return qvel


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert retargeted qpos npz to mujoco_player Stage2 format.")
    parser.add_argument("--input", required=True, help="Path to retargeted npz with qpos/fps.")
    parser.add_argument("--output", required=True, help="Path to output Stage2 npz.")
    parser.add_argument(
        "--robot-xml",
        default=str(_default_robot_xml()),
        help="MuJoCo robot XML used to reconstruct full-body poses/velocities.",
    )
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "qpos" not in data.files:
        raise ValueError(f"Expected 'qpos' in {args.input}, got keys: {data.files}")
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    fps = float(data["fps"]) if "fps" in data.files else 30.0

    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2D (T,D), got {qpos.shape}")

    model = mujoco.MjModel.from_xml_path(str(Path(args.robot_xml)))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos second dim ({qpos.shape[1]}) does not match robot nq ({model.nq})")

    dof_names, body_names, body_ids = _collect_actuated_joint_and_body_names(model)
    dof_count = len(dof_names)
    body_count = len(body_names)
    if dof_count <= 0:
        raise ValueError("No actuated joints found in robot model.")

    T = qpos.shape[0]
    qvel_all = _compute_qvel_sequence(model, qpos, fps)
    # Free joint contributes 6 velocities; remainder maps to actuated joints.
    dof_velocities = qvel_all[:, model.nv - dof_count :].astype(np.float32)
    dof_positions = qpos[:, model.nq - dof_count :].astype(np.float32)

    body_positions = np.zeros((T, body_count, 3), dtype=np.float32)
    body_rotations = np.zeros((T, body_count, 4), dtype=np.float32)
    body_linear_velocities = np.zeros((T, body_count, 3), dtype=np.float32)
    body_angular_velocities = np.zeros((T, body_count, 3), dtype=np.float32)

    sim_data = mujoco.MjData(model)
    for t in range(T):
        sim_data.qpos[:] = qpos[t]
        sim_data.qvel[:] = qvel_all[t]
        mujoco.mj_forward(model, sim_data)
        for i, bid in enumerate(body_ids):
            body_positions[t, i] = sim_data.xpos[bid]
            body_rotations[t, i] = sim_data.xquat[bid]
            # cvel = [angular(0:3), linear(3:6)]
            body_angular_velocities[t, i] = sim_data.cvel[bid, 0:3]
            body_linear_velocities[t, i] = sim_data.cvel[bid, 3:6]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.asarray(fps, dtype=np.float32),
        dof_names=np.asarray(dof_names),
        body_names=np.asarray(body_names),
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    print(f"Saved Stage2 motion to {output_path}")


if __name__ == "__main__":
    main()
