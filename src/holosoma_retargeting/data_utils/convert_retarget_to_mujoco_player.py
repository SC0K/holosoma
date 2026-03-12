from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert retargeted qpos npz to mujoco_player Stage2 format.")
    parser.add_argument("--input", required=True, help="Path to retargeted npz with qpos/fps.")
    parser.add_argument("--output", required=True, help="Path to output Stage2 npz.")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "qpos" not in data.files:
        raise ValueError(f"Expected 'qpos' in {args.input}, got keys: {data.files}")
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data.files else 30

    if qpos.shape[1] < 7:
        raise ValueError(f"qpos must have at least 7 columns (got {qpos.shape})")

    body_positions = qpos[:, :3].astype(np.float32)[:, None, :]
    body_rotations = qpos[:, 3:7].astype(np.float32)[:, None, :]
    dof_positions = qpos[:, 7:].astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=fps,
        body_positions=body_positions,
        body_rotations=body_rotations,
        dof_positions=dof_positions,
    )
    print(f"Saved Stage2 motion to {output_path}")


if __name__ == "__main__":
    main()
