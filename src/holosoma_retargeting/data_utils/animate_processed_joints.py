from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate processed joint positions from .npz.")
    parser.add_argument("--input", required=True, help="Path to processed .npz with global_joint_positions")
    parser.add_argument("--output", default=None, help="Optional output .mp4 path")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--step", type=int, default=1, help="Frame step size")
    parser.add_argument("--interval", type=int, default=33, help="Frame delay in ms (approx 30fps)")
    parser.add_argument(
        "--data_format",
        default="smplh",
        help="Data format for joint naming (e.g., smplh, smplx)",
    )
    parser.add_argument(
        "--highlight_joints",
        nargs="*",
        default=["L_Wrist", "R_Wrist"],
        help="Joint names to highlight",
    )
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "global_joint_positions" not in data.files:
        raise ValueError(f"Missing global_joint_positions in {args.input}")
    joints = data["global_joint_positions"]
    num_frames = joints.shape[0]
    start = max(args.start, 0)
    end = num_frames if args.end is None else min(args.end, num_frames)
    if start >= end:
        raise ValueError(f"Invalid frame range {start}..{end}")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    # Load joint names for highlighting
    from pathlib import Path as _Path
    import sys as _sys
    this_dir = _Path(__file__).resolve()
    _sys.path.insert(0, str(this_dir.parents[2]))
    from holosoma_retargeting.config_types import data_type as _dt  # type: ignore[import-not-found]
    joint_names = _dt.DEMO_JOINTS_REGISTRY.get(args.data_format)
    if joint_names is None:
        raise ValueError(f"Unknown data_format: {args.data_format}")
    highlight_indices = [joint_names.index(j) for j in args.highlight_joints if j in joint_names]

    pts = joints[start:end:args.step]
    mins = pts.reshape(-1, 3).min(axis=0)
    maxs = pts.reshape(-1, 3).max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    center = (mins + maxs) / 2.0
    half = span.max() * 0.6

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter([], [], [], s=10, c="orange")
    scat_hi = ax.scatter([], [], [], s=40, c="red")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def update(i):
        frame = pts[i]
        scat._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])
        if highlight_indices:
            hi = frame[highlight_indices]
            scat_hi._offsets3d = (hi[:, 0], hi[:, 1], hi[:, 2])
        else:
            scat_hi._offsets3d = ([], [], [])
        ax.set_title(f"Frame {start + i * args.step}")
        return scat, scat_hi

    anim = FuncAnimation(fig, update, frames=len(pts), interval=args.interval, blit=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=int(round(1000 / max(args.interval, 1))))
        anim.save(out_path, writer=writer)
        print(f"Saved animation to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
