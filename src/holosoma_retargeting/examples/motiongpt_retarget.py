#!/usr/bin/env python3
"""Generate motion with MotionGPT3 from text and retarget to a robot (g1)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def _find_latest_out_npy(root: Path) -> Path:
    candidates = list(root.rglob("*_out.npy"))
    if not candidates:
        raise FileNotFoundError(f"No '*_out.npy' found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_motion_npy(path: Path) -> np.ndarray:
    data = np.load(str(path))
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError(f"Unexpected motion shape {data.shape} in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionGPT3 text->motion->retarget pipeline")
    parser.add_argument("--prompt", required=False, help="Text prompt for motion generation")
    parser.add_argument("--motiongpt-root", default="MotionGPT3", help="Path to MotionGPT3 repo")
    parser.add_argument("--work-dir", default="motiongpt_retarget_runs", help="Working/output directory")
    parser.add_argument("--task-name", default="motiongpt_seq", help="Sequence name for retargeting")
    parser.add_argument("--task-type", default="robot_only", help="Retargeting task type")
    parser.add_argument("--data-format", default="humanml3d", help="Holosoma data format")
    parser.add_argument("--robot", default="g1", help="Robot type")
    parser.add_argument("--retarget-save-dir", default=None, help="Optional retarget save dir")
    parser.add_argument(
        "--y-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat MotionGPT output as y-up and convert to z-up",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to run subcommands")
    args = parser.parse_args()

    if not args.prompt:
        try:
            args.prompt = input("Enter motion prompt: ").strip()
        except EOFError:
            args.prompt = ""
        if not args.prompt:
            raise ValueError("Prompt is required (use --prompt or enter it interactively).")

    repo_root = Path(__file__).resolve().parents[4]
    motiongpt_root = (repo_root / args.motiongpt_root).resolve()
    work_dir = (repo_root / args.work_dir).resolve()

    work_dir.mkdir(parents=True, exist_ok=True)
    motion_out_dir = work_dir / "motiongpt_out"
    motion_out_dir.mkdir(parents=True, exist_ok=True)
    retarget_input_dir = work_dir / "retarget_input"
    retarget_input_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = work_dir / "prompt.txt"
    prompt_file.write_text(args.prompt.strip() + "", encoding="utf-8")

    demo_py = motiongpt_root / "demo.py"
    if not demo_py.exists():
        raise FileNotFoundError(f"MotionGPT3 demo not found: {demo_py}")

    demo_cmd = [
        args.python,
        str(demo_py),
        "--example",
        str(prompt_file),
        "--out_dir",
        str(motion_out_dir),
    ]
    subprocess.check_call(demo_cmd, cwd=str(motiongpt_root))

    latest_npy = _find_latest_out_npy(motion_out_dir)
    joints = _load_motion_npy(latest_npy)

    if args.y_up:
        from holosoma_retargeting.src.utils import transform_y_up_to_z_up

        joints = transform_y_up_to_z_up(joints)
        up_axis = 2
    else:
        up_axis = 1

    height = float(joints[..., up_axis].max() - joints[..., up_axis].min())
    out_npz = retarget_input_dir / f"{args.task_name}.npz"
    np.savez(str(out_npz), global_joint_positions=joints, height=height)

    retarget_py = repo_root / "holosoma" / "src" / "holosoma_retargeting" / "examples" / "robot_retarget.py"
    retarget_cmd = [
        args.python,
        str(retarget_py),
        "--data_path",
        str(retarget_input_dir),
        "--task-type",
        args.task_type,
        "--task-name",
        args.task_name,
        "--data_format",
        args.data_format,
        "--robot",
        args.robot,
    ]
    if args.retarget_save_dir:
        retarget_cmd += ["--save_dir", args.retarget_save_dir]

    subprocess.check_call(retarget_cmd, cwd=str(repo_root))


if __name__ == "__main__":
    main()
