#!/usr/bin/env python3
"""Generate MotionGPT3 motion from text, retarget to robot, then play in MuJoCo."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_SAVE_DIRS = {
    "robot_only": "demo_results/{robot}/robot_only/omomo",
    "object_interaction": "demo_results/{robot}/object_interaction/omomo",
    "climbing": "demo_results/{robot}/climbing/mocap_climb",
}


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


def _retarget_output_path(task_type: str, save_dir: Path, task_name: str, augmentation: bool = False) -> Path:
    if task_type == "robot_only":
        return save_dir / f"{task_name}.npz"
    suffix = "_augmented" if augmentation else "_original"
    return save_dir / f"{task_name}{suffix}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MotionGPT3 text->motion->retarget pipeline with MuJoCo playback"
    )
    parser.add_argument("--prompt", required=False, help="Text prompt for motion generation")
    parser.add_argument("--motiongpt-root", default="MotionGPT3", help="Path to MotionGPT3 repo")
    parser.add_argument("--cfg", default="./configs/test.yaml", help="MotionGPT config file")
    parser.add_argument("--work-dir", default="motiongpt_retarget_runs", help="Working/output directory")
    parser.add_argument("--task-name", default="motiongpt_seq", help="Sequence name for retargeting")
    parser.add_argument("--task-type", default="robot_only", help="Retargeting task type")
    parser.add_argument("--data-format", default="humanml3d", help="Holosoma data format")
    parser.add_argument("--robot", default="g1", help="Robot type")
    parser.add_argument("--retarget-save-dir", default=None, help="Optional retarget save dir")
    parser.add_argument("--length", type=int, default=None, help="Override motion length (frames)")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS metadata for retargeted motion")
    parser.add_argument(
        "--swap-lr",
        action="store_true",
        help="Swap left/right joints in the MotionGPT joint array (humanml3d only)",
    )
    parser.add_argument(
        "--mirror-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror motion across the X axis before retargeting (fixes left/right visual flip)",
    )
    parser.add_argument(
        "--y-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat MotionGPT output as y-up and convert to z-up",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to run subcommands")
    parser.add_argument(
        "--player-script",
        default="key-loco-man/source/isaaclab_assets/data/motion/AMASS_Retargeted_for_G1/scripts/mujoco_player.py",
        help="Path to mujoco_player.py",
    )
    parser.add_argument(
        "--player-model",
        default=None,
        help="Optional path to MuJoCo model xml (overrides player default)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip MuJoCo playback step",
    )
    parser.add_argument(
        "--player-args",
        nargs=argparse.REMAINDER,
        help="Additional args passed to mujoco_player.py (e.g. --speed 1.0)",
    )
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
    # Run demo.py
    demo_cmd = [
        args.python,
        str(demo_py),
        "--cfg",
        args.cfg,
        "--example",
        str(prompt_file),
        "--out_dir",
        str(motion_out_dir),
    ]
    if args.length:
        demo_cmd += ["--length", str(args.length)]
    subprocess.check_call(demo_cmd, cwd=str(motiongpt_root))

    latest_npy = _find_latest_out_npy(motion_out_dir)
    joints = _load_motion_npy(latest_npy)

    if args.swap_lr:
        if args.data_format != "humanml3d":
            raise ValueError("--swap-lr is only supported for data_format=humanml3d")
        from holosoma_retargeting.config_types.data_type import HUMANML3D_DEMO_JOINTS

        pairs = [
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle"),
            ("left_foot", "right_foot"),
            ("left_collar", "right_collar"),
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
        ]
        for left, right in pairs:
            li = HUMANML3D_DEMO_JOINTS.index(left)
            ri = HUMANML3D_DEMO_JOINTS.index(right)
            joints[:, [li, ri], :] = joints[:, [ri, li], :]

    if args.mirror_x:
        joints[..., 0] *= -1.0

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
    # Determine FPS for retarget metadata.
    if args.fps:
        motion_fps = int(args.fps)
    else:
        try:
            from omegaconf import OmegaConf
            cfg_assets = OmegaConf.load(str(motiongpt_root / "configs" / "assets.yaml"))
            cfg_base = OmegaConf.load(str(motiongpt_root / cfg_assets.CONFIG_FOLDER / "default.yaml"))
            cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(str(motiongpt_root / args.cfg)))
            if not cfg_exp.FULL_CONFIG:
                from motGPT.config import get_module_config
                cfg_exp = get_module_config(cfg_exp, str(motiongpt_root / cfg_assets.CONFIG_FOLDER))
            cfg = OmegaConf.merge(cfg_exp, cfg_assets)
            motion_fps = int(cfg.DATASET.HUMANML3D.FPS)
        except Exception:
            motion_fps = 30

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
        "--fps",
        str(motion_fps),
    ]
    if args.retarget_save_dir:
        retarget_cmd += ["--save_dir", args.retarget_save_dir]

    retarget_cwd = repo_root / "holosoma" / "src" / "holosoma_retargeting"
    subprocess.check_call(retarget_cmd, cwd=str(retarget_cwd))

    if args.no_play:
        return

    if args.retarget_save_dir:
        save_dir = Path(args.retarget_save_dir)
        if not save_dir.is_absolute():
            save_dir = (retarget_cwd / save_dir).resolve()
    else:
        save_dir = (retarget_cwd / DEFAULT_SAVE_DIRS[args.task_type].format(robot=args.robot)).resolve()

    retarget_npz = _retarget_output_path(args.task_type, save_dir, args.task_name, augmentation=False)
    if not retarget_npz.exists():
        raise FileNotFoundError(f"Retarget output not found: {retarget_npz}")

    player_script = (repo_root / args.player_script).resolve()
    if not player_script.exists():
        raise FileNotFoundError(f"MuJoCo player script not found: {player_script}")

    player_cwd = player_script.parent
    player_cmd = [args.python, str(player_script), str(retarget_npz)]

    if args.player_model:
        player_cmd += ["--model", args.player_model]
    else:
        model_candidates = [
            repo_root
            / "key-loco-man"
            / "source"
            / "isaaclab_assets"
            / "data"
            / "g1"
            / "g1_29dof_rev_1_0.xml",
            player_cwd / "g1_description" / "g1_29dof_rev_1_0.xml",
        ]
        for candidate in model_candidates:
            if candidate.exists():
                player_cmd += ["--model", str(candidate)]
                break

    if args.player_args:
        player_cmd += args.player_args

    subprocess.check_call(player_cmd, cwd=str(player_cwd))


if __name__ == "__main__":
    main()
