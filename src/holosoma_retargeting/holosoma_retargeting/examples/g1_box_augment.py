"""Augment object size directly from an existing G1 retargeted trajectory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import (
    build_retargeter_kwargs_from_config,
    create_task_constants,
    setup_object_data,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.qpos_layout import convert_qpos_layout, detect_qpos_layout


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def output_name(input_path: Path, scale: np.ndarray) -> Path:
    """Construct a descriptive default output filename."""
    scale_name = "_".join(f"{value:.2f}" for value in scale)
    stem = input_path.stem.removesuffix("_original")
    return input_path.with_name(f"{stem}_g1_scale_{scale_name}.npz")


def augment_g1_box(
    input_path: Path,
    scale_values: tuple[float, float, float] | np.ndarray,
    output_path: Path | None = None,
    *,
    object_name: str = "largebox",
    max_frames: int | None = None,
    foot_velocity_threshold: float = 0.01,
    no_foot_sticking: bool = False,
    visualize: bool = False,
    debug: bool = False,
    input_layout: str = "auto",
) -> tuple[Path, Path]:
    """Run one G1-to-G1 box augmentation and return motion/URDF paths."""
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input trajectory not found: {input_path}")
    scale = np.asarray(scale_values, dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError("scale_values must contain three positive values")

    with np.load(input_path, allow_pickle=False) as data:
        if "qpos" not in data:
            raise KeyError(f"{input_path} does not contain a 'qpos' array")
        source_qpos = np.asarray(data["qpos"], dtype=float)
        fps = float(data["fps"]) if "fps" in data else 30.0
        stored_layout = str(data["qpos_layout"].item()) if "qpos_layout" in data else None
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        source_qpos = source_qpos[:max_frames]
    if source_qpos.ndim != 2 or source_qpos.shape[1] < 14:
        raise ValueError(f"Expected a 2D robot-object qpos array, got {source_qpos.shape}")
    if input_layout not in {"auto", "native", "omniretarget"}:
        raise ValueError("input_layout must be 'auto', 'native', or 'omniretarget'")
    resolved_layout = stored_layout if input_layout == "auto" and stored_layout else input_layout
    if resolved_layout == "auto":
        resolved_layout = detect_qpos_layout(source_qpos)
    if resolved_layout not in {"native", "omniretarget"}:
        raise ValueError(f"Unsupported stored qpos_layout metadata: {resolved_layout!r}")
    source_qpos = convert_qpos_layout(source_qpos, resolved_layout, "native")
    print(f"Input qpos layout: {resolved_layout} (converted to native)")

    output_path = output_path.resolve() if output_path else output_name(input_path, scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_name = "scale_" + "_".join(f"{value:.2f}" for value in scale)

    robot_config = RobotConfig(
        robot_type="g1",
        robot_urdf_file=str(PACKAGE_ROOT / "models/g1/g1_29dof.urdf"),
    )
    motion_config = MotionDataConfig(data_format="smplh", robot_type="g1")
    task_config = TaskConfig(object_name=object_name)
    constants = create_task_constants(robot_config, motion_config, task_config, "object_interaction")
    constants.OBJECT_URDF_FILE = str(
        PACKAGE_ROOT / f"models/{object_name}/{object_name}.urdf"
    )
    constants.OBJECT_MESH_FILE = str(
        PACKAGE_ROOT / f"models/{object_name}/{object_name}.obj"
    )

    generated_dir = output_path.parent / "_generated_objects" / output_path.stem
    target_object_points, source_object_points, object_urdf_path = setup_object_data(
        "object_interaction",
        constants,
        None,
        smpl_scale=1.0,
        task_config=task_config,
        augmentation=True,
        object_scale_augmented=scale,
        generated_object_dir=generated_dir,
        augmentation_name=scale_name,
    )

    retargeter_config = RetargeterConfig(
        activate_foot_sticking=not no_foot_sticking,
        visualize=visualize,
        debug=debug,
    )
    retargeter = InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(
            retargeter_config,
            constants,
            object_urdf_path,
            "object_interaction",
        )
    )
    if source_qpos.shape[1] != retargeter.nq:
        raise ValueError(
            f"Input qpos width {source_qpos.shape[1]} does not match the G1-object model width {retargeter.nq}"
        )

    if no_foot_sticking:
        foot_sticking = [{"L_Toe": False, "R_Toe": False} for _ in source_qpos]
    else:
        foot_sticking = retargeter.extract_robot_foot_sticking_sequence(
            source_qpos,
            velocity_threshold=foot_velocity_threshold,
        )

    object_poses = source_qpos[:, -7:].copy()
    retargeter.retarget_motion(
        human_joint_motions=None,
        source_robot_motions=source_qpos,
        object_poses=object_poses,
        object_poses_augmented=object_poses,
        object_points_local_demo=source_object_points,
        object_points_local=target_object_points,
        foot_sticking_sequences=foot_sticking,
        q_a_init=source_qpos[0],
        q_nominal_list=source_qpos,
        original=False,
        dest_res_path=str(output_path),
        fps=fps,
        output_metadata={
            "qpos_layout": np.asarray("native"),
            "source_qpos_layout": np.asarray(resolved_layout),
            "box_augmentation_version": np.asarray(2, dtype=np.int64),
            "object_scale": scale,
            "foot_sticking_enabled": np.asarray(not no_foot_sticking),
        },
    )

    print(f"Saved G1-to-G1 augmentation to: {output_path}")
    print(f"Matching object URDF: {object_urdf_path}")
    return output_path, Path(object_urdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-optimize a retargeted G1 object motion for a scaled box without human data."
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--scale", type=float, nargs=3, metavar=("SX", "SY", "SZ"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--object-name", default="largebox")
    parser.add_argument("--max-frames", type=int, help="Optional prefix length, useful for testing")
    parser.add_argument("--foot-velocity-threshold", type=float, default=0.01)
    parser.add_argument("--no-foot-sticking", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--input-layout",
        choices=("auto", "native", "omniretarget"),
        default="auto",
        help="Input qpos field order; auto uses metadata or quaternion norms",
    )
    arguments = parser.parse_args()
    augment_g1_box(
        arguments.input_npz,
        np.asarray(arguments.scale),
        arguments.output,
        object_name=arguments.object_name,
        max_frames=arguments.max_frames,
        foot_velocity_threshold=arguments.foot_velocity_threshold,
        no_foot_sticking=arguments.no_foot_sticking,
        visualize=arguments.visualize,
        debug=arguments.debug,
        input_layout=arguments.input_layout,
    )


if __name__ == "__main__":
    main()
