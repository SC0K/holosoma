"""Configuration types for viser visualization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViserConfig:
    """Configuration for viser player visualization.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with default values.
    """

    qpos_npz: str = "rt_results/OMOMO_new/box_parallel/sub8_largebox_051_original.npz"
    """Path to .npz file with qpos data."""

    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Path to robot URDF file."""

    object_urdf: str | None = None
    """Path to object URDF file (optional)."""

    source_motion_npz: str | None = None
    """Optional source NPZ containing global_joint_positions and joint_names."""

    source_object_mesh: str | None = None
    """Optional mesh for the original object's source-pose track."""

    source_skeleton_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """XYZ offset applied to the source skeleton, useful for side-by-side viewing."""

    source_skeleton_point_size: float = 0.025
    """Displayed source-skeleton joint size in meters."""

    source_skeleton_line_width: float = 4.0
    """Displayed source-skeleton bone width."""

    show_object_axes: bool = True
    """Whether to render XYZ axes for the object frame."""

    fps: int = 30
    """Frames per second for playback."""

    assume_object_in_qpos: bool = True
    """Whether object pose is included in qpos array."""

    loop: bool = False
    """Whether to loop playback."""

    show_meshes: bool = True
    """Whether to show mesh visualizations."""

    show_source_skeleton: bool = True
    """Whether to show the original motion skeleton when provided."""

    show_source_object: bool = True
    """Whether to show the original object mesh when provided."""

    grid_width: float = 8.0
    """Grid width for visualization."""

    grid_height: float = 8.0
    """Grid height for visualization."""

    visual_fps_multiplier: int = 2
    """Visual FPS multiplier for interpolation."""

    min_fps: int = 1
    """Minimum FPS setting."""

    max_fps: int = 240
    """Maximum FPS setting."""

    min_interp_mult: int = 1
    """Minimum interpolation multiplier."""

    max_interp_mult: int = 8
    """Maximum interpolation multiplier."""
