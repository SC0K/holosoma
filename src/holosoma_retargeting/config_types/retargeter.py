"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    activate_foot_grounding: bool = False
    """Whether to softly ground detected stance-foot collision spheres in Z."""

    interpolate_failed_frames: bool = False
    """Continue after infeasible solver frames and interpolate their robot poses."""

    foot_contact_height_threshold: float = 0.04
    """Maximum source-toe height above the estimated floor for stance detection."""

    foot_contact_velocity_threshold: float = 0.01
    """Maximum per-frame source-toe XY displacement for stance detection."""

    foot_ground_height: float = 0.005
    """Target Z of G1 sole-sphere centers; 0.005 matches their 5 mm radius."""

    foot_ground_weight: float = 1000.0
    """Weight of the soft stance-foot vertical grounding objective."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
