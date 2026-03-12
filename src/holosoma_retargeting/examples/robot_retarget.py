"""
Unified robot retargeting script for all task types:
- robot_only: Robot-only retargeting with ground interaction
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain
"""

from __future__ import annotations

import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY, MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.utils import (  # noqa: E402
    augment_object_poses,
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    estimate_human_orientation,
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    load_intermimic_data,
    load_object_data,
    preprocess_motion_data,
    transform_from_human_to_world,
    transform_y_up_to_z_up,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------- Constants -----------------------------

# Task-specific defaults
DEFAULT_DATA_FORMATS = {
    "robot_only": "smplh",
    "object_interaction": "smplh",
    "climbing": "mocap",
}

DEFAULT_SAVE_DIRS = {
    "robot_only": "demo_results/{robot}/robot_only/omomo",
    "object_interaction": "demo_results/{robot}/object_interaction/omomo",
    "climbing": "demo_results/{robot}/climbing/mocap_climb",
}


# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])


# Type aliases
TaskType = Literal["robot_only", "object_interaction", "climbing"]
# DataFormat is imported from config_types.data_type


# ----------------------------- Helper Functions -----------------------------


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors (..., 3) to quaternions (..., 4) in [w, x, y, z]."""
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    small = angle < 1e-8
    axis = np.where(small, 0.0, rotvec / np.where(small, 1.0, angle))
    qw = np.cos(half)
    qxyz = axis * np.sin(half)
    quat = np.concatenate([qw, qxyz], axis=-1)
    quat = np.where(np.repeat(small, 4, axis=-1), np.array([1.0, 0.0, 0.0, 0.0]), quat)
    return quat


def _load_behave_object_poses(object_fit_path: Path) -> np.ndarray:
    """Load BEHAVE object trajectory as [qw, qx, qy, qz, x, y, z]."""
    if not object_fit_path.exists():
        raise FileNotFoundError(f"BEHAVE object params not found: {object_fit_path}")

    data = np.load(str(object_fit_path))
    rot_key = "angles" if "angles" in data else "rotvec" if "rotvec" in data else None
    trans_key = "trans" if "trans" in data else "translations" if "translations" in data else None

    if rot_key is None or trans_key is None:
        raise KeyError(
            f"{object_fit_path} must contain rotation key ('angles' or 'rotvec') "
            f"and translation key ('trans' or 'translations'). Found keys: {list(data.keys())}"
        )

    rotvec = np.asarray(data[rot_key])
    trans = np.asarray(data[trans_key])
    if rotvec.ndim != 2 or rotvec.shape[1] != 3:
        raise ValueError(f"Invalid rotation shape in {object_fit_path}: expected (T,3), got {rotvec.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Invalid translation shape in {object_fit_path}: expected (T,3), got {trans.shape}")
    if rotvec.shape[0] != trans.shape[0]:
        raise ValueError(
            f"Rotation/translation length mismatch in {object_fit_path}: {rotvec.shape[0]} vs {trans.shape[0]}"
        )

    quat_wxyz = _rotvec_to_quat_wxyz(rotvec)
    return np.concatenate([quat_wxyz, trans], axis=1)


def _resolve_behave_object_fit_path(data_path: Path, task_name: str, task_config: TaskConfig) -> Path:
    """Resolve BEHAVE object_fit_all.npz path for a sequence."""
    candidates = []
    if task_config.object_params_root is not None:
        candidates.append(task_config.object_params_root / task_name / task_config.object_params_filename)
    candidates.append(data_path / task_name / task_config.object_params_filename)
    candidates.append(data_path.parent / "behave-30fps-params-v1" / task_name / task_config.object_params_filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find BEHAVE object params file. Set --task-config.object-params-root. "
        f"Searched:\n{searched}"
    )


def _resolve_behave_info_path(data_path: Path, task_name: str, task_config: TaskConfig) -> Path:
    """Resolve BEHAVE info.json path for a sequence."""
    candidates = []
    if task_config.object_params_root is not None:
        candidates.append(task_config.object_params_root / task_name / "info.json")
    candidates.append(data_path / task_name / "info.json")
    candidates.append(data_path.parent / "behave-30fps-params-v1" / task_name / "info.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find BEHAVE info.json for {task_name}. Searched:\n{searched}")


def _resolve_behave_object_category(data_path: Path, task_name: str, task_config: TaskConfig) -> str:
    """Read BEHAVE object category from info.json."""
    info_path = _resolve_behave_info_path(data_path, task_name, task_config)
    info = json.loads(info_path.read_text())
    cat = info.get("cat")
    if not isinstance(cat, str) or not cat:
        raise KeyError(f"'cat' not found or invalid in {info_path}")
    return cat


def _resolve_behave_object_mesh_path(data_path: Path, object_cat: str, task_config: TaskConfig) -> Path:
    """Resolve BEHAVE object mesh path <objects>/<cat>/<cat>.obj."""
    roots = []
    if task_config.object_mesh_root is not None:
        roots.append(task_config.object_mesh_root)
    if task_config.object_params_root is not None:
        roots.append(task_config.object_params_root.parent / "objects")
    roots.append(data_path.parent / "objects")

    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / object_cat / f"{object_cat}.obj")
        candidates.append(root / object_cat / f"{object_cat}.ply")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find BEHAVE mesh for category '{object_cat}'. Searched:\n{searched}")


def _build_scene_xml_with_override_mesh(robot_urdf_file: str, object_name: str, object_mesh_path: Path, task_name: str) -> Path:
    """Create temp MuJoCo scene xml by overriding object mesh file path."""
    pkg_root = src_root / "holosoma_retargeting"
    robot_urdf_path = Path(robot_urdf_file)
    if not robot_urdf_path.is_absolute():
        robot_urdf_path = pkg_root / robot_urdf_path

    base_scene = Path(str(robot_urdf_path).replace(".urdf", f"_w_{object_name}.xml"))
    if not base_scene.exists() and object_name != "largebox":
        base_scene = Path(str(robot_urdf_path).replace(".urdf", "_w_largebox.xml"))
    if not base_scene.exists():
        raise FileNotFoundError(f"Base scene xml not found for object override: {base_scene}")

    tree = ET.parse(str(base_scene))
    root = tree.getroot()

    target_elem = None
    for mesh in root.findall(".//asset/mesh"):
        mesh_name = mesh.attrib.get("name", "")
        mesh_file = mesh.attrib.get("file", "")
        if mesh_name.endswith("_mesh") and ("largebox" in mesh_name or "largebox" in mesh_file):
            target_elem = mesh
            break
    if target_elem is None:
        raise ValueError(f"Could not find object mesh entry in {base_scene}")

    target_elem.set("file", str(object_mesh_path))
    # Keep the override XML next to the base scene so relative meshdir/assets paths remain valid.
    out_path = base_scene.parent / f"{base_scene.stem}_{task_name}_objoverride.xml"
    tree.write(str(out_path), encoding="utf-8", xml_declaration=False)
    return out_path


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
    task_type: str,
) -> SimpleNamespace:
    """Create combined task constants from robot and motion data configs.

    Args:
        robot_config: Robot configuration
        motion_data_config: Motion data format configuration
        task_config: Task-specific configuration
        task_type: Type of task ("robot_only", "object_interaction", "climbing")

    Returns:
        SimpleNamespace with all task constants
    """
    task_constants = SimpleNamespace()

    # Copy all attributes from robot_config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(robot_config, attr))

    # Copy legacy motion data constants (upper-case for compatibility)
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(task_constants, attr, value)

    # Task-specific object setup
    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
    elif task_type == "object_interaction":
        obj_name = task_config.object_name or "largebox"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = f"models/{obj_name}/{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = f"models/{obj_name}/{obj_name}.obj"
        task_constants.OBJECT_URDF_TEMPLATE = f"models/templates/{obj_name}.urdf.jinja"
    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.OBJECT_URDF_FILE = str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def validate_config(cfg: RetargetingConfig) -> None:
    """Validate configuration consistency.

    Args:
        cfg: Configuration arguments

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate that data_format exists in registry (if provided)
    if cfg.data_format is not None and cfg.data_format not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown data_format: '{cfg.data_format}'. "
            f"Available formats: {available}. "
            f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
        )

    # Task-specific format requirements
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap"):
        raise ValueError("Climbing task requires 'mocap' data format")
    if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh"):
        raise ValueError("Object interaction requires 'smplh' data format")
    # robot_only accepts any format in the registry (already validated above)


def create_ground_points(x_range: tuple[float, float], y_range: tuple[float, float], size: int) -> np.ndarray:
    """Create ground point meshgrid.

    Args:
        x_range: (min, max) x-coordinate range
        y_range: (min, max) y-coordinate range
        size: Number of points per dimension

    Returns:
        (N, 3) array of ground points
    """
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)


def load_motion_data(
    task_type: TaskType,
    data_format: str,
    data_path: Path,
    task_name: str,
    constants: SimpleNamespace,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load motion data based on task type and format.

    Args:
        task_type: Type of task
        data_format: Data format ("lafan", "smplh", "mocap")
        data_path: Path to data directory
        task_name: Name of the task/sequence
        constants: Task constants
        motion_data_config: Motion data configuration
        task_config: Task-specific config (used for BEHAVE object params lookup)

    Returns:
        Tuple of (human_joints, object_poses, smpl_scale)
        - human_joints: (T, J, 3) array of joint positions
        - object_poses: (T, 7) array of object poses [qw, qx, qy, qz, x, y, z]
        - smpl_scale: Scaling factor for SMPL compatibility

    Raises:
        FileNotFoundError: If required data files are not found
    """
    logger.info("Loading motion data for task: %s, format: %s", task_name, data_format)

    if task_type == "robot_only":
        if data_format == "lafan":
            npy_path = data_path / f"{task_name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"LAFAN data file not found: {npy_path}")

            human_joints = np.load(str(npy_path))
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            # LAFAN-specific spine adjustment
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "smplh":  # smplh
            pt_path = data_path / f"{task_name}.pt"
            npz_file = data_path / f"{task_name}.npz"
            if pt_path.exists():
                human_joints, object_poses = load_intermimic_data(str(pt_path))
                smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
            elif npz_file.exists():
                human_data = np.load(str(npz_file))
                human_joints = human_data["global_joint_positions"]
                human_height = human_data["height"]
                smpl_scale = constants.ROBOT_HEIGHT / human_height
            else:
                raise FileNotFoundError(f"SMPL-H data file not found: {pt_path} or {npz_file}")
        elif data_format == "mocap":
            downsample = 4
            npy_file = data_path / f"{task_name}.npy"
            if not npy_file.exists():
                raise FileNotFoundError(f"MOCAP data file not found: {npy_file}")

            human_joints = np.load(str(npy_file))[::downsample]

            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height
        elif data_format == "smplx":
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height
        else:
            # For other custom data format, if it uses consistent .npz file like SMPLX,
            # you can use the same logic as SMPLX.
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height

        # Create dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    elif task_type == "object_interaction":
        pt_path = data_path / f"{task_name}.pt"
        npz_file = data_path / f"{task_name}.npz"
        if pt_path.exists():
            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        elif npz_file.exists():
            human_data = np.load(str(npz_file))
            if "global_joint_positions" not in human_data or "height" not in human_data:
                raise KeyError(
                    f"SMPL-H npz for object_interaction must contain 'global_joint_positions' and 'height': {npz_file}"
                )

            human_joints = human_data["global_joint_positions"]
            human_height = float(np.asarray(human_data["height"]).reshape(-1)[0])
            smpl_scale = constants.ROBOT_HEIGHT / human_height

            object_fit_path = _resolve_behave_object_fit_path(data_path, task_name, task_config)
            object_poses = _load_behave_object_poses(object_fit_path)

            if human_joints.shape[0] != object_poses.shape[0]:
                n_frames = min(human_joints.shape[0], object_poses.shape[0])
                logger.warning(
                    "Frame count mismatch for %s: human=%d, object=%d. Truncating both to %d frames.",
                    task_name,
                    human_joints.shape[0],
                    object_poses.shape[0],
                    n_frames,
                )
                human_joints = human_joints[:n_frames]
                object_poses = object_poses[:n_frames]
        else:
            raise FileNotFoundError(f"Object interaction input not found: {pt_path} or {npz_file}")

    elif task_type == "climbing":
        task_dir = data_path / task_name
        npy_files = list(task_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {task_dir}")

        npy_file = npy_files[0]
        # MOCAP-specific downsample factor
        downsample = 4
        human_joints = np.load(str(npy_file))[::downsample]
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))
        default_human_height = motion_data_config.default_human_height or 1.78
        smpl_scale = constants.ROBOT_HEIGHT / default_human_height

    logger.debug(
        "Loaded %d frames, scale factor: %.4f",
        human_joints.shape[0],
        smpl_scale,
    )
    return human_joints, object_poses, smpl_scale


def setup_object_data(
    task_type: TaskType,
    constants: SimpleNamespace,
    object_dir: Path | None,
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
    object_scale_augmented: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Setup object-specific data (ground, object mesh, climbing terrain).
    Args:
        task_type: Type of task
        constants: Task constants
        object_dir: Object directory path (for climbing)
        smpl_scale: SMPL scaling factor
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        object_scale_augmented: Scale factor for augmented objects (default: [1.0, 1.0, 1.2])
    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path)
    """
    object_scale_normal = np.array([1.0, 1.0, 1.0])
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])  # For climbing task augmentation
    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(task_config.ground_range, task_config.ground_range, task_config.ground_size)
        return ground_pts, ground_pts, None

    if task_type == "object_interaction":
        # Load object data
        object_mesh_from_behave = getattr(constants, "BEHAVE_OBJECT_MESH_FILE", None)
        object_mesh_src = object_mesh_from_behave if object_mesh_from_behave is not None else constants.OBJECT_MESH_FILE
        if object_mesh_src is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")

        # Resolve package-relative paths so script works from any cwd.
        pkg_root = src_root / "holosoma_retargeting"
        object_mesh_file = Path(object_mesh_src)
        if not object_mesh_file.is_absolute():
            object_mesh_file = pkg_root / object_mesh_file
        object_urdf_file = Path(constants.OBJECT_URDF_FILE) if constants.OBJECT_URDF_FILE is not None else None
        if object_urdf_file is not None and not object_urdf_file.is_absolute():
            object_urdf_file = pkg_root / object_urdf_file

        object_local_pts, object_local_pts_demo = load_object_data(
            str(object_mesh_file), smpl_scale=smpl_scale, sample_count=100
        )
        return object_local_pts, object_local_pts_demo, str(object_urdf_file) if object_urdf_file is not None else None

    if task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")

        # Setup climbing-specific object
        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_name = Path(constants.ROBOT_URDF_FILE).name.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml")
        scene_xml_file = object_dir / scene_xml_name
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        print("object mesh file: ", constants.OBJECT_MESH_FILE)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: (
                task_config.surface_weight_high
                if p[2] > task_config.surface_weight_threshold
                else task_config.surface_weight_low
            ),
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range, task_config.climbing_ground_range, task_config.climbing_ground_size
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = object_scale_augmented
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = object_scale_normal
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        # Create scaled URDF and XML files
        scale_factors = tuple(float(value) for value in (object_scale * smpl_scale))
        object_urdf_file = create_scaled_multi_boxes_urdf(constants.OBJECT_URDF_FILE, scale_factors)
        object_asset_xml_path = create_scaled_multi_boxes_xml(str(box_asset_xml), scale_factors)
        new_scene_xml_path = create_new_scene_xml_file(str(scene_xml_file), scale_factors, object_asset_xml_path)
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file

    raise ValueError(f"Unknown task type: {task_type}")


def _compute_q_init_base(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter | None = None,
) -> np.ndarray:
    """Compute base robot pose initialization (q_init_base).
    This is a shared helper function used by both single and parallel processing.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses in format [qw, qx, qy, qz, x, y, z]
        constants: Task constants
        retargeter: Optional retargeter instance (needed for climbing)
    Returns:
        q_init_base in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints
    """
    if task_type == "robot_only":
        if data_format == "lafan":
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate(
                [human_joints[0, spine_joint_idx, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)]
            )
        else:  # smplh
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "object_interaction":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "climbing":
        if retargeter is None:
            raise ValueError("retargeter is required for climbing task")
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        spine_joint_idx = retargeter.demo_joints.index("Spine1")
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate(
            [
                human_joints[0, spine_joint_idx],
                human_quat_init,
                np.zeros(constants.ROBOT_DOF),
            ]
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return q_init_base


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert object poses from [qw, qx, qy, qz, x, y, z] to MuJoCo order [x, y, z, qw, qx, qy, qz].
    Args:
        object_poses: Object poses array of shape (T, 7) in format [qw, qx, qy, qz, x, y, z]
    Returns:
        Object poses array in MuJoCo order [x, y, z, qw, qx, qy, qz]
    """
    return object_poses[:, [4, 5, 6, 0, 1, 2, 3]]


def build_retargeter_kwargs_from_config(
    retargeter_config: RetargeterConfig,
    constants: SimpleNamespace,
    object_urdf_path: str | None,
    task_type: str,
) -> dict:
    """Build kwargs for InteractionMeshRetargeter from a RetargeterConfig.
    This is a convenience function that allows building kwargs directly from
    a RetargeterConfig without needing a full RetargetingConfig.
    Args:
        retargeter_config: Retargeter configuration
        constants: Task constants
        object_urdf_path: Path to object URDF file
        task_type: Type of task
    Returns:
        Dictionary of kwargs for InteractionMeshRetargeter
    """
    kwargs = {
        "task_constants": constants,
        "object_urdf_path": object_urdf_path,
        "q_a_init_idx": retargeter_config.q_a_init_idx,
        "activate_joint_limits": retargeter_config.activate_joint_limits,
        "activate_obj_non_penetration": retargeter_config.activate_obj_non_penetration,
        "activate_foot_sticking": retargeter_config.activate_foot_sticking,
        "penetration_tolerance": retargeter_config.penetration_tolerance,
        "foot_sticking_tolerance": retargeter_config.foot_sticking_tolerance,
        "step_size": retargeter_config.step_size,
        "visualize": retargeter_config.visualize,
        "debug": retargeter_config.debug,
        "w_nominal_tracking_init": retargeter_config.w_nominal_tracking_init,
    }
    if task_type == "climbing":
        kwargs["nominal_tracking_tau"] = retargeter_config.nominal_tracking_tau
    return kwargs


def initialize_robot_pose(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter,
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
    augmentation_translation: np.ndarray | None = None,
    augmentation_rotation: float | None = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize robot pose (q_init, q_nominal) based on task.
    Returns qpos in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints.
    Object poses are returned in MuJoCo order: [0:3] position, [3:7] quaternion.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses (assumed to be in format: [quat, pos] or [pos, quat])
        constants: Task constants
        retargeter: Retargeter instance
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        save_dir: Save directory path
        task_name: Task name
        augmentation_translation: Translation vector for augmentation (default: [0.2, 0.0, 0.0])
    Returns:
        Tuple of (q_init, q_nominal, object_poses_augmented, human_joints_modified, object_poses_modified)
        where qpos is in MuJoCo order and object_poses are in MuJoCo order
    """
    # Use default if not provided
    if augmentation_translation is None:
        augmentation_translation = _AUGMENTATION_TRANSLATION
    logger.info("Initializing robot pose")

    if task_type == "robot_only":
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    if task_type == "object_interaction":
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                augmentation_translation,
                augmentation_rotation,
            )
            # Convert object_poses to MuJoCo order
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            object_poses = convert_object_poses_to_mujoco_order(object_poses)

            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            return q_nominal[0], q_nominal, object_poses_augmented, human_joints, object_poses
        object_poses_augmented = object_poses.copy()
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
        return q_init, None, object_poses_augmented, human_joints, object_poses

    if task_type == "climbing":
        if augmentation:
            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            # Convert object_poses to MuJoCo order
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, retargeter)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    raise ValueError(f"Unknown task type: {task_type}")


def determine_output_path(
    task_type: TaskType,
    save_dir: Path,
    task_name: str,
    augmentation: bool,
) -> str:
    """Determine output file path based on task and augmentation.
    Args:
        task_type: Type of task
        save_dir: Save directory path
        task_name: Task name
        augmentation: Whether this is an augmentation run
    Returns:
        Output file path
    """
    if task_type == "robot_only":
        return str(save_dir / f"{task_name}.npz")
    if task_type in ("object_interaction", "climbing"):
        suffix = "_augmented" if augmentation else "_original"
        return str(save_dir / f"{task_name}{suffix}.npz")
    raise ValueError(f"Unknown task type: {task_type}")


# ----------------------------- Main -----------------------------


def main(cfg: RetargetingConfig) -> None:
    """Main retargeting pipeline.
    Args:
        cfg: Configuration arguments
    """
    # Validate configuration
    validate_config(cfg)

    robot = cfg.robot
    task_name = cfg.task_name
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format: str = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]
    save_dir = cfg.save_dir if cfg.save_dir is not None else Path(DEFAULT_SAVE_DIRS[task_type].format(robot=robot))
    data_path = cfg.data_path

    os.makedirs(save_dir, exist_ok=True)
    logger.info("Task: %s, Type: %s, Format: %s", task_name, task_type, data_format)
    logger.info("Data path: %s, Save dir: %s", data_path, save_dir)

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot)

    # Task-specific object setup: set default object_dir for climbing if not provided
    if task_type == "climbing" and cfg.task_config.object_dir is None:
        from dataclasses import replace

        cfg.task_config = replace(cfg.task_config, object_dir=data_path / task_name)

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=task_type,
    )

    # BEHAVE object metadata/mesh auto-resolution for object_interaction with SMPL-H data.
    if task_type == "object_interaction" and data_format == "smplh":
        try:
            behave_cat = _resolve_behave_object_category(data_path, task_name, cfg.task_config)
            behave_mesh = _resolve_behave_object_mesh_path(data_path, behave_cat, cfg.task_config)
            constants.BEHAVE_OBJECT_CATEGORY = behave_cat
            constants.BEHAVE_OBJECT_MESH_FILE = str(behave_mesh)
            constants.ROBOT_SCENE_XML_FILE = str(
                _build_scene_xml_with_override_mesh(
                    constants.ROBOT_URDF_FILE,
                    constants.OBJECT_NAME,
                    behave_mesh,
                    task_name,
                )
            )
            logger.info("BEHAVE object category: %s", behave_cat)
            logger.info("BEHAVE object mesh: %s", behave_mesh)
            logger.info("Using scene XML override: %s", constants.ROBOT_SCENE_XML_FILE)
        except Exception as e:
            logger.warning("Could not auto-resolve BEHAVE object mesh/scene override: %s", e)

    # Load motion data
    human_joints, object_poses, smpl_scale = load_motion_data(
        task_type, data_format, data_path, task_name, constants, cfg.motion_data_config, cfg.task_config
    )

    # Get toe names from motion data config (depends only on data_format)
    toe_names = cfg.motion_data_config.toe_names

    # Setup object data
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type,
        constants,
        cfg.task_config.object_dir,
        smpl_scale,
        cfg.task_config,
        cfg.augmentation,
        object_scale_augmented=_OBJECT_SCALE_AUGMENTED,
    )

    # Create retargeter
    retargeter_kwargs = build_retargeter_kwargs_from_config(cfg.retargeter, constants, object_urdf_path, task_type)
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)
    logger.info("Retargeter created")

    # Preprocess motion data
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)
    elif task_type in {"object_interaction", "climbing"}:
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            scale=smpl_scale,
            object_poses=object_poses,
        )

    # Initialize robot pose
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        task_type,
        data_format,
        human_joints,
        object_poses,
        constants,
        retargeter,
        cfg.task_config,
        cfg.augmentation,
        save_dir,
        task_name,
        augmentation_translation=_AUGMENTATION_TRANSLATION,
    )

    # Extract foot sticking sequences
    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(human_joints, retargeter.demo_joints, toe_names)

    # Task-specific foot sticking adjustments
    if task_type == "object_interaction":
        # Disable initial sticking
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    # Determine output path
    dest_res_path = determine_output_path(task_type, save_dir, task_name, cfg.augmentation)

    # Retarget motion
    logger.info("Starting retargeting...")
    retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses_augmented,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
        fps=cfg.fps,
    )
    logger.info("Retargeting complete. Results saved to: %s", dest_res_path)

    if cfg.retargeter.debug:
        input("Press Enter to exit ...")


if __name__ == "__main__":
    cfg = tyro.cli(RetargetingConfig)
    main(cfg)
