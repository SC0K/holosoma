#!/usr/bin/env python3
"""Convert OptiTrack-style FBX interaction takes to HoloSoma LAFAN data.

Run this script with Blender, for example::

    blender -b --python data_utils/convert_fbx_interaction_to_lafan.py -- \
        --input "demo_data/mocap/fbx pickup" \
        --output-dir demo_data/mocap/lafan_box --object-name carton_box

The exported BVH preserves the complete source skeleton.  The NPY/NPZ files
contain the exact 22-joint ordering expected by HoloSoma's ``lafan`` format.
NPY uses the README-compatible Y-up convention; NPZ additionally stores Z-up
joint positions and the tracked object pose for object-interaction retargeting.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import bpy
import numpy as np


LAFAN_JOINTS = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="FBX file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-name", required=True, help="Tracked FBX object name")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--downsample", type=int, default=2, help="2 converts 60 Hz to 30 Hz")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _safe_stem(path: Path, prefix: str) -> str:
    stem = path.stem
    match = re.search(r"bucket(?:_(\d+))?$", stem, flags=re.IGNORECASE)
    if match:
        suffix = match.group(1)
        return f"{prefix or 'bucket'}{('_' + suffix) if suffix else ''}"
    return prefix or re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_").lower()


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.armatures, bpy.data.meshes, bpy.data.actions):
        for block in list(datablocks):
            datablocks.remove(block)


def _source_bones(armature: bpy.types.Object) -> dict[str, str]:
    result = {}
    for bone in armature.pose.bones:
        short_name = re.sub(r"^Skeleton(?: \d+)?_", "", bone.name)
        result[short_name] = bone.name
    return result


def _frame_end(armature: bpy.types.Object) -> int:
    animation = armature.animation_data
    if animation is None or animation.action is None:
        raise ValueError("performer armature has no active action")
    return int(round(animation.action.frame_range[1]))


def _object_pose(obj: bpy.types.Object, frame_root: bpy.types.Object | None) -> np.ndarray:
    matrix = obj.matrix_world
    quat = matrix.to_quaternion()
    # FBX rigid bodies are children of a fixed +90-degree X conversion node.
    # Remove that basis rotation on the right so a Z-up mesh has identity pose
    # when the tracked object is upright, while retaining its world position.
    if frame_root is not None:
        quat = quat @ frame_root.matrix_world.to_quaternion().inverted()
    quat.normalize()
    return np.asarray((quat.w, quat.x, quat.y, quat.z, *matrix.translation), dtype=np.float64)


def _convert_one(path: Path, cfg: argparse.Namespace) -> bool:
    _clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(path))
    armature = next((obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        print(f"SKIP {path.name}: no performer armature")
        return False

    object_tracker = bpy.data.objects.get(cfg.object_name)
    if object_tracker is None:
        raise ValueError(f"tracked object {cfg.object_name!r} not found in {path.name}")
    object_root = bpy.data.objects.get(f"{cfg.object_name}_Root")
    source_bones = _source_bones(armature)
    missing = sorted(set(LAFAN_JOINTS) - {"Spine2"} - set(source_bones))
    if missing:
        raise ValueError(f"missing required bones in {path.name}: {missing}")

    start, end = 1, _frame_end(armature)
    frame_ids = list(range(start, end + 1, cfg.downsample))
    joints = np.empty((len(frame_ids), len(LAFAN_JOINTS), 3), dtype=np.float64)
    object_poses = np.empty((len(frame_ids), 7), dtype=np.float64)
    previous_quat = None
    for output_frame, source_frame in enumerate(frame_ids):
        bpy.context.scene.frame_set(source_frame)
        for joint_index, joint_name in enumerate(LAFAN_JOINTS):
            if joint_name == "Spine2":
                spine1 = armature.matrix_world @ armature.pose.bones[source_bones["Spine1"]].matrix
                neck = armature.matrix_world @ armature.pose.bones[source_bones["Neck"]].matrix
                joints[output_frame, joint_index] = 0.5 * (np.asarray(spine1.translation) + np.asarray(neck.translation))
            else:
                pose_matrix = armature.matrix_world @ armature.pose.bones[source_bones[joint_name]].matrix
                joints[output_frame, joint_index] = pose_matrix.translation
        object_poses[output_frame] = _object_pose(object_tracker, object_root)
        if previous_quat is not None and np.dot(previous_quat, object_poses[output_frame, :4]) < 0:
            object_poses[output_frame, :4] *= -1
        previous_quat = object_poses[output_frame, :4].copy()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(path, cfg.output_prefix)

    # Keep a source-fidelity BVH for archival/debugging.  FBX imports into
    # Blender in metres and Z-up; scale 100 writes conventional BVH centimetres.
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.context.scene.render.fps = 60
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    bpy.ops.export_anim.bvh(
        filepath=str(cfg.output_dir / f"{stem}.bvh"),
        frame_start=start,
        frame_end=end,
        global_scale=100.0,
        rotate_mode="NATIVE",
        root_transform_only=True,
    )

    # The README LAFAN loader swaps Y/Z.  Store NPY in that convention and
    # retain native Z-up data in NPZ for the interaction loader.
    joints_y_up = joints[..., [0, 2, 1]]
    np.save(str(cfg.output_dir / f"{stem}.npy"), joints_y_up.astype(np.float32))
    np.savez_compressed(
        str(cfg.output_dir / f"{stem}.npz"),
        global_joint_positions=joints.astype(np.float32),
        object_poses=object_poses.astype(np.float32),
        joint_names=np.asarray(LAFAN_JOINTS),
        source_fps=np.asarray(60.0),
        fps=np.asarray(60.0 / cfg.downsample),
        source_fbx=np.asarray(str(path)),
        object_name=np.asarray(cfg.object_name),
    )
    print(f"OK {path.name}: {len(frame_ids)} frames -> {stem}.bvh/.npy/.npz")
    return True


def main() -> None:
    cfg = _arguments()
    if cfg.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    paths = [cfg.input] if cfg.input.is_file() else sorted(cfg.input.glob("*.fbx"))
    if not paths:
        raise FileNotFoundError(f"no FBX files found under {cfg.input}")
    converted = sum(_convert_one(path, cfg) for path in paths)
    print(f"Converted {converted}/{len(paths)} FBX takes")


if __name__ == "__main__":
    main()
