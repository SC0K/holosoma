#!/usr/bin/env python3
"""Prepare the tracked box and bucket assets for G1 mesh retargeting."""

from __future__ import annotations

from pathlib import Path

import trimesh


ROOT = Path(__file__).resolve().parents[1]


def _write_urdf(model_dir: Path, name: str) -> None:
    (model_dir / f"{name}.urdf").write_text(
        f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}_link">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><mesh filename="{name}.obj" scale="1 1 1"/></geometry>
      <material name="mocap_object"><color rgba="0.65 0.55 0.35 0.85"/></material>
    </visual>
    <collision name="{name}">
      <geometry><mesh filename="{name}.obj" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )


def _write_bucket_urdf(bucket_dir: Path) -> None:
    """Write the full bucket with each convex proxy part kept separate."""
    proxy_dir = "bucket_proxy_minimal_v1"
    visual_parts = (
        "part_00_solid_body.obj",
        "part_01_handle_pos_lower.obj",
        "part_02_handle_pos_upper.obj",
        "part_03_handle_neg_lower.obj",
        "part_04_handle_neg_upper.obj",
        "part_05_grip.obj",
    )
    visuals = "\n".join(
        f"""    <visual name="{Path(part).stem}">
      <geometry><mesh filename="{proxy_dir}/{part}" scale="1 1 1"/></geometry>
      <material name="bucket_proxy"><color rgba="0.65 0.55 0.35 0.85"/></material>
    </visual>"""
        for part in visual_parts
    )
    collisions = "\n".join(
        f"""    <collision name="{Path(part).stem}">
      <geometry><mesh filename="{proxy_dir}/{part}" scale="1 1 1"/></geometry>
    </collision>"""
        for part in visual_parts
    )
    (bucket_dir / "bucket.urdf").write_text(
        f"""<?xml version="1.0"?>
<robot name="bucket">
  <link name="bucket_link">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
{visuals}
{collisions}
  </link>
</robot>
""",
        encoding="utf-8",
    )


def _write_g1_scene(name: str) -> None:
    source = ROOT / "models/g1/g1_29dof_w_largebox.xml"
    destination = ROOT / f"models/g1/g1_29dof_w_{name}.xml"
    scene = source.read_text(encoding="utf-8").replace("largebox", name)
    if name == "bucket":
        proxy_parts = (
            "part_00_solid_body.obj",
            "part_01_handle_pos_lower.obj",
            "part_02_handle_pos_upper.obj",
            "part_03_handle_neg_lower.obj",
            "part_04_handle_neg_upper.obj",
            "part_05_grip.obj",
        )
        single_asset = '    <mesh name="bucket_mesh" file="../../bucket/bucket.obj" scale="1 1 1"/>'
        separate_assets = "\n".join(
            f'    <mesh name="bucket_proxy_{i:02d}" '
            f'file="../../bucket/bucket_proxy_minimal_v1/{part}" scale="1 1 1"/>'
            for i, part in enumerate(proxy_parts)
        )
        scene = scene.replace(single_asset, separate_assets)

        single_geom = """        <geom name="bucket" type="mesh" mesh="bucket_mesh"
                contype="1" conaffinity="1"
                pos="0 0 0" quat="1 0 0 0"
                rgba="0.7 0.8 0.9 0.7"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>"""
        separate_geoms = "\n".join(
            f"""        <geom name="bucket_part_{i:02d}" type="mesh" mesh="bucket_proxy_{i:02d}"
                contype="1" conaffinity="1"
                pos="0 0 0" quat="1 0 0 0"
                rgba="0.7 0.8 0.9 0.7"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>"""
            for i in range(len(proxy_parts))
        )
        if single_asset in scene or single_geom not in scene:
            raise RuntimeError("Could not specialize the generated bucket collision scene")
        scene = scene.replace(single_geom, separate_geoms)
    destination.write_text(scene, encoding="utf-8")


def main() -> None:
    box_dir = ROOT / "models/mocap_box"
    box_dir.mkdir(parents=True, exist_ok=True)
    # Dimensions supplied with the recording: X=35 cm, Y=25 cm, Z=28.5 cm.
    trimesh.creation.box(extents=(0.35, 0.25, 0.285)).export(box_dir / "mocap_box.obj")
    _write_urdf(box_dir, "mocap_box")
    _write_g1_scene("mocap_box")

    bucket_dir = ROOT / "models/bucket"
    # Use the handle and grip as the interaction/collision proxy. The solid
    # bucket body is deliberately excluded so the retargeter concentrates on
    # placing the robot hand/wand relative to the handle assembly.
    proxy_dir = bucket_dir / "bucket_proxy_minimal_v1"
    handle_parts = [
        proxy_dir / "part_01_handle_pos_lower.obj",
        proxy_dir / "part_02_handle_pos_upper.obj",
        proxy_dir / "part_03_handle_neg_lower.obj",
        proxy_dir / "part_04_handle_neg_upper.obj",
        proxy_dir / "part_05_grip.obj",
    ]
    missing_parts = [part for part in handle_parts if not part.exists()]
    if missing_parts:
        raise FileNotFoundError(", ".join(map(str, missing_parts)))
    handle_mesh = trimesh.util.concatenate(
        [trimesh.load(part, force="mesh", process=False) for part in handle_parts]
    )
    handle_mesh.export(bucket_dir / "bucket.obj")
    _write_bucket_urdf(bucket_dir)
    _write_g1_scene("bucket")
    print("Prepared mocap_box and bucket OBJ/URDF/G1 MuJoCo assets")


if __name__ == "__main__":
    main()
