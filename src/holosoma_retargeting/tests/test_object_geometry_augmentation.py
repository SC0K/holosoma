from pathlib import Path

import numpy as np
import trimesh

from holosoma_retargeting.examples.parallel_robot_retarget import generate_augmentation_configs
from holosoma_retargeting.src.utils import (
    create_object_variant_scene_xml,
    create_object_variant_urdf,
    map_points_between_box_meshes,
)


def test_map_points_between_different_box_meshes(tmp_path: Path) -> None:
    source_path = tmp_path / "source.obj"
    target_path = tmp_path / "target.obj"
    trimesh.creation.box(extents=(2.0, 4.0, 6.0)).export(source_path)
    trimesh.creation.box(extents=(1.0, 8.0, 3.0)).export(target_path)

    points = np.array([[1.0, 0.0, 0.0], [0.0, -2.0, 3.0]])
    mapped = map_points_between_box_meshes(points, source_path, target_path)

    np.testing.assert_allclose(mapped, [[0.5, 0.0, 0.0], [0.0, -4.0, 1.5]])


def test_generated_geometry_files_reference_target_mesh(tmp_path: Path) -> None:
    source_mesh = tmp_path / "source.obj"
    target_mesh = tmp_path / "target.obj"
    source_mesh.write_text("v 0 0 0\n")
    target_mesh.write_text("v 0 0 0\n")
    urdf = tmp_path / "object.urdf"
    urdf.write_text(
        '<robot name="box"><link name="box"><visual><geometry>'
        '<mesh filename="source.obj" scale="1 1 1"/>'
        "</geometry></visual></link></robot>"
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><compiler meshdir="."/><asset>'
        '<mesh name="box_mesh" file="source.obj" scale="1 1 1"/>'
        "</asset><worldbody/></mujoco>"
    )

    output_urdf = Path(
        create_object_variant_urdf(urdf, tmp_path / "generated/object.urdf", mesh_path=target_mesh)
    )
    output_scene = Path(
        create_object_variant_scene_xml(
            scene,
            "box_mesh",
            tmp_path / "generated/scene.xml",
            mesh_path=target_mesh,
        )
    )

    assert str(target_mesh.resolve()) in output_urdf.read_text()
    assert str(target_mesh.resolve()) in output_scene.read_text()


def test_object_interaction_configs_include_geometry_variants() -> None:
    class Config:
        object_augmentation_mode = "all"
        object_scale_variants = ((0.8, 0.9, 1.0),)
        object_mesh_variants = (Path("small_box.obj"),)

    configs = generate_augmentation_configs("object_interaction", True, Config())
    by_name = {config["name"]: config for config in configs}

    np.testing.assert_allclose(by_name["scale_0.80_0.90_1.00"]["scale"], [0.8, 0.9, 1.0])
    assert by_name["mesh_0_small_box"]["mesh"] == Path("small_box.obj")


def test_scale_mode_only_generates_box_sizes() -> None:
    class Config:
        object_augmentation_mode = "scale"
        object_scale_variants = ((0.8, 0.8, 0.8), (1.2, 1.2, 1.2))
        object_mesh_variants = (Path("unused.obj"),)

    configs = generate_augmentation_configs("object_interaction", True, Config())

    assert [config["name"] for config in configs] == [
        "original",
        "scale_0.80_0.80_0.80",
        "scale_1.20_1.20_1.20",
    ]
