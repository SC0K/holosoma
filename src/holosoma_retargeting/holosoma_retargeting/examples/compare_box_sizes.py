"""Visualize an object mesh against a cube with a known physical size."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def add_mesh(
    axis,
    mesh: trimesh.Trimesh,
    color: str,
    alpha: float,
    label: str,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """Add a triangular mesh to a Matplotlib 3D axis."""
    vertices = np.asarray(mesh.vertices)
    if translation is not None:
        vertices = vertices + translation
    triangles = vertices[np.asarray(mesh.faces)]
    collection = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor=color,
        linewidth=0.35,
        alpha=alpha,
        label=label,
    )
    axis.add_collection3d(collection)
    return vertices


def configure_axis(axis, vertices: np.ndarray, title: str) -> None:
    """Use equal metric scaling on all three plot axes."""
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2.0
    half_range = max(maximum - minimum) * 0.6
    half_range = max(half_range, 0.2)

    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(title)
    axis.view_init(elev=22, azim=-48)


def create_comparison(
    mesh_path: Path,
    cube_size: float,
    output_path: Path,
    show: bool = False,
) -> None:
    """Render overlay and side-by-side physical-size comparisons."""
    loaded = trimesh.load(mesh_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"Could not load a triangle mesh from {mesh_path}")
    if cube_size <= 0:
        raise ValueError("cube_size must be positive")

    original = loaded.copy()
    original.apply_translation(-original.bounding_box.centroid)
    reference = trimesh.creation.box(extents=(cube_size, cube_size, cube_size))
    dimensions = np.asarray(original.extents)

    figure = plt.figure(figsize=(13, 6), constrained_layout=True)
    overlay_axis = figure.add_subplot(1, 2, 1, projection="3d")
    side_axis = figure.add_subplot(1, 2, 2, projection="3d")

    overlay_original = add_mesh(
        overlay_axis,
        original,
        color="#20A4C7",
        alpha=0.30,
        label="largebox.obj",
    )
    overlay_reference = add_mesh(
        overlay_axis,
        reference,
        color="#F28E2B",
        alpha=0.60,
        label=f"{cube_size * 100:.0f} cm cube",
    )
    configure_axis(
        overlay_axis,
        np.vstack((overlay_original, overlay_reference)),
        "Centered overlay",
    )
    overlay_axis.legend(loc="upper left")

    gap = 0.08
    original_translation = np.array([-(dimensions[0] / 2.0 + gap / 2.0), 0.0, 0.0])
    reference_translation = np.array([(cube_size / 2.0 + gap / 2.0), 0.0, 0.0])
    side_original = add_mesh(
        side_axis,
        original,
        color="#20A4C7",
        alpha=0.65,
        label="largebox.obj",
        translation=original_translation,
    )
    side_reference = add_mesh(
        side_axis,
        reference,
        color="#F28E2B",
        alpha=0.75,
        label=f"{cube_size * 100:.0f} cm cube",
        translation=reference_translation,
    )
    configure_axis(
        side_axis,
        np.vstack((side_original, side_reference)),
        "Side-by-side",
    )
    side_axis.legend(loc="upper left")

    figure.suptitle(
        "largebox.obj: "
        f"{dimensions[0] * 100:.1f} × {dimensions[1] * 100:.1f} × {dimensions[2] * 100:.1f} cm"
        f"   |   reference: {cube_size * 100:.1f} cm cube",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    print(f"largebox dimensions (m): {dimensions}")
    print(f"Saved comparison to: {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/largebox/largebox.obj",
    )
    parser.add_argument("--cube-size", type=float, default=0.30, help="Reference cube side length in meters")
    parser.add_argument("--output", type=Path, default=Path("largebox_vs_30cm_cube.png"))
    parser.add_argument("--show", action="store_true", help="Also open an interactive Matplotlib window")
    arguments = parser.parse_args()
    create_comparison(arguments.mesh, arguments.cube_size, arguments.output, arguments.show)


if __name__ == "__main__":
    main()
