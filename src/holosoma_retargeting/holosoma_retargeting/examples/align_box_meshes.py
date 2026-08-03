"""Interactively align a replacement object mesh to the original object frame."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load one triangle mesh, flattening a scene if necessary."""
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Could not load a triangle mesh from {path}")
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {path}")
    return mesh


def quaternion_wxyz(rotation: Rotation) -> np.ndarray:
    """Convert SciPy's XYZW quaternion to WXYZ."""
    xyzw = rotation.as_quat()
    return xyzw[[3, 0, 1, 2]]


def run_alignment_ui(
    original_path: Path,
    target_path: Path,
    output_path: Path,
    port: int,
) -> None:
    """Start a Viser UI for estimating the target-to-original rigid transform."""
    original_path = original_path.resolve()
    target_path = target_path.resolve()
    original = load_mesh(original_path)
    target = load_mesh(target_path)

    original_center = np.asarray(original.bounding_box.centroid)
    target_center = np.asarray(target.bounding_box.centroid)
    original_dimensions = np.asarray(original.extents)
    target_dimensions = np.asarray(target.extents)

    server = viser.ViserServer(port=port)
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_frame(
        "/original_frame",
        show_axes=True,
        axes_length=0.18,
        axes_radius=0.004,
    )
    target_frame = server.scene.add_frame(
        "/target_frame",
        show_axes=True,
        axes_length=0.18,
        axes_radius=0.004,
        origin_color=(255, 140, 30),
    )

    server.scene.add_mesh_simple(
        "/original_frame/mesh",
        vertices=np.asarray(original.vertices, dtype=np.float32),
        faces=np.asarray(original.faces, dtype=np.int32),
        color=(30, 164, 199),
        wireframe=True,
        side="double",
    )
    server.scene.add_mesh_simple(
        "/target_frame/mesh",
        vertices=np.asarray(target.vertices, dtype=np.float32),
        faces=np.asarray(target.faces, dtype=np.int32),
        color=(242, 142, 43),
        opacity=0.55,
        side="double",
    )

    server.gui.add_markdown(
        f"### Mesh orientation alignment\n"
        f"**Original:** `{original_path.name}` — "
        f"{' × '.join(f'{value * 100:.1f}' for value in original_dimensions)} cm  \n"
        f"**Target:** `{target_path.name}` — "
        f"{' × '.join(f'{value * 100:.1f}' for value in target_dimensions)} cm\n\n"
        "Adjust the target (orange) until it overlaps the original (blue wireframe)."
    )

    with server.gui.add_folder("Orientation offset (degrees)"):
        roll_slider = server.gui.add_slider(
            "Roll X",
            min=-180.0,
            max=180.0,
            step=0.5,
            initial_value=0.0,
        )
        pitch_slider = server.gui.add_slider(
            "Pitch Y",
            min=-180.0,
            max=180.0,
            step=0.5,
            initial_value=0.0,
        )
        yaw_slider = server.gui.add_slider(
            "Yaw Z",
            min=-180.0,
            max=180.0,
            step=0.5,
            initial_value=0.0,
        )

    with server.gui.add_folder("Translation offset (meters)"):
        lock_centers = server.gui.add_checkbox("Lock bounding-box centers", initial_value=True)
        x_slider = server.gui.add_slider("X", min=-0.5, max=0.5, step=0.001, initial_value=0.0)
        y_slider = server.gui.add_slider("Y", min=-0.5, max=0.5, step=0.001, initial_value=0.0)
        z_slider = server.gui.add_slider("Z", min=-0.5, max=0.5, step=0.001, initial_value=0.0)

    with server.gui.add_folder("Actions"):
        reset_button = server.gui.add_button("Reset")
        save_button = server.gui.add_button("Save offset JSON")

    status = server.gui.add_markdown("")
    current_transform: dict[str, object] = {}

    def update_target() -> None:
        euler_degrees = np.array(
            [roll_slider.value, pitch_slider.value, yaw_slider.value],
            dtype=float,
        )
        rotation = Rotation.from_euler("xyz", euler_degrees, degrees=True)
        rotation_matrix = rotation.as_matrix()
        delta_translation = np.array(
            [x_slider.value, y_slider.value, z_slider.value],
            dtype=float,
        )
        if lock_centers.value:
            center_translation = original_center - rotation_matrix @ target_center
        else:
            center_translation = np.zeros(3)
        translation = center_translation + delta_translation
        wxyz = quaternion_wxyz(rotation)

        target_frame.wxyz = wxyz
        target_frame.position = translation

        current_transform.clear()
        current_transform.update(
            {
                "source_mesh": str(original_path),
                "target_mesh": str(target_path),
                "euler_xyz_degrees": euler_degrees.tolist(),
                "quaternion_wxyz": wxyz.tolist(),
                "rotation_matrix": rotation_matrix.tolist(),
                "translation_xyz_m": translation.tolist(),
                "center_lock_enabled": bool(lock_centers.value),
                "translation_slider_xyz_m": delta_translation.tolist(),
                "transform_convention": "p_original = R_offset @ p_target + translation_xyz_m",
            }
        )
        status.content = (
            "### Current target → original offset\n"
            f"- RPY XYZ: `{euler_degrees[0]:.1f}°, {euler_degrees[1]:.1f}°, {euler_degrees[2]:.1f}°`\n"
            f"- Quaternion WXYZ: `{np.array2string(wxyz, precision=6, separator=', ')}`\n"
            f"- Translation XYZ: `{np.array2string(translation, precision=6, separator=', ')} m`"
        )

    for control in (
        roll_slider,
        pitch_slider,
        yaw_slider,
        lock_centers,
        x_slider,
        y_slider,
        z_slider,
    ):

        @control.on_update
        def _on_update(_event) -> None:
            update_target()

    @reset_button.on_click
    def _on_reset(_event) -> None:
        roll_slider.value = 0.0
        pitch_slider.value = 0.0
        yaw_slider.value = 0.0
        x_slider.value = 0.0
        y_slider.value = 0.0
        z_slider.value = 0.0
        lock_centers.value = True
        update_target()

    @save_button.on_click
    def _on_save(_event) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(current_transform, indent=2) + "\n")
        print(f"Saved orientation offset to: {output_path.resolve()}")
        status.content += f"\n- Saved to: `{output_path.resolve()}`"

    update_target()
    print(f"Original dimensions (m): {original_dimensions}")
    print(f"Target dimensions (m):   {target_dimensions}")
    print(f"Open http://localhost:{port} and adjust the orange target mesh.")
    print("Press Ctrl+C in this terminal to stop the server.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Alignment UI stopped.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-mesh",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/largebox/largebox.obj",
    )
    parser.add_argument("--target-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("box_orientation_offset.json"))
    parser.add_argument("--port", type=int, default=8080)
    arguments = parser.parse_args()
    run_alignment_ui(
        arguments.original_mesh,
        arguments.target_mesh,
        arguments.output,
        arguments.port,
    )


if __name__ == "__main__":
    main()
