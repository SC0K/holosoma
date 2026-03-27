#!/usr/bin/env python3
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


@dataclass(frozen=True)
class ViserCompareConfig:
    original_qpos_npz: str
    """Path to original .npz with qpos."""

    augmented_qpos_npz: str
    """Path to augmented .npz with qpos."""

    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Path to robot URDF."""

    object_urdf: str | None = "models/largebox/largebox.urdf"
    """Path to object URDF (optional)."""

    show_object_axes: bool = True
    """Whether to show object XYZ axes for both scenes."""

    assume_object_in_qpos: bool = True
    """Whether qpos contains object pose in the last 7 values."""

    side_offset_y: float = 1.0
    """Half-distance in Y between original (-Y) and augmented (+Y)."""

    fps: int = 30
    """Fallback FPS if npz does not include fps."""

    loop: bool = False
    """Whether to loop playback."""

    show_meshes: bool = True
    """Whether to show visual meshes."""

    grid_width: float = 10.0
    """Grid width."""

    grid_height: float = 10.0
    """Grid height."""

    visual_fps_multiplier: int = 2
    """Interpolation multiplier for smoother playback."""


def _load_qpos(path: str) -> tuple[np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float32)
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def _resolve_input_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / p,
        script_dir / p,
        repo_root / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(p)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(curr_q)
    if prev_q is None:
        return q
    return -q if float(np.dot(prev_q, q)) < 0.0 else q


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0 + u * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / s


def _interp_q(q0: np.ndarray, q1: np.ndarray, u: float, robot_dof: int, has_object: bool) -> np.ndarray:
    out = q0.copy()
    out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]
    out[3:7] = _slerp(q0[3:7], q1[3:7], u)
    out[7 : 7 + robot_dof] = (1.0 - u) * q0[7 : 7 + robot_dof] + u * q1[7 : 7 + robot_dof]
    if has_object:
        out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]
        out[-4:] = _slerp(q0[-4:], q1[-4:], u)
    return out


def make_compare_player(cfg: ViserCompareConfig) -> viser.ViserServer:
    q_orig, fps_orig = _load_qpos(cfg.original_qpos_npz)
    q_aug, fps_aug = _load_qpos(cfg.augmented_qpos_npz)

    n_frames = int(min(len(q_orig), len(q_aug)))
    if n_frames <= 0:
        raise ValueError("Both sequences must contain at least 1 frame")

    q_orig = q_orig[:n_frames]
    q_aug = q_aug[:n_frames]

    server = viser.ViserServer()

    # Anchors for side-by-side display.
    original_anchor = server.scene.add_frame("/original", position=(0.0, -cfg.side_offset_y, 0.0), show_axes=False)
    augmented_anchor = server.scene.add_frame("/augmented", position=(0.0, cfg.side_offset_y, 0.0), show_axes=False)

    original_robot_base = server.scene.add_frame("/original/robot_base", show_axes=False)
    augmented_robot_base = server.scene.add_frame("/augmented/robot_base", show_axes=False)
    original_object_base = server.scene.add_frame("/original/object_base", show_axes=cfg.show_object_axes)
    augmented_object_base = server.scene.add_frame("/augmented/object_base", show_axes=cfg.show_object_axes)

    robot_urdf_y = yourdfpy.URDF.load(_resolve_input_path(cfg.robot_urdf), load_meshes=True, build_scene_graph=True)
    original_robot = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/original/robot_base")
    augmented_robot = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/augmented/robot_base")

    original_object = None
    augmented_object = None
    if cfg.object_urdf:
        object_urdf_y = yourdfpy.URDF.load(
            _resolve_input_path(cfg.object_urdf), load_meshes=True, build_scene_graph=True
        )
        original_object = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/original/object_base")
        augmented_object = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/augmented/object_base")

    server.scene.add_grid("/grid", width=cfg.grid_width, height=cfg.grid_height, position=(0.0, 0.0, 0.0))
    server.scene.add_label("/label/original", text="original", position=(0.0, -cfg.side_offset_y, 1.3))
    server.scene.add_label("/label/augmented", text="augmented", position=(0.0, cfg.side_offset_y, 1.3))

    original_anchor.position = np.array([0.0, -cfg.side_offset_y, 0.0])
    augmented_anchor.position = np.array([0.0, cfg.side_offset_y, 0.0])

    robot_dof = len(original_robot.get_actuated_joint_limits())
    has_object = (
        cfg.assume_object_in_qpos
        and original_object is not None
        and augmented_object is not None
        and q_orig.shape[1] >= (7 + robot_dof + 7)
        and q_aug.shape[1] >= (7 + robot_dof + 7)
    )

    original_robot.show_visual = cfg.show_meshes
    augmented_robot.show_visual = cfg.show_meshes
    if original_object is not None:
        original_object.show_visual = cfg.show_meshes
    if augmented_object is not None:
        augmented_object.show_visual = cfg.show_meshes

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=int(min(fps_orig, fps_aug) or cfg.fps), min=1, max=240, step=1)
        interp_in = server.gui.add_number(
            "Visual FPS multiplier", initial_value=int(cfg.visual_fps_multiplier), min=1, max=8, step=1
        )

    with server.gui.add_folder("Display"):
        show_meshes = server.gui.add_checkbox("Show meshes", initial_value=cfg.show_meshes)
        side_offset = server.gui.add_number("Half Y offset", initial_value=float(cfg.side_offset_y), min=0.0, max=10.0, step=0.1)

    playing = {"flag": False}
    ticking = {"next": time.perf_counter()}
    fractional = {"f": float(frame_slider.value)}
    programmatic = {"flag": False}
    prev = {
        "orig_robot_q": None,
        "orig_obj_q": None,
        "aug_robot_q": None,
        "aug_obj_q": None,
    }

    def _apply_single(
        q: np.ndarray,
        robot: ViserUrdf,
        robot_base: viser.FrameHandle,
        object_urdf: ViserUrdf | None,
        object_base: viser.FrameHandle,
        robot_prev_key: str,
        obj_prev_key: str,
    ) -> None:
        joints = q[7 : 7 + robot_dof]
        if joints.shape[0] != robot_dof:
            joints = joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
        robot.update_cfg(joints)

        robot_base.position = q[0:3]
        rq = _quat_continuous(prev[robot_prev_key], q[3:7])
        prev[robot_prev_key] = rq
        robot_base.wxyz = rq

        if has_object and object_urdf is not None:
            object_base.position = q[-7:-4]
            oq = _quat_continuous(prev[obj_prev_key], q[-4:])
            prev[obj_prev_key] = oq
            object_base.wxyz = oq

    def _apply_frame(frame_float: float) -> None:
        k0 = int(np.floor(frame_float))
        k1 = (k0 + 1) % n_frames if cfg.loop else min(k0 + 1, n_frames - 1)
        u = float(frame_float - k0)

        q0 = _interp_q(q_orig[k0], q_orig[k1], u, robot_dof, has_object)
        q1 = _interp_q(q_aug[k0], q_aug[k1], u, robot_dof, has_object)

        _apply_single(
            q=q0,
            robot=original_robot,
            robot_base=original_robot_base,
            object_urdf=original_object,
            object_base=original_object_base,
            robot_prev_key="orig_robot_q",
            obj_prev_key="orig_obj_q",
        )
        _apply_single(
            q=q1,
            robot=augmented_robot,
            robot_base=augmented_robot_base,
            object_urdf=augmented_object,
            object_base=augmented_object_base,
            robot_prev_key="aug_robot_q",
            obj_prev_key="aug_obj_q",
        )

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        ticking["next"] = time.perf_counter()
        fractional["f"] = float(frame_slider.value)
        prev["orig_robot_q"] = None
        prev["orig_obj_q"] = None
        prev["aug_robot_q"] = None
        prev["aug_obj_q"] = None

    @show_meshes.on_update
    def _(_evt) -> None:
        original_robot.show_visual = bool(show_meshes.value)
        augmented_robot.show_visual = bool(show_meshes.value)
        if original_object is not None:
            original_object.show_visual = bool(show_meshes.value)
        if augmented_object is not None:
            augmented_object.show_visual = bool(show_meshes.value)

    @side_offset.on_update
    def _(_evt) -> None:
        original_anchor.position = np.array([0.0, -float(side_offset.value), 0.0])
        augmented_anchor.position = np.array([0.0, float(side_offset.value), 0.0])

    @frame_slider.on_update
    def _(_evt) -> None:
        if not programmatic["flag"]:
            playing["flag"] = False
            fractional["f"] = float(frame_slider.value)
            _apply_frame(fractional["f"])

    def _loop() -> None:
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_in.value))
                dt = 1.0 / (fps_val * mult)

                if now >= ticking["next"]:
                    f = fractional["f"] + 1.0 / mult
                    if cfg.loop:
                        f = f % n_frames
                    else:
                        f = min(f, float(n_frames - 1))
                    fractional["f"] = f
                    _apply_frame(f)

                    programmatic["flag"] = True
                    frame_slider.value = int(np.floor(f))
                    programmatic["flag"] = False
                    ticking["next"] = now + dt
                else:
                    time.sleep(min(0.002, max(0.0, ticking["next"] - now)))
            else:
                time.sleep(0.02)

    _apply_frame(0.0)
    threading.Thread(target=_loop, daemon=True).start()

    print(
        f"[viser_compare_player] Loaded {n_frames} synced frames | "
        f"robot_dof={robot_dof} | object={'yes' if has_object else 'no'}"
    )
    print("Open the viewer URL printed above. Close with Ctrl+C.")
    return server


def main(cfg: ViserCompareConfig) -> None:
    make_compare_player(cfg)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(ViserCompareConfig))
