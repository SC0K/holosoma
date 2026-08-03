"""Conversions between supported floating-base robot/object qpos layouts."""

from __future__ import annotations

from typing import Literal

import numpy as np


QposLayout = Literal["native", "omniretarget"]


def detect_qpos_layout(qpos: np.ndarray) -> QposLayout:
    """Infer layout by locating the unit quaternions at the robot and object roots."""
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] < 14 or len(qpos) == 0:
        raise ValueError(f"Expected a non-empty 2D robot/object qpos array, got {qpos.shape}")

    def quaternion_error(values: np.ndarray) -> float:
        return float(np.median(np.abs(np.linalg.norm(values, axis=1) - 1.0)))

    native_error = quaternion_error(qpos[:, 3:7]) + quaternion_error(qpos[:, -4:])
    omniretarget_error = quaternion_error(qpos[:, :4]) + quaternion_error(qpos[:, -7:-3])
    return "native" if native_error <= omniretarget_error else "omniretarget"


def convert_qpos_layout(
    qpos: np.ndarray,
    source_layout: QposLayout,
    target_layout: QposLayout = "native",
) -> np.ndarray:
    """Convert qpos without changing joint values or coordinate conventions."""
    values = np.asarray(qpos)
    if values.ndim not in (1, 2) or values.shape[-1] < 14:
        raise ValueError(f"Expected robot/object qpos with width >= 14, got {values.shape}")
    if source_layout == target_layout:
        return values.copy()

    converted = values.copy()
    if source_layout == "omniretarget" and target_layout == "native":
        converted[..., :3] = values[..., 4:7]
        converted[..., 3:7] = values[..., :4]
        converted[..., -7:-4] = values[..., -3:]
        converted[..., -4:] = values[..., -7:-3]
    elif source_layout == "native" and target_layout == "omniretarget":
        converted[..., :4] = values[..., 3:7]
        converted[..., 4:7] = values[..., :3]
        converted[..., -7:-3] = values[..., -4:]
        converted[..., -3:] = values[..., -7:-4]
    else:  # pragma: no cover - guarded by the QposLayout type in normal use
        raise ValueError(f"Unsupported qpos layout conversion: {source_layout} -> {target_layout}")
    return converted
