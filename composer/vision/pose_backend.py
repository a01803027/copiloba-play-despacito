"""Pluggable Pose Backend Protocol & STM32MP257F-DK X-LINUX-AI (COCO-17 NPU) Adapter.

Allows swapping MediaPipe Pose (macOS / desktop) with a VeriSilicon NPU MoveNet / YOLOv8-Pose
model on STM32MP257F-DK (`X-LINUX-AI` / `TIM-VX`) without changing a single line of Tier 0
PoseSanitizer, Tier 1 Kinematics, Tier 2 Pure-NumPy TCN/DTW, or the Audio Composer.
"""
from __future__ import annotations

from typing import Protocol, Sequence


# Mapping from COCO-17 keypoint indices (MoveNet / YOLOv8-Pose / X-LINUX-AI) to MediaPipe-33 indices.
COCO17_TO_MP33: tuple[tuple[int, int], ...] = (
    (0, 0),    # nose
    (1, 2),    # left_eye
    (2, 5),    # right_eye
    (3, 7),    # left_ear
    (4, 8),    # right_ear
    (5, 11),   # left_shoulder
    (6, 12),   # right_shoulder
    (7, 13),   # left_elbow
    (8, 14),   # right_elbow
    (9, 15),   # left_wrist
    (10, 16),  # right_wrist
    (11, 23),  # left_hip
    (12, 24),  # right_hip
    (13, 25),  # left_knee
    (14, 26),  # right_knee
    (15, 27),  # left_ankle
    (16, 28),  # right_ankle
)


class PoseBackend(Protocol):
    """Unified interface satisfied by MediaPipe PoseTracker and embedded NPU pose runners."""

    def process(self, image_bgr, timestamp_s: float) -> list[tuple[float, float, float, float]] | None: ...

    def draw(self, image_bgr, points, min_visibility: float = 0.5, glitch_indices: set[int] | None = None): ...

    def close(self) -> None: ...


def coco17_to_mediapipe33(
    keypoints_17: Sequence[Sequence[float]],
    *,
    yx_order: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Convert 17 normalized COCO keypoints into the 33-point `(x, y, z=0.0, conf)` layout.

    Args:
        keypoints_17: 17 entries of `(x, y, score)` (YOLOv8-Pose) or `(y, x, score)` (MoveNet when `yx_order=True`).
        yx_order: Set True for TensorFlow/TFLite MoveNet outputs where channels are `(y, x, score)`.
    """
    if len(keypoints_17) != 17:
        raise ValueError(f'Expected 17 COCO keypoints, got {len(keypoints_17)}')
    out: list[tuple[float, float, float, float]] = [(0.5, 0.5, 0.0, 0.0)] * 33
    for coco_i, mp_i in COCO17_TO_MP33:
        kp = keypoints_17[coco_i]
        if yx_order:
            y, x, score = float(kp[0]), float(kp[1]), float(kp[2]) if len(kp) >= 3 else 1.0
        else:
            x, y, score = float(kp[0]), float(kp[1]), float(kp[2]) if len(kp) >= 3 else 1.0
        out[mp_i] = (x, y, 0.0, score)
    return out
