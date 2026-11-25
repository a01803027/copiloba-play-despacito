"""Canonical torso-normalized 2D pose representation shared by Tier 1 kinematics & Tier 2 ML/DTW."""
from __future__ import annotations

from dataclasses import dataclass

# Ordered 13 anatomical landmarks tracked in canonical 2D space.
CANONICAL_JOINTS: tuple[tuple[int, str], ...] = (
    (0, 'nose'),
    (11, 'l_shoulder'),
    (12, 'r_shoulder'),
    (13, 'l_elbow'),
    (14, 'r_elbow'),
    (15, 'l_wrist'),
    (16, 'r_wrist'),
    (23, 'l_hip'),
    (24, 'r_hip'),
    (25, 'l_knee'),
    (26, 'r_knee'),
    (27, 'l_ankle'),
    (28, 'r_ankle'),
)

FEATURE_DIM = 24


@dataclass(frozen=True)
class PoseFrame:
    """Clean, glitch-guarded 2D pose snapshot in hip-centered, torso-scaled coordinates (up = +y)."""

    timestamp: float
    dt: float
    tracked: bool = False
    scale: float = 0.25
    hip_xy: tuple[float, float] = (0.5, 0.65)
    shoulder_xy: tuple[float, float] = (0.5, 0.35)
    joints_2d: dict[str, tuple[float, float]] | None = None
    vel_2d: dict[str, tuple[float, float]] | None = None
    speed_2d: dict[str, float] | None = None
    vis: dict[str, float] | None = None
    elbow_bend_2d: tuple[float, float] = (0.5, 0.5)
    arm_reach: tuple[float, float] = (0.5, 0.5)
    arm_reach_vel: tuple[float, float] = (0.0, 0.0)
    wrist_cross_metric: float = 0.5
    glitch_joints: tuple[str, ...] = ()
    feature_vector: tuple[float, ...] = (0.0,) * FEATURE_DIM

    @property
    def wiggle_suppressed(self) -> bool:
        return bool(self.glitch_joints)
