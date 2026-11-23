"""Adaptive One-Euro 3D filter and kinematic math helpers."""
from __future__ import annotations

import math


def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


class OneEuroFilter3D:
    """Adaptive low-pass filter: suppresses jitter when still, minimal lag on fast motion."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.35, d_cutoff: float = 1.2):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = (0.0, 0.0, 0.0)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, value: tuple[float, float, float], dt: float) -> tuple[float, float, float]:
        if self._x is None or dt <= 0 or dt > 0.25:
            self._x = value
            self._dx = (0.0, 0.0, 0.0)
            return value
        a_d = self._alpha(self.d_cutoff, dt)
        dx = tuple((v - prev) / dt for v, prev in zip(value, self._x))
        self._dx = tuple(prev + a_d * (cur - prev) for prev, cur in zip(self._dx, dx))
        speed = math.sqrt(sum(c * c for c in self._dx))
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self._x = tuple(prev + a * (cur - prev) for prev, cur in zip(self._x, value))
        return self._x


def elbow_bend(root, joint, tip) -> float:
    a = tuple(r - j for r, j in zip(root, joint))
    b = tuple(t - j for t, j in zip(tip, joint))
    na = math.sqrt(sum(c * c for c in a))
    nb = math.sqrt(sum(c * c for c in b))
    if na * nb < 1e-7:
        return 0.5
    cos_ang = clamp(sum(x * y for x, y in zip(a, b)) / (na * nb), -1.0, 1.0)
    return clamp(1.0 - math.acos(cos_ang) / math.pi)
