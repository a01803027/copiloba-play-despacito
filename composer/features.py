"""Pose features in torso lengths with One-Euro adaptive filtering and occlusion hysteresis."""
from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any


def clamp(x, low=0., high=1.):
    return max(low, min(high, x))


class OneEuroFilter3D:
    """Adaptive low-pass filter: suppresses jitter when still, minimal lag on fast motion."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.08, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = (0., 0., 0.)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, value: tuple[float, float, float], dt: float) -> tuple[float, float, float]:
        if self._x is None or dt <= 0 or dt > 0.25:
            self._x = value
            self._dx = (0., 0., 0.)
            return value
        a_d = self._alpha(self.d_cutoff, dt)
        dx = tuple((v - prev) / dt for v, prev in zip(value, self._x))
        self._dx = tuple(prev + a_d * (cur - prev) for prev, cur in zip(self._dx, dx))
        speed = math.sqrt(sum(c * c for c in self._dx))
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self._x = tuple(prev + a * (cur - prev) for prev, cur in zip(self._x, value))
        return self._x


@dataclass(frozen=True)
class Features:
    energy: float = 0.
    fluidity: float = 1.
    expansion: float = .3
    stillness: float = 1.
    velocity: float = 0.
    acceleration: float = 0.
    periodicity: float = 0.
    symmetry: float = 1.
    tracked: bool = True
    left_height: float = 0.5
    right_height: float = 0.5
    body_level: float = 0.5
    lateral_balance: float = 0.0
    left_impulse: float = 0.0
    right_impulse: float = 0.0
    grace: bool = False
    hands: Any = None

    def to_dict(self):
        return asdict(self)


class FeatureExtractor:
    def __init__(self, missing_grace_seconds: float = 0.30):
        self.missing_grace_seconds = missing_grace_seconds
        self._filters = [OneEuroFilter3D() for _ in range(4)]
        self.previous = None
        self.velocity = None
        self.last_time = None
        self.last_valid_time = None
        self.energy = 0.
        self.previous_valid = [False] * 4
        self.last_expansion = .3
        self.last_left_height = .5
        self.last_right_height = .5
        self.last_body_level = .5
        self.last_lateral_balance = 0.
        self.last_features = None
        self.history = deque(maxlen=90)

    def _hard_reset(self):
        for flt in self._filters:
            flt.reset()
        self.previous = self.velocity = self.last_time = self.last_valid_time = None
        self.previous_valid = [False] * 4
        self.history.clear()
        self.energy = 0.
        self.last_features = None
        return Features(tracked=False)

    def missing(self, timestamp: float | None = None):
        """Handle missing pose.

        With a timestamp, holds the last tracked state gracefully for up to
        `missing_grace_seconds` with smooth energy decay to prevent audio cuts
        on 1-3 frame MediaPipe flickers. Without a timestamp (or after grace),
        performs a hard reset.
        """
        if (
            timestamp is None
            or self.last_valid_time is None
            or self.last_features is None
            or (timestamp - self.last_valid_time) > self.missing_grace_seconds
        ):
            return self._hard_reset()

        dt = max(0., timestamp - (self.last_time or self.last_valid_time))
        decay = math.exp(-min(dt, .2) / .25)
        self.energy *= decay
        self.last_time = timestamp
        held = self.last_features
        return Features(
            energy=self.energy,
            fluidity=held.fluidity,
            expansion=self.last_expansion,
            stillness=1. - self.energy,
            velocity=held.velocity * decay,
            acceleration=0.,
            periodicity=held.periodicity * decay,
            symmetry=held.symmetry,
            tracked=True,
            left_height=self.last_left_height,
            right_height=self.last_right_height,
            body_level=self.last_body_level,
            lateral_balance=self.last_lateral_balance,
            left_impulse=0.,
            right_impulse=0.,
            grace=True,
            hands=None,
        )

    def update(self, points, timestamp, aspect=1., hands=None):
        # x and z are width-relative in MediaPipe; make y use the same unit.
        if len(points) != 33 or not all(math.isfinite(c) for p in points for c in p[:3]):
            return self.missing(timestamp)
        xyz = [(p[0], p[1] * aspect, p[2]) for p in points]
        hip = tuple((xyz[23][j] + xyz[24][j]) / 2 for j in range(3))
        shoulder = tuple((xyz[11][j] + xyz[12][j]) / 2 for j in range(3))
        scale = math.dist(hip, shoulder)
        if scale < .025:
            return self.missing(timestamp)

        # Normalized coordinates relative to hip center in torso lengths.
        positions = [tuple((xyz[i][j] - hip[j]) / scale for j in range(3)) for i in (15, 16, 27, 28)]
        dt = timestamp - self.last_time if self.last_time is not None else 1 / 30
        if dt <= 0:
            return self.last_features or Features(energy=self.energy, stillness=1 - self.energy)
        if dt > .25:
            self._hard_reset()
            dt = 1 / 30

        active = [len(points[i]) < 4 or points[i][3] >= .5 for i in (15, 16, 27, 28)]
        smoothed = []
        velocities = []
        for i, new in enumerate(positions):
            if not active[i]:
                self._filters[i].reset()
                filtered = self.previous[i] if self.previous is not None else (0., 0., 0.)
            elif self.previous is not None and self.previous_valid[i]:
                filtered = self._filters[i].filter(new, min(dt, .2))
            else:
                self._filters[i].reset()
                filtered = self._filters[i].filter(new, min(dt, .2))
            smoothed.append(filtered)
            velocities.append(
                tuple((b - a) / dt for a, b in zip(self.previous[i], filtered))
                if self.previous is not None and active[i] and self.previous_valid[i]
                else (0., 0., 0.)
            )

        positions = smoothed
        count = max(1, sum(active))
        speed = sum(math.sqrt(sum(v * v for v in p)) for p, ok in zip(velocities, active) if ok) / count

        per_limb_accel = [0., 0., 0., 0.]
        acceleration = 0.
        if self.velocity is not None:
            for i, (a, b, ok, was) in enumerate(zip(self.velocity, velocities, active, self.previous_valid)):
                if ok and was:
                    per_limb_accel[i] = math.dist(a, b) / dt
            acceleration = sum(per_limb_accel) / count

        # Asymmetric energy envelope: fast attack (90 ms), smooth release (350 ms).
        target_energy = clamp(speed / 3.0)
        tau = 0.09 if target_energy > self.energy else 0.35
        self.energy += (1.0 - math.exp(-min(dt, .2) / tau)) * (target_energy - self.energy)

        if active[0] and active[1]:
            self.last_expansion = clamp((math.dist(positions[0], positions[1]) - .5) / 3.0)
        expansion = self.last_expansion

        # In image coordinates y grows downward, so -positions[i][1] grows upward.
        # Hip is 0.0, shoulder is ~1.0, above head is ~1.8 -> map [-0.2, 1.6] to [0, 1].
        if active[0]:
            self.last_left_height = clamp((-positions[0][1] + 0.2) / 1.8)
        if active[1]:
            self.last_right_height = clamp((-positions[1][1] + 0.2) / 1.8)

        # Body vertical level from raw normalized hip y (0=top, 1=bottom -> invert).
        raw_hip_y = (points[23][1] + points[24][1]) * 0.5
        self.last_body_level = clamp(1.0 - raw_hip_y)

        # Lateral balance (-1 left, +1 right) from active wrists relative to torso center.
        if active[0] and active[1]:
            self.last_lateral_balance = clamp((positions[0][0] + positions[1][0]) * 0.5, -1.0, 1.0)
        elif active[0]:
            self.last_lateral_balance = clamp(positions[0][0] * 0.5, -1.0, 1.0)
        elif active[1]:
            self.last_lateral_balance = clamp(positions[1][0] * 0.5, -1.0, 1.0)

        left_impulse = clamp((per_limb_accel[0] - 18.0) / 40.0) if active[0] else 0.0
        right_impulse = clamp((per_limb_accel[1] - 18.0) / 40.0) if active[1] else 0.0

        pairs = [i for i in (0, 2) if active[i] and active[i + 1]]
        error = sum(
            abs(positions[i][0] + positions[i + 1][0]) + abs(positions[i][1] - positions[i + 1][1])
            for i in pairs
        ) / max(1, 2 * len(pairs))

        self.history.append((timestamp, speed))
        # Resample to 30Hz before comparing lags; variable camera FPS is allowed.
        periodicity = 0.
        if len(self.history) > 20 and timestamp - self.history[0][0] > .8:
            values = []
            history = list(self.history)
            index = 0
            for n in range(30):
                t = timestamp - (29 - n) / 30
                while index + 1 < len(history) and history[index + 1][0] <= t:
                    index += 1
                values.append(history[index][1])
            mean = sum(values) / len(values)
            centered = [v - mean for v in values]
            if sum(v * v for v in centered) > .05:
                for lag in range(8, 21):
                    a, b = centered[lag:], centered[:-lag]
                    norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
                    if norm > 1e-8:
                        periodicity = max(periodicity, sum(x * y for x, y in zip(a, b)) / norm)

        self.previous, self.velocity, self.last_time = positions, velocities, timestamp
        self.last_valid_time = timestamp
        self.previous_valid = active
        result = Features(
            energy=self.energy,
            fluidity=1. / (1. + acceleration / 20.),
            expansion=expansion,
            stillness=1. - self.energy,
            velocity=speed,
            acceleration=acceleration,
            periodicity=clamp(periodicity),
            symmetry=clamp(1. - error),
            tracked=True,
            left_height=self.last_left_height,
            right_height=self.last_right_height,
            body_level=self.last_body_level,
            lateral_balance=self.last_lateral_balance,
            left_impulse=left_impulse,
            right_impulse=right_impulse,
            grace=False,
            hands=hands,
        )
        self.last_features = result
        return result


def mock_features(seconds):
    # Alternating calm/expansive/abrupt phrases, not random isolated triggers.
    activity = .5 + .4 * math.sin(seconds * .25)
    energy = clamp(activity * (.7 + .3 * math.sin(seconds * 3) ** 2))
    left_h = clamp(.5 + .4 * math.sin(seconds * .35))
    right_h = clamp(.5 + .4 * math.cos(seconds * .28))
    return Features(
        energy=energy,
        fluidity=clamp(.7 + .25 * math.sin(seconds * .4)),
        expansion=clamp(.5 + .45 * math.sin(seconds * .18)),
        stillness=1. - energy,
        velocity=energy * 3.,
        acceleration=energy * 15.,
        periodicity=clamp(.5 + .45 * math.sin(seconds * .3)),
        symmetry=clamp(.7 + .25 * math.cos(seconds * .15)),
        tracked=True,
        left_height=left_h,
        right_height=right_h,
        body_level=clamp(.5 + .2 * math.sin(seconds * .2)),
        lateral_balance=clamp(.6 * math.sin(seconds * .22), -1., 1.),
        left_impulse=0.,
        right_impulse=0.,
        grace=False,
    )
