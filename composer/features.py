"""Full-body pose features: One-Euro adaptive filtering, occlusion hysteresis,
fast directional arm strikes, head/torso/hip dynamics, and multi-source BPM."""
from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any
from .rhythm import MultiSourceTempoTracker


def clamp(x, low=0., high=1.):
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
    # Arms: continuous height, velocity, elbow bend, and rapid directional strikes
    left_height: float = 0.5
    right_height: float = 0.5
    left_vy: float = 0.0
    right_vy: float = 0.0
    left_speed: float = 0.0
    right_speed: float = 0.0
    left_bend: float = 0.5
    right_bend: float = 0.5
    left_impulse: float = 0.0
    right_impulse: float = 0.0
    left_strike: int = 0          # -1 rapid down, 0 none, +1 rapid up
    right_strike: int = 0         # -1 rapid down, 0 none, +1 rapid up
    left_strike_strength: float = 0.0
    right_strike_strength: float = 0.0
    # Head, Shoulders, Torso, and Hip expressive features
    head_pitch: float = 0.5       # 0 looking down .. 1 head lifted
    head_tilt: float = 0.0        # -1 tilted left .. +1 tilted right
    head_nod_vel: float = 0.0
    head_nod: bool = False        # discrete crisp downward head nod event
    shoulder_lift: float = 0.0    # 0 relaxed .. 1 shrugged up (Build-up!)
    shoulder_drop: bool = False   # discrete release from shrug (Drop!)
    shoulder_tilt: float = 0.0    # -1 left shoulder up .. +1 right shoulder up
    shoulder_bounce: float = 0.0  # 0..1 rhythmic shoulder-bobbing activity
    torso_lean: float = 0.0       # -1 leaning left .. +1 leaning right
    torso_twist: float = 0.0      # 0 facing front .. 1 rotated sideways
    hip_sway: float = 0.0         # -1 swaying left .. +1 swaying right
    hip_bounce_vel: float = 0.0
    body_level: float = 0.5       # 0 crouched low .. 1 standing/elevated
    lateral_balance: float = 0.0  # -1 left .. +1 right
    # Multi-source detected body rhythm
    body_bpm: float = 100.0
    body_bpm_confidence: float = 0.0
    body_bpm_source: str = 'none'
    grace: bool = False
    hands: Any = None

    def to_dict(self):
        return asdict(self)


def _elbow_bend(root, joint, tip) -> float:
    a = tuple(r - j for r, j in zip(root, joint))
    b = tuple(t - j for t, j in zip(tip, joint))
    na = math.sqrt(sum(c * c for c in a))
    nb = math.sqrt(sum(c * c for c in b))
    if na * nb < 1e-7:
        return 0.5
    cos_ang = clamp(sum(x * y for x, y in zip(a, b)) / (na * nb), -1.0, 1.0)
    return clamp(1.0 - math.acos(cos_ang) / math.pi)


class FeatureExtractor:
    def __init__(self, missing_grace_seconds: float = 0.30, strike_cooldown: float = 0.18):
        self.missing_grace_seconds = missing_grace_seconds
        self.strike_cooldown = strike_cooldown
        # Agile One-Euro filters for extremities (low lag for rapid arm gestures).
        self._filters = [OneEuroFilter3D(min_cutoff=1.5, beta=0.35, d_cutoff=1.2) for _ in range(4)]
        # Stable One-Euro filters for head, shoulders, hips.
        self._nose_filter = OneEuroFilter3D(min_cutoff=1.3, beta=0.18, d_cutoff=1.0)
        self._hip_filter = OneEuroFilter3D(min_cutoff=1.3, beta=0.18, d_cutoff=1.0)
        self.tempo_tracker = MultiSourceTempoTracker()

        self.previous = None
        self.velocity = None
        self.last_time = None
        self.last_valid_time = None
        self.energy = 0.
        self.previous_valid = [False] * 4
        self.last_strike_time = [-1e9, -1e9]
        self.in_surge = [False, False]
        self.prev_hip_x = None
        self.hip_x_baseline = None
        self.prev_hip_y = None
        self.prev_nose_rel_y = None
        self.neck_baseline = None
        self.shrug_armed = False
        self.last_shoulder_drop = -1e9
        self.last_head_nod = -1e9
        self.in_head_nod = False
        self.prev_shoulder_tilt = 0.0
        self.shoulder_bounce = 0.0
        self.shoulder_lift = 0.0

        self.last_expansion = .3
        self.last_left_height = .5
        self.last_right_height = .5
        self.last_body_level = .5
        self.last_lateral_balance = 0.
        self.last_head_pitch = .5
        self.last_head_tilt = 0.
        self.last_shoulder_tilt = 0.
        self.last_torso_lean = 0.
        self.last_torso_twist = 0.
        self.last_hip_sway = 0.
        self.last_features = None
        self.history = deque(maxlen=90)

    def _hard_reset(self):
        for flt in self._filters:
            flt.reset()
        self._nose_filter.reset()
        self._hip_filter.reset()
        self.previous = self.velocity = self.last_time = self.last_valid_time = None
        self.previous_valid = [False] * 4
        self.in_surge = [False, False]
        self.prev_hip_x = self.hip_x_baseline = self.prev_hip_y = self.prev_nose_rel_y = None
        self.neck_baseline = None
        self.shrug_armed = False
        self.in_head_nod = False
        self.prev_shoulder_tilt = 0.0
        self.shoulder_bounce = 0.0
        self.shoulder_lift = 0.0
        self.history.clear()
        self.energy = 0.
        self.last_features = None
        return Features(tracked=False)

    def missing(self, timestamp: float | None = None):
        """Handle missing pose with 300ms grace period to prevent audio cuts on flickers."""
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
            left_bend=held.left_bend,
            right_bend=held.right_bend,
            head_pitch=self.last_head_pitch,
            head_tilt=self.last_head_tilt,
            head_nod=False,
            shoulder_lift=self.shoulder_lift * decay,
            shoulder_drop=False,
            shoulder_tilt=self.last_shoulder_tilt * decay,
            shoulder_bounce=self.shoulder_bounce * decay,
            torso_lean=self.last_torso_lean,
            torso_twist=self.last_torso_twist,
            hip_sway=self.last_hip_sway * decay,
            body_level=self.last_body_level,
            lateral_balance=self.last_lateral_balance,
            body_bpm=self.tempo_tracker.bpm,
            body_bpm_confidence=self.tempo_tracker.confidence * decay,
            body_bpm_source=self.tempo_tracker.source,
            grace=True,
            hands=None,
        )

    def update(self, points, timestamp, aspect=1., hands=None, bpm_range=(68.0, 155.0)):
        # x and z are width-relative in MediaPipe; make y use the same unit.
        if len(points) != 33 or not all(math.isfinite(c) for p in points for c in p[:3]):
            return self.missing(timestamp)
        xyz = [(p[0], p[1] * aspect, p[2]) for p in points]
        hip = tuple((xyz[23][j] + xyz[24][j]) / 2 for j in range(3))
        shoulder = tuple((xyz[11][j] + xyz[12][j]) / 2 for j in range(3))
        scale = math.dist(hip, shoulder)
        if scale < .025:
            return self.missing(timestamp)

        positions = [tuple((xyz[i][j] - hip[j]) / scale for j in range(3)) for i in (15, 16, 27, 28)]
        dt = timestamp - self.last_time if self.last_time is not None else 1 / 30
        if dt <= 0:
            return self.last_features or Features(energy=self.energy, stillness=1 - self.energy)
        if dt > .25:
            self._hard_reset()
            dt = 1 / 30
        clamped_dt = min(dt, .2)

        vis = [len(points[i]) < 4 or points[i][3] >= .5 for i in range(33)]
        active = [vis[i] for i in (15, 16, 27, 28)]
        smoothed = []
        velocities = []
        for i, new in enumerate(positions):
            if not active[i]:
                self._filters[i].reset()
                filtered = self.previous[i] if self.previous is not None else (0., 0., 0.)
            elif self.previous is not None and self.previous_valid[i]:
                filtered = self._filters[i].filter(new, clamped_dt)
            else:
                self._filters[i].reset()
                filtered = self._filters[i].filter(new, clamped_dt)
            smoothed.append(filtered)
            velocities.append(
                tuple((b - a) / dt for a, b in zip(self.previous[i], filtered))
                if self.previous is not None and active[i] and self.previous_valid[i]
                else (0., 0., 0.)
            )

        positions = smoothed
        limb_speeds = [math.sqrt(sum(v * v for v in vel)) if ok else 0.0 for vel, ok in zip(velocities, active)]
        count = max(1, sum(active))
        speed = sum(limb_speeds) / count

        per_limb_accel = [0., 0., 0., 0.]
        per_limb_decel = [0., 0., 0., 0.]
        acceleration = 0.
        if self.velocity is not None:
            for i, (a, b, ok, was) in enumerate(zip(self.velocity, velocities, active, self.previous_valid)):
                if ok and was:
                    diff = tuple((nb - na) / dt for na, nb in zip(a, b))
                    per_limb_accel[i] = math.sqrt(sum(c * c for c in diff))
                    prev_spd = math.sqrt(sum(c * c for c in a))
                    if prev_spd > 1e-6:
                        # Longitudinal acceleration along previous velocity vector (< 0 = braking).
                        per_limb_decel[i] = sum(d * v for d, v in zip(diff, a)) / prev_spd
            acceleration = sum(per_limb_accel) / count

        # --- Dual rapid-arm-strike detector (works reliably at 20-60 FPS) ---
        strikes = [0, 0]
        strike_strengths = [0.0, 0.0]
        for arm in (0, 1):
            if not (active[arm] and self.previous_valid[arm] and self.velocity is not None):
                self.in_surge[arm] = False
                continue
            cur_spd = limb_speeds[arm]
            prev_spd = math.sqrt(sum(c * c for c in self.velocity[arm]))
            # Image y grows downward -> negate vy so +vy means moving UP, -vy moving DOWN.
            cur_vy = -velocities[arm][1]
            prev_vy = -self.velocity[arm][1]
            accel = per_limb_accel[arm]
            decel = per_limb_decel[arm]

            # 1) Fast acceleration surge OR 2) sharp braking / direction reversal after fast swing.
            surge = cur_spd > 1.05 and accel > 10.0
            brake_hit = prev_spd > 1.15 and (decel < -10.0 or (abs(prev_vy) > 0.9 and cur_vy * prev_vy < 0))
            triggered = (surge and not self.in_surge[arm]) or brake_hit
            self.in_surge[arm] = surge

            if triggered and (timestamp - self.last_strike_time[arm]) >= self.strike_cooldown:
                # Choose vertical direction from whichever phase had stronger vertical motion.
                dom_vy = cur_vy if abs(cur_vy) >= abs(prev_vy) else prev_vy
                direction = 1 if dom_vy >= 0.0 else -1
                strength = clamp((max(cur_spd, prev_spd) - 0.8) / 3.2, 0.2, 1.0)
                strikes[arm] = direction
                strike_strengths[arm] = strength
                self.last_strike_time[arm] = timestamp

        # Asymmetric energy envelope: fast attack (85 ms), smooth release (350 ms).
        target_energy = clamp(speed / 2.6)
        tau = 0.085 if target_energy > self.energy else 0.35
        self.energy += (1.0 - math.exp(-clamped_dt / tau)) * (target_energy - self.energy)

        if active[0] and active[1]:
            self.last_expansion = clamp((math.dist(positions[0], positions[1]) - .5) / 2.8)
        expansion = self.last_expansion

        # Wrist heights: hip=0.0, shoulder=1.0, overhead=1.7 -> map [-0.25, 1.55] to [0, 1].
        if active[0]:
            self.last_left_height = clamp((-positions[0][1] + 0.25) / 1.8)
        if active[1]:
            self.last_right_height = clamp((-positions[1][1] + 0.25) / 1.8)

        left_vy = -velocities[0][1] if active[0] else 0.0
        right_vy = -velocities[1][1] if active[1] else 0.0

        # Elbow bends from shoulder-elbow-wrist angle.
        left_bend = _elbow_bend(xyz[11], xyz[13], xyz[15]) if (vis[11] and vis[13] and vis[15]) else 0.5
        right_bend = _elbow_bend(xyz[12], xyz[14], xyz[16]) if (vis[12] and vis[14] and vis[16]) else 0.5

        # --- Head features (nose 0, ears 7 & 8 relative to shoulder midpoint) ---
        head_nod_vel = 0.0
        head_nod = False
        nose_rel_y = 0.0
        if vis[0]:
            nose_f = self._nose_filter.filter(xyz[0], clamped_dt)
            # Vertical height of nose above shoulder midpoint in torso lengths (~0.25 down, ~0.48 neutral, ~0.68 up).
            nose_rel_y = (shoulder[1] - nose_f[1]) / scale
            self.last_head_pitch = clamp((nose_rel_y - 0.22) / 0.48)
            if self.prev_nose_rel_y is not None:
                head_nod_vel = (nose_rel_y - self.prev_nose_rel_y) / dt
                # Crisp downward head nod detection with rising-edge hysteresis + 200ms cooldown.
                nodding_now = head_nod_vel < -0.55
                if nodding_now and not self.in_head_nod and (timestamp - self.last_head_nod) >= 0.20:
                    head_nod = True
                    self.last_head_nod = timestamp
                self.in_head_nod = nodding_now
            self.prev_nose_rel_y = nose_rel_y

            raw_tilt = 0.0
            if vis[7] and vis[8]:
                ear_dx = xyz[8][0] - xyz[7][0]
                ear_dy = xyz[8][1] - xyz[7][1]
                if abs(ear_dx) > 1e-4:
                    # Sensitive tilt: ~17 degrees (0.30 rad) reaches full +/-1.0.
                    raw_tilt = clamp(math.atan2(ear_dy, abs(ear_dx)) / 0.30, -1.0, 1.0)
            else:
                nose_dx = (nose_f[0] - shoulder[0]) / scale
                raw_tilt = clamp(nose_dx / 0.26, -1.0, 1.0)
            # Subtle deadzone (+/-0.10) avoids drift when holding head straight.
            self.last_head_tilt = 0.0 if abs(raw_tilt) < 0.10 else clamp(
                math.copysign((abs(raw_tilt) - 0.10) / 0.90, raw_tilt), -1.0, 1.0
            )

        # --- Shoulder features (shrug lift -> build-up/drop, alternating tilt/bounce -> groove) ---
        shoulder_drop = False
        if vis[0]:
            if self.neck_baseline is None:
                self.neck_baseline = nose_rel_y
            elif self.shoulder_lift < 0.25:
                # Slowly adapt relaxed neck baseline only when not shrugging (tau ~ 4.0s).
                self.neck_baseline += (1.0 - math.exp(-clamped_dt / 4.0)) * (nose_rel_y - self.neck_baseline)
            # Shrugging lifts shoulders toward nose -> nose_rel_y decreases relative to baseline.
            shrug_delta = self.neck_baseline - nose_rel_y
            raw_lift = clamp((shrug_delta - 0.025) / 0.12)
            lift_tau = 0.07 if raw_lift > self.shoulder_lift else 0.12
            self.shoulder_lift += (1.0 - math.exp(-clamped_dt / lift_tau)) * (raw_lift - self.shoulder_lift)
            if self.shoulder_lift >= 0.55:
                self.shrug_armed = True
            elif self.shrug_armed and self.shoulder_lift <= 0.26 and (timestamp - self.last_shoulder_drop) >= 0.45:
                shoulder_drop = True
                self.shrug_armed = False
                self.last_shoulder_drop = timestamp

        # Alternating shoulder tilt: left shoulder up vs right shoulder up (~12.5 deg = full +/-1).
        sh_vec_dx = max(1e-4, abs(xyz[12][0] - xyz[11][0]))
        sh_vec_dy = xyz[12][1] - xyz[11][1]  # positive when right shoulder is lower (left shoulder lifted)
        raw_sh_tilt = clamp(math.atan2(sh_vec_dy, sh_vec_dx) / 0.22, -1.0, 1.0)
        self.last_shoulder_tilt = 0.0 if abs(raw_sh_tilt) < 0.06 else raw_sh_tilt
        sh_tilt_vel = abs(self.last_shoulder_tilt - self.prev_shoulder_tilt) / dt
        self.prev_shoulder_tilt = self.last_shoulder_tilt
        target_bounce = clamp(sh_tilt_vel / 3.5)
        b_tau = 0.09 if target_bounce > self.shoulder_bounce else 0.42
        self.shoulder_bounce += (1.0 - math.exp(-clamped_dt / b_tau)) * (target_bounce - self.shoulder_bounce)

        # --- Torso features (lateral lean & rotational twist) ---
        # Torso vector from hip to shoulder (in image coords, up is -y).
        torso_dx = shoulder[0] - hip[0]
        torso_dy = max(1e-4, hip[1] - shoulder[1])
        self.last_torso_lean = clamp(math.atan2(torso_dx, torso_dy) / 0.45, -1.0, 1.0)

        # Twist from shoulder Z depth difference and horizontal foreshortening.
        sh_dx = abs(xyz[12][0] - xyz[11][0]) / scale
        sh_dz = abs(xyz[12][2] - xyz[11][2]) / scale
        # Facing front: sh_dx ~ 0.65-0.95; twisted sideways: sh_dx drops toward 0.25 and sh_dz rises.
        twist_from_width = clamp((0.68 - sh_dx) / 0.45)
        twist_from_z = clamp((sh_dz - 0.15) / 0.65)
        self.last_torso_twist = clamp(max(twist_from_width, twist_from_z))

        # --- Hip features (bounce, lateral sway, vertical level) ---
        hip_f = self._hip_filter.filter(hip, clamped_dt)
        raw_hip_y = (points[23][1] + points[24][1]) * 0.5
        self.last_body_level = clamp(1.0 - raw_hip_y)

        if self.hip_x_baseline is None:
            self.hip_x_baseline = hip_f[0]
        else:
            # Slow baseline (tau ~ 1.5s) isolates rhythmic lateral hip sway from standing position.
            self.hip_x_baseline += (1.0 - math.exp(-clamped_dt / 1.5)) * (hip_f[0] - self.hip_x_baseline)
        self.last_hip_sway = clamp(((hip_f[0] - self.hip_x_baseline) / scale) / 0.35, -1.0, 1.0)

        hip_bounce_vel = 0.0
        if self.prev_hip_y is not None:
            hip_bounce_vel = (self.prev_hip_y - hip_f[1]) / (scale * dt)
        self.prev_hip_y = hip_f[1]

        # Lateral balance (-1 left, +1 right) from active wrists relative to torso center.
        if active[0] and active[1]:
            self.last_lateral_balance = clamp((positions[0][0] + positions[1][0]) * 0.5, -1.0, 1.0)
        elif active[0]:
            self.last_lateral_balance = clamp(positions[0][0] * 0.5, -1.0, 1.0)
        elif active[1]:
            self.last_lateral_balance = clamp(positions[1][0] * 0.5, -1.0, 1.0)

        left_impulse = clamp((per_limb_accel[0] - 10.0) / 28.0) if active[0] else 0.0
        right_impulse = clamp((per_limb_accel[1] - 10.0) / 28.0) if active[1] else 0.0

        # Multi-source BPM tracker (hip bounce + head nod + shoulder bounce + rhythmic arm speed).
        arm_avg_speed = 0.5 * (limb_speeds[0] + limb_speeds[1])
        body_bpm, body_bpm_conf = self.tempo_tracker.update(
            timestamp,
            hip_y=hip_f[1] / scale,
            head_y=nose_rel_y,
            arm_speed=arm_avg_speed,
            shoulder_tilt=self.last_shoulder_tilt,
            bpm_min=bpm_range[0],
            bpm_max=bpm_range[1],
        )

        pairs = [i for i in (0, 2) if active[i] and active[i + 1]]
        error = sum(
            abs(positions[i][0] + positions[i + 1][0]) + abs(positions[i][1] - positions[i + 1][1])
            for i in pairs
        ) / max(1, 2 * len(pairs))

        self.history.append((timestamp, speed))
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
            periodicity=clamp(max(periodicity, body_bpm_conf)),
            symmetry=clamp(1. - error),
            tracked=True,
            left_height=self.last_left_height,
            right_height=self.last_right_height,
            left_vy=left_vy,
            right_vy=right_vy,
            left_speed=limb_speeds[0],
            right_speed=limb_speeds[1],
            left_bend=left_bend,
            right_bend=right_bend,
            left_impulse=left_impulse,
            right_impulse=right_impulse,
            left_strike=strikes[0],
            right_strike=strikes[1],
            left_strike_strength=strike_strengths[0],
            right_strike_strength=strike_strengths[1],
            head_pitch=self.last_head_pitch,
            head_tilt=self.last_head_tilt,
            head_nod_vel=head_nod_vel,
            head_nod=head_nod,
            shoulder_lift=self.shoulder_lift,
            shoulder_drop=shoulder_drop,
            shoulder_tilt=self.last_shoulder_tilt,
            shoulder_bounce=self.shoulder_bounce,
            torso_lean=self.last_torso_lean,
            torso_twist=self.last_torso_twist,
            hip_sway=self.last_hip_sway,
            hip_bounce_vel=hip_bounce_vel,
            body_level=self.last_body_level,
            lateral_balance=self.last_lateral_balance,
            body_bpm=body_bpm,
            body_bpm_confidence=body_bpm_conf,
            body_bpm_source=self.tempo_tracker.source,
            grace=False,
            hands=hands,
        )
        self.last_features = result
        return result


def mock_features(seconds):
    # Expressive synthetic dance: alternating arm contours, torso lean, hip sway, and periodic pulse.
    activity = .5 + .4 * math.sin(seconds * .25)
    energy = clamp(activity * (.7 + .3 * math.sin(seconds * 3) ** 2))
    left_h = clamp(.5 + .4 * math.sin(seconds * .35))
    right_h = clamp(.5 + .42 * math.sin(seconds * .28))
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
        left_vy=math.cos(seconds * .35) * 0.8,
        right_vy=math.cos(seconds * .28) * 0.8,
        head_pitch=clamp(.5 + .35 * math.sin(seconds * .22)),
        head_tilt=clamp(.5 * math.sin(seconds * .30), -1., 1.),
        torso_lean=clamp(.6 * math.sin(seconds * .20), -1., 1.),
        torso_twist=clamp(.4 + .35 * math.sin(seconds * .17)),
        hip_sway=clamp(.6 * math.sin(seconds * 1.2), -1., 1.),
        body_level=clamp(.5 + .25 * math.sin(seconds * .19)),
        lateral_balance=clamp(.6 * math.sin(seconds * .22), -1., 1.),
        body_bpm=108.0,
        body_bpm_confidence=0.65,
        body_bpm_source='hip',
        grace=False,
    )
