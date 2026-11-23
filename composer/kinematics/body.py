"""Pose-only full-body studio macro extractor + 11-move biomechanical recognizer (zero fists)."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from .filters import OneEuroFilter3D, clamp, elbow_bend
from ..moves import BodyMoveRecognizer, MoveEvent
from ..rhythm import MultiSourceTempoTracker


@dataclass(frozen=True)
class Features:
    energy: float = 0.0
    fluidity: float = 1.0
    expansion: float = 0.3
    stillness: float = 1.0
    velocity: float = 0.0
    acceleration: float = 0.0
    periodicity: float = 0.0
    symmetry: float = 1.0
    tracked: bool = True
    # Arms: smooth macro heights, velocity, elbow bend
    left_height: float = 0.5
    right_height: float = 0.5
    left_vy: float = 0.0
    right_vy: float = 0.0
    left_speed: float = 0.0
    right_speed: float = 0.0
    left_bend: float = 0.5
    right_bend: float = 0.5
    # Head, Shoulders, Torso, and Hip smooth studio macro controls
    head_pitch: float = 0.5
    head_tilt: float = 0.0
    head_nod_vel: float = 0.0
    shoulder_lift: float = 0.0
    shoulder_drop: bool = False
    shoulder_tilt: float = 0.0
    shoulder_bounce: float = 0.0
    torso_lean: float = 0.0
    torso_twist: float = 0.0
    hip_sway: float = 0.0
    hip_bounce_vel: float = 0.0
    body_level: float = 0.65
    lateral_balance: float = 0.0
    # Legs & Stance continuous macros (knees 25/26 or ankles 27/28)
    legs_visible: bool = False
    legs_source: str = 'none'
    left_leg_lift: float = 0.0
    right_leg_lift: float = 0.0
    left_leg_speed: float = 0.0
    right_leg_speed: float = 0.0
    leg_energy: float = 0.0
    stance_width: float = 0.35
    # Full-Body Move & Dance Recognition outputs (11 core moves!)
    moves: tuple[MoveEvent, ...] = ()
    soundwave_rise: float = 0.0     # 0..1 continuous double-arm overhead soundwave swell
    arms_crossed: bool = False      # True while holding X-cross DJ Breakdown
    body_frozen: bool = False       # True during dramatic Freeze echo tail
    side_step_groove: bool = False  # True during rhythmic side-step dance groove
    squat_depth: float = 0.0        # 0..1 continuous deep squat underwater filter dive
    active_move_label: str = ''
    # Multi-source detected body rhythm
    body_bpm: float = 100.0
    body_bpm_confidence: float = 0.0
    body_bpm_source: str = 'none'
    grace: bool = False

    def to_dict(self):
        return asdict(self)


class FeatureExtractor:
    """Extracts smooth full-body macros + 11 biomechanical dance moves from 33-point MediaPipe Pose."""

    def __init__(self, missing_grace_seconds: float = 0.30):
        self.missing_grace_seconds = missing_grace_seconds
        self.move_recognizer = BodyMoveRecognizer()
        self._filters = [
            OneEuroFilter3D(min_cutoff=1.5, beta=0.35, d_cutoff=1.2),
            OneEuroFilter3D(min_cutoff=1.5, beta=0.35, d_cutoff=1.2),
            OneEuroFilter3D(min_cutoff=1.1, beta=0.18, d_cutoff=1.0),
            OneEuroFilter3D(min_cutoff=1.1, beta=0.18, d_cutoff=1.0),
        ]
        self._nose_filter = OneEuroFilter3D(min_cutoff=1.3, beta=0.18, d_cutoff=1.0)
        self._hip_filter = OneEuroFilter3D(min_cutoff=1.3, beta=0.18, d_cutoff=1.0)
        self.tempo_tracker = MultiSourceTempoTracker()
        self._init_state()

    def _init_state(self):
        self.previous = None
        self.velocity = None
        self.last_time = None
        self.last_valid_time = None
        self.energy = 0.0
        self.leg_energy = 0.0
        self.previous_valid = [False] * 4
        self.prev_leg_indices = [25, 26]
        self.leg_y_baseline = [None, None]
        self.last_left_leg_lift = 0.0
        self.last_right_leg_lift = 0.0
        self.last_stance_width = 0.35
        self.hip_x_baseline = None
        self.hip_y_baseline = None
        self.prev_hip_y = None
        self.prev_nose_rel_y = None
        self.neck_baseline = None
        self.ear_neck_baseline = None
        self.shrug_armed = False
        self.last_shoulder_drop = -1e9
        self.prev_shoulder_tilt = 0.0
        self.shoulder_bounce = 0.0
        self.shoulder_lift = 0.0
        self.last_expansion = 0.3
        self.last_left_height = 0.5
        self.last_right_height = 0.5
        self.last_body_level = 0.65
        self.last_lateral_balance = 0.0
        self.last_head_pitch = 0.5
        self.last_head_tilt = 0.0
        self.last_shoulder_tilt = 0.0
        self.last_torso_lean = 0.0
        self.last_torso_twist = 0.0
        self.last_hip_sway = 0.0
        self.last_features = None
        self.history = deque(maxlen=90)

    def _hard_reset(self):
        for flt in self._filters:
            flt.reset()
        self._nose_filter.reset()
        self._hip_filter.reset()
        self.move_recognizer.reset()
        self._init_state()
        return Features(tracked=False)

    def missing(self, timestamp: float | None = None):
        """300 ms grace period prevents audio cuts during brief webcam flickers."""
        if (
            timestamp is None
            or self.last_valid_time is None
            or self.last_features is None
            or (timestamp - self.last_valid_time) > self.missing_grace_seconds
        ):
            return self._hard_reset()

        dt = max(0.0, timestamp - (self.last_time or self.last_valid_time))
        decay = math.exp(-min(dt, 0.2) / 0.25)
        self.energy *= decay
        self.last_time = timestamp
        held = self.last_features
        return Features(
            energy=self.energy,
            fluidity=held.fluidity,
            expansion=self.last_expansion,
            stillness=1.0 - self.energy,
            velocity=held.velocity * decay,
            acceleration=0.0,
            periodicity=held.periodicity * decay,
            symmetry=held.symmetry,
            tracked=True,
            left_height=self.last_left_height,
            right_height=self.last_right_height,
            left_bend=held.left_bend,
            right_bend=held.right_bend,
            head_pitch=self.last_head_pitch,
            head_tilt=self.last_head_tilt,
            shoulder_lift=self.shoulder_lift * decay,
            shoulder_tilt=self.last_shoulder_tilt * decay,
            shoulder_bounce=self.shoulder_bounce * decay,
            torso_lean=self.last_torso_lean,
            torso_twist=self.last_torso_twist,
            hip_sway=self.last_hip_sway * decay,
            body_level=self.last_body_level,
            lateral_balance=self.last_lateral_balance,
            legs_visible=held.legs_visible,
            legs_source=held.legs_source,
            left_leg_lift=self.last_left_leg_lift * decay,
            right_leg_lift=self.last_right_leg_lift * decay,
            leg_energy=self.leg_energy * decay,
            stance_width=self.last_stance_width,
            moves=(),
            soundwave_rise=held.soundwave_rise * decay,
            arms_crossed=held.arms_crossed,
            body_frozen=held.body_frozen,
            side_step_groove=held.side_step_groove,
            squat_depth=held.squat_depth * decay,
            active_move_label=held.active_move_label,
            body_bpm=self.tempo_tracker.bpm,
            body_bpm_confidence=self.tempo_tracker.confidence * decay,
            body_bpm_source=self.tempo_tracker.source,
            grace=True,
        )

    def _select_legs(self, points):
        vis_ankle = [len(points[i]) < 4 or points[i][3] >= 0.65 for i in range(33)]
        vis_knee = [len(points[i]) < 4 or points[i][3] >= 0.50 for i in range(33)]
        if vis_ankle[27] and vis_ankle[28]:
            return [27, 28], 'ankles', (True, True)
        if vis_knee[25] or vis_knee[26]:
            return [25, 26], 'knees', (vis_knee[25], vis_knee[26])
        return [25, 26], 'none', (False, False)

    def update(
        self,
        points,
        timestamp: float,
        aspect: float = 1.0,
        bpm_range: tuple[float, float] = (68.0, 155.0),
    ) -> Features:
        if len(points) != 33 or not all(math.isfinite(c) for p in points for c in p[:3]):
            return self.missing(timestamp)
        xyz = [(p[0], p[1] * aspect, p[2]) for p in points]
        hip = tuple((xyz[23][j] + xyz[24][j]) / 2.0 for j in range(3))
        shoulder = tuple((xyz[11][j] + xyz[12][j]) / 2.0 for j in range(3))
        scale = math.dist(hip, shoulder)
        if scale < 0.025:
            return self.missing(timestamp)

        dt = timestamp - self.last_time if self.last_time is not None else 1.0 / 30.0
        if dt <= 0:
            return self.last_features or Features(energy=self.energy, stillness=1.0 - self.energy)
        if dt > 0.25:
            self._hard_reset()
            dt = 1.0 / 30.0
        clamped_dt = min(dt, 0.2)

        vis = [len(points[i]) < 4 or points[i][3] >= 0.5 for i in range(33)]
        leg_indices, legs_source, (leg_ok_l, leg_ok_r) = self._select_legs(points)
        for side in (0, 1):
            if leg_indices[side] != self.prev_leg_indices[side]:
                self._filters[2 + side].reset()
                self.previous_valid[2 + side] = False
                self.leg_y_baseline[side] = None
                self.prev_leg_indices[side] = leg_indices[side]

        limb_indices = (15, 16, leg_indices[0], leg_indices[1])
        raw_positions = [tuple((xyz[i][j] - hip[j]) / scale for j in range(3)) for i in limb_indices]
        active = [vis[15], vis[16], leg_ok_l, leg_ok_r]
        legs_visible = active[2] or active[3]

        positions = []
        velocities = []
        for i, new in enumerate(raw_positions):
            if not active[i]:
                self._filters[i].reset()
                filtered = self.previous[i] if self.previous is not None else (0.0, 0.0, 0.0)
            elif self.previous is not None and self.previous_valid[i]:
                filtered = self._filters[i].filter(new, clamped_dt)
            else:
                self._filters[i].reset()
                filtered = self._filters[i].filter(new, clamped_dt)
            positions.append(filtered)
            velocities.append(
                tuple((b - a) / dt for a, b in zip(self.previous[i], filtered))
                if self.previous is not None and active[i] and self.previous_valid[i]
                else (0.0, 0.0, 0.0)
            )

        limb_speeds = [math.sqrt(sum(v * v for v in vel)) if ok else 0.0 for vel, ok in zip(velocities, active)]
        count = max(1, sum(active))
        speed = sum(limb_speeds) / count

        per_limb_accel = [0.0, 0.0, 0.0, 0.0]
        per_limb_decel = [0.0, 0.0, 0.0, 0.0]
        acceleration = 0.0
        if self.velocity is not None:
            for i, (a, b, ok, was) in enumerate(zip(self.velocity, velocities, active, self.previous_valid)):
                if ok and was:
                    diff = tuple((nb - na) / dt for na, nb in zip(a, b))
                    per_limb_accel[i] = math.sqrt(sum(c * c for c in diff))
                    prev_spd = math.sqrt(sum(c * c for c in a))
                    if prev_spd > 1e-6:
                        per_limb_decel[i] = sum(d * v for d, v in zip(diff, a)) / prev_spd
            acceleration = sum(per_limb_accel) / count

        # Smooth continuous leg & stance macros.
        leg_lifts = [0.0, 0.0]
        for side in (0, 1):
            idx = 2 + side
            if not active[idx]:
                continue
            cur_y = positions[idx][1]
            if self.leg_y_baseline[side] is None:
                self.leg_y_baseline[side] = cur_y
            else:
                b_tau = 0.55 if cur_y > self.leg_y_baseline[side] else 3.5
                self.leg_y_baseline[side] += (1.0 - math.exp(-clamped_dt / b_tau)) * (cur_y - self.leg_y_baseline[side])
            lift_scale = 0.40 if legs_source == 'knees' else 0.52
            leg_lifts[side] = clamp((self.leg_y_baseline[side] - cur_y) / lift_scale)

        self.last_left_leg_lift = leg_lifts[0] if active[2] else self.last_left_leg_lift * math.exp(-clamped_dt / 0.25)
        self.last_right_leg_lift = leg_lifts[1] if active[3] else self.last_right_leg_lift * math.exp(-clamped_dt / 0.25)
        if active[2] and active[3]:
            self.last_stance_width = clamp((abs(positions[2][0] - positions[3][0]) - 0.18) / 0.85)
        raw_leg_speed = 0.5 * (limb_speeds[2] + limb_speeds[3]) if (active[2] and active[3]) else max(limb_speeds[2], limb_speeds[3])
        clean_leg_spd = max(0.0, raw_leg_speed - 0.35)
        target_leg_energy = clamp((clean_leg_spd / 2.2) + 0.50 * max(self.last_left_leg_lift, self.last_right_leg_lift)) if legs_visible else 0.0
        leg_tau = 0.12 if target_leg_energy > self.leg_energy else 0.35
        self.leg_energy += (1.0 - math.exp(-clamped_dt / leg_tau)) * (target_leg_energy - self.leg_energy)

        target_energy = clamp(speed / 5.4 + (0.20 * self.leg_energy if legs_visible else 0.0))
        e_tau = 0.12 if target_energy > self.energy else 0.42
        self.energy += (1.0 - math.exp(-clamped_dt / e_tau)) * (target_energy - self.energy)

        if active[0] and active[1]:
            self.last_expansion = clamp((math.dist(positions[0], positions[1]) - 0.5) / 2.8)
        if active[0]:
            self.last_left_height = clamp((-positions[0][1] + 0.25) / 1.8)
        if active[1]:
            self.last_right_height = clamp((-positions[1][1] + 0.25) / 1.8)

        left_vy = -velocities[0][1] if active[0] else 0.0
        right_vy = -velocities[1][1] if active[1] else 0.0
        left_leg_vy = -velocities[2][1] if active[2] else 0.0
        right_leg_vy = -velocities[3][1] if active[3] else 0.0
        left_bend = elbow_bend(xyz[11], xyz[13], xyz[15]) if (vis[11] and vis[13] and vis[15]) else 0.5
        right_bend = elbow_bend(xyz[12], xyz[14], xyz[16]) if (vis[12] and vis[14] and vis[16]) else 0.5

        # Head & shoulder continuous macros (cross-checked with ears so looking down never fakes a shrug).
        head_nod_vel = 0.0
        nose_rel_y = 0.0
        shoulder_drop = False
        if vis[0]:
            nose_f = self._nose_filter.filter(xyz[0], clamped_dt)
            nose_rel_y = (shoulder[1] - nose_f[1]) / scale
            self.last_head_pitch = clamp((nose_rel_y - 0.22) / 0.48)
            if self.prev_nose_rel_y is not None:
                head_nod_vel = (nose_rel_y - self.prev_nose_rel_y) / dt
            self.prev_nose_rel_y = nose_rel_y

            raw_tilt = 0.0
            if vis[7] and vis[8]:
                ear_dx = xyz[8][0] - xyz[7][0]
                ear_dy = xyz[8][1] - xyz[7][1]
                if abs(ear_dx) > 1e-4:
                    raw_tilt = clamp(math.atan2(ear_dy, abs(ear_dx)) / 0.30, -1.0, 1.0)
            else:
                raw_tilt = clamp(((nose_f[0] - shoulder[0]) / scale) / 0.26, -1.0, 1.0)
            self.last_head_tilt = 0.0 if abs(raw_tilt) < 0.10 else clamp(
                math.copysign((abs(raw_tilt) - 0.10) / 0.90, raw_tilt), -1.0, 1.0
            )

            avg_arm_h = 0.5 * (self.last_left_height + self.last_right_height)
            ear_rel_y = None
            look_suppress = 1.0
            if vis[7] and vis[8]:
                ear_mid_y = 0.5 * (xyz[7][1] + xyz[8][1])
                ear_rel_y = (shoulder[1] - ear_mid_y) / scale
                nose_below_ears = (nose_f[1] - ear_mid_y) / scale
                look_suppress = clamp((0.12 - nose_below_ears) / 0.08)
            if self.neck_baseline is None:
                self.neck_baseline = nose_rel_y
                if ear_rel_y is not None:
                    self.ear_neck_baseline = ear_rel_y
            elif self.shoulder_lift < 0.25 and avg_arm_h < 0.60:
                adapt = 1.0 - math.exp(-clamped_dt / 4.0)
                self.neck_baseline += adapt * (nose_rel_y - self.neck_baseline)
                if ear_rel_y is not None:
                    self.ear_neck_baseline = ear_rel_y if self.ear_neck_baseline is None else (
                        self.ear_neck_baseline + adapt * (ear_rel_y - self.ear_neck_baseline)
                    )
            arm_suppress = clamp((0.78 - avg_arm_h) / 0.18)
            nose_shrug = self.neck_baseline - nose_rel_y
            shrug_delta = min(nose_shrug, self.ear_neck_baseline - ear_rel_y) if (ear_rel_y is not None and self.ear_neck_baseline is not None) else nose_shrug
            raw_lift = clamp((shrug_delta - 0.04) / 0.12) * arm_suppress * look_suppress
            lift_tau = 0.09 if raw_lift > self.shoulder_lift else 0.14
            self.shoulder_lift += (1.0 - math.exp(-clamped_dt / lift_tau)) * (raw_lift - self.shoulder_lift)
            if self.shoulder_lift >= 0.65:
                self.shrug_armed = True
            elif self.shrug_armed and self.shoulder_lift <= 0.25 and (timestamp - self.last_shoulder_drop) >= 0.70:
                shoulder_drop = True
                self.shrug_armed = False
                self.last_shoulder_drop = timestamp

        sh_vec_dx = max(1e-4, abs(xyz[12][0] - xyz[11][0]))
        raw_sh_tilt = clamp(math.atan2(xyz[12][1] - xyz[11][1], sh_vec_dx) / 0.22, -1.0, 1.0)
        self.last_shoulder_tilt = 0.0 if abs(raw_sh_tilt) < 0.06 else raw_sh_tilt
        sh_tilt_vel = abs(self.last_shoulder_tilt - self.prev_shoulder_tilt) / dt
        self.prev_shoulder_tilt = self.last_shoulder_tilt
        target_bounce = clamp(sh_tilt_vel / 3.5)
        b_tau = 0.10 if target_bounce > self.shoulder_bounce else 0.42
        self.shoulder_bounce += (1.0 - math.exp(-clamped_dt / b_tau)) * (target_bounce - self.shoulder_bounce)

        # Torso & Hip smooth macros.
        torso_dx = shoulder[0] - hip[0]
        torso_dy = max(1e-4, hip[1] - shoulder[1])
        self.last_torso_lean = clamp(math.atan2(torso_dx, torso_dy) / 0.45, -1.0, 1.0)
        sh_dx = abs(xyz[12][0] - xyz[11][0]) / scale
        sh_dz = abs(xyz[12][2] - xyz[11][2]) / scale
        self.last_torso_twist = clamp(max(clamp((0.68 - sh_dx) / 0.45), clamp((sh_dz - 0.15) / 0.65)))

        hip_f = self._hip_filter.filter(hip, clamped_dt)
        raw_hip_y = (points[23][1] + points[24][1]) * 0.5
        if self.hip_y_baseline is None:
            self.hip_y_baseline = raw_hip_y
        else:
            self.hip_y_baseline += (1.0 - math.exp(-clamped_dt / 5.0)) * (raw_hip_y - self.hip_y_baseline)
        rel_elev = (self.hip_y_baseline - raw_hip_y) / max(scale, 0.05)
        self.last_body_level = clamp(0.65 + rel_elev * 0.90, 0.15, 1.0)

        if self.hip_x_baseline is None:
            self.hip_x_baseline = hip_f[0]
        else:
            self.hip_x_baseline += (1.0 - math.exp(-clamped_dt / 1.5)) * (hip_f[0] - self.hip_x_baseline)
        self.last_hip_sway = clamp(((hip_f[0] - self.hip_x_baseline) / scale) / 0.35, -1.0, 1.0)

        hip_bounce_vel = (self.prev_hip_y - hip_f[1]) / (scale * dt) if self.prev_hip_y is not None else 0.0
        self.prev_hip_y = hip_f[1]

        if active[0] and active[1]:
            self.last_lateral_balance = clamp((positions[0][0] + positions[1][0]) * 0.5, -1.0, 1.0)
        elif active[0]:
            self.last_lateral_balance = clamp(positions[0][0] * 0.5, -1.0, 1.0)
        elif active[1]:
            self.last_lateral_balance = clamp(positions[1][0] * 0.5, -1.0, 1.0)

        arm_avg_speed = 0.5 * (limb_speeds[0] + limb_speeds[1])
        leg_pulse = (self.last_left_leg_lift + self.last_right_leg_lift + 0.30 * (limb_speeds[2] + limb_speeds[3])) if legs_visible else 0.0
        body_bpm, body_bpm_conf = self.tempo_tracker.update(
            timestamp,
            hip_y=hip_f[1] / scale,
            head_y=nose_rel_y,
            arm_speed=arm_avg_speed,
            shoulder_tilt=self.last_shoulder_tilt,
            leg_pulse=leg_pulse,
            bpm_min=bpm_range[0],
            bpm_max=bpm_range[1],
        )

        pairs = [i for i in (0, 2) if active[i] and active[i + 1]]
        error = sum(
            abs(positions[i][0] + positions[i + 1][0]) + abs(positions[i][1] - positions[i + 1][1])
            for i in pairs
        ) / max(1, 2 * len(pairs))

        self.history.append((timestamp, speed))
        periodicity = clamp(max(self._short_periodicity(timestamp), body_bpm_conf))

        # Mirror-invariant forearm cross metric: positive when uncrossed, negative when crossed in an "X".
        sh_sign = 1.0 if (xyz[11][0] >= xyz[12][0]) else -1.0
        wrist_cross_metric = ((xyz[15][0] - xyz[16][0]) * sh_sign) / scale
        # 6-joint arm chain heights relative to hip (up is +) for ARM_WAVE propagation detection.
        chain_y = tuple((hip[1] - xyz[idx][1]) / scale for idx in (15, 13, 11, 12, 14, 16))

        move_snap = self.move_recognizer.update(
            timestamp=timestamp,
            dt=clamped_dt,
            energy=self.energy,
            speed=speed,
            expansion=self.last_expansion,
            periodicity=periodicity,
            left_height=self.last_left_height,
            right_height=self.last_right_height,
            left_vy=left_vy,
            right_vy=right_vy,
            left_speed=limb_speeds[0],
            right_speed=limb_speeds[1],
            left_bend=left_bend,
            right_bend=right_bend,
            left_decel=per_limb_decel[0],
            right_decel=per_limb_decel[1],
            arms_active=(active[0], active[1]),
            wrist_cross_metric=wrist_cross_metric,
            chain_y=chain_y,
            hip_sway=self.last_hip_sway,
            hip_bounce_vel=hip_bounce_vel,
            body_level=self.last_body_level,
            legs_visible=legs_visible,
            left_leg_lift=self.last_left_leg_lift,
            right_leg_lift=self.last_right_leg_lift,
            left_leg_vy=left_leg_vy,
            right_leg_vy=right_leg_vy,
        )

        self.previous, self.velocity, self.last_time = positions, velocities, timestamp
        self.last_valid_time = timestamp
        self.previous_valid = active

        result = Features(
            energy=self.energy,
            fluidity=1.0 / (1.0 + acceleration / 20.0),
            expansion=self.last_expansion,
            stillness=1.0 - self.energy,
            velocity=speed,
            acceleration=acceleration,
            periodicity=periodicity,
            symmetry=clamp(1.0 - error),
            tracked=True,
            left_height=self.last_left_height,
            right_height=self.last_right_height,
            left_vy=left_vy,
            right_vy=right_vy,
            left_speed=limb_speeds[0],
            right_speed=limb_speeds[1],
            left_bend=left_bend,
            right_bend=right_bend,
            head_pitch=self.last_head_pitch,
            head_tilt=self.last_head_tilt,
            head_nod_vel=head_nod_vel,
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
            legs_visible=legs_visible,
            legs_source=legs_source,
            left_leg_lift=self.last_left_leg_lift,
            right_leg_lift=self.last_right_leg_lift,
            left_leg_speed=limb_speeds[2],
            right_leg_speed=limb_speeds[3],
            leg_energy=self.leg_energy,
            stance_width=self.last_stance_width,
            moves=move_snap.events,
            soundwave_rise=move_snap.soundwave_rise,
            arms_crossed=move_snap.arms_crossed,
            body_frozen=move_snap.body_frozen,
            side_step_groove=move_snap.side_step_groove,
            squat_depth=move_snap.squat_depth,
            active_move_label=move_snap.active_label,
            body_bpm=body_bpm,
            body_bpm_confidence=body_bpm_conf,
            body_bpm_source=self.tempo_tracker.source,
            grace=False,
        )
        self.last_features = result
        return result

    def _short_periodicity(self, timestamp: float) -> float:
        if len(self.history) <= 20 or timestamp - self.history[0][0] <= 0.8:
            return 0.0
        values = []
        history = list(self.history)
        index = 0
        for n in range(30):
            t = timestamp - (29 - n) / 30.0
            while index + 1 < len(history) and history[index + 1][0] <= t:
                index += 1
            values.append(history[index][1])
        mean = sum(values) / len(values)
        centered = [v - mean for v in values]
        if sum(v * v for v in centered) <= 0.05:
            return 0.0
        periodicity = 0.0
        for lag in range(8, 21):
            a, b = centered[lag:], centered[:-lag]
            norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
            if norm > 1e-8:
                periodicity = max(periodicity, sum(x * y for x, y in zip(a, b)) / norm)
        return periodicity


def mock_features(seconds: float) -> Features:
    """Expressive synthetic studio dance for offline tests & audio auditioning."""
    activity = 0.5 + 0.35 * math.sin(seconds * 0.25)
    energy = clamp(activity * (0.72 + 0.28 * math.sin(seconds * 2.5) ** 2))
    return Features(
        energy=energy,
        fluidity=clamp(0.72 + 0.22 * math.sin(seconds * 0.4)),
        expansion=clamp(0.5 + 0.42 * math.sin(seconds * 0.18)),
        stillness=1.0 - energy,
        velocity=energy * 2.5,
        acceleration=energy * 10.0,
        periodicity=clamp(0.55 + 0.40 * math.sin(seconds * 0.3)),
        symmetry=clamp(0.75 + 0.20 * math.cos(seconds * 0.15)),
        tracked=True,
        left_height=clamp(0.5 + 0.4 * math.sin(seconds * 0.35)),
        right_height=clamp(0.5 + 0.42 * math.sin(seconds * 0.28)),
        left_vy=math.cos(seconds * 0.35) * 0.6,
        right_vy=math.cos(seconds * 0.28) * 0.6,
        head_pitch=clamp(0.5 + 0.30 * math.sin(seconds * 0.22)),
        head_tilt=clamp(0.45 * math.sin(seconds * 0.30), -1.0, 1.0),
        torso_lean=clamp(0.55 * math.sin(seconds * 0.20), -1.0, 1.0),
        torso_twist=clamp(0.35 + 0.30 * math.sin(seconds * 0.17)),
        hip_sway=clamp(0.55 * math.sin(seconds * 1.1), -1.0, 1.0),
        body_level=clamp(0.65 + 0.20 * math.sin(seconds * 0.19)),
        lateral_balance=clamp(0.55 * math.sin(seconds * 0.22), -1.0, 1.0),
        body_bpm=124.0,
        body_bpm_confidence=0.72,
        body_bpm_source='hip',
        grace=False,
    )
