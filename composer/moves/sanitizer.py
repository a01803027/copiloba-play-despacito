"""Tier 0 Pose Sanitizer: 2D-primary projection + Anti-Wiggle & Bone-Length Glitch Guard.

Eliminates monocular MediaPipe Z-depth hallucinations (10x noisier than 2D xy) and rejects
anatomically impossible 1-frame teleports or ping-pong arm wiggles before they reach any
move detector or ML window.
"""
from __future__ import annotations

import math
from ..kinematics.filters import clamp
from .pose_frame import CANONICAL_JOINTS, FEATURE_DIM, PoseFrame


class _OneEuroFilter2D:
    """Snappy 2D adaptive low-pass filter (minimal lag on fast boxing/dance motion)."""

    def __init__(self, min_cutoff: float = 2.2, beta: float = 0.65, d_cutoff: float = 1.5):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x: tuple[float, float] | None = None
        self._dx: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, value: tuple[float, float], dt: float) -> tuple[float, float]:
        if self._x is None or dt <= 0.0 or dt > 0.25:
            self._x = value
            self._dx = (0.0, 0.0)
            return value
        a_d = self._alpha(self.d_cutoff, dt)
        dx0 = (value[0] - self._x[0]) / dt
        dx1 = (value[1] - self._x[1]) / dt
        self._dx = (
            self._dx[0] + a_d * (dx0 - self._dx[0]),
            self._dx[1] + a_d * (dx1 - self._dx[1]),
        )
        speed = math.hypot(self._dx[0], self._dx[1])
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self._x = (
            self._x[0] + a * (value[0] - self._x[0]),
            self._x[1] + a * (value[1] - self._x[1]),
        )
        return self._x


def elbow_bend_2d(
    shoulder: tuple[float, float],
    elbow: tuple[float, float],
    wrist: tuple[float, float],
) -> float:
    """Compute 2D elbow bend in [0=straight, 1=fully folded], immune to monocular Z flips."""
    ax, ay = shoulder[0] - elbow[0], shoulder[1] - elbow[1]
    bx, by = wrist[0] - elbow[0], wrist[1] - elbow[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na * nb < 1e-7:
        return 0.5
    cos_ang = clamp((ax * bx + ay * by) / (na * nb), -1.0, 1.0)
    return clamp(1.0 - math.acos(cos_ang) / math.pi)


class PoseSanitizer:
    """Sanitizes raw 33-point MediaPipe Pose into a glitch-free canonical 2D PoseFrame."""

    # Anatomically impossible projected 2D bone length (in torso-heights).
    # Normal human arm segment is ~0.45-0.75 torso, up to ~1.35 under foreshortened lean;
    # >1.65 or a >0.85 single-frame stretch is a MediaPipe hallucination.
    MAX_ARM_SEG = 1.65
    MAX_ARM_SEG_JUMP = 0.85
    # Visibility gates: tighter on extremities that hallucinate during fast motion blur.
    VIS_GATES: dict[str, float] = {
        'nose': 0.50,
        'l_shoulder': 0.50,
        'r_shoulder': 0.50,
        'l_hip': 0.50,
        'r_hip': 0.50,
        'l_elbow': 0.55,
        'r_elbow': 0.55,
        'l_wrist': 0.55,
        'r_wrist': 0.55,
        'l_knee': 0.50,
        'r_knee': 0.50,
        'l_ankle': 0.60,
        'r_ankle': 0.60,
    }
    # Standing neutral 2D defaults (hip-centered, UP=+y) used when lower-body joints are off-screen
    # so closer framing or taller performers never inject MediaPipe hallucination coordinates into ML!
    NEUTRAL_FALLBACK: dict[str, tuple[float, float]] = {
        'l_knee': (-0.22, -0.75),
        'r_knee': (0.22, -0.75),
        'l_ankle': (-0.22, -1.45),
        'r_ankle': (0.22, -1.45),
    }

    def __init__(self):
        self._filters: dict[str, _OneEuroFilter2D] = {}
        for _, name in CANONICAL_JOINTS:
            if 'wrist' in name or 'elbow' in name:
                self._filters[name] = _OneEuroFilter2D(min_cutoff=2.2, beta=0.65, d_cutoff=1.5)
            else:
                self._filters[name] = _OneEuroFilter2D(min_cutoff=1.6, beta=0.35, d_cutoff=1.2)
        self.reset()

    def reset(self):
        for flt in self._filters.values():
            flt.reset()
        self._prev_pos: dict[str, tuple[float, float]] = {}
        self._prev_step: dict[str, tuple[float, float]] = {}
        self._prev_valid: dict[str, bool] = {}
        self._prev_bone: dict[str, float] = {}
        self._prev_reach: tuple[float | None, float | None] = (None, None)
        self._hip_y_base: float | None = None
        self._last_t: float | None = None
        self.total_glitches_blocked = 0

    def process(self, points, timestamp: float, aspect: float = 1.0) -> PoseFrame:
        if not points or len(points) != 33:
            return PoseFrame(timestamp=timestamp, dt=1.0 / 30.0, tracked=False)
        if not all(math.isfinite(c) for p in points for c in p[:2]):
            return PoseFrame(timestamp=timestamp, dt=1.0 / 30.0, tracked=False)

        xy = [(float(p[0]), float(p[1]) * aspect) for p in points]
        vis_raw = [float(p[3]) if len(p) >= 4 else 1.0 for p in points]

        # Core torso landmarks must be visible.
        if any(vis_raw[i] < 0.50 for i in (11, 12, 23, 24)):
            return PoseFrame(timestamp=timestamp, dt=1.0 / 30.0, tracked=False)

        hip = ((xy[23][0] + xy[24][0]) * 0.5, (xy[23][1] + xy[24][1]) * 0.5)
        shoulder = ((xy[11][0] + xy[12][0]) * 0.5, (xy[11][1] + xy[12][1]) * 0.5)
        scale = math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1])
        if scale < 0.025:
            return PoseFrame(timestamp=timestamp, dt=1.0 / 30.0, tracked=False)

        dt = (timestamp - self._last_t) if self._last_t is not None else (1.0 / 30.0)
        if dt <= 0.0:
            dt = 1.0 / 30.0
        elif dt > 0.28:
            self.reset()
            dt = 1.0 / 30.0
        clamped_dt = min(dt, 0.20)

        # Normalize all 13 canonical joints to hip-centered 2D with UP = +y.
        raw_norm: dict[str, tuple[float, float]] = {}
        vis_map: dict[str, float] = {}
        valid_map: dict[str, bool] = {}
        for idx, name in CANONICAL_JOINTS:
            raw_norm[name] = ((xy[idx][0] - hip[0]) / scale, (hip[1] - xy[idx][1]) / scale)
            v = vis_raw[idx]
            vis_map[name] = v
            valid_map[name] = v >= self.VIS_GATES.get(name, 0.50)

        glitch_joints: list[str] = []

        # 1) Anatomical bone-length sanity guard on upper arms & forearms.
        # Only triggers on genuine MediaPipe hallucinations (>1.65 torso or sudden >0.85 stretch).
        arm_bones = (
            ('l_shoulder', 'l_elbow', 'l_uarm'),
            ('l_elbow', 'l_wrist', 'l_farm'),
            ('r_shoulder', 'r_elbow', 'r_uarm'),
            ('r_elbow', 'r_wrist', 'r_farm'),
        )
        for parent, child, bone_key in arm_bones:
            if valid_map[parent] and valid_map[child]:
                dx = raw_norm[child][0] - raw_norm[parent][0]
                dy = raw_norm[child][1] - raw_norm[parent][1]
                length = math.hypot(dx, dy)
                prev_len = self._prev_bone.get(bone_key)
                impossible_len = length > self.MAX_ARM_SEG
                impossible_jump = prev_len is not None and length > 1.45 and (length - prev_len) > self.MAX_ARM_SEG_JUMP
                if impossible_len or impossible_jump:
                    glitch_joints.append(child)
                    if child in self._prev_pos:
                        prev_c = self._prev_pos[child]
                        raw_norm[child] = (
                            prev_c[0] + 0.20 * (raw_norm[child][0] - prev_c[0]),
                            prev_c[1] + 0.20 * (raw_norm[child][1] - prev_c[1]),
                        )
                else:
                    self._prev_bone[bone_key] = length

        # 2) Teleport & ping-pong wiggle guard + adaptive 2D One-Euro filtering.
        joints_2d: dict[str, tuple[float, float]] = {}
        vel_2d: dict[str, tuple[float, float]] = {}
        speed_2d: dict[str, float] = {}
        # Fast boxing punches reach ~14-18 torso/s; only reject physically impossible teleports (>22 torso/s).
        max_step = max(1.15, 22.0 * clamped_dt)

        for _, name in CANONICAL_JOINTS:
            cur = raw_norm[name]
            ok = valid_map[name]
            prev = self._prev_pos.get(name)
            was_ok = self._prev_valid.get(name, False)

            if not ok:
                self._filters[name].reset()
                if name in self.NEUTRAL_FALLBACK:
                    held = self.NEUTRAL_FALLBACK[name]
                else:
                    held = prev if prev is not None else cur
                joints_2d[name] = held
                vel_2d[name] = (0.0, 0.0)
                speed_2d[name] = 0.0
                self._prev_step[name] = (0.0, 0.0)
                continue

            if prev is not None and was_ok:
                dx = cur[0] - prev[0]
                dy = cur[1] - prev[1]
                step = math.hypot(dx, dy)

                # Reject 1-frame teleport spikes (physically impossible displacement).
                if step > max_step:
                    if name not in glitch_joints:
                        glitch_joints.append(name)
                    scale_back = (0.45 * max_step) / max(step, 1e-6)
                    cur = (prev[0] + dx * scale_back, prev[1] + dy * scale_back)
                    dx = cur[0] - prev[0]
                    dy = cur[1] - prev[1]
                    step = math.hypot(dx, dy)

                # Reject 1-frame ping-pong oscillations (arm wiggling back and forth violently).
                pstep = self._prev_step.get(name, (0.0, 0.0))
                pmag = math.hypot(pstep[0], pstep[1])
                if step > 0.35 and pmag > 0.35:
                    dot = (dx * pstep[0] + dy * pstep[1]) / (step * pmag)
                    if dot < -0.75:
                        if name not in glitch_joints:
                            glitch_joints.append(name)
                        cur = (prev[0] + 0.25 * dx, prev[1] + 0.25 * dy)

                filtered = self._filters[name].filter(cur, clamped_dt)
                vx = (filtered[0] - prev[0]) / dt
                vy = (filtered[1] - prev[1]) / dt
                spd = math.hypot(vx, vy)
                self._prev_step[name] = (filtered[0] - prev[0], filtered[1] - prev[1])
            else:
                self._filters[name].reset()
                filtered = self._filters[name].filter(cur, clamped_dt)
                vx, vy, spd = 0.0, 0.0, 0.0
                self._prev_step[name] = (0.0, 0.0)

            joints_2d[name] = filtered
            vel_2d[name] = (vx, vy)
            speed_2d[name] = spd

        # 3) Clean 2D elbow bends, outward shoulder->wrist reach velocity, & wrist cross metric.
        l_bend = (
            elbow_bend_2d(joints_2d['l_shoulder'], joints_2d['l_elbow'], joints_2d['l_wrist'])
            if (valid_map['l_shoulder'] and valid_map['l_elbow'] and valid_map['l_wrist'])
            else 0.5
        )
        r_bend = (
            elbow_bend_2d(joints_2d['r_shoulder'], joints_2d['r_elbow'], joints_2d['r_wrist'])
            if (valid_map['r_shoulder'] and valid_map['r_elbow'] and valid_map['r_wrist'])
            else 0.5
        )
        l_reach = (
            math.hypot(
                joints_2d['l_wrist'][0] - joints_2d['l_shoulder'][0],
                joints_2d['l_wrist'][1] - joints_2d['l_shoulder'][1],
            )
            if (valid_map['l_shoulder'] and valid_map['l_wrist'])
            else 0.5
        )
        r_reach = (
            math.hypot(
                joints_2d['r_wrist'][0] - joints_2d['r_shoulder'][0],
                joints_2d['r_wrist'][1] - joints_2d['r_shoulder'][1],
            )
            if (valid_map['r_shoulder'] and valid_map['r_wrist'])
            else 0.5
        )
        prev_lr, prev_rr = self._prev_reach
        l_reach_vel = (
            (l_reach - prev_lr) / dt
            if (prev_lr is not None and valid_map['l_wrist'] and self._prev_valid.get('l_wrist', False))
            else 0.0
        )
        r_reach_vel = (
            (r_reach - prev_rr) / dt
            if (prev_rr is not None and valid_map['r_wrist'] and self._prev_valid.get('r_wrist', False))
            else 0.0
        )
        self._prev_reach = (
            l_reach if valid_map['l_wrist'] else None,
            r_reach if valid_map['r_wrist'] else None,
        )
        sh_sign = 1.0 if (joints_2d['l_shoulder'][0] >= joints_2d['r_shoulder'][0]) else -1.0
        wrist_cross = (joints_2d['l_wrist'][0] - joints_2d['r_wrist'][0]) * sh_sign

        # 4) Standing hip elevation (aspect-corrected to match `scale` across 4:3 and 16:9 cameras!) & torso lean.
        hip_y_scaled = hip[1]
        if self._hip_y_base is None:
            self._hip_y_base = hip_y_scaled
        else:
            self._hip_y_base += (1.0 - math.exp(-clamped_dt / 5.0)) * (hip_y_scaled - self._hip_y_base)
        hip_rel_elev = clamp((self._hip_y_base - hip_y_scaled) / max(scale, 0.05), -1.5, 1.5)
        torso_lean = clamp(math.atan2(shoulder[0] - hip[0], max(1e-4, hip[1] - shoulder[1])) / 0.45, -1.0, 1.0)

        feature_vec = (
            joints_2d['l_shoulder'][0], joints_2d['l_shoulder'][1],
            joints_2d['r_shoulder'][0], joints_2d['r_shoulder'][1],
            joints_2d['l_elbow'][0], joints_2d['l_elbow'][1],
            joints_2d['r_elbow'][0], joints_2d['r_elbow'][1],
            joints_2d['l_wrist'][0], joints_2d['l_wrist'][1],
            joints_2d['r_wrist'][0], joints_2d['r_wrist'][1],
            clamp(vel_2d['l_wrist'][0] / 8.0, -2.0, 2.0), clamp(vel_2d['l_wrist'][1] / 8.0, -2.0, 2.0),
            clamp(vel_2d['r_wrist'][0] / 8.0, -2.0, 2.0), clamp(vel_2d['r_wrist'][1] / 8.0, -2.0, 2.0),
            l_bend, r_bend,
            joints_2d['l_knee'][0], joints_2d['l_knee'][1],
            joints_2d['r_knee'][0], joints_2d['r_knee'][1],
            hip_rel_elev, torso_lean,
        )
        assert len(feature_vec) == FEATURE_DIM

        if glitch_joints:
            self.total_glitches_blocked += len(glitch_joints)

        self._prev_pos = joints_2d
        self._prev_valid = valid_map
        self._last_t = timestamp

        return PoseFrame(
            timestamp=timestamp,
            dt=dt,
            tracked=True,
            scale=scale,
            hip_xy=hip,
            shoulder_xy=shoulder,
            joints_2d=joints_2d,
            vel_2d=vel_2d,
            speed_2d=speed_2d,
            vis=vis_map,
            elbow_bend_2d=(l_bend, r_bend),
            arm_reach=(l_reach, r_reach),
            arm_reach_vel=(l_reach_vel, r_reach_vel),
            wrist_cross_metric=wrist_cross,
            glitch_joints=tuple(glitch_joints),
            feature_vector=feature_vec,
        )
