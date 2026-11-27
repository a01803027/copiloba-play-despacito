"""Real-time 11-move Full-Body Biomechanical & Dance Recognizer (Phase 2 — Fast 2D Kinematics).

Recognizes 8 upper/full-body moves + 3 lower-body/leg moves over a rolling sanitized 2D pose buffer:
  Upper / Full-Body:
    1. HARD_PUNCH (left / right)   — outward 2D wrist-thrust impulse + combo re-arm (up to ~5/s per arm!)
    2. DOUBLE_ARM_RISE             — both arms rising overhead -> continuous Soundwave Swell
    3. OVERHEAD_DROP               — both arms slamming down from high -> Beat Drop
    4. ARMS_CROSS_CUT              — forearms crossed in an "X" at chest -> DJ Breakdown Cut
    5. OPEN_BURST                  — arms thrown wide open from chest -> Euphoric Bloom
    6. ARM_WAVE (lr / rl)          — traveling crest OR phase-lagged wave across 4-joint arm chain
    7. BODY_FREEZE                 — sudden stop after active dancing -> Echo Tail Freeze
    8. SIDE_STEP_GROOVE            — periodic lateral hip sway + bounce -> Groove Pocket Lock
  Legs / Lower-Body (works with knees 25/26 OR ankles 27/28):
    9. KNEE_STOMP (left / right)   — asymmetric single-leg high lift + crisp stomp landing (0 double-stomps)
   10. SQUAT_DIVE / SQUAT_RISE     — deep crouch filter sub-dive -> spring-up filter pop
   11. JUMP_HIT                    — upward hop above standing baseline -> Air Crash & Kick
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable
from ..kinematics.filters import clamp
from .arbiter import MoveArbiter


@dataclass(frozen=True)
class MoveEvent:
    kind: str               # e.g. 'HARD_PUNCH', 'DOUBLE_ARM_RISE', 'ARM_WAVE', 'KNEE_STOMP', ...
    side: str = 'both'      # 'left' | 'right' | 'both' | 'lr' | 'rl'
    intensity: float = 1.0  # 0..1 expressive strength
    progress: float = 1.0   # 0..1 continuous progress where applicable


@dataclass(frozen=True)
class MoveSnapshot:
    events: tuple[MoveEvent, ...] = ()
    soundwave_rise: float = 0.0    # 0..1 continuous overhead soundwave swell
    arms_crossed: bool = False     # True while holding X-cross DJ Breakdown
    body_frozen: bool = False      # True during dramatic Freeze echo pause
    side_step_groove: bool = False # True during rhythmic side-to-side groove
    squat_depth: float = 0.0       # 0..1 continuous deep squat underwater filter dive
    active_label: str = ''         # Short human-readable label for HUD


class BodyMoveRecognizer:
    """Fast 2D biomechanical state machines + cross-correlation phase wave detector."""

    def __init__(self):
        self.arbiter = MoveArbiter()
        self.reset()

    def reset(self):
        self.arbiter.reset()
        self._buf: deque[dict] = deque(maxlen=120)  # ~2-4s at 30-60 FPS
        self._cooldowns: dict[str, float] = {}
        self._punch_rearmed = [True, True]
        self._cross_held = 0.0
        self._arms_crossed = False
        self._freeze_held = 0.0
        self._body_frozen = False
        self._rise_active = False
        self._squat_armed = False
        self._squat_held = 0.0
        self._knee_armed = [False, False]
        self._step_armed = [False, False]
        self._step_peak_lift = [0.0, 0.0]
        self._last_label = ''
        self._last_label_until = -1e9

    def _ready(self, key: str, now: float, cd: float) -> bool:
        return (now - self._cooldowns.get(key, -1e9)) >= cd

    def _fire(self, events: list[MoveEvent], key: str, now: float, ev: MoveEvent, label: str, label_s: float = 0.60):
        self._cooldowns[key] = now
        events.append(ev)
        self._last_label = label
        self._last_label_until = now + label_s

    def update(
        self,
        *,
        timestamp: float,
        dt: float,
        energy: float,
        speed: float,
        expansion: float,
        periodicity: float,
        left_height: float,
        right_height: float,
        left_vy: float,
        right_vy: float,
        left_speed: float,
        right_speed: float,
        left_bend: float,
        right_bend: float,
        left_decel: float,
        right_decel: float,
        arms_active: tuple[bool, bool],
        wrist_cross_metric: float,      # < 0 when wrists are crossed in an X (mirror-invariant!)
        chain_y: tuple[float, ...],     # 6-joint heights: L_wr, L_el, L_sh, R_sh, R_el, R_wr (up is +)
        hip_sway: float,
        hip_bounce_vel: float,
        body_level: float,
        legs_visible: bool,
        left_leg_lift: float,
        right_leg_lift: float,
        left_leg_vy: float,
        right_leg_vy: float,
        left_reach: float = 0.5,
        right_reach: float = 0.5,
        left_reach_vel: float = 0.0,
        right_reach_vel: float = 0.0,
        glitch_joints: tuple[str, ...] = (),
        extra_candidates: Iterable[MoveEvent] = (),
        tier2_confidence: float = 0.0,
    ) -> MoveSnapshot:
        clamped_dt = min(max(dt, 1e-3), 0.2)
        sample = {
            't': timestamp,
            'energy': energy,
            'speed': speed,
            'exp': expansion,
            'lh': left_height,
            'rh': right_height,
            'lvy': left_vy,
            'rvy': right_vy,
            'lspd': left_speed,
            'rspd': right_speed,
            'lbend': left_bend,
            'rbend': right_bend,
            'lreach': left_reach,
            'rreach': right_reach,
            'lrvel': left_reach_vel,
            'rrvel': right_reach_vel,
            'chain': chain_y,
            'sway': hip_sway,
            'lvl': body_level,
        }
        self._buf.append(sample)
        events: list[MoveEvent] = []

        def recent(sec: float) -> list[dict]:
            cutoff = timestamp - sec
            return [s for s in self._buf if s['t'] >= cutoff]

        win_fast = recent(0.28)
        win_med = recent(0.55)
        win_wave = recent(0.95)
        win_long = recent(1.20)

        # --- 1) TWO-TIER PUNCHES: LIGHT_PUNCH (Groove Jab) vs. HARD_PUNCH (Exaggerated Power Punch) ---
        # Casual arm swings below `rvel < 2.85` / `spd < 3.80` stay in the deadzone (zero spam!).
        # Bilateral symmetric swings (OPEN_BURST / DOUBLE_ARM_RISE / OVERHEAD_DROP) are rejected immediately.
        both_plunging = (left_vy <= -2.2 and right_vy <= -2.2)
        both_thrusting_sym = (left_reach_vel >= 1.85 and right_reach_vel >= 1.85)
        both_rising_sym = (left_vy >= 1.45 and right_vy >= 1.45 and abs(left_vy - right_vy) < 1.15)
        bilateral_motion = both_plunging or both_thrusting_sym or both_rising_sym

        for side_idx, side_name, active, bend_k, spd_k, rvel_k, reach_k, decel in (
            (0, 'left', arms_active[0], 'lbend', 'lspd', 'lrvel', 'lreach', left_decel),
            (1, 'right', arms_active[1], 'rbend', 'rspd', 'rrvel', 'rreach', right_decel),
        ):
            rvel = sample[rvel_k]
            cur_bend = sample[bend_k]

            # Re-arm once outward thrust peaks/retracts (rvel < 0.70) OR after 0.28s.
            if not self._punch_rearmed[side_idx]:
                if rvel < 0.70 or self._ready(f'punch_any_{side_name}', timestamp, 0.28):
                    self._punch_rearmed[side_idx] = True

            if (
                active
                and not bilateral_motion
                and self._punch_rearmed[side_idx]
                and len(win_fast) >= 2
            ):
                max_prev_bend = max(s[bend_k] for s in win_fast[:-1])
                min_prev_reach = min(s[reach_k] for s in win_fast[:-1])
                bend_drop = max_prev_bend - cur_bend
                reach_gain = sample[reach_k] - min_prev_reach
                peak_spd = max(s[spd_k] for s in win_fast[-2:])
                peak_rvel = max(s[rvel_k] for s in win_fast[-2:])

                # Tier B (Heavy): Exaggerated, unmistakable full-power punch.
                heavy_thrust = (
                    peak_rvel >= 4.60
                    and peak_spd >= 5.20
                    and (bend_drop >= 0.15 or reach_gain >= 0.22 or cur_bend <= 0.36)
                )
                heavy_snap = (
                    peak_spd >= 6.20
                    and peak_rvel >= 3.60
                    and (bend_drop >= 0.12 or decel <= -35.0)
                )
                # Tier A (Light): Moderate rhythmic jab -> subtle in-pocket percussion accent (no loud stab!).
                light_jab = (
                    peak_rvel >= 2.85
                    and peak_spd >= 3.80
                    and (bend_drop >= 0.12 or reach_gain >= 0.16)
                )

                if (heavy_thrust or heavy_snap) and self._ready(f'punch_heavy_{side_name}', timestamp, 0.26):
                    strength = clamp((max(peak_spd, peak_rvel) - 4.5) / 4.5, 0.75, 1.0)
                    tag = 'RIGHT POWER PUNCH!' if side_name == 'right' else 'LEFT SUB BOOM!'
                    self._punch_rearmed[side_idx] = False
                    self._cooldowns[f'punch_any_{side_name}'] = timestamp
                    self._fire(events, f'punch_heavy_{side_name}', timestamp, MoveEvent('HARD_PUNCH', side_name, strength), tag, 0.45)
                elif light_jab and self._ready(f'punch_any_{side_name}', timestamp, 0.28):
                    strength = clamp((peak_rvel - 2.6) / 2.6, 0.40, 0.72)
                    tag = 'RIGHT LIGHT JAB' if side_name == 'right' else 'LEFT LIGHT JAB'
                    self._punch_rearmed[side_idx] = False
                    self._fire(events, f'punch_any_{side_name}', timestamp, MoveEvent('LIGHT_PUNCH', side_name, strength), tag, 0.32)

        # --- 2) DOUBLE_ARM_RISE (Continuous Rising Soundwave Swell) ---
        both_ok = arms_active[0] and arms_active[1]
        avg_h = 0.5 * (left_height + right_height)
        avg_vy = 0.5 * (left_vy + right_vy)
        arms_sync = abs(left_height - right_height) <= 0.25
        soundwave_rise = clamp((min(left_height, right_height) - 0.56) / 0.36) if (both_ok and arms_sync) else 0.0

        if both_ok and arms_sync and soundwave_rise >= 0.18:
            was_lower = any(0.5 * (s['lh'] + s['rh']) <= 0.52 for s in win_long)
            if not self._rise_active and (was_lower or avg_vy >= 0.35) and self._ready('double_rise', timestamp, 1.50):
                self._rise_active = True
                self._fire(
                    events, 'double_rise', timestamp,
                    MoveEvent('DOUBLE_ARM_RISE', 'both', clamp(0.65 + 0.35 * soundwave_rise), soundwave_rise),
                    'SOUNDWAVE RISE!', 1.2,
                )
        elif avg_h < 0.48:
            self._rise_active = False

        # --- 3) OVERHEAD_DROP (Slamming both arms rapidly down from overhead -> Beat Drop!) ---
        if both_ok and len(win_med) >= 3 and self._ready('overhead_drop', timestamp, 0.90):
            was_overhead = any(min(s['lh'], s['rh']) >= 0.64 for s in win_med[:-1])
            plunging = max(left_height, right_height) <= 0.48 and left_vy <= -1.6 and right_vy <= -1.6
            if was_overhead and plunging:
                strength = clamp((-avg_vy - 1.4) / 3.0, 0.70, 1.0)
                self._rise_active = False
                self._fire(events, 'overhead_drop', timestamp, MoveEvent('OVERHEAD_DROP', 'both', strength), 'OVERHEAD DROP!', 0.9)

        # --- 4) ARMS_CROSS_CUT (Forearms crossed in an "X" at chest -> Clean DJ Breakdown!) ---
        crossing_now = both_ok and wrist_cross_metric < -0.05 and (0.20 <= avg_h <= 0.80)
        if crossing_now:
            self._cross_held += clamped_dt
            if not self._arms_crossed and self._cross_held >= 0.18:
                self._arms_crossed = True
                self._fire(events, 'arms_cross', timestamp, MoveEvent('ARMS_CROSS_CUT', 'both', 1.0), 'X-CUT BREAKDOWN', 0.8)
        else:
            was_crossed = self._arms_crossed
            if wrist_cross_metric > 0.12:
                self._cross_held = 0.0
            else:
                self._cross_held = max(0.0, self._cross_held - clamped_dt * 3.0)
            if self._cross_held <= 0.02:
                self._arms_crossed = False
                if was_crossed and self._ready('uncross_drop', timestamp, 0.60):
                    self._fire(events, 'uncross_drop', timestamp, MoveEvent('OPEN_BURST', 'both', 0.90), 'X-RELEASE DROP!', 0.8)

        # --- 5) OPEN_BURST (Throwing both arms wide open from chest -> Euphoric Bloom!) ---
        if both_ok and not self._arms_crossed and len(win_med) >= 3 and self._ready('open_burst', timestamp, 0.90):
            min_exp = min(s['exp'] for s in win_med[:-1])
            dt_span = max(0.05, timestamp - win_med[0]['t'])
            exp_rate = (expansion - min_exp) / dt_span
            if min_exp <= 0.36 and expansion >= 0.68 and exp_rate >= 0.90:
                self._fire(events, 'open_burst', timestamp, MoveEvent('OPEN_BURST', 'both', clamp(expansion, 0.7, 1.0)), 'OPEN BURST!', 0.85)

        # --- 6) ARM_WAVE (Traveling crest OR continuous phase-lagged wave L->R / R->L!) ---
        if both_ok and not self._arms_crossed and len(win_wave) >= 10 and self._ready('arm_wave', timestamp, 0.85):
            wave_dir = self._detect_arm_wave(win_wave)
            if wave_dir is not None:
                tag = 'ARM WAVE L->R!' if wave_dir == 'lr' else 'ARM WAVE R->L!'
                self._fire(events, 'arm_wave', timestamp, MoveEvent('ARM_WAVE', wave_dir, 0.92), tag, 0.85)

        # --- 7) BODY_FREEZE (Sudden stop after active dancing -> Dramatic Echo Freeze!) ---
        peak_recent_energy = max((s['energy'] for s in win_long), default=0.0)
        peak_recent_speed = max((s['speed'] for s in win_long), default=0.0)
        if not self._body_frozen:
            if (peak_recent_energy >= 0.35 or peak_recent_speed >= 2.20) and speed < 0.36:
                self._freeze_held += clamped_dt
                if self._freeze_held >= 0.28 and self._ready('freeze', timestamp, 1.20):
                    self._body_frozen = True
                    self._fire(events, 'freeze', timestamp, MoveEvent('BODY_FREEZE', 'both', 1.0), 'FREEZE ECHO!', 0.9)
            else:
                self._freeze_held = 0.0
        else:
            if speed > 0.78:
                self._body_frozen = False
                self._freeze_held = 0.0

        # --- 8) SIDE_STEP_GROOVE (Periodic side-to-side hip sway & bounce -> Groove Lock) ---
        sway_span = (max(s['sway'] for s in win_long) - min(s['sway'] for s in win_long)) if win_long else 0.0
        side_step_groove = (not self._body_frozen) and sway_span >= 0.42 and periodicity >= 0.40 and energy >= 0.14

        # --- 9) LEG MOVES — Heavy KNEE_STOMP + Sensitive Rhythmic FOOT_STEP ---
        # Requiring single-leg asymmetry eliminates 100% of simultaneous hop/camera-shift false triggers.
        if legs_visible:
            both_lifted_sym = (
                left_leg_lift >= 0.12
                and right_leg_lift >= 0.12
                and abs(left_leg_lift - right_leg_lift) < 0.08
            )
            if both_lifted_sym:
                self._knee_armed = [False, False]
                self._step_armed = [False, False]
            else:
                for side_idx, side_name, lift, other_lift, vy in (
                    (0, 'left', left_leg_lift, right_leg_lift, left_leg_vy),
                    (1, 'right', right_leg_lift, left_leg_lift, right_leg_vy),
                ):
                    # High lift (>= 0.42) upgrades to heavy KNEE_STOMP and cancels light step.
                    if lift >= 0.42 and (lift - other_lift) >= 0.20:
                        self._knee_armed[side_idx] = True
                        self._step_armed[side_idx] = False
                    elif not self._knee_armed[side_idx] and 0.12 <= lift < 0.42 and (lift - other_lift) >= 0.08:
                        self._step_armed[side_idx] = True
                        self._step_peak_lift[side_idx] = max(self._step_peak_lift[side_idx], lift)

                    if self._knee_armed[side_idx]:
                        landed = lift <= 0.22 or vy <= -1.05
                        if landed and self._ready(f'knee_stomp_{side_name}', timestamp, 0.32):
                            self._knee_armed[side_idx] = False
                            self._step_armed[side_idx] = False
                            self._step_peak_lift[side_idx] = 0.0
                            tag = 'RIGHT KNEE KICK!' if side_name == 'right' else 'LEFT KNEE SNARE!'
                            self._fire(events, f'knee_stomp_{side_name}', timestamp, MoveEvent('KNEE_STOMP', side_name, 0.92), tag, 0.55)
                    elif self._step_armed[side_idx]:
                        pk = self._step_peak_lift[side_idx]
                        step_landed = lift <= max(0.08, 0.45 * pk) or vy <= -0.45
                        if step_landed and self._ready(f'foot_step_{side_name}', timestamp, 0.26):
                            self._step_armed[side_idx] = False
                            self._step_peak_lift[side_idx] = 0.0
                            step_str = clamp(0.45 + 1.15 * pk, 0.45, 0.78)
                            tag = 'RIGHT STEP GROOVE' if side_name == 'right' else 'LEFT STEP GROOVE'
                            self._fire(events, f'foot_step_{side_name}', timestamp, MoveEvent('FOOT_STEP', side_name, step_str), tag, 0.35)
                    elif lift < 0.06:
                        self._knee_armed[side_idx] = False
                        self._step_armed[side_idx] = False
                        self._step_peak_lift[side_idx] = 0.0
        else:
            self._knee_armed = [False, False]
            self._step_armed = [False, False]
            self._step_peak_lift = [0.0, 0.0]

        # --- 10) LEG MOVE 2 — SQUAT_DIVE (Deep crouch sub-dive) + SQUAT_RISE (Spring-up pop!) ---
        squat_depth = clamp((0.50 - body_level) / 0.28)
        if squat_depth >= 0.42:
            self._squat_held += clamped_dt
            if self._squat_held >= 0.14:
                self._squat_armed = True
        else:
            if self._squat_armed and (body_level >= 0.56 or hip_bounce_vel >= 0.95) and self._ready('squat_rise', timestamp, 0.80):
                self._squat_armed = False
                self._squat_held = 0.0
                self._fire(events, 'squat_rise', timestamp, MoveEvent('SQUAT_RISE', 'both', 0.95), 'SQUAT POP!', 0.85)
            elif squat_depth < 0.15:
                self._squat_held = 0.0
                if body_level >= 0.54:
                    self._squat_armed = False

        # --- 11) LEG MOVE 3 — JUMP_HIT (Explosive upward hop above standing baseline!) ---
        if hip_bounce_vel >= 1.55 and body_level >= 0.76 and not self._squat_armed and self._ready('jump_hit', timestamp, 0.85):
            strength = clamp((hip_bounce_vel - 1.35) / 2.2, 0.70, 1.0)
            self._knee_armed = [False, False]
            self._fire(events, 'jump_hit', timestamp, MoveEvent('JUMP_HIT', 'both', strength), 'JUMP AIR HIT!', 0.85)

        # Arbitrate Tier 1 + Tier 2 candidates together (dual-channel priority, spotlight lock & intent shield).
        combined = events + list(extra_candidates)
        accepted_events = self.arbiter.arbitrate(
            combined,
            timestamp=timestamp,
            glitch_joints=glitch_joints,
            soundwave_rise=soundwave_rise,
            arms_crossed=self._arms_crossed,
            tier2_confidence=tier2_confidence,
        )

        active_label = self._last_label if timestamp < self._last_label_until else (
            'X-CUT BREAKDOWN' if self._arms_crossed else (
                f'SOUNDWAVE {int(soundwave_rise * 100)}%' if soundwave_rise >= 0.20 else (
                    f'SQUAT DIVE {int(squat_depth * 100)}%' if squat_depth >= 0.42 else (
                        'FREEZE ECHO' if self._body_frozen else (
                            'GROOVE LOCKED' if side_step_groove else ''
                        )
                    )
                )
            )
        )

        return MoveSnapshot(
            events=tuple(accepted_events),
            soundwave_rise=soundwave_rise,
            arms_crossed=self._arms_crossed,
            body_frozen=self._body_frozen,
            side_step_groove=side_step_groove,
            squat_depth=squat_depth,
            active_label=active_label,
        )

    @staticmethod
    def _lag_corr(a: list[float], b: list[float], lag: int) -> float:
        """Normalized cross-correlation where `b` trails `a` by `lag` frames (>0)."""
        n = len(a) - lag
        if n < 6:
            return 0.0
        sa = a[:n]
        sb = b[lag:]
        ma = sum(sa) / n
        mb = sum(sb) / n
        ca = [x - ma for x in sa]
        cb = [y - mb for y in sb]
        va = sum(x * x for x in ca)
        vb = sum(y * y for y in cb)
        if va < 1e-6 or vb < 1e-6:
            return 0.0
        return sum(x * y for x, y in zip(ca, cb)) / math.sqrt(va * vb)

    @classmethod
    def _detect_arm_wave(cls, window: list[dict]) -> str | None:
        """Detect a traveling wave across L_wrist -> L_elbow -> R_elbow -> R_wrist (or reverse).

        Uses two complementary paths:
          Path A: Direct monotonic peak propagation (single clean wave pass).
          Path B: Positive phase-lag cross-correlation across adjacent joints (continuous fluid waving!).
        """
        indices = (0, 1, 4, 5)  # 0=L_wrist, 1=L_elbow, 4=R_elbow, 5=R_wrist
        series = [[s['chain'][j] for s in window] for j in indices]
        spans = [max(vals) - min(vals) for vals in series]

        # Reject static or tiny jitter; both wrists & elbows must actually oscillate vertically.
        if min(spans[0], spans[3]) < 0.11 or min(spans[1], spans[2]) < 0.07:
            return None

        # Path A: Clean single traveling crest across the 4-joint chain.
        peak_times = [window[max(range(len(vals)), key=lambda i: vals[i])]['t'] for vals in series]
        lr_deltas = [peak_times[i + 1] - peak_times[i] for i in range(3)]
        if all(0.02 <= d <= 0.45 for d in lr_deltas) and 0.14 <= (peak_times[-1] - peak_times[0]) <= 1.15:
            return 'lr'
        rl_deltas = [peak_times[i] - peak_times[i + 1] for i in range(3)]
        if all(0.02 <= d <= 0.45 for d in rl_deltas) and 0.14 <= (peak_times[0] - peak_times[-1]) <= 1.15:
            return 'rl'

        # Path B: Continuous phase-lagged cross-correlation (works even during multi-cycle fluid waving!).
        # Compute vertical velocity series to remove slow DC arm-height drift.
        n_frames = len(window)
        if n_frames < 12:
            return None
        vels = [
            [
                (vals[i] - vals[i - 1]) / max(0.01, window[i]['t'] - window[i - 1]['t'])
                for i in range(1, n_frames)
            ]
            for vals in series
        ]

        max_lag = min(7, len(vels[0]) // 3)
        if max_lag < 2:
            return None

        def chain_score(order: tuple[int, int, int, int]) -> float:
            # Check positive lag correlation across the 3 adjacent hops + end-to-end wrist lag.
            pair_scores = []
            for a_i, b_i in ((order[0], order[1]), (order[1], order[2]), (order[2], order[3])):
                best_pos = max(cls._lag_corr(vels[a_i], vels[b_i], lag) for lag in range(1, max_lag + 1))
                zero_lag = cls._lag_corr(vels[a_i], vels[b_i], 0)
                # Require genuinely lagged propagation (not in-phase jumping/bouncing!).
                if best_pos < 0.48 or best_pos < zero_lag + 0.04:
                    return 0.0
                pair_scores.append(best_pos)
            # Also check overall wrist-to-wrist lag (must NOT be in-phase simultaneous raising!).
            wr_zero = cls._lag_corr(vels[order[0]], vels[order[3]], 0)
            wr_lag = max(cls._lag_corr(vels[order[0]], vels[order[3]], lag) for lag in range(2, min(12, len(vels[0]) // 2)))
            if wr_lag < 0.45 or wr_lag < wr_zero + 0.12:
                return 0.0
            return sum(pair_scores) / 3.0 + 0.35 * wr_lag

        lr_score = chain_score((0, 1, 2, 3))
        rl_score = chain_score((3, 2, 1, 0))
        if lr_score >= 0.72 and lr_score > rl_score + 0.06:
            return 'lr'
        if rl_score >= 0.72 and rl_score > lr_score + 0.06:
            return 'rl'
        return None
