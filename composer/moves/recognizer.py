"""Real-time 11-move Full-Body Biomechanical & Dance Recognizer (Pose-Only, Zero Fists).

Recognizes 8 upper/full-body moves + 3 lower-body/leg moves over a rolling 1.5s 3D pose buffer:
  Upper / Full-Body:
    1. HARD_PUNCH (left / right)   — rapid elbow extension snap + high wrist thrust
    2. DOUBLE_ARM_RISE             — both arms rising overhead -> continuous Soundwave Riser
    3. OVERHEAD_DROP               — both arms slamming down from high -> Beat Drop
    4. ARMS_CROSS_CUT              — forearms crossed in an "X" at chest -> DJ Breakdown Cut
    5. OPEN_BURST                  — arms thrown wide open from chest -> Euphoric Bloom
    6. ARM_WAVE (lr / rl)          — traveling vertical wave across L/R wrist-elbow-shoulder
    7. BODY_FREEZE                 — sudden stop after active dancing -> Echo Tail Freeze
    8. SIDE_STEP_GROOVE            — periodic lateral hip sway + bounce -> Groove Pocket Lock
  Legs / Lower-Body (works with knees 25/26 OR ankles 27/28):
    9. KNEE_STOMP (left / right)   — high knee/foot lift (>=0.40) + crisp stomp landing
   10. SQUAT_DIVE / SQUAT_RISE     — deep crouch filter sub-dive -> spring-up filter pop
   11. JUMP_HIT                    — upward hop above standing baseline -> Air Crash & Kick
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from ..kinematics.filters import clamp


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
    """Multi-joint 3D spatiotemporal state machines over a rolling 1.5s pose buffer."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._buf: deque[dict] = deque(maxlen=60)  # ~2s at 30 FPS
        self._cooldowns: dict[str, float] = {}
        self._cross_held = 0.0
        self._arms_crossed = False
        self._freeze_held = 0.0
        self._body_frozen = False
        self._rise_active = False
        self._squat_armed = False
        self._squat_held = 0.0
        self._knee_armed = [False, False]
        self._last_label = ''
        self._last_label_until = -1e9

    def _ready(self, key: str, now: float, cd: float) -> bool:
        return (now - self._cooldowns.get(key, -1e9)) >= cd

    def _fire(self, events: list[MoveEvent], key: str, now: float, ev: MoveEvent, label: str, label_s: float = 0.70):
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
            'chain': chain_y,
            'sway': hip_sway,
            'lvl': body_level,
        }
        self._buf.append(sample)
        events: list[MoveEvent] = []

        # Helper to query window of samples within `sec` seconds ago.
        def recent(sec: float) -> list[dict]:
            cutoff = timestamp - sec
            return [s for s in self._buf if s['t'] >= cutoff]

        win_fast = recent(0.30)
        win_med = recent(0.55)
        win_long = recent(1.20)

        # --- 1) HARD_PUNCH (Left / Right): rapid elbow extension snap + explosive wrist velocity ---
        # Distinguishes a genuine punch (bent elbow snapping straight fast!) from wavy arm dancing.
        for side_idx, side_name, active, bend_k, spd_k, decel in (
            (0, 'left', arms_active[0], 'lbend', 'lspd', left_decel),
            (1, 'right', arms_active[1], 'rbend', 'rspd', right_decel),
        ):
            if active and len(win_fast) >= 3 and self._ready(f'punch_{side_name}', timestamp, 0.55):
                cur_bend = sample[bend_k]
                max_prev_bend = max(s[bend_k] for s in win_fast[:-1])
                bend_drop = max_prev_bend - cur_bend
                peak_spd = max(s[spd_k] for s in win_fast[-3:])
                # Elbow snapped straight by >= ~30 deg (0.17) into extended (<0.28) with explosive thrust!
                elbow_punch = bend_drop >= 0.17 and cur_bend <= 0.28 and peak_spd >= 4.2
                # Also allow explosive straight-arm boxing jab/hook with extreme thrust + crisp stop.
                power_jab = peak_spd >= 7.5 and decel <= -48.0
                if elbow_punch or power_jab:
                    strength = clamp((peak_spd - 3.2) / 4.0, 0.65, 1.0)
                    tag = 'RIGHT POWER PUNCH!' if side_name == 'right' else 'LEFT SUB BOOM!'
                    self._fire(events, f'punch_{side_name}', timestamp, MoveEvent('HARD_PUNCH', side_name, strength), tag)

        # --- 2) DOUBLE_ARM_RISE (Continuous Rising Soundwave!) ---
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
        if both_ok and not self._arms_crossed and len(win_med) >= 3 and self._ready('open_burst', timestamp, 1.00):
            min_exp = min(s['exp'] for s in win_med[:-1])
            dt_span = max(0.05, timestamp - win_med[0]['t'])
            exp_rate = (expansion - min_exp) / dt_span
            if min_exp <= 0.36 and expansion >= 0.68 and exp_rate >= 0.90:
                self._fire(events, 'open_burst', timestamp, MoveEvent('OPEN_BURST', 'both', clamp(expansion, 0.7, 1.0)), 'OPEN BURST!', 0.85)

        # --- 6) ARM_WAVE (Traveling vertical wave across 6-joint arm chain L->R or R->L!) ---
        if both_ok and len(win_long) >= 12 and self._ready('arm_wave', timestamp, 1.25):
            wave_dir = self._detect_arm_wave(win_long)
            if wave_dir is not None:
                tag = 'ARM WAVE L->R!' if wave_dir == 'lr' else 'ARM WAVE R->L!'
                self._fire(events, 'arm_wave', timestamp, MoveEvent('ARM_WAVE', wave_dir, 0.92), tag, 0.95)

        # --- 7) BODY_FREEZE (Sudden stop after active dancing -> Dramatic Echo Freeze!) ---
        peak_recent_energy = max((s['energy'] for s in win_long), default=0.0)
        if not self._body_frozen:
            if peak_recent_energy >= 0.44 and speed < 0.36:
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
        side_step_groove = (not self._body_frozen) and sway_span >= 0.48 and periodicity >= 0.44 and energy >= 0.18

        # --- 9) LEG MOVE 1 — KNEE_STOMP (High knee/foot lift >=0.40 + crisp stomp down!) ---
        if legs_visible:
            for side_idx, side_name, lift, vy in (
                (0, 'left', left_leg_lift, left_leg_vy),
                (1, 'right', right_leg_lift, right_leg_vy),
            ):
                if lift >= 0.38:
                    self._knee_armed[side_idx] = True
                if self._knee_armed[side_idx]:
                    landed = lift <= 0.24 or vy <= -1.25
                    if landed and self._ready(f'knee_stomp_{side_name}', timestamp, 0.45):
                        self._knee_armed[side_idx] = False
                        tag = 'RIGHT KNEE KICK!' if side_name == 'right' else 'LEFT KNEE SNARE!'
                        self._fire(events, f'knee_stomp_{side_name}', timestamp, MoveEvent('KNEE_STOMP', side_name, 0.92), tag)
                elif lift < 0.10:
                    self._knee_armed[side_idx] = False
        else:
            self._knee_armed = [False, False]

        # --- 10) LEG MOVE 2 — SQUAT_DIVE (Deep crouch sub-dive) + SQUAT_RISE (Spring-up pop!) ---
        squat_depth = clamp((0.50 - body_level) / 0.28)
        if squat_depth >= 0.42:
            self._squat_held += clamped_dt
            if self._squat_held >= 0.14:
                self._squat_armed = True
        else:
            if self._squat_armed and (body_level >= 0.56 or hip_bounce_vel >= 1.05) and self._ready('squat_rise', timestamp, 0.80):
                self._squat_armed = False
                self._squat_held = 0.0
                self._fire(events, 'squat_rise', timestamp, MoveEvent('SQUAT_RISE', 'both', 0.95), 'SQUAT POP!', 0.85)
            elif squat_depth < 0.15:
                self._squat_held = 0.0
                if body_level >= 0.54:
                    self._squat_armed = False

        # --- 11) LEG MOVE 3 — JUMP_HIT (Explosive upward hop above standing baseline!) ---
        if hip_bounce_vel >= 1.65 and body_level >= 0.78 and not self._squat_armed and self._ready('jump_hit', timestamp, 0.90):
            strength = clamp((hip_bounce_vel - 1.4) / 2.2, 0.70, 1.0)
            self._fire(events, 'jump_hit', timestamp, MoveEvent('JUMP_HIT', 'both', strength), 'JUMP AIR HIT!', 0.85)

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
            events=tuple(events),
            soundwave_rise=soundwave_rise,
            arms_crossed=self._arms_crossed,
            body_frozen=self._body_frozen,
            side_step_groove=side_step_groove,
            squat_depth=squat_depth,
            active_label=active_label,
        )

    @staticmethod
    def _detect_arm_wave(window: list[dict]) -> str | None:
        """Detect a traveling vertical peak across L_wrist -> L_elbow -> R_elbow -> R_wrist (or reverse)."""
        # Use 4 high-amplitude joints: 0=L_wrist, 1=L_elbow, 4=R_elbow, 5=R_wrist
        indices = (0, 1, 4, 5)
        peak_times: list[float] = []
        for j in indices:
            vals = [s['chain'][j] for s in window]
            lo, hi = min(vals), max(vals)
            if (hi - lo) < 0.14:
                return None
            # Find time of maximum height for this joint in the window.
            best_i = max(range(len(vals)), key=lambda i: vals[i])
            peak_times.append(window[best_i]['t'])

        # Check Left -> Right monotonic propagation (each segment peaks 25..420ms after previous).
        lr_deltas = [peak_times[i + 1] - peak_times[i] for i in range(3)]
        if all(0.025 <= d <= 0.42 for d in lr_deltas) and 0.18 <= (peak_times[-1] - peak_times[0]) <= 1.15:
            return 'lr'

        # Check Right -> Left monotonic propagation.
        rl_deltas = [peak_times[i] - peak_times[i + 1] for i in range(3)]
        if all(0.025 <= d <= 0.42 for d in rl_deltas) and 0.18 <= (peak_times[0] - peak_times[-1]) <= 1.15:
            return 'rl'
        return None
