"""Central Dual-Channel Move Priority, Temporal Spotlight & Intent-Shield Arbiter (v2).

Resolves simultaneous AND cross-frame move collisions across:
  - Upper-Body Channel: OVERHEAD_DROP > ARMS_CROSS_CUT > CUSTOM_TIER2 > DOUBLE_ARM_RISE
                        > ARM_WAVE > OPEN_BURST > BODY_FREEZE > HARD_PUNCH > LIGHT_PUNCH
  - Lower-Body Channel: JUMP_HIT > SQUAT_RISE > KNEE_STOMP > FOOT_STEP

Key musical guarantees:
  1. Independent Upper vs. Lower channels so footwork and upper-body gestures can groove together,
     while at most ONE move per channel can fire in a given frame.
  2. Rolling Temporal Spotlight Lock: major upper-body moves claim a ~0.55s–0.75s exclusivity window
     so intermediate/trailing arm motion cannot leak accidental punches or bursts.
  3. Tier 2 Intent Shield: when a custom Tier 2 (DTW / Keras TCN) dance move is actively forming
     (`tier2_confidence >= 0.48`), opportunistic Tier 1 sub-gestures (`LIGHT_PUNCH`, `HARD_PUNCH`,
     `OPEN_BURST`, `ARM_WAVE`) are suppressed before they can clutter the audio.
"""
from __future__ import annotations

from typing import Iterable

LOWER_BODY_MOVES: frozenset[str] = frozenset({
    'JUMP_HIT',
    'SQUAT_RISE',
    'KNEE_STOMP',
    'FOOT_STEP',
})

TIER1_KNOWN_MOVES: frozenset[str] = frozenset({
    'OVERHEAD_DROP',
    'ARMS_CROSS_CUT',
    'JUMP_HIT',
    'DOUBLE_ARM_RISE',
    'SQUAT_RISE',
    'ARM_WAVE',
    'OPEN_BURST',
    'BODY_FREEZE',
    'HARD_PUNCH',
    'KNEE_STOMP',
    'LIGHT_PUNCH',
    'FOOT_STEP',
})

CUSTOM_TIER2_PRIORITY = 90

# Higher value wins within its body channel (Upper vs. Lower).
MOVE_PRIORITY: dict[str, int] = {
    'OVERHEAD_DROP': 98,
    'ARMS_CROSS_CUT': 95,
    'JUMP_HIT': 92,
    # Custom Tier 2 moves default to CUSTOM_TIER2_PRIORITY = 90
    'DOUBLE_ARM_RISE': 86,
    'SQUAT_RISE': 84,
    'ARM_WAVE': 82,
    'OPEN_BURST': 78,
    'BODY_FREEZE': 75,
    'HARD_PUNCH': 68,
    'KNEE_STOMP': 66,
    'LIGHT_PUNCH': 56,
    'FOOT_STEP': 54,
}


def is_custom_move(kind: str) -> bool:
    """Return True when `kind` is a user-recorded Tier 2 (DTW / Keras TCN) dance move."""
    return kind not in TIER1_KNOWN_MOVES


def move_priority(kind: str) -> int:
    """Return priority score (custom Tier 2 moves outrank generic Tier 1 punches/waves/bursts)."""
    return MOVE_PRIORITY.get(kind, CUSTOM_TIER2_PRIORITY)


def move_channel(kind: str) -> str:
    """Return `'lower'` for leg/hip-elevation moves and `'upper'` for arm/torso/custom moves."""
    return 'lower' if kind in LOWER_BODY_MOVES else 'upper'


class MoveArbiter:
    """Dual-channel (Upper / Lower) priority, temporal spotlight, and anti-wiggle arbiter."""

    UPPER_SPOTLIGHT_SECONDS = 0.55
    CUSTOM_SPOTLIGHT_SECONDS = 0.75
    LOWER_SPOTLIGHT_SECONDS = 0.45
    TIER2_SHIELD_CONFIDENCE = 0.48

    def __init__(self):
        self.reset()

    def reset(self):
        self._upper_lock_until = -1e9
        self._upper_lock_prio = 0
        self._upper_lock_kind = ''
        self._lower_lock_until = -1e9
        self._lower_lock_prio = 0
        self._lower_lock_kind = ''
        self.suppressed_count = 0

    def arbitrate(
        self,
        candidates: Iterable,
        *,
        timestamp: float,
        glitch_joints: tuple[str, ...] = (),
        soundwave_rise: float = 0.0,
        arms_crossed: bool = False,
        tier2_confidence: float = 0.0,
    ) -> list:
        """Return conflict-free MoveEvents (at most 1 Upper-Body + 1 Lower-Body move per frame)."""
        # Release state-tied locks immediately once the dancer exits that continuous posture.
        if self._upper_lock_kind == 'ARMS_CROSS_CUT' and not arms_crossed:
            self._upper_lock_until = -1e9
            self._upper_lock_prio = 0
            self._upper_lock_kind = ''
        elif self._upper_lock_kind == 'DOUBLE_ARM_RISE' and soundwave_rise < 0.10:
            self._upper_lock_until = -1e9
            self._upper_lock_prio = 0
            self._upper_lock_kind = ''

        items = list(candidates)
        if not items:
            return []

        glitch_set = set(glitch_joints)
        left_arm_glitch = ('l_wrist' in glitch_set) or ('l_elbow' in glitch_set)
        right_arm_glitch = ('r_wrist' in glitch_set) or ('r_elbow' in glitch_set)

        # Simultaneous L+R leg hits in the exact same frame are anatomically a hop or camera shift.
        stomp_sides = {ev.side for ev in items if ev.kind == 'KNEE_STOMP'}
        step_sides = {ev.side for ev in items if ev.kind == 'FOOT_STEP'}
        double_stomp = ('left' in stomp_sides and 'right' in stomp_sides)
        double_step = ('left' in step_sides and 'right' in step_sides)

        # Tier 2 Intent Shield: active when a custom dance move is forming (>= 0.48) or firing now.
        has_custom_now = any(is_custom_move(ev.kind) for ev in items)
        tier2_shield = has_custom_now or (tier2_confidence >= self.TIER2_SHIELD_CONFIDENCE)

        accepted = []
        channel_taken = {'upper': False, 'lower': False}

        ordered = sorted(items, key=lambda e: (-move_priority(e.kind), -e.intensity))
        for ev in ordered:
            kind = ev.kind
            prio = move_priority(kind)
            chan = move_channel(kind)

            # 1) Anti-wiggle hardware/pose glitch guard.
            if kind in ('HARD_PUNCH', 'LIGHT_PUNCH'):
                if (ev.side == 'left' and left_arm_glitch) or (ev.side == 'right' and right_arm_glitch):
                    self.suppressed_count += 1
                    continue
                if arms_crossed:
                    self.suppressed_count += 1
                    continue
            if kind == 'OPEN_BURST' and (left_arm_glitch and right_arm_glitch):
                self.suppressed_count += 1
                continue

            # 2) Symmetric double-leg glitch guard.
            if (kind == 'KNEE_STOMP' and double_stomp) or (kind == 'FOOT_STEP' and double_step):
                self.suppressed_count += 1
                continue

            # 3) Tier 2 Intent Shield: suppress lower-priority Tier 1 upper-body sub-moves while custom move forms.
            if chan == 'upper' and tier2_shield and not is_custom_move(kind) and prio < CUSTOM_TIER2_PRIORITY:
                self.suppressed_count += 1
                continue

            # 4) Same-frame single-slot rule per body channel (highest priority wins the channel).
            if channel_taken[chan]:
                self.suppressed_count += 1
                continue

            # 5) Cross-frame rolling temporal spotlight lock (only strictly higher priority can interrupt).
            if chan == 'upper' and timestamp < self._upper_lock_until and prio <= self._upper_lock_prio:
                self.suppressed_count += 1
                continue
            if chan == 'lower' and timestamp < self._lower_lock_until and prio <= self._lower_lock_prio:
                self.suppressed_count += 1
                continue

            # Accept this move for its channel and update temporal spotlight locks if applicable.
            channel_taken[chan] = True
            accepted.append(ev)

            if chan == 'upper':
                if is_custom_move(kind):
                    self._upper_lock_until = timestamp + self.CUSTOM_SPOTLIGHT_SECONDS
                    self._upper_lock_prio = prio
                    self._upper_lock_kind = kind
                elif prio >= MOVE_PRIORITY['BODY_FREEZE']:
                    # Major spotlight moves (Drop, X-Cross, Arm Rise, Arm Wave, Open Burst, Freeze) lock out punches.
                    self._upper_lock_until = timestamp + self.UPPER_SPOTLIGHT_SECONDS
                    self._upper_lock_prio = prio
                    self._upper_lock_kind = kind
            else:
                if kind in ('JUMP_HIT', 'SQUAT_RISE'):
                    self._lower_lock_until = timestamp + self.LOWER_SPOTLIGHT_SECONDS
                    self._lower_lock_prio = prio
                    self._lower_lock_kind = kind
                elif kind == 'KNEE_STOMP':
                    # Prevent stomp landing rebound from immediately double-triggering FOOT_STEP.
                    self._lower_lock_until = timestamp + 0.30
                    self._lower_lock_prio = MOVE_PRIORITY['KNEE_STOMP'] - 1
                    self._lower_lock_kind = kind

        return accepted
