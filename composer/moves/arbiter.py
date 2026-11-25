"""Central Move Priority & Mutual-Exclusion Arbiter.

Resolves simultaneous conflicting detections across Tier 1 (fast kinematics) and
Tier 2 (temporal ML / DTW) and suppresses ballistic false positives on frames
flagged by the Tier 0 Anti-Wiggle PoseSanitizer.
"""
from __future__ import annotations

from typing import Iterable


# Higher value wins when mutually exclusive moves fire in the same frame/window.
MOVE_PRIORITY: dict[str, int] = {
    'OVERHEAD_DROP': 95,
    'JUMP_HIT': 92,
    'DOUBLE_ARM_RISE': 88,
    'ARMS_CROSS_CUT': 85,
    'SQUAT_RISE': 82,
    'OPEN_BURST': 78,
    'ARM_WAVE': 76,
    'BODY_FREEZE': 74,
    'HARD_PUNCH': 68,
    'KNEE_STOMP': 66,
}


class MoveArbiter:
    """Filters, prioritizes, and de-conflicts raw candidate MoveEvents."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._last_overhead_t = -1e9
        self._last_jump_t = -1e9
        self._last_rise_t = -1e9
        self.suppressed_count = 0

    def arbitrate(
        self,
        candidates: Iterable,
        *,
        timestamp: float,
        glitch_joints: tuple[str, ...] = (),
        soundwave_rise: float = 0.0,
        arms_crossed: bool = False,
    ) -> list:
        """Return clean, conflict-free MoveEvents ordered by priority."""
        items = list(candidates)
        if not items:
            return []

        kinds_now = {ev.kind for ev in items}
        if 'OVERHEAD_DROP' in kinds_now:
            self._last_overhead_t = timestamp
        if 'JUMP_HIT' in kinds_now:
            self._last_jump_t = timestamp
        if 'DOUBLE_ARM_RISE' in kinds_now:
            self._last_rise_t = timestamp

        glitch_set = set(glitch_joints)
        left_arm_glitch = ('l_wrist' in glitch_set) or ('l_elbow' in glitch_set)
        right_arm_glitch = ('r_wrist' in glitch_set) or ('r_elbow' in glitch_set)

        # Simultaneous L+R KNEE_STOMP in the exact same frame is anatomically a hop/camera shift.
        stomp_sides = {ev.side for ev in items if ev.kind == 'KNEE_STOMP'}
        simultaneous_double_stomp = ('left' in stomp_sides and 'right' in stomp_sides)

        accepted = []
        seen_keys: set[tuple[str, str]] = set()

        for ev in sorted(items, key=lambda e: (-MOVE_PRIORITY.get(e.kind, 50), -e.intensity)):
            key = (ev.kind, ev.side)
            if key in seen_keys:
                continue

            # 1) Block ballistic arm events triggered by a clamped MediaPipe arm wiggle.
            if ev.kind == 'HARD_PUNCH':
                if (ev.side == 'left' and left_arm_glitch) or (ev.side == 'right' and right_arm_glitch):
                    self.suppressed_count += 1
                    continue
                # Suppress accidental punch firing during DOUBLE_ARM_RISE or OVERHEAD_DROP.
                if (
                    'DOUBLE_ARM_RISE' in kinds_now
                    or 'OVERHEAD_DROP' in kinds_now
                    or (timestamp - self._last_rise_t) < 0.25
                    or (timestamp - self._last_overhead_t) < 0.25
                ):
                    self.suppressed_count += 1
                    continue
                # Suppress punch while holding X-cross breakdown.
                if arms_crossed:
                    self.suppressed_count += 1
                    continue

            if ev.kind == 'OPEN_BURST' and (left_arm_glitch and right_arm_glitch):
                self.suppressed_count += 1
                continue

            # 2) Suppress impossible simultaneous double-leg stomps or stomps right on a jump.
            if ev.kind == 'KNEE_STOMP':
                if simultaneous_double_stomp or 'JUMP_HIT' in kinds_now or (timestamp - self._last_jump_t) < 0.35:
                    self.suppressed_count += 1
                    continue

            seen_keys.add(key)
            accepted.append(ev)

        return accepted
