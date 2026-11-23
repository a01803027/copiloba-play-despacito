"""Core musical math primitives: scale degree mapping, smooth voice leading, and beat-grid quantization."""
from __future__ import annotations

from typing import Sequence

SAMPLE_RATE = 48_000


def degree_to_midi(degree: int, base_midi: int, scale: Sequence[int]) -> int:
    n = len(scale)
    return base_midi + 12 * (degree // n) + scale[degree % n]


def voice_lead(
    prev_notes: Sequence[int] | None,
    chord_degree: int,
    voicing: Sequence[int],
    base_midi: int,
    scale: Sequence[int],
    left_height: float = 0.5,
    lo_midi: int = 36,
    hi_midi: int = 84,
) -> tuple[int, ...]:
    """Build a smoothly voice-led chord anchored at the root and shaped by left arm height."""
    if not voicing:
        return ()
    root = max(lo_midi, min(hi_midi, degree_to_midi(chord_degree + voicing[0], base_midi, scale)))
    center = root + 5 + int(round(max(0.0, min(1.0, left_height)) * 12))
    prev_center = sum(prev_notes[1:]) / max(1, len(prev_notes) - 1) if prev_notes and len(prev_notes) > 1 else center
    blend_center = 0.65 * center + 0.35 * prev_center

    result = [root]
    for offset in voicing[1:]:
        raw = degree_to_midi(chord_degree + offset, base_midi, scale)
        best = raw
        best_dist = abs(raw - blend_center)
        for shift in (-12, 12, 24):
            cand = raw + shift
            if root + 2 <= cand <= hi_midi:
                dist = abs(cand - blend_center)
                if dist < best_dist:
                    best = cand
                    best_dist = dist
        result.append(max(lo_midi, min(hi_midi, best)))
    return tuple(result)


def quantize_gesture_frame(
    current_frame: int,
    next_grid_frame: float | None,
    grid_period: float,
    early_window_ms: float = 65.0,
) -> int:
    """Snap a discrete gesture cleanly onto the upcoming musical subdivision if within `early_window_ms`,
    otherwise schedule safely 8 ms ahead (`+384` samples) for zero late commands."""
    immediate = current_frame + 384
    if next_grid_frame is None or grid_period <= 0:
        return immediate
    early_frames = int(SAMPLE_RATE * early_window_ms / 1000.0)
    delta = int(round(next_grid_frame)) - current_frame
    if 0 < delta <= early_frames:
        return int(round(next_grid_frame))
    return immediate
