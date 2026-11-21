"""Procedural music generation primitives: Euclidean rhythms (Bjorklund),
contour-guided Markov melody, harmonic voice leading, and beat-grid quantization."""
import math
from typing import Sequence

SAMPLE_RATE = 48_000


def euclidean_rhythm(pulses: int, steps: int = 16, rotation: int = 0) -> tuple[bool, ...]:
    """Generate a Bjorklund Euclidean rhythm E(pulses, steps) rotated by `rotation`."""
    if steps <= 0:
        return ()
    k = max(0, min(steps, int(pulses)))
    if k == 0:
        return (False,) * steps
    if k == steps:
        return (True,) * steps

    # Bresenham/Bjorklund equivalent: evenly spaces k pulses across n steps.
    pattern = [((i * k) % steps) < k for i in range(steps)]
    # Ensure step 0 is always a pulse before rotation (standard Toussaint convention).
    if rotation:
        rot = rotation % steps
        pattern = pattern[-rot:] + pattern[:-rot]
    return tuple(pattern)


def degree_to_midi(degree: int, base_midi: int, scale: Sequence[int]) -> int:
    n = len(scale)
    return base_midi + 12 * (degree // n) + scale[degree % n]


class ContourMarkovMelody:
    """Constrained Markov melodic generator attracted to continuous arm height.

    - Raising the arm pulls the melodic contour upward step-by-step in scale.
    - Lowering the arm pulls the contour downward smoothly.
    - Slow arm motion yields stepwise legato intervals; fast motion allows wider arpeggio leaps.
    """

    def __init__(self, initial_degree: int = 4):
        self.current_degree = initial_degree

    def reset(self, degree: int = 4):
        self.current_degree = degree

    def next_degree(
        self,
        arm_height: float,
        arm_speed: float,
        chord_degree: int,
        chord_voicing: Sequence[int],
        scale_len: int,
        strong_beat: bool,
        rng,
    ) -> int:
        # Map arm height [0..1] across ~2 octaves of scale degrees [0..14].
        target = max(0.0, min(14.0, arm_height * 14.0))
        # Slow movement -> smooth nearby steps; fast movement -> expressive wider leaps.
        max_leap = 2 if arm_speed < 0.7 else (4 if arm_speed < 1.8 else 6)
        chord_classes = {(chord_degree + offset) % scale_len for offset in chord_voicing}

        lo = max(0, self.current_degree - max_leap)
        hi = min(14, self.current_degree + max_leap)
        # Always include the degree nearest to target so sustained raises/lowers never lag.
        target_deg = int(round(target))
        candidates = list(range(lo, hi + 1))
        if target_deg not in candidates:
            candidates.append(target_deg)

        weights: list[float] = []
        for cand in candidates:
            dist_target = abs(cand - target)
            dist_step = abs(cand - self.current_degree)
            # Strong contour gravity toward arm height.
            w = math.exp(-0.75 * dist_target)
            # Harmonic gravity: chord tones shine on strong beats.
            in_chord = (cand % scale_len) in chord_classes
            if strong_beat:
                w *= 2.8 if in_chord else 0.35
            else:
                w *= 1.4 if in_chord else 0.85
            # Mild Markov step preference (avoid repeating the exact same note too often when arm moves).
            if dist_step == 0 and dist_target > 0.7:
                w *= 0.35
            elif 1 <= dist_step <= 2:
                w *= 1.25
            weights.append(max(w, 1e-6))

        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        chosen = candidates[-1]
        for cand, w in zip(candidates, weights):
            acc += w
            if pick <= acc:
                chosen = cand
                break
        self.current_degree = chosen
        return chosen


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
    # Anchor the root cleanly; lift upper voices smoothly when left arm rises.
    root = max(lo_midi, min(hi_midi, degree_to_midi(chord_degree + voicing[0], base_midi, scale)))
    center = root + 5 + int(round(max(0.0, min(1.0, left_height)) * 12))
    prev_center = sum(prev_notes[1:]) / max(1, len(prev_notes) - 1) if prev_notes and len(prev_notes) > 1 else center
    blend_center = 0.65 * center + 0.35 * prev_center

    result = [root]
    for offset in voicing[1:]:
        raw = degree_to_midi(chord_degree + offset, base_midi, scale)
        # Choose octave inversion closest to blend_center while staying above root.
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
    early_window_ms: float = 45.0,
) -> int:
    """Snap a rapid body gesture to the upcoming 16th-note grid if within `early_window_ms`,
    otherwise trigger immediately (~2.7 ms ahead) for tight perceived responsiveness."""
    immediate = current_frame + 128
    if next_grid_frame is None or grid_period <= 0:
        return immediate
    early_frames = int(SAMPLE_RATE * early_window_ms / 1000.0)
    delta = int(round(next_grid_frame)) - current_frame
    if 0 < delta <= early_frames:
        return int(round(next_grid_frame))
    return immediate
