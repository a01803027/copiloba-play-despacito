"""Structured musical arrangement engine:
- 4-bar Call & Response phrase cycle (A -> A' -> B -> A'' + Bar-4 Fill)
- Musical sections (INTRO / GROOVE / CHORUS / BUILDUP / DROP) driven by energy & shoulders
- Human chord strumming (micro-arpeggiation)
- Diatonic phrase contour transposition following right arm height
"""
from typing import Sequence
from .features import Features, clamp
from .procedural import degree_to_midi
from .styles import PhraseNote, StyleBlueprint

RATE = 48_000


class SectionController:
    """Manages musical sections smoothly with hysteresis + shoulder Build-up/Drop."""

    def __init__(self):
        self.section = 'INTRO'
        self.drop_until = -1e9

    def reset(self):
        self.section = 'INTRO'
        self.drop_until = -1e9

    def update(self, live_energy: float, features: Features, seconds: float) -> str:
        # Shrugging shoulders up immediately engages musical BUILD-UP tension.
        if features.shoulder_lift >= 0.48:
            self.section = 'BUILDUP'
            return self.section

        # Releasing shrugged shoulders triggers an energetic DROP!
        if features.shoulder_drop:
            self.drop_until = seconds + 4.0
            self.section = 'DROP'
            return self.section

        if seconds < self.drop_until and live_energy >= 0.14:
            self.section = 'DROP'
            return self.section

        # Smooth hysteresis between INTRO, GROOVE, and CHORUS based on body energy.
        if self.section == 'INTRO':
            if live_energy >= 0.20:
                self.section = 'GROOVE'
        elif self.section == 'GROOVE':
            if live_energy < 0.14:
                self.section = 'INTRO'
            elif live_energy >= 0.50:
                self.section = 'CHORUS'
        else:  # CHORUS, BUILDUP, DROP
            if live_energy < 0.14:
                self.section = 'INTRO'
            elif live_energy < 0.40:
                self.section = 'GROOVE'
            else:
                self.section = 'CHORUS'
        return self.section


class PhraseSequencer:
    """Executes A - A' - B - A'' phrases coherently over 4-bar cycles."""

    @staticmethod
    def active_motif(style: StyleBlueprint, bar_in_cycle: int) -> tuple[PhraseNote, ...]:
        # Bar 0 = A, Bar 1 = A' (repetition builds hook!), Bar 2 = B (response), Bar 3 = A'' (resolution).
        return style.motif_b if bar_in_cycle == 2 else style.motif_a

    @staticmethod
    def notes_at_slot(
        style: StyleBlueprint,
        bar_in_cycle: int,
        slot: int,
        chord_degree: int,
        right_height: float,
        section: str,
        head_tilt: float,
    ) -> list[tuple[int, int, float]]:
        """Return list of (midi_note, length_steps, velocity_scale) for this 16th-note slot."""
        if section == 'INTRO' and right_height < 0.20:
            return []

        motif = PhraseSequencer.active_motif(style, bar_in_cycle)
        matching = [n for n in motif if n.step == slot]
        if not matching:
            return []

        # In INTRO, play only anchor notes (first half of motif) gently so pad breathes.
        if section == 'INTRO' and slot >= 8:
            return []

        # Continuous right arm height transposes the memorable phrase diatonically (-3..+7 degrees).
        # Raising the arm lifts the phrase note-by-note; lowering brings it smoothly down.
        reg_shift = int(round((clamp(right_height) - 0.30) * 10.0))
        # Bar 3 (A'') resolves slightly calmer on the final note.
        bar_nudge = 1 if (bar_in_cycle == 2 and section in ('CHORUS', 'DROP')) else 0

        out: list[tuple[int, int, float]] = []
        for note_spec in matching:
            deg = chord_degree + note_spec.degree_offset + reg_shift + bar_nudge
            midi = max(48, min(96, degree_to_midi(deg, style.base_midi, style.scale)))
            vel_scale = 0.72 if section == 'INTRO' else (1.0 if section in ('CHORUS', 'DROP') else 0.88)
            out.append((midi, note_spec.length_steps, vel_scale))

            # Unmistakable HEAD TILT effect: tilting head adds a lush parallel harmony voice!
            # Tilt right (> +0.28) -> +3rd above (+2 scale degrees); tilt left (< -0.28) -> +5th (+4 degrees).
            if abs(head_tilt) > 0.28:
                harm_interval = 2 if head_tilt > 0 else 4
                harm_midi = max(52, min(96, degree_to_midi(deg + harm_interval, style.base_midi, style.scale)))
                if harm_midi != midi:
                    out.append((harm_midi, note_spec.length_steps, vel_scale * 0.78))
        return out


def strum_chord_schedule(
    notes: Sequence[int],
    target_frame: int,
    strum_ms: float,
    duration_frames: int,
    velocity: float,
) -> list[tuple[int, float, int, int]]:
    """Return list of (midi_note, velocity, on_frame, off_frame) with human strum timing."""
    step_frames = int(RATE * max(0.0, strum_ms) / 1000.0)
    scheduled = []
    for idx, note in enumerate(notes):
        on_frame = target_frame + idx * step_frames
        # Slight natural accent on root and top melody tone of the voicing.
        v_mod = 1.0 if idx in (0, len(notes) - 1) else 0.90
        scheduled.append((note, clamp(velocity * v_mod, 0.15, 0.95), on_frame, on_frame + duration_frames))
    return scheduled
