"""Studio musical arrangement engine:
- Cohesive 4-bar Call & Response phrases (A -> A' -> B -> A'') in a STABLE singable register
- Hysteretic 3-zone octave/voicing expression (prevents frantic 3-octave pitch scrambling!)
- Clean musical sections (INTRO / GROOVE / CHORUS / BUILDUP / DROP) with DJ Choke & Hand Burst support
"""
from __future__ import annotations

from typing import Sequence
from .features import Features, clamp
from .procedural import degree_to_midi
from .styles import PhraseNote, StyleBlueprint

RATE = 48_000


class SectionController:
    """Manages musical sections smoothly with hysteresis, hand bursts, and shoulder Build-up/Drop."""

    def __init__(self):
        self.section = 'INTRO'
        self.drop_until = -1e9

    def reset(self):
        self.section = 'INTRO'
        self.drop_until = -1e9

    def update(self, live_energy: float, features: Features, seconds: float, drop_trigger: bool = False) -> str:
        # Deliberate shoulder shrug enters clean BUILD-UP tension.
        if features.shoulder_lift >= 0.65:
            self.section = 'BUILDUP'
            return self.section

        # Releasing DJ Breakdown toggle or shoulder drop triggers clean DROP!
        if drop_trigger or features.shoulder_drop:
            self.drop_until = seconds + 4.0
            self.section = 'DROP'
            return self.section

        if seconds < self.drop_until and live_energy >= 0.12:
            self.section = 'DROP'
            return self.section

        # Smooth hysteresis between INTRO, GROOVE, and CHORUS based on body energy.
        if self.section == 'INTRO':
            if live_energy >= 0.18:
                self.section = 'GROOVE'
        elif self.section == 'GROOVE':
            if live_energy < 0.12:
                self.section = 'INTRO'
            elif live_energy >= 0.48:
                self.section = 'CHORUS'
        else:
            if live_energy < 0.12:
                self.section = 'INTRO'
            elif live_energy < 0.36:
                self.section = 'GROOVE'
            else:
                self.section = 'CHORUS'
        return self.section


class PhraseSequencer:
    """Executes memorable A - A' - B - A'' phrases in a cohesive, singable register."""

    @staticmethod
    def active_motif(style: StyleBlueprint, bar_in_cycle: int) -> tuple[PhraseNote, ...]:
        return style.motif_b if bar_in_cycle == 2 else style.motif_a

    @staticmethod
    def register_shift(right_height: float, prev_zone: int = 0) -> int:
        """Hysteretic 3-zone musical register (-1 = low octave, 0 = core register, +1 = high octave).

        Avoids frantic frame-by-frame pitch teleportation while keeping right-arm height
        cleanly musical and expressive!
        """
        h = clamp(right_height)
        if prev_zone == -1:
            return 0 if h > 0.32 else -1
        if prev_zone == 1:
            return 0 if h < 0.68 else 1
        if h < 0.24:
            return -1
        if h > 0.76:
            return 1
        return 0

    @staticmethod
    def notes_at_slot(
        style: StyleBlueprint,
        bar_in_cycle: int,
        slot: int,
        chord_degree: int,
        right_height: float,
        section: str,
        head_tilt: float,
        register_zone: int | None = None,
    ) -> list[tuple[int, int, float]]:
        """Return list of (midi_note, length_steps, velocity_scale) for this 16th-note slot."""
        if section == 'INTRO' and right_height < 0.20:
            return []

        motif = PhraseSequencer.active_motif(style, bar_in_cycle)
        matching = [n for n in motif if n.step == slot]
        if not matching:
            return []

        if section == 'INTRO' and slot >= 8:
            return []

        zone = PhraseSequencer.register_shift(right_height) if register_zone is None else register_zone
        # Shift cleanly by full musical octaves (+/- scale length = +/- 12 semitones) so the motif
        # ALWAYS sounds intentional and singable instead of scrambling harmony!
        octave_deg = zone * len(style.scale)
        # Subtle expressive velocity lift with continuous right-arm height.
        expr_vel = 0.76 + 0.24 * clamp(right_height)

        out: list[tuple[int, int, float]] = []
        for note_spec in matching:
            deg = chord_degree + note_spec.degree_offset + octave_deg
            midi = max(52, min(88, degree_to_midi(deg, style.base_midi, style.scale)))
            vel_scale = (0.70 if section == 'INTRO' else (0.96 if section in ('CHORUS', 'DROP') else 0.85)) * expr_vel
            out.append((midi, note_spec.length_steps, vel_scale))

            # Head tilt adds a lush parallel harmony voice (+3rd right, +5th left).
            if abs(head_tilt) > 0.28:
                harm_interval = 2 if head_tilt > 0 else 4
                harm_midi = max(55, min(91, degree_to_midi(deg + harm_interval, style.base_midi, style.scale)))
                if harm_midi != midi:
                    out.append((harm_midi, note_spec.length_steps, vel_scale * 0.75))
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
        v_mod = 1.0 if idx in (0, len(notes) - 1) else 0.90
        scheduled.append((note, clamp(velocity * v_mod, 0.15, 0.92), on_frame, on_frame + duration_frames))
    return scheduled
