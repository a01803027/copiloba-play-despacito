"""Dynamic Backing Groove + 11-Move Full-Body Studio Translator (Pose-Only, Zero Fists).

Every recognized body & leg move maps cleanly to a genre-adapted studio musical signature:
  - HARD_PUNCH (R / L)   -> Octave Power Stab + Snare Crack (R) / Deep Sub-Bass Boom + Kick (L)
  - DOUBLE_ARM_RISE      -> C++ Sweeping Soundwave Riser (MIDI 120) + Ascending Shimmer Swell
  - OVERHEAD_DROP        -> Beat Drop (Kick + Crash + Deep Sub + Full Strummed Chord)
  - ARMS_CROSS_CUT       -> Clean DJ Breakdown Cut (drums & bass cut, chords sweep warm)
  - OPEN_BURST           -> Euphoric 3-Note Harmonic Fanfare + Chorus Bloom
  - ARM_WAVE (L->R/R->L) -> 6-Note Liquid Stereo-Panned Arpeggio Ripple
  - BODY_FREEZE          -> Dramatic Echo Tail Freeze (drums cut, dotted-8th delay rings out)
  - SIDE_STEP_GROOVE     -> Syncopated Groove Pocket & Tight BPM Lock
  - KNEE_STOMP (R / L)   -> Deep Kick + Sub Root (R) / Genre Snare + Bass 5th (L)
  - SQUAT_DIVE / RISE    -> Underwater Low-Pass Sub-Dive -> Spring-Up Filter Snap-Open Pop
  - JUMP_HIT             -> Uplifting Air Crash + Kick + High Sparkle Chord
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from .arrangement import PhraseSequencer, SectionController, strum_chord_schedule
from .features import Features, clamp
from .procedural import degree_to_midi, quantize_gesture_frame, voice_lead
from .styles import REGISTRY, STYLE_ORDER, STYLES, StyleBlueprint, get_style

RATE = 48000
RISER_MIDI = 120  # Dedicated studio Soundwave Riser voice in C++20 DSP (limb 1)
SCALE = (0, 2, 3, 5, 7, 8, 10)
PROGRESSION = (0, 5, 3, 4)


@dataclass(frozen=True)
class Event:
    limb: int
    kind: int
    note: int
    value: float
    frame: int


def degree_note(degree, base=48, scale=SCALE):
    return degree_to_midi(degree, base, scale)


class Composer:
    preset = 'Dream Ambient'

    def __init__(self, seed=7, genre='ambient', tempo_lock=False):
        self.random = random.Random(seed)
        self.profile: StyleBlueprint = get_style(genre)
        self.genre_key = self.profile.key
        self.preset = self.profile.label
        self.tempo_lock = tempo_lock
        self.bpm_synced = False
        self.bpm_sync_source = 'none'

        self.sections = SectionController()
        self.section = 'INTRO'
        self.bar_in_cycle = 0
        self.head_harmony_active = False
        self.shoulder_groove_active = False
        self.beat_active = True
        self.dj_choke_active = False
        self._register_zone = 0
        self._riser_playing = False
        self._wave_dir = 0          # +1 L->R, -1 R->L
        self._wave_start = -1e9

        self.prev_chord_notes: tuple[int, ...] = ()
        self.next_frame = None
        self.tick = 0
        self.context = Features()
        self.fast_context = Features()
        self.window = deque(maxlen=90)
        self.fast_window = deque(maxlen=12)
        self.medium_at = 0.0
        self.bpm = self.profile.bpm_default
        self.chord_degree = self.profile.progression[0].degree
        self.current_chord = self.profile.progression[0]
        self.current_chord_name = self.current_chord.name
        self.phrase_number = None

        self.strike_punch = 0.0
        self.strike_punch_at = -1e9
        self._last_seconds = 0.0
        self._peak_bpm_conf = 0.0
        self._peak_bpm = self.profile.bpm_default
        self._peak_bpm_source = 'none'
        self._bpm_sync_until = -1e9
        self.presets_dirty = True
        self.skipped_ticks = 0

    def set_genre(self, genre_name: str):
        profile = get_style(genre_name)
        if profile.key == self.genre_key:
            return
        self.profile = profile
        self.genre_key = profile.key
        self.preset = profile.label
        if not self.tempo_lock:
            self.bpm = profile.bpm_default
        self.current_chord = profile.progression[0]
        self.chord_degree = self.current_chord.degree
        self.current_chord_name = self.current_chord.name
        self.prev_chord_notes = ()
        self.phrase_number = None
        self.presets_dirty = True

    def cycle_genre(self, step: int = 1) -> str:
        nxt = REGISTRY.next_key(self.genre_key, step)
        self.set_genre(nxt)
        return nxt

    def toggle_tempo_lock(self) -> bool:
        self.tempo_lock = not self.tempo_lock
        return self.tempo_lock

    def toggle_beat(self) -> bool:
        self.beat_active = not self.beat_active
        return self.beat_active

    def bpm_sync_range(self) -> tuple[float, float]:
        p = self.profile
        return (max(64.0, p.bpm_min - 8.0), min(180.0, p.bpm_max + 12.0))

    def controls(self, features: Features):
        """Continuous Studio Macro Automation sculpted by posture & continuous body move states."""
        p = self.profile
        if not features.tracked:
            return {
                'bpm': self.bpm,
                'energy': 0.0,
                'cutoff': [700.0, 900.0, 1000.0, 2500.0],
                'reverb': [0.45, 0.35, 0.03, 0.02],
                'gain': [0.0, 0.0, 0.0, 0.0],
                'bend': [0.0, 0.0, 0.0, 0.0],
                'pan': [0.0, 0.0, 0.0, 0.0],
                'delay_send': 0.2,
                'resonance': 0.8,
                'sidechain': 0.0,
            }

        in_buildup = self.section == 'BUILDUP'
        in_drop = self.section == 'DROP'
        choke = features.arms_crossed
        frozen = features.body_frozen
        rise = features.soundwave_rise
        squat = features.squat_depth

        punch = self.strike_punch * math.exp(-max(0.0, self._last_seconds - self.strike_punch_at) / 0.18)
        if punch < 0.01:
            punch = 0.0

        # Active stereo pan sweep during ARM_WAVE (0.75s sweep across L <-> R).
        wave_elapsed = self._last_seconds - self._wave_start
        wave_pan_override = None
        if self._wave_dir != 0 and 0.0 <= wave_elapsed <= 0.75:
            frac = wave_elapsed / 0.75
            wave_pan_override = self._wave_dir * (-0.85 + 1.70 * frac)

        # Gentle breathing ambient pad stays alive in stillness; full-body macros open the mix smoothly.
        pad_level = clamp(0.30 + 0.50 * features.energy + 0.12 * features.expansion + 0.12 * rise + (0.10 if in_drop else 0.0))
        lead_level = clamp(
            (0.14 + 0.56 * clamp(features.energy * 0.80 + 0.35 * features.right_height)) * (0.55 + 0.45 * features.expansion)
            + 0.24 * punch + 0.24 * rise
        )

        # Dynamic Backing Groove sculpted by core energy, legs/stance, and side-step dance bounce.
        groove_bonus = 0.18 if features.side_step_groove else 0.0
        groove_drive = clamp(
            0.55 * features.energy + 0.35 * (features.leg_energy if features.legs_visible else features.energy)
            + 0.20 * abs(features.hip_sway) + 0.15 * features.shoulder_bounce + groove_bonus
        )
        cut_mult = 0.08 if (choke or frozen) else (0.45 if in_buildup else (1.12 if in_drop else 1.0))
        beat_mult = 1.0 if self.beat_active else 0.08
        # During SQUAT_DIVE, bass rumbles warm & deep while upper frequencies dive underwater!
        bass_squat_boost = 1.0 + 0.24 * squat
        bass_level = clamp((0.14 + 0.64 * groove_drive + 0.14 * punch) * cut_mult * bass_squat_boost)
        drum_level = clamp((0.14 + 0.72 * groove_drive + 0.16 * punch) * (0.08 if (choke or frozen) else 1.0) * beat_mult)

        # Filter Macros: Left arm sculpts chords, Right arm sculpts lead, Squat dives underwater, Rise sweeps skyward!
        level_mod = clamp(0.50 + 0.75 * features.body_level, 0.50, 1.12)
        squat_filter = 1.0 - 0.68 * squat  # Deep squat -> warm underwater 32% filter dive!
        shrug_sweep = 1.0 + 0.55 * features.shoulder_lift

        pad_bright = clamp(0.32 + 0.38 * features.left_height + 0.22 * features.expansion + 0.12 * features.head_pitch)
        pad_cut = (p.cutoff_pad_range[0] + (p.cutoff_pad_range[1] - p.cutoff_pad_range[0]) * pad_bright) * level_mod * shrug_sweep * squat_filter + 1400.0 * punch + 2400.0 * rise
        if choke:
            pad_cut = min(pad_cut, 1300.0)
        pad_cut = clamp(pad_cut, 380.0, 10500.0)

        lead_bright = clamp(0.35 + 0.42 * features.right_height + 0.15 * features.head_pitch + 0.12 * features.expansion)
        lead_cut = (p.cutoff_lead_range[0] + (p.cutoff_lead_range[1] - p.cutoff_lead_range[0]) * lead_bright) * level_mod * shrug_sweep * squat_filter + 3200.0 * punch + 3200.0 * rise
        if choke:
            lead_cut = min(lead_cut, 1700.0)
        lead_cut = clamp(lead_cut, 650.0, 13200.0)

        rev_lift = clamp(0.52 * features.fluidity + 0.32 * features.torso_twist + 0.32 * rise + (0.35 if (choke or frozen) else 0.0))
        pad_rev = p.reverb_pad_range[0] + (p.reverb_pad_range[1] - p.reverb_pad_range[0]) * rev_lift
        lead_rev = p.reverb_lead_range[0] + (p.reverb_lead_range[1] - p.reverb_lead_range[0]) * rev_lift

        tilt_mag = abs(features.head_tilt)
        lead_pan = wave_pan_override if wave_pan_override is not None else clamp(
            0.60 * features.head_tilt + 0.30 * features.lateral_balance, -0.82, 0.82
        )
        pad_pan = clamp(-0.30 * features.torso_lean - 0.20 * features.head_tilt, -0.45, 0.45)
        delay_send = clamp(
            p.delay_send * (0.50 + 0.35 * features.fluidity) + 0.44 * tilt_mag + 0.22 * rise + (0.38 if frozen else (0.18 if choke else 0.0)),
            0.0, 0.86,
        )

        resonance = clamp(
            p.resonance + 0.90 * features.shoulder_lift + 0.40 * abs(features.torso_lean) + 0.55 * rise + 0.45 * squat + (0.40 if choke else 0.0),
            0.65, 2.55,
        )
        lead_bend = clamp((features.right_bend - 0.5) * 0.30 + (1.0 - features.symmetry) * 0.08, -0.35, 0.35)

        sc_drive = clamp(0.35 + 0.55 * features.energy + (0.20 if in_drop else 0.0))
        stance_boost = max(0.0, features.stance_width - 0.35) if features.legs_visible else 0.0
        bass_cut = clamp((620.0 + 1600.0 * clamp(groove_drive + 0.35 * stance_boost)) * (1.0 - 0.35 * squat), 300.0, 2900.0)
        drum_cut = clamp((8200.0 + 1800.0 * features.body_level + 800.0 * punch) * (1.0 - 0.55 * squat), 2200.0, 10800.0)

        return {
            'bpm': self.bpm,
            'energy': features.energy,
            'cutoff': [pad_cut, lead_cut, bass_cut, drum_cut],
            'reverb': [pad_rev, lead_rev, 0.03, 0.02],
            'gain': [pad_level * 0.96, lead_level * 0.96, bass_level * 0.72, drum_level * 0.82],
            'bend': [0.0, lead_bend, 0.0, 0.0],
            'pan': [pad_pan, lead_pan, 0.0, clamp(0.14 * features.shoulder_tilt, -0.25, 0.25)],
            'delay_send': delay_send,
            'resonance': resonance,
            'sidechain': clamp(p.sidechain * sc_drive * (0.0 if (choke or frozen or in_buildup or not self.beat_active) else 1.0), 0.0, 0.78),
        }

    @staticmethod
    def _average(window, tracked):
        keys = (
            'energy', 'fluidity', 'expansion', 'stillness',
            'velocity', 'acceleration', 'periodicity', 'symmetry',
            'left_height', 'right_height', 'left_speed', 'right_speed',
            'head_pitch', 'head_tilt', 'shoulder_lift', 'shoulder_tilt', 'shoulder_bounce',
            'torso_lean', 'torso_twist', 'hip_sway', 'body_level', 'lateral_balance',
            'leg_energy', 'stance_width',
        )
        values = {key: sum(getattr(f, key) for f in window) / len(window) for key in keys}
        return Features(**values, tracked=tracked)

    def _update_bpm(self, features: Features, seconds: float):
        p = self.profile
        if self.tempo_lock:
            self.bpm_synced = False
            self.bpm_sync_source = 'locked'
            self._peak_bpm_conf = 0.0
            return
        peak_conf = max(self._peak_bpm_conf, features.body_bpm_confidence if features.tracked else 0.0)
        peak_bpm = self._peak_bpm if self._peak_bpm_conf >= features.body_bpm_confidence else features.body_bpm
        peak_src = self._peak_bpm_source if self._peak_bpm_conf >= features.body_bpm_confidence else features.body_bpm_source
        self._peak_bpm_conf = 0.0

        lo_bpm, hi_bpm = p.bpm_min, p.bpm_max
        max_step = 2.0
        if features.tracked and peak_conf >= 0.48:
            target = clamp(peak_bpm, lo_bpm, hi_bpm)
            delta = clamp(target - self.bpm, -max_step, max_step)
            self.bpm = clamp(self.bpm + delta, lo_bpm, hi_bpm)
            self.bpm_synced = True
            self.bpm_sync_source = peak_src or 'body'
            self._bpm_sync_until = seconds + 2.0
        elif features.tracked and seconds < self._bpm_sync_until:
            self.bpm_synced = True
        else:
            bpm_span = p.bpm_max - p.bpm_min
            target = p.bpm_min + bpm_span * clamp(0.50 * self.context.energy + 0.30 * self.context.periodicity + 0.20 * self.context.shoulder_bounce)
            delta = clamp(target - self.bpm, -max_step * 0.5, max_step * 0.5)
            self.bpm = clamp(self.bpm + delta, lo_bpm, hi_bpm)
            self.bpm_synced = False
            self.bpm_sync_source = 'auto'

    def _choose_chord(self, phrase_index: int):
        p = self.profile
        if self.context.energy < 0.14:
            return p.progression[0]
        if (self.context.energy > 0.72 and self.context.fluidity < 0.35) or self.context.torso_twist > 0.78:
            return p.tension_chord
        if self.context.head_pitch > 0.72 and self.context.expansion > 0.55 and phrase_index % 2 == 1:
            return p.bright_chord
        return p.progression[phrase_index % len(p.progression)]

    def _emit_drop_impact(self, q_frame: int, features: Features, events: list[Event]):
        p = self.profile
        if self._riser_playing:
            events.append(Event(1, 1, RISER_MIDI, 0.0, q_frame))
            self._riser_playing = False
        sub_root = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
        events.extend((
            Event(3, 0, 36, 0.96, q_frame),
            Event(3, 0, 49, 0.74, q_frame),
            Event(2, 0, sub_root, 0.90, q_frame),
            Event(2, 1, sub_root, 0.0, q_frame + int(RATE * 0.60)),
        ))
        drop_chord = voice_lead(
            self.prev_chord_notes, self.chord_degree, self.current_chord.voicing,
            p.base_midi, p.scale, left_height=max(0.65, features.left_height),
        )
        for note, vel, on_f, off_f in strum_chord_schedule(drop_chord, q_frame, p.strum_ms, int(RATE * 1.4), 0.76):
            events.extend((Event(0, 0, note, vel, on_f), Event(0, 1, note, 0.0, off_f)))

    def _translate_moves(self, features: Features, frame: int, seconds: float, interval: float, events: list[Event]) -> bool:
        """Translate all recognized full-body & leg MoveEvents into genre-authentic studio sounds."""
        p = self.profile
        hit_frame = frame + 384  # 8 ms immediate punch feel (zero late_commands!)
        drop_fired = False

        # Release soundwave riser cleanly if arms drop back down below chest.
        if self._riser_playing and features.soundwave_rise < 0.08:
            events.append(Event(1, 1, RISER_MIDI, 0.0, hit_frame))
            self._riser_playing = False

        for mv in features.moves:
            if mv.kind == 'HARD_PUNCH':
                self.strike_punch = max(self.strike_punch, mv.intensity)
                self.strike_punch_at = seconds
                if mv.side == 'right':
                    # Right Hard Punch: Explosive Octave Power Stab + Crisp Genre Clap/Snare Crack!
                    oct_deg = self._register_zone * len(p.scale)
                    n_root = max(58, min(84, degree_to_midi(self.chord_degree + 4 + oct_deg, p.base_midi, p.scale)))
                    n_oct = min(92, n_root + 12)
                    events.extend((
                        Event(1, 0, n_root, clamp(0.88 + 0.12 * mv.intensity), hit_frame),
                        Event(1, 1, n_root, 0.0, hit_frame + int(RATE * 0.34)),
                        Event(1, 0, n_oct, clamp(0.84 + 0.14 * mv.intensity), hit_frame),
                        Event(1, 1, n_oct, 0.0, hit_frame + int(RATE * 0.38)),
                        Event(3, 0, 38, clamp(0.82 + 0.16 * mv.intensity), hit_frame),
                    ))
                else:
                    # Left Hard Punch: Deep Sub-Bass 808 Boom + Heavy Club Kick Slam!
                    sub_note = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
                    chord_hit = max(52, min(78, degree_to_midi(self.chord_degree + 2, p.base_midi, p.scale)))
                    events.extend((
                        Event(3, 0, 36, clamp(0.90 + 0.10 * mv.intensity), hit_frame),
                        Event(2, 0, sub_note, clamp(0.90 + 0.10 * mv.intensity), hit_frame),
                        Event(2, 1, sub_note, 0.0, hit_frame + int(RATE * 0.45)),
                        Event(0, 0, chord_hit, clamp(0.76 + 0.18 * mv.intensity), hit_frame),
                        Event(0, 1, chord_hit, 0.0, hit_frame + int(RATE * 0.55)),
                    ))

            elif mv.kind == 'DOUBLE_ARM_RISE':
                # Launch Studio Sweeping Soundwave Riser (MIDI 120) + Ascending Shimmer Swell!
                self._riser_playing = True
                events.append(Event(1, 0, RISER_MIDI, clamp(0.76 + 0.22 * mv.intensity), hit_frame))
                step = int(interval * 0.5)
                for i, deg_off in enumerate((4, 6, 8)):
                    note = max(60, min(91, degree_to_midi(self.chord_degree + deg_off, p.base_midi, p.scale)))
                    on_f = hit_frame + i * step
                    events.extend((
                        Event(0, 0, note, clamp(0.58 + 0.10 * i), on_f),
                        Event(0, 1, note, 0.0, on_f + int(RATE * 0.90)),
                    ))

            elif mv.kind == 'OVERHEAD_DROP':
                q_drop = quantize_gesture_frame(frame, self.next_frame, interval)
                self.beat_active = True
                self.strike_punch = 1.0
                self.strike_punch_at = seconds
                self._emit_drop_impact(q_drop, features, events)
                drop_fired = True

            elif mv.kind == 'OPEN_BURST':
                q_hit = quantize_gesture_frame(frame, self.next_frame, interval)
                self.beat_active = True
                self.strike_punch = max(self.strike_punch, 0.90)
                self.strike_punch_at = seconds
                # Euphoric 3-note ascending harmonic fanfare + open hat splash!
                fanfare = tuple(
                    max(55, min(91, degree_to_midi(self.chord_degree + d, p.base_midi, p.scale)))
                    for d in (0, 4, 8)
                )
                for note, vel, on_f, off_f in strum_chord_schedule(fanfare, q_hit, 35.0, int(RATE * 1.1), 0.84):
                    events.extend((Event(0, 0, note, vel, on_f), Event(0, 1, note, 0.0, off_f)))
                events.append(Event(3, 0, 46, 0.72, q_hit))
                drop_fired = True

            elif mv.kind == 'ARM_WAVE':
                # 6-note Liquid Stereo-Panned Arpeggio Ripple synchronized with the arm wave direction!
                self._wave_dir = 1 if mv.side == 'lr' else -1
                self._wave_start = seconds
                step = max(3600, int(interval * 0.45))  # silky ~80-110ms cascade
                degrees = (0, 2, 4, 6, 8, 11) if mv.side == 'lr' else (11, 8, 6, 4, 2, 0)
                for idx, deg_off in enumerate(degrees):
                    midi = max(55, min(91, degree_to_midi(self.chord_degree + deg_off, p.base_midi, p.scale)))
                    on_f = hit_frame + idx * step
                    vel = clamp(0.68 + 0.24 * math.sin(math.pi * (idx + 0.5) / len(degrees)))
                    events.extend((
                        Event(1, 0, midi, vel, on_f),
                        Event(1, 1, midi, 0.0, on_f + int(RATE * 0.34)),
                    ))

            elif mv.kind == 'BODY_FREEZE':
                # Let a single lush resonant bell chord ring into the dotted-8th echo tail!
                ring = max(60, min(88, degree_to_midi(self.chord_degree + 8, p.base_midi, p.scale)))
                events.extend((
                    Event(1, 0, ring, 0.78, hit_frame),
                    Event(1, 1, ring, 0.0, hit_frame + int(RATE * 0.55)),
                ))

            elif mv.kind == 'KNEE_STOMP':
                self.strike_punch = max(self.strike_punch, 0.80)
                self.strike_punch_at = seconds
                if mv.side == 'right':
                    sub_root = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
                    events.extend((
                        Event(3, 0, 36, 0.96, hit_frame),
                        Event(2, 0, sub_root, 0.92, hit_frame),
                        Event(2, 1, sub_root, 0.0, hit_frame + int(RATE * 0.36)),
                    ))
                else:
                    bass_fifth = max(31, min(57, degree_to_midi(self.chord_degree + 4, max(28, p.base_midi - 12), p.scale)))
                    events.extend((
                        Event(3, 0, 38, 0.90, hit_frame),
                        Event(2, 0, bass_fifth, 0.84, hit_frame),
                        Event(2, 1, bass_fifth, 0.0, hit_frame + int(RATE * 0.30)),
                    ))

            elif mv.kind == 'SQUAT_RISE':
                # Explosive filter snap-open + punchy Kick & Bright Chord Pop!
                self.strike_punch = 0.98
                self.strike_punch_at = seconds
                sub_root = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
                events.extend((
                    Event(3, 0, 36, 0.95, hit_frame),
                    Event(2, 0, sub_root, 0.90, hit_frame),
                    Event(2, 1, sub_root, 0.0, hit_frame + int(RATE * 0.40)),
                ))
                pop_chord = voice_lead(self.prev_chord_notes, self.chord_degree, self.current_chord.voicing, p.base_midi, p.scale, left_height=0.85)
                for note, vel, on_f, off_f in strum_chord_schedule(pop_chord, hit_frame, p.strum_ms, int(RATE * 0.90), 0.82):
                    events.extend((Event(0, 0, note, vel, on_f), Event(0, 1, note, 0.0, off_f)))
                drop_fired = True

            elif mv.kind == 'JUMP_HIT':
                # Uplifting Festival Air Crash + Kick + High Sparkle Octave!
                self.strike_punch = 0.95
                self.strike_punch_at = seconds
                hi_note = max(67, min(93, degree_to_midi(self.chord_degree + 11, p.base_midi, p.scale)))
                events.extend((
                    Event(3, 0, 36, 0.96, hit_frame),
                    Event(3, 0, 49, 0.72, hit_frame),
                    Event(1, 0, hi_note, 0.88, hit_frame),
                    Event(1, 1, hi_note, 0.0, hit_frame + int(RATE * 0.45)),
                ))
                drop_fired = True

        return drop_fired

    def update(self, features: Features, frame: int, seconds: float, lookahead: int = 2400):
        events: list[Event] = []
        p = self.profile
        self._last_seconds = seconds
        if self.presets_dirty:
            for limb, preset_id in enumerate(p.presets):
                events.append(Event(limb, 4, preset_id, 0.0, 0))
            self.presets_dirty = False

        if features.tracked and features.body_bpm_confidence >= self._peak_bpm_conf:
            self._peak_bpm_conf = features.body_bpm_confidence
            self._peak_bpm = features.body_bpm
            self._peak_bpm_source = features.body_bpm_source

        self.window.append(features)
        self.fast_window.append(features)
        self.fast_context = self._average(self.fast_window, features.tracked)

        if seconds >= self.medium_at:
            self.context = self._average(self.window, features.tracked)
            self._update_bpm(features, seconds)
            self.medium_at = seconds + 0.50

        if self.next_frame is None:
            self.next_frame = frame + 512

        interval = RATE * 15 / self.bpm
        if self.next_frame < frame:
            missed = int((frame - self.next_frame) // interval) + 1
            self.next_frame += missed * interval
            self.tick += missed
            self.skipped_ticks += missed

        if not features.tracked:
            self.head_harmony_active = False
            self.shoulder_groove_active = False
            self.dj_choke_active = False
            while self.next_frame <= frame + lookahead:
                self.next_frame += interval
                self.tick += 1
            return events

        live_energy = clamp(0.65 * self.fast_context.energy + 0.35 * self.context.energy)
        self.head_harmony_active = abs(features.head_tilt) > 0.28
        self.shoulder_groove_active = features.shoulder_bounce > 0.55 or abs(features.shoulder_tilt) > 0.45
        self.dj_choke_active = features.arms_crossed or features.body_frozen

        # Translate all 11 recognized full-body & leg moves cleanly!
        drop_fired = self._translate_moves(features, frame, seconds, interval, events)
        self.section = self.sections.update(live_energy, features, seconds, drop_trigger=drop_fired)

        if features.shoulder_drop:
            q_drop = quantize_gesture_frame(frame, self.next_frame, interval)
            self.beat_active = True
            self.strike_punch = 1.0
            self.strike_punch_at = seconds
            self._emit_drop_impact(q_drop, features, events)

        # --- SEQUENCER CLOCK (Dynamic Backing Groove — Clean, Musical, Cohesive) ---
        while self.next_frame <= frame + lookahead:
            target = round(self.next_frame)
            slot = self.tick % 16
            bar_idx = self.tick // 16
            self.bar_in_cycle = bar_idx % 4
            chord_period = 32 if p.key == 'ambient' else 16
            phrase_id = self.tick // chord_period

            if self.phrase_number != phrase_id:
                self.phrase_number = phrase_id
                self.current_chord = self._choose_chord(phrase_id)
                self.chord_degree = self.current_chord.degree
                self.current_chord_name = self.current_chord.name

            if slot % 4 == 0:
                self._register_zone = PhraseSequencer.register_shift(features.right_height, self._register_zone)

            swing_amt = int(interval * p.swing * (1.0 + 0.30 * abs(self.fast_context.hip_sway))) if (slot % 2 == 1) else 0
            paused = self.dj_choke_active

            # Layer 0 — Voice-led Warm Chords / Pads with human strum.
            comp_active = (slot == 0) if (self.section == 'INTRO' or chord_period == 32 or paused) else (slot in p.comping_steps)
            if comp_active and not features.body_frozen and (self.tick % chord_period == 0 or slot != 0):
                chord_notes = voice_lead(
                    self.prev_chord_notes, self.chord_degree, self.current_chord.voicing,
                    p.base_midi, p.scale, left_height=self.fast_context.left_height,
                )
                self.prev_chord_notes = chord_notes
                chord_dur = int(RATE * min(p.chord_duration_s, (interval * chord_period / RATE) * 0.94))
                chord_vel = 0.42 + 0.28 * live_energy
                for note, vel, on_f, off_f in strum_chord_schedule(chord_notes, target + swing_amt, p.strum_ms, chord_dur, chord_vel):
                    events.extend((Event(0, 0, note, vel, on_f), Event(0, 1, note, 0.0, off_f)))

            # Layer 1 — Stable-Register Cohesive 4-Bar Melody (pauses cleanly during Freeze!).
            melody_notes = [] if features.body_frozen else PhraseSequencer.notes_at_slot(
                style=p,
                bar_in_cycle=self.bar_in_cycle,
                slot=slot,
                chord_degree=self.chord_degree,
                right_height=features.right_height,
                section=self.section,
                head_tilt=features.head_tilt,
                register_zone=self._register_zone,
            )
            for midi, length_steps, vel_scale in melody_notes:
                vel = clamp((0.32 + 0.34 * max(live_energy, features.right_height * 0.7)) * vel_scale, 0.24, 0.72)
                dur_frames = int(max(RATE * 0.16, interval * length_steps * 0.90))
                events.extend((
                    Event(1, 0, midi, vel, target + swing_amt),
                    Event(1, 1, midi, 0.0, target + swing_amt + dur_frames),
                ))

            # Layer 2 — Tight Genre Bassline (SIDE_STEP_GROOVE unlocks rich syncopated octave groove!).
            if self.section in ('GROOVE', 'CHORUS', 'DROP') and not paused:
                for b_note in p.bass_pattern:
                    if b_note.step == slot:
                        bass_base = max(28, p.base_midi - 12)
                        oct_lift = 12 if ((self.shoulder_groove_active or features.side_step_groove) and slot >= 8 and slot % 4 != 0) else 0
                        midi = max(28, min(58, degree_to_midi(self.chord_degree + b_note.degree_offset, bass_base, p.scale) + oct_lift))
                        bass_e = clamp(0.60 * live_energy + 0.40 * (features.leg_energy if features.legs_visible else live_energy))
                        vel = clamp(0.42 + 0.38 * bass_e, 0.30, 0.88)
                        dur_frames = int(max(RATE * 0.12, interval * b_note.length_steps * 0.85))
                        events.extend((
                            Event(2, 0, midi, vel, target + swing_amt),
                            Event(2, 1, midi, 0.0, target + swing_amt + dur_frames),
                        ))

            # Layer 3 — Studio Genre Drum Kit (clean on-grid groove; pauses on X-Cut & Body Freeze!).
            drums = p.drums
            drums_playing = self.beat_active and not paused
            is_bar4_fill = (self.bar_in_cycle == 3 and slot >= 12 and self.section in ('CHORUS', 'DROP') and drums_playing)
            if self.section == 'BUILDUP' and self.beat_active and not paused:
                if slot % 2 == 0:
                    roll_vel = clamp(0.28 + 0.44 * features.shoulder_lift, 0.26, 0.74)
                    events.append(Event(3, 0, 38, roll_vel, target))
            elif is_bar4_fill:
                for f_step, f_note in drums.fill_bar4:
                    if f_step == slot:
                        events.append(Event(3, 0, f_note, clamp(0.48 + 0.34 * live_energy, 0.35, 0.88), target + swing_amt))
            elif self.section in ('GROOVE', 'CHORUS', 'DROP') and drums_playing:
                use_full = self.section in ('CHORUS', 'DROP') or features.side_step_groove
                kicks = drums.kick_full if use_full else drums.kick_groove
                hats = drums.hat_full if use_full else drums.hat_groove
                if slot in kicks:
                    events.append(Event(3, 0, 36, clamp(0.52 + 0.38 * live_energy, 0.38, 0.92), target))
                if use_full and slot in drums.snare:
                    events.append(Event(3, 0, 38, clamp(0.40 + 0.36 * live_energy, 0.32, 0.84), target + swing_amt))
                if slot in hats:
                    hat_note = 46 if (use_full and slot in drums.open_hat and features.expansion > 0.55) else 42
                    events.append(Event(3, 0, hat_note, clamp(0.22 + 0.24 * live_energy, 0.16, 0.62), target + swing_amt))
                if (self.shoulder_groove_active or features.side_step_groove) and use_full and slot in drums.ghost_snare:
                    events.append(Event(3, 0, 38, clamp(0.20 + 0.18 * features.shoulder_bounce, 0.18, 0.40), target + swing_amt))

            self.tick += 1
            self.next_frame += interval

        return events
