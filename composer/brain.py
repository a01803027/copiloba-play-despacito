"""Multi-genre generative dance composer with dual-window responsiveness."""
from collections import deque
from dataclasses import dataclass
import random
from .features import Features, clamp
from .genres import GENRES, GENRE_ORDER, GenreProfile, get_genre

RATE = 48000
SCALE = (0, 2, 3, 5, 7, 8, 10)
PROGRESSION = (0, 5, 3, 4)
MOTIF = (0, 2, 4, 2, 1, 2, 0, 4)


@dataclass(frozen=True)
class Event:
    limb: int
    kind: int
    note: int
    value: float
    frame: int


def degree_note(degree, base=48, scale=SCALE):
    n = len(scale)
    return base + 12 * (degree // n) + scale[degree % n]


class Composer:
    preset = 'Contemporary Ambient'

    def __init__(self, seed=7, genre='ambient'):
        self.random = random.Random(seed)
        self.profile: GenreProfile = get_genre(genre)
        self.genre_key = self.profile.key
        self.preset = self.profile.label
        self.next_frame = None
        self.tick = 0
        self.context = Features()
        self.fast_context = Features()
        self.window = deque(maxlen=90)
        self.fast_window = deque(maxlen=15)
        self.medium_at = 0.
        self.bpm = self.profile.bpm_min + (self.profile.bpm_max - self.profile.bpm_min) * 0.35
        self.chord_degree = self.profile.progression[0].degree
        self.current_chord = self.profile.progression[0]
        self.current_chord_name = self.current_chord.name
        self.phrase_number = None
        self.last_accent = -1e9
        self.presets_dirty = True
        self.skipped_ticks = 0

    def set_genre(self, genre_name: str):
        profile = get_genre(genre_name)
        if profile.key == self.genre_key:
            return
        self.profile = profile
        self.genre_key = profile.key
        self.preset = profile.label
        self.bpm = profile.bpm_min + (profile.bpm_max - profile.bpm_min) * 0.35
        self.current_chord = profile.progression[0]
        self.chord_degree = self.current_chord.degree
        self.current_chord_name = self.current_chord.name
        self.phrase_number = None
        self.presets_dirty = True

    def controls(self, features: Features):
        # Fast layer: immediate expressive shaping on every frame.
        p = self.profile
        level = (.18 + .68 * features.energy) if features.tracked else 0.
        avg_height = 0.5 * (features.left_height + features.right_height)
        pad_cut = p.cutoff_pad_range[0] + (p.cutoff_pad_range[1] - p.cutoff_pad_range[0]) * clamp(
            0.65 * features.expansion + 0.35 * avg_height
        )
        lead_cut = p.cutoff_lead_range[0] + (p.cutoff_lead_range[1] - p.cutoff_lead_range[0]) * clamp(
            0.55 * (1.0 - features.fluidity) + 0.45 * features.right_height
        )
        pad_rev = p.reverb_pad_range[0] + (p.reverb_pad_range[1] - p.reverb_pad_range[0]) * features.fluidity
        lead_rev = p.reverb_lead_range[0] + (p.reverb_lead_range[1] - p.reverb_lead_range[0]) * features.fluidity

        # Spatial stereo pan follows the dancer's lateral balance.
        lead_pan = clamp(features.lateral_balance * 0.75, -0.8, 0.8)
        pad_pan = clamp(-features.lateral_balance * 0.30, -0.4, 0.4)

        # Raising arms / body increases filter resonance slightly for musical build-ups.
        resonance = clamp(p.resonance + 0.6 * avg_height * features.energy, 0.6, 2.8)
        delay_send = clamp(p.delay_send * (0.6 + 0.6 * features.fluidity), 0.0, 0.85)

        return {
            'bpm': self.bpm,
            'energy': features.energy,
            'cutoff': [pad_cut, lead_cut, 2200.0 + 1800.0 * features.energy, 5500.0],
            'reverb': [pad_rev, lead_rev, 0.08, 0.14],
            'gain': [
                level * 0.92,
                level * (0.45 + 0.50 * features.expansion),
                level * 0.62,
                level * 0.42,
            ],
            'bend': [0., (1. - features.symmetry) * .20, 0., 0.],
            'pan': [pad_pan, lead_pan, 0.0, 0.18],
            'delay_send': delay_send,
            'resonance': resonance,
            'sidechain': p.sidechain * clamp(0.4 + 0.6 * features.energy),
        }

    @staticmethod
    def _average(window, tracked):
        keys = (
            'energy', 'fluidity', 'expansion', 'stillness',
            'velocity', 'acceleration', 'periodicity', 'symmetry',
            'left_height', 'right_height', 'body_level', 'lateral_balance',
        )
        values = {key: sum(getattr(f, key) for f in window) / len(window) for key in keys}
        return Features(**values, tracked=tracked)

    def update(self, features: Features, frame: int, seconds: float, lookahead: int = 2400):
        events = []
        p = self.profile
        if self.presets_dirty:
            for limb, preset_id in enumerate(p.presets):
                events.append(Event(limb, 4, preset_id, 0., 0))
            self.presets_dirty = False

        self.window.append(features)
        self.fast_window.append(features)
        self.fast_context = self._average(self.fast_window, features.tracked)

        if seconds >= self.medium_at:
            self.context = self._average(self.window, features.tracked)
            bpm_span = p.bpm_max - p.bpm_min
            target_bpm = p.bpm_min + bpm_span * clamp(0.8 * self.context.energy + 0.2 * self.context.periodicity)
            self.bpm = target_bpm
            self.medium_at = seconds + 0.75

        if self.next_frame is None:
            self.next_frame = frame + 512

        interval = RATE * 15 / self.bpm
        if self.next_frame < frame:
            missed = int((frame - self.next_frame) // interval) + 1
            self.next_frame += missed * interval
            self.tick += missed
            self.skipped_ticks += missed

        if not features.tracked:
            while self.next_frame <= frame + lookahead:
                self.next_frame += interval
                self.tick += 1
            return events

        # Blend fast and medium energy so rhythmic density responds quickly (~0.5s).
        live_energy = clamp(0.65 * self.fast_context.energy + 0.35 * self.context.energy)
        avg_height = 0.5 * (self.fast_context.left_height + self.fast_context.right_height)

        while self.next_frame <= frame + lookahead:
            target = round(self.next_frame)
            slot = self.tick % 16
            phrase = (self.tick // 32) % len(p.progression)

            if self.phrase_number != self.tick // 32:
                self.phrase_number = self.tick // 32
                if self.context.energy < .15:
                    self.current_chord = p.progression[0]
                elif self.context.energy > .70 and self.context.fluidity < .35:
                    self.current_chord = p.tension_chord
                else:
                    self.current_chord = p.progression[phrase]
                self.chord_degree = self.current_chord.degree
                self.current_chord_name = self.current_chord.name

            # Layer 0 (Chords / Pad / Warm Keys): every 2 bars (or 1 bar for high-energy House/LoFi).
            chord_period = 32 if p.key in ('ambient', 'cinematic') else 16
            if self.tick % chord_period == 0:
                chord_base = p.base_midi + (12 if (avg_height > 0.78 and p.key != 'ambient') else 0)
                chord_dur = int(RATE * min(p.chord_duration_s, (interval * chord_period / RATE) * 0.92))
                for offset in self.current_chord.voicing:
                    note = degree_note(self.chord_degree + offset, chord_base, p.scale)
                    note = max(36, min(84, note))
                    events.extend((
                        Event(0, 0, note, .48 + .15 * live_energy, target),
                        Event(0, 1, note, 0., target + chord_dur),
                    ))

            # Layer 1 (Lead / Melody / Arpeggio): responds to live_energy, height, and fluidity.
            stride = p.lead_stride_low if live_energy < .52 else p.lead_stride_high
            if slot % stride == 0 and (slot % 4 == 0 or self.random.random() < live_energy):
                motif_deg = p.motif[(self.tick // stride) % len(p.motif)]
                variation = self.random.choice((-1, 0, 1)) if self.fast_context.symmetry < .5 else 0
                octave_lift = 12 if (self.fast_context.expansion > .62 or self.fast_context.right_height > .72) else 0
                lead_base = p.base_midi + octave_lift
                note = degree_note(self.chord_degree + motif_deg + variation, lead_base, p.scale)
                note = max(48, min(96, note))
                velocity = .34 + .40 * live_energy
                swing_amt = int(interval * p.swing * (1.2 - 0.5 * self.fast_context.fluidity)) if (self.tick // stride) % 2 else 0
                lead_dur = int(RATE * p.lead_duration_s)
                events.extend((
                    Event(1, 0, note, velocity, target + swing_amt),
                    Event(1, 1, note, 0., target + swing_amt + lead_dur),
                ))

            # Layer 2 (Kick or Pulsing Sub-Bass depending on genre).
            if p.presets[2] == 11:
                # Sub-bass synth bus (e.g., Cinematic Pulse).
                if live_energy > p.bass_threshold and slot in p.bass_steps:
                    bass_note = degree_note(self.chord_degree, max(28, p.base_midi - 12), p.scale)
                    bass_dur = int(RATE * 0.24)
                    events.extend((
                        Event(2, 0, bass_note, .42 + .35 * live_energy, target),
                        Event(2, 1, bass_note, 0., target + bass_dur),
                    ))
            else:
                # Kick drum bus.
                if live_energy > p.kick_threshold and slot in p.kick_steps:
                    events.append(Event(2, 0, 36, .35 + .40 * live_energy, target))

            # Layer 3 (Hi-Hat / Snare / Percussive top layer).
            if live_energy > p.hat_threshold and slot in p.hat_steps:
                hat_note = 38 if p.presets[3] == 3 else 42
                hat_vel = .24 + .22 * live_energy
                events.append(Event(3, 0, hat_note, hat_vel, target))

            self.tick += 1
            self.next_frame += interval

        # Expressive accents from abrupt wrist acceleration (directional left/right height).
        impulse = max(features.left_impulse, features.right_impulse)
        if (features.acceleration > 35 or impulse > 0.45) and features.energy > .35 and seconds - self.last_accent > .28:
            accent_height = features.right_height if features.right_impulse >= features.left_impulse else features.left_height
            accent_base = p.base_midi + (24 if accent_height > 0.68 else 12)
            accent_note = degree_note(self.chord_degree + 4, accent_base, p.scale)
            accent_note = max(52, min(96, accent_note))
            events.extend((
                Event(1, 0, accent_note, .58, frame + 256),
                Event(1, 1, accent_note, 0., frame + 256 + int(RATE * 0.35)),
            ))
            self.last_accent = seconds

        return events
