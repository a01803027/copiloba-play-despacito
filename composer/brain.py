"""Coherent multi-style dance composer with phrase memory, sections, and full-body expression.

- 12+ modular JSON styles loaded from composer/styles/presets/
- Structured 4-bar Call & Response phrases (A -> A' -> B -> A'' + Bar-4 Fills)
- Unmistakable Head & Shoulder control:
  * Shoulder Shrug -> Build-up (rising filter + 16th roll), Release -> Drop!
  * Shoulder Bounce / Tilt -> Ghost snare notes & octave bass movement
  * Head Tilt -> Dual-voice melodic harmonization + audible ping-pong delay
  * Head Nod -> Quantized rhythmic percussion accent + BPM sync
- Continuous right arm contour transposition + left arm voice-led chord strumming
- Gentle breathing ambient pad in stillness + auto-synced BPM (toggleable with T)
"""
from collections import deque
from dataclasses import dataclass
import random
from .arrangement import PhraseSequencer, SectionController, strum_chord_schedule
from .features import Features, clamp
from .procedural import degree_to_midi, quantize_gesture_frame, voice_lead
from .styles import REGISTRY, STYLE_ORDER, STYLES, StyleBlueprint, get_style

RATE = 48000
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
    preset = 'Contemporary Ambient'

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

        self.prev_chord_notes: tuple[int, ...] = ()
        self.next_frame = None
        self.tick = 0
        self.context = Features()
        self.fast_context = Features()
        self.window = deque(maxlen=90)
        self.fast_window = deque(maxlen=12)
        self.medium_at = 0.
        self.bpm = self.profile.bpm_default
        self.chord_degree = self.profile.progression[0].degree
        self.current_chord = self.profile.progression[0]
        self.current_chord_name = self.current_chord.name
        self.phrase_number = None
        self.last_accent = -1e9
        self.last_left_strike = -1e9
        self.last_head_nod = -1e9
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

    def bpm_sync_range(self) -> tuple[float, float]:
        p = self.profile
        return (max(64.0, p.bpm_min - 8.0), min(180.0, p.bpm_max + 12.0))

    def controls(self, features: Features):
        p = self.profile
        if not features.tracked:
            return {
                'bpm': self.bpm,
                'energy': 0.,
                'cutoff': [600., 800., 1200., 2500.],
                'reverb': [0.4, 0.3, 0.05, 0.1],
                'gain': [0., 0., 0., 0.],
                'bend': [0., 0., 0., 0.],
                'pan': [0., 0., 0., 0.],
                'delay_send': 0.2,
                'resonance': 0.8,
                'sidechain': 0.0,
            }

        in_buildup = self.section == 'BUILDUP' or features.shoulder_lift >= 0.48
        in_drop = self.section == 'DROP'

        # Gentle breathing ambient pad stays alive softly in complete stillness;
        # other layers rise smoothly with energy and section.
        pad_level = clamp(0.24 + 0.62 * features.energy + (0.12 if in_drop else 0.0))
        lead_level = clamp(
            0.12 + 0.72 * clamp(features.energy * 1.10 + 0.25 * features.right_speed)
        ) * (0.50 + 0.50 * features.expansion)
        # During BUILDUP, slightly duck bass so the subsequent DROP hits hard!
        bass_mult = 0.45 if in_buildup else (1.12 if in_drop else 1.0)
        bass_level = clamp((0.06 + 0.78 * clamp(features.energy + 0.25 * abs(features.hip_sway) + 0.25 * features.shoulder_bounce)) * bass_mult)
        drum_level = clamp(0.06 + 0.80 * clamp(features.energy + 0.25 * features.shoulder_bounce + (0.25 if in_buildup else 0.0)))

        # Body vertical level (crouch -> warm sub-drop; tall -> open).
        level_mod = 0.58 + 0.58 * features.body_level
        # Shrugging shoulders up sweeps filters sharply upward for unmistakable Build-up tension!
        shrug_sweep = 1.0 + 0.85 * features.shoulder_lift

        pad_bright = clamp(0.42 * features.expansion + 0.38 * features.left_height + 0.20 * features.head_pitch)
        pad_cut = (p.cutoff_pad_range[0] + (p.cutoff_pad_range[1] - p.cutoff_pad_range[0]) * pad_bright) * level_mod * shrug_sweep
        pad_cut = clamp(pad_cut, 240.0, 14500.0)

        lead_bright = clamp(0.45 * features.right_height + 0.25 * features.head_pitch + 0.30 * (1.0 - features.fluidity))
        lead_cut = (p.cutoff_lead_range[0] + (p.cutoff_lead_range[1] - p.cutoff_lead_range[0]) * lead_bright) * level_mod * shrug_sweep
        lead_cut = clamp(lead_cut, 350.0, 15500.0)

        rev_lift = clamp(0.65 * features.fluidity + 0.35 * features.torso_twist)
        pad_rev = p.reverb_pad_range[0] + (p.reverb_pad_range[1] - p.reverb_pad_range[0]) * rev_lift
        lead_rev = p.reverb_lead_range[0] + (p.reverb_lead_range[1] - p.reverb_lead_range[0]) * rev_lift

        # Head tilt clearly steers stereo image AND dramatically boosts Ping-Pong Delay!
        tilt_mag = abs(features.head_tilt)
        lead_pan = clamp(0.65 * features.head_tilt + 0.35 * features.lateral_balance, -0.88, 0.88)
        pad_pan = clamp(-0.35 * features.torso_lean - 0.25 * features.head_tilt, -0.5, 0.5)
        delay_send = clamp(p.delay_send * (0.50 + 0.40 * features.fluidity) + 0.48 * tilt_mag, 0.0, 0.90)

        # Shoulder shrug + torso lean build dramatic filter resonance.
        resonance = clamp(p.resonance + 1.15 * features.shoulder_lift + 0.50 * abs(features.torso_lean), 0.65, 3.2)
        lead_bend = clamp((features.right_bend - 0.5) * 0.5 + (1.0 - features.symmetry) * 0.12, -0.5, 0.5)

        sc_drive = clamp(0.35 + 0.55 * features.energy + (0.20 if in_drop else 0.0))

        return {
            'bpm': self.bpm,
            'energy': features.energy,
            'cutoff': [
                pad_cut,
                lead_cut,
                clamp(440.0 + 1850.0 * clamp(features.energy + 0.3 * features.body_level), 220.0, 3800.0),
                clamp((3400.0 + 5500.0 * features.body_level) * (1.0 + 0.35 * features.shoulder_lift), 1800.0, 14000.0),
            ],
            'reverb': [pad_rev, lead_rev, 0.06, 0.14],
            'gain': [pad_level, lead_level, bass_level, drum_level],
            'bend': [0., lead_bend, 0., 0.],
            'pan': [pad_pan, lead_pan, 0.0, clamp(0.18 * features.shoulder_tilt, -0.30, 0.30)],
            'delay_send': delay_send,
            'resonance': resonance,
            'sidechain': clamp(p.sidechain * sc_drive * (0.2 if in_buildup else 1.0), 0.0, 0.82),
        }

    @staticmethod
    def _average(window, tracked):
        keys = (
            'energy', 'fluidity', 'expansion', 'stillness',
            'velocity', 'acceleration', 'periodicity', 'symmetry',
            'left_height', 'right_height', 'left_speed', 'right_speed',
            'head_pitch', 'head_tilt', 'shoulder_lift', 'shoulder_tilt', 'shoulder_bounce',
            'torso_lean', 'torso_twist', 'hip_sway', 'body_level', 'lateral_balance',
        )
        values = {key: sum(getattr(f, key) for f in window) / len(window) for key in keys}
        return Features(**values, tracked=tracked)

    def _update_bpm(self, features: Features, seconds: float):
        p = self.profile
        if self.tempo_lock:
            self.bpm_synced = False
            self.bpm_sync_source = 'locked'
            return
        lo_bpm, hi_bpm = self.bpm_sync_range()
        if features.tracked and features.body_bpm_confidence >= 0.42:
            target = clamp(features.body_bpm, lo_bpm, hi_bpm)
            self.bpm += 0.18 * (target - self.bpm)
            self.bpm_synced = True
            self.bpm_sync_source = features.body_bpm_source
        elif seconds >= self.medium_at:
            bpm_span = p.bpm_max - p.bpm_min
            target = p.bpm_min + bpm_span * clamp(0.55 * self.context.energy + 0.25 * self.context.periodicity + 0.20 * self.context.shoulder_bounce)
            self.bpm += 0.28 * (target - self.bpm)
            self.bpm_synced = False
            self.bpm_sync_source = 'auto'

    def _choose_chord(self, phrase_index: int):
        p = self.profile
        if self.context.energy < .14:
            return p.progression[0]
        if (self.context.energy > .70 and self.context.fluidity < .35) or self.context.torso_twist > .75:
            return p.tension_chord
        if self.context.head_pitch > .72 and self.context.expansion > .55 and phrase_index % 2 == 1:
            return p.bright_chord
        return p.progression[phrase_index % len(p.progression)]

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
            self._update_bpm(features, seconds)
            self.medium_at = seconds + 0.65
        elif features.body_bpm_confidence >= 0.42 and not self.tempo_lock:
            self._update_bpm(features, seconds)

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
            while self.next_frame <= frame + lookahead:
                self.next_frame += interval
                self.tick += 1
            return events

        live_energy = clamp(0.68 * self.fast_context.energy + 0.32 * self.context.energy)
        self.section = self.sections.update(live_energy, features, seconds)
        self.head_harmony_active = abs(features.head_tilt) > 0.28
        self.shoulder_groove_active = features.shoulder_bounce > 0.22 or abs(features.shoulder_tilt) > 0.30

        # Instant SHOULDER DROP impact (releasing a shoulder shrug -> big Drop hit!).
        if features.shoulder_drop:
            q_drop = quantize_gesture_frame(frame, self.next_frame, interval)
            sub_root = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
            events.extend((
                Event(3, 0, 36, 0.96, q_drop),
                Event(3, 0, 46, 0.68, q_drop),
                Event(2, 0, sub_root, 0.92, q_drop),
                Event(2, 1, sub_root, 0., q_drop + int(RATE * 0.65)),
            ))
            drop_chord = voice_lead(self.prev_chord_notes, self.chord_degree, self.current_chord.voicing, p.base_midi, p.scale, left_height=max(0.65, features.left_height))
            for note, vel, on_f, off_f in strum_chord_schedule(drop_chord, q_drop, p.strum_ms, int(RATE * 1.4), 0.72):
                events.extend((Event(0, 0, note, vel, on_f), Event(0, 1, note, 0., off_f)))

        # Instant HEAD NOD accent (quantized rhythmic hi-hat/rimshot hit!).
        if features.head_nod and (seconds - self.last_head_nod) >= 0.18:
            q_nod = quantize_gesture_frame(frame, self.next_frame, interval)
            nod_note = 46 if self.section in ('CHORUS', 'DROP') else 42
            events.append(Event(3, 0, nod_note, clamp(0.45 + 0.35 * live_energy, 0.35, 0.85), q_nod))
            self.last_head_nod = seconds

        while self.next_frame <= frame + lookahead:
            target = round(self.next_frame)
            slot = self.tick % 16
            bar_idx = self.tick // 16
            self.bar_in_cycle = bar_idx % 4
            # Ambient & Cinematic change chords every 2 bars (32 ticks); rhythmic styles change every bar (16 ticks).
            chord_period = 32 if p.key in ('ambient', 'cinematic') else 16
            phrase_id = self.tick // chord_period

            if self.phrase_number != phrase_id:
                self.phrase_number = phrase_id
                self.current_chord = self._choose_chord(phrase_id)
                self.chord_degree = self.current_chord.degree
                self.current_chord_name = self.current_chord.name

            swing_amt = int(interval * p.swing * (1.0 + 0.35 * abs(self.fast_context.hip_sway))) if (slot % 2 == 1) else 0

            # Layer 0 — Voice-led Chord Comping with human strum.
            comp_active = (slot == 0) if (self.section == 'INTRO' or chord_period == 32) else (slot in p.comping_steps)
            if comp_active and (self.tick % chord_period == 0 or slot != 0):
                chord_notes = voice_lead(
                    self.prev_chord_notes,
                    self.chord_degree,
                    self.current_chord.voicing,
                    p.base_midi,
                    p.scale,
                    left_height=self.fast_context.left_height,
                )
                self.prev_chord_notes = chord_notes
                chord_dur = int(RATE * min(p.chord_duration_s, (interval * chord_period / RATE) * 0.94))
                chord_vel = 0.36 + 0.28 * live_energy
                for note, vel, on_f, off_f in strum_chord_schedule(chord_notes, target + swing_amt, p.strum_ms, chord_dur, chord_vel):
                    events.extend((Event(0, 0, note, vel, on_f), Event(0, 1, note, 0., off_f)))

            # Layer 1 — Structured A - A' - B - A'' Melody guided by Right Arm height + Head Tilt harmony!
            melody_notes = PhraseSequencer.notes_at_slot(
                style=p,
                bar_in_cycle=self.bar_in_cycle,
                slot=slot,
                chord_degree=self.chord_degree,
                right_height=features.right_height,
                section=self.section,
                head_tilt=features.head_tilt,
            )
            for midi, length_steps, vel_scale in melody_notes:
                vel = clamp((0.36 + 0.42 * max(live_energy, clamp(features.right_speed / 2.2))) * vel_scale, 0.22, 0.95)
                dur_frames = int(max(RATE * 0.14, interval * length_steps * 0.88))
                events.extend((
                    Event(1, 0, midi, vel, target + swing_amt),
                    Event(1, 1, midi, 0., target + swing_amt + dur_frames),
                ))

            # Layer 2 — Authentic Style Bassline (anchored to the 1; shoulder bounce adds expressive octave lift!).
            if self.section in ('GROOVE', 'CHORUS', 'DROP'):
                for b_note in p.bass_pattern:
                    if b_note.step == slot:
                        bass_base = max(28, p.base_midi - 12)
                        oct_lift = 12 if (self.shoulder_groove_active and slot >= 8 and slot % 4 != 0) else 0
                        midi = max(28, min(60, degree_to_midi(self.chord_degree + b_note.degree_offset, bass_base, p.scale) + oct_lift))
                        vel = clamp(0.45 + 0.40 * live_energy, 0.30, 0.95)
                        dur_frames = int(max(RATE * 0.12, interval * b_note.length_steps * 0.85))
                        events.extend((
                            Event(2, 0, midi, vel, target + swing_amt),
                            Event(2, 1, midi, 0., target + swing_amt + dur_frames),
                        ))

            # Layer 3 — Polyphonic Drum Kit (Preset 12) anchored to bar + Bar-4 fills + Shrug Build-up roll!
            drums = p.drums
            is_bar4_fill = (self.bar_in_cycle == 3 and slot >= 12 and self.section in ('CHORUS', 'DROP'))
            if self.section == 'BUILDUP':
                # Shoulder shrug Build-up: rising 8th/16th snare + hat tension roll!
                roll_stride = 1 if features.shoulder_lift > 0.72 else 2
                if slot % roll_stride == 0:
                    roll_vel = clamp(0.32 + 0.55 * features.shoulder_lift, 0.30, 0.92)
                    events.append(Event(3, 0, 38, roll_vel, target))
                    events.append(Event(3, 0, 42, roll_vel * 0.65, target))
            elif is_bar4_fill:
                for f_step, f_note in drums.fill_bar4:
                    if f_step == slot:
                        events.append(Event(3, 0, f_note, clamp(0.50 + 0.38 * live_energy, 0.35, 0.95), target + swing_amt))
            elif self.section in ('GROOVE', 'CHORUS', 'DROP'):
                kicks = drums.kick_full if self.section in ('CHORUS', 'DROP') else drums.kick_groove
                hats = drums.hat_full if self.section in ('CHORUS', 'DROP') else drums.hat_groove
                if slot in kicks:
                    events.append(Event(3, 0, 36, clamp(0.48 + 0.42 * live_energy, 0.35, 0.98), target))
                if self.section in ('CHORUS', 'DROP') and slot in drums.snare:
                    events.append(Event(3, 0, 38, clamp(0.40 + 0.42 * live_energy, 0.32, 0.94), target + swing_amt))
                if slot in hats:
                    hat_note = 46 if (self.section in ('CHORUS', 'DROP') and slot in drums.open_hat and features.expansion > 0.55) else 42
                    events.append(Event(3, 0, hat_note, clamp(0.22 + 0.28 * live_energy, 0.16, 0.72), target + swing_amt))
                # Alternating shoulder groove injects crisp musical ghost snares!
                if self.shoulder_groove_active and slot in drums.ghost_snare:
                    events.append(Event(3, 0, 38, clamp(0.24 + 0.22 * features.shoulder_bounce, 0.20, 0.52), target + swing_amt))

            self.tick += 1
            self.next_frame += interval

        # --- Immediate Beat-Quantized Reactions to Fast Directional Arm Strikes ---
        if (features.right_strike != 0 or (features.acceleration > 28 and features.energy > 0.35)) and seconds - self.last_accent > 0.20:
            q_frame = quantize_gesture_frame(frame, self.next_frame, interval)
            strength = max(features.right_strike_strength, 0.55)
            base_deg = int(round(features.right_height * 12.0))
            if features.right_strike >= 0:
                n1 = max(55, min(96, degree_to_midi(self.chord_degree + max(4, base_deg), p.base_midi, p.scale)))
                n2 = max(58, min(96, degree_to_midi(self.chord_degree + max(4, base_deg) + 2, p.base_midi, p.scale)))
                step = int(interval * 0.5)
                events.extend((
                    Event(1, 0, n1, clamp(0.55 + 0.35 * strength), q_frame),
                    Event(1, 1, n1, 0., q_frame + int(RATE * 0.25)),
                    Event(1, 0, n2, clamp(0.62 + 0.32 * strength), q_frame + step),
                    Event(1, 1, n2, 0., q_frame + step + int(RATE * 0.35)),
                ))
            else:
                n_hit = max(48, min(84, degree_to_midi(self.chord_degree + max(0, base_deg), p.base_midi, p.scale)))
                events.extend((
                    Event(1, 0, n_hit, clamp(0.65 + 0.30 * strength), q_frame),
                    Event(1, 1, n_hit, 0., q_frame + int(RATE * 0.30)),
                    Event(3, 0, 38, clamp(0.45 + 0.40 * strength), q_frame),
                ))
            self.last_accent = seconds

        if features.left_strike != 0 and seconds - self.last_left_strike > 0.22:
            q_frame = quantize_gesture_frame(frame, self.next_frame, interval)
            strength = max(features.left_strike_strength, 0.55)
            if features.left_strike > 0:
                top_note = max(60, min(92, degree_to_midi(self.chord_degree + 8, p.base_midi, p.scale)))
                events.extend((
                    Event(0, 0, top_note, clamp(0.50 + 0.35 * strength), q_frame),
                    Event(0, 1, top_note, 0., q_frame + int(RATE * 0.90)),
                ))
            else:
                sub_note = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
                events.extend((
                    Event(3, 0, 36, clamp(0.60 + 0.35 * strength), q_frame),
                    Event(2, 0, sub_note, clamp(0.60 + 0.32 * strength), q_frame),
                    Event(2, 1, sub_note, 0., q_frame + int(RATE * 0.35)),
                ))
            self.last_left_strike = seconds

        return events
