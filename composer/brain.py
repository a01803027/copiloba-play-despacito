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
import math
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
        self.last_leg_strike = [-1e9, -1e9]
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

        in_buildup = self.section == 'BUILDUP' or features.shoulder_lift >= 0.62
        in_drop = self.section == 'DROP'
        # Decaying ~240ms visceral transient punch envelope triggered by deliberate arm punches & leg stomps.
        punch = self.strike_punch * math.exp(-max(0.0, self._last_seconds - self.strike_punch_at) / 0.16)
        if punch < 0.01:
            punch = 0.0

        # Gentle breathing ambient pad stays alive softly in complete stillness;
        # other layers obey their dedicated body zones cleanly!
        pad_level = clamp(0.24 + 0.56 * features.energy + (0.12 if in_drop else 0.0) + 0.12 * punch)
        lead_level = clamp(
            (0.10 + 0.64 * clamp(features.energy * 0.95 + 0.35 * features.right_speed)) * (0.50 + 0.50 * features.expansion)
            + 0.26 * punch
        )
        # When legs are visible, YOUR LEGS DIRECTLY OWN THE RHYTHM SECTION (bass & drums)!
        # Still legs -> drums/bass drop way back; stepping/dancing legs -> full punchy rhythm!
        if features.legs_visible:
            leg_bass = clamp(0.08 + 0.86 * clamp(features.leg_energy * 1.15 + 0.20 * abs(features.hip_sway) + 0.20 * max(0.0, features.stance_width - 0.35)) + 0.14 * punch)
            leg_drum = clamp(0.04 + 0.92 * clamp(features.leg_energy * 1.18 + (0.25 if in_buildup else 0.0)) + 0.16 * punch)
            bass_mult = 0.45 if in_buildup else (1.12 if in_drop else 1.0)
            bass_level = clamp(leg_bass * bass_mult)
            drum_level = leg_drum
        else:
            rhythm_drive = clamp(features.energy + 0.25 * abs(features.hip_sway) + 0.25 * features.shoulder_bounce)
            bass_mult = 0.45 if in_buildup else (1.12 if in_drop else 1.0)
            bass_level = clamp((0.06 + 0.78 * rhythm_drive + 0.14 * punch) * bass_mult)
            drum_level = clamp(0.06 + 0.78 * clamp(rhythm_drive + (0.25 if in_buildup else 0.0)) + 0.16 * punch)

        # Body vertical level: neutral sitting/standing is ~0.65 (full open brightness ~1.0);
        # actively crouching (<0.45) closes filters into a warm sub-drop.
        level_mod = clamp(0.45 + 0.85 * features.body_level, 0.50, 1.15)
        # Shrugging shoulders up sweeps filters sharply upward for unmistakable Build-up tension!
        shrug_sweep = 1.0 + 0.65 * features.shoulder_lift
        stance_boost = max(0.0, features.stance_width - 0.35) if features.legs_visible else 0.0

        pad_bright = clamp(0.32 + 0.35 * features.expansion + 0.25 * features.left_height + 0.15 * features.head_pitch)
        pad_cut = (p.cutoff_pad_range[0] + (p.cutoff_pad_range[1] - p.cutoff_pad_range[0]) * pad_bright) * level_mod * shrug_sweep + 1800.0 * punch
        pad_cut = clamp(pad_cut, 420.0, 11500.0)

        lead_bright = clamp(0.34 + 0.38 * features.right_height + 0.15 * features.head_pitch + 0.16 * (1.0 - features.fluidity))
        lead_cut = (p.cutoff_lead_range[0] + (p.cutoff_lead_range[1] - p.cutoff_lead_range[0]) * lead_bright) * level_mod * shrug_sweep + 3800.0 * punch
        lead_cut = clamp(lead_cut, 700.0, 13200.0)

        rev_lift = clamp(0.60 * features.fluidity + 0.40 * features.torso_twist)
        pad_rev = p.reverb_pad_range[0] + (p.reverb_pad_range[1] - p.reverb_pad_range[0]) * rev_lift
        lead_rev = p.reverb_lead_range[0] + (p.reverb_lead_range[1] - p.reverb_lead_range[0]) * rev_lift

        # Head tilt clearly steers stereo image AND dramatically boosts Ping-Pong Delay!
        tilt_mag = abs(features.head_tilt)
        lead_pan = clamp(0.65 * features.head_tilt + 0.35 * features.lateral_balance, -0.88, 0.88)
        pad_pan = clamp(-0.35 * features.torso_lean - 0.25 * features.head_tilt, -0.5, 0.5)
        delay_send = clamp(p.delay_send * (0.50 + 0.35 * features.fluidity) + 0.45 * tilt_mag, 0.0, 0.85)

        # Shoulder shrug + torso lean build dramatic filter resonance (capped smoothly to avoid fizz).
        resonance = clamp(p.resonance + 1.15 * features.shoulder_lift + 0.45 * abs(features.torso_lean) + 0.25 * punch, 0.65, 2.75)
        lead_bend = clamp((features.right_bend - 0.5) * 0.4 + (1.0 - features.symmetry) * 0.10, -0.45, 0.45)

        sc_drive = clamp(0.35 + 0.55 * features.energy + (0.20 if in_drop else 0.0) + 0.20 * punch)

        # Keep Drum bus (limb 3) crisp yet strictly free of ultrasonic (>10.5 kHz) hiss, and DRY (reverb=0.02)!
        drum_cut = clamp(7800.0 + 1800.0 * features.body_level + 1200.0 * punch, 6500.0, 10800.0)
        bass_cut = clamp(480.0 + 1500.0 * clamp(features.energy + 0.25 * features.body_level + 0.35 * stance_boost), 260.0, 2900.0)

        return {
            'bpm': self.bpm,
            'energy': features.energy,
            'cutoff': [
                pad_cut,
                lead_cut,
                bass_cut,
                drum_cut,
            ],
            'reverb': [pad_rev, lead_rev, 0.03, 0.02],
            'gain': [pad_level, lead_level, bass_level * 0.84, drum_level * 0.94],
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
            self._peak_bpm_conf = 0.0
            return
        # Consume peak rhythm confidence observed anywhere inside the 0.5s window so brief
        # dips never prevent BPM sync, while keeping smooth slew (+/- 2.0 BPM per step).
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
            # Hold steady locked tempo through brief 1-2s phrase transitions.
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
        self._last_seconds = seconds
        if self.presets_dirty:
            for limb, preset_id in enumerate(p.presets):
                events.append(Event(limb, 4, preset_id, 0., 0))
            self.presets_dirty = False

        # Capture peak body BPM confidence on EVERY frame so 0.5s updates never miss a rhythm lock.
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
            while self.next_frame <= frame + lookahead:
                self.next_frame += interval
                self.tick += 1
            return events

        live_energy = clamp(0.68 * self.fast_context.energy + 0.32 * self.context.energy)
        self.section = self.sections.update(live_energy, features, seconds)
        self.head_harmony_active = abs(features.head_tilt) > 0.30
        self.shoulder_groove_active = features.shoulder_bounce > 0.55 or abs(features.shoulder_tilt) > 0.45

        # Instant SHOULDER DROP impact (releasing a shoulder shrug -> big Drop hit with Crash 49!).
        if features.shoulder_drop:
            q_drop = quantize_gesture_frame(frame, self.next_frame, interval)
            sub_root = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
            events.extend((
                Event(3, 0, 36, 0.94, q_drop),
                Event(3, 0, 49, 0.78, q_drop),
                Event(2, 0, sub_root, 0.90, q_drop),
                Event(2, 1, sub_root, 0., q_drop + int(RATE * 0.65)),
            ))
            drop_chord = voice_lead(self.prev_chord_notes, self.chord_degree, self.current_chord.voicing, p.base_midi, p.scale, left_height=max(0.65, features.left_height))
            for note, vel, on_f, off_f in strum_chord_schedule(drop_chord, q_drop, p.strum_ms, int(RATE * 1.4), 0.74):
                events.extend((Event(0, 0, note, vel, on_f), Event(0, 1, note, 0., off_f)))

        # Instant HEAD NOD accent (quantized rhythmic hi-hat/crash accent with 450ms cooldown).
        if features.head_nod and (seconds - self.last_head_nod) >= 0.45:
            q_nod = quantize_gesture_frame(frame, self.next_frame, interval)
            nod_note = 46 if self.section in ('CHORUS', 'DROP') else 42
            events.append(Event(3, 0, nod_note, clamp(0.50 + 0.30 * live_energy, 0.40, 0.85), q_nod))
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
            # When right arm is down and resting, give the melody space to breathe unless in DROP.
            right_arm_resting = features.right_height < 0.18 and features.right_speed < 0.40 and self.section != 'DROP'
            melody_notes = [] if right_arm_resting else PhraseSequencer.notes_at_slot(
                style=p,
                bar_in_cycle=self.bar_in_cycle,
                slot=slot,
                chord_degree=self.chord_degree,
                right_height=features.right_height,
                section=self.section,
                head_tilt=features.head_tilt,
            )
            for midi, length_steps, vel_scale in melody_notes:
                # Keep autopilot melody balanced (0.22..0.64) so intentional fast arm punches (0.95..1.0)
                # cut through the mix with undeniable +6 dB visceral punch!
                vel = clamp((0.26 + 0.34 * max(live_energy, clamp(features.right_speed / 2.5))) * vel_scale, 0.20, 0.64)
                dur_frames = int(max(RATE * 0.14, interval * length_steps * 0.88))
                events.extend((
                    Event(1, 0, midi, vel, target + swing_amt),
                    Event(1, 1, midi, 0., target + swing_amt + dur_frames),
                ))

            # Layer 2 — Authentic Style Bassline (when legs are visible, legs drive bass groove & density!).
            legs_drive_bass = (not features.legs_visible) or (features.leg_energy >= 0.14) or (slot == 0)
            if self.section in ('GROOVE', 'CHORUS', 'DROP') and legs_drive_bass:
                for b_note in p.bass_pattern:
                    if b_note.step == slot:
                        bass_base = max(28, p.base_midi - 12)
                        oct_lift = 12 if (self.shoulder_groove_active and slot >= 8 and slot % 4 != 0) else 0
                        midi = max(28, min(60, degree_to_midi(self.chord_degree + b_note.degree_offset, bass_base, p.scale) + oct_lift))
                        bass_e = features.leg_energy if features.legs_visible else live_energy
                        vel = clamp(0.40 + 0.44 * bass_e, 0.28, 0.94)
                        dur_frames = int(max(RATE * 0.12, interval * b_note.length_steps * 0.85))
                        events.extend((
                            Event(2, 0, midi, vel, target + swing_amt),
                            Event(2, 1, midi, 0., target + swing_amt + dur_frames),
                        ))

            # Layer 3 — Polyphonic Drum Kit (Preset 12): when legs are visible, YOUR LEGS CONTROL THE BEAT!
            drums = p.drums
            legs_want_drums = (not features.legs_visible) or (features.leg_energy >= 0.14)
            legs_want_full = (not features.legs_visible) or (features.leg_energy >= 0.38)
            is_bar4_fill = (self.bar_in_cycle == 3 and slot >= 12 and self.section in ('CHORUS', 'DROP') and legs_want_full)
            if self.section == 'BUILDUP':
                # Clean musical 8th-note rising snare tension (zero frantic 16th-note popping!).
                if slot % 2 == 0:
                    roll_vel = clamp(0.26 + 0.42 * features.shoulder_lift, 0.24, 0.72)
                    events.append(Event(3, 0, 38, roll_vel, target))
                    if slot % 4 == 0:
                        events.append(Event(3, 0, 42, roll_vel * 0.55, target))
            elif is_bar4_fill:
                for f_step, f_note in drums.fill_bar4:
                    if f_step == slot:
                        events.append(Event(3, 0, f_note, clamp(0.48 + 0.36 * live_energy, 0.35, 0.92), target + swing_amt))
            elif self.section in ('GROOVE', 'CHORUS', 'DROP') and legs_want_drums:
                use_full = self.section in ('CHORUS', 'DROP') and legs_want_full
                drum_e = features.leg_energy if features.legs_visible else live_energy
                kicks = drums.kick_full if use_full else drums.kick_groove
                hats = drums.hat_full if use_full else drums.hat_groove
                if slot in kicks:
                    events.append(Event(3, 0, 36, clamp(0.50 + 0.42 * drum_e, 0.36, 0.96), target))
                if use_full and slot in drums.snare:
                    events.append(Event(3, 0, 38, clamp(0.38 + 0.40 * drum_e, 0.30, 0.88), target + swing_amt))
                if slot in hats:
                    hat_note = 46 if (use_full and slot in drums.open_hat and features.expansion > 0.55) else 42
                    events.append(Event(3, 0, hat_note, clamp(0.20 + 0.26 * drum_e, 0.16, 0.65), target + swing_amt))
                # Alternating shoulder groove injects subtle musical ghost snares when dancing actively!
                if self.shoulder_groove_active and use_full and slot in drums.ghost_snare:
                    events.append(Event(3, 0, 38, clamp(0.22 + 0.18 * features.shoulder_bounce, 0.18, 0.44), target + swing_amt))

            self.tick += 1
            self.next_frame += interval

        # --- Immediate Clean Musical Punches (8ms safe offset = zero late_commands, zero drum popcorn!) ---
        if (features.right_strike != 0 or (features.acceleration > 52 and features.energy > 0.50)) and seconds - self.last_accent > 0.42:
            hit_frame = frame + 384  # ~8 ms: imperceptible instant feel AND zero late_commands!
            strength = max(features.right_strike_strength, 0.68)
            self.strike_punch = max(self.strike_punch, strength)
            self.strike_punch_at = seconds
            base_deg = int(round(features.right_height * 12.0))
            if features.right_strike >= 0:
                n1 = max(55, min(94, degree_to_midi(self.chord_degree + max(4, base_deg), p.base_midi, p.scale)))
                n2 = max(58, min(96, degree_to_midi(self.chord_degree + max(4, base_deg) + 2, p.base_midi, p.scale)))
                step = int(interval * 0.5)
                events.extend((
                    Event(1, 0, n1, clamp(0.86 + 0.14 * strength), hit_frame),
                    Event(1, 1, n1, 0., hit_frame + int(RATE * 0.30)),
                    Event(1, 0, n2, clamp(0.88 + 0.12 * strength), hit_frame + step),
                    Event(1, 1, n2, 0., hit_frame + step + int(RATE * 0.40)),
                ))
            else:
                n_hit = max(48, min(84, degree_to_midi(self.chord_degree + max(0, base_deg), p.base_midi, p.scale)))
                n_oct = min(96, n_hit + 12)
                events.extend((
                    Event(1, 0, n_hit, clamp(0.88 + 0.12 * strength), hit_frame),
                    Event(1, 1, n_hit, 0., hit_frame + int(RATE * 0.34)),
                    Event(1, 0, n_oct, clamp(0.76 + 0.16 * strength), hit_frame),
                    Event(1, 1, n_oct, 0., hit_frame + int(RATE * 0.28)),
                ))
            self.last_accent = seconds

        if features.left_strike != 0 and seconds - self.last_left_strike > 0.45:
            hit_frame = frame + 384
            strength = max(features.left_strike_strength, 0.68)
            self.strike_punch = max(self.strike_punch, strength)
            self.strike_punch_at = seconds
            if features.left_strike > 0:
                top_note = max(60, min(92, degree_to_midi(self.chord_degree + 8, p.base_midi, p.scale)))
                mid_note = max(53, min(84, degree_to_midi(self.chord_degree + 4, p.base_midi, p.scale)))
                events.extend((
                    Event(0, 0, mid_note, clamp(0.74 + 0.20 * strength), hit_frame),
                    Event(0, 1, mid_note, 0., hit_frame + int(RATE * 0.80)),
                    Event(0, 0, top_note, clamp(0.80 + 0.18 * strength), hit_frame),
                    Event(0, 1, top_note, 0., hit_frame + int(RATE * 0.95)),
                ))
            else:
                sub_note = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
                chord_hit = max(48, min(72, degree_to_midi(self.chord_degree, p.base_midi, p.scale)))
                events.extend((
                    Event(3, 0, 36, clamp(0.84 + 0.16 * strength), hit_frame),
                    Event(2, 0, sub_note, clamp(0.82 + 0.16 * strength), hit_frame),
                    Event(2, 1, sub_note, 0., hit_frame + int(RATE * 0.38)),
                    Event(0, 0, chord_hit, clamp(0.70 + 0.20 * strength), hit_frame),
                    Event(0, 1, chord_hit, 0., hit_frame + int(RATE * 0.45)),
                ))
            self.last_left_strike = seconds

        # --- Explicit Full-Body Leg Stomps (fires ONLY on real knee/foot lift + stomp, never standing jitter!) ---
        if features.right_leg_strike and (seconds - self.last_leg_strike[1]) >= 0.38:
            hit_frame = frame + 384
            self.strike_punch = max(self.strike_punch, 0.72)
            self.strike_punch_at = seconds
            sub_root = max(28, min(52, degree_to_midi(self.chord_degree, max(28, p.base_midi - 12), p.scale)))
            events.extend((
                Event(3, 0, 36, 0.96, hit_frame),
                Event(2, 0, sub_root, 0.88, hit_frame),
                Event(2, 1, sub_root, 0., hit_frame + int(RATE * 0.32)),
            ))
            self.last_leg_strike[1] = seconds

        if features.left_leg_strike and (seconds - self.last_leg_strike[0]) >= 0.38:
            hit_frame = frame + 384
            self.strike_punch = max(self.strike_punch, 0.62)
            self.strike_punch_at = seconds
            bass_fifth = max(31, min(57, degree_to_midi(self.chord_degree + 4, max(28, p.base_midi - 12), p.scale)))
            events.extend((
                Event(3, 0, 38, 0.86, hit_frame),
                Event(2, 0, bass_fifth, 0.80, hit_frame),
                Event(2, 1, bass_fifth, 0., hit_frame + int(RATE * 0.28)),
            ))
            self.last_leg_strike[0] = seconds

        return events
