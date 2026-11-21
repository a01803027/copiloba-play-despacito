import unittest
from composer.brain import Composer
from composer.features import Features
from composer.procedural import ContourMarkovMelody, euclidean_rhythm, quantize_gesture_frame, voice_lead


class ProceduralMusicTests(unittest.TestCase):
    def test_euclidean_rhythm_patterns_and_rotation(self):
        # E(4, 16) = classic four-on-the-floor (steps 0, 4, 8, 12).
        four = euclidean_rhythm(4, 16)
        self.assertEqual(sum(four), 4)
        self.assertEqual([i for i, x in enumerate(four) if x], [0, 4, 8, 12])
        # E(3, 8) = classic tresillo (3 evenly spaced pulses).
        tresillo = euclidean_rhythm(3, 8)
        self.assertEqual(sum(tresillo), 3)
        # Rotation shifts pulses cyclically without changing count.
        rotated = euclidean_rhythm(4, 16, rotation=2)
        self.assertEqual([i for i, x in enumerate(rotated) if x], [2, 6, 10, 14])

    def test_right_arm_contour_drives_melody_pitch(self):
        # Raising the right arm slowly must produce strictly higher average lead pitch than lowering it.
        low_brain = Composer(seed=11, genre='ambient')
        high_brain = Composer(seed=11, genre='ambient')
        low_notes, high_notes = [], []
        for n in range(180):
            ev_lo = low_brain.update(Features(energy=.55, right_height=.10, right_speed=.5), n * 1600, n / 30)
            ev_hi = high_brain.update(Features(energy=.55, right_height=.92, right_speed=.5), n * 1600, n / 30)
            low_notes.extend(e.note for e in ev_lo if e.kind == 0 and e.limb == 1)
            high_notes.extend(e.note for e in ev_hi if e.kind == 0 and e.limb == 1)
        self.assertGreater(len(low_notes), 4)
        self.assertGreater(len(high_notes), 4)
        self.assertGreater(sum(high_notes) / len(high_notes), sum(low_notes) / len(low_notes) + 6.0)

    def test_rapid_arm_strikes_trigger_distinct_quantized_events(self):
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=.5), 0, 0.0)
        # Right arm rapid UP triggers ascending 2-note arpeggio on limb 1.
        up_events = brain.update(
            Features(energy=.6, right_height=.8, right_strike=1, right_strike_strength=.85),
            1600, 0.5,
        )
        lead_ups = [e for e in up_events if e.kind == 0 and e.limb == 1]
        self.assertGreaterEqual(len(lead_ups), 2)
        self.assertGreater(lead_ups[-1].note, lead_ups[-2].note)

        # Left arm rapid DOWN triggers punchy kick (limb 3, note 36) + sub-bass (limb 2).
        down_events = brain.update(
            Features(energy=.6, left_height=.2, left_strike=-1, left_strike_strength=.9),
            4800, 1.0,
        )
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 36 for e in down_events))
        self.assertTrue(any(e.kind == 0 and e.limb == 2 for e in down_events))

    def test_auto_bpm_sync_and_lock(self):
        brain = Composer(seed=7, genre='house')
        start_bpm = brain.bpm
        # With calibrated smooth slew (+/-1.5 BPM per 0.5s), over 4 seconds it locks smoothly onto 130 BPM.
        for n in range(120):
            brain.update(
                Features(energy=.6, body_bpm=130.0, body_bpm_confidence=0.85, body_bpm_source='hip'),
                n * 1600, n / 30,
            )
        self.assertTrue(brain.bpm_synced)
        self.assertAlmostEqual(brain.bpm, 130.0, delta=1.5)
        self.assertNotAlmostEqual(brain.bpm, start_bpm, delta=2.0)

        # Locking tempo prevents further BPM drift.
        locked_bpm = brain.bpm
        self.assertTrue(brain.toggle_tempo_lock())
        for n in range(120, 180):
            brain.update(
                Features(energy=.6, body_bpm=118.0, body_bpm_confidence=0.90, body_bpm_source='hip'),
                n * 1600, n / 30,
            )
        self.assertAlmostEqual(brain.bpm, locked_bpm, places=4)

    def test_head_torso_hip_influence_controls_and_harmony(self):
        brain = Composer(seed=7, genre='ambient')
        c_neutral = brain.controls(Features(energy=.5, head_tilt=0.0, torso_lean=0.0, body_level=0.7))
        c_expressive = brain.controls(Features(energy=.5, head_tilt=0.8, torso_lean=0.8, body_level=0.2))
        # Head tilt steers stereo pan and boosts delay send.
        self.assertGreater(c_expressive['pan'][1], c_neutral['pan'][1] + 0.3)
        self.assertGreater(c_expressive['delay_send'], c_neutral['delay_send'])
        # Torso lean increases filter resonance.
        self.assertGreater(c_expressive['resonance'], c_neutral['resonance'] + 0.3)
        # Crouching (low body_level) closes filter cutoff (sub-drop effect).
        self.assertLess(c_expressive['cutoff'][0], c_neutral['cutoff'][0])

    def test_stillness_keeps_soft_breathing_pad(self):
        brain = Composer(seed=7, genre='ambient')
        ctrl = brain.controls(Features(energy=0.0, stillness=1.0, tracked=True))
        # Gentle breathing pad stays alive (>0.15) while bass/drums stay minimal.
        self.assertGreater(ctrl['gain'][0], 0.18)
        self.assertLess(ctrl['gain'][0], 0.35)

    def test_quantize_gesture_snaps_within_early_window(self):
        # 1000 frames ahead (~20.8 ms < 45 ms) snaps exactly to next_grid.
        self.assertEqual(quantize_gesture_frame(10000, 11000.0, 6000.0), 11000)
        # 4000 frames ahead (~83 ms > 45 ms) triggers immediately (+128).
        self.assertEqual(quantize_gesture_frame(10000, 14000.0, 6000.0), 10128)

    def test_voice_leading_follows_left_height(self):
        scale = (0, 2, 3, 5, 7, 8, 10)
        low_chord = voice_lead(None, 0, (0, 2, 4, 8), 48, scale, left_height=0.05)
        high_chord = voice_lead(None, 0, (0, 2, 4, 8), 48, scale, left_height=0.95)
        self.assertEqual(low_chord[0], 48)
        self.assertGreater(sum(high_chord[1:]), sum(low_chord[1:]))

    def test_head_tilt_triggers_dual_voice_harmony_and_delay(self):
        solo_brain = Composer(seed=7, genre='house')
        harm_brain = Composer(seed=7, genre='house')
        solo_notes, harm_notes = [], []
        for n in range(90):
            solo_notes.extend(e for e in solo_brain.update(Features(energy=.65, head_tilt=0.0), n * 1600, n / 30) if e.kind == 0 and e.limb == 1)
            harm_notes.extend(e for e in harm_brain.update(Features(energy=.65, head_tilt=0.75), n * 1600, n / 30) if e.kind == 0 and e.limb == 1)
        # Tilting head adds a parallel harmony voice -> ~2x lead notes!
        self.assertGreaterEqual(len(harm_notes), int(len(solo_notes) * 1.7))
        self.assertTrue(harm_brain.head_harmony_active)
        self.assertFalse(solo_brain.head_harmony_active)

    def test_shoulder_shrug_buildup_and_drop_impact(self):
        brain = Composer(seed=7, genre='house')
        # Warm up in GROOVE.
        for n in range(20):
            brain.update(Features(energy=.5), n * 1600, n / 30)
        # Shrug shoulders up -> enters BUILDUP and triggers snare roll + high resonance.
        buildup_events = []
        for n in range(20, 45):
            buildup_events += brain.update(Features(energy=.5, shoulder_lift=0.82), n * 1600, n / 30)
        self.assertEqual(brain.section, 'BUILDUP')
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 38 for e in buildup_events))
        c_build = brain.controls(Features(energy=.5, shoulder_lift=0.85))
        c_relax = brain.controls(Features(energy=.5, shoulder_lift=0.0))
        self.assertGreater(c_build['resonance'], c_relax['resonance'] + 0.8)

        # Release shrug (shoulder_drop=True) -> enters DROP and fires kick + crash (49) + sub-bass + strummed chord!
        drop_events = brain.update(Features(energy=.6, shoulder_lift=0.1, shoulder_drop=True), 45 * 1600, 45 / 30)
        self.assertEqual(brain.section, 'DROP')
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 36 for e in drop_events))
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 49 for e in drop_events))
        self.assertTrue(any(e.kind == 0 and e.limb == 2 for e in drop_events))
        self.assertTrue(any(e.kind == 0 and e.limb == 0 for e in drop_events))

    def test_head_nod_triggers_percussion_accent(self):
        brain = Composer(seed=7, genre='latin')
        brain.update(Features(energy=.5), 0, 0.0)
        ev = brain.update(Features(energy=.5, head_nod=True), 1600, 0.5)
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note in (42, 46) for e in ev))

    def test_arm_strike_punch_envelope_and_immediate_impact(self):
        brain = Composer(seed=7, genre='house')
        base_feat = Features(energy=.45, right_height=.5)
        brain.update(base_feat, 0, 0.0)
        ctrl_before = brain.controls(base_feat)
        # Fast right arm strike fires immediately (frame + 384 = 8ms, zero late commands) with loud lead punch!
        strike_feat = Features(energy=.45, right_height=.65, right_strike=1, right_strike_strength=0.95)
        ev = brain.update(strike_feat, 4800, 0.5)
        ctrl_after = brain.controls(strike_feat)
        immediate_hits = [e for e in ev if e.kind == 0 and e.frame == 5184]
        self.assertTrue(any(e.limb == 1 and e.value >= 0.90 for e in immediate_hits))
        self.assertGreater(ctrl_after['cutoff'][1], ctrl_before['cutoff'][1] + 2000.0)
        # Drum bus stays dry (no hiss reverb wash) and ultrasonic-capped.
        self.assertLessEqual(ctrl_after['reverb'][3], 0.05)
        self.assertLessEqual(ctrl_after['cutoff'][3], 11000.0)

    def test_leg_stomps_and_footwork_drive_rhythm_section(self):
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=.3, legs_visible=True, legs_source='knees'), 0, 0.0)
        ev_r = brain.update(Features(energy=.4, legs_visible=True, legs_source='knees', right_leg_strike=True), 4800, 0.5)
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 36 and e.value >= 0.9 for e in ev_r))
        self.assertTrue(any(e.kind == 0 and e.limb == 2 for e in ev_r))
        ev_l = brain.update(Features(energy=.4, legs_visible=True, legs_source='knees', left_leg_strike=True), 9600, 1.0)
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 38 for e in ev_l))
        self.assertTrue(any(e.kind == 0 and e.limb == 2 for e in ev_l))
        c_still_legs = brain.controls(Features(energy=.5, legs_visible=True, leg_energy=0.02))
        c_active_legs = brain.controls(Features(energy=.5, legs_visible=True, leg_energy=0.85, stance_width=0.9))
        self.assertGreater(c_active_legs['gain'][2], c_still_legs['gain'][2] + 0.45)
        self.assertGreater(c_active_legs['gain'][3], c_still_legs['gain'][3] + 0.55)
        # When legs are visible and STILL, autopilot drums cleanly drop out; moving legs slam them back in!
        still_brain = Composer(seed=7, genre='house')
        dance_brain = Composer(seed=7, genre='house')
        still_drums, dance_drums = 0, 0
        for i in range(60):
            still_drums += sum(1 for e in still_brain.update(Features(energy=.55, legs_visible=True, leg_energy=0.02), i * 1600, i / 30) if e.kind == 0 and e.limb == 3)
            dance_drums += sum(1 for e in dance_brain.update(Features(energy=.55, legs_visible=True, leg_energy=0.75), i * 1600, i / 30) if e.kind == 0 and e.limb == 3)
        self.assertEqual(still_drums, 0)
        self.assertGreater(dance_drums, 8)

    def test_bpm_sync_latches_peak_confidence_inside_window(self):
        brain = Composer(seed=7, genre='house')
        # Peak confidence occurs mid-window (t=0.30s) then dips right before t=0.50s tick.
        for i in range(20):
            t = i / 30.0
            conf = 0.78 if i == 9 else 0.20
            brain.update(Features(energy=.5, body_bpm=129.0, body_bpm_confidence=conf, body_bpm_source='legs'), i * 1600, t)
        self.assertTrue(brain.bpm_synced)
        self.assertEqual(brain.bpm_sync_source, 'legs')


if __name__ == '__main__':
    unittest.main()
