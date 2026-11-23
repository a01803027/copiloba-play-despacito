import unittest
from composer.brain import Composer, RISER_MIDI
from composer.features import Features
from composer.moves import MoveEvent
from composer.procedural import quantize_gesture_frame, voice_lead


class StudioPerformanceComposerTests(unittest.TestCase):
    def test_right_arm_zones_shift_melody_cleanly_by_musical_octave(self):
        low_brain = Composer(seed=11, genre='ambient')
        high_brain = Composer(seed=11, genre='ambient')
        low_notes, high_notes = [], []
        for n in range(180):
            ev_lo = low_brain.update(Features(energy=.55, right_height=.10), n * 1600, n / 30)
            ev_hi = high_brain.update(Features(energy=.55, right_height=.92), n * 1600, n / 30)
            low_notes.extend(e.note for e in ev_lo if e.kind == 0 and e.limb == 1)
            high_notes.extend(e.note for e in ev_hi if e.kind == 0 and e.limb == 1)
        self.assertGreaterEqual(len(low_notes), 4)
        self.assertGreaterEqual(len(high_notes), 4)
        self.assertGreater(sum(high_notes) / len(high_notes), sum(low_notes) / len(low_notes) + 10.0)

    def test_auto_bpm_sync_latch_and_lock(self):
        brain = Composer(seed=7, genre='house')
        start_bpm = brain.bpm
        for n in range(120):
            brain.update(
                Features(energy=.6, body_bpm=130.0, body_bpm_confidence=0.85, body_bpm_source='hip'),
                n * 1600, n / 30,
            )
        self.assertTrue(brain.bpm_synced)
        self.assertAlmostEqual(brain.bpm, 130.0, delta=1.5)
        self.assertNotAlmostEqual(brain.bpm, start_bpm, delta=2.0)

        locked_bpm = brain.bpm
        self.assertTrue(brain.toggle_tempo_lock())
        for n in range(120, 180):
            brain.update(
                Features(energy=.6, body_bpm=118.0, body_bpm_confidence=0.90, body_bpm_source='hip'),
                n * 1600, n / 30,
            )
        self.assertAlmostEqual(brain.bpm, locked_bpm, places=4)

        brain2 = Composer(seed=7, genre='house')
        for i in range(20):
            conf = 0.78 if i == 9 else 0.20
            brain2.update(Features(energy=.5, body_bpm=128.0, body_bpm_confidence=conf, body_bpm_source='legs'), i * 1600, i / 30)
        self.assertTrue(brain2.bpm_synced)
        self.assertEqual(brain2.bpm_sync_source, 'legs')

    def test_head_torso_hip_macros_and_stillness_pad(self):
        brain = Composer(seed=7, genre='ambient')
        c_neutral = brain.controls(Features(energy=.5, head_tilt=0.0, torso_lean=0.0, body_level=0.7))
        c_expressive = brain.controls(Features(energy=.5, head_tilt=0.8, torso_lean=0.8, body_level=0.2))
        self.assertGreater(c_expressive['pan'][1], c_neutral['pan'][1] + 0.3)
        self.assertGreater(c_expressive['delay_send'], c_neutral['delay_send'])
        self.assertGreater(c_expressive['resonance'], c_neutral['resonance'] + 0.25)
        self.assertLess(c_expressive['cutoff'][0], c_neutral['cutoff'][0])

        c_still = brain.controls(Features(energy=0.0, stillness=1.0, tracked=True))
        self.assertGreater(c_still['gain'][0], 0.22)
        self.assertLess(c_still['gain'][0], 0.42)
        self.assertLessEqual(c_still['reverb'][3], 0.05)

    def test_quantize_gesture_and_voice_leading(self):
        self.assertEqual(quantize_gesture_frame(10000, 11000.0, 6000.0), 11000)
        self.assertEqual(quantize_gesture_frame(10000, 16000.0, 6000.0), 10384)
        scale = (0, 2, 3, 5, 7, 8, 10)
        low_chord = voice_lead(None, 0, (0, 2, 4, 8), 48, scale, left_height=0.05)
        high_chord = voice_lead(None, 0, (0, 2, 4, 8), 48, scale, left_height=0.95)
        self.assertEqual(low_chord[0], 48)
        self.assertGreater(sum(high_chord[1:]), sum(low_chord[1:]))

    def test_head_tilt_harmony_and_shoulder_shrug_drop(self):
        solo_brain = Composer(seed=7, genre='house')
        harm_brain = Composer(seed=7, genre='house')
        solo_notes, harm_notes = [], []
        for n in range(90):
            solo_notes.extend(e for e in solo_brain.update(Features(energy=.65, head_tilt=0.0), n * 1600, n / 30) if e.kind == 0 and e.limb == 1)
            harm_notes.extend(e for e in harm_brain.update(Features(energy=.65, head_tilt=0.75), n * 1600, n / 30) if e.kind == 0 and e.limb == 1)
        self.assertGreaterEqual(len(harm_notes), int(len(solo_notes) * 1.7))
        self.assertTrue(harm_brain.head_harmony_active)

        b = Composer(seed=7, genre='house')
        for n in range(25):
            b.update(Features(energy=.6, shoulder_lift=0.80), n * 1600, n / 30)
        self.assertEqual(b.section, 'BUILDUP')
        drop_ev = b.update(Features(energy=.7, shoulder_lift=0.1, shoulder_drop=True), 25 * 1600, 25 / 30)
        self.assertEqual(b.section, 'DROP')
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 49 for e in drop_ev))

    def test_soundwave_riser_and_overhead_drop_translation(self):
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=.5), 0, 0.0)
        ev_rise = brain.update(
            Features(energy=.6, soundwave_rise=0.75, moves=(MoveEvent('DOUBLE_ARM_RISE', 'both', 0.9, 0.75),)),
            4800, 0.5,
        )
        self.assertTrue(any(e.kind == 0 and e.limb == 1 and e.note == RISER_MIDI for e in ev_rise))
        ev_drop = brain.update(
            Features(energy=.7, soundwave_rise=0.0, moves=(MoveEvent('OVERHEAD_DROP', 'both', 0.95),)),
            24000, 1.5,
        )
        # Overhead drop cleanly releases riser (NOTE_OFF 120) and fires Kick + Crash + Sub!
        self.assertTrue(any(e.kind == 1 and e.limb == 1 and e.note == RISER_MIDI for e in ev_drop))
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 36 for e in ev_drop))
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 49 for e in ev_drop))


if __name__ == '__main__':
    unittest.main()
