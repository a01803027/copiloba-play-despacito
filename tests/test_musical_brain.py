import unittest

from feature_extractors import LimbFeatures, Posture, TemporalFeatures
from musical_brain import (EventType, HybridBeatQuantizer, LimbStateMachine,
                           MusicalBrain, PlayState, RECORD_PREVIOUS_GRID, SCALES, ScaleQuantizer)


def limbs(posture=None):
    return [LimbFeatures(i, valid=True, posture=posture if i == 0 else None) for i in range(4)]


class MusicalBrainTests(unittest.TestCase):
    def test_scale_edges_and_drum_mapping(self):
        for name, notes in SCALES.items():
            scale = ScaleQuantizer(name)
            self.assertEqual(scale.note(0, -3), notes[0])
            self.assertEqual(scale.note(1, 3), notes[-1])
            self.assertTrue(all(scale.note(0, i / 100) in notes for i in range(-30, 121)))
            self.assertEqual((scale.note(2, 0), scale.note(3, 0)), (38, 36))

    def test_edges_occlusion_and_exclusive_states(self):
        states = LimbStateMachine()
        self.assertEqual(states.update(limbs(Posture.FIST))[0].kind, EventType.MUTE_TOGGLE)
        self.assertEqual(states.update(limbs()), [])
        self.assertEqual(states.update(limbs(Posture.FIST)), [])
        self.assertEqual(states.play_state[0], PlayState.MUTED)
        self.assertEqual([event.kind for event in states.update(limbs(Posture.PINCH))],
                         [EventType.MUTE_TOGGLE, EventType.LOOP_LOCK])
        self.assertEqual(states.play_state[0], PlayState.LOOPING)
        states.update(limbs(Posture.OPEN))
        self.assertEqual(states.update(limbs(Posture.PINCH))[0].value, 0)
        self.assertEqual(states.play_state[0], PlayState.LIVE)

    def test_preset_cycle_with_modifier(self):
        states = LimbStateMachine(stationary_leg_modifier=True)
        for _ in range(8):
            self.assertEqual([x.limb for x in states.update(limbs(Posture.PEACE))], [0, 2])
            states.update(limbs(Posture.OPEN))
        self.assertEqual(states.active_preset_id, [0, 1, 7, 6])

    def test_beat_windows_and_engine_phase(self):
        quantizer = HybridBeatQuantizer()
        for frame, result in [(4080, (6000, 0)), (4079, (0, 0)),
                              (6000, (0, RECORD_PREVIOUS_GRID)),
                              (7680, (0, RECORD_PREVIOUS_GRID)), (7681, (0, 0))]:
            quantizer.sync(frame, 120)
            self.assertEqual(quantizer.quantize(), result)
        quantizer.sync(6200, 120, (8000 << 16) | 5500)
        self.assertEqual(quantizer.quantize(), (8000, 0))
        self.assertEqual(quantizer.quantize(7900), (8000, 0))

    def test_remix_controls_and_fixed_song_tempo(self):
        brain = MusicalBrain(bpm=100)
        idle, _ = brain.update(limbs(), TemporalFeatures(), 0)
        bounce, _ = brain.update(limbs(), TemporalFeatures(1, (1, 0, 0, 0), 160, 1), 0)
        self.assertGreater(bounce.stem_volumes[0], idle.stem_volumes[0])
        self.assertEqual(bounce.bpm, 100)
