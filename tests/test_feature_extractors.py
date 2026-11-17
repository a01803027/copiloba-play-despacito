import math
import unittest

import numpy as np

from feature_extractors import FastKinematicsExtractor, HandPostureClassifier, HipBouncePLL, Posture, SlowTemporalExtractor
from perception_layer import PerceptionFrame


def frame(t, position=0., visible=True):
    pose = np.zeros((33, 4))
    pose[:, 3] = 1
    pose[15, 0] = position
    pose[15, 3] = visible
    return PerceptionFrame(t, pose, None, None, pose[:, :3], .5 + .025 * math.sin(2 * math.pi * 1.7 * t))


class FeatureTests(unittest.TestCase):
    def test_sine_strikes_isolate_limb_and_debounce(self):
        extractor = FastKinematicsExtractor()
        hits = []
        for i in range(120):
            t = i / 60
            limbs = extractor.update(frame(t, .4 * math.sin(4 * math.pi * t)))
            self.assertFalse(any(limb.hit for limb in limbs[1:]))
            if limbs[0].hit:
                hits.append(t)
                self.assertLessEqual(limbs[0].velocity, 1.)
        self.assertGreater(len(hits), 4)
        self.assertTrue(all(b - a >= .08 - 1e-12 for a, b in zip(hits, hits[1:])))

    def test_occlusion_does_not_trigger_on_reappearance(self):
        extractor = FastKinematicsExtractor()
        extractor.update(frame(1))
        extractor.update(frame(1.02, 8, False))
        self.assertEqual(extractor.update(frame(1.04, 10))[0].speed, 0)

    def test_hand_hysteresis(self):
        classifier = HandPostureClassifier()
        hand = np.zeros((21, 3))
        hand[9, 0] = 1
        self.assertIsNone(classifier.update(hand))
        self.assertIsNone(classifier.update(hand))
        self.assertEqual(classifier.update(hand), Posture.FIST)

    def test_bounce_tempo_and_stillness(self):
        for expected in (80, 100, 120, 160):
            pll = HipBouncePLL()
            for i in range(420):
                pll.update(.5 + .02 * math.sin(2 * math.pi * expected * i / 3600))
                if i % 12 == 0:
                    bpm, confidence = pll.estimate()
            self.assertAlmostEqual(bpm, expected, delta=2)
            self.assertGreater(confidence, .8)
        pll = HipBouncePLL()
        for _ in range(200):
            pll.update(.5)
        self.assertEqual(pll.estimate(), (120., 0.))

    def test_temporal_uses_real_timestamps_and_fallback(self):
        for rate in (30, 60):
            extractor = SlowTemporalExtractor()
            for i in range(rate * 7):
                result = extractor.update(frame(i / rate, math.sin(i / rate)))
            self.assertAlmostEqual(sum(result.groove), 1.0)
            self.assertAlmostEqual(result.target_bpm, 102, delta=2)
            self.assertGreater(result.bpm_confidence, .75)
