"""Phase 4 & STM32MP257F-DK tests: kick arbitration, 3.3ms move latency, warm arm-raise, custom Tier 2 moves & COCO-17 NPU adapter."""
import unittest
from composer.brain import Composer, KICK_MIN_GAP_FRAMES, MOVE_HIT_CUSHION_FRAMES, RATE
from composer.features import FeatureExtractor, Features
from composer.moves import MoveEvent
from composer.vision import coco17_to_mediapipe33


class Phase4AudioAndStm32Tests(unittest.TestCase):
    def test_kick_arbitration_prevents_double_flam_kick_flooding(self):
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=0.7), 0, 0.0)

        # Rapid sequence of 5 left punches spaced 100ms apart (4800 frames) + House 4-on-the-floor sequencer.
        all_kicks = []
        all_bass = []
        for i in range(1, 10):
            frame = i * 4800  # 100 ms steps
            t = i * 0.10
            mvs = (MoveEvent('HARD_PUNCH', 'left', 0.95),) if i <= 5 else ()
            evs = brain.update(Features(energy=0.75, moves=mvs), frame, t)
            for e in evs:
                if e.kind == 0 and e.limb == 3 and e.note == 36:
                    all_kicks.append(e.frame)
                if e.kind == 0 and e.limb == 2:
                    all_bass.append(e.frame)

        # Every left punch still hits sub-bass immediately, while kicks stay spaced cleanly (no machine-gun flams!).
        self.assertGreaterEqual(len(all_bass), 5)
        all_kicks.sort()
        for a, b in zip(all_kicks, all_kicks[1:]):
            self.assertGreaterEqual(b - a, KICK_MIN_GAP_FRAMES, f'Double-flammed kicks too close: {a} vs {b}')

    def test_move_hit_latency_is_ultra_low_and_tamed_arm_rise_resonance(self):
        self.assertLessEqual(MOVE_HIT_CUSHION_FRAMES, 192)  # <= 4 ms scheduling cushion
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=0.5), 0, 0.0)

        # Immediate hit schedules at frame + MOVE_HIT_CUSHION_FRAMES (~3.3ms).
        evs = brain.update(Features(energy=0.6, moves=(MoveEvent('HARD_PUNCH', 'right', 0.9),)), 4800, 0.1)
        lead_ons = [e.frame for e in evs if e.kind == 0 and e.limb == 1]
        self.assertIn(4800 + MOVE_HIT_CUSHION_FRAMES, lead_ons)

        # Holding arms raised (soundwave_rise=1.0) should add only gentle warm resonance (< +0.25), not a harsh siren!
        c_down = brain.controls(Features(energy=0.5, soundwave_rise=0.0))
        c_up = brain.controls(Features(energy=0.5, soundwave_rise=1.0))
        self.assertLess(c_up['resonance'] - c_down['resonance'], 0.25)

    def test_custom_tier2_dance_move_emits_musical_harmonic_bloom(self):
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=0.5), 0, 0.0)
        evs = brain.update(Features(energy=0.6, moves=(MoveEvent('BODY_ROLL', 'both', 0.92),)), 9600, 0.2)
        lead_ons = [e for e in evs if e.kind == 0 and e.limb == 1]
        self.assertGreaterEqual(len(lead_ons), 4)

    def test_stm32_coco17_npu_adapter_feeds_cleanly_into_feature_extractor(self):
        # Construct realistic standing COCO-17 keypoints in (y, x, score) MoveNet format.
        def mk_coco(r_wr_xy=(0.70, 0.50), r_el_xy=(0.66, 0.48)):
            kp = [(0.5, 0.5, 0.95)] * 17
            coords_xy = {
                0: (0.50, 0.20),  # nose
                3: (0.47, 0.20),  # l_ear
                4: (0.53, 0.20),  # r_ear
                5: (0.40, 0.35),  # l_shoulder
                6: (0.60, 0.35),  # r_shoulder
                7: (0.34, 0.48),  # l_elbow
                8: r_el_xy,       # r_elbow
                9: (0.30, 0.52),  # l_wrist
                10: r_wr_xy,      # r_wrist
                11: (0.44, 0.65), # l_hip
                12: (0.56, 0.65), # r_hip
                13: (0.44, 0.85), # l_knee
                14: (0.56, 0.85), # r_knee
                15: (0.44, 0.98), # l_ankle
                16: (0.56, 0.98), # r_ankle
            }
            for idx, (x, y) in coords_xy.items():
                kp[idx] = (y, x, 0.96)
            return kp

        ext = FeatureExtractor()
        for i in range(5):
            pts33 = coco17_to_mediapipe33(mk_coco((0.56, 0.42), (0.62, 0.48)), yx_order=True)
            f = ext.update(pts33, i / 30.0)
            self.assertTrue(f.tracked)
            self.assertTrue(f.legs_visible)

        # Punch right arm outward using pure COCO-17 NPU keypoints -> triggers HARD_PUNCH!
        punches = []
        for k, (wr, el) in enumerate(
            (((0.74, 0.39), (0.67, 0.42)), ((0.94, 0.36), (0.72, 0.36))),
            start=5,
        ):
            pts33 = coco17_to_mediapipe33(mk_coco(wr, el), yx_order=True)
            f = ext.update(pts33, k / 30.0)
            punches.extend(m for m in f.moves if m.kind == 'HARD_PUNCH' and m.side == 'right')
        self.assertGreaterEqual(len(punches), 1)
        self.assertGreater(RATE, 0)


if __name__ == '__main__':
    unittest.main()
