import math
import unittest
from composer.features import FeatureExtractor
from composer.rhythm import MultiSourceTempoTracker


def _base_pose(
    nose=(0.5, 0.20, 0.0),
    l_ear=(0.47, 0.20, 0.0),
    r_ear=(0.53, 0.20, 0.0),
    l_sh=(0.40, 0.35, 0.0),
    r_sh=(0.60, 0.35, 0.0),
    l_el=(0.35, 0.50, 0.0),
    r_el=(0.65, 0.50, 0.0),
    l_wr=(0.32, 0.55, 0.0),
    r_wr=(0.68, 0.55, 0.0),
    l_hip=(0.44, 0.65, 0.0),
    r_hip=(0.56, 0.65, 0.0),
):
    pts = [(0.5, 0.5, 0.0, 0.0)] * 33
    mapping = {
        0: nose, 7: l_ear, 8: r_ear,
        11: l_sh, 12: r_sh,
        13: l_el, 14: r_el,
        15: l_wr, 16: r_wr,
        23: l_hip, 24: r_hip,
    }
    for idx, (x, y, z) in mapping.items():
        pts[idx] = (x, y, z, 1.0)
    return pts


class RhythmAndBodyFeaturesTests(unittest.TestCase):
    def test_tempo_tracker_locks_at_variable_fps_hip_and_head(self):
        # 105 BPM hip bounce sampled at 24 FPS (variable-FPS webcam scenario).
        tracker = MultiSourceTempoTracker(initial_bpm=90.0)
        target_bpm = 105.0
        freq = target_bpm / 60.0
        bpm, conf = 90.0, 0.0
        for i in range(96):  # 4 seconds at 24 FPS
            t = i / 24.0
            hip_y = 0.65 + 0.06 * math.sin(2.0 * math.pi * freq * t)
            bpm, conf = tracker.update(t, hip_y=hip_y, head_y=0.4, arm_speed=0.0, bpm_min=70.0, bpm_max=150.0)
        self.assertGreater(conf, 0.6)
        self.assertAlmostEqual(bpm, target_bpm, delta=4.0)
        self.assertEqual(tracker.source, 'hip')

        # 124 BPM head nodding at 30 FPS with static hips.
        tracker2 = MultiSourceTempoTracker(initial_bpm=100.0)
        target_bpm2 = 124.0
        freq2 = target_bpm2 / 60.0
        for i in range(120):
            t = i / 30.0
            head_y = 0.45 + 0.05 * math.sin(2.0 * math.pi * freq2 * t)
            bpm, conf = tracker2.update(t, hip_y=0.65, head_y=head_y, arm_speed=0.0, bpm_min=80.0, bpm_max=150.0)
        self.assertGreater(conf, 0.6)
        self.assertAlmostEqual(bpm, target_bpm2, delta=4.0)
        self.assertEqual(tracker2.source, 'head')

    def test_fast_arm_up_and_down_strikes_at_25fps(self):
        ext = FeatureExtractor()
        # Baseline stillness for 5 frames at 25 FPS.
        for i in range(5):
            ext.update(_base_pose(r_wr=(0.68, 0.65, 0.0)), i / 25.0)

        # Rapid upward raise of right arm over 4 frames (160 ms).
        up_strikes = []
        for k, y in enumerate((0.55, 0.40, 0.22, 0.15, 0.15), start=5):
            f = ext.update(_base_pose(r_wr=(0.68, y, 0.0)), k / 25.0)
            if f.right_strike != 0:
                up_strikes.append((f.right_strike, f.right_strike_strength))
        self.assertTrue(any(d == 1 for d, _ in up_strikes), f'Expected upward strike, got {up_strikes}')

        # Wait past cooldown (6 frames = 240 ms) then rapid downward slash.
        for i in range(10, 16):
            ext.update(_base_pose(r_wr=(0.68, 0.15, 0.0)), i / 25.0)

        down_strikes = []
        for k, y in enumerate((0.28, 0.48, 0.66, 0.72, 0.72), start=16):
            f = ext.update(_base_pose(r_wr=(0.68, y, 0.0)), k / 25.0)
            if f.right_strike != 0:
                down_strikes.append((f.right_strike, f.right_strike_strength))
        self.assertTrue(any(d == -1 for d, _ in down_strikes), f'Expected downward strike, got {down_strikes}')

    def test_head_torso_and_hip_features_respond(self):
        ext = FeatureExtractor()
        neutral = ext.update(_base_pose(), 0.0)
        # Head lifted + tilted right, torso leaning right + twisted, hip swayed right & crouched realistically.
        pose = _base_pose(
            nose=(0.64, 0.16, 0.0),
            l_ear=(0.59, 0.14, 0.0),
            r_ear=(0.67, 0.20, 0.0),
            l_sh=(0.56, 0.42, -0.18),
            r_sh=(0.66, 0.42, 0.18),
            l_hip=(0.50, 0.72, 0.0),
            r_hip=(0.62, 0.72, 0.0),
        )
        expressive = neutral
        for i in range(1, 8):
            expressive = ext.update(pose, i / 30.0)
        self.assertGreater(expressive.head_pitch, neutral.head_pitch)
        self.assertGreater(expressive.head_tilt, 0.2)
        self.assertGreater(expressive.torso_lean, 0.15)
        self.assertGreater(expressive.torso_twist, 0.3)
        self.assertGreater(expressive.hip_sway, 0.1)
        self.assertLess(expressive.body_level, neutral.body_level)

    def test_shoulder_shrug_drop_bounce_and_head_nod(self):
        ext = FeatureExtractor()
        # Relaxed baseline for 10 frames.
        for i in range(10):
            ext.update(_base_pose(), i / 30.0)

        # Shrug shoulders up toward nose (l_sh/r_sh rise from 0.35 -> 0.28).
        max_lift = 0.0
        for i in range(10, 20):
            f = ext.update(_base_pose(l_sh=(0.40, 0.28, 0.0), r_sh=(0.60, 0.28, 0.0)), i / 30.0)
            max_lift = max(max_lift, f.shoulder_lift)
        self.assertGreater(max_lift, 0.65)

        # Release shoulders back down -> must fire discrete shoulder_drop event!
        dropped = False
        for i in range(20, 30):
            f = ext.update(_base_pose(l_sh=(0.40, 0.35, 0.0), r_sh=(0.60, 0.35, 0.0)), i / 30.0)
            dropped |= f.shoulder_drop
        self.assertTrue(dropped, 'Expected shoulder_drop on shrug release')

        # Alternating shoulder bobbing produces high shoulder_bounce and tilt.
        bounces = []
        for i in range(30, 50):
            dy = 0.04 * math.sin(i * 0.8)
            f = ext.update(_base_pose(l_sh=(0.40, 0.35 - dy, 0.0), r_sh=(0.60, 0.35 + dy, 0.0)), i / 30.0)
            bounces.append(f.shoulder_bounce)
        self.assertGreater(max(bounces), 0.35)

        # Crisp downward head nod triggers head_nod event.
        nodded = False
        for k, ny in enumerate((0.20, 0.23, 0.27, 0.29, 0.25, 0.20), start=50):
            f = ext.update(_base_pose(nose=(0.50, ny, 0.0)), k / 30.0)
            nodded |= f.head_nod
        self.assertTrue(nodded, 'Expected head_nod on downward nod')


if __name__ == '__main__':
    unittest.main()
