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
        tracker = MultiSourceTempoTracker(initial_bpm=90.0)
        target_bpm = 105.0
        freq = target_bpm / 60.0
        bpm, conf = 90.0, 0.0
        for i in range(96):
            t = i / 24.0
            hip_y = 0.65 + 0.06 * math.sin(2.0 * math.pi * freq * t)
            bpm, conf = tracker.update(t, hip_y=hip_y, head_y=0.4, arm_speed=0.0, bpm_min=70.0, bpm_max=150.0)
        self.assertGreater(conf, 0.6)
        self.assertAlmostEqual(bpm, target_bpm, delta=4.0)
        self.assertEqual(tracker.source, 'hip')

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

    def test_head_torso_and_hip_features_respond(self):
        ext = FeatureExtractor()
        neutral = ext.update(_base_pose(), 0.0)
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

    def test_shoulder_shrug_drop_bounce_and_look_down_rejection(self):
        ext = FeatureExtractor()
        for i in range(10):
            ext.update(_base_pose(), i / 30.0)

        max_lift = 0.0
        for i in range(10, 20):
            f = ext.update(_base_pose(l_sh=(0.40, 0.28, 0.0), r_sh=(0.60, 0.28, 0.0)), i / 30.0)
            max_lift = max(max_lift, f.shoulder_lift)
        self.assertGreater(max_lift, 0.65)

        dropped = False
        for i in range(20, 30):
            f = ext.update(_base_pose(l_sh=(0.40, 0.35, 0.0), r_sh=(0.60, 0.35, 0.0)), i / 30.0)
            dropped |= f.shoulder_drop
        self.assertTrue(dropped, 'Expected shoulder_drop on shrug release')

        bounces = []
        for i in range(30, 50):
            dy = 0.04 * math.sin(i * 0.8)
            f = ext.update(_base_pose(l_sh=(0.40, 0.35 - dy, 0.0), r_sh=(0.60, 0.35 + dy, 0.0)), i / 30.0)
            bounces.append(f.shoulder_bounce)
        self.assertGreater(max(bounces), 0.35)

        for i in range(50, 62):
            f = ext.update(_base_pose(l_sh=(0.40, 0.30, 0.0), r_sh=(0.60, 0.30, 0.0), l_wr=(0.35, 0.05, 0.0), r_wr=(0.65, 0.05, 0.0)), i / 30.0)
        self.assertLess(f.shoulder_lift, 0.25)

        for i in range(62, 74):
            f = ext.update(_base_pose(nose=(0.50, 0.28, 0.0), l_ear=(0.47, 0.20, 0.0), r_ear=(0.53, 0.20, 0.0)), i / 30.0)
        self.assertLess(f.shoulder_lift, 0.25)

    def test_leg_continuous_macros_knee_fallback_and_legs_bpm(self):
        ext = FeatureExtractor()

        def pose_with_knees(lk, rk, vis_ankles=False):
            pts = _base_pose()
            pts[25] = (*lk, 0.95)
            pts[26] = (*rk, 0.95)
            if vis_ankles:
                pts[27] = (lk[0], lk[1] + 0.18, 0.0, 0.95)
                pts[28] = (rk[0], rk[1] + 0.18, 0.0, 0.95)
            return pts

        for i in range(6):
            f = ext.update(pose_with_knees((0.44, 0.85, 0.0), (0.56, 0.85, 0.0)), i / 30.0)
        self.assertTrue(f.legs_visible)
        self.assertEqual(f.legs_source, 'knees')

        max_lift = 0.0
        for k, ry in enumerate((0.80, 0.72, 0.65, 0.74, 0.85), start=6):
            f = ext.update(pose_with_knees((0.44, 0.85, 0.0), (0.56, ry, 0.0)), k / 30.0)
            max_lift = max(max_lift, f.right_leg_lift)
        self.assertGreater(max_lift, 0.45)

        f_wide = ext.update(pose_with_knees((0.30, 0.85, 0.0), (0.70, 0.85, 0.0), vis_ankles=True), 12 / 30.0)
        self.assertEqual(f_wide.legs_source, 'ankles')
        self.assertGreater(f_wide.stance_width, 0.6)

        tracker = MultiSourceTempoTracker(initial_bpm=100.0)
        bpm, conf = 100.0, 0.0
        for i in range(120):
            t = i / 30.0
            pulse = max(0.0, math.sin(2.0 * math.pi * 2.0 * t))
            bpm, conf = tracker.update(t, hip_y=0.65, head_y=0.4, arm_speed=0.0, leg_pulse=pulse, bpm_min=80.0, bpm_max=150.0)
        self.assertGreater(conf, 0.55)
        self.assertAlmostEqual(bpm, 120.0, delta=4.0)
        self.assertEqual(tracker.source, 'legs')


if __name__ == '__main__':
    unittest.main()
