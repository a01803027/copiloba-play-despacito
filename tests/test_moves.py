"""Comprehensive tests for all 11 Full-Body & Leg Biomechanical Moves + Musical Translation."""
import math
import unittest
from composer.brain import Composer, RISER_MIDI
from composer.features import FeatureExtractor, Features
from composer.moves import MoveEvent


def _pose(
    l_sh=(0.40, 0.35, 0.0),
    r_sh=(0.60, 0.35, 0.0),
    l_el=(0.34, 0.48, 0.0),
    r_el=(0.66, 0.48, 0.0),
    l_wr=(0.30, 0.52, 0.0),
    r_wr=(0.70, 0.52, 0.0),
    l_hip=(0.44, 0.65, 0.0),
    r_hip=(0.56, 0.65, 0.0),
    l_knee=(0.44, 0.85, 0.0),
    r_knee=(0.56, 0.85, 0.0),
):
    pts = [(0.5, 0.5, 0.0, 0.0)] * 33
    mapping = {
        0: (0.50, 0.20, 0.0), 7: (0.47, 0.20, 0.0), 8: (0.53, 0.20, 0.0),
        11: l_sh, 12: r_sh,
        13: l_el, 14: r_el,
        15: l_wr, 16: r_wr,
        23: l_hip, 24: r_hip,
        25: l_knee, 26: r_knee,
    }
    for idx, coord in mapping.items():
        pts[idx] = (*coord, 0.98)
    return pts


class FullBodyMoveRecognitionTests(unittest.TestCase):
    def test_hard_punch_requires_elbow_snap_and_rejects_smooth_arm_waving(self):
        ext = FeatureExtractor()
        # Start with right arm bent at guard (~95 deg: sh=(0.60,0.35), el=(0.62,0.50), wr=(0.54,0.42)).
        for i in range(6):
            ext.update(_pose(r_el=(0.62, 0.50, 0.0), r_wr=(0.54, 0.42, 0.0)), i / 30.0)

        # Explosive punch outward: elbow snaps straight to (0.72, 0.36) and wrist thrusts to (0.95, 0.36)!
        punches = []
        seq = (
            ((0.66, 0.43, 0.0), (0.72, 0.39, 0.0)),
            ((0.72, 0.36, 0.0), (0.94, 0.36, 0.0)),
            ((0.72, 0.36, 0.0), (0.95, 0.36, 0.0)),
        )
        for k, (el, wr) in enumerate(seq, start=6):
            f = ext.update(_pose(r_el=el, r_wr=wr), k / 30.0)
            punches.extend(mv for mv in f.moves if mv.kind == 'HARD_PUNCH' and mv.side == 'right')
        self.assertGreaterEqual(len(punches), 1, 'Expected HARD_PUNCH right on explosive elbow snap')

        # Smooth sinusoidal arm waving (straight arm, low-medium speed) must fire ZERO punches!
        false_punches = 0
        for i in range(25, 85):
            y = 0.48 + 0.14 * math.sin(i * 0.22)
            f = ext.update(_pose(r_el=(0.72, (0.35 + y) * 0.5, 0.0), r_wr=(0.82, y, 0.0)), i / 30.0)
            false_punches += sum(1 for mv in f.moves if mv.kind == 'HARD_PUNCH')
        self.assertEqual(false_punches, 0)

    def test_double_arm_rise_soundwave_and_overhead_drop(self):
        ext = FeatureExtractor()
        # Arms down at waist.
        for i in range(6):
            ext.update(_pose(l_wr=(0.32, 0.62, 0.0), r_wr=(0.68, 0.62, 0.0)), i / 30.0)

        # Raise both arms synchronously overhead -> DOUBLE_ARM_RISE + high soundwave_rise!
        rises = []
        max_sw = 0.0
        for k in range(6, 20):
            y = 0.62 - (k - 5) * 0.045
            f = ext.update(_pose(l_wr=(0.34, y, 0.0), r_wr=(0.66, y, 0.0)), k / 30.0)
            rises.extend(mv for mv in f.moves if mv.kind == 'DOUBLE_ARM_RISE')
            max_sw = max(max_sw, f.soundwave_rise)
        self.assertGreaterEqual(len(rises), 1)
        self.assertGreater(max_sw, 0.75)

        # Slam both arms down rapidly -> OVERHEAD_DROP!
        drops = []
        for k, y in enumerate((0.18, 0.42, 0.68, 0.72), start=20):
            f = ext.update(_pose(l_wr=(0.34, y, 0.0), r_wr=(0.66, y, 0.0)), k / 30.0)
            drops.extend(mv for mv in f.moves if mv.kind == 'OVERHEAD_DROP')
        self.assertGreaterEqual(len(drops), 1)

    def test_arms_cross_cut_and_open_burst(self):
        ext = FeatureExtractor()
        for i in range(5):
            ext.update(_pose(l_wr=(0.32, 0.48, 0.0), r_wr=(0.68, 0.48, 0.0)), i / 30.0)

        # Cross wrists in front of chest (l_wr.x > r_wr.x opposite to l_sh.x < r_sh.x) for 0.25s -> ARMS_CROSS_CUT!
        crossed_seen = False
        for i in range(5, 16):
            f = ext.update(_pose(l_wr=(0.58, 0.45, 0.0), r_wr=(0.42, 0.45, 0.0)), i / 30.0)
            crossed_seen |= f.arms_crossed
        self.assertTrue(crossed_seen)

        # Throw both arms wide open -> OPEN_BURST!
        bursts = []
        for k, dx in enumerate((0.18, 0.34, 0.46), start=16):
            f = ext.update(_pose(l_wr=(0.50 - dx, 0.45, 0.0), r_wr=(0.50 + dx, 0.45, 0.0)), k / 30.0)
            bursts.extend(mv for mv in f.moves if mv.kind == 'OPEN_BURST')
        self.assertGreaterEqual(len(bursts), 1)

    def test_arm_wave_propagation_left_to_right(self):
        ext = FeatureExtractor()
        waves = []
        # Synthesize a traveling wave peaking sequentially: L_wr (t=0.25s) -> L_el (0.40s) -> R_el (0.60s) -> R_wr (0.78s).
        for i in range(32):
            t = i / 30.0
            def bump(t0):
                return 0.14 * math.exp(-((t - t0) / 0.08) ** 2)
            f = ext.update(
                _pose(
                    l_wr=(0.25, 0.50 - bump(0.25), 0.0),
                    l_el=(0.33, 0.48 - bump(0.40), 0.0),
                    r_el=(0.67, 0.48 - bump(0.60), 0.0),
                    r_wr=(0.75, 0.50 - bump(0.78), 0.0),
                ),
                t,
            )
            waves.extend(mv for mv in f.moves if mv.kind == 'ARM_WAVE')
        self.assertTrue(any(mv.side == 'lr' for mv in waves), f'Expected lr ARM_WAVE, got {waves}')

    def test_body_freeze_after_active_dancing(self):
        ext = FeatureExtractor()
        # Dance energetically for 18 frames.
        for i in range(18):
            dy = 0.22 * math.sin(i * 1.1)
            ext.update(_pose(l_wr=(0.30, 0.45 + dy, 0.0), r_wr=(0.70, 0.45 - dy, 0.0)), i / 30.0)
        # Suddenly freeze completely still for 15 frames -> BODY_FREEZE!
        frozen = False
        for i in range(18, 35):
            f = ext.update(_pose(l_wr=(0.30, 0.45, 0.0), r_wr=(0.70, 0.45, 0.0)), i / 30.0)
            frozen |= f.body_frozen
        self.assertTrue(frozen)

    def test_three_leg_moves_knee_stomp_squat_dive_rise_and_jump_hit(self):
        ext = FeatureExtractor()
        # Baseline standing.
        for i in range(8):
            ext.update(_pose(), i / 30.0)

        # 1) KNEE_STOMP (right): high knee lift (>=0.40) then crisp stomp down!
        stomps = []
        for k, ry in enumerate((0.76, 0.66, 0.62, 0.72, 0.85, 0.85), start=8):
            f = ext.update(_pose(r_knee=(0.56, ry, 0.0)), k / 30.0)
            stomps.extend(mv for mv in f.moves if mv.kind == 'KNEE_STOMP' and mv.side == 'right')
        self.assertGreaterEqual(len(stomps), 1)

        # 2) SQUAT_DIVE + SQUAT_RISE: dip hips down by 0.14 for 0.25s, then spring back up!
        max_squat = 0.0
        for i in range(15, 25):
            f = ext.update(_pose(l_sh=(0.40, 0.48, 0.0), r_sh=(0.60, 0.48, 0.0), l_hip=(0.44, 0.78, 0.0), r_hip=(0.56, 0.78, 0.0)), i / 30.0)
            max_squat = max(max_squat, f.squat_depth)
        self.assertGreater(max_squat, 0.55)

        squat_rises = []
        for k, dy in enumerate((0.08, 0.02, 0.0, 0.0), start=25):
            f = ext.update(_pose(l_sh=(0.40, 0.35 + dy, 0.0), r_sh=(0.60, 0.35 + dy, 0.0), l_hip=(0.44, 0.65 + dy, 0.0), r_hip=(0.56, 0.65 + dy, 0.0)), k / 30.0)
            squat_rises.extend(mv for mv in f.moves if mv.kind == 'SQUAT_RISE')
        self.assertGreaterEqual(len(squat_rises), 1)

        # 3) JUMP_HIT: settle standing then launch hips/shoulders rapidly upward!
        for i in range(30, 42):
            ext.update(_pose(), i / 30.0)
        jumps = []
        for k, lift in enumerate((0.05, 0.12, 0.16), start=42):
            f = ext.update(_pose(l_sh=(0.40, 0.35 - lift, 0.0), r_sh=(0.60, 0.35 - lift, 0.0), l_hip=(0.44, 0.65 - lift, 0.0), r_hip=(0.56, 0.65 - lift, 0.0)), k / 30.0)
            jumps.extend(mv for mv in f.moves if mv.kind == 'JUMP_HIT')
        self.assertGreaterEqual(len(jumps), 1)

    def test_composer_translates_all_moves_into_distinct_studio_events(self):
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=.5), 0, 0.0)

        # HARD_PUNCH right -> octave lead stab + snare crack.
        ev_pr = brain.update(Features(energy=.6, moves=(MoveEvent('HARD_PUNCH', 'right', 0.95),)), 4800, 0.5)
        self.assertGreaterEqual(sum(1 for e in ev_pr if e.kind == 0 and e.limb == 1), 2)
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 38 for e in ev_pr))

        # HARD_PUNCH left -> heavy kick + sub-bass boom (no spammy crash).
        ev_pl = brain.update(Features(energy=.6, moves=(MoveEvent('HARD_PUNCH', 'left', 0.95),)), 9600, 1.0)
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 36 for e in ev_pl))
        self.assertTrue(any(e.kind == 0 and e.limb == 2 for e in ev_pl))

        # ARM_WAVE -> 6-note stereo liquid arpeggio ripple!
        ev_w = brain.update(Features(energy=.6, moves=(MoveEvent('ARM_WAVE', 'lr', 0.9),)), 14400, 1.5)
        self.assertGreaterEqual(sum(1 for e in ev_w if e.kind == 0 and e.limb == 1), 6)

        # ARMS_CROSS_CUT -> cuts drums cleanly; SQUAT_DIVE -> dives filter underwater!
        c_normal = brain.controls(Features(energy=.6, left_height=.7))
        c_cross = brain.controls(Features(energy=.6, left_height=.7, arms_crossed=True))
        c_squat = brain.controls(Features(energy=.6, left_height=.7, squat_depth=0.9))
        self.assertLess(c_cross['gain'][3], c_normal['gain'][3] * 0.2)
        self.assertLess(c_squat['cutoff'][0], c_normal['cutoff'][0] * 0.55)
        self.assertGreater(c_squat['gain'][2], c_normal['gain'][2] * 1.1)

        # KNEE_STOMP (right/left), SQUAT_RISE, JUMP_HIT all produce clean distinct hits.
        ev_ks = brain.update(Features(energy=.6, moves=(MoveEvent('KNEE_STOMP', 'right', 0.9),)), 19200, 2.0)
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 36 for e in ev_ks))
        ev_jh = brain.update(Features(energy=.6, moves=(MoveEvent('JUMP_HIT', 'both', 0.95),)), 24000, 2.5)
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note == 49 for e in ev_jh))
        ev_sr = brain.update(Features(energy=.6, moves=(MoveEvent('SQUAT_RISE', 'both', 0.95),)), 28800, 3.0)
        self.assertTrue(any(e.kind == 0 and e.limb == 0 for e in ev_sr))
        ev_sw = brain.update(Features(energy=.6, soundwave_rise=0.8, moves=(MoveEvent('DOUBLE_ARM_RISE', 'both', 0.9),)), 33600, 3.5)
        self.assertTrue(any(e.kind == 0 and e.limb == 1 and e.note == RISER_MIDI for e in ev_sw))


if __name__ == '__main__':
    unittest.main()
