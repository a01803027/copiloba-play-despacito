"""Phase 1 & 2 tests: Tier 0 2D PoseSanitizer, MoveArbiter, rapid combos, fluid arm wave & asymmetric stomps."""
import math
import unittest
from composer.features import FeatureExtractor
from composer.moves import FEATURE_DIM, MoveArbiter, MoveEvent, PoseSanitizer
from tests.test_moves import _pose


class PoseSanitizerAndArbiterTests(unittest.TestCase):
    def test_z_depth_hallucination_produces_zero_speed_and_bend_spike(self):
        ext = FeatureExtractor()
        # Establish clean baseline.
        for i in range(5):
            ext.update(_pose(r_wr=(0.70, 0.50, 0.0)), i / 30.0)

        # Inject a massive monocular Z-depth hallucination (+0.95 -> -0.95 in 1 frame) with identical (x, y)!
        f1 = ext.update(_pose(r_wr=(0.70, 0.50, 0.95)), 5 / 30.0)
        f2 = ext.update(_pose(r_wr=(0.70, 0.50, -0.95)), 6 / 30.0)

        self.assertLess(f1.right_speed, 0.05)
        self.assertLess(f2.right_speed, 0.05)
        self.assertEqual(f1.moves, ())
        self.assertEqual(f2.moves, ())

    def test_bone_length_and_teleport_wiggle_are_clamped_and_flagged(self):
        san = PoseSanitizer()
        for i in range(4):
            pf = san.process(_pose(r_el=(0.66, 0.48, 0.0), r_wr=(0.72, 0.52, 0.0)), i / 30.0)
            self.assertTrue(pf.tracked)
            self.assertEqual(len(pf.feature_vector), FEATURE_DIM)
            self.assertFalse(pf.wiggle_suppressed)

        # Teleport right wrist wildly across the entire screen (impossible forearm length > 2.5 torso).
        pf_glitch = san.process(_pose(r_el=(0.66, 0.48, 0.0), r_wr=(1.45, -0.35, 0.0)), 4 / 30.0)
        self.assertTrue(pf_glitch.wiggle_suppressed)
        self.assertIn('r_wrist', pf_glitch.glitch_joints)
        self.assertGreater(san.total_glitches_blocked, 0)

    def test_arbiter_suppresses_conflicting_punch_on_rise_and_double_knee_stomps(self):
        arb = MoveArbiter()

        # 1) Simultaneous HARD_PUNCH + DOUBLE_ARM_RISE -> DOUBLE_ARM_RISE wins, punch suppressed.
        out = arb.arbitrate(
            [MoveEvent('HARD_PUNCH', 'right', 0.9), MoveEvent('DOUBLE_ARM_RISE', 'both', 0.8)],
            timestamp=1.0,
        )
        self.assertEqual([e.kind for e in out], ['DOUBLE_ARM_RISE'])

        # 2) Simultaneous left + right KNEE_STOMP in same frame -> both suppressed as hop/shift glitch.
        out2 = arb.arbitrate(
            [MoveEvent('KNEE_STOMP', 'left', 0.9), MoveEvent('KNEE_STOMP', 'right', 0.9)],
            timestamp=2.0,
        )
        self.assertEqual(out2, [])

        # 3) HARD_PUNCH on a frame where that wrist was flagged for wiggle -> suppressed!
        out3 = arb.arbitrate(
            [MoveEvent('HARD_PUNCH', 'left', 0.95)],
            timestamp=3.0,
            glitch_joints=('l_wrist',),
        )
        self.assertEqual(out3, [])
        self.assertGreaterEqual(arb.suppressed_count, 4)

    def test_dual_channel_spotlight_and_tier2_intent_shield(self):
        arb = MoveArbiter()

        # 1) Upper-body ARM_WAVE + lower-body FOOT_STEP in same frame -> BOTH pass (independent channels!),
        #    while a competing upper-body HARD_PUNCH in the same frame is suppressed.
        out = arb.arbitrate(
            [
                MoveEvent('ARM_WAVE', 'lr', 0.90),
                MoveEvent('HARD_PUNCH', 'right', 0.95),
                MoveEvent('FOOT_STEP', 'right', 0.65),
            ],
            timestamp=10.0,
        )
        self.assertEqual([e.kind for e in out], ['ARM_WAVE', 'FOOT_STEP'])

        # 2) Rolling temporal spotlight: 0.30s after ARM_WAVE, trailing HARD_PUNCH / LIGHT_PUNCH is suppressed,
        #    while OVERHEAD_DROP (higher priority 98 > 82) IS allowed to cut through!
        self.assertEqual(arb.arbitrate([MoveEvent('HARD_PUNCH', 'left', 0.95)], timestamp=10.30), [])
        self.assertEqual(
            [e.kind for e in arb.arbitrate([MoveEvent('OVERHEAD_DROP', 'both', 0.95)], timestamp=10.35)],
            ['OVERHEAD_DROP'],
        )

        # 3) Tier 2 Intent Shield: when a custom Tier 2 move is forming (tier2_confidence >= 0.48),
        #    opportunistic Tier 1 punches and bursts are blocked!
        out_shield = arb.arbitrate(
            [MoveEvent('HARD_PUNCH', 'right', 0.92), MoveEvent('OPEN_BURST', 'both', 0.88)],
            timestamp=15.0,
            tier2_confidence=0.58,
        )
        self.assertEqual(out_shield, [])

    def test_two_tier_light_vs_heavy_punches_and_sparse_feet_drums(self):
        from composer.brain import Composer
        from composer.features import Features

        ext = FeatureExtractor()
        guard_r = ((0.62, 0.50, 0.0), (0.56, 0.42, 0.0))
        for i in range(6):
            ext.update(_pose(r_el=guard_r[0], r_wr=guard_r[1]), i / 30.0)

        # Moderate jab (~0.19 dx over 2 frames -> rvel ~ 3.5, below 4.6 heavy threshold) -> LIGHT_PUNCH only!
        light_seen, heavy_seen = 0, 0
        for k, (el, wr) in enumerate(
            (((0.66, 0.46, 0.0), (0.65, 0.40, 0.0)), ((0.69, 0.43, 0.0), (0.75, 0.39, 0.0))),
            start=6,
        ):
            f = ext.update(_pose(r_el=el, r_wr=wr), k / 30.0)
            light_seen += sum(1 for m in f.moves if m.kind == 'LIGHT_PUNCH')
            heavy_seen += sum(1 for m in f.moves if m.kind == 'HARD_PUNCH')
        self.assertGreaterEqual(light_seen, 1)
        self.assertEqual(heavy_seen, 0)

        # LIGHT_PUNCH in Composer triggers subtle percussion ONLY (zero loud lead synth stabs!).
        brain = Composer(seed=7, genre='house')
        brain.update(Features(energy=0.4, legs_visible=True, leg_energy=0.3), 0, 0.0)
        jab_evs = brain.update(
            Features(energy=0.4, legs_visible=True, leg_energy=0.3, moves=(MoveEvent('LIGHT_PUNCH', 'right', 0.55),)),
            1200,
            0.05,
        )
        self.assertEqual([e for e in jab_evs if e.kind == 0 and e.limb == 1], [])
        self.assertTrue(any(e.kind == 0 and e.limb == 3 and e.note in (42, 46) for e in jab_evs))

        # Sparse drums when feet stay planted (`legs_visible=True, leg_energy=0.0`) vs active footwork (`leg_energy=0.75`).
        sparse_brain = Composer(seed=7, genre='house')
        footwork_brain = Composer(seed=7, genre='house')
        sparse_evs, foot_evs = [], []
        for n in range(180):
            sparse_evs.extend(
                sparse_brain.update(
                    Features(energy=0.45, right_height=0.8, legs_visible=True, leg_energy=0.02),
                    n * 1600,
                    n / 30.0,
                )
            )
            foot_evs.extend(
                footwork_brain.update(
                    Features(energy=0.65, right_height=0.8, legs_visible=True, leg_energy=0.78),
                    n * 1600,
                    n / 30.0,
                )
            )
        sparse_hats_snares = [e for e in sparse_evs if e.kind == 0 and e.limb == 3 and e.note in (38, 42, 46)]
        foot_hats_snares = [e for e in foot_evs if e.kind == 0 and e.limb == 3 and e.note in (38, 42, 46)]
        self.assertEqual(sparse_hats_snares, [], 'Planted feet should produce zero busy hi-hats or snares')
        self.assertGreaterEqual(len(foot_hats_snares), 15, 'Active footwork should unlock full hats and snares')

        # Sensitive FOOT_STEP: gentle single-foot dance step (lift ~0.25 < 0.42 stomp threshold) fires FOOT_STEP!
        ext_step = FeatureExtractor()
        for i in range(6):
            ext_step.update(_pose(l_knee=(0.44, 0.85, 0.0), r_knee=(0.56, 0.85, 0.0)), i / 30.0)
        steps_seen = []
        for k, ry in enumerate((0.83, 0.815, 0.815, 0.84, 0.85, 0.85), start=6):
            f_st = ext_step.update(_pose(l_knee=(0.44, 0.85, 0.0), r_knee=(0.56, ry, 0.0)), k / 30.0)
            steps_seen.extend(m.kind for m in f_st.moves)
        self.assertIn('FOOT_STEP', steps_seen)
        self.assertNotIn('KNEE_STOMP', steps_seen)

    def test_rapid_sequential_boxing_combo_registers_every_punch(self):
        ext = FeatureExtractor()
        # Guard stance for 5 frames.
        guard_l = ((0.38, 0.48, 0.0), (0.44, 0.42, 0.0))
        guard_r = ((0.62, 0.48, 0.0), (0.56, 0.42, 0.0))
        t_idx = 0
        for _ in range(5):
            ext.update(_pose(l_el=guard_l[0], l_wr=guard_l[1], r_el=guard_r[0], r_wr=guard_r[1]), t_idx / 30.0)
            t_idx += 1

        # Throw a fast 6-punch alternating combo (L-R-L-R-L-R), each punch lasting ~0.27s (~3.7 punches/s!).
        detected = []
        for rep in range(6):
            side = 'left' if rep % 2 == 0 else 'right'
            # 3 frames extend + 5 frames retract = 8 frames (0.267s) per punch.
            for step in range(8):
                frac = min(1.0, step / 2.0) if step <= 2 else max(0.0, 1.0 - (step - 2) / 5.0)
                if side == 'left':
                    l_el = (0.38 - 0.10 * frac, 0.48 - 0.12 * frac, 0.0)
                    l_wr = (0.44 - 0.36 * frac, 0.42 - 0.06 * frac, 0.0)
                    r_el, r_wr = guard_r
                else:
                    l_el, l_wr = guard_l
                    r_el = (0.62 + 0.10 * frac, 0.48 - 0.12 * frac, 0.0)
                    r_wr = (0.56 + 0.36 * frac, 0.42 - 0.06 * frac, 0.0)
                f = ext.update(_pose(l_el=l_el, l_wr=l_wr, r_el=r_el, r_wr=r_wr), t_idx / 30.0)
                detected.extend((mv.side, t_idx) for mv in f.moves if mv.kind == 'HARD_PUNCH')
                t_idx += 1

        left_count = sum(1 for s, _ in detected if s == 'left')
        right_count = sum(1 for s, _ in detected if s == 'right')
        self.assertGreaterEqual(left_count, 3, f'Expected >=3 left punches in fast combo, got {detected}')
        self.assertGreaterEqual(right_count, 3, f'Expected >=3 right punches in fast combo, got {detected}')

    def test_continuous_sinusoidal_arm_wave_triggers_repeatedly(self):
        ext = FeatureExtractor()
        waves = []
        # 2.5 seconds of smooth continuous traveling sine wave L->R across wrists & elbows.
        for i in range(75):
            t = i / 30.0
            omega = 2.0 * math.pi * 1.25  # 1.25 Hz wave
            f = ext.update(
                _pose(
                    l_wr=(0.24, 0.48 - 0.09 * math.sin(omega * (t - 0.00)), 0.0),
                    l_el=(0.33, 0.46 - 0.07 * math.sin(omega * (t - 0.10)), 0.0),
                    r_el=(0.67, 0.46 - 0.07 * math.sin(omega * (t - 0.22)), 0.0),
                    r_wr=(0.76, 0.48 - 0.09 * math.sin(omega * (t - 0.32)), 0.0),
                ),
                t,
            )
            waves.extend(mv.side for mv in f.moves if mv.kind == 'ARM_WAVE')
        self.assertGreaterEqual(waves.count('lr'), 2, f'Expected repeated lr ARM_WAVE detections, got {waves}')

    def test_symmetric_both_knees_lift_never_fires_double_knee_stomp(self):
        ext = FeatureExtractor()
        for i in range(8):
            ext.update(_pose(l_knee=(0.44, 0.85, 0.0), r_knee=(0.56, 0.85, 0.0)), i / 30.0)

        # Lift BOTH knees symmetrically together (hop / camera shift) and drop back down -> ZERO stomps!
        stomps = []
        for k, ky in enumerate((0.75, 0.66, 0.64, 0.74, 0.85, 0.85), start=8):
            f = ext.update(_pose(l_knee=(0.44, ky, 0.0), r_knee=(0.56, ky, 0.0)), k / 30.0)
            stomps.extend(mv for mv in f.moves if mv.kind == 'KNEE_STOMP')
        self.assertEqual(stomps, [], f'Symmetric lift should fire zero KNEE_STOMP, got {stomps}')


if __name__ == '__main__':
    unittest.main()
