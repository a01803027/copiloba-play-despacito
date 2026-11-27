"""Unit tests for the modular BLE Low Energy / Serial ESP32 Wearable LED Shirt (`composer/wearable/`)."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from composer.app import _build_parser
from composer.features import Features
from composer.moves import MoveEvent
from composer.wearable import (
    DEFAULT_DEVICE_NAME,
    LedShirtOrchestrator,
    MockWearableTransport,
    NUS_RX_CHAR_UUID,
    NUS_SERVICE_UUID,
    STYLE_PALETTE_IDS,
    StripConfig,
    create_shirt_orchestrator,
    encode_config,
    encode_fx,
    encode_left_arm,
    encode_off,
    encode_right_arm,
    encode_state_frame,
    encode_test_pattern,
)


class WearableProtocolTests(unittest.TestCase):
    def test_single_ble_packet_budget_and_newline_framing(self):
        cfg = StripConfig.parse('60,30,60')
        samples = [
            encode_left_arm(0),
            encode_left_arm(999, max_leds=150),
            encode_right_arm(-5),
            encode_right_arm(150, max_leds=150),
            encode_off(),
            encode_test_pattern(),
            encode_config(cfg),
            encode_state_frame(
                left_fill=150,
                right_fill=150,
                energy_0_99=99,
                palette_id=5,
                beat_pulse=True,
            ),
            encode_fx('PL', 1.0),
            encode_fx('WR', 0.0),
            encode_fx('CU', 0.85),
        ]
        for cmd in samples:
            raw = cmd.encode('ascii')
            self.assertTrue(cmd.endswith('\n'), f'Missing newline in {cmd!r}')
            self.assertLessEqual(
                len(raw),
                20,
                f'BLE GATT base-MTU payload exceeded (got {len(raw)} > 20 bytes): {cmd!r}',
            )

    def test_strip_config_defaults_to_symmetric_shoulder_to_wrist_indexing(self):
        cfg = StripConfig.parse('25,10,25')
        self.assertEqual((cfg.left_leds, cfg.chest_leds, cfg.right_leds), (25, 10, 25))
        self.assertEqual(cfg.total_leds, 60)
        self.assertTrue(cfg.reverse_left)
        self.assertEqual(encode_config(cfg), 'CFG 25,10,25,1\n')

        cfg_no_rev = StripConfig.parse('20,8,20', reverse_left=False)
        self.assertEqual(encode_config(cfg_no_rev), 'CFG 20,8,20,0\n')

        with self.assertRaises(ValueError):
            StripConfig.parse('25,10')
        with self.assertRaises(ValueError):
            StripConfig.parse('0,10,25')
        with self.assertRaises(ValueError):
            StripConfig.parse('200,10,25')

    def test_nus_uuid_constants_and_legacy_commands(self):
        self.assertEqual(DEFAULT_DEVICE_NAME, 'CV-DAW-SHIRT')
        self.assertTrue(NUS_SERVICE_UUID.startswith('6e400001-'))
        self.assertTrue(NUS_RX_CHAR_UUID.startswith('6e400002-'))
        self.assertEqual(encode_left_arm(12), 'L12\n')
        self.assertEqual(encode_right_arm(8), 'R8\n')
        self.assertEqual(encode_off(), 'OFF\n')


class WearableOrchestratorTests(unittest.TestCase):
    def test_initial_config_beat_sync_and_style_palettes(self):
        mock_tx = MockWearableTransport()
        cfg = StripConfig(left_leds=25, chest_leds=10, right_leds=25, reverse_left=True)
        shirt = LedShirtOrchestrator(mock_tx, config=cfg, state_hz=25.0)

        # Immediately queues initial strip configuration on creation.
        self.assertEqual(mock_tx.events, ['CFG 25,10,25,1\n'])

        comp = SimpleNamespace(genre_key='synthwave', tick=0, beat_active=True)
        feat = Features(tracked=True, left_height=0.80, right_height=0.40, energy=0.65)

        shirt.update(feat, comp, now=1.000)
        self.assertEqual(len(mock_tx.states), 1)
        # First observation initializes beat index (no false beat_pulse=1 on frame 0).
        self.assertEqual(mock_tx.states[-1], f'S20,10,64,{STYLE_PALETTE_IDS["synthwave"]},0\n')

        # Sub-interval frame (10 ms later < 40 ms @ 25 Hz) without beat change is coalesced/skipped.
        comp.tick = 3
        shirt.update(feat, comp, now=1.010)
        self.assertEqual(len(mock_tx.states), 1)

        # Quarter-note boundary (tick=4 -> beat_idx=1) bypasses rate-limit and fires B=1 immediately!
        comp.tick = 4
        shirt.update(feat, comp, now=1.016)
        self.assertEqual(len(mock_tx.states), 2)
        self.assertEqual(mock_tx.states[-1], f'S20,10,64,{STYLE_PALETTE_IDS["synthwave"]},1\n')

    def test_discrete_tier1_and_custom_tier2_moves_never_dropped(self):
        mock_tx = MockWearableTransport()
        shirt = LedShirtOrchestrator(mock_tx, state_hz=20.0)
        mock_tx.events.clear()

        # Rapid sequence inside 20 ms (< 50 ms state interval): discrete FX must ALL be delivered.
        f1 = Features(
            tracked=True,
            moves=(MoveEvent(kind='HARD_PUNCH', side='right', intensity=0.92),),
        )
        f2 = Features(
            tracked=True,
            moves=(MoveEvent(kind='ARM_WAVE', side='lr', intensity=0.88),),
        )
        f3 = Features(
            tracked=True,
            moves=(MoveEvent(kind='KAKAKA', side='both', intensity=0.95),),  # Custom Tier 2 move!
        )
        f4 = Features(
            tracked=True,
            moves=(MoveEvent(kind='LIGHT_PUNCH', side='right', intensity=0.65),),  # Subtle jab -> capped KR!
        )
        shirt.update(f1, None, now=2.000)
        shirt.update(f2, None, now=2.010)
        shirt.update(f3, None, now=2.020)
        shirt.update(f4, None, now=2.030)

        self.assertEqual(
            mock_tx.events,
            ['FX PR,91\n', 'FX WL,87\n', 'FX CU,94\n', 'FX KR,42\n'],
        )

    def test_xcross_freeze_and_tracking_loss_transitions(self):
        mock_tx = MockWearableTransport()
        shirt = LedShirtOrchestrator(mock_tx, state_hz=25.0)
        mock_tx.events.clear()
        comp = SimpleNamespace(genre_key='house', tick=0, beat_active=True)

        # Enter X-Cross DJ breakdown: triggers FX XC once and dims arm fills to 20%.
        f_cross = Features(tracked=True, left_height=1.0, right_height=1.0, energy=0.5, arms_crossed=True)
        shirt.update(f_cross, comp, now=5.00)
        self.assertIn('FX XC,89\n', mock_tx.events)
        self.assertEqual(mock_tx.states[-1], 'S5,5,50,1,0\n')

        # Beat ticks while crossed do NOT fire beat flashes.
        comp.tick = 4
        shirt.update(f_cross, comp, now=5.10)
        self.assertTrue(mock_tx.states[-1].endswith(',0\n'))

        # Tracking lost -> sends OFF\n exactly once until re-tracked.
        f_lost = Features(tracked=False)
        shirt.update(f_lost, comp, now=5.20)
        shirt.update(f_lost, comp, now=5.30)
        self.assertEqual(mock_tx.events.count('OFF\n'), 1)

        shirt.close()
        self.assertFalse(mock_tx.connected)

    def test_factory_off_and_app_cli_flags(self):
        self.assertIsNone(create_shirt_orchestrator('off'))
        mock_shirt = create_shirt_orchestrator('mock', leds_spec='30,12,30', verbose_mock=False)
        self.assertIsNotNone(mock_shirt)
        assert mock_shirt is not None
        self.assertEqual(mock_shirt.config.total_leds, 72)
        mock_shirt.close()

        parser = _build_parser()
        args = parser.parse_args(['--wearable', 'ble', '--wearable-leds', '24,8,24'])
        self.assertEqual(args.wearable, 'ble')
        self.assertEqual(args.wearable_leds, '24,8,24')
        self.assertFalse(args.wearable_no_reverse_left)


if __name__ == '__main__':
    unittest.main()
