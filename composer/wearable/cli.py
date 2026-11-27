"""Standalone CLI for testing & calibrating the ESP32 BLE / Serial WS2812B LED Shirt.

Usage (works identically on STM32MP257F-DK and macOS):
  1) Verify physical strip segments (Red=Left Arm, Green=Chest, Blue=Right Arm):
      .venv-mac/bin/python -m composer.wearable.cli --mode ble --leds 25,10,25 --test-segments

  2) Run a 15-second simulated 124 BPM House music + move lightshow (punches, arm waves, drops):
      .venv-mac/bin/python -m composer.wearable.cli --mode ble --leds 25,10,25 --demo

  3) Test on protoboard over USB cable before enabling Bluetooth:
      .venv-mac/bin/python -m composer.wearable.cli --mode serial --port /dev/cu.usbserial-0001 --demo
"""
from __future__ import annotations

import argparse
import math
import time
from .ble_transport import create_wearable_transport
from .protocol import (
    DEFAULT_DEVICE_NAME,
    StripConfig,
    encode_config,
    encode_fx,
    encode_off,
    encode_state_frame,
    encode_test_pattern,
)


def main():
    p = argparse.ArgumentParser(description='CV-DAW Wearable ESP32 LED Shirt Calibration & BLE Test Tool')
    p.add_argument('--mode', choices=('ble', 'serial', 'mock'), default='ble')
    p.add_argument('--name', default=DEFAULT_DEVICE_NAME, help='BLE peripheral name (default: CV-DAW-SHIRT)')
    p.add_argument('--addr', default=None, help='Optional explicit BLE MAC / UUID')
    p.add_argument('--port', default=None, help='USB Serial port when --mode serial')
    p.add_argument('--leds', default='25,10,25', help='Segment LED counts: left,chest,right (default: 25,10,25)')
    p.add_argument('--no-reverse-left', action='store_true', help='Do not invert left-arm shoulder->wrist indexing')
    p.add_argument('--test-segments', action='store_true', help='Light Red=Left, Green=Chest, Blue=Right for 4s')
    p.add_argument('--demo', action='store_true', help='Run 15s 124-BPM music & dance FX showcase')
    p.add_argument('--off', action='store_true', help='Send OFF and exit')
    args = p.parse_args()

    cfg = StripConfig.parse(args.leds, reverse_left=not args.no_reverse_left)
    transport = create_wearable_transport(
        args.mode,
        device_name=args.name,
        address=args.addr,
        serial_port=args.port,
        initial_commands=(encode_config(cfg),),
        verbose_mock=(args.mode == 'mock'),
    )
    assert transport is not None

    try:
        if args.mode == 'ble':
            print(f'Scanning for BLE shirt "{args.name}"...', flush=True)
            deadline = time.monotonic() + 10.0
            while not transport.connected and time.monotonic() < deadline:
                time.sleep(0.1)
            if not transport.connected:
                print('WARNING: BLE device not found within 10s (continuing in background auto-connect).', flush=True)

        if args.off:
            transport.send_event(encode_off())
            time.sleep(0.3)
            return

        if args.test_segments:
            print(f'Sending CFG ({cfg.left_leds},{cfg.chest_leds},{cfg.right_leds}) + TEST segment pattern...', flush=True)
            transport.send_event(encode_config(cfg))
            transport.send_event(encode_test_pattern())
            time.sleep(4.0)

        if args.demo:
            print('Running 15s 124-BPM music + choreography LED showcase...', flush=True)
            bpm = 124.0
            beat_period = 60.0 / bpm
            started = time.monotonic()
            last_beat = -1
            fx_schedule = [
                (2.0, 'PL'),
                (3.2, 'PR'),
                (5.0, 'WL'),
                (7.2, 'WR'),
                (9.2, 'UP'),
                (11.2, 'DR'),
                (13.0, 'CU'),
            ]
            fx_idx = 0
            while (time.monotonic() - started) < 15.0:
                t = time.monotonic() - started
                beat_num = int(t / beat_period)
                beat_hit = beat_num != last_beat
                last_beat = beat_num

                while fx_idx < len(fx_schedule) and t >= fx_schedule[fx_idx][0]:
                    code = fx_schedule[fx_idx][1]
                    print(f'  t={t:4.1f}s -> FX {code}', flush=True)
                    transport.send_event(encode_fx(code, 0.95))
                    fx_idx += 1

                lh = 0.5 + 0.45 * math.sin(t * 2.1)
                rh = 0.5 + 0.45 * math.cos(t * 1.7)
                energy = 0.55 + 0.35 * math.sin(t * 0.8) ** 2
                transport.send_state(
                    encode_state_frame(
                        left_fill=int(round(lh * cfg.left_leds)),
                        right_fill=int(round(rh * cfg.right_leds)),
                        energy_0_99=int(round(energy * 99)),
                        palette_id=int(t // 3.0) % 6,
                        beat_pulse=beat_hit,
                    )
                )
                time.sleep(0.04)
            transport.send_event(encode_off())
            time.sleep(0.3)
    finally:
        transport.close()


if __name__ == '__main__':
    main()
