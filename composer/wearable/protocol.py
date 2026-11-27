"""Compact ASCII-over-BLE-NUS protocol & strip geometry for the ESP32 WS2812B LED Shirt.

Every command is a single `\n`-terminated ASCII line guaranteed <= 20 bytes so it fits inside
a single unfragmented BLE 4.0+ GATT Write-Without-Response packet (default 23-byte ATT MTU ->
20-byte payload), while remaining 100% human-readable in a USB Serial Monitor and backward-
compatible with basic `L<idx>\n`, `R<idx>\n`, and `OFF\n` commands.

Physical strip layout (continuous WS2812B on ESP32 GPIO 13):
    Left Arm (0 .. left_leds-1)  ->  Chest/Back (.. +chest_leds)  ->  Right Arm (.. +right_leds)
"""
from __future__ import annotations

from dataclasses import dataclass

# Standard Nordic UART Service (NUS) UUIDs — universally supported on BlueZ (STM32), CoreBluetooth (Mac), & ESP32.
NUS_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
NUS_RX_CHAR_UUID = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'  # Host writes commands here (Write / WriteWithoutResponse)
NUS_TX_CHAR_UUID = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'  # ESP32 optional telemetry notify

DEFAULT_DEVICE_NAME = 'CV-DAW-SHIRT'

STYLE_PALETTE_IDS: dict[str, int] = {
    'ambient': 0,    # Cyan / Deep Violet aurora
    'house': 1,      # Warm Gold / Magenta / Club Cyan
    'lofi': 2,       # Cozy Amber / Rose Lo-Fi glow
    'latin': 3,      # Fiery Coral / Tropical Emerald
    'synthwave': 4,  # Neon Hot-Pink / Cyber Outrun Blue
    'dnb': 5,        # Electric Lime / Ice White breakbeat
}


@dataclass(frozen=True)
class StripConfig:
    """Configurable pixel counts for Left Arm -> Chest -> Right Arm continuous WS2812B strip.

    On a continuous strip wired Left-Wrist (0) -> Left-Shoulder -> Chest -> Right-Shoulder -> Right-Wrist,
    `reverse_left=True` (the default, matching ESP32 firmware) makes logical index 0 on BOTH arms start
    symmetrically at the shoulder and grow outward toward the wrist.
    """

    left_leds: int = 25
    chest_leds: int = 10
    right_leds: int = 25
    reverse_left: bool = True

    @property
    def total_leds(self) -> int:
        return self.left_leds + self.chest_leds + self.right_leds

    @classmethod
    def parse(cls, spec: str, reverse_left: bool = True) -> 'StripConfig':
        """Parse `'25,10,25'` into a validated `StripConfig`."""
        parts = [int(p.strip()) for p in spec.split(',') if p.strip()]
        if len(parts) != 3 or any(p < 1 or p > 150 for p in parts):
            raise ValueError(f'Expected 3 LED counts left,chest,right in [1..150], got {spec!r}')
        return cls(left_leds=parts[0], chest_leds=parts[1], right_leds=parts[2], reverse_left=reverse_left)


def _clamp_int(val: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(val))))


def encode_left_arm(index: int, max_leds: int = 25) -> str:
    """Legacy-compatible command: `L<0..max_leds>\\n`."""
    return f'L{_clamp_int(index, 0, max_leds)}\n'


def encode_right_arm(index: int, max_leds: int = 25) -> str:
    """Legacy-compatible command: `R<0..max_leds>\\n`."""
    return f'R{_clamp_int(index, 0, max_leds)}\n'


def encode_off() -> str:
    """Turn all LEDs off cleanly: `OFF\\n`."""
    return 'OFF\n'


def encode_test_pattern() -> str:
    """Trigger 3-second segment calibration pattern (Red=Left, Green=Chest, Blue=Right): `TEST\\n`."""
    return 'TEST\n'


def encode_config(cfg: StripConfig) -> str:
    """Live-configure segment sizes on ESP32 without reflashing: `CFG <L>,<C>,<R>,<revL>\\n`."""
    return f'CFG {cfg.left_leds},{cfg.chest_leds},{cfg.right_leds},{1 if cfg.reverse_left else 0}\n'


def encode_state_frame(
    *,
    left_fill: int,
    right_fill: int,
    energy_0_99: int,
    palette_id: int,
    beat_pulse: bool,
) -> str:
    """Compact music-synced state frame (<= 17 bytes): `S<L>,<R>,<E>,<P>,<B>\\n`."""
    l_v = _clamp_int(left_fill, 0, 150)
    r_v = _clamp_int(right_fill, 0, 150)
    e_v = _clamp_int(energy_0_99, 0, 99)
    p_v = _clamp_int(palette_id, 0, 9)
    b_v = 1 if beat_pulse else 0
    return f'S{l_v},{r_v},{e_v},{p_v},{b_v}\n'


def encode_fx(fx_code: str, intensity: float = 1.0) -> str:
    """Trigger a 120 FPS one-shot choreography effect on the ESP32: `FX <CODE>,<0..99>\\n`.

    Supported codes:
      - `PL` / `PR` : Left / Right Hard Punch shockwave (Chest -> Wrist)
      - `WL` / `WR` : Arm Wave L->R (`0 -> total-1`) / R->L (`total-1 -> 0`)
      - `UP`        : Double-Arm Rise warm aurora swell (Chest -> Both Wrists)
      - `DR`        : Overhead Drop / Jump / Squat-Rise full-strip burst
      - `XC`        : X-Cross DJ Breakdown (arms dim, chest deep heartbeat)
      - `FR`        : Body Freeze crystal sparkle hold
      - `KL` / `KR` : Left / Right Knee Stomp chest-to-shoulder bass pulse
      - `CU`        : Custom Tier 2 ML/DTW dance move prismatic spiral bloom
    """
    code = fx_code.strip().upper()[:2]
    lvl = _clamp_int(intensity * 99.0, 10, 99)
    return f'FX {code},{lvl}\n'
