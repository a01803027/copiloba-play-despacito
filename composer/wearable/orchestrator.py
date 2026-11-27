"""Music- & Move-Synchronized LED Shirt Orchestrator (STM32MP257F-DK & macOS).

Translates CV-DAW's real-time body posture, 11 biomechanical moves, custom Tier 2 ML/DTW
dance moves, musical style palette, and exact quarter-note BPM beat clock into compact
`\n`-terminated ASCII commands streamed over BLE Low Energy (or USB Serial) to the ESP32 shirt.
"""
from __future__ import annotations

from .ble_transport import WearableTransport, create_wearable_transport
from .protocol import (
    DEFAULT_DEVICE_NAME,
    STYLE_PALETTE_IDS,
    StripConfig,
    encode_config,
    encode_fx,
    encode_off,
    encode_state_frame,
)

# Map CV-DAW MoveEvent(kind, side) -> 2-char ESP32 choreography FX code.
_MOVE_TO_FX: dict[tuple[str, str], str] = {
    ('HARD_PUNCH', 'left'): 'PL',
    ('HARD_PUNCH', 'right'): 'PR',
    ('LIGHT_PUNCH', 'left'): 'KL',
    ('LIGHT_PUNCH', 'right'): 'KR',
    ('ARM_WAVE', 'lr'): 'WL',
    ('ARM_WAVE', 'rl'): 'WR',
    ('DOUBLE_ARM_RISE', 'both'): 'UP',
    ('OVERHEAD_DROP', 'both'): 'DR',
    ('OPEN_BURST', 'both'): 'DR',
    ('SQUAT_RISE', 'both'): 'DR',
    ('JUMP_HIT', 'both'): 'DR',
    ('ARMS_CROSS_CUT', 'both'): 'XC',
    ('BODY_FREEZE', 'both'): 'FR',
    ('KNEE_STOMP', 'left'): 'KL',
    ('KNEE_STOMP', 'right'): 'KR',
    ('FOOT_STEP', 'left'): 'KL',
    ('FOOT_STEP', 'right'): 'KR',
}


class LedShirtOrchestrator:
    """Bridges `(Features, Composer)` -> non-blocking `WearableTransport` at ~25 Hz + instant FX."""

    def __init__(
        self,
        transport: WearableTransport,
        config: StripConfig | None = None,
        *,
        state_hz: float = 25.0,
    ):
        self.transport = transport
        self.config = config or StripConfig()
        self.min_state_interval = 1.0 / max(5.0, min(60.0, state_hz))
        self._last_state_t = -1e9
        self._last_beat_idx: int | None = None
        self._was_tracked = True
        self._was_crossed = False
        self._was_frozen = False
        # Push initial strip geometry so ESP32 matches configured LED counts immediately.
        self.transport.send_event(encode_config(self.config))

    def update(self, features, composer=None, *, now: float | None = None) -> None:
        """Send instant choreography FX triggers + rate-limited (~25 Hz) music-synced state frames."""
        import time as _time

        ts = float(now if now is not None else _time.monotonic())

        if not getattr(features, 'tracked', False):
            if self._was_tracked:
                self.transport.send_event(encode_off())
                self._was_tracked = False
            return
        self._was_tracked = True

        # 1) Instant discrete choreography MoveEvents (never rate-dropped!).
        for mv in getattr(features, 'moves', ()):
            code = _MOVE_TO_FX.get((mv.kind, mv.side))
            if code is None:
                code = _MOVE_TO_FX.get((mv.kind, 'both'), 'CU')  # Custom Tier 2 ML/DTW dance move -> 'CU'!
            raw_int = float(getattr(mv, 'intensity', 0.9))
            fx_int = min(0.42, raw_int) if mv.kind in ('LIGHT_PUNCH', 'FOOT_STEP') else raw_int
            self.transport.send_event(encode_fx(code, fx_int))

        # Continuous hold transitions (X-Cross breakdown & Freeze).
        crossed = bool(getattr(features, 'arms_crossed', False))
        frozen = bool(getattr(features, 'body_frozen', False))
        if crossed and not self._was_crossed:
            self.transport.send_event(encode_fx('XC', 0.90))
        if frozen and not self._was_frozen:
            self.transport.send_event(encode_fx('FR', 0.90))
        self._was_crossed = crossed
        self._was_frozen = frozen

        # 2) Detect exact quarter-note beat boundary from Composer (4 sixteenth-note ticks = 1 beat!).
        beat_hit = False
        palette_id = 1
        if composer is not None:
            palette_id = STYLE_PALETTE_IDS.get(getattr(composer, 'genre_key', 'house'), 1)
            tick = int(getattr(composer, 'tick', 0))
            beat_idx = tick // 4
            if self._last_beat_idx is None:
                self._last_beat_idx = beat_idx
            elif beat_idx != self._last_beat_idx:
                self._last_beat_idx = beat_idx
                if getattr(composer, 'beat_active', True) and not (crossed or frozen):
                    beat_hit = True

        # Always flush immediately on a beat tick; otherwise rate-limit state updates to `state_hz` (~25 Hz).
        if not beat_hit and (ts - self._last_state_t) < self.min_state_interval:
            return
        self._last_state_t = ts

        lh = max(0.0, min(1.0, float(getattr(features, 'left_height', 0.0))))
        rh = max(0.0, min(1.0, float(getattr(features, 'right_height', 0.0))))
        energy = max(0.0, min(1.0, float(getattr(features, 'energy', 0.0))))

        # Dim arms gracefully during X-Cross DJ breakdown so the chest heartbeat stands out.
        arm_scale = 0.20 if crossed else 1.0
        left_fill = int(round(lh * arm_scale * self.config.left_leds))
        right_fill = int(round(rh * arm_scale * self.config.right_leds))
        energy_int = int(round(energy * 99.0))

        self.transport.send_state(
            encode_state_frame(
                left_fill=left_fill,
                right_fill=right_fill,
                energy_0_99=energy_int,
                palette_id=palette_id,
                beat_pulse=beat_hit,
            )
        )

    def close(self) -> None:
        self.transport.close()


def create_shirt_orchestrator(
    mode: str = 'off',
    *,
    leds_spec: str = '25,10,25',
    reverse_left: bool = True,
    device_name: str = DEFAULT_DEVICE_NAME,
    address: str | None = None,
    serial_port: str | None = None,
    verbose_mock: bool = True,
) -> LedShirtOrchestrator | None:
    """Build a ready-to-use `LedShirtOrchestrator` from CLI options (returns `None` when `mode == 'off'`)."""
    cfg = StripConfig.parse(leds_spec, reverse_left=reverse_left)
    transport = create_wearable_transport(
        mode,
        device_name=device_name,
        address=address,
        serial_port=serial_port,
        verbose_mock=verbose_mock,
    )
    if transport is None:
        return None
    return LedShirtOrchestrator(transport, config=cfg)
