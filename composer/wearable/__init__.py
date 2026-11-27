"""Modular BLE Low Energy / USB Serial Wearable LED Shirt Orchestration (`composer/wearable/`)."""
from .ble_transport import (
    BleNusTransport,
    MockWearableTransport,
    SerialTransport,
    WearableTransport,
    create_wearable_transport,
)
from .orchestrator import LedShirtOrchestrator, create_shirt_orchestrator
from .protocol import (
    DEFAULT_DEVICE_NAME,
    NUS_RX_CHAR_UUID,
    NUS_SERVICE_UUID,
    NUS_TX_CHAR_UUID,
    STYLE_PALETTE_IDS,
    StripConfig,
    encode_config,
    encode_fx,
    encode_left_arm,
    encode_off,
    encode_right_arm,
    encode_state_frame,
    encode_test_pattern,
)

__all__ = [
    'BleNusTransport',
    'DEFAULT_DEVICE_NAME',
    'LedShirtOrchestrator',
    'MockWearableTransport',
    'NUS_RX_CHAR_UUID',
    'NUS_SERVICE_UUID',
    'NUS_TX_CHAR_UUID',
    'STYLE_PALETTE_IDS',
    'SerialTransport',
    'StripConfig',
    'WearableTransport',
    'create_shirt_orchestrator',
    'create_wearable_transport',
    'encode_config',
    'encode_fx',
    'encode_left_arm',
    'encode_off',
    'encode_right_arm',
    'encode_state_frame',
    'encode_test_pattern',
]
