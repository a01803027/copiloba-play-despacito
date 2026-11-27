"""Non-blocking BLE Low Energy (Nordic UART Service) & USB Serial transports for the LED Shirt.

Runs BLE I/O on an isolated daemon asyncio thread so the 60 Hz vision + musical clock on
STM32MP257F-DK (OpenSTLinux BlueZ) or macOS (CoreBluetooth) NEVER blocks for even 0.1 ms.
Coalesces continuous state frames (`S...`) to only the freshest frame while guaranteeing
delivery of discrete choreography event triggers (`FX ...`, `CFG ...`, `TEST`, `OFF`).
"""
from __future__ import annotations

import asyncio
from collections import deque
import queue
import threading
import time
from typing import Protocol

from .protocol import DEFAULT_DEVICE_NAME, NUS_RX_CHAR_UUID


class WearableTransport(Protocol):
    """Unified non-blocking transport interface for BLE NUS, USB Serial, and Mock testing."""

    def send_state(self, line: str) -> None: ...

    def send_event(self, line: str) -> None: ...

    @property
    def connected(self) -> bool: ...

    def close(self) -> None: ...


class MockWearableTransport:
    """In-memory transport for unit tests and dry-run console inspection."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.states: deque[str] = deque(maxlen=200)
        self.events: list[str] = []
        self._closed = False

    @property
    def connected(self) -> bool:
        return not self._closed

    def send_state(self, line: str) -> None:
        if not self._closed:
            self.states.append(line)
            if self.verbose:
                print(f'[WEARABLE-MOCK STATE] {line.strip()}', flush=True)

    def send_event(self, line: str) -> None:
        if not self._closed:
            self.events.append(line)
            if self.verbose:
                print(f'[WEARABLE-MOCK EVENT] {line.strip()}', flush=True)

    def close(self) -> None:
        self._closed = True


class SerialTransport:
    """Non-blocking USB Serial transport (`115200` baud) for protoboard bench testing."""

    def __init__(self, port: str, baudrate: int = 115200):
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError('USB serial requires pyserial (`pip install pyserial`).') from exc

        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.05, write_timeout=0.05)
        self._events: queue.Queue[str] = queue.Queue(maxsize=32)
        self._latest_state: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name='wearable-serial')
        self._thread.start()

    @property
    def connected(self) -> bool:
        return self._connected and not self._stop.is_set()

    def send_state(self, line: str) -> None:
        with self._lock:
            self._latest_state = line

    def send_event(self, line: str) -> None:
        try:
            self._events.put_nowait(line)
        except queue.Full:
            try:
                self._events.get_nowait()
                self._events.put_nowait(line)
            except queue.Empty:
                pass

    def _worker(self) -> None:
        while not self._stop.is_set():
            sent_any = False
            while True:
                try:
                    ev = self._events.get_nowait()
                except queue.Empty:
                    break
                self._write(ev)
                sent_any = True

            with self._lock:
                st = self._latest_state
                self._latest_state = None
            if st is not None:
                self._write(st)
                sent_any = True

            if not sent_any:
                time.sleep(0.005)

    def _write(self, line: str) -> None:
        try:
            self._ser.write(line.encode('ascii', errors='ignore'))
        except Exception:
            self._connected = False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.5)
        try:
            self._ser.write(b'OFF\n')
            self._ser.close()
        except Exception:
            pass
        self._connected = False


class BleNusTransport:
    """Asynchronous BLE Low Energy (Nordic UART Service) client on a dedicated background thread.

    Compatible out-of-the-box with:
      - STM32MP257F-DK (OpenSTLinux 6.6 Scarthgap via BlueZ D-Bus)
      - macOS (Apple Silicon via CoreBluetooth)
    Automatically scans by `device_name` (or connects directly to `address`), writes commands
    with `response=False` (zero ACK round-trip lag), and auto-reconnects if the wearable resets.
    """

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        address: str | None = None,
        *,
        initial_commands: tuple[str, ...] = (),
        reconnect_delay_s: float = 2.0,
    ):
        try:
            import bleak  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'BLE wearable transport requires `bleak` (`pip install bleak`). '
                'On STM32MP257F-DK run: `pip3 install bleak` (uses BlueZ D-Bus).'
            ) from exc

        self.device_name = device_name
        self.address = address
        self.reconnect_delay_s = reconnect_delay_s
        self._events: queue.Queue[str] = queue.Queue(maxsize=32)
        self._last_cfg: str | None = None
        self._connect_count = 0
        self._lock = threading.Lock()
        for cmd in initial_commands:
            self.send_event(cmd)
        self._latest_state: str | None = None
        self._stop = threading.Event()
        self._connected = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name='wearable-ble-nus')
        self._thread.start()

    @property
    def connected(self) -> bool:
        return self._connected

    def send_state(self, line: str) -> None:
        with self._lock:
            self._latest_state = line

    def send_event(self, line: str) -> None:
        if line.startswith('CFG '):
            with self._lock:
                self._last_cfg = line
        try:
            self._events.put_nowait(line)
        except queue.Full:
            try:
                self._events.get_nowait()
                self._events.put_nowait(line)
            except queue.Empty:
                pass

    def _run_loop(self) -> None:
        asyncio.run(self._ble_main())

    async def _ble_main(self) -> None:
        from bleak import BleakClient, BleakScanner

        while not self._stop.is_set():
            try:
                target = self.address
                if not target:
                    dev = await BleakScanner.find_device_by_name(self.device_name, timeout=4.0)
                    if dev is None:
                        await asyncio.sleep(self.reconnect_delay_s)
                        continue
                    target = dev

                async with BleakClient(target, timeout=8.0) as client:
                    self._connect_count += 1
                    self._connected = True
                    target_label = getattr(target, 'address', str(target))
                    print(f'[WEARABLE BLE] Connected to {self.device_name} ({target_label})', flush=True)

                    # Restore strip geometry automatically if the ESP32 power-cycled & reconnected.
                    if self._connect_count > 1:
                        with self._lock:
                            cfg_cmd = self._last_cfg
                        if cfg_cmd is not None:
                            await client.write_gatt_char(NUS_RX_CHAR_UUID, cfg_cmd.encode('ascii'), response=False)

                    while not self._stop.is_set() and client.is_connected:
                        sent_any = False
                        # 1) Flush discrete high-priority FX / CFG / TEST events first.
                        while True:
                            try:
                                ev = self._events.get_nowait()
                            except queue.Empty:
                                break
                            await client.write_gatt_char(NUS_RX_CHAR_UUID, ev.encode('ascii'), response=False)
                            sent_any = True

                        # 2) Send only the latest coalesced continuous state frame (`S...`).
                        with self._lock:
                            st = self._latest_state
                            self._latest_state = None
                        if st is not None:
                            await client.write_gatt_char(NUS_RX_CHAR_UUID, st.encode('ascii'), response=False)
                            sent_any = True

                        await asyncio.sleep(0.008 if sent_any else 0.015)

                    # Turn off strip cleanly when stopping intentionally.
                    if self._stop.is_set() and client.is_connected:
                        await client.write_gatt_char(NUS_RX_CHAR_UUID, b'OFF\n', response=False)
            except Exception as exc:
                if self._connected:
                    print(f'[WEARABLE BLE] Disconnected ({exc}); retrying...', flush=True)
                self._connected = False
                if not self._stop.is_set():
                    await asyncio.sleep(self.reconnect_delay_s)
            finally:
                self._connected = False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._connected = False


def create_wearable_transport(
    mode: str = 'off',
    *,
    device_name: str = DEFAULT_DEVICE_NAME,
    address: str | None = None,
    serial_port: str | None = None,
    initial_commands: tuple[str, ...] = (),
    verbose_mock: bool = False,
) -> WearableTransport | None:
    """Factory for `'off'`, `'ble'`, `'serial'`, or `'mock'` wearable transports."""
    choice = (mode or 'off').strip().lower()
    if choice in ('off', 'none', ''):
        return None
    if choice == 'mock':
        m = MockWearableTransport(verbose=verbose_mock)
        for cmd in initial_commands:
            m.send_event(cmd)
        return m
    if choice == 'serial':
        if not serial_port:
            raise ValueError('--wearable serial requires --wearable-port (e.g. /dev/ttyUSB0 or /dev/cu.usbserial-*)')
        s = SerialTransport(port=serial_port)
        for cmd in initial_commands:
            s.send_event(cmd)
        return s
    if choice == 'ble':
        return BleNusTransport(device_name=device_name, address=address, initial_commands=initial_commands)
    raise ValueError(f'Unknown wearable mode: {mode!r} (choose off, ble, serial, mock)')
