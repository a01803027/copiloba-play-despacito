#!/usr/bin/env python3
"""60 Hz producer. SharedMemory owns the mapping; native C++ owns atomic access."""
import argparse
import ctypes as C
import math
import os
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

class AudioCommand(C.Structure):
    _fields_ = [('limb_id', C.c_uint8), ('event_type', C.c_uint8),
                ('note_or_param', C.c_uint8), ('reserved', C.c_uint8),
                ('value', C.c_float), ('target_frame', C.c_uint64)]

assert C.sizeof(AudioCommand) == 16
assert AudioCommand.value.offset == 4 and AudioCommand.target_frame.offset == 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, default=Path(__file__).parent / 'build/libcv_daw_ipc.so')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 1:
        parser.error('--seconds must be finite and >=1')
    native = C.CDLL(str(args.library.resolve()))
    signatures = {
        'cv_daw_abi_size': ([], C.c_uint32),
        'cv_daw_command_size': ([], C.c_uint32),
        'cv_daw_attach': ([C.c_void_p, C.c_uint32], C.c_int),
        'cv_daw_claim': ([C.c_void_p, C.c_uint32], C.c_int),
        'cv_daw_release': ([C.c_void_p, C.c_uint32], None),
        'cv_daw_push': ([C.c_void_p, C.POINTER(AudioCommand), C.c_uint32], C.c_int),
        'cv_daw_control': ([C.c_void_p, C.c_uint32, C.c_uint32, C.c_float, C.c_uint32], C.c_int),
        'cv_daw_metric': ([C.c_void_p, C.c_uint32], C.c_uint64),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(native, name)
        function.argtypes, function.restype = arguments, result
    # Python <=3.12 otherwise registers this C++-owned segment for unlink at exit.
    # Never let the mock sender unlink a live audio engine's mapping.
    if sys.version_info >= (3, 13):
        shm = shared_memory.SharedMemory(name='cv_daw_shm', create=False, track=False)
    else:
        from multiprocessing import resource_tracker
        shm = shared_memory.SharedMemory(name='cv_daw_shm', create=False)
        resource_tracker.unregister(shm._name, 'shared_memory')
    pointer = C.c_void_p(C.addressof(C.c_char.from_buffer(shm.buf)))
    token = os.getpid()
    claimed = False
    sent = 0
    try:
        if native.cv_daw_command_size() != C.sizeof(AudioCommand) or shm.size < native.cv_daw_abi_size():
            raise RuntimeError('C++/Python ABI size mismatch')
        if not native.cv_daw_attach(pointer, shm.size):
            raise RuntimeError('audio engine not ready or incompatible IPC version')
        if not native.cv_daw_claim(pointer, token):
            raise RuntimeError('another producer already owns the SPSC queue')
        claimed = True
        before = native.cv_daw_metric(pointer, 1)
        initial_frame = native.cv_daw_metric(pointer, 0)
        def send(limb, event, param, value, target=0):
            nonlocal sent
            command = AudioCommand(limb, event, param, 0, value, target)
            result = native.cv_daw_push(pointer, C.byref(command), token)
            if result != 1:
                raise RuntimeError(f'IPC enqueue failed ({result}); producer must not retry indefinitely')
            sent += 1
        # presets: saw, sine, kick, snare; preset 4=hat, 5=stab can be selected too.
        for limb, preset in enumerate((0, 1, 2, 3)):
            send(limb, 4, preset, 0)
        start = time.monotonic()
        frame = 0
        while time.monotonic()-start < args.seconds:
            t = time.monotonic()-start
            for limb in range(4):
                phase = t * 1.3 + limb
                values = ((0, 500 + 6500*(1+math.sin(phase))),
                          (1, .15+.1*math.sin(phase)), (2, 2*math.sin(phase*.5)),
                          (5, .35+.15*math.sin(phase)))
                for group, value in values:
                    if native.cv_daw_control(pointer, group, limb, value, token) != 1:
                        raise RuntimeError('control write rejected')
            native.cv_daw_control(pointer, 3, 0, 120+10*math.sin(t*.3), token)
            native.cv_daw_control(pointer, 4, 0, .5+.4*math.sin(t), token)
            if frame % 8 == 0:
                limb = (frame // 8) % 4
                now_frame = native.cv_daw_metric(pointer, 0)
                send(limb, 0, (60, 63, 65, 67)[limb], .8, now_frame+512)
                send(limb, 1, (60, 63, 65, 67)[limb], 0, now_frame+4800)
            # Record for first four seconds, then demonstrate locking/unlocking.
            if frame == 240:
                send(0, 3, 0, 1)
            if frame == 480:
                send(0, 3, 0, 0)
            frame += 1
            delay = start + frame/60 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        # Verify native consumer acknowledgment, not merely successful writes.
        deadline = time.monotonic()+1
        while native.cv_daw_metric(pointer, 1)-before < sent and time.monotonic()<deadline:
            time.sleep(.01)
        consumed = native.cv_daw_metric(pointer, 1)-before
        advanced = native.cv_daw_metric(pointer, 0)-initial_frame
        drops = native.cv_daw_metric(pointer, 2)
        invalid = native.cv_daw_metric(pointer, 3)
        overflow = native.cv_daw_metric(pointer, 5)
        if consumed != sent or advanced == 0 or drops or invalid or overflow:
            raise RuntimeError(f'IPC failed: sent={sent} consumed={consumed} frames={advanced} drops={drops} invalid={invalid} pending_full={overflow}')
        print(f'IPC PASS: sent={sent}, consumed={consumed}, audio_frames_advanced={advanced}, zero event drops')
        print(f'XRUNs={native.cv_daw_metric(pointer,8)}, stem_starved_frames={native.cv_daw_metric(pointer,7)}')
    finally:
        if claimed:
            native.cv_daw_release(pointer, token)
        shm.close()

if __name__ == '__main__':
    main()
