# Modules 4 and 5 — Cortex-A35 / OpenSTLinux

This directory is a standalone C++20 implementation of the requested IPC ABI and
miniaudio audio engine. It is independent of the earlier `include/cvdaw` engine:
use **one** engine/brain pipeline at a time. Its shared memory is `/cv_daw_shm`,
not the earlier `/cvdaw`. The existing musical brain/frontend must use this new
ABI/native producer bridge to drive this engine.

## Files and build

* `ipc_bridge.hpp`: 16-byte/aligned command ABI, 256-command wait-free SPSC,
  atomic controls, validated POSIX mapping/ownership.
* `ipc_bridge_c.cpp`: shared library exposing acquire/release atomic operations
  to Python; direct ctypes field writes into C++ atomics are not supported.
* `dsp_primitives.hpp`: smoothing, ADSR, eight-voice pool, RAM samples,
  octave-bandlimited Fourier saw tables, sine table, Cytomic TPT SVF, reverb,
  scalar/NEON bounded rational limiter.
* `audio_engine.hpp`: pure DSP `processBlock`, frame clock, bounded future-event
  heap, four two-bar pattern tables, independent stem volume/mute controls.
* `audio_engine.cpp`: WAV decoding at startup, miniaudio ALSA runtime, device
  geometry verification, RT startup hook and offline IPC demonstration.
* `rt_startup.hpp`: process memory lock, actual device-thread affinity/FIFO/FPCR.
* `stem_worker.hpp`: four preloaded stereo stems and real Rubber Band streaming
  worker; only its preallocated FIFO is touched from the DSP thread.
* `vision_mock_sender.py`: 60 Hz controls/events and native-consumer verification.

On a normal Linux build host, install CMake, a C++20 compiler, pthreads and Rubber
Band >=3.0 development files (Debian/Ubuntu package `librubberband-dev`). ALSA is
dynamically loaded by miniaudio; `libasound.so.2` must exist on the target.
Miniaudio 0.11.21 is vendored with its license, so no configure-time download is
needed. Build natively on the board:

```sh
cmake -S realtime -B realtime/build \
  -DCMAKE_BUILD_TYPE=Release -DCV_DAW_CORTEX_A35=ON \
  -DCV_DAW_WITH_RUBBERBAND=ON
cmake --build realtime/build --parallel 2
ctest --test-dir realtime/build --output-on-failure
```

The optimized targets use `-O3 -mcpu=cortex-a35 -ffast-math -ftree-vectorize`.
NEON is part of ARMv8-A AArch64: do not add the ARM32-only `-mfpu=neon` option.
Input finite checks inspect IEEE-754 bits so `-ffast-math` cannot erase them.
`-DCV_DAW_FAST_MATH=OFF` is available for numerical comparisons.

To cross-build in an **OpenSTLinux SDK Linux shell**, source the installed
Cortex-A35 SDK setup script first. This script discovers a single installed
matching SDK rather than assuming a BSP release directory:

```sh
sdk_setup=$(find /opt/st -type f -name 'environment-setup-cortexa35*-linux' -print)
test "$(printf '%s\n' "$sdk_setup" | wc -l)" -eq 1
. "$sdk_setup"
: "${OE_CMAKE_TOOLCHAIN_FILE:=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake}"
test -f "$OE_CMAKE_TOOLCHAIN_FILE"
cmake -S realtime -B realtime/build-stm32 \
  -DCMAKE_TOOLCHAIN_FILE="$OE_CMAKE_TOOLCHAIN_FILE" \
  -DCMAKE_BUILD_TYPE=Release -DCV_DAW_CORTEX_A35=ON \
  -DCV_DAW_WITH_RUBBERBAND=ON
cmake --build realtime/build-stm32 --parallel 2
```

If multiple SDKs are installed, select the setup script for the image deployed
on the board before sourcing it. Include `rubberband` in the Yocto image and its
development package in the SDK. pkg-config must resolve to the **target** sysroot,
not workstation libraries. ST documents the SDK layout and setup in
[Standard SDK directory structure](https://wiki.st.com/stm32mpu/index.php/Standard_SDK_directory_structure)
and [CMake SDK setup](https://wiki.st.com/stm32mpu/wiki/How_to_build_Qt_applications_using_CMake).
Do not run cross-built tests on the workstation without an AArch64 emulator.

For host-only functional testing, omit the Cortex flag and disable Rubber Band
if it is not installed. This build still contains real sample/synth/looper DSP:

```sh
cmake -S realtime -B realtime/build -DCV_DAW_WITH_RUBBERBAND=OFF
cmake --build realtime/build --parallel 2
ctest --test-dir realtime/build --output-on-failure
```

The root build also supports `-DCVDAW_MODULES45=ON`. Production Linux builds
require Rubber Band by default. A build explicitly disabling it rejects `--stems`
instead of pretending to time-stretch.

## Linux RT privileges

These commands are to run **on the board**, using an administrator account:

```sh
sudo groupadd -f audio
sudo usermod -aG audio,video "$USER"
sudo tee -a /etc/security/limits.conf >/dev/null <<'LIMITS'
# CV-DAW real-time audio (PAM login sessions)
@audio - rtprio 95
@audio - memlock unlimited
LIMITS
```

Log out and log back in, then verify:

```sh
id
ulimit -r
ulimit -l
```

Expect an RT priority limit >=90 and unlimited locked memory. PAM sessions must
load `pam_limits.so`. Systemd services do not inherit PAM limits: install
`cv-daw-engine.service` below, which explicitly sets the limits.

The initializer calls `mlockall(MCL_CURRENT | MCL_FUTURE)` after decoding,
preallocation, device creation and worker creation. The ALSA backend start hook
runs on **miniaudio's actual audio pthread**, before playback/prefill. It calls
`pthread_setaffinity_np(pthread_self(), ..., CPU_SET(1))`, prefaults worker stack,
sets FPCR.FZ, then calls `pthread_setschedparam(..., SCHED_FIFO, priority=90)`.
Failures abort startup. ARMv8-A FZ handles single/double subnormal inputs and
outputs; Cortex-A35 has no independent DAZ/FIZ flag. See the
[Arm instruction reference](https://documentation-service.arm.com/static/6245c734b059dc5ff9a8bdab).

## Kernel isolation

OpenSTLinux boot configuration location varies by boot medium/BSP. On the stock
STM32MP257x-DK flow the serial log names `/mmc0_extlinux/extlinux.conf` on bootfs;
under Linux this is normally `/boot/mmc0_extlinux/extlinux.conf`. Check the serial
log and locate the active file first:

```sh
find /boot -type f -iname '*extlinux*.conf' -print
cat /boot/mmc0_extlinux/extlinux.conf
```

For that stock path, the following exact commands preserve all existing boot
arguments and append the requested isolation settings to each boot-menu APPEND
line. If your serial boot log names a different file, pass that active file as
the final argument instead. Keep the backup for serial-console recovery:

```sh
sudo cp -a /boot/mmc0_extlinux/extlinux.conf /boot/mmc0_extlinux/extlinux.conf.before-cvdaw
sudo python3 - /boot/mmc0_extlinux/extlinux.conf <<'BOOT'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
extra = ('isolcpus=1', 'nohz_full=1', 'rcu_nocbs=1', 'irqaffinity=0')
found = False
for i, line in enumerate(lines):
    if re.match(r'^\s*append\s+', line, re.I):
        found = True
        tokens = line.split()
        for option in extra:
            key = option.split('=')[0] + '='
            tokens = [t for t in tokens if not t.startswith(key)]
            tokens.append(option)
        lines[i] = '  ' + ' '.join(tokens)
if not found:
    raise SystemExit('No APPEND lines: no change made')
p.write_text('\n'.join(lines) + '\n')
BOOT
sudo reboot
```

After reboot:

```sh
cat /proc/cmdline
cat /sys/devices/system/cpu/isolated
cat /proc/interrupts
```

Expect CPU `1` isolated. Use PREEMPT_RT, performance governor and appropriate
thermal management. Route non-audio IRQs/workqueues and vision/NPU work to CPU 0;
configure irqbalance so it does not undo affinity. Kernel/per-CPU/managed IRQs
need board-specific inspection; `isolcpus=1` alone does not provide exclusive
execution or a latency guarantee. Do not starve the IRQ thread needed by ALSA.
ST's board boot sequence is documented in
[Execute basic commands](https://wiki.stmicroelectronics.cn/stm32mpu/wiki/Getting_started/STM32MP2_boards/STM32MP257x-DK/Let%27s_start/Execute_basic_commands).

## Run and verify IPC

For the default synthesized kick/snare/hat/stab bank:

```sh
taskset -c 0 ./realtime/build/cv-daw-engine
```

The miniaudio context is explicitly ALSA-only, device `hw:0,0`, callback format
f32, 48 kHz stereo, 128 frames and two periods. Native integer sample-format
conversion is permitted for codecs lacking f32; native sample rate, channels,
period length and period count must match exactly or startup fails. There is no
PulseAudio/PipeWire/default-device fallback. miniaudio owns I/O waits outside
`processBlock()`. ALSA EPIPE recovery is intercepted, counted and rejected;
underruns stop the stream instead of silently recovering.

Load actual percussion WAV files and/or all four stems at startup:

```sh
taskset -c 0 ./realtime/build/cv-daw-engine \
  --samples kick.wav snare.wav hat.wav stab.wav \
  --stems 120 drums.wav bass.wav vocals.wav other.wav
```

WAVs decode/resample into preallocated interleaved float32 stereo at startup;
non-finite samples are rejected and finite amplitudes are clamped to [-1,1].
Sample limit is 15 seconds; stem limit is 15 minutes **per stem**. All stems must
have equal lengths and already aligned downbeats. Allocate RAM for all decoded
samples (48000*2*4 bytes per second per stem) plus working memory before enabling
`mlockall`. Samples gate after two seconds with a 120 ms release; explicit note-off
also releases them. Built-in kick/snare/hat/stab samples work without asset files.
Presets 0=saw, 1=sine, 2=kick, 3=snare, 4=hat, 5=stab.

In a second terminal, as the same Unix user:

```sh
taskset -c 0 python3 realtime/vision_mock_sender.py \
  --library realtime/build/libcv_daw_ipc.so --seconds 10
```

The harness verifies ABI size/offsets, claims the sole producer slot, publishes
60 Hz controls and timestamped hits, and requires C++ consumer acknowledgments,
advancing master clock, zero command drops and no invalid/scheduler events.
On macOS the library extension is `.dylib`. A non-RT offline IPC demonstration is
also available on macOS/Linux:

```sh
./realtime/build/cv-daw-engine --offline 7500
# In a second terminal, run the mock sender with the matching library path.
```

Only the audio creator unlinks shared memory on orderly shutdown; Python never
unlinks it. Python 3.13+ uses `track=False`; earlier versions unregister only this
externally owned segment from the resource tracker. A crash can leave the segment
or producer claim. Stop **all** clients before removing stale `/dev/shm/cv_daw_shm`
and restarting the session. Never reset ring indexes or recreate a mapping while
another process is attached. IPC uses lock-free C++ atomics under the shared
Linux AArch64/toolchain ABI; producer and consumer must use matching binaries.

## Command and DSP behavior

`AudioCommand` is exactly 16 bytes, alignment 16: limb/event/note/reserved at
bytes 0/1/2/3, value at 4, target frame at 8. `target_frame=0` means the earliest
sample of the next processing block that ingests the command. Future commands
use the engine's absolute rendered-frame domain, not wall-clock nanoseconds or
the DAC clock. Commands scheduled in the past execute at the earliest available
sample and increment `late_commands`. Queue capacity and future heap capacity
are both 256; queue-full drops newest, heap-full counts and drops newest.
Events with the same target frame retain producer order, including preset/mute
changes before note triggers. Future
events farther than ten seconds are rejected. One bounded 256-event ingestion
pass per block prevents producer floods from extending callback work indefinitely.

* NOTE_ON: limb 0..3, MIDI note 0..127, velocity in [0,1]. Allocates a pool voice
  or steals the oldest. Live hits record to the nearest sixteenth-note slot.
* NOTE_OFF: release matching voices for that limb and note.
* MUTE_TOGGLE: toggle limb mute and immediately stop its active voices.
* LOOP_LOCK: value >=0.5 locks, <0.5 unlocks. Locked limbs ignore live NOTE_ON and
  replay their table on sixteenth boundaries. Muting preserves the pattern.
* PRESET_CHANGE: note_or_param selects preset 0..5; existing voices keep their
  selected waveform/sample until release.
* STEM_MUTE: limb MASTER, note_or_param 0..3; toggle that stem mute with smoothing.

Each limb table contains 32 slots and at most eight hits per slot. Live operation
is overdubbing: existing hits persist across two-bar turns; a full slot replaces
its last hit. There is no unbounded event list or callback allocation. The sample
clock and fractional beat phase preserve tempo continuity; smoothed BPM controls
samples per sixteenth (`48000*15/BPM`). Quarter-note beat semantics assume 4/4.
The engine does not perform motion-to-grid early/late quantization; that policy
belongs in the producer, which supplies target frames.

Controls are read atomically once per block, then smoothed per sample: cutoff
20..18000 Hz, send 0..1, bend -12..12 semitones, BPM 60..200, energy 0..1, stem
volumes 0..1. Filter low/band/high outputs are available; the engine uses lowpass.
Reverb uses two fixed cross-coupled comb delays. The rational master saturator
bounds output to [-1,1] with no lookahead; it is not a true-peak oversampling
limiter. NEON handles two stereo frames per limiter iteration on AArch64.

Rubber Band construction, padding/delay compensation, streaming and ratio updates
run on a CPU-0 worker. The DSP thread stores a target BPM and consumes independent
four-stem frames from a preallocated FIFO. The time ratio is source/target BPM.
4096-frame prebuffering (~85 ms), stretch latency and ratio slew limit stem tempo
response; they do not delay live limb voices. Starvation counts missing frames
and emits silence on that bus. Worker starvation/CPU budget under vision load
requires target measurement. Independent stretchers do not guarantee perfect
transient phase coherence across all four stems. See the
[Rubber Band streaming API](https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandStretcher.html).

## Verification and limits

The tests cover queue full/empty/wrap and 200000 concurrent transfers, POSIX
cross-process attachment, command layout, smoothing convergence, filter stability,
finite limiter bounds, exact sample scheduling, 32-slot/two-bar loop playback,
tempo changes, sample note-off, stem mixing and allocation-free rendering. Linux
adds allocator/pthread linker guards (including uncontended mutexes) and a
seccomp child which kills the process if a tested processBlock call makes
any syscall (exit is allowed only to finish the test). This is runtime evidence
for the exercised paths, not a formal WCET proof. A Python/C++ process test runs
the mock sender against the real offline engine and requires acknowledgments.

No board is connected here. Host tests cannot prove Cortex-A35 WCET, ALSA XRUN
behavior, driver timestamps or <35 ms physical motion-to-sound latency. Measure
all of those on STM32MP257F-DK under simultaneous camera/NPU/stem/thermal stress.
The renderer deliberately has no wall-clock calls; use an external profiler or
instrument the callback wrapper, outside processBlock, for timing measurements.
