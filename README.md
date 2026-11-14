# CV-DAW

**New to the project? Read [START_HERE.md](START_HERE.md) first.** It explains what
was built and walks you through the first Mac test without assuming C++ knowledge.

The dedicated miniaudio/Cortex-A35 Modules 4–5 implementation and its Python IPC
harness are in [realtime/README.md](realtime/README.md). It uses `/cv_daw_shm` and
a separate command ABI from the initial pipeline below.

A C++20 real-time audio core with a non-real-time Python camera frontend for
STM32MP257F-DK/OpenSTLinux. The existing Python/pygame experiments are preserved.
This is an initial implementation, not a board-qualified release: no target
hardware, compiled NPU models, trained groove model, or song stems were available.

## Build and test

Host (macOS or Linux; tests need permission to create POSIX shared memory):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
```

Linux board or OpenSTLinux SDK, with ALSA, Rubber Band >=3.0 and libsndfile
headers/libraries in the sysroot:

```sh
# For cross-compilation, first source the SDK's environment-setup-* script.
cmake -S . -B build-target -DCVDAW_TARGET=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-target --parallel 2
```

Install the OpenSTLinux equivalents of `alsa-lib-dev`, `rubberband-dev`,
`libsndfile-dev`, `v4l-utils`, Python NumPy, PyGObject, GStreamer core/base/good
plugins, and a board-compatible MediaPipe package or custom inference backend.
`requirements.txt` is retained for the original desktop prototype; its wheels
are not a target provisioning recipe. `.github/workflows/build.yml` compiles all
Linux backends and runs the portable tests. It has not yet run in this workspace.

## Run on the board

Run all processes as the same user. Grant that user audio/video device access,
`memlock` and RT scheduling privileges (see systemd templates in `deploy/`). Use
a PREEMPT_RT kernel and configure CPU isolation as described in
[the target checklist](docs/target-validation.md). Start audio first:

```sh
./build/cvdaw-audio 128
# Separate terminal; use camera-specific manual exposure units:
python3 perception/run.py --exposure 50
# Optional third terminal; four aligned 48 kHz stereo WAV files, original BPM:
./build/cvdaw-stems 120 drums.wav bass.wav harmony.wav vocals.wav
```

These examples assume the board binaries are installed in `build/`. For systemd,
install the repository at `/opt/cvdaw`, create the `cvdaw` user, adapt the camera
exposure, and install the two service templates. They do not configure the kernel,
IRQ routing, sensor, or NPU automatically. The brain retries audio startup for
up to five seconds. Audio rejects unsupported PCM geometry instead of resampling.
It tries native FLOAT_LE, then S32_LE, at `hw:0,0`, 48 kHz, two channels, two
128- or 256-frame periods.

The creator uses exclusive `/cvdaw` shared memory ownership. Shut down all clients
before restarting audio. A crash can leave `/dev/shm/cvdaw`; after verifying all
CV-DAW processes have stopped, explicitly remove that object before restarting.
Never unlink/recreate it while clients remain attached. One brain and one stem
worker are allowed per session; restart the session after a worker crash.

## Implemented modules

* `motion.hpp`: One Euro smoothing, aspect-corrected torso normalization, four
  wrist/ankle velocity and acceleration streams, deceleration onsets, 80 ms
  debounce, polar controls; resets after >100 ms tracking gaps.
* `perception/run.py`: bounded latest-frame GStreamer capture at negotiated
  60 FPS, manual V4L2 exposure setup, capture PTS mapping, CPU MediaPipe reference
  pose/hands, three-frame posture confirmation, optional asynchronous temporal
  TFLite inference. Tracking loss expires the audio heartbeat after 250 ms.
* `music.hpp`: per-limb LIVE/MUTED/LOOPING modes, C minor pentatonic pitch,
  32-slot/two-bar overdub recorder, sample-frame loop scheduling, preset changes (arms start as sine voices, legs as drums),
  tempo phase continuity, configurable hybrid quantization. Each slot stores the
  most recent note for that limb. Pinch plays the recorded pattern; open resumes
  live overdubbing, fist mutes, peace cycles sine/saw/drum presets.
* `ipc.hpp`: ABI-tagged POSIX shared memory, cache-line-separated bounded SPSC
  queues, atomic controls/metrics, bounded audio-clock snapshot retries. Both
  processes must use the same ABI/toolchain and lock-free ARM64 atomic operations.
* `dsp.hpp`, `audio.cpp`: bounded scheduling, eight voices per limb, startup-built
  wavetables and RAM kick sample, per-limb state-variable lowpass, stereo stem
  bus, tanh saturation, allocation-free render path, affinity, `mlockall`, FIFO
  priority 80, exact ALSA parameter checking and partial-write handling. An XRUN
  increments telemetry and stops the engine; it is never hidden by recovery.
* `stems.cpp`: four preloaded aligned song stems stretched by Rubber Band R2,
  start-padding/delay compensation and mixed into a prebuffered SPSC FIFO. It
  runs on Core 0 so variable-cost stretch processing stays outside live DSP.

## Policies and remaining integration

The strict default waits at most 8 ms for an early grid tick. `--full-quantize`
implements the requested 40 ms early window, which by itself contradicts a strict
<35 ms motion-to-sound deadline. Late hits within 35 ms play immediately and are
recorded at the previous grid line. Events outside the windows play immediately
and record at the next grid line. Neither policy can undo camera/inference delay.
At 60 FPS, sampled deceleration detection may require multiple observations.
A One Euro filter has nonzero delay, including during fast movement.

The slow fallback is a bounded hip-bounce phase/period tracker plus smoothed
kinetic energy. It does **not** claim trained dance-style inference. Pass
`--temporal-model model.tflite --delegate /path/to/vendor_delegate.so` to enable
an actual trained model. Input is `[1,30,12]`, chronological four-limb torso
velocities, single output `[energy,p_style0,p_style1,p_style2,p_style3]`; both
float32 and per-tensor int8/uint8 I/O are supported. Model input normalization,
training labels and quantization calibration must match this contract. Results
older than 500 ms are discarded. Groove outputs are published as four atomic
controls; they do not automatically change instrument presets.

For NPU pose/hands, use `--backend your_package.backend`. The module exports
`create_backend()` returning an object with `infer(rgb)` and `close()`. `infer`
returns `(pose, hands)`: pose is 33 `(x,y,z,visibility)` tuples in MediaPipe
normalized image coordinates, hands maps `'Left'`/`'Right'` to 21 `(x,y,z)` points
with MediaPipe selfie handedness. The frontend corrects unmirrored camera labels;
use `--mirrored` if input is already mirrored. The backend owns detector/ROI
tracking, int8 preprocessing/dequantization, delegate creation and landmark
postprocessing. Vendor model binaries and their tensor/delegate contracts are
required to implement and validate this adapter. CPU MediaPipe inference does
not implicitly use the NPU. The optional temporal worker alone loads the supplied
TFLite delegate.

The stem FIFO prebuffers 4096 frames (~85 ms); Rubber Band adds processing delay.
This latency is separate from live notes and limits tempo-control responsiveness.
Core 0 feasibility with camera + inference + four stretchers is unproven. Stereo
stems must already share duration and aligned downbeats; no beat-phase alignment
to the dancer is currently performed. The simple saw wavetable aliases at high
pitches. The master uses tanh with no lookahead and is not a true-peak limiter.
The default RAM kick is synthetic; `Engine::load_drum` accepts a mono sample before
real-time startup, but there is no runtime sample-file loader/UI yet.

See [target validation](docs/target-validation.md) for measurement and acceptance.
ALSA configuration follows its [PCM documentation](https://www.alsa-project.org/alsa-doc/alsa-lib/group___p_c_m.html).
Rubber Band streaming and startup compensation follow its
[API contract](https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandStretcher.html).
