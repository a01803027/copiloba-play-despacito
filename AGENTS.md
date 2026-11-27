# AGENTS.md — Multi-Agent & Multi-Branch Engineering Guide for CV-DAW

Read this file **before** planning or editing code on any branch. CV-DAW is a **hard real-time** full-body interactive musical instrument and wearable LED system developed on **macOS (Apple Silicon)** and deployed onto the **STM32MP257F-DK** embedded Linux/NPU platform + **ESP32 BLE Wearable Shirt**.

Every agent working in parallel across branches **must** uphold the hardware, real-time, and interface contracts below so branches merge cleanly and never break embedded deployment.

---

## 1. Dual-Platform Architecture at a Glance

| Layer | macOS Dev / Training Host | STM32MP257F-DK Embedded Target | Interface Contract |
| :--- | :--- | :--- | :--- |
| **Camera & Vision** | Webcam + MediaPipe Pose `.task` (`lite` @ `320px`) on `AsyncPoseWorker` | V4L2 / CSI `--gstreamer` + **VeriSilicon 1.35 TOPS NPU** MoveNet INT8 (`StaiMpuPoseTracker` via `stai_mpu`) | `create_pose_backend()` $\rightarrow$ 33-point `(x, y, z, vis)` list (15 COCO-shared joints populated) |
| **Tier 0–2 Move Engine** | Python `3.13` (`numpy`) | OpenSTLinux `6.6` Scarthgap Python 3 (`numpy` only — **zero TF/MediaPipe**) | `PoseSanitizer` $\rightarrow$ `24-D` `PoseFrame` $\rightarrow$ Tier 2 (`DTW`/`TCN`) + Tier 1 (`BodyMoveRecognizer`) $\rightarrow$ `MoveArbiter` $\rightarrow$ `Features` |
| **Composer & Arranger** | `composer/brain.py` (`60 Hz` loop) | Cortex-A35 Core 0 (`60 Hz` loop) | `Composer.update()` & `Composer.controls()` $\rightarrow$ `Event`s + `ControlSnapshot` |
| **Real-Time DSP Engine** | `realtime/build-mac/cv-daw-engine` (CoreAudio) | `realtime/build/cv-daw-engine` (ALSA, Cortex-A35 Core 1 `SCHED_FIFO`) | Lock-free POSIX `/cv_daw_shm` (**ABI v3**, `4544` bytes, SPSC ring + `SeqLock`) |
| **Wearable LED Shirt** | `BleNusTransport` (CoreBluetooth) / `SerialTransport` | `BleNusTransport` (BlueZ D-Bus over onboard Murata BLE) | Daemon thread $\rightarrow$ `<= 20`-byte `\n` ASCII over BLE NUS (`CV-DAW-SHIRT`) $\rightarrow$ ESP32 `GPIO 13` WS2812B |

---

## 2. Non-Negotiable STM32MP257F-DK Compatibility Rules

1. **15-Joint 2D `(x, y)` Intersection ONLY**:
   - All kinematics, move detectors, and ML features **must only use the 15 joints shared between MediaPipe-33 and COCO-17 MoveNet**: `nose (0)`, `ears (7, 8)`, `shoulders (11, 12)`, `elbows (13, 14)`, `wrists (15, 16)`, `hips (23, 24)`, `knees (25, 26)`, `ankles (27, 28)` (defined in `CANONICAL_JOINTS` in `composer/moves/pose_frame.py`).
   - **NEVER** use monocular `z` depth for velocities, angles, or ML features (MediaPipe `z` has 10× noise spikes, and COCO-17 MoveNet has no `z` at all).
   - **NEVER** depend on finger, toe, or face-mesh indices (`1..6, 9..10, 17..22, 29..32`).
   - Always keep graceful off-screen leg/knee fallbacks (`legs_visible` check + standing-neutral fallback in `PoseSanitizer`).

2. **Zero Heavy Imports in Runtime Code (Strict Lazy Imports)**:
   - `composer/kinematics/`, `composer/moves/`, `composer/brain.py`, `composer/arrangement.py`, `composer/procedural.py`, `composer/styles/`, `composer/vision/pose_backend.py`, and `composer/wearable/` **must import cleanly using ONLY Python stdlib + `numpy` (+ `cv2` where needed)**.
   - Keep `mediapipe`, `stai_mpu`, `bleak`, and `serial` **lazy-imported** inside their specific backend/transport classes so the board runs even when optional packages are absent.
   - Keep `tensorflow` and `keras` **strictly confined to `ml/`** (offline training/export on Mac). On-board Tier 2 inference runs in pure NumPy (`composer/moves/tcn_runtime.py`, `< 1.0 ms` on Cortex-A35, `< 100 KB` `.npz`) or INT8 `.tflite` (`--export-tflite-int8`).

3. **Preserve Embedded Camera, Headless & ALSA Flags**:
   - Never break `--pose-backend {auto,mediapipe,stai-npu,movenet-tflite}`, `--npu-model`, `--gstreamer`, `--no-window`, `--alsa-device`, `--period-frames`, or `--alsa-periods` in `composer/app.py` and `composer/moves/lab.py`.
   - Keep `-DCV_DAW_CORTEX_A35=ON` defaults in `realtime/CMakeLists.txt` (`CV_DAW_VOICES_PER_LIMB=8`, `CV_DAW_WITH_RUBBERBAND=OFF`, `-O3 -mcpu=cortex-a35 -ffast-math -ftree-vectorize`) and `scripts/build_stm32mp257.sh` working.

---

## 3. Hard Real-Time & Low-Latency Invariants

### 3.1 C++20 Audio Thread (`realtime/`)
- **Callback Budget**: `48 kHz` stereo `float32`, `128` frames = **`2666 µs` max** (`5333 µs` at `--period-frames 256` on STM32).
- **Forbidden Inside `AudioEngine::processBlock()` & `data_callback`**:
  - **ZERO** heap allocations (`new`, `delete`, `malloc`, `free`, `std::vector::push_back`, `std::string` resizing).
  - **ZERO** locks/mutexes (`std::mutex`, `pthread_mutex_*`) or condition variables.
  - **ZERO** syscalls, sleeps, or console/file I/O (`printf`, `std::cout`, `write`).
  - *(Note: Linux CTest links `-Wl,--wrap=malloc/free/pthread_mutex_*` via `realtime/tests/rt_guards.cpp` and will fail immediately if violated!)*
- **Audio Quality**: Pre-allocate all voice/FX state; ramp gain/filter/sidechain changes smoothly to prevent clicks/pops; keep `< 120 Hz` mono-sub crossover and `190 ms` cross-bus kick arbitration (`KICK_MIN_GAP_FRAMES`); **never modify vendored `realtime/third_party/miniaudio.h`**.

### 3.2 Shared-Memory IPC ABI (`realtime/ipc_bridge.hpp` $\leftrightarrow$ `composer/ipc.py`)
- POSIX `/cv_daw_shm` uses **ABI v3** (`sizeof(SharedMemory) == 4544` bytes).
- **Frozen Across Parallel Branches**: Do **NOT** alter `Command`, `ControlSnapshot`, or `SharedMemory` struct layouts unless explicitly assigned an IPC ABI task. Any ABI change must atomically update `ABI_VERSION`, `ipc_bridge.hpp`, `ipc_bridge_c.cpp`, `composer/ipc.py`, and `realtime/tests/rt_tests.cpp`.

### 3.3 Non-Blocking 60 Hz Python Loop & BLE Wearable Link
- `composer/app.py`'s main loop runs at `60 Hz` to schedule musical ticks with low jitter. **Never perform blocking camera reads, ML training, disk flushes, or BLE/Serial writes on the main thread.**
- Camera (`LatestCamera`), pose inference (`AsyncPoseWorker`), and wearable transport (`BleNusTransport` / `SerialTransport`) must stay on daemon threads with lock-free/non-blocking queues.
- **BLE NUS Packet Budget (`composer/wearable/protocol.py` & `wearable_esp32/`)**:
  - Every wearable command **must** end in `\n` and encode to **`<= 20` ASCII bytes** (single BLE GATT `WriteWithoutResponse` packet without ATT fragmentation).
  - Keep `FastLED.setMaxPowerInVoltsAndMilliamps(5, 1600)` in `wearable_esp32/playera_led_esp32/playera_led_esp32.ino` to protect the `5V >= 2A` power bank from brownout resets.

---

## 4. Musical & Recognition Design Philosophy (Do Not Regress!)

When tuning moves or music generation, preserve the established body-to-band architecture:
1. **Lower Body = Rhythm Engine (`composer/kinematics/body.py`, `composer/rhythm.py`, `composer/brain.py`)**:
   - Legs, feet, and hips carry **62% of global energy**, **80% of drum/bass drive**, and **master BPM priority** (`legs > hip > head > shoulders > arms`).
   - **Planted Feet = Minimal/Sparse Rhythm**: When `legs_visible=True` and legs/hips stay still (`lower_drive < 0.14`), drums & bass **must stay minimal/sparse** (soft downbeat pulse + sustained root bass, zero busy hi-hats/snares/fills) while upper-body movements still play lush chords, voicings, and lead melodies.
   - Two-tier leg moves: sensitive `FOOT_STEP` (`lift 0.12..0.41`) for pocket groove steps + high asymmetric `KNEE_STOMP` (`lift >= 0.42`) for heavy kick/snare hits.
2. **Upper Body = Harmony, Lead & Dual-Channel Arbitrated Moves (`composer/moves/arbiter.py`, `recognizer.py`)**:
   - `MoveArbiter` separates **`upper`** and **`lower`** channels (max 1 move per channel per frame), enforces rolling `0.55–0.75 s` temporal spotlight locks, and activates the **Tier 2 Intent Shield** (`tier2_confidence >= 0.48` blocks opportunistic Tier 1 punches/bursts while a custom move is forming).
   - Three-zone punches: casual dance deadzone (`reach_vel < 2.85`), beat-quantized `LIGHT_PUNCH` percussion (`2.85..4.59`, zero synth stabs), and exaggerated `HARD_PUNCH` power stabs (`>= 4.60`).
3. **Custom Tier 2 Moves = Multi-Bar Groove & Section Director (`composer/brain.py`)**:
   - Any move kind outside `TIER1_KNOWN_MOVES` (`is_custom_move(kind) == True`) automatically acts as a **3-bar (`48`-tick) Groove & Section Director** (`CHORUS`/`DROP` + 3 harmonic/rhythmic groove flavors) with continuous pre-completion `tier2_anticipation` (`0..1`).

---

## 5. Parallel Branch Ownership & Conflict-Free Extension Map

When multiple agents work on separate branches simultaneously, stay inside your assigned module boundary and use the repository's **zero-conflict extension hooks**:

| Workstream / Agent Role | Primary Files Owned | Frozen Boundary / How to Extend Without Conflicts |
| :--- | :--- | :--- |
| **A. Custom Moves & ML Training** | `ml/train_move_net.py`, `composer/moves/dtw.py`, `composer/moves/tcn_runtime.py`, `composer/moves/lab_diagnose.py` | Keep `FEATURE_DIM = 24` (`PoseFrame.feature_vector`) and `.npz` weight keys unchanged. Exported models drop into `models/dance_tcn.npz`. **Never delete user clips in `datasets/moves/`**. |
| **B. Tier 1 Kinematics & Arbiter** | `composer/moves/sanitizer.py`, `composer/moves/recognizer.py`, `composer/moves/arbiter.py`, `composer/kinematics/body.py` | Keep existing `Features` dataclass field names/defaults backward-compatible (add new fields with defaults at the end). |
| **C. Musical Styles & Composer** | `composer/styles/presets/*.json`, `composer/arrangement.py`, `composer/brain.py`, `composer/procedural.py` | **New genres**: simply add `<style>.json` in `composer/styles/presets/` (`StyleRegistry` auto-discovers it!). Keep limb bus IDs (`0`=Chords, `1`=Lead, `2`=Bass, `3`=Drums) and `RISER_MIDI = 120`. |
| **D. C++20 DSP & ALSA Engine** | `realtime/dsp_primitives.hpp`, `realtime/audio_engine.hpp`, `realtime/audio_engine.cpp`, `realtime/CMakeLists.txt` | Preserve preset IDs (`8..17`), `SharedMemory` ABI v3, Cortex-A35 voice/math flags, and C++20 readability formatting (`2`-space/clean clang-format style, never format `miniaudio.h`). |
| **E. STM32 Vision & Camera** | `composer/vision/pose_backend.py`, `composer/vision/camera.py`, `scripts/build_stm32mp257.sh`, `docs/STM32MP257F_DK.md` | Preserve `PoseBackend` protocol (`process(bgr, ts) -> list[33]`, `draw()`, `close()`) and `coco17_to_mediapipe33()`. |
| **F. Wearable ESP32 & BLE** | `composer/wearable/*`, `wearable_esp32/*`, `tests/test_wearable.py` | Keep `create_shirt_orchestrator()` signature and `<= 20`-byte `\n` NUS protocol backward-compatible (`L`, `R`, `OFF`, `S`, `FX`, `CFG`, `TEST`). |
| **G. UI / HUD & Moves Lab** | `composer/ui/*`, `composer/moves/lab.py` | Read-only consumer of `Features` and `Composer`; never mutate `Composer` state outside `handle_window_key()`. |

> **Shared Hotspot Warning**: `composer/app.py`, `docs/CAPABILITIES.md`, and `README.md` are touched by multiple workstreams. Keep edits in `composer/app.py` minimal (delegate logic to subpackages like `composer/wearable/` or `composer/vision/`) so Git merges stay trivial.

---

## 6. Mandatory Pre-Merge Verification Checklist

Before marking work complete on any branch, run and pass all applicable checks:

1. **Python Unit Test Suite (Fast, Sandboxed — Required for ALL Python/ML/Wearable changes)**:
   ```bash
   MPLCONFIGDIR=/tmp/mplcache .venv-mac/bin/python -m unittest discover -s tests -v
   ```
   - Verify **zero regressions** across tracking, sanitizer/arbiter, Tier 1 & Tier 2 moves, lower-body rhythm, composer, STM32 NPU adapter, and wearable BLE protocol (`<= 20` bytes).
   - Verify no top-level `mediapipe` / `tensorflow` / `bleak` import leaked into core runtime:
     ```bash
     .venv-mac/bin/python -m unittest tests/test_phase4_stm32.py -v
     ```

2. **C++20 Build & Full Real-Time IPC CTest Suite (Required for ANY `realtime/`, `ipc`, or `brain.py` change)**:
   ```bash
   /opt/homebrew/bin/cmake --build realtime/build-mac --parallel 4
   /opt/homebrew/bin/ctest --test-dir realtime/build-mac --output-on-failure
   ```
   *(Note: On macOS sandboxed agent environments, `ctest` opens POSIX `/cv_daw_shm` and may require running outside the sandbox if `shm_open` returns `Operation not permitted`.)*

3. **Living Documentation Update**:
   - Whenever moves, thresholds, musical styles, CLI flags, wearable FX, or DSP features change, update **[docs/CAPABILITIES.md](docs/CAPABILITIES.md)** (and **[docs/STM32MP257F_DK.md](docs/STM32MP257F_DK.md)** / **[wearable_esp32/README.md](wearable_esp32/README.md)** if deployment or hardware steps changed) in the same branch.
