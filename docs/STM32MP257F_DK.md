# Deploying CV-DAW on STM32MP257F-DK (OpenSTLinux / Dual Cortex-A35 + 1.35 TOPS NPU)

CV-DAW's C++20 DSP engine (`realtime/`) and pure-NumPy Move Recognition pipeline (`composer/moves/`) are engineered to run cleanly on the **STM32MP257F-DK**.

---

## 1. Cross-Compiling the C++20 Audio Engine & IPC Library

On a **Linux x86_64 workstation** with the ST OpenSTLinux SDK installed:

```bash
# 1. Source the OpenSTLinux Cortex-A35 SDK
source /opt/st/.../environment-setup-cortexa35-ostl-linux

# 2. Build both cv-daw-engine and libcv_daw_ipc.so
chmod +x scripts/build_stm32mp257.sh
./scripts/build_stm32mp257.sh
```

What `-DCV_DAW_CORTEX_A35=ON` configures automatically in [realtime/CMakeLists.txt](file:///Users/yucateco/workspace/copiloba-play-despacito/realtime/CMakeLists.txt):
- `-O3 -mcpu=cortex-a35 -ffast-math -ftree-vectorize` (ARMv8-A AArch64 NEON limiter + `FPCR` denormal flush + `cntvct_el0` hardware cycle counter).
- **`CV_DAW_WITH_RUBBERBAND=OFF`** by default (zero external library dependencies beyond `pthreads`, `librt`, `libdl`, and runtime `libasound.so.2`).
- **`CV_DAW_VOICES_PER_LIMB=8`** (tuned for generous Cortex-A35 real-time headroom; override with `-DCV_DAW_VOICES_PER_LIMB=12` if desired).

Deploy both binaries to `realtime/build/` on the board:
```bash
ssh root@<board-ip> "mkdir -p ~/cv-daw/realtime/build"
scp realtime/build-stm32/cv-daw-engine realtime/build-stm32/libcv_daw_ipc.so \
    root@<board-ip>:~/cv-daw/realtime/build/
```

---

## 2. Verifying Cortex-A35 DSP Headroom & ALSA Audio on the Board

### Step A — Measure Offline DSP Callback Budget (Zero Audio Hardware Needed)
```bash
cd ~/cv-daw
./realtime/build/cv-daw-engine --offline 2000 --demo --profile
```
- At `128` frames / `48 kHz`, your hard real-time budget is **`2666 µs`** (`5333 µs` if using `--period-frames 256`).
- Verify `callback_max_us` stays comfortably below budget and `drops=0 late=0`.

### Step B — List ALSA Audio Devices & Test Live Playback
```bash
aplay -l
# Friendly mode (auto-converts float32 -> S16_LE / S32_LE on onboard SAI codec or USB DAC):
./realtime/build/cv-daw-engine --demo --seconds 8 --alsa-device default
# Or target a specific hardware card with plug conversion:
./realtime/build/cv-daw-engine --demo --seconds 8 --alsa-device plughw:0,0 --period-frames 256 --alsa-periods 3
```

> **Note on RT Privileges**: By default on Linux, `cv-daw-engine` attempts `mlockall()` + `SCHED_FIFO` priority `90` (CPU Core 1) and prints a friendly warning (continuing in normal priority) if run without root/RT limits. Pass `--strict-alsa --strict-rt` for strict hardware qualification (after configuring `/etc/security/limits.conf` as described in [realtime/README.md](file:///Users/yucateco/workspace/copiloba-play-despacito/realtime/README.md)).

---

## 3. Running Vision & Move Recognition on STM32MP257F-DK

### Zero TensorFlow Required on the Board!
- Train your custom dance recognizer on macOS/PC with Keras 3 (`ml/train_move_net.py`) and copy only the tiny **`models/dance_tcn.npz`** (`~56 KB`) to the board.
- On the board, **Tier 0 (`PoseSanitizer`) + Tier 1 (`BodyMoveRecognizer`) + Tier 2 (`NumpyTcnModel` / `StreamingDtwMatcher`)** require **ONLY Python 3 + NumPy** (`< 1 ms/frame` on Cortex-A35)!

### Connecting the STM32MP257F-DK 1.35 TOPS NPU (X-LINUX-AI)
Running CPU MediaPipe Pose on Cortex-A35 is slow (~3–6 FPS). Instead, run MoveNet or YOLOv8-Pose on the onboard **VeriSilicon NPU (`X-LINUX-AI` / `TIM-VX`)** and convert the 17 COCO keypoints directly using [coco17_to_mediapipe33()](file:///Users/yucateco/workspace/copiloba-play-despacito/composer/vision/pose_backend.py):

```python
from composer.vision import coco17_to_mediapipe33
from composer.features import FeatureExtractor

extractor = FeatureExtractor(tcn_model_path='models/dance_tcn.npz')
points_33 = coco17_to_mediapipe33(movenet_kp17, yx_order=True)
features = extractor.update(points_33, timestamp_s, aspect=frame_h / frame_w)
```

### Launching Full Audio + GStreamer Camera on the Board
```bash
python3 -m composer.app \
  --mode camera \
  --gstreamer "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=true max-buffers=1" \
  --style house \
  --launch-engine \
  --alsa-device plughw:0,0 \
  --period-frames 256 \
  --alsa-periods 3 \
  --no-window
```
