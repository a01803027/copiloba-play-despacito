# STM32MP257F-DK Complete Deployment Guide (OpenSTLinux 6.6 Scarthgap)

This guide takes you step-by-step from a fresh **STM32MP257F-DK running OpenSTLinux 6.6 (Yocto Scarthgap)** to running the full CV-DAW pipeline:
- **VeriSilicon NPU (1.35 TOPS)** → **MoveNet Lightning INT8** (`X-LINUX-AI` / `stai_mpu`) at 25–40+ FPS
- **Cortex-A35 Core 0** → GStreamer camera + Tier 0 Anti-Wiggle `PoseSanitizer` + Tier 1 Kinematics + Tier 2 Pure-NumPy TCN (`models/dance_tcn.npz`, `~1 ms`, zero TensorFlow on board!) + Python Composer
- **Cortex-A35 Core 1 (`SCHED_FIFO`)** → Lock-free C++20 ALSA Studio DSP Engine (`realtime/build/cv-daw-engine`)

> **You do NOT need to rebuild Yocto!** OpenSTLinux 6.6 Scarthgap supports installing `X-LINUX-AI` and the NPU MoveNet INT8 model directly over Ethernet/Wi-Fi via `apt-get`.

---

## Step 1 — Install X-LINUX-AI & MoveNet INT8 NPU Model on the Board (via `apt-get`)

Connect your STM32MP257F-DK to the internet (Ethernet or Wi-Fi), SSH in (`ssh root@<board-ip>`), and run:

```bash
# 1. Update package lists & install ST's X-LINUX-AI package manager
apt-get update
apt-get install -y x-linux-ai-tool

# 2. Install the NPU-accelerated Python Pose Estimation package (stai_mpu + MoveNet INT8 .nb)
x-linux-ai -i stai-mpu-pose-estimation-python-ovx-npu
```

Verify that Python, NumPy, OpenCV, `stai_mpu`, and the NPU MoveNet model are installed:
```bash
python3 -c "import numpy, cv2, stai_mpu; print('Board Python AI stack OK!')"
find /usr/local/x-linux-ai/pose-estimation -name "*.nb" -o -name "*.tflite"
```
*(You should see a MoveNet `.nb` / `.nbg` or `.tflite` file under `/usr/local/x-linux-ai/pose-estimation/models/`. CV-DAW's `--pose-backend auto` / `stai-npu` auto-discovers this path!)*

> **Note if `apt-get` cannot find `x-linux-ai-tool`**: Make sure the board date/time is synced (`date` — TLS fails if clock is in 1970; run `timedatectl set-ntp true` or `date -s "2026-11-25 12:00:00"`) and verify `/etc/apt/sources.list.d/packages.openstlinux.st.com.list` points to your Scarthgap (`v6.x`) feed.

---

## Step 2 — Clone This Repository Directly on the Board via Git

Since your board has `git` installed, clone your GitHub repo straight onto the board (and pull future Mac tweaks with a single `git pull`):

```bash
cd ~
git clone <YOUR_GITHUB_REPO_URL> cv-daw
cd cv-daw
git checkout cv-daw-integracion
```

Verify the Python recognition layer works immediately with **zero MediaPipe and zero TensorFlow**:
```bash
python3 -m unittest discover -s tests -p "test_phase4_stm32.py" -v
```

---

## Step 3 — Compile the C++20 DSP Engine & IPC Library (`cv-daw-engine` + `libcv_daw_ipc.so`)

You have **two options** — pick whichever is faster for your setup:

### Option A — Cross-Compile on a Linux Host with the ST OpenSTLinux SDK *(Recommended & Fastest)*
```bash
# On your Linux workstation with ST OpenSTLinux SDK installed:
source /opt/st/.../environment-setup-cortexa35-ostl-linux
./scripts/build_stm32mp257.sh

# Copy both built artifacts to ~/cv-daw/realtime/build/ on the board:
ssh root@<board-ip> "mkdir -p ~/cv-daw/realtime/build"
scp realtime/build-stm32/cv-daw-engine realtime/build-stm32/libcv_daw_ipc.so \
    root@<board-ip>:~/cv-daw/realtime/build/
```

### Option B — Compile Natively Directly on the STM32MP257F-DK *(If `g++` & `cmake` are on your image)*
Check if `g++` and `cmake` exist (`which g++ cmake`); if missing, install them via `apt-get install -y g++ cmake make` (if available in your image feed) and run:
```bash
cd ~/cv-daw
./scripts/build_stm32mp257.sh realtime/build
```

What `-DCV_DAW_CORTEX_A35=ON` automatically sets in [realtime/CMakeLists.txt](file:///Users/yucateco/workspace/copiloba-play-despacito/realtime/CMakeLists.txt):
- `-O3 -mcpu=cortex-a35 -ffast-math -ftree-vectorize` (ARMv8-A NEON limiter, `FPCR` denormal flush, `cntvct_el0` cycle profiling)
- `CV_DAW_WITH_RUBBERBAND=OFF` (zero extra Yocto SDK dependencies)
- `CV_DAW_VOICES_PER_LIMB=8` (generous real-time DSP headroom on Cortex-A35)

---

## Step 4 — Verify Cortex-A35 DSP Headroom & ALSA Audio on the Board

### 4.1 Offline DSP Callback Benchmark (No Speaker Needed)
```bash
cd ~/cv-daw
./realtime/build/cv-daw-engine --offline 2000 --demo --profile
```
- Check `callback_max_us`: budget is **`2666 µs`** at `128` frames (`5333 µs` at `--period-frames 256`).
- Confirm `drops=0 late=0 xruns=0`.

### 4.2 Find Your ALSA Audio Device & Play Live Test Tone
```bash
aplay -l
```
Run the built-in 8-second demo through your audio output (headphone jack / HDMI / USB audio DAC):
```bash
# Uses friendly ALSA format auto-conversion (float32 -> S16_LE / S32_LE):
./realtime/build/cv-daw-engine --demo --seconds 8 --alsa-device default

# Or target a specific hardware card from `aplay -l` (e.g. card 0, device 0):
./realtime/build/cv-daw-engine --demo --seconds 8 --alsa-device plughw:0,0 --period-frames 256 --alsa-periods 3
```
*(Non-root users: `cv-daw-engine` automatically falls back cleanly with a warning if `mlockall`/`SCHED_FIFO` lacks privileges. Pass `--strict-alsa --strict-rt` only after configuring `/etc/security/limits.conf` as documented in [realtime/README.md](file:///Users/yucateco/workspace/copiloba-play-despacito/realtime/README.md).)*

---

## Step 5 — Verify Camera & MoveNet INT8 NPU in Moves Lab (Zero Audio)

Find your camera device (`ls /dev/video*` or `v4l2-ctl --list-devices`) and run Moves Lab on the board using the NPU backend:

```bash
# USB webcam (most plug-and-play):
python3 -m composer.moves.lab \
  --pose-backend stai-npu \
  --gstreamer "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=true max-buffers=1"

# If you want to specify the exact MoveNet .nb model path explicitly:
python3 -m composer.moves.lab \
  --pose-backend stai-npu \
  --npu-model /usr/local/x-linux-ai/pose-estimation/models/<your_movenet>.nb \
  --camera 0
```
- Watch the top bar for **`25–40+ FPS`** and test punches, arm raises, X-cross, and leg stomps!
- Running headless (over SSH without HDMI display)? Test with `composer.app ... --no-window`.

---

## Step 6 — Run Full Live Music Instrument & Deploy Custom ML Dance Models

### Launch Full Live CV-DAW on STM32MP257F-DK
```bash
python3 -m composer.app \
  --mode camera \
  --pose-backend stai-npu \
  --gstreamer "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=true max-buffers=1" \
  --style house \
  --launch-engine \
  --alsa-device plughw:0,0 \
  --period-frames 256 \
  --alsa-periods 3 \
  --no-window
```

### Deploying Custom Trained Dance Moves (`models/dance_tcn.npz`)
1. Record clips in Moves Lab on your Mac (or directly on the board with `--record-move`, then `scp` clips to Mac).
2. Train on your Mac with Keras 3:
   ```bash
   .venv-mac/bin/python ml/train_move_net.py \
     --data datasets/moves \
     --out models/dance_tcn.npz \
     --export-tflite-int8 models/dance_tcn_int8.tflite
   ```
3. Copy ONLY **`models/dance_tcn.npz`** (`~62 KB`) to the board (or commit/push & `git pull` on the board):
   ```bash
   scp models/dance_tcn.npz root@<board-ip>:~/cv-daw/models/
   ```
   `composer.app` and `composer.moves.lab` auto-load `models/dance_tcn.npz` automatically in **pure NumPy (`~1 ms` on Cortex-A35 Core 0, zero TensorFlow required on the board)**!
   *(If you ever want to compile the dance TCN onto the NPU with `stedgeai`, `models/dance_tcn_int8.tflite` is already full-integer INT8 quantized & calibrated!)*

---

## Step 7 — Connect the ESP32 BLE Low Energy Wearable LED Shirt (`CV-DAW-SHIRT`)

The STM32MP257F-DK includes an onboard **Murata LBEE5KL1YN Wi-Fi / Bluetooth Low Energy 5.x** module managed by **BlueZ (`bluetoothd`)** on OpenSTLinux 6.6 Scarthgap. CV-DAW talks to the ESP32 LED shirt over **BLE Nordic UART Service (`NUS`)** on an isolated daemon thread (`<= 20`-byte `WriteWithoutResponse` packets, zero pairing dialog, zero impact on the 60 Hz audio/vision loop).

### 7.1 Enable BLE & Install `bleak` on OpenSTLinux 6.6 Scarthgap
```bash
# 1. Ensure the onboard Bluetooth controller is up and powered on:
systemctl start bluetooth
bluetoothctl power on

# 2. Install bleak (pure-Python BlueZ D-Bus client):
python3 -m pip install --break-system-packages bleak
```

### 7.2 Calibrate Physical Shirt Strip Segments Over BLE (No Reflash Needed)
Power on the ESP32 DevKit (`wearable_esp32/playera_led_esp32/playera_led_esp32.ino`, advertising as `CV-DAW-SHIRT`) and verify the **Left Arm (Red) → Chest (Green) → Right Arm (Blue)** pixel boundaries:
```bash
cd ~/cv-daw
python3 -m composer.wearable.cli --mode ble --leds 25,10,25 --test-segments
python3 -m composer.wearable.cli --mode ble --leds 25,10,25 --demo
```

### 7.3 Launch Full Live CV-DAW + NPU Pose + ALSA Audio + BLE LED Shirt
Simply append `--wearable ble --wearable-leds <left>,<chest>,<right>` to your launch command:
```bash
python3 -m composer.app \
  --mode camera \
  --pose-backend stai-npu \
  --gstreamer "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=true max-buffers=1" \
  --style house \
  --launch-engine \
  --alsa-device plughw:0,0 \
  --period-frames 256 \
  --alsa-periods 3 \
  --wearable ble \
  --wearable-leds 25,10,25 \
  --no-window
```

---

## Step 8 — Board Troubleshooting Cheat Sheet

| Issue on STM32MP257F-DK | Fix |
| :--- | :--- |
| `Audio device initialization failed` | Run `aplay -l`, then pass `--alsa-device plughw:<card>,<dev> --period-frames 256 --alsa-periods 3`. Never use raw `hw:0,0` unless your codec natively supports `FLOAT_LE`. |
| `No STAI NPU pose model found` | Run `find /usr/local -name "*.nb" -o -name "*.tflite"` and pass the exact file with `--npu-model <path>`. |
| Camera won't open with `--camera 0` | Check `ls /dev/video*` (STM32 DCMIPP/CSI pipelines often live on `/dev/video0` or `/dev/video1` and require `--gstreamer "v4l2src device=/dev/videoX ! ... ! appsink drop=true max-buffers=1"`). |
| Low MoveNet detection rate in dim room | Improve frontal lighting or lower `min_valid_score` in `StaiMpuPoseTracker` (default `0.28`). |
| Audio underruns (`xruns > 0`) | Use `--period-frames 256 --alsa-periods 3`, verify Core 1 `SCHED_FIFO` limits, and confirm `CV_DAW_VOICES_PER_LIMB=8` in the `audio-ready` startup line. |
| BLE shirt `CV-DAW-SHIRT` not found | Run `bluetoothctl power on` and `bluetoothctl scan le` to confirm `CV-DAW-SHIRT` is advertising; if `hci0` is down on STM32MP257F-DK, run `rfkill unblock bluetooth && hciconfig hci0 up`. |
