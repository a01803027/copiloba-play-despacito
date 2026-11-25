# CV-DAW — Instrumento Musical Controlado por el Cuerpo

Sistema generativo interactivo en tiempo real que traduce el movimiento corporal (MediaPipe Pose Tasks 1.2+ en Python 3.13) en música polifónica mediante un motor DSP C++20 conectado por memoria compartida sin bloqueos (`/cv_daw_shm`).
En macOS reproduce por **CoreAudio**; en Linux ARM64 (Raspberry Pi 5 / STM32) usa **ALSA**.

**Empieza por [MAC_SETUP.md](MAC_SETUP.md) · Capacidades: [docs/CAPABILITIES.md](docs/CAPABILITIES.md) · Guía del Moves Lab y ML: [docs/MOVES_LAB_GUIDE.md](docs/MOVES_LAB_GUIDE.md) · Guía STM32MP257F-DK: [docs/STM32MP257F_DK.md](docs/STM32MP257F_DK.md).**

> **IMPORTANT — STM32MP257F-DK Compatibility Mandate (All Future Tweaks)**
> macOS is the development & training workstation; **STM32MP257F-DK (OpenSTLinux 6.6 Scarthgap, dual Cortex-A35 + 1.35 TOPS VeriSilicon NPU)** is the embedded target. Every change or tweak to this repository **MUST** preserve STM32MP257F-DK compatibility:
> 1. **15-Joint COCO-17 Intersection Only**: All kinematics, moves, and ML features must only depend on the 15 joints shared between MediaPipe-33 and COCO-17 MoveNet (`nose, ears, shoulders, elbows, wrists, hips, knees, ankles`) in **2D `(x, y)`** (never rely on MediaPipe-only `z` depth or finger/face landmarks).
> 2. **Zero MediaPipe / TensorFlow Imports in Board Runtime**: Runtime code (`composer/moves/`, `composer/kinematics/`, `composer/brain.py`, `composer/vision/pose_backend.py`) must run with **Python 3 + NumPy + OpenCV + `stai_mpu`** only. Keep Keras/TensorFlow strictly inside offline training (`ml/`), exporting to pure-NumPy `.npz` (`< 100 KB`) or INT8 `.tflite`.
> 3. **Pluggable Vision & Camera**: Always go through `create_pose_backend()` (`mediapipe` on Mac, `stai-npu` MoveNet INT8 on STM32) and keep `--gstreamer` / `--no-window` working.
> 4. **Real-Time C++20 & ALSA Portability**: Zero heap allocations, locks, or syscalls inside `AudioEngine::processBlock()`, preserve ABI v3 `SharedMemory`, keep `CV_DAW_VOICES_PER_LIMB=8` and `CV_DAW_WITH_RUBBERBAND=OFF` defaults on `-DCV_DAW_CORTEX_A35=ON`, and keep flexible ALSA flags (`--alsa-device`, `--period-frames`, non-root fallback).

## Ejecución rápida

```sh
source .venv-mac/bin/activate
python -m composer.app --mode camera --style house --launch-engine
```

- Cambia entre los 6 estilos musicales con **`1`–`6`** (o **`N`/`P`**).
- Activa/desactiva la batería con **`Space`/`B`** y bloquea el BPM con **`T`**.
- Laboratorio de movimientos (sin audio, para probar y depurar detección): `python -m composer.moves.lab` (o `--replay artifacts/diag_*.jsonl`).
- Diagnóstico en vivo (WAV + JSONL + reporte): `python -m composer.app --mode camera --style house --launch-engine --diagnose 30`.
- Sin cámara: `python -m composer.app --mode mock --style house --launch-engine --seconds 20`.

## Estructura del repositorio

| Carpeta / archivo | Responsabilidad |
| --- | --- |
| `composer/` | Visión pose-only (`tracker.py`, `vision/`), cinemática (`kinematics/`, `features.py`, `rhythm.py`), reconocimiento de 11 movimientos (`moves/`), arreglista (`arrangement.py`, `brain.py`, `procedural.py`), 6 estilos JSON (`styles/presets/`), HUD (`ui/`), diagnóstico (`recorder.py`, `diagnose.py`) y app (`app.py`) |
| `realtime/` | Motor DSP C++20 de estudio (`audio_engine.*`, `dsp_primitives.hpp`), puente IPC lock-free (`ipc_bridge*`), `miniaudio` y pruebas RT |
| `models/` | Modelos MediaPipe `.task` locales (`pose_landmarker_full.task`, `pose_landmarker_lite.task`) |
| `tests/` | Pruebas unitarias de tracking, 11 movimientos, ritmo, estilos, arreglista, diagnóstico y cámara |
| `docs/` | [CAPABILITIES.md](docs/CAPABILITIES.md), [STM32MP257F_DK.md](docs/STM32MP257F_DK.md), arquitectura del sistema y validación en hardware |
| `ml/` | Entrenamiento Keras 3 1D-TCN y exportación a `.npz` pure-NumPy (`<0.5 ms`) / `.tflite` (`train_move_net.py`) |
| `scripts/` | Compilación cruzada OpenSTLinux para STM32MP257F-DK (`build_stm32mp257.sh`) |
| `requirements-mac.txt` | Dependencias Python 3.13+ (MediaPipe + OpenCV + TensorFlow/Keras 3) |
