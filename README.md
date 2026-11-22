# CV-DAW — Instrumento Musical Controlado por el Cuerpo

Sistema generativo interactivo en tiempo real que traduce el movimiento corporal (MediaPipe Pose Tasks 1.2+ en Python 3.14) en música polifónica mediante un motor DSP C++20 conectado por memoria compartida sin bloqueos (`/cv_daw_shm`).
En macOS reproduce por **CoreAudio**; en Linux ARM64 (Raspberry Pi 5 / STM32) usa **ALSA**.

**Empieza por [MAC_SETUP.md](MAC_SETUP.md) · Guía completa de capacidades y movimientos: [docs/CAPABILITIES.md](docs/CAPABILITIES.md).**

## Ejecución rápida

```sh
source .venv-mac/bin/activate
python -m composer.app --mode camera --style house --launch-engine
```

- Cambia entre los 6 estilos musicales con **`1`–`6`** (o **`N`/`P`**).
- Activa/desactiva la batería con **`Space`/`B`** y bloquea el BPM con **`T`**.
- Diagnóstico en vivo (WAV + JSONL + reporte): `python -m composer.app --mode camera --style house --launch-engine --diagnose 30`.
- Sin cámara: `python -m composer.app --mode mock --style house --launch-engine --seconds 20`.

## Estructura del repositorio

| Carpeta / archivo | Responsabilidad |
| --- | --- |
| `composer/` | Visión pose-only (`tracker.py`, `vision/`), cinemática (`kinematics/`, `features.py`, `rhythm.py`), reconocimiento de 11 movimientos (`moves/`), arreglista (`arrangement.py`, `brain.py`, `procedural.py`), 6 estilos JSON (`styles/presets/`), HUD (`ui/`), diagnóstico (`recorder.py`, `diagnose.py`) y app (`app.py`) |
| `realtime/` | Motor DSP C++20 de estudio (`audio_engine.*`, `dsp_primitives.hpp`), puente IPC lock-free (`ipc_bridge*`), `miniaudio` y pruebas RT |
| `models/` | Modelos MediaPipe `.task` locales (`pose_landmarker_full.task`, `pose_landmarker_lite.task`) |
| `tests/` | Pruebas unitarias de tracking, 11 movimientos, ritmo, estilos, arreglista, diagnóstico y cámara |
| `docs/` | [CAPABILITIES.md](docs/CAPABILITIES.md), arquitectura del sistema y validación en hardware |
| `requirements-mac.txt` | Dependencias Python 3.14+ |
