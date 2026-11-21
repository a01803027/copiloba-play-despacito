# CV-DAW — Instrumento Musical Controlado por el Cuerpo

Sistema generativo interactivo en tiempo real que traduce el movimiento corporal (MediaPipe Pose Tasks 1.2+ en Python 3.14) en música polifónica mediante un motor DSP C++20 conectado por memoria compartida sin bloqueos (`/cv_daw_shm`).
En macOS reproduce por **CoreAudio**; en Linux ARM64 (Raspberry Pi 5 / STM32) usa **ALSA**.

**Empieza por [MAC_SETUP.md](MAC_SETUP.md).**

## Ejecución rápida

```sh
source .venv-mac/bin/activate
python -m composer.app --mode camera --style synthwave --launch-engine
```

- Cambia entre estilos musicales con **`N`/`P`** (o **`1`–`9`,`0`,`-`,`=`**).
- Bloquea o desbloquea el auto-sync de BPM con **`T`**.
- Sin cámara: `python -m composer.app --mode mock --style house --launch-engine --seconds 20`.

## Estructura del repositorio

| Carpeta / archivo | Responsabilidad |
| --- | --- |
| `composer/` | Visión MediaPipe Tasks (`tracker.py`), rasgos corporales (`features.py`, `rhythm.py`), arreglista (`arrangement.py`, `brain.py`, `procedural.py`), estilos JSON (`styles/presets/`), IPC (`ipc.py`) y app (`app.py`) |
| `realtime/` | Motor DSP C++20 (`audio_engine.*`, `dsp_primitives.hpp`), puente IPC lock-free (`ipc_bridge*`), `miniaudio` y pruebas RT |
| `models/` | Modelos MediaPipe `.task` locales (`pose_landmarker_full.task`, `pose_landmarker_lite.task`, `hand_landmarker.task`) |
| `tests/` | Pruebas unitarias de tracking, ritmo, estilos, arreglista y cámara |
| `docs/` | Arquitectura del sistema y plan de validación en hardware |
| `requirements-mac.txt` | Dependencias Python 3.14+ |
