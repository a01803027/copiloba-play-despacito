# CV-DAW — Contemporary Ambient

Compositor generativo controlado por danza. La integración actual conecta
OpenCV/MediaPipe en Python con el DSP C++20 mediante memoria compartida.
En macOS reproduce por **CoreAudio**; el destino STM32 conserva **ALSA**.

**Empieza por [MAC_SETUP.md](MAC_SETUP.md).** Incluye instalación, entorno Python,
compilación, prueba sin cámara y aplicación con cámara, paso a paso.

- [Qué hacía el prototipo original](docs/EXISTING_SYSTEM_EXPLAINED.md).
- [Arquitectura y reglas musicales](docs/SYSTEM_ARCHITECTURE.md).
- [Resultados reales y limitaciones](docs/IMPLEMENTATION_REPORT.md).

La aplicación nueva se ejecuta como `python -m composer.app`. El modo `mock`
permite componer sin cámara; `camera` usa la cámara; `video` usa un archivo local.
La opción `--launch-engine` inicia y cierra automáticamente el motor C++.
Consulta la guía antes de ejecutarla: requiere compilar y activar `.venv-mac`.

## Organización

| Carpeta/archivo | Uso |
| --- | --- |
| `BrazoIzquierdo.py`, `BodyTrackingModule.py`, `HandTrackingModule.py` | Prototipo original de Mili, conservado |
| `composer/` | Características, compositor en tres tiempos, puente Python y aplicación integrada |
| `realtime/` | Motor miniaudio, IPC versión 2, DSP y pruebas Mac/Linux |
| `requirements-mac.txt` | Dependencias fijadas para la integración con Python 3.11 |
| `include/cvdaw/`, `src/`, `perception/` | Primera implementación Linux; tiene otro protocolo y no se combina con el compositor nuevo |
| `artifacts/` | WAV y métricas locales generadas por las pruebas; ignorados por Git |

La memoria del sistema integrado se llama `/cv_daw_shm`. La implementación Linux
anterior usa `/cvdaw`. No son intercambiables ni deben ejecutarse como si fueran
una sola cadena.

Las cinco pruebas CTest de `realtime/build-mac` pasaron en el entorno de desarrollo.
También se ejecutaron CoreAudio, cámara y MediaPipe sobre vídeo; el alcance exacto
está en el informe. Sigue pendiente una sesión de danza escuchada por la persona
usuaria y la validación de latencia/XRUNs en STM32. Ningún resultado Mac demuestra
comportamiento determinista en el Cortex-A35.
