# Informe de implementación y verificación

## Estado y alcance

Trabajo realizado en **tuve-un-sueno**, desde el commit auditado `a84344d`.
El árbol estaba limpio al comenzar. No se cambió a main, no se hizo pull, merge,
reset, commit ni push. Los cambios quedan locales para revisión.

Se implementó una cadena ejecutable en Mac: cámara/vídeo/mock → características
→ compositor → memoria compartida → DSP C++ → CoreAudio. Se comprobó software y
acceso al dispositivo. **Falta que una persona baile delante de la cámara y
confirme auditivamente la respuesta musical.** No se presenta esa validación
pendiente como un éxito ya conseguido.

## Qué se conservó de Mili

No se modificaron `BrazoIzquierdo.py`, `BodyTrackingModule.py`,
`HandTrackingModule.py`, `requirements.txt` ni los WAV originales.
La aplicación nueva importa directamente su `BodyDetector`, llama `findBody()` y
lee `results.pose_landmarks` para conservar coordenadas y confianza originales.
No usa `findPosition()` para componer porque ese método pierde profundidad y
confianza al convertir todo a píxeles enteros.

La auditoría encontró el índice 9 de YOLO usado en un detector MediaPipe. Se
explicó en el documento de auditoría sin corregir silenciosamente la prueba
original. El compositor usa índices MediaPipe 15/16/27/28 para muñecas/tobillos.

## Archivos creados

| Archivo | Para qué sirve |
| --- | --- |
| `composer/__init__.py` | Paquete Python de la nueva integración |
| `composer/features.py` | Rasgos físicos normalizados, suavizado y gestión de oclusiones |
| `composer/brain.py` | Tres capas temporales; escala, tríadas, motivos y variación con semilla |
| `composer/ipc.py` | Cliente del puente nativo, controles, confirmaciones y apagado suave |
| `composer/app.py` | Modos cámara/vídeo/mock, ventana, inicio del motor, cierre y métricas |
| `requirements-mac.txt` | Dependencias probadas con Python 3.11, sin alterar el entorno anterior |
| `realtime/hardware_clock.hpp` | Contador ARM accesible sin llamadas al sistema para perfilar |
| `realtime/wav_writer.hpp` | Exportador PCM16 estéreo llamado solo desde el modo offline |
| `realtime/tests/test_wav.py` | Verifica WAV real generado por C++ |
| `realtime/tests/test_composer_ipc.py` | Verifica compositor Python → IPC → DSP → WAV |
| `tests/test_composer.py` | Reglas musicales, escala, repetibilidad, densidad, tracking y controles |
| `tools/check_mili.py` | Ejecuta el main original con vídeo real y adaptadores de hardware silenciosos |
| `MAC_SETUP.md` | Instalación y ejecución desde cero, con cada comando explicado |
| `docs/EXISTING_SYSTEM_EXPLAINED.md` | Auditoría previa a modificar código |
| `docs/SYSTEM_ARCHITECTURE.md` | Arquitectura real, composición y restricciones |
| `docs/IMPLEMENTATION_REPORT.md` | Este informe |
| `docs/TEST_RESULTS.json` | Copia de resultados numéricos, sin imágenes ni vídeo de cámara |

Los entornos `.venv-mac` y los resultados `artifacts/` se generaron localmente y
están ignorados por Git. Ahí están los WAV y JSON de las sesiones de prueba.

## Archivos modificados

| Archivo | Cambio y motivo |
| --- | --- |
| `realtime/audio_engine.cpp` | CoreAudio en Apple, ALSA en Linux, audición limitada, WAV offline, perfilado, duración y mensajes de inicio |
| `realtime/audio_engine.hpp` | Ganancias independientes suavizadas: expresividad continua sin crear una nota por imagen |
| `realtime/ipc_bridge.hpp` | ABI versión 2; ganancias y telemetría usando espacio antes reservado dentro de los bloques alineados |
| `realtime/ipc_bridge_c.cpp` | Grupo de control 6 para ganancias y lectura de tiempos del callback |
| `realtime/rt_startup.hpp` | La función de bloqueo de memoria devuelve no-soportado fuera de Linux; macOS no pide políticas Linux |
| `realtime/CMakeLists.txt` | Build portable y cinco pruebas, usando el intérprete seleccionado |
| `realtime/tests/rt_tests.cpp` | Añade comprobación de silencio por ganancias sin asignaciones |
| `.gitignore` | Ignora entorno Mac y resultados locales |
| `README.md`, `START_HERE.md`, `realtime/README.md` | Enlazan la guía actual y eliminan la descripción obsoleta de Mac como solo offline |

No se sustituyó el otro motor de `src/` ni su IPC `/cvdaw`. La integración usa
`realtime/` y `/cv_daw_shm`. El cambio de ABI exige recompilar ejecutable y puente
juntos; adjuntar una versión anterior se rechaza.

## Pruebas ejecutadas, en orden A–F

### A — Prototipo de Mili

La primera importación en `.venv` falló: faltaba MediaPipe y ese entorno usa
Python 3.14. Se creó `.venv-mac` con Python 3.11 y dependencias separadas.

`tools/check_mili.py` ejecutó el `main()` **original** con 30 cuadros de
`brazo.mp4`, inferencia MediaPipe real, carga de los ocho WAV y audio SDL dummy.
Se observó un cambio de zona y terminó correctamente. La captura y GUI se
redirigen solo en ese proceso de prueba; no se cambia el archivo original.

La primera ejecución en sandbox falló al crear NSOpenGLPixelFormat. Al repetir
con acceso a la sesión gráfica, MediaPipe informó contexto Metal/Apple M4 y pasó.
**No se verificó aquí el main original con webcam física y sonido escuchado.**

### B — DSP sin cámara

C++ generó `artifacts/dsp-audition.wav`: PCM16, 48 kHz, dos canales y 384000 frames
(ocho segundos). La prueba comprobó duración y amplitud no nula acotada.
El archivo es reproducible con `afplay`; generar el WAV no equivale a escucharlo.
Los valores concretos de RMS y pico están en `TEST_RESULTS.json`.

También quedó `artifacts/contemporary-ambient.wav`, una grabación de 12.016 s
generada mediante el compositor Python completo: pico 0.12082 y RMS 0.01464.
Es una demostración ambiental relativamente suave; ajuste el volumen de salida
para escucharla. No se hizo una evaluación estética humana automática.

### C — IPC Python/C++

La prueba existente `modules45_python_ipc` pasó. La nueva `composer_ipc` también
pasó: compositor, controles de ganancia, comandos futuros y WAV no silencioso.
En una sesión offline de 12 s se enviaron 101 eventos, 57 NOTE_ON; no hubo pérdidas,
comandos inválidos, saturación del scheduler ni ticks saltados del compositor.
La lectura de confirmaciones no se confunde con la ejecución de todos los eventos
futuros: verifica que el consumidor recibió el lote.

### D — CoreAudio sin cámara

Se ejecutó realmente `cv-daw-engine --demo --seconds 3 --profile`:
CoreAudio inició, el callback produjo 144128 frames en 1126 llamadas y el proceso
terminó limpiamente. Media del segmento instrumentado: 25.77 µs; máximo 31.48 µs.

Se ejecutó después el compositor **mock con CoreAudio** durante unos 10 s:
78 eventos, 43 NOTE_ON, energía aproximada 0.35..0.897, cero pérdidas y cero
comandos tardíos. Media del callback 23.71 µs; máximo 34.24 µs.
**Esto demuestra acceso y procesamiento CoreAudio, no confirmación de audición.**

### E — Cámara y vídeo reales

Se abrió la cámara física del Mac con OpenCV/AVFoundation y se procesó con
MediaPipe. La sesión final de unos 3.03 s obtuvo **45 cuadros**, pero **0 poses
válidas y 0 NOTE_ON**. El sistema bajó las ganancias y no inventó movimiento.
El acceso a la cámara funciona; falta la prueba con una persona bien encuadrada.
La ventana gráfica de la aplicación nueva también queda por verificar manualmente:
estas ejecuciones usaron `--no-window`.

Como prueba reproducible, se procesó `brazo.mp4` con la cadena nueva. Inicialmente
se exigían todos los tobillos visibles y no componía: el vídeo recorta las piernas.
Se corrigió la política para exigir torso y alguna muñeca, y excluir extremidades
ocultas en lugar de utilizar sus coordenadas no confiables.

Resultado final con vídeo: **163 cuadros con seguimiento**, 27.12 FPS de proceso,
energía 0..0.493, **33 eventos y 16 NOTE_ON**, sin pérdidas/rechazos/retrasos. Se
produjo `artifacts/video-composition.wav`. Es evidencia de movimiento humano
**grabado** cambiando características y composición; no sustituye la sesión de
danza en vivo.

### F — Rendimiento observado

| Sesión | FPS de proceso | Callback medio / máximo | Ida y vuelta IPC |
| --- | --- | --- | --- |
| Mock + CoreAudio, ~10 s | 27.05 (control simulado) | 23.71 / 34.24 µs | 2.086 y 0.350 ms |
| Vídeo + DSP offline, ~6 s | 27.12 | Sin callback de dispositivo | 1.341 y 2.604 ms |
| Cámara + CoreAudio, ~3 s, sin pose válida | 14.87 | 16.58 / 23.85 µs | 3.791 y 1.824 ms |

Los 128 frames a 48 kHz representan 2.667 ms. La muestra de tiempos del DSP es
pequeña y corresponde al Mac, no al STM32. En la cámara hubo un ciclo Python de
hasta **851 ms** durante el arranque/espera; el compositor saltó un tick en lugar
de disparar eventos atrasados. No se cumple ni se afirma un límite físico de 35 ms.

La ida y vuelta se mide desde Python hasta observar avance del consumidor e
incluye sondeo y el siguiente bloque; son dos observaciones por sesión, no una
estimación estadística p99. El tiempo físico cámara → altavoz no fue medido.
El contador XRUN de Linux apareció en cero en Mac, pero **no instrumenta todos los
fallos de CoreAudio** y no demuestra ausencia de dropouts.

## Estado final de las pruebas automáticas

- CTest: **5/5 aprobadas**, 0 fallidas, unos 4.99 s en la ejecución final registrada.
- `unittest discover` de `tests/`: **11 pruebas aprobadas**, incluyendo nueve del compositor y dos de postura existentes.
- Python compila los módulos nuevos sin errores de sintaxis.
- `git diff --check` pasó.
- Los archivos originales de Mili no tienen diferencias frente a HEAD.

Los guards allocator/mutex y seccomp específicos de Linux permanecen en el
repositorio, pero **no se ejecutaron en este macOS**. No se ha compilado aquí el
nuevo cambio para la placa ni ejecutado CI remoto.

## Fallos observados y pendientes

1. Falta inicial de MediaPipe: resuelta con entorno separado.
2. NSOpenGLPixelFormat dentro del sandbox: resuelto al ejecutar con acceso gráfico;
   una sesión realmente headless puede seguir requiriendo otra estrategia.
3. `pip check` con pip 26.2 informó «mediapipe 0.10.21 is not supported on this
   platform». Es coherente con el problema de metadatos del wheel
   [reportado en MediaPipe #6112](https://github.com/google-ai-edge/mediapipe/issues/6112).
   No se modificaron paquetes para ocultarlo. Importación e inferencia ARM real sí
   funcionaron; la reproducibilidad del empaquetado sigue siendo una limitación.
4. Primer intento de vídeo con política de cuerpo completo: resuelto con confianza
   por extremidad y pruebas de oclusión/reaparición.
5. Cámara sin pose válida: **pendiente**, requiere persona, encuadre e iluminación.
6. Audición subjetiva, ajuste de sonoridad/timbre, pausas musicales y calibración
   personalizada: **pendientes** de probar con la persona usuaria.
7. No hay medición física de latencia ni prueba prolongada de dropouts; no hay
   compensación de latencia Bluetooth ni sincronización con audio externo.
8. Las envolventes y cuatro timbres son deliberadamente simples. No se añadió un
   gran modelo de IA, ni NPU, ni stems al primer hito integrado.

## Próximo paso hacia STM32

Primero validar la experiencia musical en Mac siguiendo `MAC_SETUP.md`. Después:
compilar el mismo `realtime/` con el SDK OpenSTLinux y Cortex-A35, comprobar ABI
versión 2, adaptar captura al sensor, perfilar pose en CPU/NPU con modelos reales,
y trasladar el compositor manteniendo un solo productor. Reducir el retardo de
suavizado/planificación solo después de medir su efecto musical y de estabilidad.

Verificar PREEMPT_RT, prioridades e IRQs, bloqueo de memoria, formato ALSA,
consumo de CPU y térmica. Ejecutar guards/seccomp en Linux y pruebas largas de
XRUNs. Medir movimiento a DAC con referencia externa. Los resultados Mac no
sustituyen ninguno de esos criterios de aceptación en hardware.

## Ajuste posterior: webcam del Mac y vista de vídeo

Se detectó que la cámara usada antes era Continuity Camera del iPhone.
`system_profiler` identificó MacBook Pro Camera e iPhone; OpenCV 4.11 ordena sus
uniqueID, por lo que en este equipo el iPhone era 0 y el Mac 1. Esa comprobación
corrige la interpretación de las pruebas de cámara anteriores: verificaban una
fuente accesible desde macOS, no necesariamente la webcam integrada.

Se añadió `composer/cameras.py` para preferir la webcam integrada y recalcular su
índice cuando cambia la lista. `--camera` conserva la selección manual. Si no se
encuentra una integrada, se informa el problema sin abrir el iPhone silenciosamente.
El orden se corresponde con el backend AVFoundation de OpenCV 4.11 fijado en las
dependencias. La ventana de vídeo se crea explícitamente y se puede redimensionar.
Cuatro pruebas adicionales de selección de cámara pasaron.
