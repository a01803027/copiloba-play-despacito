# Qué existía antes de integrar el compositor

## Auditoría inicial

Se revisó `/Users/HaeJin/DJ` antes de editar: rama **tuve-un-sueno**, árbol de
trabajo limpio, commit `a84344d`. Estaban `BrazoIzquierdo.py`,
`BodyTrackingModule.py`, `HandTrackingModule.py` y los motores C++.
No se cambió de rama ni se hizo pull, merge o reset.

## El prototipo de Mili

```text
OpenCV abre cámara 0 → lee imagen → BodyDetector ejecuta MediaPipe Pose
→ devuelve puntos en píxeles → BrazoIzquierdo calcula una zona vertical
→ pygame reproduce un WAV → OpenCV muestra imagen y datos
```

En [BrazoIzquierdo.py](../BrazoIzquierdo.py), `main()` inicia `pygame.mixer`,
carga ocho archivos WAV desde la carpeta actual y reserva el canal 0.
`cv2.VideoCapture(0)` abre la cámara predeterminada. `cap.read()` obtiene cada
imagen; no hay captura paralela ni garantía de 60 imágenes por segundo.

[BodyTrackingModule.py](../BodyTrackingModule.py) importa **MediaPipe**, no YOLO.
`BodyDetector.__init__()` crea `mp.solutions.pose.Pose`, con complejidad 1 y
confianzas 0.5. `findBody()` convierte BGR a RGB, llama `pose.process()` y dibuja
el esqueleto. `findPosition()` transforma los 33 puntos normalizados en listas
`[id, x_en_píxeles, y_en_píxeles]`. No devuelve profundidad ni confianza.
`maxBody` y `bodyNo` no implementan seguimiento de varias personas.

### Cómo convierte altura en sonido

En la versión auditada, `BrazoIzquierdo.py` usa `lmList[9][2]` (línea 76), aunque
su comentario dice «muñeca izquierda en YOLO». **El detector es MediaPipe: el
punto 9 corresponde a la boca; la muñeca izquierda es 15.** Por eso no se debe
interpretar esta versión como un mapeo correcto de muñeca sin corregir ese índice.
Se conserva el archivo original para no alterar la prueba de Mili silenciosamente.
La aplicación integrada nueva usa explícitamente 15 y 16 para las muñecas.

Si `h` es la altura de imagen y `cy` la altura del punto:

```text
altura_invertida = h - cy
porcentaje = limitar(100 × altura_invertida / h, 0, 100)
zona = mínimo(8, entero(porcentaje / 12.5) + 1)
```

Hay ocho franjas. Al cambiar de franja, reproduce el WAV correspondiente:
do, re, mi, fa, la, si, sol, do sostenido, en ese orden concreto del arreglo.
El orden no es una escala ascendente. El «porcentaje de intensidad» selecciona
la zona: **no modifica el volumen**. Un solo canal reemplaza el sonido anterior.
No hay armonía, composición, metrónomo ni interpretación del baile completo.
`q` cierra la ventana y libera cámara/audio.

[HandTrackingModule.py](../HandTrackingModule.py) usa MediaPipe Hands, pero
`BrazoIzquierdo.py` no lo llama. No hay gestos de manos en ese prototipo.

### Dependencias y ejecución independiente

El `requirements.txt` auditado enumera ultralytics, OpenCV y pygame, pero omite
MediaPipe, que el detector realmente importa. En el entorno `.venv` existente,
la importación inicial falló con `ModuleNotFoundError: mediapipe` y Python 3.14.
Se añade un archivo separado `requirements-mac.txt` y un entorno `.venv-mac`
con Python 3.11, sin reemplazar ese entorno ni el archivo original.

Desde la raíz, con las dependencias del nuevo entorno instaladas:

```sh
source .venv-mac/bin/activate
python BrazoIzquierdo.py
```

La carpeta actual debe contener los WAV. Necesita autorización de cámara, salida
de audio y una sesión gráfica. Consulte [MAC_SETUP.md](../MAC_SETUP.md).
Pasar una importación o procesar un vídeo no demuestra que la cámara física funcione.

## Qué había en C++

| Ubicación | Función real antes de este trabajo | Limitación |
| --- | --- | --- |
| `include/cvdaw/`, `src/audio.cpp` | Primer DSP, IPC, musical brain y ALSA directo | Linux; memoria `/cvdaw` |
| `src/brain.cpp`, `perception/run.py` | Protocolo de texto de pose y frontend GStreamer | No conecta al ABI nuevo; captura pensada para Linux |
| `realtime/ipc_bridge.hpp` | Cola SPSC de 256 comandos, controles atómicos y mmap | Funciona Mac/Linux con el mismo ABI |
| `realtime/ipc_bridge_c.cpp` | Funciones nativas para que Python opere atomics | El acceso correcto es por estas funciones, no escribiendo floats crudos |
| `realtime/dsp_primitives.hpp` | Tablas limitadas en armónicos, voces, ADSR, filtro, reverb, limitador | DSP funcional; no es compositor |
| `realtime/audio_engine.hpp` | Mezcla, reloj de muestras, comandos futuros, patrones de dos compases | Recibe notas, no interpreta movimiento |
| `realtime/audio_engine.cpp` | miniaudio ALSA y ejecución silenciosa offline | Antes no reproducía en Mac: solo backend null |
| `realtime/stem_worker.hpp` | Worker Rubber Band opcional para cuatro stems | Necesita biblioteca y archivos; no se requiere para la primera integración |
| `realtime/vision_mock_sender.py` | Notas y barridos simulados por IPC | Prueba de conexión, no composición expresiva |

Había dos implementaciones, no una sola cadena conectada. Sus formatos de comandos
y nombres de memoria son distintos. Se conserva la primera y se integra sobre
`realtime/`, que ya tiene miniaudio y el puente nativo multiplataforma.

## Cómo se comunicaban Python y C++

El prototipo de Mili **no se comunicaba** con C++: todo su audio era pygame.
El mock de `realtime` sí usaba `multiprocessing.shared_memory`, ctypes y la
biblioteca `libcv_daw_ipc` para enviar comandos a `/cv_daw_shm`.
La integración nueva reutiliza ese puente y la clase de detección corporal de
Mili; cambia el significado musical en un módulo aparte.
