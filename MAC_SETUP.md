# CV-DAW en tu Mac: guía desde cero

## Qué vas a conseguir

Primero escucharás una composición controlada por movimiento simulado. Después
usarás la cámara para influir con tu cuerpo. La composición usa Do menor, motivos
repetidos y reglas de energía, fluidez y expansión: no una nota por franja de altura.

Los archivos de Mili siguen intactos. También puedes ejecutar su prueba aparte.
Esta guía es para **macOS Apple Silicon**. No copies aquí los comandos de permisos
Linux ni uses `sudo` para arrancar el motor.

Abre Terminal con **Command + Espacio**, escribe «Terminal» y pulsa Return.
Copia un recuadro cada vez, sin copiar los símbolos que lo rodean. Pulsa Return
y espera a que termine antes de seguir. Si un comando falla, conserva el error.

## 1. Comprueba la carpeta y la rama

```sh
cd /Users/HaeJin/DJ
```

`cd` entra en tu carpeta del proyecto. Todos los comandos siguientes parten de
ella. Si moviste la carpeta, utiliza su ubicación real.

```sh
pwd
```

Muestra la carpeta actual: debe terminar en `/DJ`.

```sh
git branch --show-current
```

Muestra la rama activa. Debe decir **tuve-un-sueno**. Si dice otra cosa, detente:
esta guía no cambia ramas y no debe trabajar sobre `main`.

```sh
git status --short
```

Muestra archivos modificados o nuevos. Los cambios de esta integración son
esperados. No uses reset, checkout ni pull para «limpiarlos».

## 2. Comprueba e instala herramientas

```sh
uname -m
```

Muestra el tipo de procesador. En Apple Silicon nativo esperamos `arm64`.
Si dice `x86_64`, tu terminal podría estar ejecutándose con Rosetta.

```sh
xcode-select -p
```

Comprueba que existen herramientas de compilación de Apple. Si devuelve una
ruta, puedes saltar el siguiente comando.

```sh
xcode-select --install
```

**Solo si faltan las herramientas**, abre su instalador. Incluye el compilador
que convierte el código C++ en un programa. Espera a que termine la instalación.

```sh
/opt/homebrew/bin/brew --version
```

Comprueba Homebrew, el instalador de herramientas que usaremos. Si existe,
salta el siguiente comando de instalación.

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Solo si no tienes Homebrew**, descarga y ejecuta su instalador oficial. Puede
pedir tu contraseña e instrucciones de confirmación. La contraseña no se muestra
al escribir. La referencia oficial es [Homebrew Installation](https://docs.brew.sh/Installation).
No necesitas reinstalarlo si el comando anterior ya mostró una versión.

```sh
/opt/homebrew/bin/brew install cmake python@3.11
```

Instala CMake y Python 3.11. CMake organiza la compilación; se fija Python 3.11
para usar la API MediaPipe existente. No utilizamos el Python 3.14 del entorno
anterior. Si Homebrew indica que ya están instalados, continúa.

## 3. Crea un entorno Python separado

Un entorno es una carpeta con las bibliotecas de esta aplicación. Así no se
mezclan con otros proyectos. Se usa `.venv-mac`; no se reemplaza `.venv`.

```sh
/opt/homebrew/bin/python3.11 -m venv .venv-mac
```

Crea el entorno. Si esta integración ya dejó `.venv-mac` preparado, puedes
saltar este paso; no hace falta crearlo cada día.

```sh
source .venv-mac/bin/activate
```

Activa ese entorno **en esta pestaña**. A partir de ahora `python` y `pip`
corresponden al proyecto. Cada pestaña nueva necesita su propia activación.

```sh
python --version
```

Debe mostrar Python **3.11.x**. Si muestra 3.14, no continúes instalando paquetes:
revisa que activaste `.venv-mac` y no el entorno anterior.

```sh
python -m pip install -r requirements-mac.txt
```

Instala las versiones fijadas de MediaPipe, OpenCV, pygame y sus dependencias.
OpenCV-contrib ya proporciona `cv2`; no instales simultáneamente otro paquete
OpenCV en este entorno. pygame solo es necesario para la aplicación original.
No hace falta instalar ultralytics para el detector actual de Mili.

```sh
python -c "import cv2, mediapipe, pygame; print('Dependencias importadas correctamente')"
```

Comprueba que Python puede cargar las bibliotecas. Algunos mensajes de MediaPipe
o pygame son informativos. El fallo que debemos resolver es un traceback.

MediaPipe 0.10.21 contiene un problema conocido de metadatos de arquitectura:
algunas herramientas pueden decir «not supported on this platform» al inspeccionar
la distribución. Aquí se pudo importar y ejecutar inferencia ARM real; no ignores
un fallo real de importación. Véase [incidencia oficial #6112](https://github.com/google-ai-edge/mediapipe/issues/6112).

## 4. Compila el motor C++

```sh
cmake -S realtime -B realtime/build-mac -DCMAKE_BUILD_TYPE=Release -DCV_DAW_WITH_RUBBERBAND=OFF -DPython3_EXECUTABLE="$PWD/.venv-mac/bin/python"
```

Prepara un build específico del Mac en `realtime/build-mac`. Usa el Python del
entorno para las pruebas. Desactiva los stems/Rubber Band porque no hacen falta
para la primera composición. No pongas `CV_DAW_CORTEX_A35=ON` en el Mac.

```sh
cmake --build realtime/build-mac --parallel 2
```

Compila el motor y el puente. Espera a que aparezca `Built target cv-daw-engine`.
Un warning aislado del archivo incluido miniaudio no significa que haya fallado.

```sh
ctest --test-dir realtime/build-mac --output-on-failure
```

Ejecuta las pruebas de DSP, conexión Python/C++, WAV y reglas musicales.
No tengas otro motor abierto mientras corren: usan la misma memoria compartida.
El resultado esperado de esta versión es **5 pruebas aprobadas**.

## 5. Ejecuta la prueba original de Mili, si quieres compararla

```sh
python BrazoIzquierdo.py
```

Abre la cámara y reproduce los WAV originales con pygame. Se ejecuta desde la
raíz porque busca allí `do2.wav`, `re2.wav`, etc. Pulsa **q** con la ventana
seleccionada para salir. El archivo se conservó: todavía usa el índice 9 de
MediaPipe por una confusión con YOLO; no esperes que represente bien la muñeca.
La explicación y el cálculo de zonas están en
[EXISTING_SYSTEM_EXPLAINED.md](docs/EXISTING_SYSTEM_EXPLAINED.md).

```sh
python tools/check_mili.py
```

Alternativa de diagnóstico: ejecuta el `main()` original con 30 imágenes de
`brazo.mp4`, MediaPipe real y audio dummy silencioso. No abre la cámara ni prueba
los altavoces. Es útil para separar un problema de dependencias de uno de hardware.

## 6. Escucha el DSP sin cámara y sin Python compositor

```sh
mkdir -p artifacts
```

Crea una carpeta para WAV y métricas. No se sube a Git porque contiene resultados
locales. Si ya existe, el comando no la borra.

```sh
./realtime/build-mac/cv-daw-engine --render artifacts/dsp-audition.wav --seconds 8 --demo
```

C++ calcula ocho segundos de acordes y escribe un WAV. Esto no usa un dispositivo
de audio; no se escucha nada mientras se genera.

```sh
afplay artifacts/dsp-audition.wav
```

El reproductor del Mac abre ese WAV. Aquí sí debes escuchar sonido por la salida
que tengas seleccionada en Ajustes del Sistema → Sonido. Control+C lo detiene.

```sh
./realtime/build-mac/cv-daw-engine --demo --seconds 5 --profile
```

Prueba directamente CoreAudio durante cinco segundos. Al comenzar debe aparecer
`audio-ready backend=CoreAudio`. Al terminar imprime tiempos del callback.
Esta es una audición del DSP; no es todavía el compositor de danza.

## 7. Escucha la composición sin cámara: camino recomendado

```sh
python -m composer.app --mode mock --launch-engine --seconds 30 --metrics artifacts/mi-mock.json
```

Inicia el motor C++ y el compositor Python juntos. `mock` simula movimientos
fluidos/abruptos y expansión; la música evoluciona durante treinta segundos.
`--launch-engine` evita tener que manejar una segunda pestaña. El archivo JSON
contiene las mediciones. Al finalizar, el programa cierra su propio motor.

```sh
python -m composer.app --mode mock --launch-engine --offline-audio --record artifacts/mi-composicion.wav --seconds 12 --metrics artifacts/mi-render.json
```

Genera una demostración **silenciosa mientras se calcula**, con la cadena real
Python → memoria compartida → DSP. Usa `--offline-audio` porque la escritura WAV
no se permite en el callback de audio en vivo.

```sh
afplay artifacts/mi-composicion.wav
```

Escucha el resultado grabado. La semilla predeterminada es 7; puedes añadir
`--seed 12` al compositor para otras pequeñas variaciones controladas.

## 8. Prueba el motor y el mock antiguo por separado (opcional)

Sirve para entender la conexión. No es necesario si el apartado 7 ya funciona.

En **pestaña A**, desde la carpeta del proyecto y con el entorno activado:

```sh
./realtime/build-mac/cv-daw-engine --profile
```

Deja abierto el motor. Aún no hay notas: espera mensajes. No uses `--demo` si vas
a conectar un controlador Python, porque solo puede haber un productor.

Abre la **pestaña B** con Command+T:

```sh
cd /Users/HaeJin/DJ
```

Entra en el proyecto también en esta pestaña.

```sh
source .venv-mac/bin/activate
```

Activa el mismo entorno en la pestaña nueva.

```sh
python realtime/vision_mock_sender.py --library realtime/build-mac/libcv_daw_ipc.dylib --seconds 10
```

Envía notas y barridos de filtro durante diez segundos. Debe terminar con
`IPC PASS`. Usa `.dylib` en Mac, no `.so`. Este es el mock básico anterior;
el modo mock del compositor del apartado 7 demuestra mejor la intención musical.
Al terminar, vuelve a A y pulsa **Control+C** para cerrar el motor.

## 9. Usa la cámara para componer

Primero detén cualquier motor o mock anterior. Sitúa el Mac para ver hombros,
caderas y manos. Para interpretar piernas, aléjate hasta incluir los pies.
Usa iluminación suficiente y deja una sola persona delante de la cámara.

```sh
python -m composer.app --mode camera --launch-engine --metrics artifacts/mi-danza.json
```

Abre la webcam integrada del Mac, detector, compositor y CoreAudio, con vídeo
en una ventana de 960×720. La selección automática evita el iPhone y muestra
el nombre y el índice elegidos. En la configuración comprobada, MacBook Pro Camera
es el índice 1 y el iPhone es el 0. Si cambian los dispositivos, se vuelve a buscar
la webcam integrada por nombre. No utilices `--no-window` si quieres ver el vídeo.
Puedes forzar una selección añadiendo `--camera 1` al comando anterior.
macOS puede pedir permiso de
Cámara para Terminal/Python. Acéptalo si quieres usar esta función. Si fue negado,
revisa Ajustes del Sistema → Privacidad y seguridad → Cámara y vuelve a abrir
Terminal si es necesario.

La ventana muestra energía, expansión y BPM. Prueba lo siguiente:

1. Mantén el torso visible y mueve lentamente ambos brazos: observa energía.
2. Abre y cierra el espacio entre brazos: cambia el filtro y el registro.
3. Alterna movimiento fluido y cortes: cambia articulación, ritmo y tensión.
4. Aumenta la actividad durante varios segundos: aparecen más notas/percusión.
5. Quédate quieta/o: disminuye la densidad y regresa una armonía más reposada.

Hay tres tiempos de respuesta: volumen/filtro rápidos, densidad en segundos y
armonía en frases de dos compases. No todos los movimientos producen una nota
inmediata. Si no se detecta el torso, baja el sonido y aparece una indicación.

Pulsa **q** con la ventana seleccionada o **Control+C** en Terminal para salir.
La salida de esta prueba todavía debe confirmarse contigo bailando y escuchando:
una cámara abierta, por sí sola, no demuestra un instrumento musical validado.

```sh
python -m composer.app --mode video --video brazo.mp4 --launch-engine --seconds 6
```

Alternativa reproducible: usa el vídeo del repositorio en lugar de la cámara.
Conserva detector real, compositor, IPC y CoreAudio. Es útil para probar antes
de colocarte delante de la cámara.

## 10. Cómo detener todo y salir

**Control+C** es una combinación del teclado, no un comando para escribir.
En el compositor con `--launch-engine`, detiene Python y su hijo C++.
Si abriste ambos por separado, detén primero el controlador y después el motor.
No cierres simplemente la ventana de Terminal mientras el motor sigue activo.

```sh
deactivate
```

Desactiva el entorno de esta pestaña cuando ya hayas parado los programas.
No desinstala paquetes ni borra archivos.

## 11. Volver otro día

```sh
cd /Users/HaeJin/DJ
```

Entra de nuevo en el proyecto.

```sh
git branch --show-current
```

Confirma de nuevo `tuve-un-sueno`. No hace falta descargar ni mezclar ramas.

```sh
source .venv-mac/bin/activate
```

Activa el entorno ya creado. No reinstales paquetes cada vez.

```sh
python -m composer.app --mode mock --launch-engine --seconds 15
```

Comprueba primero sonido sin cámara.

```sh
python -m composer.app --mode camera --launch-engine
```

Después prueba la danza. Si cambiaste archivos C++, vuelve a ejecutar el comando
de compilación del apartado 4 antes de arrancar. Si cambió `requirements-mac.txt`,
vuelve a instalar sus dependencias. No hace falta recrear el entorno por rutina.

## 12. Problemas habituales

| Mensaje/síntoma | Acción |
| --- | --- |
| `No module named mediapipe` | Activa `.venv-mac` e instala `requirements-mac.txt`, no solo `requirements.txt`. |
| `mediapipe has no attribute solutions` | Confirma que estás usando la versión fijada 0.10.21 y Python 3.11. |
| `cmake: command not found` | Usa `/opt/homebrew/bin/cmake` en lugar de `cmake`, o revisa instalación de Homebrew. |
| Falta `libcv_daw_ipc.dylib` o el ejecutable | Compila `realtime/build-mac`; no uses la carpeta `build` antigua. |
| `ABI incompatible` | Detén todos los procesos y recompila biblioteca y motor juntos; la nueva versión es 2. |
| `shm_open: File exists` | Otro motor está abierto o una salida abrupta dejó la conexión. No borres memoria mientras siga abierto un cliente. Comparte el error para hacer limpieza segura. |
| `another producer` / «otro controlador» | Cierra el mock antes de abrir cámara; no combines `--demo` con Python. |
| `NSOpenGLPixelFormat` | Ejecuta desde Terminal en tu sesión gráfica local, no desde un sandbox/headless sin acceso al contexto gráfico. |
| Cámara cerrada o no entrega imágenes | Revisa permisos, desconecta apps que la usen y vuelve a iniciar la aplicación. |
| Energía siempre cero / no hay notas | Deben verse hombros, caderas y al menos una muñeca con suficiente confianza. Aleja/ilumina la cámara. |
| Se ve el cuerpo pero el movimiento no cambia rápido la armonía | Los acordes se deciden en frases; prueba unos 20 segundos y observa también filtro/volumen. |
| `audio-ready` pero no escuchas | Comprueba salida de Sonido, volumen y auriculares/Bluetooth; prueba el WAV con `afplay`. |
| `Audio device initialization failed` | Comparte mensaje completo y prueba la salida predeterminada del Mac. No uses instrucciones ALSA para corregir CoreAudio. |
| Avisos de TensorFlow/feedback/landmark | Pueden ser advertencias de MediaPipe. Busca si también hay traceback o salida con error. |
| Error de permisos de memoria en pruebas dentro de un agente | Repite `ctest` en tu Terminal normal; las pruebas necesitan POSIX shared memory. |

Si necesitas ayuda, envía **el comando exacto y toda su salida**, y di si estabas
en modo mock, cámara o vídeo. No necesitas saber interpretar el error.

## Documentos de apoyo

- [Qué hacía Mili y qué motores existían](docs/EXISTING_SYSTEM_EXPLAINED.md).
- [Cómo se conecta y cómo compone](docs/SYSTEM_ARCHITECTURE.md).
- [Qué se probó realmente y qué falta](docs/IMPLEMENTATION_REPORT.md).

La guía usa un entorno virtual según la [documentación de Python](https://docs.python.org/3/library/venv.html)
y el backend CoreAudio descrito por [miniaudio](https://miniaud.io/docs/manual/).
