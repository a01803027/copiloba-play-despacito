# Manual del instrumento CV-DAW

## A. Primera sesión y alcance

La entrada del instrumento de cuatro extremidades es `python -m composer.performance`. Funciona con cámara, vídeo o movimiento sintético. La cámara permite tocar dos voces melódicas y dos percusivas, cambiar sonidos por lado, silenciarlos, repetir golpes y mezclar cuatro pistas de acompañamiento preparadas. El seguimiento de pose y manos usa MediaPipe por CPU en la configuración base. La detección opcional por modelo externo o acelerador y el seguimiento de tempo por rebote son rutas que conviene validar con el equipo y los movimientos reales. Los modelos externos son opcionales y se aportan por separado.

Sigue [MAC_SETUP.md](../MAC_SETUP.md) para instalar dependencias y compilar. En macOS, con ese entorno activo y desde la raíz:

```sh
python -m composer.performance --mode camera --bpm 120
python -m composer.performance --mode mock --offline-audio --seconds 5 --record artifacts/prueba.wav
python -m composer.performance --mode video --video brazo.mp4 --seconds 10
```

Ejecuta una sesión a la vez. La segunda orden genera un WAV para escuchar después y prueba las cuatro vías sin cámara ni bocinas. En vídeo los golpes dependen de las extremidades visibles. En directo, permite acceso a la cámara y encuadra hombros, caderas, muñecas y tobillos. Usa luz frontal y deja espacio para mover los brazos sin tapar las manos ni cruzar las piernas. Elige otra cámara con `--camera N`. Pulsa `q` con la ventana activa o Ctrl+C en la terminal. La ventana muestra FPS de procesamiento, coordenadas verticales antes y después de normalizar, BPM objetivo, estados por extremidad y el índice del último golpe enviado. Los puntos verdes marcan puntas con visibilidad suficiente. Al salir, el informe `hits` cuenta disparos en orden izquierdo/derecho de brazos y luego izquierdo/derecho de piernas. Con `--no-window` usa Ctrl+C o `--seconds`.

La interfaz actual carece de metrónomo audible, edición o borrado de loops, grabación de la salida en vivo, elección directa de presets desde la ventana, separación automática de canciones y salida MIDI a otros equipos. La grabación con `--record` requiere `--offline-audio`.

## B. Mapa corporal

Izquierda y derecha son las de tu cuerpo. En los cuatro canales, un golpe es mover la punta y frenar, en cualquier dirección. Se necesitan ambos hombros y ambas caderas con visibilidad ≥ 0,5, más la muñeca o el tobillo del canal con visibilidad ≥ 0,5. Se comparan cuadros consecutivos separados por hasta 200 ms. El frenado debe cruzar por debajo de −18 longitudes de torso/s² en la dirección de la velocidad anterior, con rapidez anterior > 1,8 longitudes de torso/s y al menos 80 ms desde el último golpe de ese canal. Solo el estado `LIVE` dispara golpes nuevos.

| Parte | Movimiento, sonido inicial y parámetros | Al perder detección |
| --- | --- | --- |
| Brazo izquierdo (0) | Golpe de muñeca izquierda, diente de sierra, registro por altura en la escala | Cesa el disparo y se atenúa su bus salvo en loop |
| Brazo derecho (1) | Golpe de muñeca derecha, seno, registro por altura en la misma escala | Igual, con estado independiente |
| Pierna izquierda (2) | Golpe de tobillo izquierdo, caja DSP y nota interna fija 38 | Igual, con estado independiente |
| Pierna derecha (3) | Golpe de tobillo derecho, bombo DSP y nota interna fija 36 | Igual, con estado independiente |
| Mano izquierda | Forma de la mano, controla el estado y el preset del brazo 0. Con el modificador también puede afectar a la pierna 2 | Deja de emitir cambios de gesto. Conserva el último estado y preset. La muñeca puede seguir tocando |
| Mano derecha | Forma de la mano, controla el estado y el preset del brazo 1. Con el modificador también puede afectar a la pierna 3 | Igual para su lado |
| Cadera y movimiento global | La altura media de las caderas aporta la señal de rebote. Las velocidades de 12 puntos articulares producen energía y pesos de groove, que cambian la ganancia general y los volúmenes de los stems. Con `--follow-bounce` una confianza ≥ 0,7 permite cambiar el BPM objetivo | Se reinicia la ventana temporal y vuelve al contexto inicial de reposo. El cerebro conserva el último BPM aplicado. Si falla el torso se invalidan las cuatro puntas |

En cada golpe la rapidez pico se divide por 8 y se limita a 0,1–1 para dar intensidad. El brillo (corte del paso bajo) sube con la punta entre −0,3 y +1,2 longitudes de torso respecto a la cadera, con altura positiva hacia arriba. La distancia horizontal y en profundidad, saturada a 1,5 torsos, aumenta el envío a reverb. Flexionar el codo o la rodilla sube la afinación o velocidad de reproducción del sample. Una articulación totalmente recta da −2 semitonos, 90° da 0 y el extremo de flexión da +2. Si falta el codo/rodilla o su raíz, el bend vuelve a 0. Si desaparece la punta, el canal pierde su historial continuo de velocidad y usa controles neutros (corte ≈1.451 Hz, reverb y bend cero). Un loop bloqueado sigue audible con esos controles. El regreso de la punta empieza sin una velocidad previa válida. El torso y las manos se calculan por separado, así que una mano visible todavía puede cambiar estados aun si no hay pose válida. [Extracción](../feature_extractors.py#L81) · [Mapeo y estados](../musical_brain.py#L52)

El rebote utiliza 180 muestras a 60 Hz (unos 3 s), un paso banda de 1,33–2,66 Hz y una desviación mínima de 0,002 en coordenada vertical filtrada corregida por aspecto. Su estimación se limita a 80–160 BPM. La ventana de groove usa 30 muestras a 60 Hz y se actualiza cada seis muestras cuando está llena. Una articulación con visibilidad <0,5 en cualquiera de los dos cuadros aporta velocidad cero. [Contexto temporal](../feature_extractors.py#L142)

## C. Gestos de manos y de movimiento

La palma se mide entre muñeca y base del dedo medio. Para índice, medio, anular y meñique, la punta debe estar a más de 1,15 veces la distancia de la muñeca a su articulación intermedia para contar como extendida. La clasificación usa tres cuadros consecutivos de la misma clase para confirmar cambios. Una mano perdida o una forma sin clase interrumpe esa cuenta, pero conserva la última postura confirmada y no rearma el mismo gesto. Usa la mano abierta entre repeticiones.

| Gesto | Cómo hacerlo | Efecto y cómo revertirlo |
| --- | --- | --- |
| Mano abierta | Extiende los cuatro dedos largos | Rearma el cambio al confirmar `OPEN`. No cambia `LIVE`, `MUTED` ni `LOOPING` |
| Puño | Mantén los cuatro dedos largos sin extensión | Alterna silencio y directo. Desde loop primero lo desbloquea y luego silencia. Abre y cierra otra vez para volver a directo |
| Pinza | Junta pulgar e índice a menos de 0,25 palmas con medio, anular y meñique extendidos | Bloquea el patrón. Si estaba silenciado también reactiva el canal. Abre y repite para desbloquear |
| Paz | Extiende índice y medio, dobla anular y meñique | Pasa al preset siguiente, del 7 vuelve al 0. Abre y repite para seguir recorriendo |
| Golpe en el aire | Acelera y frena una muñeca o tobillo como si golpearas un parche imaginario | Dispara una nota en `LIVE`. El detector se rearma al salir del frenado y pasar el tiempo mínimo |
| Movimiento lento o sostenido | Varía altura, distancia o flexión mientras suena una nota o loop | Modula brillo, reverb y bend en tiempo real. Puede cambiar el balance global por energía/groove. Mantener una pose no sostiene una nota ni dispara otra |

Con `--stationary-leg-modifier` los cambios de mano se aplican también a la pierna del mismo lado si su tobillo es válido y se mueve a <0,3 torsos/s justo al confirmar el gesto. Sin esa opción no hay gesto de mano que bloquee, silencie o cambie el preset de las piernas. [Clasificador](../feature_extractors.py#L47) · [Máquina de estados](../musical_brain.py#L52)

## D. Sonidos y presets

| ID | Sonido | Origen |
| --- | --- | --- |
| 0 | Diente de sierra por bandas, brazo izquierdo al iniciar | Oscilador DSP |
| 1 | Seno, brazo derecho al iniciar | Oscilador DSP |
| 2, 3, 4, 5 | Bombo, caja, hi-hat y stab respectivamente | Samples generados en memoria al arrancar (0,5 s por sonido) o cuatro WAV aportados con `--samples KICK SNARE HAT STAB` |
| 6 | Bombo tonal con caída de frecuencia, pierna derecha inicial | Percusión DSP de 0,2 s |
| 7 | Caja con ruido y tono, pierna izquierda inicial | Percusión DSP de 0,2 s |

La señal de paz cambia el sonido de disparos futuros y de los golpes que se vuelvan a reproducir en loop. La escala inicial es `minor-pentatonic`; también existen `dorian`, `phrygian-dominant` y `major-pentatonic` con `--scale`. La altura selecciona una de nueve notas internas en los brazos. Los samples conservan su afinación base y su velocidad cambia con el bend, independientemente de la nota seleccionada. Los WAV de notas que ya están en la raíz pertenecen a la demostración `BrazoIzquierdo.py`; el instrumento integrado usa los sonidos generados salvo que se proporcionen cuatro muestras. Cada WAV de sample debe durar más de 0 y como máximo 15 s tras decodificar. [Sonidos de arranque](../realtime/audio_engine.cpp#L32) · [Voces DSP](../realtime/dsp_primitives.hpp#L87)

## E. Tempo, loops y remezcla

- El tempo inicial es 120 BPM y admite 60–200. Hay cuatro pasos por pulso. A 120 BPM cada semicorchea dura 125 ms. Un golpe hasta 40 ms antes del paso puede esperar al paso. Si llega hasta 35 ms después, suena cuando lo toma el motor y se guarda en el paso anterior. Fuera de esas ventanas suena al tomarlo el motor y se guarda en el paso más cercano al ejecutarse.
- En directo se guardan automáticamente notas e intensidades en 32 pasos, dos compases de 4/4 a razón de 16 pasos por compás. Admite ocho golpes por paso y canal. Un golpe adicional sustituye el octavo de ese paso. La pinza bloquea lo acumulado y lo hace repetir conforme avanza el reloj. El primer sonido puede tardar hasta que el reloj alcance un paso ocupado. Una pinza sobre un patrón vacío da silencio. Desbloquear permite seguir agregando golpes sin borrar los anteriores. Reiniciar la sesión vacía todos los patrones. Los loops usan el preset y los efectos del momento.
- Un stem es un WAV premezclado de una parte de la canción. Exporta batería, bajo, voz y resto desde el mismo punto, con la misma duración y al BPM real de la canción. Por ejemplo:

```sh
python -m composer.performance --bpm 120 --stems bateria.wav bajo.wav voz.wav resto.wav
```

Los cuatro WAV se convierten a estéreo de 48 kHz y deben tener igual longitud decodificada, con hasta 15 minutos por pista. Si se quiere una vuelta alineada al compás, exporta una duración que abarque compases enteros. Por defecto se leen en bucle a su velocidad original. El programa toma `--bpm` como tempo de origen y de arranque. No analiza el tempo o el primer pulso del WAV. Con `--follow-bounce` y stems se exige una compilación con Rubber Band. Así el motor puede estirar las pistas según el BPM objetivo. La posición inicial depende del inicio del motor y la alineación del material. El groove mezcla volúmenes de batería, bajo, voz y resto automáticamente. Los puños y pinzas controlan las voces corporales, que son buses independientes de los stems. La aplicación no ofrece solo por stem, búsqueda, recorte, lanzamiento de secciones ni realineación al pulso de la canción. [Flujo de stem](../realtime/stem_worker.hpp#L40) · [Grabación de patrones](../realtime/audio_engine.hpp#L62)

## F. Práctica progresiva

1. **Un brazo.** Con el derecho en reposo, haz cuatro golpes de muñeca izquierda a una altura similar, separados por medio segundo. Luego alterna dos alturas. Debes oír el diente de sierra y ver `golpe 0`. Compara el registro grave y agudo. Desplaza lentamente la muñeca durante la cola para escuchar el cambio de color sin un nuevo disparo.
2. **Ambas manos.** Alterna golpes 0 y 1 para distinguir sierra y seno. Haz puño a la izquierda durante tres cuadros o más. La primera posición de la ventana pasa a `MUTED` y la derecha sigue sonando. Abre y repite el puño para regresar a `LIVE`. Alterna paz y abierta a la derecha y comprueba el nuevo timbre en el próximo golpe. Toca dos golpes a la izquierda, haz pinza y escucha la repetición en `LOOPING`. Abre y repite la pinza para regresar a directo.
3. **Piernas.** Baja el encuadre hasta incluir ambos tobillos sin cortar hombros. Con los brazos quietos, levanta y frena un pie sin pisar fuerte. Espera el 2 con la izquierda y su caja, y el 3 con la derecha y su bombo. Si falta el punto verde, busca otro ángulo antes de aumentar la fuerza. Repite a volumen moderado y con apoyo estable.
4. **Cuatro extremidades.** Cuenta en voz alta «uno, dos, tres, cuatro» a un pulso cómodo. Prueba bombo derecho en uno, muñeca izquierda en dos, caja izquierda en tres y muñeca derecha en cuatro. Comprueba que el informe final suma disparos en las cuatro posiciones. Después bloquea un brazo con una pinza y toca encima con las piernas. Para oír el balance global, repite con cuatro WAV alineados y alterna reposo y movimiento durante varios segundos.

Consulta [cómo funciona el código](COMO_FUNCIONA_EL_CODIGO.md) y [la tabla de parámetros](PARAMETROS_DEL_INSTRUMENTO.md).
