# Playera LED Wearable — Firmware ESP32 (BLE Low Energy + WS2812B)

Esta carpeta contiene el código dedicado del **ESP32 DevKit de 30 pines** que controla la tira continua **WS2812B** montada en la playera (`Brazo Izquierdo -> Pecho/Espalda -> Brazo Derecho`), sincronizada en tiempo real por **Bluetooth Low Energy (BLE — Nordic UART Service)** con la **STM32MP257F-DK** (o tu Mac durante desarrollo).

---

## 1. Conexiones de Hardware y Reglas de Seguridad (Protoboard / Perfboard)

- **Alimentación**: Power bank de **5 V ($\ge 2\text{ A}$)** + capacitor electrolítico de **$1000\ \mu\text{F}$ (35 V)** entre `+5V` y `GND`.
- **Tierra común (`GND`)**: Compartida entre Power Bank, `GND` del ESP32, ambos lados del conversor de nivel lógico y `GND` de la tira WS2812B.
- **Datos (3.3 V $\rightarrow$ 5 V)**:
  - `GPIO 13` del ESP32 $\rightarrow$ `LV1` del conversor lógico (`LV` a `3V3` del ESP32).
  - `HV1` del conversor lógico (`HV` a `+5V`) $\rightarrow$ resistencia de **$330\ \Omega$ o $470\ \Omega$** en serie $\rightarrow$ `DIN` de la tira WS2812B (respetando la flecha **`DI -> DO`**).
- **Protección de corriente integrada**: El firmware limita automáticamente el consumo con `FastLED.setMaxPowerInVoltsAndMilliamps(5, 1600)` (`1.6 A` máx) para que la power bank de `2 A` nunca sufra caídas de tensión ni reinicie el ESP32 en los drops.
- **REGLA DE ORO AL PROGRAMAR POR USB**: **Desconecta siempre el cable de `+5V / VIN` de la power bank antes de conectar el ESP32 por USB a la computadora** para no cruzar alimentaciones.

---

## 2. Cómo Cargar el Firmware al ESP32

### Opción A — Arduino IDE
1. Instala el soporte de placas **esp32 by Espressif Systems** (incluye `BLEDevice` de fábrica) y la librería **FastLED** desde el Library Manager.
2. Abre [playera_led_esp32/playera_led_esp32.ino](playera_led_esp32/playera_led_esp32.ino).
3. Selecciona placa **`ESP32 Dev Module`**, puerto USB y sube el sketch (`115200` baudios en el Monitor Serie).

### Opción B — PlatformIO (VS Code / CLI)
```bash
cd wearable_esp32
pio run --target upload --target monitor
```

Al encender, el ESP32 anuncia automáticamente por **BLE Low Energy** el nombre **`CV-DAW-SHIRT`** y también acepta exactamente los mismos comandos por **USB Serial (`115200`)** para pruebas rápidas en protoboard.

---

## 3. Calibración Física al Cortar la Tira WS2812B (Sin Reprogramar)

Cuando llegue el rollo WS2812B y midan cuántos LEDs quedan en **Brazo Izquierdo**, **Pecho/Espalda** y **Brazo Derecho** (por defecto `25, 10, 25` = `60` LEDs):

1. Enciende el ESP32 y lanza el patrón de calibración desde tu Mac o la STM32MP257F-DK:
   ```bash
   python -m composer.wearable.cli --mode ble --leds 25,10,25 --test-segments
   ```
   *(O escribe `CFG 25,10,25,1` y luego `TEST` en el Monitor Serie de Arduino).*
2. Verás:
   - **Rojo** = Brazo Izquierdo (con punta blanca en la muñeca)
   - **Verde** = Pecho / Espalda
   - **Azul** = Brazo Derecho (con punta blanca en la muñeca)
3. Ajusta `--wearable-leds L,C,R` al lanzar `composer.app` (y actualiza `DEFAULT_LEFT_LEDS`, `DEFAULT_CHEST_LEDS`, `DEFAULT_RIGHT_LEDS` en el `.ino` cuando tengan las medidas definitivas).

---

## 4. Protocolo de Mensajes ASCII (`\n`)

Todos los mensajes terminan en `\n` y ocupan $\le 20$ bytes para viajar en un solo paquete BLE sin fragmentación (`WriteWithoutResponse`, `< 12 ms` de latencia):

| Comando | Ejemplo | Efecto en la Playera LED |
| :--- | :--- | :--- |
| `L<idx>\n` | `L18\n` | *(Compatible con diseño base)* Llena brazo izquierdo hasta el LED `18` |
| `R<idx>\n` | `R22\n` | *(Compatible con diseño base)* Llena brazo derecho hasta el LED `22` |
| `OFF\n` | `OFF\n` | Apaga toda la tira limpiamente |
| `S<L>,<R>,<E>,<P>,<B>\n` | `S16,21,78,1,1\n` | Estado musical continuo (~25 Hz): brazos `L/R`, energía `0..99`, paleta del género `0..5` (`Ambient/House/LoFi/Latin/Synthwave/DnB`), y pulso `B=1` exacto en cada beat del BPM |
| `FX PL,<int>\n` / `FX PR,<int>\n` | `FX PR,95\n` | **Hard Punch Izq / Der**: Cometa explosivo del pecho hacia el puño |
| `FX WL,<int>\n` / `FX WR,<int>\n` | `FX WL,92\n` | **Arm Wave L→R / R→L**: Ola líquida arcoíris viajando de muñeca a muñeca pasando por el pecho |
| `FX UP,<int>\n` | `FX UP,85\n` | **Double-Arm Rise**: Aurora ascendente del pecho hacia ambas manos |
| `FX DR,<int>\n` | `FX DR,99\n` | **Drop / Salto / Squat Pop**: Destello de impacto + onda expansiva |
| `FX XC,<int>\n` | `FX XC,90\n` | **X-Cross DJ Cut**: Brazos se atenúan y el pecho late en violeta profundo |
| `FX FR,<int>\n` | `FX FR,90\n` | **Body Freeze**: Destellos de hielo congelados en toda la tira |
| `FX KL,<int>\n` / `FX KR,<int>\n` | `FX KR,90\n` | **Knee Stomp Izq / Der**: Golpe de graves en pecho y hombro |
| `FX CU,<int>\n` | `FX CU,92\n` | **Movimiento Custom ML/DTW** (ej. `kakaka`): Espiral prismática en toda la playera |
| `CFG <L>,<C>,<R>,<revL>\n` | `CFG 25,10,25,1\n` | Ajusta en vivo el conteo de LEDs por tramo e inversión del brazo izquierdo |
| `TEST\n` | `TEST\n` | Muestra 3.5 s de calibración **Rojo (Izq) / Verde (Pecho) / Azul (Der)** |
