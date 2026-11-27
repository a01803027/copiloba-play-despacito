/*
 * CV-DAW Wearable LED Shirt Firmware — ESP32 DevKit (30-Pin) + WS2812B + BLE Low Energy (NUS)
 *
 * Hardware Connections (matches protoboard & final perfboard):
 *   - Power: 5V >= 2A Power Bank + 1000 uF (35V) electrolytic capacitor across +5V and GND.
 *   - Common GND shared between Power Bank, ESP32 GND, HV/LV Logic Level Shifter, and WS2812B strip.
 *   - SAFETY: Disconnect +5V/VIN from Power Bank while flashing ESP32 via PC USB!
 *   - Data Pin: ESP32 GPIO 13 -> LV1 (3.3V side of 4-ch shifter) -> HV1 (5V side) -> 330/470 Ohm -> WS2812B DIN.
 *
 * Strip Topology (single continuous WS2812B strip):
 *   [0 .. leftCount-1]                          : Left Arm
 *   [leftCount .. leftCount+chestCount-1]       : Chest / Back Core
 *   [leftCount+chestCount .. totalCount-1]      : Right Arm
 *
 * Supported `\n`-terminated ASCII Protocol (works identically over BLE NUS AND USB Serial 115200!):
 *   - Legacy / Direct:
 *       L<0..N>\n                     -> Set left arm fill index
 *       R<0..N>\n                     -> Set right arm fill index
 *       OFF\n                         -> Clear strip smoothly
 *   - Music-Synced Continuous State (~25-30 Hz from STM32MP257F-DK / Mac):
 *       S<L>,<R>,<E0..99>,<Pal0..5>,<Beat0|1>\n
 *   - 120 FPS Choreography Event FX:
 *       FX <PL|PR|WL|WR|UP|DR|XC|FR|KL|KR|CU>,<10..99>\n
 *   - Live Strip Calibration (no reflash needed when cutting the physical strip!):
 *       CFG <left>,<chest>,<right>,<revLeft0|1>\n
 *       TEST\n                        -> 3s Red=Left, Green=Chest, Blue=Right calibration check
 */

#include <Arduino.h>
#include <FastLED.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ======================= HARDWARE & DEFAULT STRIP CONFIG =======================
#define LED_DATA_PIN        13
#define MAX_STRIP_LEDS      150     // Upper RAM buffer ceiling (supports up to 2.5m @ 60 LEDs/m)
#define MAX_POWER_MILLIAMPS 1600    // Protects 5V 2A power bank from brownout (leaves 400 mA for ESP32 BLE)

#define DEFAULT_LEFT_LEDS   25      // 0..24  (Left Arm)
#define DEFAULT_CHEST_LEDS  10      // 25..34 (Chest / Back bridge)
#define DEFAULT_RIGHT_LEDS  25      // 35..59 (Right Arm)
// On a continuous Left-Wrist -> Left-Shoulder -> Chest -> Right-Shoulder -> Right-Wrist strip,
// setting DEFAULT_REV_LEFT = true makes index 0 on BOTH arms start at the shoulder and grow toward the wrist!
#define DEFAULT_REV_LEFT    true

// Nordic UART Service (NUS) BLE UUIDs — matches composer/wearable/protocol.py
#define BLE_DEVICE_NAME     "CV-DAW-SHIRT"
#define NUS_SERVICE_UUID    "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX_CHAR_UUID    "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX_CHAR_UUID    "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

// ======================= RUNTIME STATE =======================
static CRGB leds[MAX_STRIP_LEDS];

static uint16_t gLeftCount  = DEFAULT_LEFT_LEDS;
static uint16_t gChestCount = DEFAULT_CHEST_LEDS;
static uint16_t gRightCount = DEFAULT_RIGHT_LEDS;
static bool     gRevLeft    = DEFAULT_REV_LEFT;

static inline uint16_t totalLeds() {
    uint16_t sum = gLeftCount + gChestCount + gRightCount;
    return (sum <= MAX_STRIP_LEDS) ? sum : MAX_STRIP_LEDS;
}

// Live continuous state driven by STM32MP257F-DK / Mac
static uint8_t  gLeftFill   = 0;
static uint8_t  gRightFill  = 0;
static uint8_t  gEnergy     = 35;   // 0..99
static uint8_t  gPaletteId  = 1;    // 0..5 (6 CV-DAW musical styles)
static float    gBeatPulse  = 0.0f; // 0..1 decays smoothly after each BPM beat tick
static bool     gStripOff   = false;

// Active one-shot 120 FPS choreography FX state
enum FxKind : uint8_t {
    FX_NONE = 0,
    FX_PUNCH_L,
    FX_PUNCH_R,
    FX_WAVE_LR,
    FX_WAVE_RL,
    FX_RISE_UP,
    FX_DROP_BURST,
    FX_XCROSS_CUT,
    FX_FREEZE_ICE,
    FX_STOMP_L,
    FX_STOMP_R,
    FX_CUSTOM_BLOOM,
    FX_TEST_SEGMENTS
};

static FxKind   gActiveFx      = FX_NONE;
static uint32_t gFxStartMs     = 0;
static uint32_t gFxDurationMs  = 0;
static uint8_t  gFxIntensity   = 90;

// Thread-safe lock-free ring buffer for incoming ASCII lines from BLE callback + USB Serial
static constexpr size_t CMD_QUEUE_LEN = 16;
static constexpr size_t CMD_MAX_CHARS = 48;
static char             gCmdQueue[CMD_QUEUE_LEN][CMD_MAX_CHARS];
static volatile uint8_t gCmdHead = 0;
static volatile uint8_t gCmdTail = 0;
static portMUX_TYPE     gCmdMux  = portMUX_INITIALIZER_UNLOCKED;

static void enqueueLine(const char* line, size_t len) {
    if (len == 0) return;
    if (len >= CMD_MAX_CHARS) len = CMD_MAX_CHARS - 1;
    portENTER_CRITICAL(&gCmdMux);
    uint8_t nextHead = (gCmdHead + 1) % CMD_QUEUE_LEN;
    if (nextHead != gCmdTail) {
        memcpy(gCmdQueue[gCmdHead], line, len);
        gCmdQueue[gCmdHead][len] = '\0';
        gCmdHead = nextHead;
    }
    portEXIT_CRITICAL(&gCmdMux);
}

static bool dequeueLine(char* outBuf) {
    bool hasItem = false;
    portENTER_CRITICAL(&gCmdMux);
    if (gCmdTail != gCmdHead) {
        memcpy(outBuf, gCmdQueue[gCmdTail], CMD_MAX_CHARS);
        gCmdTail = (gCmdTail + 1) % CMD_QUEUE_LEN;
        hasItem = true;
    }
    portEXIT_CRITICAL(&gCmdMux);
    return hasItem;
}

// ======================= BLE NORDIC UART CALLBACKS =======================
static BLEServer* gBleServer = nullptr;
static bool       gBleConnected = false;
static char       gBleAccum[CMD_MAX_CHARS];
static size_t     gBleAccumLen = 0;

class ShirtServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer*) override {
        gBleConnected = true;
        Serial.println("[BLE] Connected to STM32 / Host!");
    }
    void onDisconnect(BLEServer* pServer) override {
        gBleConnected = false;
        Serial.println("[BLE] Disconnected — restarting BLE advertising...");
        pServer->startAdvertising();
    }
};

class NusRxCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pCharacteristic) override {
        String rx = pCharacteristic->getValue();
        for (size_t i = 0; i < rx.length(); ++i) {
            char c = rx[i];
            if (c == '\r') continue;
            if (c == '\n') {
                enqueueLine(gBleAccum, gBleAccumLen);
                gBleAccumLen = 0;
            } else if (gBleAccumLen + 1 < CMD_MAX_CHARS) {
                gBleAccum[gBleAccumLen++] = c;
            }
        }
    }
};

// ======================= SEGMENT PIXEL MAPPING HELPERS =======================
static inline uint16_t leftArmPixel(uint16_t idx) {
    // idx = 0 (shoulder) .. gLeftCount-1 (wrist) when gRevLeft is true
    return gRevLeft ? (gLeftCount - 1 - idx) : idx;
}

static inline uint16_t chestPixel(uint16_t idx) {
    return gLeftCount + idx;
}

static inline uint16_t rightArmPixel(uint16_t idx) {
    // idx = 0 (shoulder) .. gRightCount-1 (wrist)
    return gLeftCount + gChestCount + idx;
}

// 6 Genre-Authentic Color Palettes matching CV-DAW styles (0..5)
static void styleColors(uint8_t pal, CRGB& primary, CRGB& secondary, CRGB& accent) {
    switch (pal % 6) {
        case 0: // Dream Ambient: Ocean Cyan & Deep Violet
            primary = CRGB(0, 190, 255); secondary = CRGB(130, 40, 255); accent = CRGB(180, 245, 255); break;
        case 1: // Melodic House: Club Gold, Magenta & Cyan
            primary = CRGB(255, 165, 20); secondary = CRGB(255, 25, 145); accent = CRGB(60, 245, 255); break;
        case 2: // Lo-Fi / Nu-Jazz: Warm Sunset Amber & Rose
            primary = CRGB(255, 115, 35); secondary = CRGB(215, 55, 110); accent = CRGB(255, 210, 140); break;
        case 3: // Latin / Afrobeat: Fiery Coral & Emerald
            primary = CRGB(255, 65, 25); secondary = CRGB(15, 230, 110); accent = CRGB(255, 225, 45); break;
        case 4: // Retro Synthwave: Outrun Neon Pink & Cyber Blue
            primary = CRGB(255, 15, 165); secondary = CRGB(15, 210, 255); accent = CRGB(190, 80, 255); break;
        default: // Liquid DnB: Electric Lime & Ice Cyan
            primary = CRGB(85, 255, 45); secondary = CRGB(0, 215, 255); accent = CRGB(235, 255, 255); break;
    }
}

// ======================= COMMAND PARSER =======================
static void startFx(FxKind kind, uint32_t durationMs, uint8_t intensity) {
    gStripOff     = false;
    gActiveFx     = kind;
    gFxStartMs    = millis();
    gFxDurationMs = durationMs;
    gFxIntensity  = intensity;
}

static void parseCommand(const char* line) {
    if (strcmp(line, "OFF") == 0) {
        gStripOff = true;
        gActiveFx = FX_NONE;
        gLeftFill = 0;
        gRightFill = 0;
        FastLED.clear(true);
        return;
    }
    if (strcmp(line, "TEST") == 0) {
        startFx(FX_TEST_SEGMENTS, 3500, 99);
        return;
    }
    if (strncmp(line, "CFG ", 4) == 0) {
        int l = 0, c = 0, r = 0, rev = 1;
        if (sscanf(line + 4, "%d,%d,%d,%d", &l, &c, &r, &rev) >= 3) {
            if (l >= 1 && c >= 1 && r >= 1 && (l + c + r) <= MAX_STRIP_LEDS) {
                gLeftCount  = (uint16_t)l;
                gChestCount = (uint16_t)c;
                gRightCount = (uint16_t)r;
                gRevLeft    = (rev != 0);
                FastLED.clear(true);
                Serial.printf("[CFG] Left=%u Chest=%u Right=%u RevL=%d\n", gLeftCount, gChestCount, gRightCount, (int)gRevLeft);
            }
        }
        return;
    }
    if (line[0] == 'L' && line[1] >= '0' && line[1] <= '9') {
        gStripOff = false;
        gLeftFill = (uint8_t)constrain(atoi(line + 1), 0, (int)gLeftCount);
        return;
    }
    if (line[0] == 'R' && line[1] >= '0' && line[1] <= '9') {
        gStripOff = false;
        gRightFill = (uint8_t)constrain(atoi(line + 1), 0, (int)gRightCount);
        return;
    }
    if (line[0] == 'S') {
        int l = 0, r = 0, e = 0, pal = 0, beat = 0;
        if (sscanf(line + 1, "%d,%d,%d,%d,%d", &l, &r, &e, &pal, &beat) == 5) {
            gStripOff  = false;
            gLeftFill  = (uint8_t)constrain(l, 0, (int)gLeftCount);
            gRightFill = (uint8_t)constrain(r, 0, (int)gRightCount);
            gEnergy    = (uint8_t)constrain(e, 0, 99);
            gPaletteId = (uint8_t)constrain(pal, 0, 5);
            if (beat) {
                gBeatPulse = 1.0f;
            }
        }
        return;
    }
    if (strncmp(line, "FX ", 3) == 0) {
        char code[4] = {0};
        int lvl = 90;
        if (sscanf(line + 3, "%2s,%d", code, &lvl) >= 1) {
            uint8_t inten = (uint8_t)constrain(lvl, 10, 99);
            if      (strcmp(code, "PL") == 0) startFx(FX_PUNCH_L,      340, inten);
            else if (strcmp(code, "PR") == 0) startFx(FX_PUNCH_R,      340, inten);
            else if (strcmp(code, "WL") == 0) startFx(FX_WAVE_LR,      780, inten);
            else if (strcmp(code, "WR") == 0) startFx(FX_WAVE_RL,      780, inten);
            else if (strcmp(code, "UP") == 0) startFx(FX_RISE_UP,     1200, inten);
            else if (strcmp(code, "DR") == 0) startFx(FX_DROP_BURST,   520, inten);
            else if (strcmp(code, "XC") == 0) startFx(FX_XCROSS_CUT,   900, inten);
            else if (strcmp(code, "FR") == 0) startFx(FX_FREEZE_ICE,   950, inten);
            else if (strcmp(code, "KL") == 0) startFx(FX_STOMP_L,      300, inten);
            else if (strcmp(code, "KR") == 0) startFx(FX_STOMP_R,      300, inten);
            else if (strcmp(code, "CU") == 0) startFx(FX_CUSTOM_BLOOM, 820, inten);
        }
    }
}

// ======================= 120 FPS STUDIO RENDERER =======================
static void renderFrame(uint32_t nowMs) {
    if (gStripOff) return;

    uint16_t nTotal = totalLeds();
    CRGB primary, secondary, accent;
    styleColors(gPaletteId, primary, secondary, accent);

    // 1. Calibration test pattern overrides everything cleanly (Red=Left, Green=Chest, Blue=Right).
    if (gActiveFx == FX_TEST_SEGMENTS) {
        if (nowMs - gFxStartMs >= gFxDurationMs) {
            gActiveFx = FX_NONE;
        } else {
            for (uint16_t i = 0; i < gLeftCount; ++i)  leds[leftArmPixel(i)]  = CRGB(180, 0, 0);
            for (uint16_t i = 0; i < gChestCount; ++i) leds[chestPixel(i)]    = CRGB(0, 180, 0);
            for (uint16_t i = 0; i < gRightCount; ++i) leds[rightArmPixel(i)] = CRGB(0, 0, 180);
            // Bright white boundary markers at the very tips of both wrists!
            leds[leftArmPixel(gLeftCount - 1)]   = CRGB(220, 220, 220);
            leds[rightArmPixel(gRightCount - 1)] = CRGB(220, 220, 220);
            FastLED.show();
            return;
        }
    }

    // Smooth trail persistence for organic motion glow
    fadeToBlackBy(leds, nTotal, 55);

    // 2. Base Layer — Chest Core BPM Beat Pulse + Energy Glow
    float eNorm = gEnergy / 99.0f;
    uint8_t chestBri = (uint8_t)constrain((int)(32.0f + 125.0f * eNorm + 98.0f * gBeatPulse), 16, 255);
    float chestCenter = (gChestCount - 1) * 0.5f;
    for (uint16_t i = 0; i < gChestCount; ++i) {
        float distNorm = (gChestCount > 1) ? (fabsf(i - chestCenter) / (chestCenter + 0.5f)) : 0.0f;
        CRGB c = blend(accent, primary, (uint8_t)(distNorm * 200.0f));
        uint8_t falloff = (uint8_t)(chestBri * (1.0f - 0.35f * distNorm * (1.0f - gBeatPulse)));
        c.nscale8_video(falloff);
        leds[chestPixel(i)] += c;
    }
    gBeatPulse *= 0.88f; // Crisp ~110ms beat decay at 120 FPS

    // 3. Base Layer — Left & Right Arm Musical Fill Bars (grow from shoulder -> wrist)
    uint8_t armBri = (uint8_t)constrain((int)(65.0f + 160.0f * eNorm), 50, 240);
    for (uint16_t i = 0; i < gLeftFill && i < gLeftCount; ++i) {
        uint8_t grad = (uint8_t)((uint32_t)i * 255 / max((uint16_t)1, gLeftCount));
        CRGB c = blend(primary, secondary, grad);
        c.nscale8_video(armBri);
        leds[leftArmPixel(i)] = c;
    }
    for (uint16_t i = 0; i < gRightFill && i < gRightCount; ++i) {
        uint8_t grad = (uint8_t)((uint32_t)i * 255 / max((uint16_t)1, gRightCount));
        CRGB c = blend(secondary, accent, grad);
        c.nscale8_video(armBri);
        leds[rightArmPixel(i)] = c;
    }

    // 4. Overlay Layer — Active 120 FPS Choreography Event FX!
    if (gActiveFx != FX_NONE) {
        uint32_t elapsed = nowMs - gFxStartMs;
        if (elapsed >= gFxDurationMs) {
            gActiveFx = FX_NONE;
        } else {
            float prog = (float)elapsed / (float)gFxDurationMs; // 0.0 -> 1.0
            uint8_t fxBri = (uint8_t)constrain((int)(gFxIntensity * 2.55f * (1.0f - 0.35f * prog)), 40, 255);

            switch (gActiveFx) {
                case FX_PUNCH_L: {
                    // Explosive comet shooting from Chest -> Left Wrist!
                    float head = prog * (gLeftCount - 1);
                    for (uint16_t i = 0; i < gLeftCount; ++i) {
                        float d = fabsf((float)i - head);
                        if (d < 4.5f) {
                            CRGB c = blend(CRGB::White, primary, (uint8_t)(d * 55.0f));
                            c.nscale8_video((uint8_t)(fxBri * (1.0f - d / 4.5f)));
                            leds[leftArmPixel(i)] += c;
                        }
                    }
                    break;
                }
                case FX_PUNCH_R: {
                    // Explosive comet shooting from Chest -> Right Wrist!
                    float head = prog * (gRightCount - 1);
                    for (uint16_t i = 0; i < gRightCount; ++i) {
                        float d = fabsf((float)i - head);
                        if (d < 4.5f) {
                            CRGB c = blend(CRGB::White, accent, (uint8_t)(d * 55.0f));
                            c.nscale8_video((uint8_t)(fxBri * (1.0f - d / 4.5f)));
                            leds[rightArmPixel(i)] += c;
                        }
                    }
                    break;
                }
                case FX_WAVE_LR:
                case FX_WAVE_RL: {
                    // Liquid traveling wave across the ENTIRE physical strip (Left Wrist <-> Chest <-> Right Wrist)!
                    float pos = (gActiveFx == FX_WAVE_LR) ? (prog * (nTotal - 1)) : ((1.0f - prog) * (nTotal - 1));
                    for (uint16_t p = 0; p < nTotal; ++p) {
                        float d = fabsf((float)p - pos);
                        if (d < 7.0f) {
                            CRGB c = CHSV((uint8_t)(p * 4 + elapsed / 3), 200, (uint8_t)(fxBri * (1.0f - d / 7.0f)));
                            leds[p] += c;
                        }
                    }
                    break;
                }
                case FX_RISE_UP: {
                    // Warm aurora shimmer expanding symmetrically from Chest outward to both wrists!
                    uint16_t reachL = (uint16_t)(prog * gLeftCount);
                    uint16_t reachR = (uint16_t)(prog * gRightCount);
                    for (uint16_t i = 0; i < reachL; ++i) {
                        leds[leftArmPixel(i)] += CHSV((uint8_t)(140 + i * 3), 180, fxBri / 2);
                    }
                    for (uint16_t i = 0; i < reachR; ++i) {
                        leds[rightArmPixel(i)] += CHSV((uint8_t)(140 + i * 3), 180, fxBri / 2);
                    }
                    break;
                }
                case FX_DROP_BURST: {
                    // Crisp white impact flash (<90ms) -> expanding shockwave rings across chest & arms!
                    if (prog < 0.18f) {
                        fill_solid(leds, nTotal, CRGB(fxBri, fxBri, fxBri));
                    } else {
                        float ring = (prog - 0.18f) / 0.82f;
                        uint16_t idxL = (uint16_t)(ring * (gLeftCount - 1));
                        uint16_t idxR = (uint16_t)(ring * (gRightCount - 1));
                        leds[leftArmPixel(idxL)]  = CRGB::White;
                        leds[rightArmPixel(idxR)] = CRGB::White;
                    }
                    break;
                }
                case FX_XCROSS_CUT: {
                    // Deep midnight-violet chest heartbeat while arms stay dark.
                    uint8_t pulse = beatsin8(90, 40, 210);
                    for (uint16_t i = 0; i < gChestCount; ++i) {
                        leds[chestPixel(i)] = CRGB((pulse * 3) / 5, 10, pulse);
                    }
                    break;
                }
                case FX_FREEZE_ICE: {
                    // Sparkling ice-crystal shimmer hold!
                    if (random8() < 160) {
                        leds[random16(nTotal)] += CRGB(200, 245, 255);
                    }
                    break;
                }
                case FX_STOMP_L:
                case FX_STOMP_R: {
                    // Heavy bass thump across Chest + corresponding shoulder!
                    CRGB boom = (gActiveFx == FX_STOMP_R) ? primary : secondary;
                    boom.nscale8_video(fxBri);
                    for (uint16_t i = 0; i < gChestCount; ++i) leds[chestPixel(i)] += boom;
                    uint16_t shSpan = min((uint16_t)6, (gActiveFx == FX_STOMP_R) ? gRightCount : gLeftCount);
                    for (uint16_t i = 0; i < shSpan; ++i) {
                        if (gActiveFx == FX_STOMP_R) leds[rightArmPixel(i)] += boom;
                        else                         leds[leftArmPixel(i)]  += boom;
                    }
                    break;
                }
                case FX_CUSTOM_BLOOM: {
                    // Prismatic rainbow spiral bloom for custom trained Tier 2 ML/DTW dance moves!
                    for (uint16_t p = 0; p < nTotal; ++p) {
                        uint8_t wave = sin8((uint8_t)(p * 14 - elapsed / 2));
                        leds[p] += CHSV((uint8_t)(p * 6 + elapsed / 3), 230, scale8(wave, fxBri));
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    FastLED.show();
}

// ======================= USB SERIAL READER (PROTOBOARD BENCH TEST) =======================
static char   gSerAccum[CMD_MAX_CHARS];
static size_t gSerAccumLen = 0;

static void pollUsbSerial() {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            enqueueLine(gSerAccum, gSerAccumLen);
            gSerAccumLen = 0;
        } else if (gSerAccumLen + 1 < CMD_MAX_CHARS) {
            gSerAccum[gSerAccumLen++] = c;
        }
    }
}

// ======================= ARDUINO SETUP & LOOP =======================
void setup() {
    Serial.begin(115200);
    delay(150);
    Serial.println("\n=== CV-DAW Wearable LED Shirt (BLE NUS + FastLED GPIO 13) ===");

    // Initialize WS2812B strip with 1.6A hardware-safe power cap for 5V 2A power banks!
    FastLED.addLeds<WS2812B, LED_DATA_PIN, GRB>(leds, MAX_STRIP_LEDS).setCorrection(TypicalLEDStrip);
    FastLED.setMaxPowerInVoltsAndMilliamps(5, MAX_POWER_MILLIAMPS);
    FastLED.clear(true);

    // Initialize BLE Low Energy Nordic UART Service (NUS)
    BLEDevice::init(BLE_DEVICE_NAME);
    gBleServer = BLEDevice::createServer();
    gBleServer->setCallbacks(new ShirtServerCallbacks());

    BLEService* pService = gBleServer->createService(NUS_SERVICE_UUID);
    BLECharacteristic* pRxChar = pService->createCharacteristic(
        NUS_RX_CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
    );
    pRxChar->setCallbacks(new NusRxCallbacks());

    BLECharacteristic* pTxChar = pService->createCharacteristic(
        NUS_TX_CHAR_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pTxChar->addDescriptor(new BLE2902());

    pService->start();
    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(NUS_SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);  // Fast BLE connection interval (~7.5-15 ms latency!)
    BLEDevice::startAdvertising();

    Serial.println("[READY] Advertising BLE Low Energy as 'CV-DAW-SHIRT' (also listening on USB Serial 115200).");
    // Brief startup greeting sweep so you see immediately on protoboard that GPIO 13 works!
    startFx(FX_RISE_UP, 900, 75);
}

static uint32_t gLastFrameMs = 0;

void loop() {
    pollUsbSerial();

    char cmd[CMD_MAX_CHARS];
    while (dequeueLine(cmd)) {
        parseCommand(cmd);
    }

    uint32_t now = millis();
    if (now - gLastFrameMs >= 8) { // ~120 FPS smooth local animation rate
        gLastFrameMs = now;
        renderFrame(now);
    }
}
