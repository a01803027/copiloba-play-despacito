# CV-DAW — System Capabilities Reference

> **Living Document**: This file is the single source of truth for everything CV-DAW currently recognizes, synthesizes, and controls. Keep it updated whenever moves, thresholds, styles, or DSP features are tuned.

---

## 1. High-Level Pipeline

CV-DAW is a real-time full-body interactive music production instrument:

1. **Pose-Only Computer Vision (~28–30 FPS)** — MediaPipe Tasks `PoseLandmarker` (`models/pose_landmarker_full.task` or `lite`) running on a dedicated `AsyncPoseWorker` thread (`composer/vision/camera.py`) so camera inference never stalls the 60 Hz musical clock.
2. **Continuous Studio Macros + 11-Move Biomechanical Recognizer** — Scale- and translation-invariant 3D kinematics (`composer/kinematics/body.py`) with One-Euro adaptive filtering, 300 ms occlusion grace period, multi-source BPM auto-sync (`composer/rhythm.py`), and a rolling 1.5 s spatiotemporal move recognizer (`composer/moves/recognizer.py`).
3. **Dynamic Studio Composer** — Structured 4-bar `A → A' → B → A''` phrasing in a stable singable register (`composer/arrangement.py`), 6 modular JSON styles (`composer/styles/presets/`), and per-move studio sound translation (`composer/brain.py`).
4. **Lock-Free C++20 Studio DSP Engine** — Communicates over POSIX shared memory (`/cv_daw_shm`, ABI v3, 4544 bytes, zero locks, zero RT heap allocations) and plays via **CoreAudio** on macOS or **ALSA** on Linux ARM64 (Raspberry Pi 5).

---

## 2. Recognized Full-Body & Leg Moves (11 Core Moves)

All moves run purely from the 33-point 3D pose skeleton (zero fist/hand-model overhead).

### Upper & Full-Body Moves (8)

| # | Move ID | How to Perform | Musical Effect | Current Tuning |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **`HARD_PUNCH`** *(Right / Left)* | Snap a bent elbow straight at high speed (forward, cross, or side punch). Smooth arm waving is ignored. | **Right**: Immediate (`8 ms`) Octave Power Synth Stab + Crisp Genre Clap/Snare Crack + `+3.2 kHz` lead filter flash.<br>**Left**: Deep 808/Sub-Bass Boom + Heavy Kick Slam + low chord hit. | Elbow snap `≥0.17` (`~30°`) to straight `≤0.28`, `speed ≥4.2` (or power jab `≥7.5` + decel `≤-48`); cooldown `0.55 s`. |
| 2 | **`DOUBLE_ARM_RISE`** *(Continuous Soundwave)* | Raise both arms symmetrically together from waist/chest overhead. | Launches the dedicated C++ **Sweeping Soundwave Riser** (`MIDI 120`: `+1 octave` pitch glide + resonant `360 → 9160 Hz` stereo swell) + 3-note ascending shimmer chord + continuous `soundwave_rise` filter lift while held overhead. | Both wrists sync (`\|lh-rh\| ≤0.25`), `soundwave_rise ≥0.18`; cooldown `1.50 s`. Releases smoothly when arms drop `<0.08`. |
| 3 | **`OVERHEAD_DROP`** | Slam both arms rapidly down from overhead (`>0.64`) below chest (`≤0.48`). | Releases any active Soundwave Riser and fires a synchronized **Beat Drop** (`Kick 36 + Crash 49 + Deep Sub Root + Full Strummed Power Chord`) entering `DROP` section. | Both `vy ≤ -1.6`; cooldown `0.90 s`. |
| 4 | **`ARMS_CROSS_CUT`** *(Continuous State)* | Cross wrists/forearms in an **"X"** in front of your chest (mirror-invariant). | **DJ Breakdown Cut**: mutes drums & bass smoothly, sweeps chords warm (`≤1300 Hz`) with lush reverb & delay. Uncrossing triggers an immediate `OPEN_BURST` drop back into the groove. | Hold `≥0.18 s` at chest height (`0.20..0.80`). |
| 5 | **`OPEN_BURST`** | Throw both arms wide open rapidly from close to your chest. | Fires a euphoric **3-Note Ascending Harmonic Fanfare + Open Hat Splash (`46`)** and launches full `DROP` / `CHORUS` energy. | `expansion` jumps `≤0.36 → ≥0.68`, rate `≥0.90/s`; cooldown `1.00 s`. |
| 6 | **`ARM_WAVE`** *(L→R or R→L)* | Perform an arm wave so peaks ripple sequentially across `L_wrist → L_elbow → R_elbow → R_wrist` (or reverse). | Plays a **6-Note Liquid Stereo Arpeggio Ripple** in the active chord/scale while sweeping lead stereo pan smoothly Left→Right (or Right→Left) over `0.75 s`. | Monotonic neighbor peak lag `25–420 ms`, span `0.18–1.15 s`, amplitude `≥0.14`; cooldown `1.25 s`. |
| 7 | **`BODY_FREEZE`** *(Continuous State)* | Dance actively (`energy ≥0.44`), then abruptly **freeze** completely still. | **Dramatic Echo Tail Freeze**: instantly cuts drums, bass, and sequencer notes while ringing one high harmonic bell into a lush dotted-8th delay (`0.86`) & reverb tail. Resumes automatically when you move (`speed > 0.78`). | Stillness (`speed < 0.36`) held `≥0.28 s`; cooldown `1.20 s`. |
| 8 | **`SIDE_STEP_GROOVE`** *(Continuous State)* | Dance side-to-side with periodic hip sway and rhythmic bounce. | Locks into full **Groove Pocket**, upgrades kick/hat pattern to full groove, adds syncopated octave bass movement, and reinforces BPM lock. | Hip sway span `≥0.48`, `periodicity ≥0.44`, `energy ≥0.18`. |

### Leg & Lower-Body Moves (3 — Works with Knees `25/26` OR Ankles `27/28`)

| # | Move ID | How to Perform | Musical Effect | Current Tuning |
| :---: | :--- | :--- | :--- | :--- |
| 9 | **`KNEE_STOMP`** *(Right / Left)* | Lift knee/foot high above floor baseline, then stomp down sharply. | **Right**: Immediate Heavy Kick (`36`) + Sub-Bass Root.<br>**Left**: Immediate Crisp Genre Snare/Rimshot (`38`) + Bass 5th. | Arms on lift `≥0.38`, fires on landing (`≤0.24` or `vy ≤ -1.25`); cooldown `0.45 s`. |
| 10 | **`SQUAT_DIVE` & `SQUAT_RISE`** *(Continuous + Release)* | Dip hips low into a squat, hold briefly, then spring back up. | **While Low (`SQUAT_DIVE`)**: Underwater low-pass filter dive (chords/lead/drums sweep deep & warm while bass swells `+24%`).<br>**Spring Up (`SQUAT_RISE`)**: Explosive filter snap-open + Kick + Sub + Bright Strummed Chord pop! | Squat depth from `body_level < 0.50`; arms at depth `≥0.42` for `0.14 s`; fires `SQUAT_RISE` at `body_level ≥0.56` or `hip_bounce_vel ≥1.05`; cooldown `0.80 s`. |
| 11 | **`JUMP_HIT`** | Hop/jump upward noticeably above your standing baseline. | Uplifting **Air Crash (`49`) + Heavy Kick (`36`) + High Sparkle Octave Lead** and enters `DROP` momentum. | `hip_bounce_vel ≥1.65` & `body_level ≥0.78`; cooldown `0.90 s`. |

---

## 3. Continuous Full-Body Studio Macros

Between trigger moves, your entire body continuously automates studio mix parameters without creating stray pops:

- **Overall Energy**: Smoothly transitions song sections (`INTRO → GROOVE → CHORUS`) with hysteresis; complete stillness leaves a soft, warm breathing ambient pad alive.
- **Left Arm Height**: Controls chord voicing openness (voice-led inversions) and chord/pad low-pass filter cutoff.
- **Right Arm Height**: Controls lead filter brightness, velocity expression, and **hysteretic 3-zone octave register** (`low < 0.24`, `mid 0.24..0.76`, `high > 0.76`, updated on quarter-note beats so melodies stay singable and never scramble mid-phrase).
- **Arm Expansion (Width)**: Swells stereo chorus/pad fullness, lead presence, and opens hi-hats (`46`).
- **Head Tilt (Left / Right)**: Steers lead stereo panning, boosts tempo-synced ping-pong delay throws, and engages **parallel dual-voice harmony** (`+3rd` right, `+5th` left when `|head_tilt| > 0.28`).
- **Head Pitch (Looking Up)**: Brightens chord & lead filters and shifts alternating bars toward bright chord voicings (`bright_chord`).
- **Torso Lean & Twist**: Lean increases filter resonance and shifts pad panning; sideways twist enriches reverb depth and pulls harmony toward `tension_chord`.
- **Shoulders**: Deliberate shoulder shrug (`≥0.65`, ear-verified so looking down never fakes a shrug) builds rising filter tension + 8th-note snare `BUILDUP`; releasing triggers a `DROP`. Alternating shoulder bobbing adds octave bass lifts and subtle ghost snares.
- **Hips & Legs (Continuous)**: Hip sway modulates groove swing and bass momentum; wide stance (`stance_width`) warms and opens bass cutoff; leg activity (`leg_energy`, using knees automatically when ankles are off-screen) drives bass & drum energy.
- **Auto-Synced Multi-Source BPM**: 30 Hz resampled autocorrelation (`composer/rhythm.py`) across `legs`, `hip`, `head`, `shoulders`, and `arms`, peak-latched on every frame (`threshold = 0.48`, `2.0 s` hold, `±2 BPM` smooth slew inside the active genre's tempo range; toggleable with `T`).

---

## 4. Musical Styles & Genre-Authentic Drum Kits

Styles are modular JSON blueprints in `composer/styles/presets/` (drop a new `.json` file there and it is discovered automatically).

| Key | Style | BPM Range (Default) | Synth Presets (Chords / Lead / Bass) | Dedicated C++ Drum Kit (`limb 3`) |
| :---: | :--- | :---: | :--- | :--- |
| **`1`** | **Dream Ambient** (`ambient`) | `66–86` (`74`) | Analog Pad (`8`) / Marimba-Bell (`14`) / Moog Sub (`11`) | Kit `17`: Organic Felt Kick & Soft Rim Tap |
| **`2`** | **Melodic House / Anjuna** (`house`) | `118–130` (`124`) | Analog Pad (`8`) / Melodic Pluck (`13`) / Moog Sub (`11`) | Kit `12`: Club 909 Kick, Layered Clap, Silky Hats |
| **`3`** | **Lo-Fi / Nu-Jazz** (`lofi`) | `76–92` (`84`) | Studio Rhodes (`9`) / Rhodes (`9`) / Moog Sub (`11`) | Kit `13`: Dusty Boom-Bap Kick, Woody Vinyl Snare |
| **`4`** | **Latin / Afrobeat** (`latin`) | `94–108` (`100`) | Studio Rhodes (`9`) / Marimba-Nylon (`14`) / Moog Sub (`11`) | Kit `14`: Round Kick, Cross-Stick Rim/Conga, Shaker |
| **`5`** | **Retro Synthwave / 80s** (`synthwave`) | `106–122` (`114`) | Analog Pad (`8`) / Prophet Supersaw (`10`) / Moog Sub (`11`) | Kit `15`: Big 80s Kick, Gated LinnDrum Snare |
| **`6`** | **Liquid Drum & Bass** (`dnb`) | `160–176` (`168`) | Analog Pad (`8`) / Studio Rhodes (`9`) / Moog Sub (`11`) | Kit `16`: Tight Fast Breakbeat Kick & Crisp Snare |

---

## 5. C++20 Studio DSP Engine (`realtime/`)

- **Real-Time Guarantees**: Zero heap allocations, zero mutexes, zero syscalls inside `AudioEngine::processBlock()` and `data_callback` (verified by `modules45` RT test). Fixed `128`-frame blocks at `48 kHz` stereo float32.
- **Expressive Synthesis**:
  - Exponential analog-curve `Adsr` envelopes & 12-voice release-first voice stealing per bus (`voices_per_limb = 12`).
  - **Per-Voice Resonant Filter Envelopes** (`CytomicSvf` + velocity-sensitive `filt_env`) on all studio synths (`Presets 8, 9, 10, 11, 13, 14`).
  - Dedicated **Sweeping Soundwave Riser** voice (`MIDI 120` on synth buses).
- **Studio Mixing & Mastering Chain**:
  - `210–240 Hz` high-pass mud carve + `PresenceEq` midrange clarity boost on Chords & Lead.
  - Juno-style quadrature `StereoChorus` on Chords/Keys (`limb 0`).
  - Smooth attack/release sidechain ducking (`~2.6 ms` attack, `~150 ms` release — zero step clicks).
  - Tempo-synced dotted-eighth ping-pong `StereoDelay` + damped 4-channel Hadamard FDN `Reverb` (dry drums: `reverb = 0.02`).
  - `MonoSubCrossover` (`<120 Hz` centered tight mono + `1.14x` stereo widening above 120 Hz) → feed-forward SSL-style `MasterGlueCompressor` (2.4:1) → ARM NEON vectorized `limit_stereo`.

---

## 6. Live Controls, CLI Modes & Diagnostic Flight Recorder

### Live Keyboard Shortcuts (OpenCV Window)
| Key | Action |
| :--- | :--- |
| **`1` – `6`** | Jump directly to style (`Ambient`, `House`, `Lo-Fi`, `Latin`, `Synthwave`, `DnB`) |
| **`N` / `]`** or **`P` / `[`** | Next / previous musical style |
| **`Space` / `B`** | Toggle backing drum beat ON / OFF |
| **`T`** | Lock / unlock automatic body BPM sync |
| **`q`** | Quit cleanly |

### Common CLI Commands
- **Live Camera Performance**:
  ```bash
  .venv-mac/bin/python -m composer.app --mode camera --style house --launch-engine
  ```
- **Live Diagnostic Flight Recorder** (captures synchronized live CoreAudio `.wav`, per-frame `.jsonl` telemetry, and automatic `_report.md` with spectral balance, correlations, landmark visibility, and detected move counts):
  ```bash
  .venv-mac/bin/python -m composer.app --mode camera --style house --launch-engine --diagnose 30
  ```
- **Re-analyze an existing session**:
  ```bash
  .venv-mac/bin/python -m composer.diagnose artifacts/diag_<style>_<timestamp>.jsonl
  ```
- **Offline Mock / Video Testing**:
  ```bash
  .venv-mac/bin/python -m composer.app --mode mock --style lofi --launch-engine --offline-audio --no-window --diagnose 10
  .venv-mac/bin/python -m composer.app --mode video --video brazo.mp4 --launch-engine --seconds 10
  ```
- **Full Automated Test Suite** (`6/6` CTest + `37/37` Python unit tests):
  ```bash
  /opt/homebrew/bin/cmake --build realtime/build-mac --parallel 4
  /opt/homebrew/bin/ctest --test-dir realtime/build-mac --output-on-failure
  ```

---

## 7. Tuning & Change Log

| Date | Milestone / Tuning Update |
| :--- | :--- |
| **2026-11-21** | Upgraded C++20 DSP to Studio v2 (per-voice filter envelopes, exponential ADSR, Juno chorus, presence EQ, mono-sub, SSL glue compressor, 6 genre-specific drum kits `12–17`); eliminated hiss & sidechain/voice-stealing pops; refactored Python into modular `vision/`, `kinematics/`, `ui/`, `styles/`. |
| **2026-11-22** | Removed clunky `HandLandmarker` fist controls (restoring ~28–30 FPS pose tracking); added `composer/moves/` with **11 core full-body & leg moves** (`HARD_PUNCH`, `DOUBLE_ARM_RISE`, `OVERHEAD_DROP`, `ARMS_CROSS_CUT`, `OPEN_BURST`, `ARM_WAVE`, `BODY_FREEZE`, `SIDE_STEP_GROOVE`, `KNEE_STOMP`, `SQUAT_DIVE`/`SQUAT_RISE`, `JUMP_HIT`) and the C++20 **Sweeping Soundwave Riser** (`MIDI 120`). |
