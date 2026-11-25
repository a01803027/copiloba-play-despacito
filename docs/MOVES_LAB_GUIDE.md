# CV-DAW Moves Lab — Complete Field Guide

**Moves Lab** (`composer/moves/lab.py`) is CV-DAW's standalone, **zero-audio** workbench for debugging vision tracking, testing built-in move responsiveness, recording labeled dance datasets, and training/testing custom neural dance models (**Keras 3 1D-TCN** and **Few-Shot DTW**) before connecting them to the C++20 music engine.

> **STM32MP257F-DK Compatibility Rule**: Any custom move, threshold tweak, or model trained via Moves Lab **must** stay compatible with the STM32MP257F-DK target: rely only on full-body 2D silhouettes (15 joints shared with MoveNet INT8 COCO-17 — no finger/face gestures), export to pure-NumPy `models/dance_tcn.npz` (and optionally `--export-tflite-int8`), and keep `eval_stride >= 2` so Cortex-A35 Core 0 CPU load stays `< 2%`. See [STM32MP257F_DK.md](STM32MP257F_DK.md).

---

## 1. Quick Start Commands

```bash
# 1. Live debugger (zero audio — check FPS, anti-wiggle guard & Tier 1 moves):
.venv-mac/bin/python -m composer.moves.lab

# 2. Offline replay benchmark against any recorded diagnostic .jsonl session:
.venv-mac/bin/python -m composer.moves.lab --replay artifacts/diag_house_20261123_182101.jsonl

# 3. Guided clip recorder (saves labeled .npz clips into datasets/moves/<label>/):
.venv-mac/bin/python -m composer.moves.lab --record-move body_roll --clips 15 --clip-seconds 1.2
.venv-mac/bin/python -m composer.moves.lab --record-move idle_random --clips 20 --clip-seconds 1.2

# 4. Test immediately with Few-Shot DTW (zero training required — works from 3-5 clips!):
.venv-mac/bin/python -m composer.moves.lab --dtw-dir datasets/moves

# 5. Train the embedded Keras 3 1D-TCN & export pure-NumPy models/dance_tcn.npz (~62 KB):
.venv-mac/bin/python ml/train_move_net.py --data datasets/moves --out models/dance_tcn.npz

# 6. Test your trained neural network live in Moves Lab:
.venv-mac/bin/python -m composer.moves.lab --tcn-model models/dance_tcn.npz
```

---

## 2. Reading the Moves Lab HUD

When running live, Moves Lab opens a clean OpenCV window with **zero audio engine overhead**:

| HUD Area | What It Shows | What Good Looks Like |
| :--- | :--- | :--- |
| **Top Bar** | Pose model (`LITE`/`FULL`), live **FPS**, frame latency (`ms`), **GLITCHES BLOCKED**, **ARB SUPPRESSED** | `28–60 FPS` (`16–35 ms`). If `GLITCHES BLOCKED` rises rapidly while standing still, improve room lighting or step back slightly. |
| **Red Skeleton Joints** | Joints clamped on that frame by the Tier 0 **Anti-Wiggle `PoseSanitizer`** | Occasional red flashes during wild blur are normal — that means phantom spikes were blocked before triggering sounds. |
| **Top Gauges** | `L_Spd`, `R_Spd`, `L_Thrust`, `R_Thrust`, `Rise`, `Squat`, `Exp` | On a crisp punch, `L_Thrust` or `R_Thrust` spikes past `~0.45` (`2.2 torso/s`) while smooth arm waving stays near `0.0–0.15`. |
| **Center Flash Banner** | Instant badge for the most recent recognized move & intensity (`0.65–1.00`) | Should snap immediately on impact with no lingering double-triggers. |
| **Bottom-Left Ticker** | Scrolling log of the last 5 triggered moves with exact timestamps | Check rapid `L-R-L-R` boxing combos and `ARM_WAVE` triggers here. |
| **Bottom-Right Panel** | **TIER 2 ML / DTW** live probability bars (shown when `--tcn-model` or `--dtw-dir` is active) | Bars turn **green** when crossing the `78%` enter threshold; `idle_random` should dominate (`>70%`) whenever you aren't doing a move. |

**Live Hotkeys:**
- **`r`** — Reset all counters, baseline height, and recent move ticker.
- **`q`** — Quit and print a full session breakdown table in the terminal.

---

## 3. Recording a Rock-Solid Dance Dataset (Multi-Height & Multi-Position Best Practices)

Every recorded clip stores a sequence of **24 canonical hip-centered, torso-scaled 2D features** (`PoseSanitizer` output):
- **Automatically invariant** to where you stand in the frame (left/right/center), camera distance, overall height, and 4:3 vs. 16:9 camera aspect ratio.
- **Automatically knee-safe**: if a tall performer stands closer and their knees leave the frame, `PoseSanitizer` substitutes clean standing neutral coordinates instead of MediaPipe garbage.

Even with strong normalization and automatic biomechanical data augmentation during training, **dataset quality determines 90% of real-world accuracy**. Follow this protocol:

### A. Always Record an `idle_random` (Negative / Null) Class — Most Important Rule!
Without a negative class, any neural network is forced to pick one of your dance moves *even when you're just walking, scratching your head, or resting*.
- Record **at least 1.5×–2× as many `idle_random` clips** as each dance move (e.g., `20–30` clips of `idle_random` if you record `15` clips per dance move).
- **Do NOT just stand frozen during `idle_random`!** Include:
  - Standing relaxed & shifting weight
  - Walking in/out of frame
  - Casual arm movements, adjusting glasses/hair, clapping, swaying
  - Transitions *between* moves (raising/lowering hands casually)

### B. Recording Positive Move Clips (`--record-move <name>`)
```bash
.venv-mac/bin/python -m composer.moves.lab \
  --record-move body_roll \
  --clips 15 \
  --clip-seconds 1.2 \
  --countdown 2.0
```
- **Naming convention for directional moves**:
  - End left/right symmetric pairs with `_lr` / `_rl` (e.g., `arm_wave_lr`, `arm_wave_rl`) or `_left` / `_right` (e.g., `spin_left`, `spin_right`).
  - `train_move_net.py` automatically mirrors clips across paired directional labels inside the training split — giving you **2× the training data for free**!
  - Non-directional moves (e.g., `body_roll`, `salsa_step`, `shoulder_shimmy`) are mirrored into the same class automatically.
- **Vary your recording conditions across the 15–25 clips** (huge generalization boost!):
  - **Reps 1–5**: Center of frame, normal speed, facing straight ahead.
  - **Reps 6–10**: Step **closer** (waist-up / knees cropped) and **farther** (full body visible).
  - **Reps 11–15**: Turn **15–30° left and right** (yaw angle), plus a few slightly **faster** and **slower** reps.
  - **Bonus (best generalization)**: Have a **second person of different height/build** record 5–10 clips into the exact same `datasets/moves/<move_name>/` folder (clips auto-increment filenames and never overwrite!).
- **Timing tip**: Wait for the red `REC` bar, perform **one crisp repetition** centered inside the `1.2 s` capture window. During training, the pipeline automatically extracts the true-speed `0.80 s` peak-motion slice so it matches the live `0.80 s` streaming window 1:1.

---

## 4. Choosing Between Few-Shot DTW vs. Keras 3 1D-TCN

| Mode | Clips Needed | Training Step | Latency (Mac / STM32 A35) | Best Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 1 Kinematics** *(built-in)* | `0` | None | `< 0.1 ms` | Instant ballistic hits (`HARD_PUNCH`, `KNEE_STOMP`, `OVERHEAD_DROP`, `JUMP_HIT`) & holds (`ARMS_CROSS_CUT`, `SQUAT_DIVE`, `DOUBLE_ARM_RISE`). |
| **Tier 2A: Streaming DTW** (`--dtw-dir`) | **3–5 / move** | **Zero** | `~0.3 ms / ~1.2 ms` | Rapid prototyping — test a brand-new dance move 10 seconds after recording it! |
| **Tier 2B: Keras 3 1D-TCN** (`--tcn-model`) | **15–25 / move** + `idle_random` | ~10s on Mac (`train_move_net.py`) | **`~0.15 ms / ~1.0 ms`** | Production multi-person dance recognition (`~62 KB` pure-NumPy `.npz`, zero TensorFlow at runtime, strong noise & null rejection). |

---

## 5. Training & Auditing Your Neural Network

```bash
.venv-mac/bin/python ml/train_move_net.py \
  --data datasets/moves \
  --out models/dance_tcn.npz \
  --epochs 40
```

### How to Read the Training Output
`train_move_net.py` performs a **strict clip-level 80/20 split BEFORE any augmentation or mirroring** — so validation metrics reflect **100% unseen real clips**:
```text
Exported pure-NumPy TCN -> models/dance_tcn.npz (62.1 KB)
  Unseen clip val accuracy: 95.0%
    - arm_wave_lr       : 100.0% (4 unseen val clips)
    - body_roll         :  90.0% (4 unseen val clips)
    - idle_random       :  95.0% (6 unseen val clips)
  Keras vs NumPy max diff:  1.8e-07
  Pure-NumPy latency:       0.145 ms/frame
```
- **Target unseen validation accuracy**: **`≥ 88–95%`** per class.
- If one specific move scores `< 80%`:
  1. Check `datasets/moves/<move>/` and delete any botched clips where you missed the `REC` window.
  2. Record 8–10 cleaner clips of that move (or make it visually more distinct from colliding moves).
  3. Re-run `train_move_net.py`.
- Optional exports: add `--export-tflite models/dance_tcn.tflite` or `--export-keras models/dance_tcn.keras` if you want standard ML artifacts alongside the zero-dependency `.npz`.

---

## 6. Offline Replay Benchmarking (Zero Camera Needed)

After any diagnostic run (`composer.app ... --diagnose 30`), you can replay the saved `.jsonl` telemetry offline in `< 0.5 s` to test recognizer or TCN changes deterministically:

```bash
# Test Tier 1 only:
.venv-mac/bin/python -m composer.moves.lab --replay artifacts/diag_house_20261123_182101.jsonl

# Test Tier 1 + your trained Tier 2 Keras TCN together:
.venv-mac/bin/python -m composer.moves.lab \
  --replay artifacts/diag_house_20261123_182101.jsonl \
  --tcn-model models/dance_tcn.npz
```

---

## 7. Connecting Trained Models to Live Music & STM32MP257F-DK

1. **On macOS (Live Performance)**:
   Once `models/dance_tcn.npz` exists, `composer.app` **auto-loads it automatically** (or pass `--tcn-model path/to/model.npz` / `--dtw-dir datasets/moves`):
   ```bash
   .venv-mac/bin/python -m composer.app --mode camera --style house --launch-engine
   ```
   Any custom move label recognized by Tier 2 triggers a musical 4-note genre-adapted harmonic bloom immediately (and you can map custom move IDs to dedicated instruments in [composer/brain.py](file:///Users/yucateco/workspace/copiloba-play-despacito/composer/brain.py)).

2. **On STM32MP257F-DK**:
   Copy ONLY the tiny **`models/dance_tcn.npz`** (`~62 KB`) to the board — **zero TensorFlow/Keras installation is required on OpenSTLinux!** The pure-NumPy TCN executes in `~1 ms` on Cortex-A35 Core 0 (~`1.5%` CPU at 15 Hz stride), leaving the 1.35 TOPS VeriSilicon NPU dedicated to 30+ FPS pose estimation and Cortex-A35 Core 1 dedicated to real-time C++20 ALSA audio. See [docs/STM32MP257F_DK.md](file:///Users/yucateco/workspace/copiloba-play-despacito/docs/STM32MP257F_DK.md) for deployment details.

---

## 8. Troubleshooting Checklist

| Symptom in Moves Lab | Likely Cause | Fix |
| :--- | :--- | :--- |
| FPS stuck below `20 FPS` | Running `--pose-model full` or high capture resolution | Keep default `--pose-model lite --inference-width 320`. |
| Move fires randomly while standing still | Missing or too-static `idle_random` clips | Record 20+ active `idle_random` clips (casual sway/fidgeting) and retrain. |
| Model works for you but misses a taller/shorter friend | All training clips recorded at one exact distance/framing | Add 5–10 clips at closer (waist-up) & farther framing, or from a second person. |
| Punch triggers during slow dancing | Low lighting causing MediaPipe wrist jitter | Improve frontal lighting; check `GLITCHES BLOCKED` & `L_Thrust/R_Thrust` gauges. |
| Custom TCN move triggers twice on one repetition | Move lasts longer than `0.85 s` cooldown | Increase `cooldown_s` in `TemporalMoveClassifier` or make the rep crisper (~`0.8 s`). |
