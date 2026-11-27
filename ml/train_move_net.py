"""Keras 3 1D-CNN / TCN Dance Move Training & Embedded Pure-NumPy / TFLite Export Pipeline.

Audit-hardened training strategy for multi-height, multi-position, multi-body-type generalization
and sub-millisecond embedded execution on macOS & STM32MP257F-DK (Cortex-A35):
  - Zero train/val clip leakage: stratified clip-level split BEFORE augmentation or mirroring.
  - Exact train/inference time alignment: extracts true-speed `0.80 s` windows from longer clips.
  - Biomechanical domain augmentation: yaw foreshortening, height/arm-length proportion scaling,
    2D camera roll (±10°), bidirectional speed warp (0.65x..1.35x with velocity rescaling),
    random knee occlusion dropout (25% -> tall/close users without visible legs!), & landmark jitter.
  - GAP + GMP dual pooling (`~15k` params, `~62 KB`, `~0.57 MFLOPs`), label smoothing (`0.05`),
    L2 regularization, balanced class weights, EarlyStopping, and verified Keras-vs-NumPy parity.

Usage:
  Train on your recorded Moves Lab clips:
      .venv-mac/bin/python ml/train_move_net.py --data datasets/moves --out models/dance_tcn.npz

  Self-test end-to-end with synthetic dance clips:
      .venv-mac/bin/python ml/train_move_net.py --demo-synthetic --epochs 14 --out /tmp/demo_tcn.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np

# Ensure Keras 3 always uses TensorFlow inside this venv even if ~/.keras/keras.json sets torch/jax.
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from composer.moves.dtw import (
    DEFAULT_WINDOW_FRAMES,
    DEFAULT_WINDOW_SECONDS,
    NULL_CLASSES,
    estimate_active_duration,
    extract_time_window,
    resample_sequence,
)
from composer.moves.pose_frame import FEATURE_DIM
from composer.moves.tcn_runtime import NumpyTcnModel

# Feature layout inside the 24-D canonical pose vector (composer/moves/sanitizer.py):
#  0..1: l_sh(x,y)  2..3: r_sh(x,y)   4..5: l_el(x,y)   6..7: r_el(x,y)
#  8..9: l_wr(x,y) 10..11: r_wr(x,y) 12..13: l_wr_v    14..15: r_wr_v
# 16: l_bend       17: r_bend        18..19: l_knee    20..21: r_knee
# 22: hip_rel_elev 23: torso_lean
_MIRROR_X_COLS = (0, 2, 4, 6, 8, 10, 12, 14, 18, 20, 23)
_MIRROR_SWAPS = ((0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15), (16, 17), (18, 20), (19, 21))
_XY_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (18, 19), (20, 21))


def mirror_feature_sequence(seq: np.ndarray) -> np.ndarray:
    """Anatomically mirror a `(T, 24)` pose sequence left <-> right."""
    out = np.array(seq, dtype=np.float32, copy=True)
    for a, b in _MIRROR_SWAPS:
        tmp = out[:, a].copy()
        out[:, a] = out[:, b]
        out[:, b] = tmp
    for col in _MIRROR_X_COLS:
        out[:, col] = -out[:, col]
    return out


def mirror_label_name(label: str) -> str:
    """Flip directional suffixes when mirroring left <-> right."""
    low = label.lower()
    if low.endswith('_lr'):
        return label[:-3] + '_rl'
    if low.endswith('_rl'):
        return label[:-3] + '_lr'
    if low.endswith('_left'):
        return label[:-5] + '_right'
    if low.endswith('_right'):
        return label[:-6] + '_left'
    return label


def _extract_random_window(
    features: np.ndarray,
    timestamps: np.ndarray | None,
    rng: np.random.Generator,
    *,
    window_seconds: float,
    target_frames: int,
) -> tuple[np.ndarray, float]:
    """Sample a wide speed-warped (`0.55x..1.55x`) sub-window from a raw clip and return `(window, speed_factor)`."""
    arr = np.asarray(features, dtype=np.float32)
    speed_factor = float(rng.uniform(0.55, 1.55))

    if timestamps is not None and len(timestamps) == arr.shape[0]:
        t = np.asarray(timestamps, dtype=np.float64)
        total_span = float(t[-1] - t[0])
        desired_span = window_seconds * speed_factor
        if total_span > desired_span + 1e-4:
            lo = float(rng.uniform(float(t[0]), float(t[-1]) - desired_span))
            dst_t = np.linspace(lo, lo + desired_span, target_frames, dtype=np.float64)
            out = np.empty((target_frames, arr.shape[1]), dtype=np.float32)
            for col in range(arr.shape[1]):
                out[:, col] = np.interp(dst_t, t, arr[:, col])
            return out, speed_factor
        # Clip shorter than desired_span -> resample full clip.
        return resample_sequence(arr, t, target_frames=target_frames), 1.0

    # Fallback without timestamps: sample sub-window of length 72%..100%.
    n = arr.shape[0]
    frac = float(rng.uniform(0.72, 1.0))
    sub_len = max(8, int(round(n * frac)))
    start = int(rng.integers(0, max(1, n - sub_len + 1)))
    return resample_sequence(arr[start : start + sub_len], None, target_frames=target_frames), frac


def augment_clip(
    seq: np.ndarray,
    rng: np.random.Generator,
    *,
    timestamps: np.ndarray | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    target_frames: int = DEFAULT_WINDOW_FRAMES,
) -> np.ndarray:
    """Biomechanical augmentation for multi-height, multi-proportion, yaw, tilt, and leg-occlusion invariance."""
    out, speed_factor = _extract_random_window(
        seq, timestamps, rng, window_seconds=window_seconds, target_frames=target_frames
    )

    # 1) Motion-amplitude scaling around mean clip posture (smaller casual rep vs big energetic rep!)
    #    + static posture shift jitter so different resting stances never lock out another dancer.
    mean_pose = np.mean(out, axis=0, keepdims=True)
    amp_scale = float(rng.uniform(0.65, 1.40))
    posture_shift = rng.normal(0.0, 0.045, size=(1, out.shape[1])).astype(np.float32)
    posture_shift[:, 12:18] = 0.0
    out = mean_pose + posture_shift + (out - mean_pose) * amp_scale

    # 2) Yaw foreshortening (standing slightly turned) & body height/width proportion variance.
    scale_x = float(rng.uniform(0.74, 1.20))
    scale_y = float(rng.uniform(0.84, 1.16))
    for x_col, y_col in _XY_PAIRS:
        out[:, x_col] *= scale_x
        out[:, y_col] *= scale_y

    # 3) Independent arm-length proportion scaling around shoulders (short-armed vs long-armed performers).
    arm_scale = float(rng.uniform(0.82, 1.20))
    for sh_x, sh_y, el_x, el_y, wr_x, wr_y in ((0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11)):
        out[:, el_x] = out[:, sh_x] + (out[:, el_x] - out[:, sh_x]) * arm_scale
        out[:, el_y] = out[:, sh_y] + (out[:, el_y] - out[:, sh_y]) * arm_scale
        out[:, wr_x] = out[:, sh_x] + (out[:, wr_x] - out[:, sh_x]) * arm_scale
        out[:, wr_y] = out[:, sh_y] + (out[:, wr_y] - out[:, sh_y]) * arm_scale

    # 4) Consistent wrist velocity rescaling after amplitude + spatial + temporal speed warp.
    vel_gain = (amp_scale * arm_scale) / max(speed_factor, 0.45)
    out[:, 12:16] = np.clip(out[:, 12:16] * vel_gain, -2.0, 2.0)

    # 4) 2D Camera roll / tilt (±10 deg) rotated consistently across all joints & velocities.
    angle = float(rng.uniform(-0.175, 0.175))
    c_ang, s_ang = float(np.cos(angle)), float(np.sin(angle))
    for x_col, y_col in _XY_PAIRS:
        vx = out[:, x_col].copy()
        vy = out[:, y_col].copy()
        out[:, x_col] = c_ang * vx - s_ang * vy
        out[:, y_col] = s_ang * vx + c_ang * vy
    out[:, 23] = np.clip(out[:, 23] + angle / 0.45, -1.0, 1.0)

    # 5) Lower-body (knee) off-screen occlusion dropout (25% chance) -> tall/close performers!
    if float(rng.random()) < 0.25:
        out[:, 18] = -0.22
        out[:, 19] = -0.75
        out[:, 20] = 0.22
        out[:, 21] = -0.75

    # 6) Synthetic MediaPipe / NPU landmark noise.
    out += rng.normal(0.0, 0.018, size=out.shape).astype(np.float32)
    return out


def load_dataset_split(
    data_dir: Path,
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    window_seconds: float | None = None,
    aug_per_clip: int = 24,
    val_ratio: float = 0.20,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Load `datasets/moves/<label>/*.npz` with STRICT clip-level train/val split BEFORE augmentation!

    Auto-adapts `window_seconds` from the recorded clips' active duration when `window_seconds is None`,
    and warns if any move class has very low 2D visibility (`mean_std < 0.065`, e.g. frontal body roll).
    Returns `(x_train, y_train, x_val, y_val, classes, effective_window_seconds)`.
    """
    rng = np.random.default_rng(seed)
    raw_by_class: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {}
    active_durs: list[float] = []
    for move_dir in sorted(Path(data_dir).iterdir()):
        if not move_dir.is_dir():
            continue
        label = move_dir.name
        for npz_path in sorted(move_dir.glob('*.npz')):
            with np.load(npz_path, allow_pickle=False) as data:
                if 'features' not in data:
                    continue
                feats = np.asarray(data['features'], dtype=np.float32)
                ts = np.asarray(data['timestamps'], dtype=np.float64) if 'timestamps' in data else None
                if feats.ndim == 2 and feats.shape[0] >= 4 and feats.shape[1] == FEATURE_DIM:
                    raw_by_class.setdefault(label, []).append((feats, ts))
                    if label.lower() not in NULL_CLASSES:
                        active_durs.append(estimate_active_duration(feats, ts))

    if len(raw_by_class) < 2:
        raise ValueError(f'Need at least 2 move folders with .npz clips inside {data_dir}')

    eff_win_s = float(window_seconds) if window_seconds is not None else (
        float(np.median(active_durs)) if active_durs else DEFAULT_WINDOW_SECONDS
    )

    classes = sorted(raw_by_class.keys())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    if not any(c.lower() in NULL_CLASSES for c in classes):
        print('WARNING: No `idle_random` negative class found! Record `idle_random` clips to avoid false triggers.', flush=True)

    for cls_name in classes:
        if cls_name.lower() in NULL_CLASSES:
            continue
        mean_motion = float(np.mean([np.mean(np.std(f, axis=0)) for f, _ in raw_by_class[cls_name]]))
        if mean_motion < 0.065:
            print(
                f'WARNING: Move "{cls_name}" has very small 2D keypoint motion (mean_std={mean_motion:.3f}). '
                f'Moves along camera depth (like a dead-on frontal body roll) look nearly identical to idle '
                f'because 2D pose has no spine/chest-depth points — try performing it at a 35-45 deg side angle '
                f'or involving arms/elbows!',
                flush=True,
            )

    train_raw: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {c: [] for c in classes}
    val_raw: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {c: [] for c in classes}

    # Stratified clip-level split BEFORE any augmentation or mirroring (ZERO leakage!).
    for cls_name in classes:
        clips = list(raw_by_class[cls_name])
        order = rng.permutation(len(clips))
        n_val = max(1, int(round(len(clips) * val_ratio))) if len(clips) >= 2 else 0
        for rank, idx in enumerate(order):
            if rank < n_val:
                val_raw[cls_name].append(clips[int(idx)])
            else:
                train_raw[cls_name].append(clips[int(idx)])

    # Build clean unseen validation set (exact true-speed `eff_win_s` peak-motion slice).
    x_val_list: list[np.ndarray] = []
    y_val_list: list[int] = []
    for cls_name in classes:
        for feats, ts in val_raw[cls_name]:
            is_null = cls_name.lower() in NULL_CLASSES
            win = extract_time_window(
                feats, ts, window_seconds=eff_win_s, target_frames=window_frames, center_on_peak=not is_null
            )
            x_val_list.append(win)
            y_val_list.append(cls_to_idx[cls_name])

    # Build augmented training set (including anatomical mirroring across both symmetric & paired L/R classes!).
    x_tr_list: list[np.ndarray] = []
    y_tr_list: list[int] = []
    for cls_name in classes:
        c_idx = cls_to_idx[cls_name]
        m_lbl = mirror_label_name(cls_name)
        m_idx = cls_to_idx.get(m_lbl)
        is_null = cls_name.lower() in NULL_CLASSES

        for feats, ts in train_raw[cls_name]:
            base_win = extract_time_window(
                feats, ts, window_seconds=eff_win_s, target_frames=window_frames, center_on_peak=not is_null
            )
            x_tr_list.append(base_win)
            y_tr_list.append(c_idx)
            for _ in range(aug_per_clip):
                x_tr_list.append(
                    augment_clip(feats, rng, timestamps=ts, window_seconds=eff_win_s, target_frames=window_frames)
                )
                y_tr_list.append(c_idx)

            # Mirror strictly within training set (same class OR paired directional class if present!).
            if m_idx is not None:
                m_feats = mirror_feature_sequence(feats)
                x_tr_list.append(mirror_feature_sequence(base_win))
                y_tr_list.append(m_idx)
                for _ in range(max(4, aug_per_clip // 2)):
                    x_tr_list.append(
                        augment_clip(m_feats, rng, timestamps=ts, window_seconds=eff_win_s, target_frames=window_frames)
                    )
                    y_tr_list.append(m_idx)

    x_train = np.stack(x_tr_list, axis=0).astype(np.float32)
    y_train = np.array(y_tr_list, dtype=np.int32)
    perm = rng.permutation(len(y_train))
    x_train, y_train = x_train[perm], y_train[perm]

    x_val = np.stack(x_val_list, axis=0).astype(np.float32) if x_val_list else x_train[:1]
    y_val = np.array(y_val_list, dtype=np.int32) if y_val_list else y_train[:1]
    return x_train, y_train, x_val, y_val, classes, eff_win_s


def generate_synthetic_dataset(
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    clips_per_class: int = 45,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a realistic multi-person synthetic 4-class dataset (`arm_wave_lr`, `arm_wave_rl`, `body_roll`, `idle_random`)."""
    rng = np.random.default_rng(seed)
    classes = ['arm_wave_lr', 'arm_wave_rl', 'body_roll', 'idle_random']
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    t = np.linspace(0.0, DEFAULT_WINDOW_SECONDS, window_frames, dtype=np.float32)

    for cls_idx, cls_name in enumerate(classes):
        for _ in range(clips_per_class):
            seq = np.zeros((window_frames, FEATURE_DIM), dtype=np.float32)
            # Simulate diverse body proportions (wide/narrow shoulders, tall/short limbs, yaw foreshortening).
            sh_w = float(rng.uniform(0.28, 0.44))
            arm_len = float(rng.uniform(0.85, 1.15))
            seq[:, 0], seq[:, 1] = -sh_w, 1.0
            seq[:, 2], seq[:, 3] = sh_w, 1.0
            seq[:, 18], seq[:, 19] = -0.22, -0.75
            seq[:, 20], seq[:, 21] = 0.22, -0.75
            freq = float(rng.uniform(1.00, 1.55))
            phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
            amp = float(rng.uniform(0.20, 0.34)) * arm_len

            if cls_name in ('arm_wave_lr', 'arm_wave_rl'):
                lags = (0.0, 0.10, 0.22, 0.32) if cls_name == 'arm_wave_lr' else (0.32, 0.22, 0.10, 0.0)
                seq[:, 8], seq[:, 9] = -0.82 * arm_len, 0.75 + amp * np.sin(2 * np.pi * freq * (t - lags[0]) + phase0)
                seq[:, 4], seq[:, 5] = -0.56 * arm_len, 0.85 + 0.8 * amp * np.sin(2 * np.pi * freq * (t - lags[1]) + phase0)
                seq[:, 6], seq[:, 7] = 0.56 * arm_len, 0.85 + 0.8 * amp * np.sin(2 * np.pi * freq * (t - lags[2]) + phase0)
                seq[:, 10], seq[:, 11] = 0.82 * arm_len, 0.75 + amp * np.sin(2 * np.pi * freq * (t - lags[3]) + phase0)
            elif cls_name == 'body_roll':
                seq[:, 22] = 0.35 * np.sin(2 * np.pi * freq * t + phase0)
                seq[:, 23] = 0.45 * np.cos(2 * np.pi * freq * t + phase0)
                seq[:, 4], seq[:, 5] = -0.45 * arm_len, 0.55 + 0.15 * np.sin(2 * np.pi * freq * t + phase0)
                seq[:, 6], seq[:, 7] = 0.45 * arm_len, 0.55 + 0.15 * np.sin(2 * np.pi * freq * t + phase0)
                seq[:, 8], seq[:, 9] = -0.50 * arm_len, 0.35
                seq[:, 10], seq[:, 11] = 0.50 * arm_len, 0.35
            else:
                drift = float(rng.uniform(-0.2, 0.2))
                seq[:, 4], seq[:, 5] = -0.42 * arm_len, 0.45 + drift
                seq[:, 6], seq[:, 7] = 0.42 * arm_len, 0.45 - drift
                seq[:, 8], seq[:, 9] = -0.50 * arm_len, 0.20 + 0.08 * np.sin(2 * np.pi * 0.6 * t + phase0)
                seq[:, 10], seq[:, 11] = 0.50 * arm_len, 0.20 + 0.08 * np.sin(2 * np.pi * 0.6 * t + phase0)

            dt = DEFAULT_WINDOW_SECONDS / max(1, window_frames - 1)
            seq[1:, 12:14] = np.clip(np.diff(seq[:, 8:10], axis=0) / (dt * 8.0), -2.0, 2.0)
            seq[1:, 14:16] = np.clip(np.diff(seq[:, 10:12], axis=0) / (dt * 8.0), -2.0, 2.0)
            seq[:, 16:18] = float(rng.uniform(0.25, 0.65))
            seq += rng.normal(0.0, 0.02, size=seq.shape).astype(np.float32)

            x_list.append(seq)
            y_list.append(cls_idx)

    x_arr = np.stack(x_list, axis=0).astype(np.float32)
    y_arr = np.array(y_list, dtype=np.int32)
    perm = rng.permutation(len(y_arr))
    return x_arr[perm], y_arr[perm], classes


def build_keras_tcn(window_frames: int, feature_dim: int, num_classes: int):
    """Build a compact (~15k param, ~62 KB, ~0.57 MFLOPs) Keras 3 1D-TCN with GAP+GMP dual pooling."""
    import keras
    from keras import layers, regularizers

    l2 = regularizers.L2(1e-4)
    inp = keras.Input(shape=(window_frames, feature_dim), name='pose_window')
    x = layers.Conv1D(32, 5, padding='same', dilation_rate=1, use_bias=False, kernel_regularizer=l2, name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.SpatialDropout1D(0.15, name='drop1')(x)

    x = layers.Conv1D(48, 5, padding='same', dilation_rate=2, use_bias=False, kernel_regularizer=l2, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.SpatialDropout1D(0.15, name='drop2')(x)

    # Dual pooling captures BOTH continuous groove patterns (GAP) and brief dance accents (GMP).
    gap = layers.GlobalAveragePooling1D(name='gap')(x)
    gmp = layers.GlobalMaxPooling1D(name='gmp')(x)
    x = layers.Concatenate(name='pool_concat')([gap, gmp])

    x = layers.Dense(32, activation='relu', kernel_regularizer=l2, name='dense1')(x)
    x = layers.Dropout(0.25, name='drop3')(x)
    out = layers.Dense(num_classes, activation='softmax', name='dense2')(x)
    return keras.Model(inputs=inp, outputs=out, name='cv_daw_dance_tcn')


def _fold_conv_bn(conv_layer, bn_layer) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNormalization parameters directly into Conv1D kernel & bias for zero-overhead inference."""
    kernel = conv_layer.get_weights()[0].astype(np.float32)  # (K, C_in, C_out)
    gamma, beta, moving_mean, moving_var = [w.astype(np.float32) for w in bn_layer.get_weights()]
    eps = float(bn_layer.epsilon)
    scale = gamma / np.sqrt(moving_var + eps)
    folded_w = kernel * scale.reshape((1, 1, -1))
    folded_b = beta - moving_mean * scale
    return folded_w.astype(np.float32), folded_b.astype(np.float32)


def export_to_numpy_npz(
    model,
    out_path: Path,
    *,
    classes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    window_frames: int,
    window_seconds: float,
    feature_dim: int,
) -> Path:
    """Export trained Keras 3 TCN into a compact, BatchNorm-folded pure-NumPy `.npz` file."""
    c1_w, c1_b = _fold_conv_bn(model.get_layer('conv1'), model.get_layer('bn1'))
    c2_w, c2_b = _fold_conv_bn(model.get_layer('conv2'), model.get_layer('bn2'))
    d1_w, d1_b = [w.astype(np.float32) for w in model.get_layer('dense1').get_weights()]
    d2_w, d2_b = [w.astype(np.float32) for w in model.get_layer('dense2').get_weights()]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        classes=np.array(classes),
        window_frames=np.int32(window_frames),
        window_seconds=np.float32(window_seconds),
        feature_dim=np.int32(feature_dim),
        pool_mode=np.array('avg_max'),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        conv1_w=c1_w,
        conv1_b=c1_b,
        conv1_dilation=np.int32(1),
        conv2_w=c2_w,
        conv2_b=c2_b,
        conv2_dilation=np.int32(2),
        dense1_w=d1_w,
        dense1_b=d1_b,
        dense2_w=d2_w,
        dense2_b=d2_b,
    )
    return out_path


def train_and_export(args) -> dict:
    import keras

    req_win_s = getattr(args, 'window_seconds', None)
    if args.demo_synthetic:
        window_seconds = float(req_win_s if req_win_s is not None else DEFAULT_WINDOW_SECONDS)
        x_all, y_all, classes = generate_synthetic_dataset(window_frames=args.window_frames, seed=args.seed)
        split = max(1, int(0.8 * len(y_all)))
        x_train, y_train = x_all[:split], y_all[:split]
        x_val, y_val = x_all[split:], y_all[split:]
    else:
        x_train, y_train, x_val, y_val, classes, window_seconds = load_dataset_split(
            args.data,
            window_frames=args.window_frames,
            window_seconds=req_win_s,
            aug_per_clip=args.aug_per_clip,
            seed=args.seed,
        )

    num_classes = len(classes)
    mean = np.mean(x_train, axis=(0, 1)).astype(np.float32)
    std = np.maximum(np.std(x_train, axis=(0, 1)), 1e-5).astype(np.float32)
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    # One-hot targets so we can use label_smoothing=0.05 (prevents OOD softmax overconfidence!).
    y_train_oh = np.eye(num_classes, dtype=np.float32)[y_train]
    y_val_oh = np.eye(num_classes, dtype=np.float32)[y_val]

    # Balanced class weights in case one move has fewer clips than `idle_random`.
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))
    class_weight = {i: float(total / (num_classes * max(1.0, counts[i]))) for i in range(num_classes)}

    model = build_keras_tcn(args.window_frames, FEATURE_DIM, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy'],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=max(6, args.epochs // 4),
            restore_best_weights=True,
        ),
    ]
    print(
        f'Training Keras 3 Dance TCN | classes={classes} | params={model.count_params():,} | '
        f'train={len(x_train)} unseen_val={len(x_val)}',
        flush=True,
    )

    model.fit(
        x_train_n,
        y_train_oh,
        validation_data=(x_val_n, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1 if not args.quiet else 0,
    )

    out_path = export_to_numpy_npz(
        model,
        args.out,
        classes=classes,
        mean=mean,
        std=std,
        window_frames=args.window_frames,
        window_seconds=window_seconds,
        feature_dim=FEATURE_DIM,
    )

    # Verify exact numerical parity + unseen validation per-class accuracy!
    np_model = NumpyTcnModel.load(out_path)
    keras_probs = model.predict(x_val_n, verbose=0)
    t0 = time.perf_counter()
    np_probs = np.stack([np_model.predict_proba(s) for s in x_val], axis=0)
    per_frame_ms = (time.perf_counter() - t0) / max(1, len(x_val)) * 1000.0
    max_diff = float(np.max(np.abs(keras_probs - np_probs)))

    val_preds = np.argmax(np_probs, axis=1)
    val_acc = float(np.mean(val_preds == y_val))
    size_kb = out_path.stat().st_size / 1024.0

    print(f'\nExported pure-NumPy TCN -> {out_path} ({size_kb:.1f} KB)')
    print(f'  Unseen clip val accuracy: {val_acc * 100:.1f}%')
    for i, cname in enumerate(classes):
        mask = (y_val == i)
        if np.any(mask):
            cls_acc = float(np.mean(val_preds[mask] == i))
            print(f'    - {cname:<18s}: {cls_acc * 100:5.1f}% ({int(np.sum(mask))} unseen val clips)')
    print(f'  Keras vs NumPy max diff:  {max_diff:.2e}')
    print(f'  Pure-NumPy latency:       {per_frame_ms:.3f} ms/frame')

    if args.export_keras:
        args.export_keras.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.export_keras)
        print(f'  Saved Keras checkpoint -> {args.export_keras}')

    if args.export_tflite:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_bytes = converter.convert()
        args.export_tflite.parent.mkdir(parents=True, exist_ok=True)
        args.export_tflite.write_bytes(tflite_bytes)
        print(f'  Saved FP32 TFLite       -> {args.export_tflite} ({len(tflite_bytes) / 1024.0:.1f} KB)')

    int8_acc = None
    if getattr(args, 'export_tflite_int8', None):
        import tensorflow as tf

        def rep_data():
            n_cal = min(200, len(x_train_n))
            for i in range(n_cal):
                yield [x_train_n[i : i + 1].astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        int8_bytes = converter.convert()
        out_int8 = Path(args.export_tflite_int8)
        out_int8.parent.mkdir(parents=True, exist_ok=True)
        out_int8.write_bytes(int8_bytes)

        # Verify quantized INT8 accuracy directly on the unseen validation set!
        interp = tf.lite.Interpreter(model_content=int8_bytes)
        interp.allocate_tensors()
        in_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]
        in_s, in_zp = in_d['quantization']
        out_s, out_zp = out_d['quantization']
        q_preds = []
        for sample in x_val_n:
            q_in = np.clip(np.round(sample[None, ...] / in_s + in_zp), -128, 127).astype(np.int8)
            interp.set_tensor(in_d['index'], q_in)
            interp.invoke()
            q_out = interp.get_tensor(out_d['index'])[0]
            f_out = (q_out.astype(np.float32) - out_zp) * out_s
            q_preds.append(int(np.argmax(f_out)))
        int8_acc = float(np.mean(np.array(q_preds) == y_val))
        print(
            f'  Saved INT8 TFLite (NPU) -> {out_int8} ({len(int8_bytes) / 1024.0:.1f} KB | '
            f'unseen val acc: {int8_acc * 100:.1f}%)'
        )

    return {
        'out_path': str(out_path),
        'classes': classes,
        'params': int(model.count_params()),
        'val_accuracy': val_acc,
        'int8_val_accuracy': int8_acc,
        'keras_numpy_max_diff': max_diff,
        'latency_ms': per_frame_ms,
        'size_kb': size_kb,
    }


def main():
    p = argparse.ArgumentParser(description='Train Keras 3 Dance TCN and export to pure-NumPy .npz (<0.5 ms embedded)')
    p.add_argument('--data', type=Path, default=ROOT / 'datasets/moves', help='Folder with <move_name>/*.npz clips')
    p.add_argument('--out', type=Path, default=ROOT / 'models/dance_tcn.npz', help='Output pure-NumPy .npz model')
    p.add_argument('--export-keras', type=Path, default=None, help='Optional .keras model save path')
    p.add_argument('--export-tflite', type=Path, default=None, help='Optional FP32 .tflite export path')
    p.add_argument('--export-tflite-int8', type=Path, default=None,
                   help='Optional full-integer INT8 .tflite export (calibrated for STM32MP257 NPU / stedgeai)')
    p.add_argument('--window-frames', type=int, default=DEFAULT_WINDOW_FRAMES)
    p.add_argument('--window-seconds', type=float, default=None,
                   help='Window seconds (default: auto-detected from recorded clips median active duration)')
    p.add_argument('--aug-per-clip', type=int, default=24)
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1.5e-3)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--demo-synthetic', action='store_true', help='Train & verify on built-in synthetic dance clips')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    train_and_export(args)


if __name__ == '__main__':
    main()
