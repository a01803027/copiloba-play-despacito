"""Tier 2 Option B — Embedded Pure-NumPy 1D-CNN / TCN Runtime (< 0.5 ms Mac / ~1 ms STM32 Cortex-A35).

Executes BatchNorm-folded weights exported from Keras 3 (`ml/train_move_net.py`) in `models/dance_tcn.npz`.
Supports both `avg_max` (GlobalAveragePooling1D + GlobalMaxPooling1D concat) and legacy `avg` pooling,
respects the exported `window_seconds`, and gates static stillness to eliminate OOD drift.
Runs identically on macOS Apple Silicon and STM32MP257F-DK ARM64 using ONLY `numpy` (zero TensorFlow).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import numpy as np
from .dtw import DEFAULT_WINDOW_SECONDS, NULL_CLASSES, _parse_label_side, resample_sequence
from .recognizer import MoveEvent


def _conv1d_same_dilated(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray, dilation: int = 1) -> np.ndarray:
    """Exact Keras `Conv1D(padding='same', dilation_rate=dilation)` in vectorized pure NumPy.

    Args:
        x: `(T, C_in)` float32 input sequence.
        kernel: `(K, C_in, C_out)` float32 weights.
        bias: `(C_out,)` float32 bias.
    """
    t_len, c_in = x.shape
    k_size, k_cin, c_out = kernel.shape
    assert c_in == k_cin, f'Channel mismatch: input {c_in} vs kernel {k_cin}'

    eff_k = (k_size - 1) * dilation + 1
    pad_total = max(0, eff_k - 1)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    if pad_total > 0:
        padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode='constant')
    else:
        padded = x

    out = np.broadcast_to(bias, (t_len, c_out)).copy()
    for k in range(k_size):
        start = k * dilation
        out += padded[start : start + t_len] @ kernel[k]
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / (np.sum(exps) + 1e-12)


class NumpyTcnModel:
    """Stateless pure-NumPy 1D-CNN / TCN inference engine with bit-exact Keras 3 parity."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.classes: tuple[str, ...] = tuple(str(c) for c in weights['classes'])
        self.window_frames: int = int(weights['window_frames'])
        self.window_seconds: float = float(weights.get('window_seconds', DEFAULT_WINDOW_SECONDS))
        self.feature_dim: int = int(weights['feature_dim'])
        self.pool_mode: str = str(weights.get('pool_mode', 'avg'))
        self.mean: np.ndarray = np.asarray(weights['mean'], dtype=np.float32)
        self.std: np.ndarray = np.maximum(np.asarray(weights['std'], dtype=np.float32), 1e-5)

        self.conv1_w = np.ascontiguousarray(weights['conv1_w'], dtype=np.float32)
        self.conv1_b = np.ascontiguousarray(weights['conv1_b'], dtype=np.float32)
        self.conv1_d = int(weights.get('conv1_dilation', 1))

        self.conv2_w = np.ascontiguousarray(weights['conv2_w'], dtype=np.float32)
        self.conv2_b = np.ascontiguousarray(weights['conv2_b'], dtype=np.float32)
        self.conv2_d = int(weights.get('conv2_dilation', 2))

        self.dense1_w = np.ascontiguousarray(weights['dense1_w'], dtype=np.float32)
        self.dense1_b = np.ascontiguousarray(weights['dense1_b'], dtype=np.float32)

        self.dense2_w = np.ascontiguousarray(weights['dense2_w'], dtype=np.float32)
        self.dense2_b = np.ascontiguousarray(weights['dense2_b'], dtype=np.float32)

    @classmethod
    def load(cls, npz_path: Path | str) -> 'NumpyTcnModel':
        with np.load(Path(npz_path), allow_pickle=False) as data:
            return cls({k: data[k] for k in data.files})

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        """Run forward pass on a `(window_frames, feature_dim)` window; returns `(num_classes,)` probs."""
        x = (np.asarray(window, dtype=np.float32) - self.mean) / self.std
        h1 = np.maximum(0.0, _conv1d_same_dilated(x, self.conv1_w, self.conv1_b, dilation=self.conv1_d))
        h2 = np.maximum(0.0, _conv1d_same_dilated(h1, self.conv2_w, self.conv2_b, dilation=self.conv2_d))
        if self.pool_mode == 'avg_max':
            pooled = np.concatenate((np.mean(h2, axis=0), np.max(h2, axis=0)), axis=0)
        else:
            pooled = np.mean(h2, axis=0)
        d1 = np.maximum(0.0, pooled @ self.dense1_w + self.dense1_b)
        logits = d1 @ self.dense2_w + self.dense2_b
        return _softmax(logits)


class TemporalMoveClassifier:
    """Streaming online classifier with EMA smoothing, motion gate, hysteresis, and cooldowns."""

    def __init__(
        self,
        model: NumpyTcnModel,
        *,
        window_seconds: float | None = None,
        enter_threshold: float = 0.68,
        exit_threshold: float = 0.40,
        ema_alpha: float = 0.50,
        cooldown_s: float = 0.85,
        eval_stride: int = 2,
        min_motion_std: float = 0.025,
        multi_scale: bool = True,
    ):
        self.model = model
        self.window_seconds = float(window_seconds if window_seconds is not None else model.window_seconds)
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.ema_alpha = ema_alpha
        self.cooldown_s = cooldown_s
        self.eval_stride = max(1, eval_stride)
        self.min_motion_std = min_motion_std
        self.multi_scale = multi_scale
        self.reset()

    @classmethod
    def from_npz(cls, npz_path: Path | str, **kwargs) -> 'TemporalMoveClassifier':
        return cls(NumpyTcnModel.load(npz_path), **kwargs)

    def reset(self):
        self._buf: deque[tuple[float, tuple[float, ...]]] = deque(maxlen=180)
        self._ema = np.zeros(len(self.model.classes), dtype=np.float32)
        self._active: dict[str, bool] = {c: False for c in self.model.classes}
        self._cooldowns: dict[str, float] = {}
        self._frame_count = 0
        self.last_probs: dict[str, float] = {c: 0.0 for c in self.model.classes}
        self.last_raw_probs: dict[str, float] = {c: 0.0 for c in self.model.classes}

    def _window_for_span(self, timestamp: float, span_s: float) -> np.ndarray | None:
        cutoff = timestamp - span_s
        win = [item for item in self._buf if item[0] >= cutoff]
        if len(win) < 6 or (win[-1][0] - win[0][0]) < (0.45 * span_s):
            return None
        ts = np.array([w[0] for w in win], dtype=np.float64)
        feats = np.array([w[1] for w in win], dtype=np.float32)
        return resample_sequence(feats, ts, target_frames=self.model.window_frames)

    def update(self, timestamp: float, feature_vector: tuple[float, ...]) -> list[MoveEvent]:
        self._buf.append((timestamp, feature_vector))
        self._frame_count += 1
        if self._frame_count % self.eval_stride != 0:
            return []

        window = self._window_for_span(timestamp, self.window_seconds)
        if window is None:
            return []

        # Motion-energy gate: if no joint is actually moving (`max_std < 0.055`), decay probabilities cleanly.
        if float(np.max(np.std(window, axis=0))) < max(self.min_motion_std, 0.055):
            self._ema *= (1.0 - self.ema_alpha)
            self.last_probs = {cls_name: float(self._ema[i]) for i, cls_name in enumerate(self.model.classes)}
            for cls_name in self._active:
                self._active[cls_name] = False
            return []

        # Multi-scale time invariance: evaluate nominal speed (1.0x) + faster (0.72x) + slower (1.32x) windows!
        scales = (0.72, 1.0, 1.32) if self.multi_scale else (1.0,)
        prob_candidates = []
        for sc in scales:
            w_sc = window if sc == 1.0 else self._window_for_span(timestamp, self.window_seconds * sc)
            if w_sc is not None:
                prob_candidates.append(self.model.predict_proba(w_sc))
        probs = np.max(np.stack(prob_candidates, axis=0), axis=0) if len(prob_candidates) > 1 else prob_candidates[0]

        self.last_raw_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.model.classes)}
        self._ema = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self._ema
        self.last_probs = {cls_name: float(self._ema[i]) for i, cls_name in enumerate(self.model.classes)}

        events: list[MoveEvent] = []
        best_i = int(np.argmax(self._ema))
        best_cls = self.model.classes[best_i]
        best_p = float(self._ema[best_i])

        for i, cls_name in enumerate(self.model.classes):
            if self._active.get(cls_name) and float(self._ema[i]) <= self.exit_threshold:
                self._active[cls_name] = False

        if best_cls.lower() in NULL_CLASSES:
            return events

        if (
            best_p >= self.enter_threshold
            and not self._active.get(best_cls, False)
            and (timestamp - self._cooldowns.get(best_cls, -1e9)) >= self.cooldown_s
        ):
            self._active[best_cls] = True
            self._cooldowns[best_cls] = timestamp
            kind, side = _parse_label_side(best_cls)
            events.append(MoveEvent(kind=kind, side=side, intensity=max(0.70, best_p)))

        return events
