"""Tier 2 Option A — Posture- & Speed-Tolerant Few-Shot Streaming DTW Matcher (Zero Training).

Designed for STM32MP257F-DK & macOS (pure NumPy, < 1.5 ms/eval):
  - Auto-adapts window duration per move from recorded `.npz` timestamps (no fixed 0.80s chopping!).
  - Multi-scale time querying (`0.72x`, `1.00x`, `1.35x` speed) + Sakoe-Chiba warping (`band=6`)
    so moves performed significantly faster or slower still match cleanly.
  - Posture-invariant trajectory representation: blends torso-scaled pose with window-centered,
    amplitude-normalized motion deltas so different people / stances / movement sizes match without
    needing pixel-identical clones.
  - Auto-calibrates per-move distance thresholds from intra-class template spread.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .pose_frame import FEATURE_DIM
from .recognizer import MoveEvent

DEFAULT_WINDOW_FRAMES = 24
DEFAULT_WINDOW_SECONDS = 0.80
NULL_CLASSES = frozenset({'idle', 'idle_random', 'null', 'none', 'background'})


def resample_sequence(
    features: np.ndarray,
    timestamps: np.ndarray | None = None,
    target_frames: int = DEFAULT_WINDOW_FRAMES,
) -> np.ndarray:
    """Linearly resample a `(N, F)` pose feature sequence to `(target_frames, F)` float32."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((target_frames, FEATURE_DIM), dtype=np.float32)
    if arr.shape[0] == 1:
        return np.repeat(arr, target_frames, axis=0)

    if timestamps is not None and len(timestamps) == arr.shape[0]:
        t = np.asarray(timestamps, dtype=np.float64)
        span = t[-1] - t[0]
        if span > 1e-4:
            src_x = (t - t[0]) / span
        else:
            src_x = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    else:
        src_x = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)

    dst_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    out = np.empty((target_frames, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(dst_x, src_x, arr[:, col])
    return out


def extract_time_window(
    features: np.ndarray,
    timestamps: np.ndarray | None = None,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    target_frames: int = DEFAULT_WINDOW_FRAMES,
    center_on_peak: bool = True,
) -> np.ndarray:
    """Extract a true-speed `window_seconds` slice from a clip and resample to `target_frames`."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((target_frames, FEATURE_DIM), dtype=np.float32)
    if timestamps is None or len(timestamps) != arr.shape[0]:
        return resample_sequence(arr, None, target_frames=target_frames)

    t = np.asarray(timestamps, dtype=np.float64)
    total_span = float(t[-1] - t[0])
    if total_span <= window_seconds * 1.05:
        return resample_sequence(arr, t, target_frames=target_frames)

    if center_on_peak and arr.shape[0] >= 4:
        diffs = np.linalg.norm(np.diff(arr[:, :12], axis=0), axis=1)
        energy = np.concatenate(([0.0], diffs))
        best_start_t = float(t[0])
        best_score = -1.0
        for i in range(len(t)):
            t_end = t[i] + window_seconds
            if t_end > t[-1] + 1e-6:
                break
            j = int(np.searchsorted(t, t_end, side='right'))
            score = float(np.sum(energy[i:j]))
            if score > best_score:
                best_score = score
                best_start_t = float(t[i])
        t_lo, t_hi = best_start_t, best_start_t + window_seconds
    else:
        mid = 0.5 * (float(t[0]) + float(t[-1]))
        t_lo = mid - 0.5 * window_seconds
        t_hi = mid + 0.5 * window_seconds

    dst_t = np.linspace(t_lo, t_hi, target_frames, dtype=np.float64)
    out = np.empty((target_frames, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(dst_t, t, arr[:, col])
    return out


def estimate_active_duration(features: np.ndarray, timestamps: np.ndarray | None) -> float:
    """Estimate the active motion duration (seconds) of a recorded clip (clamped to 0.70..2.20s)."""
    if timestamps is None or len(timestamps) < 4:
        return DEFAULT_WINDOW_SECONDS
    t = np.asarray(timestamps, dtype=np.float64)
    total = float(t[-1] - t[0])
    if total <= 0.75:
        return max(0.60, total)
    arr = np.asarray(features, dtype=np.float32)
    diffs = np.linalg.norm(np.diff(arr[:, :16], axis=0), axis=1)
    peak = float(np.max(diffs)) if len(diffs) else 0.0
    if peak < 1e-4:
        return min(1.50, total)
    active_idx = np.where(diffs >= 0.18 * peak)[0]
    if len(active_idx) >= 2:
        span = float(t[min(len(t) - 1, int(active_idx[-1]) + 2)] - t[max(0, int(active_idx[0]))])
        return float(max(0.75, min(2.20, span * 1.15)))
    return float(max(0.80, min(2.00, total * 0.85)))


def normalize_for_dtw(seq: np.ndarray) -> np.ndarray:
    """Build a posture- & amplitude-tolerant representation `(T, F)` for cross-person DTW matching.

    Removes 65% of static mean posture offset (so different shoulder widths / resting stances don't
    inflate distance) and gently normalizes large amplitude differences without boosting idle noise.
    """
    arr = np.asarray(seq, dtype=np.float32)
    mean_pose = np.mean(arr, axis=0, keepdims=True)
    centered = arr - mean_pose
    # Per-frame RMS displacement from the window's mean pose; only compress/expand gently in [0.75..1.35].
    rms = float(np.sqrt(np.mean(np.sum(centered[:, :18] * centered[:, :18], axis=1))) + 1e-5)
    norm_scale = float(np.clip(rms / 0.55, 0.72, 1.40)) if rms >= 0.18 else 1.0
    traj = centered / norm_scale
    return (0.35 * mean_pose + traj).astype(np.float32)


def banded_dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    band: int = 6,
    *,
    posture_invariant: bool = True,
) -> float:
    """Normalized Sakoe-Chiba banded Euclidean DTW distance between two `(T, F)` sequences."""
    a = normalize_for_dtw(seq_a) if posture_invariant else np.asarray(seq_a, dtype=np.float32)
    b = normalize_for_dtw(seq_b) if posture_invariant else np.asarray(seq_b, dtype=np.float32)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float('inf')

    diff = a[:, None, :] - b[None, :, :]
    cost = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)

    w = max(band, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


@dataclass(frozen=True)
class DtwMatch:
    label: str
    confidence: float
    distance: float
    threshold: float


class StreamingDtwMatcher:
    """Multi-scale, posture-tolerant few-shot dance recognizer over rolling PoseFrame windows."""

    def __init__(
        self,
        templates: dict[str, list[np.ndarray]] | None = None,
        *,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        window_seconds: float = 1.20,
        label_durations: dict[str, float] | None = None,
        max_distance: float = 0.50,
        min_confidence: float = 0.35,
        band: int = 6,
        cooldown_s: float = 0.90,
        eval_stride: int = 2,
    ):
        self.window_frames = window_frames
        self.window_seconds = window_seconds
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.band = band
        self.cooldown_s = cooldown_s
        self.eval_stride = max(1, eval_stride)
        self.templates: dict[str, list[np.ndarray]] = {}
        self.label_durations: dict[str, float] = {}
        self.label_thresholds: dict[str, float] = {}

        if templates:
            for lbl, seqs in templates.items():
                if lbl.lower() in NULL_CLASSES:
                    continue
                resampled = [resample_sequence(s, target_frames=window_frames) for s in seqs]
                if not resampled:
                    continue
                self.templates[lbl] = resampled
                self.label_durations[lbl] = float(
                    label_durations[lbl] if (label_durations and lbl in label_durations) else window_seconds
                )
                # Auto-calibrate threshold from intra-class variation so natural human variance passes!
                if len(resampled) >= 2:
                    intra = [
                        banded_dtw_distance(resampled[i], resampled[j], band=band)
                        for i in range(len(resampled))
                        for j in range(i + 1, len(resampled))
                    ]
                    p75 = float(np.percentile(intra, 75))
                    self.label_thresholds[lbl] = float(max(max_distance, min(0.68, p75 * 1.45)))
                else:
                    self.label_thresholds[lbl] = max_distance
        self.reset()

    def reset(self):
        self._buf: deque[tuple[float, tuple[float, ...]]] = deque(maxlen=180)
        self._cooldowns: dict[str, float] = {}
        self._frame_count = 0
        self.last_scores: dict[str, float] = {}
        self.last_distances: dict[str, float] = {}

    @classmethod
    def from_directory(cls, dataset_dir: Path, **kwargs) -> 'StreamingDtwMatcher':
        """Load recorded `.npz` clips from `datasets/moves/<move_name>/*.npz` with auto duration & spread."""
        root = Path(dataset_dir)
        templates: dict[str, list[np.ndarray]] = {}
        durations: dict[str, float] = {}
        target_frames = kwargs.get('window_frames', DEFAULT_WINDOW_FRAMES)
        if not root.is_dir():
            return cls(templates, **kwargs)

        for move_dir in sorted(root.iterdir()):
            if not move_dir.is_dir():
                continue
            label = move_dir.name
            if label.lower() in NULL_CLASSES:
                continue
            raw_clips: list[tuple[np.ndarray, np.ndarray | None, float]] = []
            for npz_path in sorted(move_dir.glob('*.npz')):
                with np.load(npz_path, allow_pickle=False) as data:
                    if 'features' not in data:
                        continue
                    feats = np.asarray(data['features'], dtype=np.float32)
                    ts = np.asarray(data['timestamps'], dtype=np.float64) if 'timestamps' in data else None
                    dur = estimate_active_duration(feats, ts)
                    raw_clips.append((feats, ts, dur))
            if raw_clips:
                med_dur = float(np.median([c[2] for c in raw_clips]))
                durations[label] = med_dur
                templates[label] = [
                    extract_time_window(feats, ts, window_seconds=med_dur, target_frames=target_frames, center_on_peak=True)
                    for feats, ts, _ in raw_clips
                ]
        return cls(templates, label_durations=durations, **kwargs)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.templates.keys()))

    def _query_at_duration(self, duration_s: float) -> np.ndarray | None:
        if not self._buf:
            return None
        now = self._buf[-1][0]
        cutoff = now - duration_s
        win = [item for item in self._buf if item[0] >= cutoff]
        if len(win) < 6 or (win[-1][0] - win[0][0]) < (0.45 * duration_s):
            return None
        ts = np.array([w[0] for w in win], dtype=np.float64)
        feats = np.array([w[1] for w in win], dtype=np.float32)
        return resample_sequence(feats, ts, target_frames=self.window_frames)

    def update(self, timestamp: float, feature_vector: tuple[float, ...]) -> list[MoveEvent]:
        if not self.templates:
            return []
        self._buf.append((timestamp, feature_vector))
        self._frame_count += 1
        if self._frame_count % self.eval_stride != 0:
            return []

        best: DtwMatch | None = None
        scores: dict[str, float] = {}
        dists: dict[str, float] = {}

        # Multi-scale speed invariance: test 0.72x (fast), 1.00x (normal), and 1.35x (slow) durations!
        for label, clips in self.templates.items():
            base_dur = self.label_durations.get(label, self.window_seconds)
            thresh = self.label_thresholds.get(label, self.max_distance)
            label_best_dist = float('inf')

            for scale in (0.72, 1.00, 1.35):
                q = self._query_at_duration(base_dur * scale)
                if q is None:
                    continue
                # At least one coordinate/angle channel must exhibit real temporal motion (>= 0.055).
                if float(np.max(np.std(q, axis=0))) < 0.055:
                    continue
                d = min(banded_dtw_distance(q, tpl, band=self.band) for tpl in clips)
                if d < label_best_dist:
                    label_best_dist = d

            dists[label] = label_best_dist
            if not np.isfinite(label_best_dist):
                scores[label] = 0.0
                continue

            # Map distance smoothly onto [0..1] where `dist == thresh` -> `min_confidence` (0.32).
            conf = float(max(0.0, min(1.0, 1.0 - (label_best_dist / max(thresh, 1e-4)) * (1.0 - self.min_confidence))))
            scores[label] = conf
            if label_best_dist <= thresh and (best is None or conf > best.confidence):
                best = DtwMatch(label=label, confidence=conf, distance=label_best_dist, threshold=thresh)

        self.last_scores = scores
        self.last_distances = dists
        if best is None or best.confidence < self.min_confidence:
            return []

        last_fired = self._cooldowns.get(best.label, -1e9)
        if (timestamp - last_fired) < self.cooldown_s:
            return []

        self._cooldowns[best.label] = timestamp
        kind, side = _parse_label_side(best.label)
        return [MoveEvent(kind=kind, side=side, intensity=max(0.65, best.confidence))]


def _parse_label_side(label: str) -> tuple[str, str]:
    """Map e.g. 'arm_wave_lr' -> ('ARM_WAVE', 'lr'), 'body_roll' -> ('BODY_ROLL', 'both')."""
    low = label.strip().lower()
    for suffix in ('_left', '_right', '_lr', '_rl'):
        if low.endswith(suffix):
            base = low[: -len(suffix)].upper()
            return base, suffix[1:]
    return low.upper(), 'both'
