"""Tier 2 Option A — Few-Shot Streaming Subsequence DTW Dance Move Matcher (Zero Training).

Works immediately after recording just 3-5 example clips into `datasets/moves/<move_name>/*.npz`.
Uses banded Sakoe-Chiba DTW over time-normalized 24-frame (0.80s @ 30 Hz) canonical PoseFrame
feature windows, with optional auto-calibration against `idle_random` negative clips.
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
    """Extract a true-speed `window_seconds` slice from a clip and resample to `target_frames`.

    Prevents time-scale distortion when clips recorded at `--clip-seconds 1.2` (or longer)
    are compared against an online `0.80 s` streaming window.
    """
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((target_frames, FEATURE_DIM), dtype=np.float32)
    if timestamps is None or len(timestamps) != arr.shape[0]:
        return resample_sequence(arr, None, target_frames=target_frames)

    t = np.asarray(timestamps, dtype=np.float64)
    total_span = float(t[-1] - t[0])
    if total_span <= window_seconds * 1.05:
        return resample_sequence(arr, t, target_frames=target_frames)

    # Find the `window_seconds` sub-interval with peak motion energy (or center slice).
    if center_on_peak and arr.shape[0] >= 4:
        # Frame-to-frame motion magnitude across upper+lower body coordinates.
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


def banded_dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band: int = 4) -> float:
    """Normalized Sakoe-Chiba banded Euclidean DTW distance between two `(T, F)` sequences."""
    a = np.asarray(seq_a, dtype=np.float32)
    b = np.asarray(seq_b, dtype=np.float32)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float('inf')

    # Vectorized pairwise frame Euclidean distances: (n, m).
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


class StreamingDtwMatcher:
    """Zero-training few-shot dance recognizer over rolling 0.80s PoseFrame feature windows."""

    def __init__(
        self,
        templates: dict[str, list[np.ndarray]] | None = None,
        *,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_distance: float = 0.55,
        band: int = 4,
        cooldown_s: float = 0.85,
        eval_stride: int = 2,
    ):
        self.window_frames = window_frames
        self.window_seconds = window_seconds
        self.max_distance = max_distance
        self.band = band
        self.cooldown_s = cooldown_s
        self.eval_stride = max(1, eval_stride)
        self.templates: dict[str, list[np.ndarray]] = {}
        if templates:
            for lbl, seqs in templates.items():
                if lbl.lower() in NULL_CLASSES:
                    continue
                self.templates[lbl] = [resample_sequence(s, target_frames=window_frames) for s in seqs]
        self.reset()

    def reset(self):
        self._buf: deque[tuple[float, tuple[float, ...]]] = deque(maxlen=120)
        self._cooldowns: dict[str, float] = {}
        self._frame_count = 0
        self.last_scores: dict[str, float] = {}

    @classmethod
    def from_directory(cls, dataset_dir: Path, **kwargs) -> 'StreamingDtwMatcher':
        """Load recorded `.npz` clips from `datasets/moves/<move_name>/*.npz`."""
        root = Path(dataset_dir)
        templates: dict[str, list[np.ndarray]] = {}
        if not root.is_dir():
            return cls(templates, **kwargs)

        for move_dir in sorted(root.iterdir()):
            if not move_dir.is_dir():
                continue
            label = move_dir.name
            if label.lower() in NULL_CLASSES:
                continue
            clips: list[np.ndarray] = []
            for npz_path in sorted(move_dir.glob('*.npz')):
                with np.load(npz_path, allow_pickle=False) as data:
                    if 'features' not in data:
                        continue
                    ts = data['timestamps'] if 'timestamps' in data else None
                    clips.append(
                        extract_time_window(
                            data['features'],
                            ts,
                            window_seconds=kwargs.get('window_seconds', DEFAULT_WINDOW_SECONDS),
                            target_frames=kwargs.get('window_frames', DEFAULT_WINDOW_FRAMES),
                            center_on_peak=True,
                        )
                    )
            if clips:
                templates[label] = clips
        return cls(templates, **kwargs)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.templates.keys()))

    def update(self, timestamp: float, feature_vector: tuple[float, ...]) -> list[MoveEvent]:
        if not self.templates:
            return []
        self._buf.append((timestamp, feature_vector))
        self._frame_count += 1
        if self._frame_count % self.eval_stride != 0:
            return []

        cutoff = timestamp - self.window_seconds
        win = [item for item in self._buf if item[0] >= cutoff]
        if len(win) < 8 or (win[-1][0] - win[0][0]) < (0.45 * self.window_seconds):
            return []

        ts = np.array([w[0] for w in win], dtype=np.float64)
        feats = np.array([w[1] for w in win], dtype=np.float32)
        query = resample_sequence(feats, ts, target_frames=self.window_frames)

        # Require actual body motion over time (mean per-channel temporal std along axis=0).
        if float(np.mean(np.std(query, axis=0))) < 0.025:
            self.last_scores = {lbl: 0.0 for lbl in self.templates}
            return []

        best: DtwMatch | None = None
        scores: dict[str, float] = {}
        for label, clips in self.templates.items():
            dist = min(banded_dtw_distance(query, tpl, band=self.band) for tpl in clips)
            conf = float(max(0.0, min(1.0, 1.0 - dist / max(self.max_distance, 1e-4))))
            scores[label] = conf
            if dist <= self.max_distance and (best is None or dist < best.distance):
                best = DtwMatch(label=label, confidence=conf, distance=dist)

        self.last_scores = scores
        if best is None or best.confidence < 0.40:
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
