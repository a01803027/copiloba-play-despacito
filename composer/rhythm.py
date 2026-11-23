"""FPS-independent multi-source body tempo (BPM) tracker.

Combines hip bounce, head nodding, and rhythmic arm speed resampled onto a
uniform 30 Hz grid so webcam frame-rate jitter (15-60 FPS) never breaks
periodicity estimation.
"""
from collections import deque
import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class MultiSourceTempoTracker:
    """Autocorrelation + smooth PLL tempo estimator over head, hip, and arms."""

    RESAMPLE_HZ = 30.0
    WINDOW_SECONDS = 2.8  # ~84 samples at 30 Hz: fast lock (~1.8s) + clear peaks

    def __init__(self, initial_bpm: float = 100.0):
        self.bpm = initial_bpm
        self.confidence = 0.0
        self.source = 'none'
        self._samples: deque[tuple[float, float, float, float, float, float]] = deque(maxlen=240)
        self._last_estimate_t = -1e9

    def reset(self, bpm: float | None = None):
        self._samples.clear()
        self.confidence = 0.0
        self.source = 'none'
        self._last_estimate_t = -1e9
        if bpm is not None:
            self.bpm = bpm

    def update(
        self,
        timestamp: float,
        hip_y: float,
        head_y: float,
        arm_speed: float,
        shoulder_tilt: float = 0.0,
        leg_pulse: float = 0.0,
        bpm_min: float = 68.0,
        bpm_max: float = 155.0,
    ) -> tuple[float, float]:
        """Push a new observation and return (estimated_bpm, confidence)."""
        if self._samples and timestamp <= self._samples[-1][0]:
            return self.bpm, self.confidence
        if self._samples and (timestamp - self._samples[-1][0]) > 0.4:
            self._samples.clear()
            self.confidence = 0.0
            self.source = 'none'

        self._samples.append((timestamp, hip_y, head_y, arm_speed, shoulder_tilt, leg_pulse))
        # Prune older than WINDOW_SECONDS + 0.5s buffer.
        cutoff_t = timestamp - (self.WINDOW_SECONDS + 0.5)
        while len(self._samples) > 2 and self._samples[0][0] < cutoff_t:
            self._samples.popleft()

        span = self._samples[-1][0] - self._samples[0][0]
        if len(self._samples) < 20 or span < 1.5:
            self.confidence *= 0.94
            return self.bpm, self.confidence

        # Re-estimate at ~10 Hz to keep CPU tiny.
        if timestamp - self._last_estimate_t < 0.095:
            return self.bpm, self.confidence
        self._last_estimate_t = timestamp

        n_grid = min(int(span * self.RESAMPLE_HZ), int(self.WINDOW_SECONDS * self.RESAMPLE_HZ))
        if n_grid < 42:
            return self.bpm, self.confidence

        hip_s, head_s, arm_s, sh_s, leg_s = self._resample(timestamp, n_grid)
        candidates = (
            ('legs', leg_s, 0.03),
            ('hip', hip_s, 0.0025),
            ('head', head_s, 0.0025),
            ('shoulders', sh_s, 0.05),
            ('arms', arm_s, 0.08),
        )

        best_score = 0.0
        best_bpm = self.bpm
        best_src = 'none'
        # Evaluate lags covering 60..165 BPM at 30 Hz (lag = 1800 / bpm => 11..30).
        min_lag = max(10, int(round(60.0 * self.RESAMPLE_HZ / 165.0)))
        max_lag = min(n_grid // 2 - 1, int(round(60.0 * self.RESAMPLE_HZ / 60.0)))
        if max_lag <= min_lag + 2:
            return self.bpm, self.confidence

        for name, series, min_std in candidates:
            bpm_est, score = self._autocorr_bpm(series, min_lag, max_lag, min_std, bpm_min, bpm_max)
            if score > best_score:
                best_score = score
                best_bpm = bpm_est
                best_src = name

        if best_score >= 0.36:
            # Smooth PLL lock: faster adaptation when confidence is high.
            alpha = 0.24 + 0.30 * _clamp((best_score - 0.36) / 0.5, 0.0, 1.0)
            self.bpm = _clamp(self.bpm + alpha * (best_bpm - self.bpm), bpm_min, bpm_max)
            self.confidence = _clamp(best_score, 0.0, 1.0)
            self.source = best_src
        else:
            self.confidence *= 0.91
            if self.confidence < 0.15:
                self.source = 'none'
        return self.bpm, self.confidence

    def _resample(self, end_t: float, n_grid: int) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        samples = list(self._samples)
        hip_out: list[float] = []
        head_out: list[float] = []
        arm_out: list[float] = []
        sh_out: list[float] = []
        leg_out: list[float] = []
        idx = 0
        dt = 1.0 / self.RESAMPLE_HZ
        start_t = end_t - (n_grid - 1) * dt
        last_idx = len(samples) - 1
        for k in range(n_grid):
            t = start_t + k * dt
            while idx + 1 < last_idx and samples[idx + 1][0] < t:
                idx += 1
            t0, h0, hd0, a0, s0, l0 = samples[idx]
            if idx + 1 <= last_idx and samples[idx + 1][0] > t0:
                t1, h1, hd1, a1, s1, l1 = samples[idx + 1]
                frac = _clamp((t - t0) / (t1 - t0), 0.0, 1.0)
                hip_out.append(h0 + frac * (h1 - h0))
                head_out.append(hd0 + frac * (hd1 - hd0))
                arm_out.append(a0 + frac * (a1 - a0))
                sh_out.append(s0 + frac * (s1 - s0))
                leg_out.append(l0 + frac * (l1 - l0))
            else:
                hip_out.append(h0)
                head_out.append(hd0)
                arm_out.append(a0)
                sh_out.append(s0)
                leg_out.append(l0)
        return hip_out, head_out, arm_out, sh_out, leg_out

    def _autocorr_bpm(
        self,
        series: list[float],
        min_lag: int,
        max_lag: int,
        min_std: float,
        bpm_min: float,
        bpm_max: float,
    ) -> tuple[float, float]:
        n = len(series)
        # Remove linear drift (mean + slow moving trend) so slow posture changes don't mask beats.
        mean = sum(series) / n
        centered = [v - mean for v in series]
        # Simple 1st-order high-pass detrend (alpha ~ 0.85) to reject < 0.7 Hz sway.
        hp = [0.0] * n
        prev_x = centered[0]
        prev_y = 0.0
        for i in range(1, n):
            y = 0.82 * (prev_y + centered[i] - prev_x)
            hp[i] = y
            prev_x = centered[i]
            prev_y = y

        variance = sum(v * v for v in hp) / n
        if math.sqrt(variance) < min_std:
            return self.bpm, 0.0

        scores: list[float] = []
        for lag in range(min_lag, max_lag + 1):
            a = hp[lag:]
            b = hp[:-lag]
            norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
            scores.append(sum(x * y for x, y in zip(a, b)) / norm if norm > 1e-9 else 0.0)

        # Find local peaks; prefer shortest strong peak (fundamental period, not 2x sub-harmonic).
        peaks = [
            i for i in range(1, len(scores) - 1)
            if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1] and scores[i] > 0.35
        ]
        if not peaks:
            return self.bpm, 0.0

        strongest = max(scores[i] for i in peaks)
        best_i = next(i for i in peaks if scores[i] >= strongest - 0.06)
        score = scores[best_i]

        # Parabolic sub-sample peak refinement.
        left, right = scores[best_i - 1], scores[best_i + 1]
        denom = left + right - 2.0 * score
        offset = 0.5 * (left - right) / denom if denom < -1e-9 else 0.0
        exact_lag = min_lag + best_i + _clamp(offset, -0.5, 0.5)
        raw_bpm = (60.0 * self.RESAMPLE_HZ) / max(exact_lag, 1.0)

        # Octave resolution into preferred musical range [bpm_min, bpm_max].
        resolved = raw_bpm
        if resolved < bpm_min and resolved * 2.0 <= bpm_max + 8.0:
            resolved *= 2.0
        elif resolved > bpm_max and resolved * 0.5 >= bpm_min - 8.0:
            resolved *= 0.5
        return _clamp(resolved, bpm_min, bpm_max), score
