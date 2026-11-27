"""Moves Lab Diagnostic Flight Recorder & Dataset Health Auditor.

Records everything happening inside a zero-audio Moves Lab session (`--diagnose [SECONDS]`):
  - Raw 15-joint landmarks + sanitized 24-D canonical feature vectors
  - Per-frame Tier 2 DTW distances/thresholds/scores & Tier 2 TCN raw/EMA probabilities
  - `[Space]` / `[m]` live "Mark Attempt!" markers so any failed move attempt can be inspected
    frame-by-frame by both humans and AI
  - Automatic Dataset Health Audit of `datasets/moves/` (flags low-2D-motion moves like frontal
    body rolls, missing `idle_random`, empty folders, and intra- vs inter-class separability).
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any
import numpy as np

from ..recorder import KEY_LANDMARKS
from .dtw import NULL_CLASSES, banded_dtw_distance, estimate_active_duration, extract_time_window

FEATURE_NAMES: tuple[str, ...] = (
    'l_sh_x', 'l_sh_y', 'r_sh_x', 'r_sh_y',
    'l_el_x', 'l_el_y', 'r_el_x', 'r_el_y',
    'l_wr_x', 'l_wr_y', 'r_wr_x', 'r_wr_y',
    'l_wr_vx', 'l_wr_vy', 'r_wr_vx', 'r_wr_vy',
    'l_bend', 'r_bend',
    'l_kn_x', 'l_kn_y', 'r_kn_x', 'r_kn_y',
    'hip_elev', 'torso_lean',
)


def _round(v: Any):
    if isinstance(v, float):
        return round(v, 4) if np.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(val) for k, val in v.items()}
    return v


def audit_dataset_dir(dataset_dir: Path | None) -> dict:
    """Inspect `datasets/moves/` and return structured health metrics + actionable warnings."""
    if dataset_dir is None or not Path(dataset_dir).is_dir():
        return {'exists': False, 'classes': {}, 'warnings': ['Dataset directory does not exist yet.']}

    root = Path(dataset_dir)
    classes_info: dict[str, dict] = {}
    windows_by_class: dict[str, list[np.ndarray]] = {}
    warnings: list[str] = []

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        npz_files = sorted(sub.glob('*.npz'))
        if not npz_files:
            warnings.append(f'Empty move folder `{sub.name}/` (0 clips) — safe to delete with `rm -rf {sub}`.')
            continue

        durs = []
        motion_stds = []
        per_col_stds = []
        wins = []
        for f in npz_files:
            with np.load(f, allow_pickle=False) as d:
                if 'features' not in d:
                    continue
                feats = np.asarray(d['features'], dtype=np.float32)
                ts = np.asarray(d['timestamps'], dtype=np.float64) if 'timestamps' in d else None
                dur = float(ts[-1] - ts[0]) if (ts is not None and len(ts) > 1) else (len(feats) / 30.0)
                act_dur = estimate_active_duration(feats, ts)
                col_std = np.std(feats, axis=0)
                durs.append(dur)
                motion_stds.append(float(np.mean(col_std)))
                per_col_stds.append(col_std)
                wins.append(extract_time_window(feats, ts, window_seconds=act_dur))

        if not wins:
            continue
        avg_col_std = np.mean(np.stack(per_col_stds, axis=0), axis=0)
        top_idx = [int(i) for i in np.argsort(-avg_col_std)[:4]]
        top_channels = [f'{FEATURE_NAMES[i]}({avg_col_std[i]:.2f})' for i in top_idx]

        intra = [
            banded_dtw_distance(wins[i], wins[j])
            for i in range(len(wins))
            for j in range(i + 1, len(wins))
        ]
        intra_med = float(np.median(intra)) if intra else 0.0
        intra_max = float(np.max(intra)) if intra else 0.0
        mean_m = float(np.mean(motion_stds))

        classes_info[sub.name] = {
            'clips': len(wins),
            'median_clip_s': round(float(np.median(durs)), 2),
            'mean_motion_std': round(mean_m, 3),
            'intra_dtw_median': round(intra_med, 3),
            'intra_dtw_max': round(intra_max, 3),
            'top_channels': top_channels,
        }
        windows_by_class[sub.name] = wins

        if sub.name.lower() not in NULL_CLASSES:
            if len(wins) < 8:
                warnings.append(
                    f'`{sub.name}` only has {len(wins)} clips — works for DTW, but record 12–20 varied clips for strong TCN generalization.'
                )
            if mean_m < 0.065:
                warnings.append(
                    f'`{sub.name}` has very low 2D motion (`mean_std={mean_m:.3f}`, top={top_channels[:2]}). '
                    f'Pure depth movements facing the camera (like a frontal body roll) have almost zero 2D keypoint travel '
                    f'because Pose models lack spine/ribcage depth points — perform at a 35–45° angle or add arm/head motion!'
                )

    positive_classes = [c for c in classes_info if c.lower() not in NULL_CLASSES]
    if positive_classes and not any(c.lower() in NULL_CLASSES for c in classes_info):
        warnings.append(
            'Missing `idle_random` negative class! Record 15–25 clips of casual standing/swaying/fidgeting '
            '(`--record-move idle_random`) before training the Keras TCN so everyday motion is cleanly rejected.'
        )

    # Pairwise cross-class DTW separation check.
    cross_pairs: dict[str, float] = {}
    pos_sorted = sorted(positive_classes)
    for i in range(len(pos_sorted)):
        for j in range(i + 1, len(pos_sorted)):
            ca, cb = pos_sorted[i], pos_sorted[j]
            dists = [banded_dtw_distance(wa, wb) for wa in windows_by_class[ca] for wb in windows_by_class[cb]]
            med_cross = float(np.median(dists))
            cross_pairs[f'{ca} <-> {cb}'] = round(med_cross, 3)
            if med_cross < 0.45:
                warnings.append(f'`{ca}` and `{cb}` look very similar in 2D (`cross_dtw={med_cross:.3f}`) — risk of confusion!')

    return {
        'exists': True,
        'dataset_dir': str(root),
        'classes': classes_info,
        'cross_dtw_median': cross_pairs,
        'warnings': warnings,
    }


class LabSessionRecorder:
    """Streams per-frame Moves Lab vision, feature vectors, DTW/TCN scores & user attempt markers to JSONL."""

    def __init__(self, jsonl_path: Path, metadata: dict):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.jsonl_path.open('w', encoding='utf-8')
        self.frames_logged = 0
        self._fh.write(json.dumps({'type': 'header', 'metadata': _round(metadata)}) + '\n')

    def record_frame(
        self,
        *,
        elapsed_s: float,
        points: list[tuple[float, float, float, float]] | None,
        pose_frame,
        features,
        dtw_scores: dict[str, float] | None = None,
        dtw_distances: dict[str, float] | None = None,
        dtw_thresholds: dict[str, float] | None = None,
        tcn_probs: dict[str, float] | None = None,
        tcn_raw_probs: dict[str, float] | None = None,
        marked_attempt: bool = False,
    ):
        landmarks = None
        if points and len(points) == 33:
            landmarks = {
                name: [round(points[idx][0], 4), round(points[idx][1], 4), round(points[idx][2], 4), round(points[idx][3], 3)]
                for idx, name in KEY_LANDMARKS
            }
        entry = {
            'type': 'frame',
            't': round(elapsed_s, 4),
            'tracked': bool(features.tracked),
            'marked': bool(marked_attempt),
            'landmarks': landmarks,
            'fvec': _round(list(pose_frame.feature_vector)) if (pose_frame and pose_frame.tracked) else None,
            'kin': {
                'energy': round(float(features.energy), 3),
                'l_spd': round(float(features.left_speed), 3),
                'r_spd': round(float(features.right_speed), 3),
                'l_rvel': round(float(pose_frame.arm_reach_vel[0]), 3) if (pose_frame and pose_frame.tracked) else 0.0,
                'r_rvel': round(float(pose_frame.arm_reach_vel[1]), 3) if (pose_frame and pose_frame.tracked) else 0.0,
                'glitches': list(features.glitch_joints),
            },
            'moves': [
                {'kind': m.kind, 'side': m.side, 'intensity': round(float(m.intensity), 3)}
                for m in features.moves
            ],
            'dtw': {
                'scores': _round(dtw_scores or {}),
                'distances': _round(dtw_distances or {}),
                'thresholds': _round(dtw_thresholds or {}),
            } if dtw_scores is not None else None,
            'tcn': {
                'ema': _round(tcn_probs or {}),
                'raw': _round(tcn_raw_probs or {}),
            } if tcn_probs is not None else None,
        }
        self._fh.write(json.dumps(entry) + '\n')
        self.frames_logged += 1

    def close(self, summary: dict):
        if self._fh is not None:
            self._fh.write(json.dumps({'type': 'footer', 'frames': self.frames_logged, 'summary': _round(summary)}) + '\n')
            self._fh.flush()
            self._fh.close()
            self._fh = None


def write_lab_report(jsonl_path: Path, dataset_dir: Path | None = None) -> tuple[dict, Path]:
    """Analyze a recorded `lab_diag_*.jsonl` session and write a comprehensive `_report.md`."""
    jsonl_path = Path(jsonl_path)
    rows = [json.loads(line) for line in jsonl_path.open('r', encoding='utf-8')]
    header = next((r for r in rows if r.get('type') == 'header'), {})
    frames = [r for r in rows if r.get('type') == 'frame']

    dur = (frames[-1]['t'] - frames[0]['t']) if len(frames) >= 2 else 0.0
    fps = (len(frames) - 1) / max(dur, 1e-3) if len(frames) >= 2 else 0.0
    tracked_pct = 100.0 * sum(1 for f in frames if f.get('tracked')) / max(1, len(frames))

    move_counts: Counter[str] = Counter()
    all_detections: list[tuple[float, str, float]] = []
    mark_times: list[float] = []
    peak_scores: dict[str, tuple[float, float]] = {}  # label -> (best_score, at_t)

    for f in frames:
        t = float(f['t'])
        if f.get('marked'):
            mark_times.append(t)
        for m in f.get('moves', []):
            tag = f"{m['kind']}:{m['side']}"
            move_counts[tag] += 1
            all_detections.append((round(t, 2), tag, float(m['intensity'])))

        if f.get('dtw') and f['dtw'].get('scores'):
            for k, val in f['dtw']['scores'].items():
                if val is not None and val > peak_scores.get(f'dtw:{k}', (-1.0, 0.0))[0]:
                    peak_scores[f'dtw:{k}'] = (float(val), round(t, 2))
        if f.get('tcn') and f['tcn'].get('ema'):
            for k, val in f['tcn']['ema'].items():
                if val is not None and val > peak_scores.get(f'tcn:{k}', (-1.0, 0.0))[0]:
                    peak_scores[f'tcn:{k}'] = (float(val), round(t, 2))

    # Inspect each user-marked attempt (`[Space]` / `[m]`) in a [-1.6s .. +0.6s] window!
    attempt_diagnostics = []
    for mt in mark_times:
        win = [f for f in frames if (mt - 1.6) <= float(f['t']) <= (mt + 0.6)]
        fired = [f"{m['kind']}:{m['side']}@{f['t']:.2f}s" for f in win for m in f.get('moves', [])]
        fvecs = [f['fvec'] for f in win if f.get('fvec') is not None]
        motion_std = float(np.mean(np.std(np.asarray(fvecs, dtype=np.float32), axis=0))) if len(fvecs) >= 4 else 0.0

        best_dtw = {}
        min_dist = {}
        thresh_map = {}
        best_tcn = {}
        for f in win:
            if f.get('dtw'):
                for k, v in (f['dtw'].get('scores') or {}).items():
                    if v is not None:
                        best_dtw[k] = max(best_dtw.get(k, 0.0), float(v))
                for k, v in (f['dtw'].get('distances') or {}).items():
                    if v is not None:
                        min_dist[k] = min(min_dist.get(k, 99.0), float(v))
                thresh_map.update(f['dtw'].get('thresholds') or {})
            if f.get('tcn'):
                for k, v in (f['tcn'].get('ema') or {}).items():
                    if v is not None:
                        best_tcn[k] = max(best_tcn.get(k, 0.0), float(v))

        attempt_diagnostics.append({
            'mark_t': round(mt, 2),
            'motion_std': round(motion_std, 3),
            'fired': fired,
            'best_dtw_scores': {k: round(v, 3) for k, v in best_dtw.items()},
            'min_dtw_dists': {k: round(v, 3) for k, v in min_dist.items()},
            'dtw_thresholds': thresh_map,
            'best_tcn_probs': {k: round(v, 3) for k, v in best_tcn.items()},
        })

    ds_audit = audit_dataset_dir(dataset_dir)
    report_path = jsonl_path.with_name(jsonl_path.stem + '_report.md')

    lines = [
        '# Moves Lab Diagnostic Report',
        '',
        f'- **Duration**: `{dur:.2f} s` | **Pose FPS**: `{fps:.1f}` | **Tracked**: `{tracked_pct:.1f}%` | **Frames**: `{len(frames)}`',
        f'- **Metadata**: `{header.get("metadata", {})}`',
        f'- **Detections**: `{dict(move_counts)}`',
        f'- **Peak Tier 2 Scores**: `{ {k: f"{v[0]*100:.1f}% @ {v[1]}s" for k, v in peak_scores.items()} }`',
        '',
        '## 1. Dataset Health Audit (`datasets/moves/`)',
    ]
    if ds_audit.get('warnings'):
        for w in ds_audit['warnings']:
            lines.append(f'- **WARNING**: {w}')
    else:
        lines.append('- Dataset health looks clean!')

    if ds_audit.get('classes'):
        lines.extend([
            '',
            '| Move Folder | Clips | Med Duration | 2D Motion Std | Intra-DTW (Med / Max) | Top Moving Channels |',
            '| :--- | :---: | :---: | :---: | :---: | :--- |',
        ])
        for cname, info in ds_audit['classes'].items():
            lines.append(
                f"| `{cname}` | {info['clips']} | {info['median_clip_s']}s | {info['mean_motion_std']} | "
                f"{info['intra_dtw_median']} / {info['intra_dtw_max']} | {', '.join(info['top_channels'])} |"
            )
        if ds_audit.get('cross_dtw_median'):
            lines.append(f"\n- **Cross-class DTW separation (median)**: `{ds_audit['cross_dtw_median']}`")

    lines.extend(['', '## 2. User-Marked Attempts (`[Space]` / `[m]`)'])
    if not attempt_diagnostics:
        lines.append('- No manual markers pressed during session (press `Space` or `m` right after attempting a move!).')
    else:
        for ad in attempt_diagnostics:
            lines.append(
                f"- **Marker @ `{ad['mark_t']}s`**: 2D motion_std=`{ad['motion_std']}` | "
                f"Fired=`{ad['fired'] or 'NONE'}` | "
                f"DTW min_dist=`{ad['min_dtw_dists']}` (thresh=`{ad['dtw_thresholds']}`) | "
                f"TCN peak=`{ad['best_tcn_probs']}`"
            )

    lines.extend([
        '',
        '## 3. Detection Timeline',
        f"- `{all_detections if all_detections else 'No moves triggered'}`",
        '',
        '## 4. Second-by-Second Telemetry',
        '| Sec | Energy | L_Spd | R_Spd | Best Tier 2 Score | Triggered Moves | Marked |',
        '| :---: | :---: | :---: | :---: | :--- | :--- | :---: |',
    ])
    max_sec = int(dur) + 1
    for sec in range(max_sec):
        s_frames = [f for f in frames if sec <= float(f['t']) < (sec + 1)]
        if not s_frames:
            continue
        avg_e = sum(f['kin']['energy'] for f in s_frames) / len(s_frames)
        pk_l = max(f['kin']['l_spd'] for f in s_frames)
        pk_r = max(f['kin']['r_spd'] for f in s_frames)
        best_t2_str = '-'
        best_t2_val = -1.0
        for f in s_frames:
            for src, subk in (('dtw', 'scores'), ('tcn', 'ema')):
                if f.get(src):
                    for k, v in (f[src].get(subk) or {}).items():
                        if v is not None and float(v) > best_t2_val:
                            best_t2_val = float(v)
                            best_t2_str = f'{k}:{v*100:.0f}%'
        mvs = [f"{m['kind']}:{m['side']}" for f in s_frames for m in f.get('moves', [])]
        marked = 'MARK!' if any(f.get('marked') for f in s_frames) else ''
        lines.append(
            f'| {sec} | {avg_e:.2f} | {pk_l:.2f} | {pk_r:.2f} | {best_t2_str} | {", ".join(mvs) or "-"} | {marked} |'
        )

    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = {
        'duration_s': round(dur, 2),
        'fps': round(fps, 1),
        'detections': dict(move_counts),
        'attempt_diagnostics': attempt_diagnostics,
        'dataset_audit': ds_audit,
    }
    return summary, report_path
