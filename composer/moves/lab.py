"""Standalone Moves Lab Workbench — Pure Vision, Move Recognition, Guided Clip Recorder & ML HUD.

Usage:
  1) Live camera debugging (zero audio — Tier 1 kinematics + optional Tier 2 TCN / DTW):
      .venv-mac/bin/python -m composer.moves.lab
      .venv-mac/bin/python -m composer.moves.lab --tcn-model models/dance_tcn.npz
      .venv-mac/bin/python -m composer.moves.lab --dtw-dir datasets/moves

  2) Guided dance move clip recorder (saves labeled `(N, 24)` .npz clips for DTW & Keras TCN):
      .venv-mac/bin/python -m composer.moves.lab --record-move arm_wave_lr --clips 10 --clip-seconds 1.2
      .venv-mac/bin/python -m composer.moves.lab --record-move idle_random --clips 12 --clip-seconds 1.2

  3) Offline replay & before/after benchmark against any recorded diagnostic JSONL:
      .venv-mac/bin/python -m composer.moves.lab --replay artifacts/diag_house_20261123_182101.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import time
import numpy as np

from ..cameras import resolve_camera
from ..kinematics.body import FeatureExtractor
from ..recorder import KEY_LANDMARKS
from ..vision import AsyncPoseWorker, LatestCamera
from .pose_frame import CANONICAL_JOINTS

ROOT = Path(__file__).resolve().parents[2]
_NAME_TO_IDX: dict[str, int] = {name: idx for idx, name in CANONICAL_JOINTS}
_RECORDER_NAME_TO_IDX: dict[str, int] = {name: idx for idx, name in KEY_LANDMARKS}


def _reconstruct_33_points(landmark_dict: dict | None) -> list[tuple[float, float, float, float]] | None:
    """Reconstruct a 33-point MediaPipe list from the compact 15-landmark dict saved in diagnostic JSONL."""
    if not landmark_dict:
        return None
    pts: list[tuple[float, float, float, float]] = [(0.5, 0.5, 0.0, 0.0)] * 33
    for name, idx in _RECORDER_NAME_TO_IDX.items():
        val = landmark_dict.get(name)
        if val and len(val) >= 4:
            pts[idx] = (float(val[0]), float(val[1]), float(val[2]), float(val[3]))
    return pts


def save_pose_clip(out_dir: Path, label: str, features_list: list[tuple[float, ...]], timestamps: list[float]) -> Path:
    """Save a recorded canonical 24-D pose sequence into `datasets/moves/<label>/clip_<stamp>.npz`."""
    target_dir = Path(out_dir) / label
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    existing = len(list(target_dir.glob('*.npz')))
    out_path = target_dir / f'{label}_{stamp}_{existing + 1:03d}.npz'
    np.savez(
        out_path,
        label=np.array(label),
        features=np.asarray(features_list, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )
    return out_path


def replay_jsonl(
    jsonl_path: Path,
    aspect: float = 480.0 / 640.0,
    tcn_model_path: Path | None = None,
    dtw_dir: Path | None = None,
) -> dict:
    """Replay a recorded diagnostic JSONL through the current Tier 0 + Tier 1 + optional Tier 2 + Arbiter pipeline."""
    extractor = FeatureExtractor(tcn_model_path=tcn_model_path, dtw_dir=dtw_dir)
    old_moves: Counter[str] = Counter()
    new_moves: Counter[str] = Counter()
    old_max_spd = 0.0
    new_max_spd = 0.0
    old_double_stomps = 0
    new_double_stomps = 0
    new_timeline: list[tuple[float, str, str, float]] = []
    pose_frames = 0
    wiggle_frames = 0

    with Path(jsonl_path).open('r', encoding='utf-8') as fh:
        for line in fh:
            row = json.loads(line)
            if row.get('type') != 'frame' or not row.get('new_pose'):
                continue
            pose_frames += 1
            t = float(row['t'])
            old_f = row.get('features') or {}
            old_max_spd = max(old_max_spd, float(old_f.get('left_speed', 0.0)), float(old_f.get('right_speed', 0.0)))
            om = old_f.get('moves') or []
            if sum(1 for m in om if m.get('kind') == 'KNEE_STOMP') >= 2:
                old_double_stomps += 1
            for m in om:
                old_moves[f"{m['kind']}:{m['side']}"] += 1

            pts = _reconstruct_33_points(row.get('landmarks'))
            if pts is None:
                feat = extractor.missing(t)
            else:
                feat = extractor.update(pts, t, aspect=aspect)

            new_max_spd = max(new_max_spd, feat.left_speed, feat.right_speed)
            if feat.wiggle_suppressed:
                wiggle_frames += 1
            if sum(1 for m in feat.moves if m.kind == 'KNEE_STOMP') >= 2:
                new_double_stomps += 1
            for m in feat.moves:
                new_moves[f'{m.kind}:{m.side}'] += 1
                new_timeline.append((round(t, 2), m.kind, m.side, round(m.intensity, 2)))

    return {
        'file': str(jsonl_path),
        'pose_frames': pose_frames,
        'old_max_wrist_speed': round(old_max_spd, 2),
        'new_max_wrist_speed': round(new_max_spd, 2),
        'glitches_blocked': extractor.sanitizer.total_glitches_blocked,
        'wiggle_frames': wiggle_frames,
        'arbiter_suppressed': extractor.move_recognizer.arbiter.suppressed_count,
        'old_double_stomps': old_double_stomps,
        'new_double_stomps': new_double_stomps,
        'old_moves': dict(old_moves),
        'new_moves': dict(new_moves),
        'timeline': new_timeline,
    }


def _print_replay_report(res: dict):
    print(f"\n=== MOVES LAB OFFLINE REPLAY: {res['file']} ({res['pose_frames']} pose frames) ===")
    print(f"  Max wrist speed (old 3D vs new 2D): {res['old_max_wrist_speed']:.2f} -> {res['new_max_wrist_speed']:.2f} torso/s")
    print(f"  Anti-wiggle glitches clamped:       {res['glitches_blocked']} joints across {res['wiggle_frames']} frames")
    print(f"  Arbiter conflicts suppressed:       {res['arbiter_suppressed']}")
    print(f"  Simultaneous double-knee stomps:    {res['old_double_stomps']} -> {res['new_double_stomps']}")
    all_keys = sorted(set(res['old_moves']) | set(res['new_moves']))
    print('\n  Move Type                  | Old Log | New Pipeline')
    print('  ---------------------------+---------+-------------')
    for k in all_keys:
        print(f"  {k:<26s} | {res['old_moves'].get(k, 0):>7d} | {res['new_moves'].get(k, 0):>12d}")
    print(f"\n  Total detections: {sum(res['old_moves'].values())} -> {sum(res['new_moves'].values())}")


def _draw_lab_hud(
    cv2,
    image,
    features,
    *,
    fps: float,
    infer_ms: float,
    model_name: str,
    glitches_total: int,
    suppressed_total: int,
    ticker: deque,
    flash: tuple[str, float],
    reach_vels: tuple[float, float] = (0.0, 0.0),
    ml_probs: dict[str, float] | None = None,
    rec_overlay: tuple[str, float, tuple[int, int, int]] | None = None,
):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 128), (12, 16, 24), -1)
    cv2.rectangle(overlay, (0, h - 132), (310, h), (12, 16, 24), -1)
    if ml_probs:
        panel_h = 28 + 20 * min(8, len(ml_probs))
        cv2.rectangle(overlay, (w - 255, h - panel_h - 26), (w, h - 26), (12, 16, 24), -1)
    cv2.rectangle(overlay, (0, h - 26), (w, h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

    status = 'TRACKED' if features.tracked else 'SEARCHING...'
    status_col = (80, 240, 140) if features.tracked else (70, 90, 250)
    header = (
        f'MOVES LAB (ZERO AUDIO) | {model_name.upper()} | '
        f'{fps:4.1f} FPS ({infer_ms:4.1f} ms) | '
        f'GLITCHES BLOCKED: {glitches_total} | ARB SUPPRESSED: {suppressed_total}'
    )
    cv2.putText(image, header, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 242, 250), 1, cv2.LINE_AA)
    cv2.putText(image, status, (w - 115, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_col, 2, cv2.LINE_AA)

    gauges = (
        ('L_Spd', min(1.0, features.left_speed / 8.0), (80, 210, 255)),
        ('R_Spd', min(1.0, features.right_speed / 8.0), (255, 190, 80)),
        ('L_Thrust', max(0.0, min(1.0, reach_vels[0] / 5.0)), (100, 255, 170)),
        ('R_Thrust', max(0.0, min(1.0, reach_vels[1] / 5.0)), (255, 150, 120)),
        ('Rise', features.soundwave_rise, (235, 130, 255)),
        ('Squat', features.squat_depth, (80, 220, 255)),
        ('Exp', features.expansion, (90, 230, 175)),
    )
    bar_w = max(62, (w - 32) // len(gauges) - 10)
    for idx, (label, val, col) in enumerate(gauges):
        x0 = 14 + idx * (bar_w + 10)
        y0 = 52
        cv2.putText(image, f'{label} {val:.2f}', (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 215, 228), 1, cv2.LINE_AA)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + 10), (50, 56, 68), -1)
        fill = int(bar_w * max(0.0, min(1.0, val)))
        if fill > 0:
            cv2.rectangle(image, (x0, y0), (x0 + fill, y0 + 10), col, -1)

    state_flags = []
    if features.wiggle_suppressed:
        state_flags.append(('WIGGLE BLOCKED!', (60, 70, 255)))
    if features.arms_crossed:
        state_flags.append(('X-CUT', (255, 170, 80)))
    if features.body_frozen:
        state_flags.append(('FREEZE', (180, 200, 255)))
    if features.side_step_groove:
        state_flags.append(('GROOVE', (100, 240, 180)))

    now = time.monotonic()
    flash_txt, flash_until = flash
    if flash_txt and now < flash_until:
        cv2.putText(image, flash_txt, (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (90, 255, 180), 2, cv2.LINE_AA)
    elif features.active_move_label:
        cv2.putText(image, features.active_move_label, (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (120, 230, 255), 2, cv2.LINE_AA)

    fx = w - 185
    for flag_txt, col in state_flags:
        cv2.putText(image, flag_txt, (fx, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
        fx -= 130

    # Guided clip recording overlay banner (center).
    if rec_overlay is not None:
        banner, progress, bcol = rec_overlay
        cv2.rectangle(image, (w // 2 - 220, h // 2 - 55), (w // 2 + 220, h // 2 + 35), (15, 18, 28), -1)
        cv2.rectangle(image, (w // 2 - 220, h // 2 - 55), (w // 2 + 220, h // 2 + 35), bcol, 2)
        cv2.putText(image, banner, (w // 2 - 200, h // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.72, bcol, 2, cv2.LINE_AA)
        cv2.rectangle(image, (w // 2 - 200, h // 2 + 6), (w // 2 + 200, h // 2 + 20), (50, 56, 70), -1)
        pfill = int(400 * max(0.0, min(1.0, progress)))
        if pfill > 0:
            cv2.rectangle(image, (w // 2 - 200, h // 2 + 6), (w // 2 - 200 + pfill, h // 2 + 20), bcol, -1)

    # Tier 2 ML / DTW live probability bars (bottom-right).
    if ml_probs:
        items = sorted(ml_probs.items(), key=lambda kv: -kv[1])[:8]
        y_base = h - 26 - 20 * len(items) - 22
        cv2.putText(image, 'TIER 2 ML / DTW:', (w - 245, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 175, 255), 1, cv2.LINE_AA)
        for row_i, (cls_name, prob) in enumerate(items):
            y = y_base + 18 + row_i * 20
            cv2.putText(image, f'{cls_name[:12]:<12s}', (w - 245, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 222, 235), 1, cv2.LINE_AA)
            cv2.rectangle(image, (w - 145, y - 9), (w - 50, y + 1), (50, 56, 68), -1)
            fill = int(95 * max(0.0, min(1.0, prob)))
            col = (95, 255, 160) if prob >= 0.78 else (200, 145, 255)
            if fill > 0:
                cv2.rectangle(image, (w - 145, y - 9), (w - 145 + fill, y + 1), col, -1)
            cv2.putText(image, f'{int(prob * 100):2d}%', (w - 44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (215, 222, 235), 1, cv2.LINE_AA)

    cv2.putText(image, 'RECENT MOVES:', (10, h - 114), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 190, 215), 1, cv2.LINE_AA)
    for row_i, (ev_t, kind, side, inten) in enumerate(list(ticker)[-5:]):
        y = h - 94 + row_i * 15
        cv2.putText(
            image,
            f'{ev_t:5.1f}s  {kind}:{side} ({inten:.2f})',
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (225, 245, 235),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(image, '[r] Reset Counters   [q] Quit & Print Summary', (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 225), 1, cv2.LINE_AA)
    return image


def run_live_lab(args):
    import cv2
    from ..tracker import PoseTracker

    tcn_path = args.tcn_model
    if tcn_path is None:
        default_tcn = ROOT / 'models/dance_tcn.npz'
        if default_tcn.is_file():
            tcn_path = default_tcn

    detector = PoseTracker(model=args.pose_model, inference_width=args.inference_width)
    if args.gstreamer:
        print(f'Moves Lab — GStreamer: {args.gstreamer} | Modelo: {args.pose_model} @ {args.inference_width}px', flush=True)
        capture = LatestCamera(cv2, 0, fps=args.fps, pipeline=args.gstreamer)
    else:
        camera_index, camera_name = resolve_camera(args.camera)
        print(f'Moves Lab — Cámara: {camera_name} (índice {camera_index}) | Modelo: {args.pose_model} @ {args.inference_width}px', flush=True)
        capture = LatestCamera(cv2, camera_index, fps=args.fps)
    if tcn_path:
        print(f'  Tier 2 Keras TCN runtime loaded: {tcn_path}', flush=True)
    if args.dtw_dir:
        print(f'  Tier 2 DTW templates loaded:     {args.dtw_dir}', flush=True)

    worker = AsyncPoseWorker(capture, detector)
    extractor = FeatureExtractor(tcn_model_path=tcn_path, dtw_dir=args.dtw_dir)

    cv2.namedWindow('CV-DAW Moves Lab', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CV-DAW Moves Lab', 960, 720)

    started = time.monotonic()
    last_seq = 0
    pose_times: deque[float] = deque(maxlen=40)
    ticker: deque[tuple[float, str, str, float]] = deque(maxlen=20)
    counts: Counter[str] = Counter()
    flash = ('', 0.0)
    infer_ms = 0.0

    # Guided clip recorder state machine (active when `--record-move` is passed).
    rec_label = args.record_move
    rec_clips_done = 0
    rec_phase = 'countdown'  # 'countdown' -> 'recording' -> 'done'
    rec_phase_start = time.monotonic()
    rec_feats: list[tuple[float, ...]] = []
    rec_ts: list[float] = []
    saved_paths: list[Path] = []

    try:
        while not args.seconds or (time.monotonic() - started) < args.seconds:
            polled = worker.poll(last_seq)
            if polled is not None:
                t_start = time.monotonic()
                last_seq, ts, image, points = polled
                pose_times.append(ts)
                fps = (len(pose_times) - 1) / max(1e-3, pose_times[-1] - pose_times[0]) if len(pose_times) >= 2 else 0.0

                needed = (11, 12, 23, 24)
                if points and all(points[i][3] > 0.5 for i in needed) and any(points[i][3] > 0.5 for i in (15, 16)):
                    feat = extractor.update(points, ts, aspect=image.shape[0] / image.shape[1])
                else:
                    feat = extractor.missing(ts)

                infer_ms = 0.85 * infer_ms + 0.15 * ((time.monotonic() - t_start) * 1000.0)
                elapsed = time.monotonic() - started
                for mv in feat.moves:
                    key = f'{mv.kind}:{mv.side}'
                    counts[key] += 1
                    ticker.append((elapsed, mv.kind, mv.side, mv.intensity))
                    flash = (f'{mv.kind} ({mv.side.upper()}) {mv.intensity:.2f}', time.monotonic() + 0.65)

                pf = extractor.last_pose_frame
                rvels = pf.arm_reach_vel if (pf and pf.tracked) else (0.0, 0.0)

                # Update guided clip recorder if enabled.
                rec_overlay = None
                if rec_label is not None:
                    now = time.monotonic()
                    dt_phase = now - rec_phase_start
                    if rec_phase == 'countdown':
                        rem = max(0.0, args.countdown - dt_phase)
                        rec_overlay = (
                            f'GET READY [{rec_clips_done + 1}/{args.clips}] "{rec_label}" in {rem:.1f}s',
                            1.0 - rem / max(args.countdown, 0.1),
                            (70, 200, 255),
                        )
                        if dt_phase >= args.countdown:
                            rec_phase = 'recording'
                            rec_phase_start = now
                            rec_feats.clear()
                            rec_ts.clear()
                    elif rec_phase == 'recording':
                        if pf is not None and pf.tracked:
                            rec_feats.append(pf.feature_vector)
                            rec_ts.append(ts)
                        prog = min(1.0, dt_phase / max(args.clip_seconds, 0.1))
                        rec_overlay = (
                            f'REC [{rec_clips_done + 1}/{args.clips}] "{rec_label}" ({len(rec_feats)} f)',
                            prog,
                            (60, 70, 255),
                        )
                        if dt_phase >= args.clip_seconds:
                            if len(rec_feats) >= 8:
                                saved = save_pose_clip(args.dataset_dir, rec_label, rec_feats, rec_ts)
                                saved_paths.append(saved)
                                rec_clips_done += 1
                                print(f'Saved clip {rec_clips_done}/{args.clips}: {saved} ({len(rec_feats)} frames)', flush=True)
                            else:
                                print('Clip skipped (pose lost during capture); retrying...', flush=True)
                            if rec_clips_done >= args.clips:
                                print(f'Finished recording {rec_clips_done} clips for "{rec_label}"!', flush=True)
                                break
                            rec_phase = 'countdown'
                            rec_phase_start = now

                glitch_idxs = {_NAME_TO_IDX[g] for g in feat.glitch_joints if g in _NAME_TO_IDX}
                detector.draw(image, points, glitch_indices=glitch_idxs)
                display = cv2.flip(image, 1) if not args.no_mirror else image
                _draw_lab_hud(
                    cv2,
                    display,
                    feat,
                    fps=fps,
                    infer_ms=1000.0 / max(fps, 1.0),
                    model_name=args.pose_model,
                    glitches_total=extractor.sanitizer.total_glitches_blocked,
                    suppressed_total=extractor.move_recognizer.arbiter.suppressed_count,
                    ticker=ticker,
                    flash=flash,
                    reach_vels=rvels,
                    ml_probs=extractor.last_ml_probs or None,
                    rec_overlay=rec_overlay,
                )
                cv2.imshow('CV-DAW Moves Lab', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord('r'), ord('R')):
                extractor._hard_reset()
                counts.clear()
                ticker.clear()
                print('Moves Lab counters reset.', flush=True)
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        capture.close()
        detector.close()
        cv2.destroyAllWindows()

    dur = max(1e-3, time.monotonic() - started)
    print(f'\n=== MOVES LAB SESSION SUMMARY ({dur:.1f}s) ===')
    print(f'  Glitches blocked:  {extractor.sanitizer.total_glitches_blocked}')
    print(f'  Arbiter filtered:  {extractor.move_recognizer.arbiter.suppressed_count}')
    if saved_paths:
        print(f'  Clips recorded:    {len(saved_paths)} -> {args.dataset_dir / rec_label}')
    for k, v in counts.most_common():
        print(f'  {k:<26s}: {v}')


def main():
    p = argparse.ArgumentParser(description='CV-DAW Standalone Moves Lab (Vision, Recognition, Clip Recorder & ML HUD)')
    p.add_argument('--replay', type=Path, nargs='+', default=None,
                   help='Replay one or more recorded diagnostic .jsonl files offline')
    p.add_argument('--tcn-model', type=Path, default=None, help='Optional pure-NumPy TCN .npz model path')
    p.add_argument('--dtw-dir', type=Path, default=None, help='Optional DTW templates folder (datasets/moves)')
    p.add_argument('--record-move', type=str, default=None,
                   help='Guided clip recording mode for a move label (e.g. arm_wave_lr, body_roll, idle_random)')
    p.add_argument('--clips', type=int, default=10, help='Number of clips to record in --record-move mode')
    p.add_argument('--clip-seconds', type=float, default=1.2, help='Seconds per recorded clip (default: 1.2)')
    p.add_argument('--countdown', type=float, default=2.0, help='Countdown seconds before each clip (default: 2.0)')
    p.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/moves', help='Dataset output root')
    p.add_argument('--camera', type=int, default=None, help='Camera index override')
    p.add_argument('--gstreamer', type=str, default=None, help='Optional GStreamer camera pipeline (STM32 CSI/V4L2)')
    p.add_argument('--pose-model', choices=('lite', 'full'), default='lite', help='MediaPipe Pose model (default: lite)')
    p.add_argument('--inference-width', type=int, default=320, help='Pose inference resize width in px (default: 320)')
    p.add_argument('--fps', type=int, default=60, help='Requested camera FPS (default: 60)')
    p.add_argument('--seconds', type=float, default=0.0, help='Optional timed run (0 = until q)')
    p.add_argument('--no-mirror', action='store_true', help='Disable horizontal mirror display')
    args = p.parse_args()

    if args.replay:
        for path in args.replay:
            res = replay_jsonl(path, tcn_model_path=args.tcn_model, dtw_dir=args.dtw_dir)
            _print_replay_report(res)
        return

    run_live_lab(args)


if __name__ == '__main__':
    main()
