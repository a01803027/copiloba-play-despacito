"""Mac and OpenSTLinux entry point for the four-limb instrument."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from composer.app import ROOT, start_engine
from composer.ipc import Bridge
from feature_extractors import FastKinematicsExtractor, SlowTemporalExtractor, TemporalFeatures
from musical_brain import EventType, MusicalBrain, SCALES
from perception_layer import CameraStream, PerceptionFrame, SkeletonTracker, draw_frame


def synthetic_frame(t):
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1
    pose[[11, 12], :2] = ((-.25, -1), (.25, -1))
    pose[[23, 24], 0] = [-.2, .2]
    for i, (joint, tip) in enumerate(((13, 15), (14, 16), (25, 27), (26, 28))):
        pose[joint, 0] = pose[tip, 0] = (-1 if i % 2 == 0 else 1) * .5
        center = -.5 if i < 2 else 1.
        pose[tip, 1] = center + .36 * math.sin(2 * math.pi * (1.4 + i * .2) * t + i)
        pose[joint, 1] = center
    return PerceptionFrame(t, pose, None, None, pose[:, :3], .5 + .02 * math.sin(4 * math.pi * t))


def stop_engine(process):
    if process.poll() is None:
        process.terminate()
    output, _ = process.communicate(timeout=5)
    if output:
        print(output.strip())


def run(args):
    fast = FastKinematicsExtractor()
    slow = SlowTemporalExtractor(args.groove_model, args.delegate)
    brain = MusicalBrain(scale=args.scale, bpm=args.bpm, follow_bounce=args.follow_bounce,
                         stationary_leg_modifier=args.stationary_leg_modifier)
    with ExitStack() as stack:
        cv2 = capture = tracker = None
        if args.mode != 'mock':
            import cv2
            if not args.no_window:
                stack.callback(cv2.destroyAllWindows)
            tracker = SkeletonTracker(args.pose_backend, model=args.pose_model, delegate=args.pose_delegate)
            stack.callback(tracker.close)
        if args.mode == 'camera':
            capture = CameraStream(args.camera, device=args.device, exposure=args.exposure)
            stack.callback(capture.close)
            print(f'Cámara: {capture.name}, FPS informados: {capture.capture.negotiated_fps:g}')
        elif args.mode == 'video':
            capture = cv2.VideoCapture(str(args.video))
            stack.callback(capture.release)
            if not capture.isOpened():
                raise RuntimeError('No se pudo abrir el vídeo')
        rate = (capture.get(cv2.CAP_PROP_FPS) or 30) if args.mode == 'video' else 60
        process = start_engine(args)
        stack.callback(stop_engine, process)
        bridge = Bridge(args.library)
        stack.callback(bridge.close)
        worker = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        pending = None
        temporal = TemporalFeatures()
        started = time.monotonic()
        latest, counts, processed, tracked = None, [0] * 4, 0, 0
        while not args.seconds or time.monotonic() - started < args.seconds:
            if process.poll() is not None:
                raise RuntimeError('El motor se detuvo durante la ejecución')
            if args.mode == 'mock':
                frame = synthetic_frame(processed / rate)
            elif args.mode == 'video':
                ok, image = capture.read()
                if not ok:
                    break
                frame = tracker.process(image, processed / rate)
            else:
                timestamp, image = capture.read()
                frame = tracker.process(image, timestamp)
            limbs = fast.update(frame)
            if pending is not None and pending.done():
                temporal = pending.result()
                pending = None
            if pending is None:
                pending = worker.submit(slow.update, frame)
            sample = bridge.metric(0)
            controls, events = brain.update(limbs, temporal, sample, bridge.metric(13))
            bridge.controls(asdict(controls))
            for event in events:
                bridge.send(event)
                if event.kind == EventType.NOTE_ON:
                    counts[event.limb] += 1
                    latest = event.limb
            processed += 1
            tracked += frame.normalized_pose is not None
            if cv2 is not None and not args.no_window:
                draw_frame(image, frame, processed / max(time.monotonic() - started, .001))
                states = ' / '.join(state.name for state in brain.states.play_state)
                cv2.putText(image, f'{brain.bpm:.0f} BPM | {states} | golpe {latest}',
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 0), 1)
                cv2.imshow('CV-DAW Instrumento', image)
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
            if args.mode != 'camera':
                delay = started + processed / rate - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        report = {
            'processed_frames': processed,
            'tracked_frames': tracked,
            'camera_frames': capture.capture.count if args.mode == 'camera' else None,
            'processed_fps': processed / max(time.monotonic() - started, .001),
            'hits': counts,
            'drops': bridge.metric(2),
            'invalid': bridge.metric(3),
            'late': bridge.metric(4),
            'scheduler_overflows': bridge.metric(5),
            'stem_starved_frames': bridge.metric(7),
        }
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        if any(report[key] for key in ('drops', 'invalid', 'scheduler_overflows')):
            raise RuntimeError('El motor perdió o rechazó comandos')


def main():
    parser = argparse.ArgumentParser(description='Instrumento y remezcla CV-DAW')
    parser.add_argument('--mode', choices=('camera', 'video', 'mock'), default='camera')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--camera', type=int)
    parser.add_argument('--device')
    parser.add_argument('--exposure', type=int)
    pose = parser.add_mutually_exclusive_group()
    pose.add_argument('--pose-backend')
    pose.add_argument('--pose-model', type=Path)
    parser.add_argument('--pose-delegate')
    parser.add_argument('--groove-model', type=Path)
    parser.add_argument('--delegate')
    parser.add_argument('--stationary-leg-modifier', action='store_true')
    parser.add_argument('--follow-bounce', action='store_true')
    parser.add_argument('--scale', choices=SCALES, default='minor-pentatonic')
    parser.add_argument('--bpm', type=float, default=120.)
    parser.add_argument('--stems', nargs=4, metavar=('DRUMS', 'BASS', 'VOCALS', 'OTHER'))
    parser.add_argument('--samples', nargs=4, metavar=('KICK', 'SNARE', 'HAT', 'STAB'))
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--no-window', action='store_true')
    parser.add_argument('--offline-audio', action='store_true')
    parser.add_argument('--record')
    parser.add_argument('--metrics', type=Path)
    build = ROOT / 'realtime' / ('build-mac' if sys.platform == 'darwin' else 'build')
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--engine', default=build / 'cv-daw-engine')
    parser.add_argument('--library', default=build / f'libcv_daw_ipc.{suffix}')
    args = parser.parse_args()
    if not (math.isfinite(args.seconds) and args.seconds >= 0 and 60 <= args.bpm <= 200):
        parser.error('Duración no negativa y BPM entre 60 y 200 requeridos')
    if args.mode == 'video' and not args.video:
        parser.error('--mode video requiere --video')
    if args.record and not args.offline_audio:
        parser.error('--record requiere --offline-audio')
    if args.delegate and not args.groove_model:
        parser.error('--delegate requiere --groove-model')
    if args.pose_delegate and not args.pose_model:
        parser.error('--pose-delegate requiere --pose-model')
    args.performance = True
    try:
        run(args)
    except KeyboardInterrupt:
        print('Instrumento detenido')


if __name__ == '__main__':
    main()
