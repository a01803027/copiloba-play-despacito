"""python -m composer.app: Pose-Only ~30 FPS Full-Body Move Recognition -> Lock-Free C++20 DSP."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import selectors
import subprocess
import sys
import time
from .brain import Composer
from .cameras import resolve_camera
from .diagnose import write_report
from .features import FeatureExtractor, mock_features
from .ipc import Bridge
from .moves import CANONICAL_JOINTS
from .recorder import SessionRecorder
from .styles import STYLE_ORDER
from .ui import draw_hud, handle_window_key
from .vision import AsyncPoseWorker, LatestCamera, create_pose_backend

ROOT = Path(__file__).resolve().parents[1]


def default_build_dir():
    preferred = ROOT / ('realtime/build-mac' if sys.platform == 'darwin' else 'realtime/build')
    if preferred.is_dir():
        return preferred
    fallback = ROOT / 'realtime/build'
    return fallback if fallback.is_dir() else preferred


def start_engine(args):
    command = [str(Path(args.engine).resolve()), '--profile']
    if args.offline_audio:
        command += ['--offline', str(math.ceil((args.seconds + 5 if args.seconds else 3600) * 375))]
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            command += ['--render', args.record]
    elif args.record:
        raise ValueError('--record requiere --offline-audio: usa --diagnose para grabar audio en vivo')
    if getattr(args, 'live_record', None) and not args.offline_audio:
        Path(args.live_record).parent.mkdir(parents=True, exist_ok=True)
        command += ['--live-record', str(args.live_record)]
    if getattr(args, 'alsa_device', None):
        command += ['--alsa-device', str(args.alsa_device)]
    if getattr(args, 'period_frames', None):
        command += ['--period-frames', str(args.period_frames)]
    if getattr(args, 'alsa_periods', None):
        command += ['--alsa-periods', str(args.alsa_periods)]
    if getattr(args, 'strict_alsa', False):
        command += ['--strict-alsa']
    if getattr(args, 'strict_rt', False):
        command += ['--strict-rt']
    if args.seconds and not args.offline_audio:
        command += ['--seconds', str(args.seconds + 10)]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, text=True, bufsize=1)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            if selector.select(0.1):
                line = process.stdout.readline().strip()
                if line:
                    print(line, flush=True)
                if line.startswith(('audio-ready', 'offline-ready')):
                    return process
            if process.poll() is not None:
                raise RuntimeError('El motor terminó antes de estar listo; revisa su mensaje anterior.')
        raise RuntimeError('Timeout al iniciar el motor')
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        raise
    finally:
        selector.close()


def _clear_one_shots(features):
    if features.moves or features.shoulder_drop:
        data = features.to_dict()
        data['moves'] = ()
        data['shoulder_drop'] = False
        return features.__class__(**data)
    return features


def _build_parser():
    p = argparse.ArgumentParser(description='CV-DAW — Instrumento musical interactivo por movimientos y danza')
    p.add_argument('--mode', choices=('mock', 'camera', 'video'), default='camera')
    p.add_argument('--genre', '--style', dest='genre', choices=STYLE_ORDER, default='house',
                   help='Estilo inicial (1-6 o N/P en vivo)')
    p.add_argument('--video', type=Path)
    p.add_argument('--camera', type=int, default=None, help='Índice de cámara (por defecto webcam integrada)')
    p.add_argument('--gstreamer', type=str, default=None, help='Pipeline GStreamer opcional (e.g. STM32MP257F-DK CSI/V4L2)')
    p.add_argument('--pose-backend', choices=('auto', 'mediapipe', 'stai-npu', 'movenet-tflite'), default='auto',
                   help='Motor de pose: auto (MediaPipe en Mac, STAI NPU en STM32MP257F-DK)')
    p.add_argument('--npu-model', type=Path, default=None,
                   help='Ruta al modelo MoveNet INT8 (.nb/.tflite) para STM32 NPU o TFLite')
    p.add_argument('--pose-model', choices=('lite', 'full'), default='lite',
                   help='Modelo MediaPipe Pose (.task) — lite (30-60 FPS) por defecto')
    p.add_argument('--inference-width', type=int, default=320,
                   help='Ancho de entrada para inferencia Pose (320px = máxima velocidad, 0 = nativo)')
    p.add_argument('--tcn-model', type=Path, default=None,
                   help='Modelo Tier 2 pure-NumPy TCN (.npz) opcional (auto-carga models/dance_tcn.npz si existe)')
    p.add_argument('--dtw-dir', type=Path, default=None,
                   help='Carpeta de plantillas Tier 2 DTW opcional (e.g. datasets/moves)')
    p.add_argument('--alsa-device', type=str, default=None, help='Dispositivo ALSA Linux/STM32 (e.g. default, plughw:0,0)')
    p.add_argument('--period-frames', type=int, default=None, help='Frames por callback ALSA (128 o 256)')
    p.add_argument('--alsa-periods', type=int, default=None, help='Periodos ALSA (2..8)')
    p.add_argument('--strict-alsa', action='store_true', help='Modo ALSA exclusivo estricto (calificación RT)')
    p.add_argument('--strict-rt', action='store_true', help='Falla si mlockall/SCHED_FIFO no tienen permisos')
    p.add_argument('--tempo-lock', action='store_true', help='Bloquea el BPM inicial')
    p.add_argument('--no-mirror', action='store_true', help='Desactiva espejo en la ventana de cámara')
    p.add_argument('--seconds', type=float, default=0, help='0 = hasta q o Control+C')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--launch-engine', action='store_true')
    build_dir = default_build_dir()
    p.add_argument('--engine', default=str(build_dir / 'cv-daw-engine'))
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    p.add_argument('--library', default=str(build_dir / f'libcv_daw_ipc.{suffix}'))
    p.add_argument('--offline-audio', action='store_true', help='Prueba sin dispositivo de audio')
    p.add_argument('--record', help='WAV offline con IPC y compositor Python')
    p.add_argument('--diagnose', nargs='?', const=15.0, type=float, default=None,
                   help='Graba sesión de diagnóstico en vivo (WAV + JSONL + reporte .md)')
    p.add_argument('--diagnose-dir', type=Path, default=ROOT / 'artifacts')
    p.add_argument('--no-window', action='store_true')
    p.add_argument('--metrics', type=Path)
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.live_record = None
    diag_jsonl = diag_wav = None

    if args.diagnose is not None:
        if not math.isfinite(args.diagnose) or args.diagnose <= 0:
            parser.error('--diagnose requiere una duración positiva en segundos')
        if not args.launch_engine:
            parser.error('--diagnose requiere --launch-engine para capturar el audio en vivo')
        stamp = time.strftime('%Y%m%d_%H%M%S')
        args.diagnose_dir.mkdir(parents=True, exist_ok=True)
        diag_jsonl = args.diagnose_dir / f'diag_{args.genre}_{stamp}.jsonl'
        diag_wav = args.diagnose_dir / f'diag_{args.genre}_{stamp}.wav'
        if args.offline_audio:
            args.record = str(diag_wav)
        else:
            args.live_record = str(diag_wav)
        if not args.seconds:
            args.seconds = args.diagnose

    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds debe ser un número no negativo')
    if args.mode == 'video' and not args.video:
        parser.error('--mode video requiere --video archivo.mp4')
    if (args.record or args.offline_audio) and not args.launch_engine:
        parser.error('--record y --offline-audio requieren --launch-engine')

    process = bridge = capture = detector = pose_worker = recorder = cv2 = None
    started = time.monotonic()
    frames = tracked_frames = notes = 0
    max_loop = 0.0
    energies: list[float] = []
    probes: list[float] = []
    tcn_path = args.tcn_model
    if tcn_path is None:
        default_tcn = ROOT / 'models/dance_tcn.npz'
        if default_tcn.is_file():
            tcn_path = default_tcn
    composer = Composer(args.seed, genre=args.genre, tempo_lock=args.tempo_lock)
    extractor = FeatureExtractor(tcn_model_path=tcn_path, dtw_dir=args.dtw_dir)
    flash_state = {'move': ('', 0.0)}
    report: dict = {}

    try:
        if diag_jsonl is not None:
            recorder = SessionRecorder(
                diag_jsonl, diag_wav,
                {'mode': args.mode, 'initial_style': args.genre, 'pose_model': args.pose_model},
            )
            print(f'Grabando sesión de diagnóstico ({args.seconds:.0f}s) -> {diag_wav} + {diag_jsonl}', flush=True)

        if args.mode != 'mock':
            import cv2

            detector = create_pose_backend(
                args.pose_backend,
                pose_model=args.pose_model,
                inference_width=args.inference_width,
                npu_model=args.npu_model,
            )
            if args.mode == 'camera':
                if args.gstreamer:
                    print(f'Cámara GStreamer: {args.gstreamer}', flush=True)
                    capture = LatestCamera(cv2, 0, pipeline=args.gstreamer)
                else:
                    camera_index, camera_name = resolve_camera(args.camera)
                    print(f'Cámara: {camera_name} — índice OpenCV {camera_index}', flush=True)
                    capture = LatestCamera(cv2, camera_index)
                pose_worker = AsyncPoseWorker(capture, detector)
            else:
                capture = cv2.VideoCapture(str(args.video))
                if not capture.isOpened():
                    raise RuntimeError('No se puede leer el vídeo')
            video_fps = capture.get(cv2.CAP_PROP_FPS) if args.mode == 'video' else 60
            if not video_fps or video_fps < 1:
                video_fps = 30

        if args.launch_engine:
            process = start_engine(args)
        bridge = Bridge(args.library)
        probes.append(bridge.probe())
        initial_consumed = bridge.metric(1)
        started = time.monotonic()

        if cv2 and not args.no_window:
            cv2.namedWindow('CV-DAW Studio', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CV-DAW Studio', 960, 720)
        print(
            f'{composer.preset} ({composer.bpm:.0f} BPM) — 11 movimientos corporales y de piernas activos '
            f'(Punch, Brazos Arriba=Soundwave, X=Cut, ArmWave, Rodilla, Sentadilla, Salto). 1-6 estilo, q salir.',
            flush=True,
        )

        last_pose_seq = 0
        last_features = extractor.missing()
        last_points = None

        while not args.seconds or time.monotonic() - started < args.seconds:
            begin = time.monotonic()
            if process and process.poll() is not None:
                raise RuntimeError('El motor se detuvo durante la ejecución')
            elapsed = begin - started
            rec_label = f'REC {elapsed:04.1f}/{args.seconds:.0f}s' if recorder and args.seconds else ('REC' if recorder else None)
            is_new_pose = False

            if args.mode == 'mock':
                features = mock_features(elapsed)
                is_new_pose = True
            elif args.mode == 'camera':
                polled = pose_worker.poll(last_pose_seq)
                if polled is not None:
                    last_pose_seq, timestamp, image, points = polled
                    last_points = points
                    is_new_pose = True
                    needed = (11, 12, 23, 24)
                    if points and all(points[i][3] > 0.5 for i in needed) and any(points[i][3] > 0.5 for i in (15, 16)):
                        features = extractor.update(
                            points, timestamp, image.shape[0] / image.shape[1],
                            bpm_range=composer.bpm_sync_range(),
                        )
                    else:
                        features = extractor.missing(timestamp)
                    last_features = features
                    if features.moves and features.active_move_label:
                        flash_state['move'] = (features.active_move_label, time.monotonic() + 0.70)
                    if not args.no_window:
                        name_to_idx = {name: idx for idx, name in CANONICAL_JOINTS}
                        glitch_idxs = {name_to_idx[g] for g in features.glitch_joints if g in name_to_idx}
                        detector.draw(image, points, glitch_indices=glitch_idxs)
                        display = cv2.flip(image, 1) if not args.no_mirror else image
                        draw_hud(cv2, display, features, composer, flash_state, rec_label)
                        cv2.imshow('CV-DAW Studio', display)
                else:
                    features = _clear_one_shots(last_features)
                if not args.no_window:
                    if handle_window_key(cv2.waitKey(1) & 0xFF, composer):
                        break
            else:
                ok, image = capture.read()
                if not ok:
                    break
                timestamp = time.monotonic()
                points = detector.process(image, timestamp)
                last_points = points
                is_new_pose = True
                needed = (11, 12, 23, 24)
                if points and all(points[i][3] > 0.5 for i in needed) and any(points[i][3] > 0.5 for i in (15, 16)):
                    features = extractor.update(
                        points, timestamp, image.shape[0] / image.shape[1],
                        bpm_range=composer.bpm_sync_range(),
                    )
                else:
                    features = extractor.missing(timestamp)
                if features.moves and features.active_move_label:
                    flash_state['move'] = (features.active_move_label, time.monotonic() + 0.70)
                if not args.no_window:
                    detector.draw(image, points)
                    draw_hud(cv2, image, features, composer, flash_state, rec_label)
                    cv2.imshow('CV-DAW Studio', image)
                    if handle_window_key(cv2.waitKey(1) & 0xFF, composer):
                        break

            frame = bridge.metric(0)
            events = composer.update(features, frame, elapsed)
            ctrl = composer.controls(features)
            bridge.controls(ctrl)
            for event in events:
                bridge.send(event)
                notes += event.kind == 0
            if recorder is not None:
                recorder.record_frame(
                    elapsed_s=elapsed, audio_frame=frame, is_new_pose=is_new_pose,
                    points=last_points, features=features, composer=composer, controls=ctrl, events=events,
                )
            tracked_frames += features.tracked
            energies.append(features.energy)
            frames += 1
            max_loop = max(max_loop, (time.monotonic() - begin) * 1000)
            rate = video_fps if args.mode == 'video' else (60 if args.mode == 'camera' else 30)
            delay = 1 / rate - (time.monotonic() - begin)
            if delay > 0:
                time.sleep(delay)

        probes.append(bridge.probe())
        deadline = time.monotonic() + 1
        while bridge.metric(1) - initial_consumed < bridge.sent - 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        duration = time.monotonic() - started
        frequency = bridge.metric(12)
        calls = bridge.metric(9)
        report = {
            'mode': args.mode, 'seconds': duration, 'processed_frames': frames,
            'processed_fps': frames / max(duration, 0.001), 'tracked_frames': tracked_frames,
            'camera_frames': capture.count if isinstance(capture, LatestCamera) else None,
            'max_python_processing_ms': max_loop, 'ipc_roundtrip_ms': probes,
            'events_sent': bridge.sent, 'note_ons': notes,
            'energy_min': min(energies, default=0), 'energy_max': max(energies, default=0),
            'command_drops': bridge.metric(2), 'invalid_commands': bridge.metric(3),
            'late_commands': bridge.metric(4), 'scheduler_overflows': bridge.metric(5),
            'composer_skipped_ticks': composer.skipped_ticks,
            'callback_count': calls,
            'callback_mean_us': bridge.metric(10) / frequency / calls * 1e6 if frequency and calls else None,
            'callback_max_us': bridge.metric(11) / frequency * 1e6 if frequency and calls else None,
            'physical_camera_verified': args.mode == 'camera' and frames > 0,
            'physical_sound_verified': False,
        }
        if report['command_drops'] or report['invalid_commands'] or report['scheduler_overflows']:
            raise RuntimeError('Se detectaron pérdidas/rechazos IPC; consulta métricas')
    except KeyboardInterrupt:
        print('Deteniendo compositor...')
    finally:
        if recorder is not None:
            recorder.record_footer(report)
            recorder.close()
        if bridge:
            bridge.close()
        if process:
            if process.poll() is None:
                process.terminate()
            output, _ = process.communicate(timeout=5)
            if output:
                print(output.strip())
        if pose_worker:
            pose_worker.close()
        if isinstance(capture, LatestCamera):
            capture.close()
        elif capture:
            capture.release()
        if detector:
            detector.close()
        if cv2:
            cv2.destroyAllWindows()

    if diag_jsonl is not None and diag_jsonl.is_file():
        diag, report_md = write_report(diag_jsonl, diag_wav if (diag_wav and diag_wav.is_file()) else None)
        print(f'\n=== DIAGNÓSTICO GUARDADO ===\nWAV:     {diag_wav}\nJSONL:   {diag_jsonl}\nREPORTE: {report_md}', flush=True)
        for item in diag['findings']:
            print(f'  * {item}', flush=True)
    if report:
        print(json.dumps(report, indent=2))
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
