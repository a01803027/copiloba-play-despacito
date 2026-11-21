"""python -m composer.app: camera/video/mock -> rules -> native IPC -> C++ audio."""
import argparse
import json
import math
from pathlib import Path
import queue
import selectors
import subprocess
import sys
import threading
import time
from .brain import Composer
from .diagnose import write_report
from .features import FeatureExtractor, mock_features
from .recorder import SessionRecorder
from .styles import STYLE_ORDER
from .ipc import Bridge
from .cameras import resolve_camera

ROOT=Path(__file__).resolve().parents[1]


def default_build_dir():
    preferred = ROOT / ('realtime/build-mac' if sys.platform == 'darwin' else 'realtime/build')
    if preferred.is_dir():
        return preferred
    fallback = ROOT / 'realtime/build'
    return fallback if fallback.is_dir() else preferred


def handle_window_key(key, composer):
    """Handle live keyboard shortcuts for styles, BPM lock, and exit. Returns True to quit."""
    if key == ord('q'):
        return True
    direct_keys = (
        ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'),
        ord('7'), ord('8'), ord('9'), ord('0'), ord('-'), ord('='),
    )
    if key in direct_keys:
        idx = direct_keys.index(key)
        if idx < len(STYLE_ORDER):
            composer.set_genre(STYLE_ORDER[idx])
            print(f'Estilo [{idx + 1}/{len(STYLE_ORDER)}]: {composer.preset} ({composer.bpm:.0f} BPM)', flush=True)
    elif key in (ord('n'), ord('N'), ord(']')):
        composer.cycle_genre(1)
        idx = STYLE_ORDER.index(composer.genre_key) + 1 if composer.genre_key in STYLE_ORDER else 1
        print(f'Estilo [{idx}/{len(STYLE_ORDER)}]: {composer.preset} ({composer.bpm:.0f} BPM)', flush=True)
    elif key in (ord('p'), ord('P'), ord('[')):
        composer.cycle_genre(-1)
        idx = STYLE_ORDER.index(composer.genre_key) + 1 if composer.genre_key in STYLE_ORDER else 1
        print(f'Estilo [{idx}/{len(STYLE_ORDER)}]: {composer.preset} ({composer.bpm:.0f} BPM)', flush=True)
    elif key in (ord('t'), ord('T')):
        locked = composer.toggle_tempo_lock()
        print(f'BPM {"bloqueado" if locked else "sincronizado con el cuerpo"} ({composer.bpm:.0f} BPM)', flush=True)
    return False


def draw_hud(cv2, image, features, composer, flash_state=None, rec_status=None):
    """Draw a clean, translucent full-body musical HUD onto the display frame."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 134), (16, 18, 24), -1)
    cv2.rectangle(overlay, (0, h - 30), (w, h), (16, 18, 24), -1)
    cv2.addWeighted(overlay, 0.76, image, 0.24, 0, image)

    if not features.tracked:
        status_text, status_color = 'BUSCANDO POSE', (70, 90, 240)
    elif features.grace:
        status_text, status_color = 'GRACE', (40, 195, 255)
    else:
        status_text, status_color = 'TRACKING', (90, 230, 120)

    cv2.circle(image, (20, 22), 7, status_color, -1, cv2.LINE_AA)
    bpm_tag = 'LOCK' if getattr(composer, 'tempo_lock', False) else (
        f'SYNC:{getattr(composer, "bpm_sync_source", "").upper()}'
        if getattr(composer, 'bpm_synced', False)
        else 'AUTO'
    )
    style_idx = (STYLE_ORDER.index(composer.genre_key) + 1) if getattr(composer, 'genre_key', '') in STYLE_ORDER else 1
    section = getattr(composer, 'section', 'INTRO')
    bar_num = getattr(composer, 'bar_in_cycle', 0) + 1
    chord_name = getattr(composer, 'current_chord_name', '')
    title = f'[{style_idx}/{len(STYLE_ORDER)}] {getattr(composer, "preset", "CV-DAW")} | {composer.bpm:.0f} BPM ({bpm_tag}) | {chord_name} | {section} [{bar_num}/4]'
    cv2.putText(image, title, (34, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 244, 248), 2, cv2.LINE_AA)
    if rec_status:
        cv2.circle(image, (w - 255, 22), 6, (40, 40, 250), -1, cv2.LINE_AA)
        cv2.putText(image, rec_status, (w - 242, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 90, 255), 2, cv2.LINE_AA)
    cv2.putText(image, status_text, (w - 135, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1, cv2.LINE_AA)

    footer = 'Estilos: [1] Ambient [2] House [3] Lo-Fi [4] Latin [5] Synthwave [6] DnB  |  [N/P]  [T] Lock BPM  [q] Salir'
    cv2.putText(image, footer, (16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 222, 232), 1, cv2.LINE_AA)

    if not features.tracked:
        cv2.putText(
            image,
            'Necesito ver hombros, caderas y al menos una mano (alejate para incluir todo el cuerpo)',
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 190, 210),
            1,
            cv2.LINE_AA,
        )
        return image

    legs_label = f'Piernas ({features.legs_source[:3].upper()})' if features.legs_visible else 'Piernas (NO)'
    meters = (
        ('Energia', features.energy, (60, 210, 255)),
        ('Brazo Izq', features.left_height, (130, 220, 255)),
        ('Brazo Der', features.right_height, (220, 130, 245)),
        ('Hombros', max(features.shoulder_lift, features.shoulder_bounce), (70, 220, 150)),
        (legs_label, features.leg_energy, (255, 185, 70) if features.legs_visible else (110, 110, 120)),
    )
    bar_w = max(70, (w - 52) // len(meters) - 16)
    for idx, (label, val, color) in enumerate(meters):
        x0 = 16 + idx * (bar_w + 16)
        y0 = 56
        cv2.putText(image, f'{label} {val:.2f}', (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 216, 224), 1, cv2.LINE_AA)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + 10), (55, 60, 72), -1)
        fill = int(bar_w * max(0.0, min(1.0, val)))
        if fill > 0:
            cv2.rectangle(image, (x0, y0), (x0 + fill, y0 + 10), color, -1)

    body_line = (
        f'Cabeza tilt:{features.head_tilt:+.2f}  |  '
        f'Hombros shrug:{features.shoulder_lift:.2f} bounce:{features.shoulder_bounce:.2f}  |  '
        f'Cadera sway:{features.hip_sway:+.2f}  |  '
        f'Piernas:{"OK(" + features.legs_source + ") stance:" + f"{features.stance_width:.2f}" if features.legs_visible else "aleja camara para ver rodillas/pies"}'
    )
    cv2.putText(image, body_line, (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 205, 225), 1, cv2.LINE_AA)

    # Live expressive badges so the user immediately sees Head, Shoulder, Arm & Leg triggers.
    badges = []
    if section == 'BUILDUP':
        badges.append(('SHRUG BUILD-UP!', (40, 195, 255)))
    elif section == 'DROP':
        badges.append(('DROP!', (220, 90, 255)))
    if getattr(composer, 'head_harmony_active', False):
        badges.append(('HEAD HARMONY + ECHO', (235, 145, 255)))
    if getattr(composer, 'shoulder_groove_active', False):
        badges.append(('SHOULDER GROOVE', (100, 240, 180)))
    if flash_state:
        now = time.monotonic()
        for key_name, color in (
            ('nod', (100, 255, 130)),
            ('left', (80, 235, 255)),
            ('right', (255, 165, 235)),
            ('l_leg', (255, 190, 80)),
            ('r_leg', (90, 220, 255)),
        ):
            txt, until = flash_state.get(key_name, ('', 0.0))
            if now < until and txt:
                badges.append((txt, color))

    bx = 16
    for text, color in badges:
        cv2.putText(image, text, (bx, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA)
        bx += max(110, len(text) * 10 + 16)
    return image


class AsyncPoseWorker:
    """Runs MediaPipe pose inference on a dedicated thread so the musical clock never stalls."""

    def __init__(self, capture, detector):
        self._capture = capture
        self._detector = detector
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                timestamp, image = self._capture.get()
                points = self._detector.process(image, timestamp)
            except Exception as exc:
                if not self._stop.is_set():
                    with self._lock:
                        self._error = exc
                break
            with self._lock:
                self._seq += 1
                self._latest = (self._seq, timestamp, image, points)

    def poll(self, last_seq: int):
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._latest is None or self._latest[0] == last_seq:
                return None
            return self._latest

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)


def start_engine(args):
    command=[str(Path(args.engine).resolve()),'--profile']
    if getattr(args, 'performance', False):
        command += ['--performance', str(args.bpm)]
        if args.stems and args.follow_bounce:
            command += ['--stretch-stems']
        if args.stems:
            command += ['--stems', str(args.bpm), *(str(Path(stem).resolve()) for stem in args.stems)]
        if args.samples:
            command += ['--samples', *(str(Path(sample).resolve()) for sample in args.samples)]
    if args.offline_audio:
        command+=['--offline',str(math.ceil((args.seconds+5 if args.seconds else 3600)*375))]
        if args.record:
            Path(args.record).parent.mkdir(parents=True,exist_ok=True)
            command+=['--render',args.record]
    elif args.record:
        raise ValueError('--record requiere --offline-audio: usa --diagnose para grabar audio en vivo')
    if getattr(args, 'live_record', None) and not args.offline_audio:
        Path(args.live_record).parent.mkdir(parents=True, exist_ok=True)
        command += ['--live-record', str(args.live_record)]
    if args.seconds and not args.offline_audio:
        command+=['--seconds',str(args.seconds+10)]
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,text=True,bufsize=1)
    selector=selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ)
    deadline=time.monotonic()+15
    try:
        while time.monotonic()<deadline:
            if selector.select(.1):
                line=process.stdout.readline().strip()
                if line:
                    print(line,flush=True)
                if line.startswith(('audio-ready','offline-ready')):
                    return process
            if process.poll() is not None:
                raise RuntimeError('El motor terminó antes de estar listo; revisa su mensaje anterior.')
        raise RuntimeError('Timeout al iniciar el motor')
    except BaseException:
        process.terminate();process.wait(timeout=5)
        raise
    finally:
        selector.close()


class LatestCamera:
    def __init__(self,cv2,index, *, fps=30, pipeline=None):
        self.cv2=cv2
        if pipeline:
            self.cap=cv2.VideoCapture(pipeline,cv2.CAP_GSTREAMER)
        elif sys.platform=='darwin':
            self.cap=cv2.VideoCapture(index,cv2.CAP_AVFOUNDATION)
        else:
            self.cap=cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError('No se pudo abrir la cámara. Revisa permisos de Cámara para Terminal y cierra otras apps.')
        if not pipeline:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            self.cap.set(cv2.CAP_PROP_FPS,fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.negotiated_fps=self.cap.get(cv2.CAP_PROP_FPS)
        self.frames=queue.Queue(maxsize=1)
        self.stop=threading.Event()
        self.count=0
        self.worker=threading.Thread(target=self.read,daemon=True)
        self.worker.start()

    def read(self):
        while not self.stop.is_set():
            ok,image=self.cap.read()
            if not ok:
                break
            self.count+=1
            item=(time.monotonic(),image)
            try:
                self.frames.put_nowait(item)
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait(item)
                except queue.Full:
                    pass

    def get(self):
        try:
            return self.frames.get(timeout=2)
        except queue.Empty as exc:
            raise RuntimeError('La cámara no está entregando imágenes') from exc

    def close(self):
        self.stop.set()
        self.worker.join(timeout=3)
        if self.worker.is_alive():
            raise RuntimeError('La captura no se detuvo a tiempo; cierra la aplicación antes de reiniciar la cámara.')
        self.cap.release()


def main():
    p=argparse.ArgumentParser(description='Compositor Contemporary Ambient controlado por danza')
    p.add_argument('--mode',choices=('mock','camera','video'),default='camera')
    p.add_argument('--genre','--style',dest='genre',choices=STYLE_ORDER,default='ambient',help='Estilo musical inicial (12 estilos JSON; N/P o 1-9,0,-,= en vivo)')
    p.add_argument('--video',type=Path)
    p.add_argument('--camera',type=int,default=None,help='Índice manual; por defecto se prefiere la webcam integrada del Mac sobre el iPhone')
    p.add_argument('--pose-model',choices=('full','lite'),default='full',help='Modelo MediaPipe Tasks (.task en models/)')
    p.add_argument('--tempo-lock',action='store_true',help='Bloquea el BPM inicial en lugar de sincronizarlo al ritmo corporal')
    p.add_argument('--no-mirror',action='store_true',help='Desactiva el efecto espejo en la ventana de cámara')
    p.add_argument('--seconds',type=float,default=0,help='0 = hasta q o Control+C')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--launch-engine',action='store_true')
    build_dir=default_build_dir()
    p.add_argument('--engine',default=str(build_dir/'cv-daw-engine'))
    suffix='dylib' if sys.platform=='darwin' else 'so'
    p.add_argument('--library',default=str(build_dir/f'libcv_daw_ipc.{suffix}'))
    p.add_argument('--offline-audio',action='store_true',help='Prueba sin dispositivo de audio')
    p.add_argument('--record',help='WAV de integración offline, con IPC y compositor Python')
    p.add_argument('--diagnose',nargs='?',const=15.0,type=float,default=None,help='Graba clip de diagnóstico en vivo (WAV + traza JSONL + reporte .md); por defecto 15 s')
    p.add_argument('--diagnose-dir',type=Path,default=ROOT/'artifacts',help='Carpeta de salida para los archivos de --diagnose')
    p.add_argument('--no-window',action='store_true')
    p.add_argument('--metrics',type=Path)
    args=p.parse_args()
    args.live_record = None
    diag_jsonl = None
    diag_wav = None
    if args.diagnose is not None:
        if not math.isfinite(args.diagnose) or args.diagnose <= 0:
            p.error('--diagnose requiere una duración positiva en segundos (p. ej. --diagnose 15)')
        if not args.launch_engine:
            p.error('--diagnose requiere --launch-engine para capturar el audio en vivo')
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
    if not math.isfinite(args.seconds) or args.seconds<0:
        p.error('--seconds debe ser un número no negativo')
    if args.mode=='video' and not args.video:
        p.error('--mode video requiere --video archivo.mp4')
    if (args.record or args.offline_audio) and not args.launch_engine:
        p.error('--record y --offline-audio requieren --launch-engine')
    process=None;bridge=None;capture=None;detector=None;pose_worker=None;recorder=None;cv2=None
    started=time.monotonic();frames=0;tracked_frames=0;max_loop=0;energies=[];probes=[];notes=0
    composer=Composer(args.seed,genre=args.genre,tempo_lock=args.tempo_lock);extractor=FeatureExtractor()
    flash_state={'left':('',0.0),'right':('',0.0),'nod':('',0.0),'l_leg':('',0.0),'r_leg':('',0.0)}
    report={}
    try:
        if diag_jsonl is not None:
            recorder=SessionRecorder(diag_jsonl,diag_wav,{'mode':args.mode,'initial_style':args.genre,'pose_model':args.pose_model})
            print(f'Grabando sesión de diagnóstico ({args.seconds:.0f}s) -> {diag_wav} + {diag_jsonl}',flush=True)
        # Open vision before starting audio so permission prompts never leave a
        # playing background child. Uses modern MediaPipe Tasks PoseLandmarker.
        if args.mode!='mock':
            import cv2
            from .tracker import PoseTracker
            detector=PoseTracker(model=args.pose_model)
            if args.mode=='camera':
                camera_index,camera_name=resolve_camera(args.camera)
                print(f'Cámara: {camera_name} — índice OpenCV {camera_index}',flush=True)
                capture=LatestCamera(cv2,camera_index)
                pose_worker=AsyncPoseWorker(capture,detector)
            else:
                capture=cv2.VideoCapture(str(args.video))
                if not capture.isOpened():
                    raise RuntimeError('No se puede leer el vídeo')
            video_fps=capture.get(cv2.CAP_PROP_FPS) if args.mode=='video' else 60
            if not video_fps or video_fps<1:
                video_fps=30
        if args.launch_engine:
            process=start_engine(args)
        bridge=Bridge(args.library)
        probes.append(bridge.probe())
        initial_consumed=bridge.metric(1)
        started=time.monotonic()
        if cv2 and not args.no_window:
            cv2.namedWindow('CV-DAW Contemporary Ambient',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CV-DAW Contemporary Ambient',960,720)
            print('Vista de vídeo abierta. Selecciona la ventana y pulsa q para salir.',flush=True)
        print(f'{composer.preset} — mueve brazos, cabeza, torso y cadera; 1-6 o N/P estilo, T lock BPM, q salir.',flush=True)
        last_pose_seq=0
        last_features=extractor.missing()
        last_points=None
        while not args.seconds or time.monotonic()-started<args.seconds:
            begin=time.monotonic()
            if process and process.poll() is not None:
                raise RuntimeError('El motor se detuvo durante la ejecución')
            elapsed=begin-started
            rec_label=f'REC {elapsed:04.1f}/{args.seconds:.0f}s' if recorder and args.seconds else ('REC' if recorder else None)
            is_new_pose=False
            if args.mode=='mock':
                features=mock_features(elapsed)
                is_new_pose=True
            elif args.mode=='camera':
                polled=pose_worker.poll(last_pose_seq)
                if polled is not None:
                    last_pose_seq,timestamp,image,points=polled
                    last_points=points
                    is_new_pose=True
                    needed=(11,12,23,24)
                    if points and all(points[i][3]>.5 for i in needed) and any(points[i][3]>.5 for i in (15,16)):
                        features=extractor.update(points,timestamp,image.shape[0]/image.shape[1],bpm_range=composer.bpm_sync_range())
                    else:
                        features=extractor.missing(timestamp)
                    last_features=features
                    if features.left_strike!=0:
                        flash_state['left']=(f'PUNCH IZQ {"UP" if features.left_strike>0 else "DROP"}!',time.monotonic()+0.35)
                    if features.right_strike!=0:
                        flash_state['right']=(f'PUNCH DER {"UP" if features.right_strike>0 else "HIT"}!',time.monotonic()+0.35)
                    if features.left_leg_strike:
                        flash_state['l_leg']=('PIE IZQ SNARE!',time.monotonic()+0.32)
                    if features.right_leg_strike:
                        flash_state['r_leg']=('PIE DER KICK!',time.monotonic()+0.32)
                    if features.head_nod:
                        flash_state['nod']=('HEAD NOD!',time.monotonic()+0.30)
                    if not args.no_window:
                        detector.draw(image,points)
                        display=cv2.flip(image,1) if not args.no_mirror else image
                        draw_hud(cv2,display,features,composer,flash_state,rec_label)
                        cv2.imshow('CV-DAW Contemporary Ambient',display)
                else:
                    # Keep musical clock ticking smoothly without duplicating one-shot events.
                    if (
                        last_features.left_strike!=0 or last_features.right_strike!=0
                        or last_features.left_leg_strike or last_features.right_leg_strike
                        or last_features.head_nod or last_features.shoulder_drop
                    ):
                        features=last_features.__class__(**{
                            **last_features.to_dict(),
                            'left_strike':0,'right_strike':0,
                            'left_leg_strike':False,'right_leg_strike':False,
                            'head_nod':False,'shoulder_drop':False,
                        })
                    else:
                        features=last_features
                if not args.no_window:
                    key=cv2.waitKey(1)&0xff
                    if handle_window_key(key,composer):
                        break
            else:
                ok,image=capture.read()
                if not ok:
                    break
                timestamp=time.monotonic()
                points=detector.process(image,timestamp)
                last_points=points
                is_new_pose=True
                needed=(11,12,23,24)
                if points and all(points[i][3]>.5 for i in needed) and any(points[i][3]>.5 for i in (15,16)):
                    features=extractor.update(points,timestamp,image.shape[0]/image.shape[1],bpm_range=composer.bpm_sync_range())
                else:
                    features=extractor.missing(timestamp)
                if features.left_strike!=0:
                    flash_state['left']=(f'PUNCH IZQ {"UP" if features.left_strike>0 else "DROP"}!',time.monotonic()+0.35)
                if features.right_strike!=0:
                    flash_state['right']=(f'PUNCH DER {"UP" if features.right_strike>0 else "HIT"}!',time.monotonic()+0.35)
                if features.left_leg_strike:
                    flash_state['l_leg']=('PIE IZQ SNARE!',time.monotonic()+0.32)
                if features.right_leg_strike:
                    flash_state['r_leg']=('PIE DER KICK!',time.monotonic()+0.32)
                if features.head_nod:
                    flash_state['nod']=('HEAD NOD!',time.monotonic()+0.30)
                if not args.no_window:
                    detector.draw(image,points)
                    draw_hud(cv2,image,features,composer,flash_state,rec_label)
                    cv2.imshow('CV-DAW Contemporary Ambient',image)
                    key=cv2.waitKey(1)&0xff
                    if handle_window_key(key,composer):
                        break
            frame=bridge.metric(0)
            events=composer.update(features,frame,elapsed)
            ctrl=composer.controls(features)
            bridge.controls(ctrl)
            for event in events:
                bridge.send(event)
                notes+=event.kind==0
            if recorder is not None:
                recorder.record_frame(
                    elapsed_s=elapsed,
                    audio_frame=frame,
                    is_new_pose=is_new_pose,
                    points=last_points,
                    features=features,
                    composer=composer,
                    controls=ctrl,
                    events=events,
                )
            tracked_frames+=features.tracked
            energies.append(features.energy)
            frames+=1
            max_loop=max(max_loop,(time.monotonic()-begin)*1000)
            rate=video_fps if args.mode=='video' else (60 if args.mode=='camera' else 30)
            delay=1/rate-(time.monotonic()-begin)
            if delay>0:
                time.sleep(delay)
        probes.append(bridge.probe())
        # Wait for queue ingestion, not for future musical events to execute.
        deadline=time.monotonic()+1
        while bridge.metric(1)-initial_consumed<bridge.sent-1 and time.monotonic()<deadline:
            time.sleep(.001)
        duration=time.monotonic()-started
        frequency=bridge.metric(12);calls=bridge.metric(9)
        report={'mode':args.mode,'seconds':duration,'processed_frames':frames,'processed_fps':frames/max(duration,.001),
                'tracked_frames':tracked_frames,
                'camera_frames':capture.count if isinstance(capture,LatestCamera) else None,
                'max_python_processing_ms':max_loop,'ipc_roundtrip_ms':probes,'events_sent':bridge.sent,'note_ons':notes,
                'energy_min':min(energies,default=0),'energy_max':max(energies,default=0),
                'command_drops':bridge.metric(2),'invalid_commands':bridge.metric(3),'late_commands':bridge.metric(4),
                'scheduler_overflows':bridge.metric(5),'composer_skipped_ticks':composer.skipped_ticks,
                'callback_count':calls,'callback_mean_us':bridge.metric(10)/frequency/calls*1e6 if frequency and calls else None,
                'callback_max_us':bridge.metric(11)/frequency*1e6 if frequency and calls else None,
                'physical_camera_verified':args.mode=='camera' and frames>0,'physical_sound_verified':False}
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
            output,_=process.communicate(timeout=5)
            if output:
                print(output.strip())
        if pose_worker:
            pose_worker.close()
        if isinstance(capture,LatestCamera):
            capture.close()
        elif capture:
            capture.release()
        if detector:
            detector.close()
        if cv2:
            cv2.destroyAllWindows()
    if diag_jsonl is not None and diag_jsonl.is_file():
        diag,report_md=write_report(diag_jsonl,diag_wav if (diag_wav and diag_wav.is_file()) else None)
        print(f'\n=== DIAGNÓSTICO GUARDADO ===\nWAV:     {diag_wav}\nJSONL:   {diag_jsonl}\nREPORTE: {report_md}',flush=True)
        for item in diag['findings']:
            print(f'  * {item}',flush=True)
    if report:
        print(json.dumps(report,indent=2))
        if args.metrics:
            args.metrics.parent.mkdir(parents=True,exist_ok=True)
            args.metrics.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
