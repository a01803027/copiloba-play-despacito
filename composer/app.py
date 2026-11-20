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
from .features import FeatureExtractor, mock_features
from .genres import GENRE_ORDER
from .ipc import Bridge
from .cameras import resolve_camera

ROOT=Path(__file__).resolve().parents[1]


def default_build_dir():
    preferred = ROOT / ('realtime/build-mac' if sys.platform == 'darwin' else 'realtime/build')
    if preferred.is_dir():
        return preferred
    fallback = ROOT / 'realtime/build'
    return fallback if fallback.is_dir() else preferred


def draw_hud(cv2, image, features, composer):
    """Draw a clean, translucent musical HUD onto the display frame."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    if not features.tracked:
        status_text, status_color = 'BUSCANDO POSE', (70, 90, 240)
    elif features.grace:
        status_text, status_color = 'GRACE', (40, 195, 255)
    else:
        status_text, status_color = 'TRACKING', (90, 230, 120)

    cv2.circle(image, (22, 24), 7, status_color, -1, cv2.LINE_AA)
    title = f'{getattr(composer, "preset", "CV-DAW")}  |  BPM {composer.bpm:.0f}'
    chord_name = getattr(composer, 'current_chord_name', '')
    if chord_name:
        title += f'  |  {chord_name}'
    cv2.putText(image, title, (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 244, 248), 2, cv2.LINE_AA)
    cv2.putText(image, status_text, (w - 155, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1, cv2.LINE_AA)

    if not features.tracked:
        cv2.putText(
            image,
            'Necesito ver hombros, caderas y al menos una mano (alejate para incluir piernas)',
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 190, 210),
            1,
            cv2.LINE_AA,
        )
        return image

    meters = (
        ('Energia', features.energy, (60, 210, 255)),
        ('Expansion', features.expansion, (120, 230, 130)),
        ('Fluidez', features.fluidity, (240, 170, 90)),
        ('Altura', 0.5 * (features.left_height + features.right_height), (220, 130, 245)),
    )
    bar_w = max(70, (w - 60) // len(meters) - 20)
    for idx, (label, val, color) in enumerate(meters):
        x0 = 18 + idx * (bar_w + 20)
        y0 = 56
        cv2.putText(image, f'{label} {val:.2f}', (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 216, 224), 1, cv2.LINE_AA)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + 12), (55, 60, 72), -1)
        fill = int(bar_w * max(0.0, min(1.0, val)))
        if fill > 0:
            cv2.rectangle(image, (x0, y0), (x0 + fill, y0 + 12), color, -1)

    footer = 'Teclas: [1] Ambient  [2] House  [3] Lo-Fi  [4] Cinematic  |  [q] Salir'
    cv2.putText(image, footer, (18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 225, 235), 1, cv2.LINE_AA)
    return image


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
        raise ValueError('--record requiere --offline-audio: nunca se escribe un archivo desde el callback')
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
    p.add_argument('--genre',choices=GENRE_ORDER,default='ambient',help='Género musical inicial (cambiable en vivo con teclas 1-4)')
    p.add_argument('--video',type=Path)
    p.add_argument('--camera',type=int,default=None,help='Índice manual; por defecto se prefiere la webcam integrada del Mac sobre el iPhone')
    p.add_argument('--pose-model',choices=('full','lite'),default='full',help='Modelo MediaPipe Tasks (.task en models/)')
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
    p.add_argument('--no-window',action='store_true')
    p.add_argument('--metrics',type=Path)
    args=p.parse_args()
    if not math.isfinite(args.seconds) or args.seconds<0:
        p.error('--seconds debe ser un número no negativo')
    if args.mode=='video' and not args.video:
        p.error('--mode video requiere --video archivo.mp4')
    if (args.record or args.offline_audio) and not args.launch_engine:
        p.error('--record y --offline-audio requieren --launch-engine')
    process=None;bridge=None;capture=None;detector=None;cv2=None
    started=time.monotonic();frames=0;tracked_frames=0;max_loop=0;energies=[];probes=[];notes=0
    composer=Composer(args.seed,genre=args.genre);extractor=FeatureExtractor()
    report={}
    try:
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
            else:
                capture=cv2.VideoCapture(str(args.video))
                if not capture.isOpened():
                    raise RuntimeError('No se puede leer el vídeo')
            video_fps=capture.get(cv2.CAP_PROP_FPS) if args.mode=='video' else 30
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
        print('Contemporary Ambient — mueve todo el cuerpo; q o Control+C para salir.',flush=True)
        while not args.seconds or time.monotonic()-started<args.seconds:
            begin=time.monotonic()
            if process and process.poll() is not None:
                raise RuntimeError('El motor se detuvo durante la ejecución')
            elapsed=begin-started
            if args.mode=='mock':
                features=mock_features(elapsed)
            else:
                if args.mode=='camera':
                    timestamp,image=capture.get()
                else:
                    ok,image=capture.read()
                    if not ok:
                        break
                    timestamp=time.monotonic()
                points=detector.process(image,timestamp)
                needed=(11,12,23,24)
                if points and all(points[i][3]>.5 for i in needed) and any(points[i][3]>.5 for i in (15,16)):
                    features=extractor.update(points,timestamp,image.shape[0]/image.shape[1])
                else:
                    features=extractor.missing(timestamp)
                if not args.no_window:
                    detector.draw(image,points)
                    display=cv2.flip(image,1) if (args.mode=='camera' and not args.no_mirror) else image
                    draw_hud(cv2,display,features,composer)
                    cv2.imshow('CV-DAW Contemporary Ambient',display)
                    key=cv2.waitKey(1)&0xff
                    if key==ord('q'):
                        break
                    if key in (ord('1'),ord('2'),ord('3'),ord('4')):
                        selected=GENRE_ORDER[key-ord('1')]
                        composer.set_genre(selected)
                        print(f'Género cambiado a: {composer.preset}',flush=True)
            frame=bridge.metric(0)
            events=composer.update(features,frame,elapsed)
            bridge.controls(composer.controls(features))
            for event in events:
                bridge.send(event)
                notes+=event.kind==0
            tracked_frames+=features.tracked
            energies.append(features.energy)
            frames+=1
            max_loop=max(max_loop,(time.monotonic()-begin)*1000)
            rate=video_fps if args.mode=='video' else 30
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
        if bridge:
            bridge.close()
        if process:
            if process.poll() is None:
                process.terminate()
            output,_=process.communicate(timeout=5)
            if output:
                print(output.strip())
        if isinstance(capture,LatestCamera):
            capture.close()
        elif capture:
            capture.release()
        if detector:
            detector.close()
        if cv2:
            cv2.destroyAllWindows()
    if report:
        print(json.dumps(report,indent=2))
        if args.metrics:
            args.metrics.parent.mkdir(parents=True,exist_ok=True)
            args.metrics.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
