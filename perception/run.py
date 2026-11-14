#!/usr/bin/env python3
"""Non-RT latest-frame camera frontend. No audio/DSP runs in Python."""
import argparse
import importlib
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque


def posture(points):
    """Rotation-invariant geometric baseline: OPEN/FIST/PINCH/PEACE."""
    def distance(a, b):
        return math.sqrt(sum((points[a][i] - points[b][i]) ** 2 for i in range(3)))
    palm = max(distance(0, 9), 1e-6)
    if distance(4, 8) / palm < 0.25:
        return 2
    extended = [distance(0, tip) > 1.15 * distance(0, pip)
                for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]]
    if extended == [True, True, False, False]:
        return 3
    if sum(extended) <= 1:
        return 1
    return 0


class MediaPipeBackend:
    """CPU reference implementation; NPU delegates require a board-specific adapter."""
    def __init__(self):
        import mediapipe as mp
        self.pose = mp.solutions.pose.Pose(model_complexity=0, smooth_landmarks=False)
        self.hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0)

    def infer(self, rgb):
        p = self.pose.process(rgb)
        h = self.hands.process(rgb)
        pose = None
        if p.pose_landmarks:
            pose = [(v.x, v.y, v.z, v.visibility) for v in p.pose_landmarks.landmark]
        hands = {}
        if h.multi_hand_landmarks:
            for landmarks, handedness in zip(h.multi_hand_landmarks, h.multi_handedness):
                hands[handedness.classification[0].label] = [(v.x, v.y, v.z) for v in landmarks.landmark]
        return pose, hands

    def close(self):
        self.pose.close()
        self.hands.close()


class TemporalWorker:
    """Latest-only model jobs; inference never blocks the fast path.

    Model contract: [1,30,12] float or int8 velocities in torso lengths/sec;
    single output [energy, groove probabilities...]. PLL BPM remains in C++.
    """
    def __init__(self, model, delegate):
        self.jobs = queue.Queue(maxsize=1)
        self.result = None
        self.error = None
        self.stop = threading.Event()
        self.model, self.delegate = model, delegate
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, velocities):
        try:
            self.jobs.put_nowait(velocities)
        except queue.Full:
            pass

    def run(self):
        try:
            import numpy as np
            from tflite_runtime.interpreter import Interpreter, load_delegate
            delegates = [load_delegate(self.delegate)] if self.delegate else []
            interpreter = Interpreter(model_path=self.model, experimental_delegates=delegates, num_threads=1)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
            if list(inp['shape']) != [1, 30, 12]:
                raise ValueError('temporal input must be [1,30,12]')
            while not self.stop.is_set():
                try:
                    timestamp, window = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                x = np.array([window], dtype=np.float32)
                if inp['dtype'] != np.float32:
                    scale, zero = inp['quantization']
                    if scale <= 0:
                        raise ValueError('missing per-tensor quantization scale')
                    limits = np.iinfo(inp['dtype'])
                    x = np.clip(np.rint(x / scale + zero), limits.min, limits.max).astype(inp['dtype'])
                interpreter.set_tensor(inp['index'], x)
                interpreter.invoke()
                y = interpreter.get_tensor(out['index']).reshape(-1)
                if out['dtype'] != np.float32:
                    scale, zero = out['quantization']
                    y = (y.astype(np.float32) - zero) * scale
                if len(y) != 5 or not np.all(np.isfinite(y)):
                    raise ValueError('invalid energy/groove output')
                self.result = timestamp, float(np.clip(y[0], 0, 1)), y[1:].tolist()
        except Exception as exc:
            self.error = exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain', default='build/cvdaw-brain')
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--exposure', type=int, required=True, help='V4L2 exposure_absolute units; sensor-specific')
    parser.add_argument('--backend', help='Python module exporting create_backend().infer(rgb) -> (pose,hands)')
    parser.add_argument('--temporal-model')
    parser.add_argument('--delegate', help='Vendor TFLite delegate .so for temporal model')
    parser.add_argument('--mirrored', action='store_true', help='Input is mirrored; MediaPipe handedness is already correct')
    parser.add_argument('--full-quantize', action='store_true', help='Allow 40ms waits; incompatible with strict <35ms')
    args = parser.parse_args()
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {0})
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import numpy as np
    Gst.init(None)
    # Fail closed if manual exposure controls are unsupported; do not silently use auto.
    subprocess.run(['v4l2-ctl', '-d', args.device, '--set-ctrl',
                    f'exposure_auto=1,exposure_absolute={args.exposure}'], check=True)
    backend = importlib.import_module(args.backend).create_backend() if args.backend else MediaPipeBackend()
    temporal = TemporalWorker(args.temporal_model, args.delegate) if args.temporal_model else None
    pipeline = Gst.parse_launch(
        f'v4l2src device={args.device} do-timestamp=false ! '
        f'video/x-raw,width={args.width},height={args.height},framerate=60/1 ! '
        'queue max-size-buffers=1 leaky=downstream ! videoconvert ! video/x-raw,format=RGB ! '
        'appsink name=frames max-buffers=1 drop=true sync=false')
    sink = pipeline.get_by_name('frames')
    command = [args.brain] + (['--full-quantize'] if args.full_quantize else [])
    brain = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, bufsize=1)
    window = deque(maxlen=30)
    history = [deque(maxlen=3), deque(maxlen=3)]
    gestures = [0, 0, 0, 0]
    previous = None
    last_stamp = None
    frames, gaps = 0, 0
    start = time.monotonic()
    try:
        pipeline.set_state(Gst.State.PLAYING)
        while brain.poll() is None:
            sample = sink.emit('try-pull-sample', 1_000_000_000)
            if sample is None:
                raise RuntimeError('camera timeout: inspect negotiated caps and V4L2 controls')
            buffer = sample.get_buffer()
            if buffer.pts == Gst.CLOCK_TIME_NONE:
                raise RuntimeError('camera has no capture PTS')
            clock = pipeline.get_clock()
            # Map pipeline running-time PTS to CLOCK_MONOTONIC. Timestamp quality still
            # depends on the sensor/driver (SOE vs EOF), to be measured on the board.
            stamp = time.monotonic_ns() + pipeline.get_base_time() + buffer.pts - clock.get_time()
            if last_stamp is not None and stamp <= last_stamp:
                continue
            dt = (stamp-last_stamp)/1e9 if last_stamp else 1/60
            last_stamp = stamp
            frames += 1
            gaps += dt > .025
            if time.monotonic_ns() - stamp > 50_000_000:
                continue
            ok, mapped = buffer.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                # GStreamer RGB rows are padded to four-byte alignment.
                stride = (args.width * 3 + 3) & ~3
                rgb = np.ndarray((args.height, args.width, 3), dtype=np.uint8,
                                 buffer=mapped.data, strides=(stride, 3, 1)).copy()
            finally:
                buffer.unmap(mapped)
            pose, hands = backend.infer(rgb)
            if pose is None or len(pose) != 33 or any(pose[i][3] < .6 for i in (11,12,15,16,23,24,27,28)):
                previous = None
                window.clear()
                continue
            aspect = args.height / args.width
            xyz = [(p[0], p[1]*aspect, p[2]) for p in pose]
            if not all(math.isfinite(v) for p in xyz for v in p):
                continue
            for i, label in enumerate(('Left', 'Right')):
                # MediaPipe assumes mirrored/selfie input. Native camera frames are not mirrored.
                source = label if args.mirrored else ('Right' if label == 'Left' else 'Left')
                if source in hands:
                    history[i].append(posture(hands[source]))
                    if len(history[i]) == 3 and len(set(history[i])) == 1:
                        gestures[i] = history[i][0]
                else:
                    history[i].clear()  # Keep state during a brief hand occlusion.
            hip = [(xyz[23][j]+xyz[24][j])/2 for j in range(3)]
            shoulder = [(xyz[11][j]+xyz[12][j])/2 for j in range(3)]
            scale = math.dist(hip, shoulder)
            if scale < .025:
                previous = None
                continue
            position = [(xyz[i][j]-hip[j])/scale for i in (15,16,27,28) for j in range(3)]
            if dt > .1:
                window.clear()
                previous = None
            if previous is not None:
                window.append([(a-b)/dt for a,b in zip(position, previous)])
            previous = position
            if temporal:
                if temporal.error:
                    raise RuntimeError('temporal inference failed') from temporal.error
                if len(window) == 30:
                    temporal.submit((stamp, list(window)))
            values = [str(stamp)] + [str(v) for p in xyz for v in p] + [str(g) for g in gestures]
            # Model energy is optional; BPM is always estimated by the C++ phase tracker.
            if temporal and temporal.result and stamp-temporal.result[0] < 500_000_000:
                values += [str(temporal.result[1])] + [str(p) for p in temporal.result[2]]
            brain.stdin.write(' '.join(values)+'\n')
            brain.stdin.flush()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        backend.close()
        if temporal:
            temporal.stop.set()
            temporal.thread.join(timeout=2)
        brain.stdin.close()
        try:
            brain.wait(timeout=2)
        except subprocess.TimeoutExpired:
            brain.terminate()
            brain.wait()
        elapsed = time.monotonic()-start
        print(f'capture samples={frames} effective_fps={frames/max(elapsed,.001):.2f} gaps_over_25ms={gaps}')
    if brain.returncode:
        raise SystemExit(brain.returncode)


if __name__ == '__main__':
    main()
