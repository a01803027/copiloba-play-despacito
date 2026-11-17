"""Portable camera and timestamped anatomical landmarks."""
from dataclasses import dataclass
import importlib
import logging
import math
import re
import subprocess
import sys
import time

import numpy as np


@dataclass(frozen=True)
class PerceptionFrame:
    timestamp: float
    pose_landmarks: np.ndarray | None
    left_hand: np.ndarray | None
    right_hand: np.ndarray | None
    normalized_pose: np.ndarray | None
    hip_y: float | None
    sequence: int = 0
    filtered_left_hand: np.ndarray | None = None
    filtered_right_hand: np.ndarray | None = None


class CameraStream:
    def __init__(self, index=None, *, device=None, exposure=None):
        import cv2
        from composer.app import LatestCamera
        from composer.cameras import resolve_camera

        pipeline = None
        if device is not None:
            if sys.platform != 'linux' or not device.startswith('/dev/video') or not device[10:].isdigit():
                raise ValueError('Se requiere /dev/videoN en Linux')
            if exposure is None or exposure <= 0:
                raise ValueError('Indica la exposición manual del sensor con --exposure')
            command = ['v4l2-ctl', '-d', device]
            controls = subprocess.check_output([*command, '--list-ctrls'], text=True)
            names = set(re.findall(r'^\s*(\w+)\s+0x', controls, re.MULTILINE))
            auto = 'auto_exposure' if 'auto_exposure' in names else 'exposure_auto'
            shutter = 'exposure_time_absolute' if 'exposure_time_absolute' in names else 'exposure_absolute'
            if not {auto, shutter} <= names:
                raise RuntimeError('El sensor no expone controles V4L2 de obturación manual')
            subprocess.run([*command, '--set-ctrl', f'{auto}=1'], check=True)
            subprocess.run([*command, '--set-ctrl', f'{shutter}={exposure}'], check=True)
            settings = subprocess.check_output([*command, '--get-ctrl', f'{auto},{shutter}'], text=True)
            readback = dict(re.findall(r'^(\w+):\s*(-?\d+)', settings, re.MULTILINE))
            if readback.get(auto) != '1' or readback.get(shutter) != str(exposure):
                raise RuntimeError(f'El sensor no mantuvo la exposición solicitada: {settings.strip()}')
            pipeline = (
                f'v4l2src device={device} ! video/x-raw,width=640,height=480,framerate=60/1 ! '
                'queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
                'videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false'
            )
        elif exposure is not None:
            raise ValueError('--exposure requiere --device')
        camera, self.name = (0, device) if pipeline else resolve_camera(index)
        self.capture = LatestCamera(cv2, camera, fps=60, pipeline=pipeline)

    def read(self):
        return self.capture.get()

    def close(self):
        self.capture.close()


class VectorizedOneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=.04, d_cutoff=1.):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.timestamp = self.raw = self.filtered = self.derivative = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * math.pi * dt * cutoff))

    def update(self, values, timestamp):
        if self.timestamp is not None and timestamp <= self.timestamp:
            return self.filtered
        dt = timestamp - self.timestamp if self.timestamp is not None else math.inf
        if dt > .2:
            self.filtered = values.copy()
            self.derivative = np.zeros_like(values)
        else:
            derivative = (values - self.raw) / dt
            self.derivative += self.alpha(self.d_cutoff, dt) * (derivative - self.derivative)
            cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
            self.filtered = self.filtered + self.alpha(cutoff, dt) * (values - self.filtered)
        self.raw, self.timestamp = values.copy(), timestamp
        return self.filtered


class TorsoNormalizer:
    @staticmethod
    def normalize(pose):
        hip = (pose[23] + pose[24]) * .5
        shoulder = (pose[11] + pose[12]) * .5
        scale = max(float(np.linalg.norm(shoulder - hip)), 1e-4)
        return (pose - hip) / scale, float(hip[1])


class TFLiteSkeletonBackend:
    """Full-frame RGB graph with detection and anatomical left/right outputs."""
    def __init__(self, model, delegate):
        from tflite_model import TFLiteModel

        self.model = TFLiteModel(model, delegate)
        batch, self.height, self.width, channels = map(int, self.model.input['shape'])
        expected = {(1, 33, 4), (1, 2, 21, 3), (1, 2)}
        actual = {tuple(item['shape']) for item in self.model.outputs}
        if batch != 1 or channels != 3 or min(self.height, self.width) < 1 or actual != expected:
            raise ValueError('El grafo TFLite no cumple el contrato de pose y manos')

    def infer(self, rgb):
        import cv2

        image = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        outputs = {value.shape: value[0] for value in self.model.invoke(image[None] / np.float32(255))}
        pose = outputs[(1, 33, 4)]
        hands = outputs[(1, 2, 21, 3)]
        confidence = outputs[(1, 2)]
        # Preserve the shared backend's MediaPipe label convention.
        result = {label: hand for label, hand, score in zip(('Right', 'Left'), hands, confidence)
                  if score >= .5}
        return pose, result

    def close(self):
        self.model = None


class SkeletonTracker:
    def __init__(self, backend=None, *, model=None, delegate=None):
        from perception.run import MediaPipeBackend

        self.backend = None
        if backend:
            self.backend = self._attempt(backend, lambda: importlib.import_module(backend).create_backend())
        elif model:
            self.backend = self._attempt(str(model), lambda: TFLiteSkeletonBackend(model, delegate))
        self.accelerated = self.backend is not None
        if self.backend is None:
            self.backend = MediaPipeBackend()
        self.filters = [VectorizedOneEuroFilter() for _ in range(3)]
        self.sequence = 0
        self.pose_visibility = None

    @staticmethod
    def _attempt(name, factory):
        try:
            return factory()
        except Exception as exc:
            logging.warning('No se pudo activar %s. Se usará CPU. %s', name, exc)
            return None

    @staticmethod
    def array(values, shape):
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        return array if array.shape == shape and np.isfinite(array).all() else None

    def process(self, image, timestamp):
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            pose, hands = self.backend.infer(rgb)
        except Exception:
            if not self.accelerated:
                raise
            logging.exception('Falló el backend acelerado. Se usará CPU.')
            from perception.run import MediaPipeBackend
            self.backend.close()
            self.backend = MediaPipeBackend()
            self.accelerated = False
            self.pose_visibility = None
            for smoothing in self.filters:
                smoothing.reset()
            pose, hands = self.backend.infer(rgb)
        self.sequence += 1
        pose = self.array(pose, (33, 4))
        # MediaPipe Hands assumes a mirrored input. Capture is unmirrored.
        left = self.array(hands.get('Right'), (21, 3))
        right = self.array(hands.get('Left'), (21, 3))
        aspect = np.array([1., image.shape[0] / image.shape[1], 1.], dtype=np.float32)
        normalized = hip_y = None
        if pose is not None and np.all(pose[[11, 12, 23, 24], 3] >= .5):
            visibility = pose[:, 3] >= .5
            coordinates = pose[:, :3] * aspect
            smoothing = self.filters[0]
            if self.pose_visibility is not None and timestamp > smoothing.timestamp:
                reacquired = visibility & ~self.pose_visibility
                smoothing.filtered[reacquired] = coordinates[reacquired]
                smoothing.raw[reacquired] = coordinates[reacquired]
                smoothing.derivative[reacquired] = 0
            filtered = smoothing.update(coordinates, timestamp)
            normalized, hip_y = TorsoNormalizer.normalize(filtered)
            self.pose_visibility = visibility
        else:
            self.filters[0].reset()
            self.pose_visibility = None
        filtered_hands = []
        for hand, smoothing in zip((left, right), self.filters[1:]):
            if hand is None:
                smoothing.reset()
            filtered_hands.append(smoothing.update(hand * aspect, timestamp) if hand is not None else None)
        return PerceptionFrame(timestamp, pose, left, right, normalized, hip_y, self.sequence, *filtered_hands)

    def close(self):
        self.backend.close()


def draw_frame(image, frame, fps):
    import cv2

    cv2.putText(image, f'{fps:.1f} FPS | y cruda / y normalizada', (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
    if frame.normalized_pose is not None:
        for row, index in enumerate((15, 16, 27, 28), 2):
            raw = frame.pose_landmarks[index]
            normalized = frame.normalized_pose[index]
            cv2.putText(image, f'{index}: {raw[1]:.2f} / {normalized[1]:+.2f}', (10, 22 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
            if raw[3] >= .5:
                cv2.circle(image, (int(raw[0] * image.shape[1]), int(raw[1] * image.shape[0])),
                           5, (0, 255, 0), -1)


def main():
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int)
    parser.add_argument('--device')
    parser.add_argument('--exposure', type=int)
    parser.add_argument('--backend')
    parser.add_argument('--pose-model')
    parser.add_argument('--pose-delegate')
    args = parser.parse_args()
    tracker = SkeletonTracker(args.backend, model=args.pose_model, delegate=args.pose_delegate)
    camera = None
    try:
        camera = CameraStream(args.camera, device=args.device, exposure=args.exposure)
        started = time.monotonic()
        while True:
            timestamp, image = camera.read()
            frame = tracker.process(image, timestamp)
            draw_frame(image, frame, frame.sequence / max(time.monotonic() - started, .001))
            cv2.imshow('CV-DAW Percepcion', image)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
    finally:
        tracker.close()
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
