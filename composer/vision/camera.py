"""Threaded low-latency camera capture and async Pose-Only vision worker (~28-30 FPS)."""
from __future__ import annotations

import queue
import sys
import threading
import time


class LatestCamera:
    """Drops stale camera frames so MediaPipe always sees the freshest frame."""

    def __init__(self, cv2, index, *, fps=30, pipeline=None):
        self.cv2 = cv2
        if pipeline:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif sys.platform == 'darwin':
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError('No se pudo abrir la cámara. Revisa permisos de Cámara para Terminal y cierra otras apps.')
        if not pipeline:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.negotiated_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.count = 0
        self.worker = threading.Thread(target=self.read, daemon=True)
        self.worker.start()

    def read(self):
        while not self.stop.is_set():
            ok, image = self.cap.read()
            if not ok:
                break
            self.count += 1
            item = (time.monotonic(), image)
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


class AsyncPoseWorker:
    """Runs PoseLandmarker on a dedicated thread so the 60 Hz musical clock never stalls."""

    def __init__(self, capture, pose_detector):
        self._capture = capture
        self._pose = pose_detector
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
                points = self._pose.process(image, timestamp)
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
