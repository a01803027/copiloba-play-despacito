"""Modern MediaPipe Tasks API (1.2+) PoseTracker for Python 3.14+.

Uses mediapipe.tasks.python.vision.PoseLandmarker in VIDEO mode with local
.task models in models/ (pose_landmarker_full.task / pose_landmarker_lite.task).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'

MODEL_FILES = {
    'full': MODELS_DIR / 'pose_landmarker_full.task',
    'lite': MODELS_DIR / 'pose_landmarker_lite.task',
}

# Fallback connections if PoseLandmarksConnections is unavailable.
DEFAULT_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
)


def resolve_pose_model(model: str = 'full') -> Path:
    """Resolve 'full', 'lite', or an explicit .task path."""
    candidate = MODEL_FILES.get(model, Path(model))
    if not candidate.is_file():
        raise FileNotFoundError(
            f'Modelo MediaPipe no encontrado: {candidate}. '
            f'Asegúrate de tener pose_landmarker_full.task o pose_landmarker_lite.task en {MODELS_DIR}.'
        )
    return candidate


class PoseTracker:
    """MediaPipe Tasks PoseLandmarker wrapper with monotonic video timestamps."""

    def __init__(
        self,
        model: str = 'full',
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        self.model_path = resolve_pose_model(model)

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self._connections = [
            (c.start, c.end)
            for c in getattr(vision.PoseLandmarksConnections, 'POSE_LANDMARKS', ())
        ] or list(DEFAULT_CONNECTIONS)

    def process(self, image_bgr, timestamp_s: float) -> Optional[list[tuple[float, float, float, float]]]:
        """Run pose estimation on a BGR OpenCV frame and return 33 (x, y, z, visibility) points."""
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(timestamp_s * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        if len(landmarks) != 33:
            return None
        return [
            (
                float(lm.x),
                float(lm.y),
                float(lm.z),
                float(lm.visibility if lm.visibility is not None else 1.0),
            )
            for lm in landmarks
        ]

    def draw(
        self,
        image_bgr,
        points: Optional[list[tuple[float, float, float, float]]],
        min_visibility: float = 0.5,
    ):
        """Draw pose skeleton cleanly onto a BGR OpenCV frame."""
        if not points:
            return image_bgr
        import cv2

        h, w = image_bgr.shape[:2]
        pixel_pts = [(int(p[0] * w), int(p[1] * h), p[3]) for p in points]

        for start, end in self._connections:
            if start < len(pixel_pts) and end < len(pixel_pts):
                x1, y1, v1 = pixel_pts[start]
                x2, y2, v2 = pixel_pts[end]
                if v1 >= min_visibility and v2 >= min_visibility:
                    cv2.line(image_bgr, (x1, y1), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)

        key_limbs = {15, 16, 27, 28}
        for idx, (x, y, vis) in enumerate(pixel_pts):
            if vis < min_visibility:
                continue
            color = (0, 215, 255) if idx in key_limbs else (80, 220, 120)
            radius = 6 if idx in key_limbs else 3
            cv2.circle(image_bgr, (x, y), radius, color, -1, cv2.LINE_AA)
        return image_bgr

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
