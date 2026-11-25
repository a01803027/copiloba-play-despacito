"""Pluggable Pose Backend Protocol & STM32MP257F-DK X-LINUX-AI (MoveNet INT8 NPU / TFLite) Trackers.

Allows swapping MediaPipe Pose (macOS / desktop) with MoveNet SinglePose INT8 on the
STM32MP257F-DK 1.35 TOPS VeriSilicon NPU (`X-LINUX-AI` / `stai_mpu`) or TFLite CPU without
changing a single line of Tier 0 PoseSanitizer, Tier 1 Kinematics, Tier 2 Pure-NumPy TCN/DTW,
or the C++20 Audio Composer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence
import numpy as np

# Mapping from COCO-17 keypoint indices (MoveNet / YOLOv8-Pose / X-LINUX-AI) to MediaPipe-33 indices.
COCO17_TO_MP33: tuple[tuple[int, int], ...] = (
    (0, 0),    # nose
    (1, 2),    # left_eye
    (2, 5),    # right_eye
    (3, 7),    # left_ear
    (4, 8),    # right_ear
    (5, 11),   # left_shoulder
    (6, 12),   # right_shoulder
    (7, 13),   # left_elbow
    (8, 14),   # right_elbow
    (9, 15),   # left_wrist
    (10, 16),  # right_wrist
    (11, 23),  # left_hip
    (12, 24),  # right_hip
    (13, 25),  # left_knee
    (14, 26),  # right_knee
    (15, 27),  # left_ankle
    (16, 28),  # right_ankle
)

# Standard skeleton connections drawn for COCO-17 / MediaPipe-33 shared joints.
SHARED_SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
)

# Default directories where ST X-LINUX-AI installs MoveNet NPU models on STM32MP257F-DK.
DEFAULT_STAI_MODEL_DIRS: tuple[Path, ...] = (
    Path('/usr/local/x-linux-ai/pose-estimation/models'),
    Path('/usr/local/demo-ai/pose-estimation/models'),
)


class PoseBackend(Protocol):
    """Unified interface satisfied by MediaPipe PoseTracker and STM32 NPU / TFLite MoveNet trackers."""

    def process(self, image_bgr, timestamp_s: float) -> list[tuple[float, float, float, float]] | None: ...

    def draw(self, image_bgr, points, min_visibility: float = 0.5, glitch_indices: set[int] | None = None): ...

    def close(self) -> None: ...


def calibrate_movenet_score(raw_score: float, min_valid_score: float = 0.28) -> float:
    """Map MoveNet INT8 confidence scores (~0.28..0.75) cleanly onto MediaPipe's visibility scale (0..1).

    Ensures downstream `PoseSanitizer` visibility gates (0.50..0.60) treat MoveNet identically to MediaPipe:
      - `raw_score < min_valid_score` -> mapped below 0.45 (rejected cleanly)
      - `raw_score == min_valid_score` -> 0.58 (passes wrist/elbow/torso gates)
      - `raw_score >= 0.65` -> 1.00
    """
    s = float(raw_score)
    if s < min_valid_score:
        return max(0.0, (s / max(min_valid_score, 1e-4)) * 0.45)
    frac = (s - min_valid_score) / max(0.65 - min_valid_score, 1e-4)
    return min(1.0, 0.58 + 0.42 * frac)


def coco17_to_mediapipe33(
    keypoints_17: Sequence[Sequence[float]],
    *,
    yx_order: bool = False,
    calibrate_score: bool = False,
    min_valid_score: float = 0.28,
) -> list[tuple[float, float, float, float]]:
    """Convert 17 normalized COCO keypoints into the 33-point `(x, y, z=0.0, vis)` layout.

    Args:
        keypoints_17: 17 entries of `(x, y, score)` (YOLOv8-Pose) or `(y, x, score)` (MoveNet when `yx_order=True`).
        yx_order: Set True for TensorFlow/TFLite/STAI MoveNet outputs where channels are `(y, x, score)`.
        calibrate_score: Set True for MoveNet INT8 outputs to align confidence range with MediaPipe gates.
    """
    if len(keypoints_17) != 17:
        raise ValueError(f'Expected 17 COCO keypoints, got {len(keypoints_17)}')
    out: list[tuple[float, float, float, float]] = [(0.5, 0.5, 0.0, 0.0)] * 33
    for coco_i, mp_i in COCO17_TO_MP33:
        kp = keypoints_17[coco_i]
        if yx_order:
            y, x, score = float(kp[0]), float(kp[1]), float(kp[2]) if len(kp) >= 3 else 1.0
        else:
            x, y, score = float(kp[0]), float(kp[1]), float(kp[2]) if len(kp) >= 3 else 1.0
        vis = calibrate_movenet_score(score, min_valid_score=min_valid_score) if calibrate_score else score
        out[mp_i] = (x, y, 0.0, vis)
    return out


def draw_shared_skeleton(
    image_bgr,
    points: list[tuple[float, float, float, float]] | None,
    min_visibility: float = 0.5,
    glitch_indices: set[int] | None = None,
):
    """Draw the shared 15-joint skeleton onto a BGR OpenCV frame (zero MediaPipe dependency)."""
    if not points:
        return image_bgr
    import cv2

    h, w = image_bgr.shape[:2]
    pixel_pts = [(int(p[0] * w), int(p[1] * h), p[3]) for p in points]
    glitches = glitch_indices or set()

    for start, end in SHARED_SKELETON_CONNECTIONS:
        if start < len(pixel_pts) and end < len(pixel_pts):
            x1, y1, v1 = pixel_pts[start]
            x2, y2, v2 = pixel_pts[end]
            if v1 >= min_visibility and v2 >= min_visibility:
                color = (50, 60, 255) if (start in glitches or end in glitches) else (220, 220, 220)
                cv2.line(image_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    key_limbs = {15, 16, 27, 28}
    for _, mp_i in COCO17_TO_MP33:
        x, y, vis = pixel_pts[mp_i]
        if vis < min_visibility:
            continue
        if mp_i in glitches:
            col, rad = (40, 40, 255), 8
        elif mp_i in key_limbs:
            col, rad = (0, 215, 255), 6
        else:
            col, rad = (80, 220, 120), 4
        cv2.circle(image_bgr, (x, y), rad, col, -1, cv2.LINE_AA)
    return image_bgr


def _preprocess_movenet_input(
    image_bgr,
    in_h: int,
    in_w: int,
    dtype: np.dtype,
    quant_scale: float = 0.0,
    quant_zp: int = 0,
) -> np.ndarray:
    """Resize BGR image to RGB `(1, in_h, in_w, 3)` matching MoveNet's expected dtype & quantization."""
    import cv2

    resized = cv2.resize(image_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(rgb, axis=0)
    if dtype == np.uint8:
        return arr.astype(np.uint8)
    if dtype == np.int8:
        if quant_scale > 0.0:
            q = np.round(arr.astype(np.float32) / 255.0 / quant_scale + quant_zp)
            return np.clip(q, -128, 127).astype(np.int8)
        return (arr.astype(np.int16) - 128).astype(np.int8)
    return arr.astype(np.float32)


def _extract_quant_params(info: Any) -> tuple[float, int]:
    """Read `(scale, zero_point)` safely across STAI MPU (`get_qparams`) and TFLite tensor dicts."""
    if hasattr(info, 'get_qparams'):
        qp = info.get_qparams()
        scale = float(getattr(qp, 'scale', 0.0) or 0.0)
        zp = int(getattr(qp, 'zero_point', 0) or 0)
        return scale, zp
    if isinstance(info, dict):
        qp = info.get('quantization', (0.0, 0))
        if isinstance(qp, (list, tuple)) and len(qp) >= 2:
            return float(qp[0] or 0.0), int(qp[1] or 0)
    return 0.0, 0


def _decode_movenet_output(raw: np.ndarray, out_scale: float = 0.0, out_zp: int = 0) -> np.ndarray:
    """Convert raw MoveNet output `(1, 1, 17, 3)` (float32 or quantized int8/uint8) to float32 `(17, 3)`."""
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.integer) and out_scale > 0.0:
        arr = (arr.astype(np.float32) - float(out_zp)) * float(out_scale)
    else:
        arr = arr.astype(np.float32)
    arr = np.squeeze(arr)
    if arr.shape != (17, 3):
        arr = arr.reshape((17, 3))
    return arr


class StaiMpuPoseTracker:
    """STM32MP257F-DK VeriSilicon NPU MoveNet tracker using ST's `stai_mpu` Python API (zero MediaPipe)."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        use_hw_acceleration: bool = True,
        min_valid_score: float = 0.28,
        yx_order: bool = True,
        network_override: Any = None,
    ):
        self.model_path = Path(model_path)
        self.min_valid_score = float(min_valid_score)
        self.yx_order = bool(yx_order)

        if network_override is not None:
            self._net = network_override
        else:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f'STAI NPU pose model not found: {self.model_path}. '
                    f'Install X-LINUX-AI pose package: `x-linux-ai -i stai-mpu-pose-estimation-python-ovx-npu`.'
                )
            from stai_mpu import stai_mpu_network

            self._net = stai_mpu_network(str(self.model_path), use_hw_acceleration=use_hw_acceleration)

        in_info = self._net.get_input_infos()[0]
        out_info = self._net.get_output_infos()[0]
        in_shape = tuple(in_info.get_shape() if hasattr(in_info, 'get_shape') else in_info['shape'])
        self._in_h = int(in_shape[1]) if len(in_shape) >= 3 else 192
        self._in_w = int(in_shape[2]) if len(in_shape) >= 3 else 192
        raw_dtype = in_info.get_dtype() if hasattr(in_info, 'get_dtype') else in_info.get('dtype', np.uint8)
        self._in_dtype = np.dtype(raw_dtype)
        self._in_scale, self._in_zp = _extract_quant_params(in_info)
        self._out_scale, self._out_zp = _extract_quant_params(out_info)

    def process(self, image_bgr, timestamp_s: float) -> list[tuple[float, float, float, float]] | None:
        del timestamp_s
        inp = _preprocess_movenet_input(
            image_bgr, self._in_h, self._in_w, self._in_dtype, self._in_scale, self._in_zp
        )
        self._net.set_input(0, inp)
        self._net.run()
        raw_out = self._net.get_output(index=0)
        kp17 = _decode_movenet_output(raw_out, self._out_scale, self._out_zp)
        if float(np.max(kp17[:, 2])) < self.min_valid_score:
            return None
        return coco17_to_mediapipe33(
            kp17,
            yx_order=self.yx_order,
            calibrate_score=True,
            min_valid_score=self.min_valid_score,
        )

    def draw(self, image_bgr, points, min_visibility: float = 0.5, glitch_indices: set[int] | None = None):
        return draw_shared_skeleton(image_bgr, points, min_visibility=min_visibility, glitch_indices=glitch_indices)

    def close(self) -> None:
        self._net = None


class TfliteMoveNetPoseTracker:
    """Portable MoveNet SinglePose `.tflite` tracker (works on Mac or Linux CPU via LiteRT / TFLite)."""

    def __init__(self, model_path: Path | str, *, min_valid_score: float = 0.28, yx_order: bool = True):
        self.model_path = Path(model_path)
        self.min_valid_score = float(min_valid_score)
        self.yx_order = bool(yx_order)
        if not self.model_path.is_file():
            raise FileNotFoundError(f'MoveNet TFLite model not found: {self.model_path}')

        interpreter_cls = None
        for mod_name in ('ai_edge_litert.interpreter', 'tflite_runtime.interpreter'):
            try:
                mod = __import__(mod_name, fromlist=['Interpreter'])
                interpreter_cls = mod.Interpreter
                break
            except ImportError:
                continue
        if interpreter_cls is None:
            import tensorflow as tf
            interpreter_cls = tf.lite.Interpreter

        self._interp = interpreter_cls(model_path=str(self.model_path))
        self._interp.allocate_tensors()
        in_det = self._interp.get_input_details()[0]
        out_det = self._interp.get_output_details()[0]
        self._in_idx = int(in_det['index'])
        self._out_idx = int(out_det['index'])
        self._in_h = int(in_det['shape'][1])
        self._in_w = int(in_det['shape'][2])
        self._in_dtype = np.dtype(in_det['dtype'])
        self._in_scale, self._in_zp = _extract_quant_params(in_det)
        self._out_scale, self._out_zp = _extract_quant_params(out_det)

    def process(self, image_bgr, timestamp_s: float) -> list[tuple[float, float, float, float]] | None:
        del timestamp_s
        inp = _preprocess_movenet_input(
            image_bgr, self._in_h, self._in_w, self._in_dtype, self._in_scale, self._in_zp
        )
        self._interp.set_tensor(self._in_idx, inp)
        self._interp.invoke()
        raw_out = self._interp.get_tensor(self._out_idx)
        kp17 = _decode_movenet_output(raw_out, self._out_scale, self._out_zp)
        if float(np.max(kp17[:, 2])) < self.min_valid_score:
            return None
        return coco17_to_mediapipe33(
            kp17,
            yx_order=self.yx_order,
            calibrate_score=True,
            min_valid_score=self.min_valid_score,
        )

    def draw(self, image_bgr, points, min_visibility: float = 0.5, glitch_indices: set[int] | None = None):
        return draw_shared_skeleton(image_bgr, points, min_visibility=min_visibility, glitch_indices=glitch_indices)

    def close(self) -> None:
        self._interp = None


def discover_stai_pose_model() -> Path | None:
    """Locate an installed ST X-LINUX-AI MoveNet `.nb` / `.tflite` pose model on STM32MP257F-DK."""
    for folder in DEFAULT_STAI_MODEL_DIRS:
        if not folder.is_dir():
            continue
        for pattern in ('*movenet*.nb', '*pose*.nb', '*movenet*.tflite', '*pose*.tflite'):
            matches = sorted(folder.rglob(pattern))
            if matches:
                return matches[0]
    return None


def create_pose_backend(
    backend: str = 'auto',
    *,
    pose_model: str = 'lite',
    inference_width: int = 320,
    npu_model: Path | str | None = None,
) -> PoseBackend:
    """Instantiate the requested pose tracker without importing MediaPipe on STM32 NPU boards."""
    choice = backend.strip().lower()
    if choice == 'stai-npu':
        model = Path(npu_model) if npu_model else discover_stai_pose_model()
        if model is None:
            raise FileNotFoundError(
                'No STAI NPU pose model specified or found. Pass --npu-model /path/to/movenet.nb '
                'or install `x-linux-ai -i stai-mpu-pose-estimation-python-ovx-npu`.'
            )
        return StaiMpuPoseTracker(model)

    if choice == 'movenet-tflite':
        if not npu_model:
            raise ValueError('--pose-backend movenet-tflite requires --npu-model /path/to/movenet.tflite')
        return TfliteMoveNetPoseTracker(npu_model)

    if choice == 'mediapipe':
        from ..tracker import PoseTracker
        return PoseTracker(model=pose_model, inference_width=inference_width)

    if choice == 'auto':
        if npu_model is not None:
            try:
                import stai_mpu  # noqa: F401
                return StaiMpuPoseTracker(npu_model)
            except ImportError:
                return TfliteMoveNetPoseTracker(npu_model)
        try:
            import mediapipe  # noqa: F401
            from ..tracker import PoseTracker
            return PoseTracker(model=pose_model, inference_width=inference_width)
        except ImportError:
            found = discover_stai_pose_model()
            if found is not None:
                return StaiMpuPoseTracker(found)
            raise RuntimeError(
                'Neither MediaPipe nor an STM32 X-LINUX-AI MoveNet model was found. '
                'On STM32MP257F-DK install X-LINUX-AI (`x-linux-ai -i stai-mpu-pose-estimation-python-ovx-npu`) '
                'or pass `--pose-backend stai-npu --npu-model /path/to/movenet.nb`.'
            )

    raise ValueError(f'Unknown pose backend: {backend!r} (choose auto, mediapipe, stai-npu, movenet-tflite)')
