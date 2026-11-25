"""Threaded low-latency camera capture, Pose-only worker, and pluggable PoseBackends (MediaPipe + STM32 NPU)."""
from .camera import AsyncPoseWorker, LatestCamera
from .pose_backend import (
    COCO17_TO_MP33,
    PoseBackend,
    StaiMpuPoseTracker,
    TfliteMoveNetPoseTracker,
    calibrate_movenet_score,
    coco17_to_mediapipe33,
    create_pose_backend,
    discover_stai_pose_model,
    draw_shared_skeleton,
)

__all__ = [
    'AsyncPoseWorker',
    'COCO17_TO_MP33',
    'LatestCamera',
    'PoseBackend',
    'StaiMpuPoseTracker',
    'TfliteMoveNetPoseTracker',
    'calibrate_movenet_score',
    'coco17_to_mediapipe33',
    'create_pose_backend',
    'discover_stai_pose_model',
    'draw_shared_skeleton',
]
