"""Modular full-body move & dance recognition (Tier 0 Sanitizer -> Tier 1 Kinematics + Tier 2 DTW/TCN -> Arbiter)."""
from .arbiter import (
    CUSTOM_TIER2_PRIORITY,
    LOWER_BODY_MOVES,
    MOVE_PRIORITY,
    TIER1_KNOWN_MOVES,
    MoveArbiter,
    is_custom_move,
    move_channel,
    move_priority,
)
from .dtw import (
    DEFAULT_WINDOW_FRAMES,
    DEFAULT_WINDOW_SECONDS,
    DtwMatch,
    StreamingDtwMatcher,
    banded_dtw_distance,
    estimate_active_duration,
    extract_time_window,
    normalize_for_dtw,
    resample_sequence,
)
from .lab_diagnose import LabSessionRecorder, audit_dataset_dir, write_lab_report
from .pose_frame import CANONICAL_JOINTS, FEATURE_DIM, PoseFrame
from .recognizer import BodyMoveRecognizer, MoveEvent, MoveSnapshot
from .sanitizer import PoseSanitizer, elbow_bend_2d
from .tcn_runtime import NumpyTcnModel, TemporalMoveClassifier

__all__ = [
    'BodyMoveRecognizer',
    'CANONICAL_JOINTS',
    'CUSTOM_TIER2_PRIORITY',
    'DEFAULT_WINDOW_FRAMES',
    'DEFAULT_WINDOW_SECONDS',
    'DtwMatch',
    'FEATURE_DIM',
    'LOWER_BODY_MOVES',
    'LabSessionRecorder',
    'MOVE_PRIORITY',
    'MoveArbiter',
    'MoveEvent',
    'MoveSnapshot',
    'NumpyTcnModel',
    'PoseFrame',
    'PoseSanitizer',
    'StreamingDtwMatcher',
    'TIER1_KNOWN_MOVES',
    'TemporalMoveClassifier',
    'audit_dataset_dir',
    'banded_dtw_distance',
    'elbow_bend_2d',
    'estimate_active_duration',
    'extract_time_window',
    'is_custom_move',
    'move_channel',
    'move_priority',
    'normalize_for_dtw',
    'resample_sequence',
    'write_lab_report',
]
