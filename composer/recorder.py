"""Synchronized flight recorder for Vision -> Features -> Composer -> Audio diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KEY_LANDMARKS = (
    (0, 'nose'),
    (7, 'l_ear'),
    (8, 'r_ear'),
    (11, 'l_shoulder'),
    (12, 'r_shoulder'),
    (13, 'l_elbow'),
    (14, 'r_elbow'),
    (15, 'l_wrist'),
    (16, 'r_wrist'),
    (23, 'l_hip'),
    (24, 'r_hip'),
    (25, 'l_knee'),
    (26, 'r_knee'),
    (27, 'l_ankle'),
    (28, 'r_ankle'),
)


def _round_value(value: Any):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return [_round_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _round_value(v) for k, v in value.items()}
    return value


class SessionRecorder:
    """Streams per-frame vision landmarks, features, composer state, controls, and IPC events to JSONL."""

    def __init__(self, jsonl_path: Path, wav_path: Path | None = None, metadata: dict | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.wav_path = Path(wav_path) if wav_path else None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.jsonl_path.open('w', encoding='utf-8')
        self.frames_logged = 0
        self.first_audio_frame: int | None = None
        self.last_audio_frame: int | None = None
        header = {
            'type': 'header',
            'wav_path': str(self.wav_path) if self.wav_path else None,
            'metadata': metadata or {},
        }
        self._file.write(json.dumps(header) + '\n')

    def record_frame(
        self,
        *,
        elapsed_s: float,
        audio_frame: int,
        is_new_pose: bool,
        points: list[tuple[float, float, float, float]] | None,
        features,
        composer,
        controls: dict,
        events: list,
    ):
        if self.first_audio_frame is None:
            self.first_audio_frame = int(audio_frame)
        self.last_audio_frame = int(audio_frame)

        landmarks = None
        if points and len(points) == 33:
            landmarks = {
                name: [round(points[idx][0], 4), round(points[idx][1], 4), round(points[idx][2], 4), round(points[idx][3], 3)]
                for idx, name in KEY_LANDMARKS
            }

        feat_dict = {
            k: _round_value(v)
            for k, v in features.to_dict().items()
            if k != 'hands'
        }
        entry = {
            'type': 'frame',
            't': round(elapsed_s, 4),
            'audio_frame': int(audio_frame),
            'new_pose': bool(is_new_pose),
            'landmarks': landmarks,
            'features': feat_dict,
            'composer': {
                'style': getattr(composer, 'genre_key', ''),
                'section': getattr(composer, 'section', ''),
                'bar': getattr(composer, 'bar_in_cycle', 0) + 1,
                'chord': getattr(composer, 'current_chord_name', ''),
                'bpm': round(float(getattr(composer, 'bpm', 0.0)), 2),
                'bpm_synced': bool(getattr(composer, 'bpm_synced', False)),
                'bpm_source': getattr(composer, 'bpm_sync_source', 'none'),
                'head_harmony': bool(getattr(composer, 'head_harmony_active', False)),
                'shoulder_groove': bool(getattr(composer, 'shoulder_groove_active', False)),
                'beat_active': bool(getattr(composer, 'beat_active', True)),
                'dj_choke': bool(getattr(composer, 'dj_choke_active', False)),
            },
            'controls': _round_value(controls),
            'events': [
                {
                    'limb': int(e.limb),
                    'kind': int(e.kind),
                    'note': int(e.note),
                    'val': round(float(e.value), 3),
                    'frame': int(e.frame),
                }
                for e in events
            ],
        }
        self._file.write(json.dumps(entry) + '\n')
        self.frames_logged += 1

    def record_footer(self, report: dict):
        footer = {
            'type': 'footer',
            'frames_logged': self.frames_logged,
            'first_audio_frame': self.first_audio_frame,
            'last_audio_frame': self.last_audio_frame,
            'report': _round_value(report),
        }
        self._file.write(json.dumps(footer) + '\n')

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
