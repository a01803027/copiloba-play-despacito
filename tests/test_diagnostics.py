import math
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np

from composer.brain import Composer
from composer.diagnose import analyze_session, write_report
from composer.features import Features
from composer.recorder import SessionRecorder


class DiagnosticFlightRecorderTests(unittest.TestCase):
    def test_recorder_and_analyzer_produce_report_and_correlations(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            jsonl_path = tmp_dir / 'session.jsonl'
            wav_path = tmp_dir / 'session.wav'

            # Write a 4-second synthetic stereo WAV whose amplitude rises with time (matching rising energy).
            rate = 48000
            t = np.arange(rate * 4, dtype=np.float32) / rate
            env = 0.05 + 0.35 * (t / 4.0)
            left = env * np.sin(2.0 * math.pi * 220.0 * t)
            right = env * np.sin(2.0 * math.pi * 221.5 * t)
            pcm = np.clip(np.column_stack([left, right]) * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(pcm.tobytes())

            composer = Composer(seed=7, genre='house')
            rec = SessionRecorder(jsonl_path, wav_path, {'mode': 'test'})
            pts = [(0.5, 0.5, 0.0, 0.9)] * 33
            for i in range(120):
                sec = i / 30.0
                energy = 0.1 + 0.8 * (sec / 4.0)
                rh = 0.1 + 0.85 * (sec / 4.0)
                lh = 0.2 + 0.7 * (sec / 4.0)
                feat = Features(energy=energy, right_height=rh, left_height=lh, head_tilt=0.5 if sec > 2.0 else 0.0)
                events = composer.update(feat, i * 1600, sec)
                ctrl = composer.controls(feat)
                rec.record_frame(
                    elapsed_s=sec,
                    audio_frame=i * 1600,
                    is_new_pose=True,
                    points=pts,
                    features=feat,
                    composer=composer,
                    controls=ctrl,
                    events=events,
                )
            rec.record_footer({'command_drops': 0})
            rec.close()

            diag, report_md = write_report(jsonl_path, wav_path)
            self.assertTrue(report_md.is_file())
            self.assertEqual(diag['control_frames'], 120)
            self.assertGreater(diag['correlations']['energy_vs_audio_rms'], 0.85)
            self.assertGreater(diag['correlations']['left_height_vs_pad_cutoff'], 0.85)
            self.assertIsNotNone(diag['audio'])
            self.assertGreater(diag['audio']['rms_dbfs'], -30.0)
            self.assertGreater(len(diag['timeline']), 3)


if __name__ == '__main__':
    unittest.main()
