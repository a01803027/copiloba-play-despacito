"""Phase 3 tests: Few-Shot Streaming DTW, Keras 3 1D-TCN training -> pure-NumPy .npz parity & streaming runtime."""
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np

from composer.features import FeatureExtractor
from composer.moves import (
    DEFAULT_WINDOW_FRAMES,
    FEATURE_DIM,
    NumpyTcnModel,
    PoseSanitizer,
    StreamingDtwMatcher,
    TemporalMoveClassifier,
    banded_dtw_distance,
    extract_time_window,
    resample_sequence,
)
from composer.moves.lab import save_pose_clip
from ml.train_move_net import (
    generate_synthetic_dataset,
    load_dataset_split,
    mirror_feature_sequence,
    mirror_label_name,
    train_and_export,
)
from tests.test_moves import _pose


class Tier2DanceRecognitionTests(unittest.TestCase):
    def test_dtw_matches_few_shot_template_and_rejects_idle(self):
        x, y, classes = generate_synthetic_dataset(window_frames=DEFAULT_WINDOW_FRAMES, clips_per_class=6, seed=11)
        idx_wave = classes.index('arm_wave_lr')
        idx_roll = classes.index('body_roll')
        idx_idle = classes.index('idle_random')

        wave_clips = [x[i] for i in range(len(y)) if y[i] == idx_wave]
        roll_clips = [x[i] for i in range(len(y)) if y[i] == idx_roll]
        idle_clips = [x[i] for i in range(len(y)) if y[i] == idx_idle]

        # Same-class DTW distance should be much smaller than cross-class distance.
        d_same = banded_dtw_distance(wave_clips[0], wave_clips[1])
        d_diff = banded_dtw_distance(wave_clips[0], roll_clips[0])
        self.assertLess(d_same, d_diff * 0.65)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for clip in wave_clips[:3]:
                save_pose_clip(root, 'arm_wave_lr', [tuple(row) for row in clip], list(np.linspace(0.0, 0.8, len(clip))))
            for clip in roll_clips[:3]:
                save_pose_clip(root, 'body_roll', [tuple(row) for row in clip], list(np.linspace(0.0, 0.8, len(clip))))
            for clip in idle_clips[:2]:
                save_pose_clip(root, 'idle_random', [tuple(row) for row in clip], list(np.linspace(0.0, 0.8, len(clip))))

            matcher = StreamingDtwMatcher.from_directory(root, max_distance=0.90, eval_stride=1)
            self.assertEqual(matcher.labels, ('arm_wave_lr', 'body_roll'))

            # Stream a fresh body_roll clip -> should fire BODY_ROLL!
            events = []
            test_clip = roll_clips[4]
            for i, vec in enumerate(test_clip):
                events.extend(matcher.update(i / 30.0, tuple(float(v) for v in vec)))
            self.assertTrue(any(ev.kind == 'BODY_ROLL' for ev in events), f'Expected BODY_ROLL from DTW, got {events}')

            # Stream stillness -> zero detections.
            matcher.reset()
            still_events = []
            still_vec = tuple(float(v) for v in idle_clips[0][0])
            for i in range(30):
                still_events.extend(matcher.update(i / 30.0, still_vec))
            self.assertEqual(still_events, [])

    def test_mirror_augmentation_flips_directional_labels_and_coordinates(self):
        seq = np.zeros((10, FEATURE_DIM), dtype=np.float32)
        seq[:, 8] = -0.8   # l_wrist x
        seq[:, 10] = 0.3   # r_wrist x
        mirrored = mirror_feature_sequence(seq)
        self.assertAlmostEqual(float(mirrored[0, 8]), -0.3, places=5)
        self.assertAlmostEqual(float(mirrored[0, 10]), 0.8, places=5)
        self.assertEqual(mirror_label_name('arm_wave_lr'), 'arm_wave_rl')
        self.assertEqual(mirror_label_name('body_roll'), 'body_roll')
        self.assertEqual(resample_sequence(seq, target_frames=24).shape, (24, FEATURE_DIM))

    def test_keras_tcn_train_export_numpy_parity_and_streaming_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = Path(tmp) / 'test_tcn.npz'
            args = types.SimpleNamespace(
                data=Path(tmp),
                out=out_npz,
                export_keras=None,
                export_tflite=None,
                window_frames=DEFAULT_WINDOW_FRAMES,
                aug_per_clip=4,
                epochs=12,
                batch_size=32,
                lr=2.5e-3,
                seed=7,
                demo_synthetic=True,
                quiet=True,
            )
            stats = train_and_export(args)

            # 1) High validation accuracy on synthetic dance moves.
            self.assertGreater(stats['val_accuracy'], 0.90)
            # 2) Bit-exact numerical parity between Keras 3 and pure-NumPy runtime!
            self.assertLess(stats['keras_numpy_max_diff'], 1e-4)
            # 3) Embedded-friendly size (< 200 KB) & fast pure-NumPy latency (< 2.0 ms/frame).
            self.assertLess(stats['size_kb'], 200.0)
            self.assertLess(stats['latency_ms'], 2.0)

            # 4) Streaming TemporalMoveClassifier detects a live synthetic `body_roll` & rejects `idle_random`.
            clf = TemporalMoveClassifier.from_npz(out_npz, eval_stride=1, enter_threshold=0.70)
            x, y, classes = generate_synthetic_dataset(window_frames=DEFAULT_WINDOW_FRAMES, clips_per_class=4, seed=99)
            roll_clip = x[int(np.where(y == classes.index('body_roll'))[0][0])]
            idle_clip = x[int(np.where(y == classes.index('idle_random'))[0][0])]

            fired = []
            for i, vec in enumerate(roll_clip):
                fired.extend(clf.update(i / 30.0, tuple(float(v) for v in vec)))
            self.assertTrue(any(ev.kind == 'BODY_ROLL' for ev in fired), f'Expected BODY_ROLL from TCN, got {fired}')

            clf.reset()
            idle_fired = []
            for i, vec in enumerate(idle_clip):
                idle_fired.extend(clf.update(i / 30.0, tuple(float(v) for v in vec)))
            self.assertEqual(idle_fired, [], f'Expected zero moves on idle_random, got {idle_fired}')

            # 5) FeatureExtractor accepts `tcn_model_path` cleanly without breaking normal pipeline.
            ext = FeatureExtractor(tcn_model_path=out_npz)
            self.assertIsNotNone(ext.tcn_classifier)
            loaded = NumpyTcnModel.load(out_npz)
            self.assertEqual(loaded.feature_dim, FEATURE_DIM)
            self.assertEqual(loaded.pool_mode, 'avg_max')

    def test_pose_sanitizer_height_position_aspect_invariance_and_knee_occlusion_safety(self):
        # Person A: tall / close (torso=0.36) standing on the left of a 4:3 frame.
        # Person B: shorter / farther (torso=0.18) standing on the right of a 16:9 frame.
        def shifted_scaled(dx, dy, s, aspect, knees_visible=True):
            # Build identical anatomical pose in isotropic coordinates, then project back to normalized image.
            base = _pose(
                l_sh=(0.40, 0.35, 0.0), r_sh=(0.60, 0.35, 0.0),
                l_el=(0.30, 0.45, 0.0), r_el=(0.70, 0.45, 0.0),
                l_wr=(0.24, 0.38, 0.0), r_wr=(0.76, 0.38, 0.0),
                l_hip=(0.44, 0.65, 0.0), r_hip=(0.56, 0.65, 0.0),
                l_knee=(0.44, 0.85, 0.0), r_knee=(0.56, 0.85, 0.0),
            )
            pts = []
            for idx, (x, y, z, v) in enumerate(base):
                nx = dx + (x - 0.5) * s
                ny = dy + ((y - 0.65) * s) / aspect
                if idx in (25, 26) and not knees_visible:
                    # Simulate wild off-screen MediaPipe hallucination when knees leave bottom of frame!
                    pts.append((0.1, 1.95, 0.0, 0.12))
                else:
                    pts.append((nx, ny, z, v))
            return pts

        san_a, san_b = PoseSanitizer(), PoseSanitizer()
        pf_a = san_a.process(shifted_scaled(0.32, 0.55, 1.2, 480 / 640, knees_visible=True), 0.0, aspect=480 / 640)
        pf_b = san_b.process(shifted_scaled(0.72, 0.48, 0.6, 720 / 1280, knees_visible=True), 0.0, aspect=720 / 1280)
        # Upper-body normalized coordinates + elbow bends + hip elevation must match across height/position/aspect!
        for i in range(18):
            self.assertAlmostEqual(pf_a.feature_vector[i], pf_b.feature_vector[i], places=3)

        # Off-screen knees with garbage coords (0.1, 1.95, vis=0.12) must cleanly fall back to neutral (-0.22, -0.75)!
        san_occ = PoseSanitizer()
        pf_occ = san_occ.process(shifted_scaled(0.50, 0.55, 1.2, 480 / 640, knees_visible=False), 0.0, aspect=480 / 640)
        self.assertAlmostEqual(pf_occ.feature_vector[18], -0.22, places=3)
        self.assertAlmostEqual(pf_occ.feature_vector[19], -0.75, places=3)
        self.assertAlmostEqual(pf_occ.feature_vector[20], 0.22, places=3)
        self.assertAlmostEqual(pf_occ.feature_vector[21], -0.75, places=3)

    def test_clip_level_train_val_split_prevents_leakage_and_preserves_time_scale(self):
        # Create 10 distinct 1.6s clips across 2 classes and verify clip-level split + 0.80s peak extraction.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls_name, base_val in (('move_a', 0.5), ('idle_random', 0.1)):
                for rep in range(5):
                    ts = np.linspace(0.0, 1.6, 48, dtype=np.float64)
                    feats = np.full((48, FEATURE_DIM), base_val + 0.01 * rep, dtype=np.float32)
                    # Put a sharp 0.5s pulse in the middle of the 1.6s clip.
                    feats[18:32, 9] += 0.6 * np.sin(np.linspace(0, np.pi, 14))
                    save_pose_clip(root, cls_name, [tuple(r) for r in feats], list(ts))

            x_tr, y_tr, x_val, y_val, classes = load_dataset_split(root, aug_per_clip=4, val_ratio=0.20, seed=3)
            self.assertEqual(classes, ['idle_random', 'move_a'])
            # 1 unseen validation clip per class = 2 val windows (never augmented!), while train has augmented copies.
            self.assertEqual(len(y_val), 2)
            self.assertGreater(len(y_tr), 30)
            # Peak time-window extraction on a 1.6s clip returns exact 24-frame window containing the pulse peak!
            sample_feats = np.zeros((48, FEATURE_DIM), dtype=np.float32)
            sample_feats[20:30, 9] = 1.0
            win = extract_time_window(sample_feats, np.linspace(0.0, 1.6, 48), window_seconds=0.80, target_frames=24)
            self.assertEqual(win.shape, (24, FEATURE_DIM))
            self.assertGreater(float(np.max(win[:, 9])), 0.9)


if __name__ == '__main__':
    unittest.main()
