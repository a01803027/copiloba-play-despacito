from pathlib import Path
import unittest
from composer.tracker import PoseTracker, resolve_pose_model

ROOT = Path(__file__).resolve().parents[1]


class TrackerTests(unittest.TestCase):
    def test_resolve_models(self):
        self.assertTrue(resolve_pose_model('full').is_file())
        self.assertTrue(resolve_pose_model('lite').is_file())
        with self.assertRaises(FileNotFoundError):
            resolve_pose_model('nonexistent-model.task')

    def test_pose_landmarker_on_sample_video(self):
        video_path = ROOT / 'brazo.mp4'
        if not video_path.is_file():
            self.skipTest('brazo.mp4 not present')
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        self.assertTrue(cap.isOpened())
        detected = 0
        try:
            with PoseTracker(model='lite') as tracker:
                for idx in range(15):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # Also verify duplicate/non-monotonic timestamp handling.
                    pts = tracker.process(frame, idx / 30.0 if idx % 2 else 0.0)
                    if pts is not None:
                        self.assertEqual(len(pts), 33)
                        detected += 1
        finally:
            cap.release()
        self.assertGreater(detected, 0)


if __name__ == '__main__':
    unittest.main()
