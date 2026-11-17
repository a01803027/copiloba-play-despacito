import unittest

import numpy as np

from perception_layer import SkeletonTracker, TorsoNormalizer, VectorizedOneEuroFilter


class PerceptionLayerTests(unittest.TestCase):
    def test_filter_equation_and_reset(self):
        f = VectorizedOneEuroFilter(beta=0)
        start = np.zeros((33, 3))
        end = np.ones_like(start)
        f.update(start, 1)
        np.testing.assert_allclose(f.update(end, 1.02), f.alpha(1.2, .02))
        np.testing.assert_array_equal(f.update(end, 2), end)

    def test_torso_scale_and_translation(self):
        points = np.arange(99, dtype=float).reshape(33, 3)
        first, hip = TorsoNormalizer.normalize(points)
        second, _ = TorsoNormalizer.normalize(points * 3 + [7, 2, 5])
        np.testing.assert_allclose(first, second)
        self.assertEqual(hip, (points[23, 1] + points[24, 1]) / 2)

    def test_contract_rejects_bad_landmarks(self):
        self.assertIsNone(SkeletonTracker.array([[0, 0, 0]], (21, 3)))
        self.assertIsNone(SkeletonTracker.array(np.full((33, 4), np.nan), (33, 4)))

    def test_raw_contract_handedness_and_occlusion(self):
        from unittest.mock import patch
        from perception_layer import PerceptionFrame

        pose = np.zeros((33, 4), dtype=np.float32)
        pose[:, 3] = 1
        pose[[23, 24], 1] = .6
        pose[[11, 12], 1] = .3
        hand = np.full((21, 3), .1, dtype=np.float32)

        def rgb_passthrough(image, _conversion):
            return image

        class Backend:
            def infer(self, rgb):
                return pose, {'Right': hand}

        import sys
        from types import SimpleNamespace
        cv2 = SimpleNamespace(cvtColor=rgb_passthrough, COLOR_BGR2RGB=0)
        with patch('perception.run.MediaPipeBackend', return_value=Backend()), patch.dict(sys.modules, {'cv2': cv2}):
            tracker = SkeletonTracker()
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = tracker.process(image, 1.)
            self.assertIsInstance(result, PerceptionFrame)
            np.testing.assert_array_equal(result.left_hand, hand)
            self.assertIsNone(result.right_hand)
            self.assertAlmostEqual(result.hip_y, .45, places=6)
            pose[15, 3] = 0
            tracker.process(image, 1.02)
            pose[15] = [.3, .4, 0, 1]
            result = tracker.process(image, 1.04)
            self.assertAlmostEqual(result.normalized_pose[15, 0], 4 / 3, places=6)

    def test_tflite_graph_contract_accepts_named_shapes_independently_of_order(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from perception_layer import TFLiteSkeletonBackend

        pose = np.zeros((1, 33, 4), dtype=np.float32)
        hands = np.zeros((1, 2, 21, 3), dtype=np.float32)
        hands[0, 1] = 1
        scores = np.array([[.2, .9]], dtype=np.float32)
        model = SimpleNamespace(
            input={'shape': (1, 2, 2, 3)},
            outputs=[{'shape': value.shape} for value in (scores, pose, hands)],
            invoke=lambda image: (scores, pose, hands) if image.max() == 1 else (),
        )
        cv2 = SimpleNamespace(resize=lambda image, *_args, **_kwargs: image, INTER_LINEAR=0)
        with patch('tflite_model.TFLiteModel', return_value=model), patch.dict(sys.modules, {'cv2': cv2}):
            backend = TFLiteSkeletonBackend('pose.tflite', 'delegate.so')
            actual_pose, actual_hands = backend.infer(np.full((2, 2, 3), 255, np.uint8))
        np.testing.assert_array_equal(actual_pose, pose[0])
        self.assertEqual(set(actual_hands), {'Left'})
        np.testing.assert_array_equal(actual_hands['Left'], hands[0, 1])
