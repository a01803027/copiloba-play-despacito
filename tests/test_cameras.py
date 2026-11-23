import unittest
from composer.cameras import select_mac_camera

class CameraSelectionTests(unittest.TestCase):
    def test_matches_opencv_unique_id_order(self):
        # Discovery order differs from OpenCV's sorted indexing.
        mac={'_name':'MacBook Pro Camera','spcamera_unique-id':'6C70','spcamera_model-id':'MacBook Pro Camera'}
        phone={'_name':'Phone Camera','spcamera_unique-id':'0EED','spcamera_model-id':'iPhone15,3'}
        self.assertEqual(select_mac_camera([mac,phone]),(1,'MacBook Pro Camera'))

    def test_mac_returns_to_zero_without_phone(self):
        self.assertEqual(select_mac_camera([{'_name':'MacBook Pro Camera','spcamera_unique-id':'6C70'}])[0],0)

    def test_no_silent_phone_fallback(self):
        with self.assertRaises(RuntimeError):
            select_mac_camera([{'_name':'Phone Camera','spcamera_unique-id':'0EED','spcamera_model-id':'iPhone15,3'}])

    def test_no_devices(self):
        with self.assertRaises(RuntimeError):
            select_mac_camera([])
