import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('frontend', Path(__file__).parents[1] / 'perception' / 'run.py')
frontend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontend)

class PostureTests(unittest.TestCase):
    def hand(self, fingers):
        p = [(0., 0., 0.) for _ in range(21)]
        p[9] = (0., 1., 0.)
        p[4] = (-1., .5, 0.)
        for tip, pip, extended in zip((8,12,16,20), (6,10,14,18), fingers):
            p[pip] = (.1, 1., 0.)
            p[tip] = (.1, 2. if extended else .5, 0.)
        return p

    def test_states(self):
        self.assertEqual(frontend.posture(self.hand([1,1,1,1])), 0)
        self.assertEqual(frontend.posture(self.hand([0,0,0,0])), 1)
        self.assertEqual(frontend.posture(self.hand([1,1,0,0])), 3)
        p = self.hand([1,1,1,1])
        p[4] = p[8]
        self.assertEqual(frontend.posture(p), 2)

    def test_rotation_scale_translation(self):
        p = self.hand([1,1,0,0])
        q = [(3-2*y, 4+2*x, 2*z) for x,y,z in p]
        self.assertEqual(frontend.posture(q), frontend.posture(p))
