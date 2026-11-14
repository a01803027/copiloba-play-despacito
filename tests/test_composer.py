import math
import unittest
from composer.brain import Composer, degree_note
from composer.features import FeatureExtractor, Features, mock_features

class ComposerTests(unittest.TestCase):
    def sequence(self,feature,seed=7):
        brain=Composer(seed);events=[]
        for n in range(360):
            events+=brain.update(feature,n*1600,n/30)
        return brain,events

    def test_seed_and_scale(self):
        _,a=self.sequence(Features(energy=.8,symmetry=.2))
        _,b=self.sequence(Features(energy=.8,symmetry=.2))
        self.assertEqual(a,b)
        allowed={0,2,3,5,7,8,10}
        for event in a:
            if event.kind==0 and event.limb<2:
                self.assertIn(event.note%12,allowed)
                self.assertGreaterEqual(event.value,0)
                self.assertLessEqual(event.value,1)

    def test_density_follows_energy(self):
        _,low=self.sequence(Features(energy=.05))
        _,high=self.sequence(Features(energy=.9))
        self.assertGreater(sum(e.kind==0 for e in high),sum(e.kind==0 for e in low))
        self.assertFalse(any(e.kind==0 and e.limb>=2 for e in low))
        self.assertTrue(any(e.kind==0 and e.limb>=2 for e in high))

    def test_harmony_changes_after_phrase(self):
        _,events=self.sequence(Features(energy=.5))
        chord_roots={e.note for e in events if e.kind==0 and e.limb==0}
        self.assertIn(48,chord_roots)
        self.assertIn(56,chord_roots)

    def test_tracking_loss_silences(self):
        brain,events=self.sequence(Features(tracked=False))
        self.assertFalse(any(e.kind==0 for e in events))
        self.assertEqual(brain.controls(Features(tracked=False))['gain'],[0]*4)

    def test_stall_skips_without_burst(self):
        brain=Composer();brain.update(Features(),0,0)
        result=brain.update(Features(),480000,10)
        self.assertGreater(brain.skipped_ticks,0)
        self.assertLess(len(result),20)
        self.assertTrue(all(e.frame>=480000 for e in result))

    def test_features_scale_and_translation(self):
        p=[(0.,0.,0.)]*33
        for i,value in {11:(-.2,.2,0),12:(.2,.2,0),23:(-.2,.6,0),24:(.2,.6,0),15:(-.5,.3,0),16:(.5,.3,0),27:(-.2,1.,0),28:(.2,1.,0)}.items():
            p[i]=value
        a,b=FeatureExtractor(),FeatureExtractor()
        x=a.update(p,1)
        y=b.update([tuple(v*2+3 for v in point) for point in p],1)
        self.assertAlmostEqual(x.expansion,y.expansion)
        self.assertAlmostEqual(x.symmetry,y.symmetry)
        p[15]=(-.8,.1,0)
        moved=a.update(p,1+1/30)
        self.assertGreater(moved.energy,0)
        self.assertTrue(math.isfinite(moved.acceleration))
        self.assertFalse(a.missing().tracked)

    def test_invisible_feet_do_not_create_motion(self):
        p=[(0.,0.,0.,1.)]*33
        for i,value in {11:(-.2,.2,0,1),12:(.2,.2,0,1),23:(-.2,.6,0,1),24:(.2,.6,0,1),15:(-.5,.3,0,1),16:(.5,.3,0,1),27:(-.2,1.,0,0),28:(.2,1.,0,0)}.items():
            p[i]=value
        extractor=FeatureExtractor()
        extractor.update(p,1)
        p[27]=(90.,90.,0.,0.)
        self.assertEqual(extractor.update(p,1+1/30).energy,0)
        p[27]=(-.2,1.,0.,1.)
        self.assertEqual(extractor.update(p,1+2/30).energy,0)

    def test_tension_and_fast_controls(self):
        tension,_=self.sequence(Features(energy=.9,fluidity=.1))
        release,_=self.sequence(Features(energy=.05))
        self.assertEqual(tension.chord_degree,4)
        self.assertEqual(release.chord_degree,0)
        high=tension.controls(Features(energy=.9,expansion=.9))
        low=tension.controls(Features(energy=.1,expansion=.1))
        self.assertGreater(high['gain'][0],low['gain'][0])
        self.assertGreater(high['cutoff'][0],low['cutoff'][0])

    def test_mock_bounded(self):
        for n in range(300):
            f=mock_features(n/30)
            for key in ('energy','fluidity','expansion','stillness','periodicity','symmetry'):
                self.assertGreaterEqual(getattr(f,key),0)
                self.assertLessEqual(getattr(f,key),1)

if __name__=='__main__':
    unittest.main()
