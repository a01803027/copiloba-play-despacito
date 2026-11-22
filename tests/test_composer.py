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
            for key in ('energy','fluidity','expansion','stillness','periodicity','symmetry','left_height','right_height','body_level'):
                self.assertGreaterEqual(getattr(f,key),0)
                self.assertLessEqual(getattr(f,key),1)
            self.assertGreaterEqual(f.lateral_balance,-1)
            self.assertLessEqual(f.lateral_balance,1)

    def test_occlusion_grace_period_and_expiry(self):
        p=[(0.,0.,0.,1.)]*33
        for i,value in {11:(-.2,.2,0,1),12:(.2,.2,0,1),23:(-.2,.6,0,1),24:(.2,.6,0,1),15:(-.5,.2,0,1),16:(.5,.2,0,1)}.items():
            p[i]=value
        extractor=FeatureExtractor(missing_grace_seconds=0.30)
        extractor.update(p,1.0)
        p[15]=(-.8,0.,0.,1.)
        active=extractor.update(p,1.0+1/30)
        self.assertTrue(active.tracked)
        self.assertFalse(active.grace)
        self.assertGreater(active.energy,0)
        # Brief 100 ms flicker stays tracked in grace mode with decayed energy.
        flicker=extractor.missing(1.0+1/30+0.10)
        self.assertTrue(flicker.tracked)
        self.assertTrue(flicker.grace)
        self.assertLess(flicker.energy,active.energy)
        # Missing beyond 300 ms performs hard reset.
        lost=extractor.missing(1.0+1/30+0.40)
        self.assertFalse(lost.tracked)
        self.assertEqual(lost.energy,0.)

    def test_left_right_heights_and_lateral_balance(self):
        p=[(0.,0.,0.,1.)]*33
        # Hip y=0.6, shoulder y=0.2 (torso=0.4). Left wrist high (y=0.0), right wrist low (y=0.6).
        for i,value in {11:(-.2,.2,0,1),12:(.2,.2,0,1),23:(-.2,.6,0,1),24:(.2,.6,0,1),15:(.0,.0,0,1),16:(.6,.6,0,1)}.items():
            p[i]=value
        extractor=FeatureExtractor()
        f=extractor.update(p,1.0)
        self.assertGreater(f.left_height,f.right_height)
        self.assertGreater(f.lateral_balance,0.0)

    def test_all_genres_produce_distinct_musical_output(self):
        from composer.styles import STYLE_ORDER
        bpms={}
        for genre in STYLE_ORDER:
            brain=Composer(seed=7,genre=genre)
            events=[]
            for n in range(180):
                events+=brain.update(Features(energy=.75,expansion=.7,fluidity=.6,left_height=.6,right_height=.7,lateral_balance=.4),n*1600,n/30)
            note_ons=[e for e in events if e.kind==0]
            self.assertGreater(len(note_ons),10,f'genre {genre} produced too few notes')
            ctrl=brain.controls(Features(energy=.75,expansion=.7,fluidity=.6,lateral_balance=.5))
            self.assertIn('pan',ctrl)
            self.assertIn('delay_send',ctrl)
            self.assertIn('resonance',ctrl)
            self.assertIn('sidechain',ctrl)
            bpms[genre]=brain.bpm
        self.assertGreater(bpms['house'],bpms['ambient'])
        # Live genre switching updates preset and triggers preset change events.
        brain=Composer(seed=7,genre='ambient')
        brain.update(Features(energy=.5),0,0.0)
        brain.set_genre('house')
        switched=brain.update(Features(energy=.5),1600,1/30)
        self.assertTrue(any(e.kind==4 for e in switched))
        self.assertEqual(brain.genre_key,'house')

if __name__=='__main__':
    unittest.main()
