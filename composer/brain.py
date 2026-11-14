"""Three temporal layers; seeded rules in C minor, no trained model required."""
from collections import deque
from dataclasses import dataclass
import random
from .features import Features, clamp

RATE=48000
SCALE=(0,2,3,5,7,8,10)
PROGRESSION=(0,5,3,4)  # i, VI, iv, v: all triads from C natural minor.
MOTIF=(0,2,4,2,1,2,0,4)

@dataclass(frozen=True)
class Event:
    limb: int
    kind: int
    note: int
    value: float
    frame: int


def degree_note(degree, base=48):
    return base+12*(degree//7)+SCALE[degree%7]


class Composer:
    preset='Contemporary Ambient'

    def __init__(self, seed=7):
        self.random=random.Random(seed)
        self.next_frame=None
        self.tick=0
        self.context=Features()
        self.window=deque(maxlen=120)
        self.medium_at=0.
        self.bpm=76.
        self.chord_degree=0
        self.phrase_number=None
        self.last_accent=-1e9
        self.started=False
        self.skipped_ticks=0

    def controls(self, features):
        # Fast: movement shapes ongoing sound, even between composed note events.
        level=(.15+.65*features.energy) if features.tracked else 0.
        return {
            'bpm':self.bpm,'energy':features.energy,
            'cutoff':[500+7000*features.expansion,900+9000*(1-features.fluidity),1800,3500],
            'reverb':[.25+.5*features.fluidity,.2+.4*features.fluidity,.08,.12],
            'gain':[level,level*(.4+.5*features.expansion),level*.45,level*.3],
            'bend':[0.,(1-features.symmetry)*.25,0.,0.],
        }

    def update(self, features, frame, seconds, lookahead=2400):
        events=[]
        if not self.started:
            for limb,preset in enumerate((1,0,2,4)):
                events.append(Event(limb,4,preset,0.,0))
            self.started=True
        self.window.append(features)
        if seconds>=self.medium_at:
            keys=('energy','fluidity','expansion','stillness','velocity','acceleration','periodicity','symmetry')
            values={key:sum(getattr(f,key) for f in self.window)/len(self.window) for key in keys}
            self.context=Features(**values,tracked=features.tracked)
            self.bpm=68+20*self.context.energy+4*self.context.periodicity
            self.medium_at=seconds+1.
        if self.next_frame is None:
            self.next_frame=frame+512
        # If inference stalls, advance musical time without bursting old notes.
        interval=RATE*15/self.bpm
        if self.next_frame<frame:
            missed=int((frame-self.next_frame)//interval)+1
            self.next_frame+=missed*interval
            self.tick+=missed
            self.skipped_ticks+=missed
        if not features.tracked:
            # Preserve phrase position but never generate new notes from an occluded person.
            while self.next_frame<=frame+lookahead:
                self.next_frame+=interval
                self.tick+=1
            return events
        while self.next_frame<=frame+lookahead:
            target=round(self.next_frame)
            slot=self.tick%16
            phrase=(self.tick//32)%len(PROGRESSION)
            if self.phrase_number!=self.tick//32:
                self.phrase_number=self.tick//32
                if self.context.energy<.15:
                    self.chord_degree=0  # stillness releases tension to tonic
                elif self.context.energy>.7 and self.context.fluidity<.35:
                    self.chord_degree=4  # abrupt energetic phrase leans toward v
                else:
                    self.chord_degree=PROGRESSION[phrase]
            # Slow: 2-bar chord and an 8-note motif. Expansion raises register;
            # asymmetry introduces constrained scale-neighbour variation.
            if self.tick%32==0:
                for degree in (self.chord_degree,self.chord_degree+2,self.chord_degree+4):
                    note=degree_note(degree)
                    events.extend((Event(0,0,note,.5,target),Event(0,1,note,0.,target+int(RATE*1.7))))
            stride=4 if self.context.energy<.55 else 2
            if slot%stride==0 and (slot%4==0 or self.random.random()<self.context.energy):
                motif=MOTIF[(self.tick//stride)%len(MOTIF)]
                variation=self.random.choice((-1,0,1)) if self.context.symmetry<.5 else 0
                note=degree_note(self.chord_degree+motif+variation,60 if self.context.expansion>.65 else 48)
                velocity=.35+.35*self.context.energy
                swing=int(interval*.12*(1-self.context.fluidity)) if (self.tick//stride)%2 else 0
                events.extend((Event(1,0,note,velocity,target+swing),Event(1,1,note,0.,target+swing+int(RATE*.4))))
            # Medium: quiet periodic kick/hat. Stillness removes percussion.
            if self.context.energy>.25 and slot%8==0:
                events.append(Event(2,0,36,.25+.25*self.context.energy,target))
            if self.context.energy>.6 and slot%4==2:
                events.append(Event(3,0,42,.2,target))
            self.tick+=1
            self.next_frame+=interval
        # Abrupt expressive accents are rate-limited and use a chord tone.
        if features.acceleration>35 and features.energy>.4 and seconds-self.last_accent>.3:
            events.append(Event(1,0,degree_note(self.chord_degree+4,60),.55,frame+256))
            self.last_accent=seconds
        return events
