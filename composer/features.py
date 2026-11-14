"""Pose features in torso lengths; no camera/audio dependencies."""
from collections import deque
from dataclasses import asdict, dataclass
import math


def clamp(x, low=0., high=1.):
    return max(low, min(high, x))


@dataclass(frozen=True)
class Features:
    energy: float = 0.
    fluidity: float = 1.
    expansion: float = .3
    stillness: float = 1.
    velocity: float = 0.
    acceleration: float = 0.
    periodicity: float = 0.
    symmetry: float = 1.
    tracked: bool = True

    def to_dict(self):
        return asdict(self)


class FeatureExtractor:
    def __init__(self):
        self.previous = None
        self.velocity = None
        self.last_time = None
        self.energy = 0.
        self.previous_valid = [False]*4
        self.last_expansion = .3
        self.history = deque(maxlen=90)

    def missing(self):
        self.previous = self.velocity = self.last_time = None
        self.previous_valid = [False]*4
        self.history.clear()
        self.energy = 0.
        return Features(tracked=False)

    def update(self, points, timestamp, aspect=1.):
        # x and z are width-relative in MediaPipe; make y use the same unit.
        if len(points) != 33 or not all(math.isfinite(c) for p in points for c in p[:3]):
            return self.missing()
        xyz = [(p[0], p[1]*aspect, p[2]) for p in points]
        hip = tuple((xyz[23][j]+xyz[24][j])/2 for j in range(3))
        shoulder = tuple((xyz[11][j]+xyz[12][j])/2 for j in range(3))
        scale = math.dist(hip, shoulder)
        if scale < .025:
            return self.missing()
        positions = [tuple((xyz[i][j]-hip[j])/scale for j in range(3)) for i in (15,16,27,28)]
        dt = timestamp-self.last_time if self.last_time is not None else 1/30
        if dt <= 0:
            return Features(energy=self.energy, stillness=1-self.energy)
        if dt > .2:
            self.missing()
        active = [len(points[i])<4 or points[i][3]>=.5 for i in (15,16,27,28)]
        alpha = 1-math.exp(-min(dt,.2)/.04)
        smoothed=[]
        velocities=[]
        for i,new in enumerate(positions):
            if not active[i]:
                new=self.previous[i] if self.previous is not None else (0.,0.,0.)
            elif self.previous is not None and self.previous_valid[i]:
                new=tuple(a+alpha*(b-a) for a,b in zip(self.previous[i],new))
            smoothed.append(new)
            velocities.append(tuple((b-a)/dt for a,b in zip(self.previous[i],new))
                              if self.previous is not None and active[i] and self.previous_valid[i] else (0.,0.,0.))
        positions=smoothed
        count=max(1,sum(active))
        speed = sum(math.sqrt(sum(v*v for v in p)) for p,ok in zip(velocities,active) if ok)/count
        acceleration = 0.
        if self.velocity is not None:
            acceleration = sum(math.dist(a,b)/dt for a,b,ok,was in zip(self.velocity,velocities,active,self.previous_valid) if ok and was)/count
        self.energy += (1-math.exp(-min(dt,.2)/.3))*(clamp(speed/3)-self.energy)
        if active[0] and active[1]:
            self.last_expansion=clamp((math.dist(positions[0],positions[1])-.5)/3)
        expansion=self.last_expansion
        pairs=[i for i in (0,2) if active[i] and active[i+1]]
        error = sum(abs(positions[i][0]+positions[i+1][0])+abs(positions[i][1]-positions[i+1][1]) for i in pairs)/max(1,2*len(pairs))
        self.history.append((timestamp,speed))
        # Resample to 30Hz before comparing lags; variable camera FPS is allowed.
        periodicity = 0.
        if len(self.history)>20 and timestamp-self.history[0][0]>.8:
            values=[]
            history=list(self.history)
            index=0
            for n in range(30):
                t=timestamp-(29-n)/30
                while index+1<len(history) and history[index+1][0]<=t:
                    index+=1
                values.append(history[index][1])
            mean=sum(values)/len(values)
            centered=[v-mean for v in values]
            if sum(v*v for v in centered)>.05:
                for lag in range(8,21):
                    a,b=centered[lag:],centered[:-lag]
                    norm=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
                    if norm>1e-8:
                        periodicity=max(periodicity,sum(x*y for x,y in zip(a,b))/norm)
        self.previous,self.velocity,self.last_time=positions,velocities,timestamp
        self.previous_valid=active
        return Features(self.energy,1/(1+acceleration/20),expansion,1-self.energy,
                        speed,acceleration,clamp(periodicity),clamp(1-error))


def mock_features(seconds):
    # Alternating calm/expansive/abrupt phrases, not random isolated triggers.
    activity=.5+.4*math.sin(seconds*.25)
    energy=clamp(activity*(.7+.3*math.sin(seconds*3)**2))
    return Features(energy,clamp(.7+.25*math.sin(seconds*.4)),
                    clamp(.5+.45*math.sin(seconds*.18)),1-energy,
                    energy*3,energy*15,clamp(.5+.45*math.sin(seconds*.3)),
                    clamp(.7+.25*math.cos(seconds*.15)))
