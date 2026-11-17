"""Four-limb instrument, gestures, and beat-aligned audio commands."""
from dataclasses import dataclass
from enum import IntEnum
import math

from feature_extractors import Posture, TemporalFeatures

SAMPLE_RATE = 48_000
RECORD_PREVIOUS_GRID = 1
SCALES = {
    'minor-pentatonic': (60, 63, 65, 67, 70, 72, 75, 79, 82),
    'dorian': (60, 62, 63, 65, 67, 69, 70, 72, 74),
    'phrygian-dominant': (60, 61, 64, 65, 67, 68, 70, 72, 73),
    'major-pentatonic': (60, 62, 64, 67, 69, 72, 74, 76, 79),
}


class EventType(IntEnum):
    NOTE_ON = 0
    MUTE_TOGGLE = 2
    LOOP_LOCK = 3
    PRESET_CHANGE = 4


class PlayState(IntEnum):
    LIVE = 0
    MUTED = 1
    LOOPING = 2


@dataclass(frozen=True)
class AudioCommand:
    limb: int
    kind: EventType
    note: int = 0
    value: float = 0.
    frame: int = 0
    flags: int = 0


@dataclass(frozen=True)
class ControlFrame:
    cutoff: tuple[float, ...]
    reverb: tuple[float, ...]
    bend: tuple[float, ...]
    gain: tuple[float, ...]
    stem_volumes: tuple[float, float, float, float]
    bpm: float
    energy: float


class LimbStateMachine:
    def __init__(self, stationary_leg_modifier=False):
        self.play_state = [PlayState.LIVE] * 4
        self.active_preset_id = [0, 1, 7, 6]
        self.last_posture = [None, None]
        self.stationary_leg_modifier = stationary_leg_modifier

    def initial_presets(self):
        return [AudioCommand(i, EventType.PRESET_CHANGE, preset)
                for i, preset in enumerate(self.active_preset_id)]

    def update(self, limbs):
        commands = []
        for hand in range(2):
            posture = limbs[hand].posture
            if posture is None or posture == self.last_posture[hand]:
                continue
            self.last_posture[hand] = posture
            targets = [hand]
            leg = limbs[hand + 2]
            if self.stationary_leg_modifier and leg.valid and leg.speed < .3:
                targets.append(hand + 2)
            for limb in targets:
                state = self.play_state[limb]
                if posture == Posture.PEACE:
                    preset = (self.active_preset_id[limb] + 1) % 8
                    self.active_preset_id[limb] = preset
                    commands.append(AudioCommand(limb, EventType.PRESET_CHANGE, preset))
                elif posture == Posture.FIST:
                    if state == PlayState.LOOPING:
                        commands.append(AudioCommand(limb, EventType.LOOP_LOCK))
                    commands.append(AudioCommand(limb, EventType.MUTE_TOGGLE))
                    self.play_state[limb] = PlayState.LIVE if state == PlayState.MUTED else PlayState.MUTED
                elif posture == Posture.PINCH:
                    if state == PlayState.MUTED:
                        commands.append(AudioCommand(limb, EventType.MUTE_TOGGLE))
                    locked = state != PlayState.LOOPING
                    commands.append(AudioCommand(limb, EventType.LOOP_LOCK, value=float(locked)))
                    self.play_state[limb] = PlayState.LOOPING if locked else PlayState.LIVE
        return commands


class ScaleQuantizer:
    def __init__(self, name='minor-pentatonic'):
        self.notes = SCALES[name]

    def note(self, limb, height):
        if limb >= 2:
            return (38, 36)[limb - 2]
        position = min(1., max(0., (height + .3) / 1.5))
        return self.notes[min(len(self.notes) - 1, int(position * len(self.notes)))]


class HybridBeatQuantizer:
    def __init__(self, bpm=120.):
        self.sync(0, bpm)

    def sync(self, sample_frame, bpm, grid_snapshot=0):
        self.frame = sample_frame
        self.period = SAMPLE_RATE * 15 / bpm
        if grid_snapshot:
            self.upcoming = grid_snapshot >> 16
            self.period = grid_snapshot & 0xffff
        else:
            self.upcoming = math.ceil(sample_frame / self.period) * self.period

    def quantize(self, onset_frame=None):
        onset = self.frame if onset_frame is None else onset_frame
        nearest = self.upcoming + round((onset - self.upcoming) / self.period) * self.period
        delta = (onset - nearest) * 1000 / SAMPLE_RATE
        if 0 <= delta <= 35:
            return 0, RECORD_PREVIOUS_GRID
        if -40 <= delta < 0 and nearest > self.frame:
            return int(round(nearest)), 0
        return 0, 0


def stem_arrangement(temporal):
    bounce, flow, staccato, idle = temporal.groove
    energy = temporal.energy
    return (
        min(1., .30 + .35 * bounce + .25 * staccato + .25 * energy),
        min(1., .40 + .30 * bounce + .20 * flow + .15 * energy),
        min(1., .45 + .35 * flow + .16 * idle),
        min(1., .40 + .28 * flow + .20 * idle + .12 * energy),
    )


class MusicalBrain:
    def __init__(self, *, scale='minor-pentatonic', bpm=120., follow_bounce=False,
                 stationary_leg_modifier=False):
        self.states = LimbStateMachine(stationary_leg_modifier)
        self.scale = ScaleQuantizer(scale)
        self.quantizer = HybridBeatQuantizer(bpm)
        self.bpm = bpm
        self.follow_bounce = follow_bounce
        self.started = False

    def update(self, limbs, temporal: TemporalFeatures, sample_frame, grid_snapshot=0):
        if self.follow_bounce and temporal.bpm_confidence >= .7:
            self.bpm = temporal.target_bpm
        self.quantizer.sync(sample_frame, self.bpm, grid_snapshot)
        commands = [] if self.started else self.states.initial_presets()
        self.started = True
        commands.extend(self.states.update(limbs))
        for limb in limbs:
            if limb.valid and limb.hit and self.states.play_state[limb.limb] == PlayState.LIVE:
                target, flags = self.quantizer.quantize()
                commands.append(AudioCommand(limb.limb, EventType.NOTE_ON,
                                             self.scale.note(limb.limb, limb.height),
                                             limb.velocity, target, flags))
        controls = ControlFrame(
            tuple(180 * (65 ** limb.cutoff) for limb in limbs),
            tuple(.6 * limb.reverb for limb in limbs),
            tuple(4 * (limb.bend - .5) for limb in limbs),
            tuple(float(limb.valid or self.states.play_state[limb.limb] == PlayState.LOOPING)
                  for limb in limbs),
            stem_arrangement(temporal), self.bpm, temporal.energy,
        )
        return controls, commands
