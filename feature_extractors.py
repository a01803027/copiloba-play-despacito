"""Independent fast kinematics and windowed motion context."""
from dataclasses import dataclass
from enum import IntEnum
import math
from pathlib import Path

import numpy as np

from perception_layer import PerceptionFrame


class Posture(IntEnum):
    OPEN = 0
    FIST = 1
    PINCH = 2
    PEACE = 3


JOINTS = np.array([(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28)])
ENDPOINTS = JOINTS[:, 2]


@dataclass(frozen=True)
class LimbFeatures:
    limb: int
    valid: bool = False
    height: float = 0.
    speed: float = 0.
    hit: bool = False
    velocity: float = .1
    cutoff: float = .5
    reverb: float = 0.
    bend: float = .5
    posture: Posture | None = None
    velocity_xyz: tuple[float, float, float] = (0., 0., 0.)
    acceleration_xyz: tuple[float, float, float] = (0., 0., 0.)


@dataclass(frozen=True)
class TemporalFeatures:
    energy: float = 0.
    groove: tuple[float, float, float, float] = (0., 0., 0., 1.)
    target_bpm: float = 120.
    bpm_confidence: float = 0.


class HandPostureClassifier:
    def __init__(self):
        self.current = self.candidate = None
        self.count = 0

    @staticmethod
    def classify(hand):
        scale = np.linalg.norm(hand[9] - hand[0])
        if scale < 1e-4:
            return None
        points = (hand - hand[0]) / scale
        tips = np.linalg.norm(points[[8, 12, 16, 20]], axis=1)
        pips = np.linalg.norm(points[[6, 10, 14, 18]], axis=1)
        extended = (tips > 1.15 * pips).tolist()
        if all(extended[1:]) and np.linalg.norm(points[4] - points[8]) < .25:
            return Posture.PINCH
        if extended == [True, True, False, False]:
            return Posture.PEACE
        if not any(extended):
            return Posture.FIST
        return Posture.OPEN if all(extended) else None

    def update(self, hand):
        value = self.classify(hand) if hand is not None else None
        if value is None:
            self.candidate, self.count = None, 0
            return None
        self.count = self.count + 1 if value == self.candidate else 1
        self.candidate = value
        if self.count >= 3:
            self.current = value
        return self.current


class FastKinematicsExtractor:
    def __init__(self, min_speed=1.8, deceleration=18., cooldown=.08):
        self.min_speed, self.deceleration, self.cooldown = min_speed, deceleration, cooldown
        self.timestamp = None
        self.position = np.zeros((4, 3))
        self.velocity = np.zeros((4, 3))
        self.valid = np.zeros(4, dtype=bool)
        self.last_hit = np.full(4, -np.inf)
        self.peak = np.zeros(4)
        self.braking = np.zeros(4, dtype=bool)
        self.hands = [HandPostureClassifier(), HandPostureClassifier()]

    def update(self, frame: PerceptionFrame):
        dt = frame.timestamp - self.timestamp if self.timestamp is not None else math.inf
        if dt <= 0:
            return tuple(LimbFeatures(i) for i in range(4))
        self.timestamp = frame.timestamp
        postures = [classifier.update(hand) for classifier, hand in zip(
            self.hands, (frame.filtered_left_hand if frame.filtered_left_hand is not None else frame.left_hand,
                         frame.filtered_right_hand if frame.filtered_right_hand is not None else frame.right_hand))]
        pose = frame.normalized_pose
        visible = frame.pose_landmarks[:, 3] >= .5 if pose is not None else np.zeros(33, dtype=bool)
        valid = visible[ENDPOINTS]
        positions = pose[ENDPOINTS] if pose is not None else np.zeros_like(self.position)
        continuous = valid & self.valid & (dt <= .2)
        velocity = np.zeros_like(self.velocity)
        acceleration = np.zeros_like(self.velocity)
        velocity[continuous] = (positions[continuous] - self.position[continuous]) / dt
        acceleration[continuous] = (velocity[continuous] - self.velocity[continuous]) / dt
        speed = np.linalg.norm(velocity, axis=1)
        previous_speed = np.linalg.norm(self.velocity, axis=1)
        projection = np.sum(acceleration * self.velocity, axis=1) / np.maximum(previous_speed, 1e-9)
        braking = continuous & (projection < -self.deceleration)
        hit = braking & ~self.braking & (previous_speed > self.min_speed) & (
            frame.timestamp - self.last_hit >= self.cooldown)
        peak = np.maximum(self.peak, previous_speed)
        self.last_hit[hit] = frame.timestamp
        self.peak = np.where(continuous & ~braking, np.maximum(peak, speed), speed)
        result = []
        for i, (root, joint, tip) in enumerate(JOINTS):
            posture = postures[i] if i < 2 else None
            if not valid[i]:
                result.append(LimbFeatures(i, posture=posture))
                continue
            height = -float(positions[i, 1])
            radius = math.hypot(positions[i, 0], positions[i, 2])
            bend = .5
            if visible[root] and visible[joint]:
                a, b = pose[root] - pose[joint], pose[tip] - pose[joint]
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                if denominator > 1e-7:
                    bend = 1 - math.acos(float(np.clip(a @ b / denominator, -1, 1))) / math.pi
            result.append(LimbFeatures(i, True, height, float(speed[i]), bool(hit[i]),
                                       min(1., max(.1, float(peak[i]) / 8)),
                                       min(1., max(0., (height + .3) / 1.5)),
                                       min(1., radius / 1.5), bend, posture,
                                       tuple(velocity[i]), tuple(acceleration[i])))
        self.position, self.velocity, self.valid, self.braking = positions, velocity, valid, braking
        return tuple(result)


class HipBouncePLL:
    """Second-order bandpass and correlation-based period smoothing at 60 Hz."""
    def __init__(self):
        self.ring = np.zeros(180)
        self.count = 0
        self.bpm, self.confidence = 120., 0.
        self.state = np.zeros(2)
        self.origin = None
        low, high = (math.tan(math.pi * hz / 60) for hz in (1.33, 2.66))
        bandwidth = high - low
        center2 = low * high
        norm = 1 + bandwidth + center2
        self.b0 = bandwidth / norm
        self.a1 = 2 * (center2 - 1) / norm
        self.a2 = (1 - bandwidth + center2) / norm

    def update(self, y):
        if self.origin is None:
            self.origin = y
        x = y - self.origin
        filtered = self.b0 * x + self.state[0]
        self.state[0] = self.state[1] - self.a1 * filtered
        self.state[1] = -self.b0 * x - self.a2 * filtered
        self.ring[self.count % len(self.ring)] = filtered
        self.count += 1

    def estimate(self):
        self.confidence = 0.
        if self.count < len(self.ring):
            return self.bpm, self.confidence
        y = np.roll(self.ring, -(self.count % len(self.ring)))
        y -= y.mean()
        if np.std(y) < .002:
            return self.bpm, self.confidence
        scores = []
        for lag in range(21, 47):
            a, b = y[lag:], y[:-lag]
            scores.append(float(a @ b / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))))
        peaks = [i for i in range(1, len(scores) - 1)
                 if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]]
        if peaks:
            strongest = max(scores[i] for i in peaks)
            best = next(i for i in peaks if scores[i] >= strongest - .05)
            score = scores[best]
            if score >= .7:
                left, right = scores[best - 1], scores[best + 1]
                curvature = left + right - 2 * score
                offset = .5 * (left - right) / curvature if curvature < -1e-9 else 0.
                candidate = float(np.clip(3600 / (21 + best + offset), 80, 160))
                self.bpm += .25 * (candidate - self.bpm)
                self.confidence = score
        return self.bpm, self.confidence


class GrooveModel:
    def __init__(self, path, delegate=None):
        path = Path(path)
        if path.suffix == '.onnx':
            import onnxruntime as ort
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            name = session.get_inputs()[0].name
            self.predict = lambda x: session.run(None, {name: x})[0]
        elif path.suffix == '.tflite':
            from tflite_model import TFLiteModel
            try:
                model = TFLiteModel(path, delegate)
            except Exception:
                if delegate is None:
                    raise
                import logging
                logging.exception('Falló el delegado de groove. Se usará TFLite en CPU.')
                model = TFLiteModel(path)
            if len(model.outputs) != 1:
                raise ValueError('El modelo de groove debe tener un tensor de salida')
            self.predict = lambda x: model.invoke(x)[0]
        else:
            raise ValueError('El modelo de groove debe ser .tflite u .onnx')

    def logits(self, values):
        result = np.asarray(self.predict(values[None].astype(np.float32))).reshape(-1)
        if result.shape != (4,) or not np.isfinite(result).all():
            raise ValueError('El modelo debe entregar cuatro logits finitos')
        return result


class SlowTemporalExtractor:
    def __init__(self, model=None, delegate=None):
        self.model = GrooveModel(model, delegate) if model and Path(model).is_file() else None
        self.ring = np.zeros((30, 36), dtype=np.float32)
        self.hip = HipBouncePLL()
        self.timestamp = self.position = self.last_hip = self.visible = None
        self.samples = 0
        self.result = TemporalFeatures()

    def update(self, frame: PerceptionFrame):
        if frame.normalized_pose is None or frame.hip_y is None:
            if self.position is not None:
                self.hip = HipBouncePLL()
                self.samples = 0
                self.result = TemporalFeatures(target_bpm=self.result.target_bpm)
            self.timestamp = self.position = self.visible = None
            return self.result
        indices = JOINTS.reshape(-1)
        points = frame.normalized_pose[indices]
        visibility = frame.pose_landmarks[indices, 3] >= .5
        t = frame.timestamp
        if self.timestamp is not None and t <= self.timestamp:
            return self.result
        if self.timestamp is None or t - self.timestamp > .2:
            self.hip = HipBouncePLL()
            self.samples = 0
            self.next_tick = t + 1 / 60
            self.result = TemporalFeatures(target_bpm=self.result.target_bpm)
        else:
            dt = t - self.timestamp
            velocity = (points - self.position) / dt
            velocity[~(visibility & self.visible)] = 0
            while self.next_tick <= t + 1e-9:
                fraction = min(1., (self.next_tick - self.timestamp) / dt)
                self.hip.update(self.last_hip + fraction * (frame.hip_y - self.last_hip))
                self.ring[self.samples % len(self.ring)] = velocity.reshape(-1)
                self.samples += 1
                self.next_tick += 1 / 60
                if self.samples >= 30 and self.samples % 6 == 0:
                    self.result = self.analyze()
        self.timestamp, self.position, self.last_hip, self.visible = t, points, frame.hip_y, visibility
        return self.result

    def analyze(self):
        data = np.roll(self.ring, -(self.samples % len(self.ring)), axis=0)
        energy = min(1., float(np.mean(data ** 2)) / 3)
        variation = min(1., float(np.mean(np.diff(data, axis=0) ** 2)))
        bpm, confidence = self.hip.estimate()
        logits = (self.model.logits(data) if self.model else
                  np.array([2 * confidence + energy, 2 * energy * (1 - variation),
                            3 * variation, 3 * (1 - energy) * (1 - variation)]))
        weights = np.exp(logits - np.max(logits))
        return TemporalFeatures(energy, tuple(weights / weights.sum()), bpm, confidence)


if __name__ == '__main__':
    import sys
    import unittest
    sys.argv[1:] = ['discover', '-s', 'tests', '-p', 'test_feature_extractors.py']
    unittest.main(module=None)
