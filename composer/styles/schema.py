"""Typed schema and JSON loader for modular CV-DAW musical styles."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ChordSpec:
    degree: int
    name: str
    voicing: tuple[int, ...]


@dataclass(frozen=True)
class PhraseNote:
    step: int          # 0..15 sixteenth-note slot within a 4/4 bar
    degree_offset: int # scale-degree offset relative to current chord root
    length_steps: int  # duration in 16th-note steps (1..16)


@dataclass(frozen=True)
class DrumPatterns:
    kick_groove: tuple[int, ...]
    kick_full: tuple[int, ...]
    snare: tuple[int, ...]
    hat_groove: tuple[int, ...]
    hat_full: tuple[int, ...]
    open_hat: tuple[int, ...]
    ghost_snare: tuple[int, ...]
    fill_bar4: tuple[tuple[int, int], ...]  # (step, midi_drum_note: 36/38/42/46)


@dataclass(frozen=True)
class StyleBlueprint:
    key: str
    label: str
    order: int
    base_midi: int
    scale: tuple[int, ...]
    bpm_default: float
    bpm_min: float
    bpm_max: float
    swing: float
    presets: tuple[int, int, int, int]  # (chords, lead, bass, drum_kit)
    strum_ms: float
    progression: tuple[ChordSpec, ...]
    bright_chord: ChordSpec
    tension_chord: ChordSpec
    comping_steps: tuple[int, ...]
    chord_duration_s: float
    motif_a: tuple[PhraseNote, ...]
    motif_b: tuple[PhraseNote, ...]
    bass_pattern: tuple[PhraseNote, ...]
    drums: DrumPatterns
    cutoff_pad_range: tuple[float, float]
    cutoff_lead_range: tuple[float, float]
    reverb_pad_range: tuple[float, float]
    reverb_lead_range: tuple[float, float]
    delay_send: float
    resonance: float
    sidechain: float


def _chord(raw: dict) -> ChordSpec:
    return ChordSpec(
        degree=int(raw['degree']),
        name=str(raw['name']),
        voicing=tuple(int(v) for v in raw['voicing']),
    )


def _notes(raw_list: list) -> tuple[PhraseNote, ...]:
    out = []
    for item in raw_list:
        step, deg, length = int(item[0]), int(item[1]), int(item[2])
        if not (0 <= step < 16 and length >= 1):
            raise ValueError(f'Nota de frase inválida: {item}')
        out.append(PhraseNote(step=step, degree_offset=deg, length_steps=length))
    return tuple(out)


def load_style_json(path: Path) -> StyleBlueprint:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    presets = tuple(int(x) for x in data['presets'])
    if len(presets) != 4:
        raise ValueError(f'{path}: presets debe tener exactamente 4 IDs')
    scale = tuple(int(x) for x in data['scale'])
    if len(scale) < 5:
        raise ValueError(f'{path}: scale debe tener al menos 5 grados')
    progression = tuple(_chord(c) for c in data['progression'])
    if not progression:
        raise ValueError(f'{path}: progression vacía')
    d = data['drums']
    drums = DrumPatterns(
        kick_groove=tuple(int(x) for x in d['kick_groove']),
        kick_full=tuple(int(x) for x in d['kick_full']),
        snare=tuple(int(x) for x in d['snare']),
        hat_groove=tuple(int(x) for x in d['hat_groove']),
        hat_full=tuple(int(x) for x in d['hat_full']),
        open_hat=tuple(int(x) for x in d['open_hat']),
        ghost_snare=tuple(int(x) for x in d['ghost_snare']),
        fill_bar4=tuple((int(s), int(n)) for s, n in d['fill_bar4']),
    )
    fx = data['fx']
    return StyleBlueprint(
        key=str(data['key']).strip().lower(),
        label=str(data['label']),
        order=int(data.get('order', 100)),
        base_midi=int(data['base_midi']),
        scale=scale,
        bpm_default=float(data['bpm_default']),
        bpm_min=float(data['bpm_min']),
        bpm_max=float(data['bpm_max']),
        swing=float(data['swing']),
        presets=presets,
        strum_ms=float(data.get('strum_ms', 16.0)),
        progression=progression,
        bright_chord=_chord(data['bright_chord']),
        tension_chord=_chord(data['tension_chord']),
        comping_steps=tuple(int(x) for x in data['comping_steps']),
        chord_duration_s=float(data['chord_duration_s']),
        motif_a=_notes(data['motifs']['a']),
        motif_b=_notes(data['motifs']['b']),
        bass_pattern=_notes(data['bass_pattern']),
        drums=drums,
        cutoff_pad_range=(float(fx['cutoff_pad'][0]), float(fx['cutoff_pad'][1])),
        cutoff_lead_range=(float(fx['cutoff_lead'][0]), float(fx['cutoff_lead'][1])),
        reverb_pad_range=(float(fx['reverb_pad'][0]), float(fx['reverb_pad'][1])),
        reverb_lead_range=(float(fx['reverb_lead'][0]), float(fx['reverb_lead'][1])),
        delay_send=float(fx['delay_send']),
        resonance=float(fx['resonance']),
        sidechain=float(fx['sidechain']),
    )
