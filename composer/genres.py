"""Musical genre profiles: harmony, voicings, rhythms, synth presets, and FX."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChordSpec:
    degree: int
    name: str
    # Scale-degree offsets from `degree` (e.g. (0, 2, 4, 6) = 7th chord, (0, 2, 4, 8) = add9).
    voicing: tuple[int, ...]


@dataclass(frozen=True)
class GenreProfile:
    key: str
    label: str
    base_midi: int
    scale: tuple[int, ...]
    progression: tuple[ChordSpec, ...]
    tension_chord: ChordSpec
    motif: tuple[int, ...]
    # Presets for (limb 0 chords, limb 1 lead/arp, limb 2 kick/bass, limb 3 hat/snare)
    # 0=saw, 1=sine, 2=kick, 3=snare, 4=hat, 5=stab, 6=dsp_kick, 7=dsp_snare,
    # 8=warm_unison_pad, 9=fm_warm_keys, 10=supersaw_pluck, 11=deep_sub_bass
    presets: tuple[int, int, int, int]
    bpm_min: float
    bpm_max: float
    swing: float
    cutoff_pad_range: tuple[float, float]
    cutoff_lead_range: tuple[float, float]
    reverb_pad_range: tuple[float, float]
    reverb_lead_range: tuple[float, float]
    delay_send: float
    resonance: float
    sidechain: float
    kick_threshold: float
    hat_threshold: float
    kick_steps: tuple[int, ...]
    hat_steps: tuple[int, ...]
    bass_threshold: float
    bass_steps: tuple[int, ...]
    chord_duration_s: float
    lead_duration_s: float
    lead_stride_low: int
    lead_stride_high: int


GENRES: dict[str, GenreProfile] = {
    'ambient': GenreProfile(
        key='ambient',
        label='Contemporary Ambient',
        base_midi=48,  # C3
        scale=(0, 2, 3, 5, 7, 8, 10),  # C natural minor
        progression=(
            ChordSpec(0, 'Cm9', (0, 2, 4, 8)),
            ChordSpec(5, 'Abmaj7', (0, 2, 4, 6)),
            ChordSpec(3, 'Fm9', (0, 2, 4, 8)),
            ChordSpec(4, 'Gm7', (0, 2, 4, 6)),
        ),
        tension_chord=ChordSpec(4, 'Gm11', (0, 2, 4, 6)),
        motif=(0, 2, 4, 2, 1, 2, 0, 4),
        presets=(8, 9, 2, 4),
        bpm_min=68.0,
        bpm_max=92.0,
        swing=0.10,
        cutoff_pad_range=(550.0, 6200.0),
        cutoff_lead_range=(950.0, 8800.0),
        reverb_pad_range=(0.40, 0.85),
        reverb_lead_range=(0.30, 0.72),
        delay_send=0.36,
        resonance=0.85,
        sidechain=0.15,
        kick_threshold=0.25,
        hat_threshold=0.60,
        kick_steps=(0, 8),
        hat_steps=(2, 6, 10, 14),
        bass_threshold=0.25,
        bass_steps=(0,),
        chord_duration_s=3.2,
        lead_duration_s=0.55,
        lead_stride_low=4,
        lead_stride_high=2,
    ),
    'house': GenreProfile(
        key='house',
        label='Deep Melodic House',
        base_midi=41,  # F2
        scale=(0, 2, 3, 5, 7, 8, 10),  # F natural minor
        progression=(
            ChordSpec(0, 'Fm9', (0, 2, 4, 6)),
            ChordSpec(5, 'Dbmaj7', (0, 2, 4, 6)),
            ChordSpec(6, 'Eb6', (0, 2, 4, 5)),
            ChordSpec(4, 'Cm7', (0, 2, 4, 6)),
        ),
        tension_chord=ChordSpec(4, 'Cm9', (0, 2, 4, 8)),
        motif=(0, 4, 2, 6, 4, 2, 1, 4),
        presets=(8, 10, 2, 4),
        bpm_min=118.0,
        bpm_max=126.0,
        swing=0.06,
        cutoff_pad_range=(450.0, 5800.0),
        cutoff_lead_range=(800.0, 10500.0),
        reverb_pad_range=(0.28, 0.65),
        reverb_lead_range=(0.22, 0.55),
        delay_send=0.40,
        resonance=1.25,
        sidechain=0.55,
        kick_threshold=0.18,
        hat_threshold=0.38,
        kick_steps=(0, 4, 8, 12),  # 4-on-the-floor
        hat_steps=(2, 6, 10, 14),  # off-beat house hat
        bass_threshold=0.22,
        bass_steps=(0, 3, 6, 10, 14),
        chord_duration_s=2.4,
        lead_duration_s=0.28,
        lead_stride_low=4,
        lead_stride_high=2,
    ),
    'lofi': GenreProfile(
        key='lofi',
        label='Lo-Fi Chill',
        base_midi=48,  # C3
        scale=(0, 2, 3, 5, 7, 9, 10),  # C Dorian (warm jazzy 6th)
        progression=(
            ChordSpec(0, 'Cm9', (0, 2, 4, 6)),
            ChordSpec(3, 'F13', (0, 2, 4, 6)),
            ChordSpec(6, 'Bbmaj7', (0, 2, 4, 6)),
            ChordSpec(2, 'Ebmaj9', (0, 2, 4, 8)),
        ),
        tension_chord=ChordSpec(4, 'Gm9', (0, 2, 4, 8)),
        motif=(0, 2, 5, 4, 2, 3, 1, 0),
        presets=(9, 9, 2, 3),
        bpm_min=76.0,
        bpm_max=88.0,
        swing=0.20,
        cutoff_pad_range=(600.0, 4200.0),
        cutoff_lead_range=(900.0, 5200.0),
        reverb_pad_range=(0.24, 0.58),
        reverb_lead_range=(0.20, 0.48),
        delay_send=0.22,
        resonance=0.78,
        sidechain=0.22,
        kick_threshold=0.20,
        hat_threshold=0.42,
        kick_steps=(0, 7, 10),  # laid-back boom-bap kick
        hat_steps=(4, 12),      # snare backbeat on limb 3
        bass_threshold=0.20,
        bass_steps=(0, 10),
        chord_duration_s=2.8,
        lead_duration_s=0.45,
        lead_stride_low=4,
        lead_stride_high=2,
    ),
    'cinematic': GenreProfile(
        key='cinematic',
        label='Cinematic Pulse',
        base_midi=50,  # D3
        scale=(0, 2, 3, 5, 7, 8, 10),  # D natural minor
        progression=(
            ChordSpec(0, 'Dm9', (0, 2, 4, 8)),
            ChordSpec(5, 'Bbmaj7', (0, 2, 4, 6)),
            ChordSpec(2, 'Fmaj7', (0, 2, 4, 6)),
            ChordSpec(6, 'Csus4', (0, 3, 4, 6)),
        ),
        tension_chord=ChordSpec(4, 'Am9', (0, 2, 4, 8)),
        motif=(0, 4, 7, 4, 2, 4, 6, 4),
        presets=(8, 10, 11, 2),
        bpm_min=92.0,
        bpm_max=112.0,
        swing=0.04,
        cutoff_pad_range=(500.0, 7500.0),
        cutoff_lead_range=(1000.0, 11000.0),
        reverb_pad_range=(0.50, 0.92),
        reverb_lead_range=(0.35, 0.78),
        delay_send=0.48,
        resonance=1.10,
        sidechain=0.35,
        kick_threshold=0.28,
        hat_threshold=0.52,
        kick_steps=(0, 6, 12),
        hat_steps=(0, 8),
        bass_threshold=0.18,
        bass_steps=(0, 2, 4, 6, 8, 10, 12, 14),  # pulsing 8th-note sub bass
        chord_duration_s=3.4,
        lead_duration_s=0.32,
        lead_stride_low=2,
        lead_stride_high=1,
    ),
}

GENRE_ORDER = ('ambient', 'house', 'lofi', 'cinematic')


def get_genre(name: str) -> GenreProfile:
    key = (name or 'ambient').strip().lower()
    if key not in GENRES:
        raise ValueError(f'Género desconocido: {name!r}. Opciones: {", ".join(GENRE_ORDER)}')
    return GENRES[key]
