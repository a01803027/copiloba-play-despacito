"""Dynamic StyleRegistry: auto-discovers modular JSON style presets."""
from pathlib import Path
from .schema import ChordSpec, DrumPatterns, PhraseNote, StyleBlueprint, load_style_json

PRESETS_DIR = Path(__file__).resolve().parent / 'presets'


class StyleRegistry:
    def __init__(self, presets_dir: Path = PRESETS_DIR):
        self.presets_dir = Path(presets_dir)
        self._styles: dict[str, StyleBlueprint] = {}
        self.reload()

    def reload(self):
        loaded: list[StyleBlueprint] = []
        if self.presets_dir.is_dir():
            for path in sorted(self.presets_dir.glob('*.json')):
                loaded.append(load_style_json(path))
        loaded.sort(key=lambda s: (s.order, s.key))
        self._styles = {style.key: style for style in loaded}

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._styles.keys())

    @property
    def styles(self) -> dict[str, StyleBlueprint]:
        return dict(self._styles)

    def get(self, name: str) -> StyleBlueprint:
        key = (name or 'ambient').strip().lower()
        if key not in self._styles:
            raise ValueError(f'Estilo desconocido: {name!r}. Disponibles: {", ".join(self.keys)}')
        return self._styles[key]

    def next_key(self, current: str, step: int = 1) -> str:
        order = self.keys
        if not order:
            return current
        idx = order.index(current) if current in order else 0
        return order[(idx + step) % len(order)]


REGISTRY = StyleRegistry()
STYLES = REGISTRY.styles
STYLE_ORDER = REGISTRY.keys


def get_style(name: str) -> StyleBlueprint:
    return REGISTRY.get(name)


__all__ = [
    'ChordSpec',
    'DrumPatterns',
    'PhraseNote',
    'StyleBlueprint',
    'StyleRegistry',
    'REGISTRY',
    'STYLES',
    'STYLE_ORDER',
    'get_style',
    'load_style_json',
]
