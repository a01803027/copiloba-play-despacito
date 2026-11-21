"""Backwards-compatible facade over the modular JSON style registry in composer.styles."""
from .styles import (
    ChordSpec,
    REGISTRY,
    STYLE_ORDER as GENRE_ORDER,
    STYLES as GENRES,
    StyleBlueprint as GenreProfile,
    get_style as get_genre,
)

__all__ = ['ChordSpec', 'GenreProfile', 'GENRES', 'GENRE_ORDER', 'REGISTRY', 'get_genre']
