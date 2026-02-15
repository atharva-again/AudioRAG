"""Audio source providers."""

from __future__ import annotations

from audiorag.source.discovery import discover_sources
from audiorag.source.local import LocalSource

__all__ = [
    "LocalSource",
    "discover_sources",
]
