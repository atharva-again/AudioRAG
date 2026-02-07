"""Chunking strategy protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from audiorag.core.models import ChunkMetadata, TranscriptionSegment


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(
        self,
        segments: list[TranscriptionSegment],
        source_url: str,
        title: str,
    ) -> list[ChunkMetadata]: ...
