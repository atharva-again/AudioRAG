"""Pydantic data models for AudioRAG."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""

    start_time: float
    end_time: float
    text: str
    source_url: str
    title: str
    metadata: dict[str, Any] = {}


class Source(BaseModel):
    """A source document with relevance score."""

    text: str
    start_time: float
    end_time: float
    source_url: str
    title: str
    relevance_score: float
    metadata: dict[str, Any] = {}


class QueryResult(BaseModel):
    """Result of a query with sources."""

    answer: str
    sources: list[Source]


class BatchIndexFailure(BaseModel):
    """Failure details for one source in a batch indexing run."""

    source_url: str
    stage: str
    error_message: str


class BatchIndexResult(BaseModel):
    """Structured outcome for a batch indexing run."""

    inputs: list[str] = Field(default_factory=list)
    discovered_sources: list[str] = Field(default_factory=list)
    indexed_sources: list[str] = Field(default_factory=list)
    skipped_sources: list[str] = Field(default_factory=list)
    failures: list[BatchIndexFailure] = Field(default_factory=list)


class SourceMetadata(BaseModel):
    """Pre-flight metadata from a source provider."""

    duration: float | None = None
    title: str | None = None
    raw: Any | None = Field(default=None, exclude=True)


class AudioFile(BaseModel):
    """Audio file metadata."""

    path: Path
    source_url: str
    title: str
    duration: float | None = None


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio."""

    start_time: float
    end_time: float
    text: str


class IndexingStatus(StrEnum):
    """Pipeline stages for audio indexing."""

    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    SPLITTING = "splitting"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    CHUNKING = "chunking"
    CHUNKED = "chunked"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    COMPLETED = "completed"
    FAILED = "failed"
