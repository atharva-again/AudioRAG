"""Pydantic data models for AudioRAG."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, computed_field


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""

    start_time: float
    end_time: float
    text: str
    source_url: str
    video_title: str


class Source(BaseModel):
    """A source document with relevance score."""

    text: str
    start_time: float
    end_time: float
    source_url: str
    video_title: str
    relevance_score: float

    @computed_field
    @property
    def youtube_timestamp_url(self) -> str:
        """Generate YouTube URL with timestamp."""
        return f"{self.source_url}&t={int(self.start_time)}"


class QueryResult(BaseModel):
    """Result of a query with sources."""

    answer: str
    sources: list[Source]


class AudioFile(BaseModel):
    """Audio file metadata."""

    path: Path
    source_url: str
    video_title: str
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
