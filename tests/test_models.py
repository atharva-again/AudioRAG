"""Comprehensive unit tests for Pydantic models in AudioRAG."""

import json
from pathlib import Path

from audiorag.core.models import (
    AudioFile,
    ChunkMetadata,
    IndexingStatus,
    QueryResult,
    Source,
    TranscriptionSegment,
)


class TestChunkMetadata:
    """Test suite for ChunkMetadata model."""

    def test_valid_creation(self):
        """Test creating a valid ChunkMetadata instance."""
        chunk = ChunkMetadata(
            start_time=0.0,
            end_time=5.5,
            text="Sample text",
            source_url="https://youtube.com/watch?v=abc123",
            title="Test Video",
        )
        assert chunk.start_time == 0.0
        assert chunk.end_time == 5.5
        assert chunk.text == "Sample text"
        assert chunk.source_url == "https://youtube.com/watch?v=abc123"
        assert chunk.title == "Test Video"

    def test_field_types_validation(self):
        """Test that field types are properly validated."""
        # Valid float times
        chunk = ChunkMetadata(
            start_time=1.5,
            end_time=10.75,
            text="Text",
            source_url="https://example.com",
            title="Title",
        )
        assert isinstance(chunk.start_time, float)
        assert isinstance(chunk.end_time, float)

    def test_string_fields_required(self):
        """Test that string fields are required and accept empty strings."""
        # Empty strings are valid in Pydantic
        chunk = ChunkMetadata(
            start_time=0.0,
            end_time=5.0,
            text="",
            source_url="",
            title="",
        )
        assert chunk.text == ""
        assert chunk.source_url == ""
        assert chunk.title == ""

    def test_negative_times(self):
        """Test that negative times are accepted (edge case)."""
        chunk = ChunkMetadata(
            start_time=-1.0,
            end_time=5.0,
            text="Text",
            source_url="https://example.com",
            title="Title",
        )
        assert chunk.start_time == -1.0

    def test_zero_duration(self):
        """Test chunk with zero duration (start_time == end_time)."""
        chunk = ChunkMetadata(
            start_time=5.0,
            end_time=5.0,
            text="Instant text",
            source_url="https://example.com",
            title="Title",
        )
        assert chunk.start_time == chunk.end_time

    def test_serialization(self):
        """Test model serialization to dict."""
        chunk = ChunkMetadata(
            start_time=1.0,
            end_time=3.0,
            text="Test",
            source_url="https://example.com",
            title="Title",
        )
        data = chunk.model_dump()
        assert data["start_time"] == 1.0
        assert data["end_time"] == 3.0
        assert data["text"] == "Test"

    def test_json_serialization(self):
        """Test model serialization to JSON."""
        chunk = ChunkMetadata(
            start_time=1.0,
            end_time=3.0,
            text="Test",
            source_url="https://example.com",
            title="Title",
        )
        json_str = chunk.model_dump_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["start_time"] == 1.0

    def test_deserialization(self):
        """Test model deserialization from dict."""
        data = {
            "start_time": 2.5,
            "end_time": 7.5,
            "text": "Deserialized",
            "source_url": "https://example.com",
            "title": "Title",
        }
        chunk = ChunkMetadata(**data)  # type: ignore
        assert chunk.start_time == 2.5
        assert chunk.text == "Deserialized"

    def test_large_time_values(self):
        """Test with large time values (long videos)."""
        chunk = ChunkMetadata(
            start_time=3600.0,  # 1 hour
            end_time=7200.0,  # 2 hours
            text="Long video chunk",
            source_url="https://example.com",
            title="Long Video",
        )
        assert chunk.start_time == 3600.0
        assert chunk.end_time == 7200.0


class TestSource:
    """Test suite for Source model."""

    def test_valid_creation(self):
        """Test creating a valid Source instance."""
        source = Source(
            text="Source text",
            start_time=10.0,
            end_time=15.0,
            source_url="https://youtube.com/watch?v=abc123",
            title="Test Video",
            relevance_score=0.95,
        )
        assert source.text == "Source text"
        assert source.start_time == 10.0
        assert source.end_time == 15.0
        assert source.relevance_score == 0.95

    def test_relevance_score_range(self):
        """Test relevance score with various values."""
        # Test with 0.0
        source1 = Source(
            text="Text",
            start_time=0.0,
            end_time=5.0,
            source_url="https://example.com",
            title="Title",
            relevance_score=0.0,
        )
        assert source1.relevance_score == 0.0

        # Test with 1.0
        source2 = Source(
            text="Text",
            start_time=0.0,
            end_time=5.0,
            source_url="https://example.com",
            title="Title",
            relevance_score=1.0,
        )
        assert source2.relevance_score == 1.0

    def test_serialization(self):
        source = Source(
            text="Text",
            start_time=10.0,
            end_time=15.0,
            source_url="https://youtube.com/watch?v=abc123",
            title="Title",
            relevance_score=0.8,
        )
        data = source.model_dump()
        assert data["title"] == "Title"
        assert data["relevance_score"] == 0.8

    def test_json_serialization(self):
        source = Source(
            text="Text",
            start_time=20.0,
            end_time=25.0,
            source_url="https://youtube.com/watch?v=xyz789",
            title="Title",
            relevance_score=0.75,
        )
        json_str = source.model_dump_json()
        data = json.loads(json_str)
        assert data["title"] == "Title"

    def test_deserialization(self):
        """Test model deserialization from dict."""
        data = {
            "text": "Deserialized text",
            "start_time": 5.0,
            "end_time": 10.0,
            "source_url": "https://example.com",
            "title": "Title",
            "relevance_score": 0.85,
        }
        source = Source(**data)  # type: ignore
        assert source.text == "Deserialized text"
        assert source.relevance_score == 0.85


class TestQueryResult:
    """Test suite for QueryResult model."""

    def test_valid_creation_with_sources(self):
        """Test creating a QueryResult with sources."""
        sources = [
            Source(
                text="Source 1",
                start_time=0.0,
                end_time=5.0,
                source_url="https://example.com",
                title="Video 1",
                relevance_score=0.9,
            ),
            Source(
                text="Source 2",
                start_time=10.0,
                end_time=15.0,
                source_url="https://example.com",
                title="Video 2",
                relevance_score=0.8,
            ),
        ]
        result = QueryResult(answer="Test answer", sources=sources)
        assert result.answer == "Test answer"
        assert len(result.sources) == 2
        assert result.sources[0].relevance_score == 0.9

    def test_empty_sources_list(self):
        """Test QueryResult with empty sources list."""
        result = QueryResult(answer="Answer with no sources", sources=[])
        assert result.answer == "Answer with no sources"
        assert len(result.sources) == 0

    def test_single_source(self):
        """Test QueryResult with single source."""
        source = Source(
            text="Single source",
            start_time=0.0,
            end_time=5.0,
            source_url="https://example.com",
            title="Video",
            relevance_score=0.95,
        )
        result = QueryResult(answer="Answer", sources=[source])
        assert len(result.sources) == 1
        assert result.sources[0].text == "Single source"

    def test_multiple_sources(self):
        """Test QueryResult with multiple sources."""
        sources = [
            Source(
                text=f"Source {i}",
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                source_url="https://example.com",
                title=f"Video {i}",
                relevance_score=0.9 - (i * 0.1),
            )
            for i in range(5)
        ]
        result = QueryResult(answer="Multi-source answer", sources=sources)
        assert len(result.sources) == 5

    def test_serialization(self):
        """Test QueryResult serialization."""
        source = Source(
            text="Text",
            start_time=0.0,
            end_time=5.0,
            source_url="https://example.com",
            title="Video",
            relevance_score=0.8,
        )
        result = QueryResult(answer="Answer", sources=[source])
        data = result.model_dump()
        assert data["answer"] == "Answer"
        assert len(data["sources"]) == 1

    def test_json_serialization(self):
        """Test QueryResult JSON serialization."""
        source = Source(
            text="Text",
            start_time=0.0,
            end_time=5.0,
            source_url="https://example.com",
            title="Video",
            relevance_score=0.8,
        )
        result = QueryResult(answer="Answer", sources=[source])
        json_str = result.model_dump_json()
        data = json.loads(json_str)
        assert data["answer"] == "Answer"
        assert len(data["sources"]) == 1

    def test_deserialization(self):
        """Test QueryResult deserialization."""
        data = {
            "answer": "Deserialized answer",
            "sources": [
                {
                    "text": "Source text",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "source_url": "https://example.com",
                    "title": "Video",
                    "relevance_score": 0.85,
                }
            ],
        }
        result = QueryResult(**data)  # type: ignore
        assert result.answer == "Deserialized answer"
        assert len(result.sources) == 1


class TestAudioFile:
    """Test suite for AudioFile model."""

    def test_valid_creation_with_path(self):
        """Test creating AudioFile with Path object."""
        path = Path("/tmp/audio.mp3")
        audio = AudioFile(
            path=path,
            source_url="https://example.com",
            title="Test Video",
        )
        assert audio.path == path
        assert isinstance(audio.path, Path)
        assert audio.source_url == "https://example.com"
        assert audio.title == "Test Video"

    def test_path_from_string(self):
        """Test creating AudioFile with string path (should convert to Path)."""
        audio = AudioFile(
            path="/tmp/audio.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
        )
        assert isinstance(audio.path, Path)
        assert str(audio.path) == "/tmp/audio.mp3"

    def test_duration_optional(self):
        """Test that duration is optional."""
        audio = AudioFile(
            path="/tmp/audio.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
        )
        assert audio.duration is None

    def test_duration_with_value(self):
        """Test AudioFile with duration specified."""
        audio = AudioFile(
            path="/tmp/audio.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
            duration=120.5,
        )
        assert audio.duration == 120.5

    def test_duration_zero(self):
        """Test AudioFile with zero duration."""
        audio = AudioFile(
            path="/tmp/audio.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
            duration=0.0,
        )
        assert audio.duration == 0.0

    def test_duration_large_value(self):
        """Test AudioFile with large duration (long video)."""
        audio = AudioFile(
            path="/tmp/audio.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
            duration=86400.0,  # 24 hours
        )
        assert audio.duration == 86400.0

    def test_relative_path(self):
        """Test AudioFile with relative path."""
        audio = AudioFile(
            path="./audio/file.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
        )
        assert isinstance(audio.path, Path)
        assert str(audio.path) == "audio/file.mp3"

    def test_path_with_spaces(self):
        """Test AudioFile with path containing spaces."""
        audio = AudioFile(
            path="/tmp/my audio file.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
        )
        assert isinstance(audio.path, Path)
        assert "my audio file" in str(audio.path)

    def test_serialization(self):
        """Test AudioFile serialization."""
        audio = AudioFile(
            path="/tmp/audio.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
            duration=100.0,
        )
        data = audio.model_dump()
        assert str(data["path"]) == "/tmp/audio.mp3"
        assert data["duration"] == 100.0

    def test_json_serialization(self):
        """Test AudioFile JSON serialization."""
        audio = AudioFile(
            path="/tmp/audio.mp3",  # type: ignore
            source_url="https://example.com",
            title="Test Video",
            duration=100.0,
        )
        json_str = audio.model_dump_json()
        data = json.loads(json_str)
        assert data["path"] == "/tmp/audio.mp3"

    def test_deserialization(self):
        """Test AudioFile deserialization."""
        data = {
            "path": "/tmp/audio.mp3",
            "source_url": "https://example.com",
            "title": "Test Video",
            "duration": 150.0,
        }
        audio = AudioFile(**data)  # type: ignore
        assert isinstance(audio.path, Path)
        assert audio.duration == 150.0


class TestTranscriptionSegment:
    """Test suite for TranscriptionSegment model."""

    def test_valid_creation(self):
        """Test creating a valid TranscriptionSegment."""
        segment = TranscriptionSegment(
            start_time=0.0,
            end_time=5.0,
            text="Transcribed text",
        )
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.text == "Transcribed text"

    def test_field_types(self):
        """Test that field types are correct."""
        segment = TranscriptionSegment(
            start_time=1.5,
            end_time=10.75,
            text="Text",
        )
        assert isinstance(segment.start_time, float)
        assert isinstance(segment.end_time, float)
        assert isinstance(segment.text, str)

    def test_empty_text(self):
        """Test TranscriptionSegment with empty text."""
        segment = TranscriptionSegment(
            start_time=0.0,
            end_time=5.0,
            text="",
        )
        assert segment.text == ""

    def test_long_text(self):
        """Test TranscriptionSegment with long text."""
        long_text = "A" * 10000
        segment = TranscriptionSegment(
            start_time=0.0,
            end_time=5.0,
            text=long_text,
        )
        assert len(segment.text) == 10000

    def test_negative_times(self):
        """Test TranscriptionSegment with negative times."""
        segment = TranscriptionSegment(
            start_time=-1.0,
            end_time=5.0,
            text="Text",
        )
        assert segment.start_time == -1.0

    def test_zero_duration(self):
        """Test TranscriptionSegment with zero duration."""
        segment = TranscriptionSegment(
            start_time=5.0,
            end_time=5.0,
            text="Instant text",
        )
        assert segment.start_time == segment.end_time

    def test_special_characters_in_text(self):
        """Test TranscriptionSegment with special characters."""
        text = "Hello! @#$%^&*() hello"
        segment = TranscriptionSegment(
            start_time=0.0,
            end_time=5.0,
            text=text,
        )
        assert segment.text == text

    def test_multiline_text(self):
        """Test TranscriptionSegment with multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        segment = TranscriptionSegment(
            start_time=0.0,
            end_time=5.0,
            text=text,
        )
        assert "\n" in segment.text

    def test_serialization(self):
        """Test TranscriptionSegment serialization."""
        segment = TranscriptionSegment(
            start_time=1.0,
            end_time=3.0,
            text="Test",
        )
        data = segment.model_dump()
        assert data["start_time"] == 1.0
        assert data["end_time"] == 3.0
        assert data["text"] == "Test"

    def test_json_serialization(self):
        """Test TranscriptionSegment JSON serialization."""
        segment = TranscriptionSegment(
            start_time=1.0,
            end_time=3.0,
            text="Test",
        )
        json_str = segment.model_dump_json()
        data = json.loads(json_str)
        assert data["start_time"] == 1.0

    def test_deserialization(self):
        """Test TranscriptionSegment deserialization."""
        data = {
            "start_time": 2.5,
            "end_time": 7.5,
            "text": "Deserialized",
        }
        segment = TranscriptionSegment(**data)  # type: ignore
        assert segment.start_time == 2.5
        assert segment.text == "Deserialized"


class TestIndexingStatus:
    """Test suite for IndexingStatus enum."""

    def test_enum_values_exist(self):
        """Test that all expected enum values exist."""
        assert IndexingStatus.DOWNLOADING.value == "downloading"
        assert IndexingStatus.DOWNLOADED.value == "downloaded"
        assert IndexingStatus.SPLITTING.value == "splitting"
        assert IndexingStatus.TRANSCRIBING.value == "transcribing"
        assert IndexingStatus.TRANSCRIBED.value == "transcribed"
        assert IndexingStatus.CHUNKING.value == "chunking"
        assert IndexingStatus.CHUNKED.value == "chunked"
        assert IndexingStatus.EMBEDDING.value == "embedding"
        assert IndexingStatus.EMBEDDED.value == "embedded"
        assert IndexingStatus.COMPLETED.value == "completed"
        assert IndexingStatus.FAILED.value == "failed"

    def test_enum_count(self):
        """Test that all expected enum members exist."""
        expected_count = 11
        assert len(IndexingStatus) == expected_count

    def test_enum_string_comparison(self):
        """Test comparing enum values with strings."""
        status = IndexingStatus.DOWNLOADING
        assert status == "downloading"
        assert status.value == "downloading"

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        status = IndexingStatus("downloading")
        assert status == IndexingStatus.DOWNLOADING

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        statuses = list(IndexingStatus)
        assert len(statuses) == 11
        assert IndexingStatus.DOWNLOADING in statuses
        assert IndexingStatus.COMPLETED in statuses

    def test_enum_in_model(self):
        """Test using enum in a Pydantic model context."""
        # Create a simple model with enum field
        from pydantic import BaseModel

        class StatusModel(BaseModel):
            status: IndexingStatus

        model = StatusModel(status=IndexingStatus.DOWNLOADING)
        assert model.status == IndexingStatus.DOWNLOADING

    def test_enum_serialization(self):
        """Test enum serialization."""
        from pydantic import BaseModel

        class StatusModel(BaseModel):
            status: IndexingStatus

        model = StatusModel(status=IndexingStatus.COMPLETED)
        data = model.model_dump()
        assert data["status"] == "completed"

    def test_enum_json_serialization(self):
        """Test enum JSON serialization."""
        from pydantic import BaseModel

        class StatusModel(BaseModel):
            status: IndexingStatus

        model = StatusModel(status=IndexingStatus.FAILED)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["status"] == "failed"

    def test_enum_deserialization(self):
        """Test enum deserialization from string."""
        from pydantic import BaseModel

        class StatusModel(BaseModel):
            status: IndexingStatus

        data = {"status": "embedding"}
        model = StatusModel(**data)  # type: ignore
        assert model.status == IndexingStatus.EMBEDDING

    def test_all_status_transitions(self):
        """Test that all status values are valid."""
        valid_statuses = [
            "downloading",
            "downloaded",
            "splitting",
            "transcribing",
            "transcribed",
            "chunking",
            "chunked",
            "embedding",
            "embedded",
            "completed",
            "failed",
        ]
        for status_str in valid_statuses:
            status = IndexingStatus(status_str)
            assert status.value == status_str

    def test_enum_string_type(self):
        """Test that IndexingStatus is a StrEnum."""
        # StrEnum members should be comparable to strings
        assert IndexingStatus.DOWNLOADING == "downloading"
        assert IndexingStatus.DOWNLOADING == "downloading"
