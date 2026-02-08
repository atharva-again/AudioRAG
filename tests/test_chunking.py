"""Comprehensive unit tests for chunk_transcription function."""

from audiorag.chunking import chunk_transcription
from audiorag.core.models import ChunkMetadata, TranscriptionSegment


class TestChunkingBasic:
    """Test basic grouping by duration."""

    def test_single_segment_under_duration(self):
        """Test single segment that doesn't reach chunk duration."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=2.0, text="Hello world"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 2.0

    def test_multiple_segments_single_chunk(self):
        """Test multiple segments grouped into single chunk."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=2.0, text="Hello"),
            TranscriptionSegment(start_time=2.0, end_time=4.0, text="world"),
            TranscriptionSegment(start_time=4.0, end_time=6.0, text="test"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world test"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 6.0

    def test_multiple_segments_multiple_chunks(self):
        """Test segments split across multiple chunks."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=3.0, text="First"),
            TranscriptionSegment(start_time=3.0, end_time=6.0, text="second"),
            TranscriptionSegment(start_time=6.0, end_time=9.0, text="third"),
            TranscriptionSegment(start_time=9.0, end_time=12.0, text="fourth"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=6,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 2
        assert chunks[0].text == "First second"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 6.0
        assert chunks[1].text == "third fourth"
        assert chunks[1].start_time == 6.0
        assert chunks[1].end_time == 12.0

    def test_exact_duration_boundary(self):
        """Test segment that exactly hits chunk duration."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="Exactly"),
            TranscriptionSegment(start_time=5.0, end_time=10.0, text="five"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Exactly five"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 10.0

    def test_exceeds_duration_boundary(self):
        """Test segment that exceeds chunk duration."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="First"),
            TranscriptionSegment(start_time=5.0, end_time=11.0, text="second"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "First second"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 11.0


class TestChunkingEmptySegments:
    """Test filtering empty and whitespace segments."""

    def test_empty_text_segment_filtered(self):
        """Test that segments with empty text are included but create extra spaces."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=2.0, text="Hello"),
            TranscriptionSegment(start_time=2.0, end_time=4.0, text=""),
            TranscriptionSegment(start_time=4.0, end_time=6.0, text="world"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        # Empty segments create extra spaces when joined
        assert chunks[0].text == "Hello  world"

    def test_whitespace_only_segment_filtered(self):
        """Test that segments with only whitespace are included but create extra spaces."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=2.0, text="Hello"),
            TranscriptionSegment(start_time=2.0, end_time=4.0, text="   "),
            TranscriptionSegment(start_time=4.0, end_time=6.0, text="world"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        # Whitespace segments create extra spaces when joined
        assert chunks[0].text == "Hello     world"

    def test_all_empty_segments_returns_empty_list(self):
        """Test that all empty segments result in empty chunk list."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=2.0, text=""),
            TranscriptionSegment(start_time=2.0, end_time=4.0, text="  "),
            TranscriptionSegment(start_time=4.0, end_time=6.0, text="\t\n"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 0

    def test_mixed_empty_and_content_chunks(self):
        """Test multiple chunks with some containing empty segments."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=3.0, text="First"),
            TranscriptionSegment(start_time=3.0, end_time=6.0, text=""),
            TranscriptionSegment(start_time=6.0, end_time=9.0, text="second"),
            TranscriptionSegment(start_time=9.0, end_time=12.0, text="   "),
            TranscriptionSegment(start_time=12.0, end_time=15.0, text="third"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=6,
            source_url="https://example.com/video",
            title="Test Video",
        )
        # Chunks are created when duration boundary is hit
        # Chunk 1: 0-6 (First + empty)
        # Chunk 2: 6-12 (second + whitespace)
        # Chunk 3: 12-15 (third)
        assert len(chunks) == 3
        assert chunks[0].text == "First"
        assert chunks[1].text == "second"
        assert chunks[2].text == "third"


class TestChunkingNoOverlap:
    """Test that chunks don't overlap."""

    def test_chunks_have_sequential_boundaries(self):
        """Test that chunk end_time matches next chunk start_time."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=3.0, text="A"),
            TranscriptionSegment(start_time=3.0, end_time=6.0, text="B"),
            TranscriptionSegment(start_time=6.0, end_time=9.0, text="C"),
            TranscriptionSegment(start_time=9.0, end_time=12.0, text="D"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=6,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 2
        assert chunks[0].end_time == chunks[1].start_time

    def test_no_time_gaps_between_chunks(self):
        """Test that there are no gaps between consecutive chunks."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=2.5, text="Segment1"),
            TranscriptionSegment(start_time=2.5, end_time=5.0, text="Segment2"),
            TranscriptionSegment(start_time=5.0, end_time=7.5, text="Segment3"),
            TranscriptionSegment(start_time=7.5, end_time=10.0, text="Segment4"),
            TranscriptionSegment(start_time=10.0, end_time=12.5, text="Segment5"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=5,
            source_url="https://example.com/video",
            title="Test Video",
        )
        for i in range(len(chunks) - 1):
            assert chunks[i].end_time == chunks[i + 1].start_time

    def test_chunks_cover_full_timeline(self):
        """Test that chunks cover the entire timeline without gaps."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="A"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="B"),
            TranscriptionSegment(start_time=2.0, end_time=3.0, text="C"),
            TranscriptionSegment(start_time=3.0, end_time=4.0, text="D"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=2,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert chunks[0].start_time == 0.0
        assert chunks[-1].end_time == 4.0
        for i in range(len(chunks) - 1):
            assert chunks[i].end_time == chunks[i + 1].start_time


class TestChunkingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input_list(self):
        """Test with empty segment list."""
        chunks = chunk_transcription(
            segments=[],
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 0

    def test_single_segment(self):
        """Test with single segment."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="Single"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Single"

    def test_very_small_chunk_duration(self):
        """Test with very small chunk duration."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=0.5, text="A"),
            TranscriptionSegment(start_time=0.5, end_time=1.0, text="B"),
            TranscriptionSegment(start_time=1.0, end_time=1.5, text="C"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=0.5,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 3
        assert chunks[0].text == "A"
        assert chunks[1].text == "B"
        assert chunks[2].text == "C"

    def test_very_large_chunk_duration(self):
        """Test with very large chunk duration."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=10.0, text="A"),
            TranscriptionSegment(start_time=10.0, end_time=20.0, text="B"),
            TranscriptionSegment(start_time=20.0, end_time=30.0, text="C"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=1000,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "A B C"

    def test_single_segment_exact_boundary(self):
        """Test single segment that exactly matches chunk duration."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=10.0, text="Exact"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Exact"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 10.0

    def test_fractional_timestamps(self):
        """Test with fractional timestamps."""
        segments = [
            TranscriptionSegment(start_time=0.123, end_time=2.456, text="Frac1"),
            TranscriptionSegment(start_time=2.456, end_time=5.789, text="Frac2"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=5,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 1
        assert chunks[0].start_time == 0.123
        assert chunks[0].end_time == 5.789

    def test_large_number_of_segments(self):
        """Test with large number of segments."""
        segments = [
            TranscriptionSegment(
                start_time=float(i),
                end_time=float(i + 1),
                text=f"Segment{i}",
            )
            for i in range(100)
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/video",
            title="Test Video",
        )
        assert len(chunks) == 10
        assert chunks[0].start_time == 0.0
        assert chunks[-1].end_time == 100.0


class TestChunkingMetadata:
    """Test that metadata is properly preserved."""

    def test_source_url_preserved(self):
        """Test that source_url is preserved in all chunks."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=3.0, text="A"),
            TranscriptionSegment(start_time=3.0, end_time=6.0, text="B"),
            TranscriptionSegment(start_time=6.0, end_time=9.0, text="C"),
        ]
        source_url = "https://youtube.com/watch?v=abc123"
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=6,
            source_url=source_url,
            title="Test Video",
        )
        for chunk in chunks:
            assert chunk.source_url == source_url

    def test_title_preserved(self):
        """Test that title is preserved in all chunks."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=3.0, text="A"),
            TranscriptionSegment(start_time=3.0, end_time=6.0, text="B"),
            TranscriptionSegment(start_time=6.0, end_time=9.0, text="C"),
        ]
        expected_title = "My Amazing Video"
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=6,
            source_url="https://example.com/video",
            title=expected_title,
        )
        for chunk in chunks:
            assert chunk.title == expected_title

    def test_metadata_with_special_characters(self):
        """Test metadata with special characters."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="Content"),
        ]
        source_url = "https://example.com/video?id=123&lang=en"
        expected_title = "Video: Part 1 & 2 (Special Edition)"
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url=source_url,
            title=expected_title,
        )
        assert chunks[0].source_url == source_url
        assert chunks[0].title == expected_title

    def test_metadata_with_unicode(self):
        """Test metadata with unicode characters."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="Content"),
        ]
        source_url = "https://example.com/video"
        expected_title = "Video Title"
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url=source_url,
            title=expected_title,
        )
        assert chunks[0].title == expected_title

    def test_empty_metadata_strings(self):
        """Test with empty metadata strings."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="Content"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="",
            title="",
        )
        assert chunks[0].source_url == ""
        assert chunks[0].title == ""

    def test_chunk_metadata_type(self):
        """Test that returned objects are ChunkMetadata instances."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="Test"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChunkMetadata)


class TestChunkingTextJoining:
    """Test text joining and formatting."""

    def test_segments_joined_with_space(self):
        """Test that segments are joined with single space."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="Hello"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="world"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].text == "Hello world"

    def test_multiple_segments_joined(self):
        """Test joining multiple segments."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="The"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="quick"),
            TranscriptionSegment(start_time=2.0, end_time=3.0, text="brown"),
            TranscriptionSegment(start_time=3.0, end_time=4.0, text="fox"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].text == "The quick brown fox"

    def test_text_with_punctuation(self):
        """Test text containing punctuation."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="Hello,"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="world!"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].text == "Hello, world!"

    def test_text_with_special_characters(self):
        """Test text with special characters."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="Hello@#$"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="world%^&"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].text == "Hello@#$ world%^&"

    def test_text_with_unicode(self):
        """Test text with unicode characters."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="hello"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="world"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].text == "hello world"

    def test_text_with_emoji(self):
        """Test text with emoji."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="Hello"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="world"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].text == "Hello world"


class TestChunkingTimestamps:
    """Test timestamp preservation and accuracy."""

    def test_chunk_start_time_from_first_segment(self):
        """Test that chunk start_time comes from first segment."""
        segments = [
            TranscriptionSegment(start_time=5.0, end_time=7.0, text="A"),
            TranscriptionSegment(start_time=7.0, end_time=9.0, text="B"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].start_time == 5.0

    def test_chunk_end_time_from_last_segment(self):
        """Test that chunk end_time comes from last segment."""
        segments = [
            TranscriptionSegment(start_time=5.0, end_time=7.0, text="A"),
            TranscriptionSegment(start_time=7.0, end_time=12.0, text="B"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].end_time == 12.0

    def test_timestamps_preserved_across_chunks(self):
        """Test that timestamps are accurately preserved."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=2.5, text="A"),
            TranscriptionSegment(start_time=2.5, end_time=5.0, text="B"),
            TranscriptionSegment(start_time=5.0, end_time=7.5, text="C"),
            TranscriptionSegment(start_time=7.5, end_time=10.0, text="D"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=5,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 5.0
        assert chunks[1].start_time == 5.0
        assert chunks[1].end_time == 10.0

    def test_fractional_timestamp_accuracy(self):
        """Test accuracy with fractional timestamps."""
        segments = [
            TranscriptionSegment(start_time=0.123, end_time=2.456, text="A"),
            TranscriptionSegment(start_time=2.456, end_time=4.789, text="B"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com",
            title="Title",
        )
        assert chunks[0].start_time == 0.123
        assert chunks[0].end_time == 4.789


class TestChunkingComplexScenarios:
    """Test complex real-world scenarios."""

    def test_realistic_transcription_scenario(self):
        """Test realistic transcription with varied segment lengths."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.5, text="Welcome"),
            TranscriptionSegment(start_time=1.5, end_time=3.2, text="to"),
            TranscriptionSegment(start_time=3.2, end_time=5.8, text="the"),
            TranscriptionSegment(start_time=5.8, end_time=8.1, text="podcast"),
            TranscriptionSegment(start_time=8.1, end_time=10.5, text="episode"),
            TranscriptionSegment(start_time=10.5, end_time=12.3, text="today"),
            TranscriptionSegment(start_time=12.3, end_time=14.7, text="we"),
            TranscriptionSegment(start_time=14.7, end_time=17.2, text="discuss"),
            TranscriptionSegment(start_time=17.2, end_time=19.8, text="important"),
            TranscriptionSegment(start_time=19.8, end_time=22.1, text="topics"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=10,
            source_url="https://example.com/podcast",
            title="Podcast Episode 1",
        )
        # Chunk 1: 0-10.5 (10.5 seconds, hits boundary)
        # Chunk 2: 10.5-22.1 (remaining segments)
        assert len(chunks) == 2
        assert chunks[0].start_time == 0.0
        assert chunks[-1].end_time == 22.1

    def test_chunks_with_varying_durations(self):
        """Test that chunks can have varying durations."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="A"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text="B"),
            TranscriptionSegment(start_time=2.0, end_time=3.0, text="C"),
            TranscriptionSegment(start_time=3.0, end_time=8.0, text="D"),
            TranscriptionSegment(start_time=8.0, end_time=9.0, text="E"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=5,
            source_url="https://example.com",
            title="Title",
        )
        # First chunk: 0-8 (8 seconds, exceeds 5 second boundary at segment D)
        # Second chunk: 8-9 (remaining)
        assert len(chunks) == 2
        assert chunks[0].end_time == 8.0
        assert chunks[1].start_time == 8.0

    def test_many_small_segments_into_chunks(self):
        """Test many small segments grouped into chunks."""
        segments = [
            TranscriptionSegment(
                start_time=float(i * 0.5),
                end_time=float((i + 1) * 0.5),
                text=f"Word{i}",
            )
            for i in range(40)
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=5,
            source_url="https://example.com",
            title="Title",
        )
        assert len(chunks) == 4
        assert chunks[0].start_time == 0.0
        assert chunks[-1].end_time == 20.0

    def test_alternating_empty_and_content_segments(self):
        """Test alternating empty and content segments."""
        segments = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="A"),
            TranscriptionSegment(start_time=1.0, end_time=2.0, text=""),
            TranscriptionSegment(start_time=2.0, end_time=3.0, text="B"),
            TranscriptionSegment(start_time=3.0, end_time=4.0, text=""),
            TranscriptionSegment(start_time=4.0, end_time=5.0, text="C"),
            TranscriptionSegment(start_time=5.0, end_time=6.0, text=""),
            TranscriptionSegment(start_time=6.0, end_time=7.0, text="D"),
        ]
        chunks = chunk_transcription(
            segments=segments,
            chunk_duration_seconds=5,
            source_url="https://example.com",
            title="Title",
        )
        assert len(chunks) == 2
        # Empty segments create extra spaces when joined
        assert chunks[0].text == "A  B  C"
        assert chunks[1].text == "D"
