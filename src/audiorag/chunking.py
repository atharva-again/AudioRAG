"""Time-based text chunking for transcription segments."""

from audiorag.models import ChunkMetadata, TranscriptionSegment


def chunk_transcription(
    segments: list[TranscriptionSegment],
    chunk_duration_seconds: int | float,
    source_url: str,
    video_title: str,
) -> list[ChunkMetadata]:
    """Group transcription segments into non-overlapping time-based chunks.

    Args:
        segments: List of transcription segments from STT provider.
        chunk_duration_seconds: Target duration for each chunk in seconds.
        source_url: Source URL for all chunks.
        video_title: Video title for all chunks.

    Returns:
        List of ChunkMetadata objects grouped by duration, with empty chunks filtered out.
    """
    if not segments:
        return []

    chunks: list[ChunkMetadata] = []
    current_chunk_segments: list[TranscriptionSegment] = []
    current_chunk_start_time: float | None = None

    for segment in segments:
        # Initialize start time for new chunk
        if current_chunk_start_time is None:
            current_chunk_start_time = segment.start_time

        # Add segment to current chunk
        current_chunk_segments.append(segment)

        # Calculate accumulated duration
        accumulated_duration = segment.end_time - current_chunk_start_time

        # If we've reached or exceeded the chunk duration, finalize the current chunk
        if accumulated_duration >= chunk_duration_seconds:
            # Create chunk from all accumulated segments (including the one
            # that triggered the boundary)
            chunk_segments = current_chunk_segments
            current_chunk_segments = []
            current_chunk_start_time = None

            # Create ChunkMetadata if text is not empty
            chunk_text = " ".join(s.text for s in chunk_segments).strip()
            if chunk_text:
                chunk = ChunkMetadata(
                    start_time=chunk_segments[0].start_time,
                    end_time=chunk_segments[-1].end_time,
                    text=chunk_text,
                    source_url=source_url,
                    video_title=video_title,
                )
                chunks.append(chunk)

    # Handle remaining segments
    if current_chunk_segments:
        chunk_text = " ".join(s.text for s in current_chunk_segments).strip()
        if chunk_text:
            chunk = ChunkMetadata(
                start_time=current_chunk_segments[0].start_time,
                end_time=current_chunk_segments[-1].end_time,
                text=chunk_text,
                source_url=source_url,
                video_title=video_title,
            )
            chunks.append(chunk)

    return chunks
