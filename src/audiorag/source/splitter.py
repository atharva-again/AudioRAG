"""Audio file splitter using pydub."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from audiorag.core.logging_config import get_logger

if TYPE_CHECKING:
    import structlog

logger = get_logger(__name__)


class AudioSplitter:
    """Splits audio files into smaller chunks if they exceed size limit."""

    def __init__(self, max_size_mb: float = 25.0) -> None:
        """Initialize AudioSplitter.

        Args:
            max_size_mb: Maximum file size in MB before splitting (default: 25.0)
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._logger = logger.bind(
            provider="audio_splitter",
            max_size_mb=max_size_mb,
        )

    async def split_if_needed(self, audio_path: Path, output_dir: Path | None = None) -> list[Path]:
        """Split audio file if it exceeds size limit.

        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save split files (default: same as input)

        Returns:
            List of audio file paths (original if no split needed, or split chunks)

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        operation_logger = self._logger.bind(
            audio_path=str(audio_path),
            operation="split_if_needed",
        )

        if not audio_path.exists():
            operation_logger.error("audio_file_not_found")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = audio_path.stat().st_size
        operation_logger.debug("checking_file_size", file_size_bytes=file_size)

        if file_size <= self.max_size_bytes:
            operation_logger.info("no_split_needed", file_size_bytes=file_size)
            return [audio_path]

        operation_logger.info("splitting_required", file_size_bytes=file_size)
        try:
            return await asyncio.to_thread(
                self._split_sync,
                audio_path,
                output_dir or audio_path.parent,
                operation_logger,
            )
        except Exception as e:
            operation_logger.error(
                "split_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _split_sync(
        self,
        audio_path: Path,
        output_dir: Path,
        operation_logger: structlog.stdlib.BoundLogger,
    ) -> list[Path]:
        """Synchronous split implementation."""
        try:
            from pydub import AudioSegment

            output_dir.mkdir(parents=True, exist_ok=True)

            operation_logger.debug("loading_audio_file")
            audio = AudioSegment.from_file(str(audio_path))

            file_size = audio_path.stat().st_size
            duration_ms = len(audio)

            num_chunks = (file_size // self.max_size_bytes) + 1
            chunk_duration_ms = int((duration_ms / num_chunks) * 0.9)

            operation_logger.debug(
                "calculated_split_params",
                num_chunks=num_chunks,
                chunk_duration_ms=chunk_duration_ms,
                total_duration_ms=duration_ms,
            )

            chunks: list[Path] = []
            stem = audio_path.stem
            suffix = audio_path.suffix

            for i, start_ms in enumerate(range(0, duration_ms, chunk_duration_ms)):
                end_ms = min(start_ms + chunk_duration_ms, duration_ms)
                chunk = audio[start_ms:end_ms]

                chunk_path = output_dir / f"{stem}_part{i + 1:03d}{suffix}"
                chunk.export(str(chunk_path), format=suffix.lstrip("."))
                chunks.append(chunk_path)

            operation_logger.info("split_completed", chunks_count=len(chunks))
            return chunks

        except Exception as e:
            operation_logger.error(
                "split_sync_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Failed to split audio file {audio_path}: {e}") from e
