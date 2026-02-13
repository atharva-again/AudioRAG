"""Audio file splitter using ffmpeg."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from audiorag.core.exceptions import ProviderError
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

    def _ensure_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg"):
            raise ProviderError(
                message=(
                    "audio_splitter requires ffmpeg but it's not installed or not in PATH. "
                    "Please install ffmpeg to use audio splitting."
                ),
                provider="audio_splitter",
                retryable=False,
            )

    async def split_if_needed(self, audio_path: Path, output_dir: Path | None = None) -> list[Path]:
        """Split audio file if it exceeds size limit.

        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save split files (default: same as input)

        Returns:
            List of audio file paths (original if no split needed, or split chunks)

        Raises:
            ProviderError: If audio file doesn't exist or splitting fails
        """
        self._ensure_ffmpeg()

        operation_logger = self._logger.bind(
            audio_path=str(audio_path),
            operation="split_if_needed",
        )

        if not audio_path.exists():
            operation_logger.error("audio_file_not_found")
            raise ProviderError(
                message=f"audio_splitter split_if_needed failed: file not found: {audio_path}",
                provider="audio_splitter",
                retryable=False,
            )

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
        """Synchronous split implementation using ffmpeg."""
        output_dir.mkdir(parents=True, exist_ok=True)

        file_size = audio_path.stat().st_size
        num_chunks = (file_size // self.max_size_bytes) + 1

        stem = audio_path.stem
        suffix = audio_path.suffix.lstrip(".")

        chunks: list[Path] = []

        ffprobe_path = shutil.which("ffprobe")
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffprobe_path or not ffmpeg_path:
            raise ProviderError(
                message=("audio_splitter requires ffmpeg but it's not installed or not in PATH."),
                provider="audio_splitter",
                retryable=False,
            )

        try:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            total_duration = float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            raise ProviderError(
                message=f"audio_splitter: failed to get audio duration: {e}",
                provider="audio_splitter",
                retryable=False,
            ) from e

        chunk_duration = total_duration / num_chunks

        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = output_dir / f"{stem}_part{i + 1:03d}.{suffix}"

            try:
                subprocess.run(
                    [
                        ffmpeg_path,
                        "-y",
                        "-i",
                        str(audio_path),
                        "-ss",
                        str(start_time),
                        "-t",
                        str(chunk_duration),
                        "-c",
                        "copy",
                        str(chunk_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                chunks.append(chunk_path)
            except subprocess.CalledProcessError as e:
                raise ProviderError(
                    message=f"audio_splitter: ffmpeg failed to create chunk {i + 1}: {e.stderr}",
                    provider="audio_splitter",
                    retryable=False,
                ) from e

        operation_logger.info("split_completed", chunks_count=len(chunks))
        return chunks
