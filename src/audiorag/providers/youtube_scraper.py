"""YouTube audio scraper using yt-dlp."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yt_dlp  # type: ignore

from audiorag.logging_config import get_logger
from audiorag.models import AudioFile
from audiorag.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

if TYPE_CHECKING:
    import structlog

logger = get_logger(__name__)


class YouTubeScraper:
    """Downloads audio from YouTube videos using yt-dlp."""

    def __init__(self, retry_config: RetryConfig | None = None) -> None:
        """Initialize YouTubeScraper and validate FFmpeg availability.

        Args:
            retry_config: Retry configuration. Uses default if not provided.
        """
        self._logger = logger.bind(provider="youtube_scraper")
        self._retry_config = retry_config or RetryConfig()

        if not shutil.which("ffmpeg"):
            self._logger.error("ffmpeg_not_found")
            raise RuntimeError(
                "FFmpeg is not installed or not in PATH. "
                "Please install FFmpeg to use YouTubeScraper."
            )
        self._logger.debug("ffmpeg_validated")

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for YouTube download operations."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(ConnectionError, TimeoutError, RuntimeError),
        )

    async def download(self, url: str, output_dir: Path, audio_format: str = "mp3") -> AudioFile:
        """Download audio from YouTube video.

        Args:
            url: YouTube video URL
            output_dir: Directory to save the audio file
            audio_format: Audio format (default: mp3)

        Returns:
            AudioFile with metadata

        Raises:
            RuntimeError: If download fails
        """
        operation_logger = self._logger.bind(
            url=url,
            output_dir=str(output_dir),
            audio_format=audio_format,
            operation="download",
        )
        operation_logger.info("download_started")

        output_dir.mkdir(parents=True, exist_ok=True)

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _download_with_retry() -> Any:
            return self._download_sync(url, output_dir, audio_format, operation_logger)

        try:
            result = await asyncio.to_thread(_download_with_retry)
            operation_logger.info(
                "download_completed",
                video_title=result.video_title,
                duration_seconds=result.duration,
                file_path=str(result.path),
            )
            return result
        except Exception as e:
            operation_logger.error(
                "download_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _download_sync(
        self,
        url: str,
        output_dir: Path,
        audio_format: str,
        operation_logger: structlog.stdlib.BoundLogger,
    ) -> AudioFile:
        """Synchronous download implementation.

        Args:
            url: YouTube video URL
            output_dir: Directory to save the audio file
            audio_format: Audio format
            operation_logger: Logger instance with context

        Returns:
            AudioFile with metadata

        Raises:
            RuntimeError: If download fails
        """
        output_template = str(output_dir / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_format,
                }
            ],
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                operation_logger.debug("extracting_video_info")
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError(f"Failed to extract video info from {url}")

                video_id = info["id"]
                video_title = info.get("title", "Unknown")
                duration = info.get("duration")

                audio_path = output_dir / f"{video_id}.{audio_format}"

                if not audio_path.exists():
                    raise RuntimeError(f"Downloaded file not found at {audio_path}")

                return AudioFile(
                    path=audio_path,
                    source_url=url,
                    video_title=video_title,
                    duration=duration,
                )

        except Exception as e:
            operation_logger.error(
                "download_sync_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Failed to download audio from {url}: {e}") from e
