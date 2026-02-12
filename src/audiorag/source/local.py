"""Local file audio source provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from audiorag.core.exceptions import ProviderError
from audiorag.core.logging_config import get_logger
from audiorag.core.models import AudioFile, SourceMetadata

if TYPE_CHECKING:
    import structlog

logger = get_logger(__name__)


class LocalSource:
    """Audio source for local files and directories.

    Supports:
    - Single audio files
    - Directories of audio files
    - Automatic duration detection using pydub
    """

    async def get_metadata(self, url: str) -> Any:
        """Fetch metadata for a local file."""
        return None

    async def download(
        self,
        source_path: str,
        output_dir: Path,
        audio_format: str = "mp3",
        metadata: SourceMetadata | None = None,
    ) -> AudioFile:
        """Get audio file info from local path.

        Args:
            source_path: Path to audio file or directory
            output_dir: Output directory (not used for local files)
            audio_format: Target format (not used for local files)
            metadata: Not used for local source.

        Returns:
            AudioFile with metadata
        """
        operation_logger = self._logger.bind(
            source_path=source_path,
            operation="download",
        )
        operation_logger.info("local_source_started")

        path = Path(source_path)

        if not path.exists():
            operation_logger.error("path_not_found")
            raise ProviderError(
                message=f"local_source download failed: path not found: {source_path}",
                provider="local_source",
                retryable=False,
            )

        if path.is_dir():
            operation_logger.error("is_directory", error="Use AudioSplitter for directories")
            raise ProviderError(
                message=f"local_source download failed: path is a directory: {source_path}",
                provider="local_source",
                retryable=False,
            )

        # Derive title from filename
        title = path.stem.replace("_", " ").replace("-", " ").strip()

        # Get duration using pydub
        duration = await self._get_duration(path, operation_logger)

        return AudioFile(
            path=path,
            source_url=f"file://{path.absolute()}",
            title=title,
            duration=duration,
        )

    async def _get_duration(
        self, audio_path: Path, operation_logger: structlog.stdlib.BoundLogger
    ) -> float | None:
        """Get audio duration in seconds."""
        try:
            from pydub import AudioSegment

            def _get_sync() -> float:
                audio = AudioSegment.from_file(str(audio_path))
                return len(audio) / 1000.0

            return await asyncio.to_thread(_get_sync)
        except Exception as e:
            operation_logger.warning("duration_detection_failed", error=str(e))
            return None
