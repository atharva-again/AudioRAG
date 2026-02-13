"""Local file audio source provider."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

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
    - Automatic duration detection using ffprobe
    """

    def __init__(self) -> None:
        self._logger = logger.bind(provider="local_source")

    async def get_metadata(self, url: str) -> SourceMetadata:
        """Fetch metadata for a local file."""
        path_str = url.replace("file://", "") if url.startswith("file://") else url
        path = Path(path_str)
        file_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
        title = path.stem.replace("_", " ").replace("-", " ").strip()
        return SourceMetadata(
            title=title,
            raw={"id": file_id, "filepath": str(path.resolve())},
        )

    async def download(
        self,
        url: str,
        output_dir: Path,
        audio_format: str = "mp3",
        metadata: SourceMetadata | None = None,
    ) -> AudioFile:
        """Get audio file info from local path.

        Args:
            url: Path to audio file (file:// URL or local path)
            output_dir: Output directory (not used for local files)
            audio_format: Target format (not used for local files)
            metadata: Not used for local source.

        Returns:
            AudioFile with metadata
        """
        operation_logger = self._logger.bind(
            source_path=url,
            operation="download",
        )
        operation_logger.info("local_source_started")

        path = Path(url)

        if not path.exists():
            operation_logger.error("path_not_found")
            raise ProviderError(
                message=f"local_source download failed: path not found: {url}",
                provider="local_source",
                retryable=False,
            )

        if path.is_dir():
            operation_logger.error("is_directory", error="Use AudioSplitter for directories")
            raise ProviderError(
                message=f"local_source download failed: path is a directory: {url}",
                provider="local_source",
                retryable=False,
            )

        # Derive title from filename
        title = path.stem.replace("_", " ").replace("-", " ").strip()

        # Get duration using ffprobe
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
        """Get audio duration in seconds using ffprobe."""
        import shutil
        import subprocess

        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            operation_logger.warning("duration_detection_failed", error="ffprobe not found")
            return None

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
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            operation_logger.warning("duration_detection_failed", error=str(e))
            return None
