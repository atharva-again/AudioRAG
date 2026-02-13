"""HTTP URL audio source provider."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from audiorag.core.exceptions import ProviderError
from audiorag.core.logging_config import get_logger
from audiorag.core.models import AudioFile, SourceMetadata

if TYPE_CHECKING:
    import structlog

logger = get_logger(__name__)

# Supported audio extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".webm"}


class URLSource:
    """Audio source for direct HTTP URL downloads.

    Downloads audio from direct URLs. Use YouTubeSource for YouTube videos.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(provider="url_source")

    async def get_metadata(self, url: str) -> SourceMetadata | None:
        """Fetch metadata for a direct URL."""
        return None

    async def download(
        self,
        url: str,
        output_dir: Path,
        audio_format: str = "mp3",
        metadata: SourceMetadata | None = None,
    ) -> AudioFile:
        """Download audio from HTTP URL.

        Args:
            url: Direct URL to audio file
            output_dir: Directory to save downloaded file
            audio_format: Target audio format for conversion
            metadata: Not used for URL source.

        Returns:
            AudioFile with metadata
        """
        operation_logger = self._logger.bind(
            source_url=url,
            output_dir=str(output_dir),
            audio_format=audio_format,
            operation="download",
        )
        operation_logger.info("url_download_started")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename from URL or content-disposition
        filename = await self._get_filename(url, operation_logger)
        if not filename:
            filename = f"audio_{hash(url) % 1000000:06d}.{audio_format}"
        else:
            # Use the extension from the audio_format, not the original
            stem = Path(filename).stem
            filename = f"{stem}.{audio_format}"

        output_path = output_dir / filename

        # Download the file
        content = await self._download_file(url, operation_logger)

        # Write to output file
        await asyncio.to_thread(output_path.write_bytes, content)

        # Get duration using ffprobe
        duration = await self._get_duration(output_path, operation_logger)

        # Derive title from filename
        title = Path(filename).stem.replace("_", " ").replace("-", " ").strip()

        return AudioFile(
            path=output_path,
            source_url=url,
            title=title,
            duration=duration,
        )

    async def _get_filename(
        self, url: str, operation_logger: structlog.stdlib.BoundLogger
    ) -> str | None:
        """Extract filename from URL or Content-Disposition header."""
        try:
            import aiohttp  # type: ignore[import]

            async with aiohttp.ClientSession() as session:
                async with session.head(url, allow_redirects=True) as response:
                    # Try Content-Disposition header first
                    content_disposition = response.headers.get("Content-Disposition")
                    if content_disposition:
                        match = re.search(r'filename="?([^";\n]+)', content_disposition)
                        if match:
                            return match.group(1).strip()

        except Exception as e:
            operation_logger.warning("filename_detection_failed", error=str(e))

        # Fall back to URL path
        try:
            from urllib.parse import unquote

            path = unquote(url.split("?", maxsplit=1)[0])
            return Path(path).name
        except Exception:
            return None

    async def _download_file(
        self, url: str, operation_logger: structlog.stdlib.BoundLogger
    ) -> bytes:
        """Download file content."""
        try:
            import aiohttp  # type: ignore[import]

            async with aiohttp.ClientSession() as session, session.get(url) as response:
                if response.status >= 400:
                    raise ProviderError(
                        message=f"url_source download failed: HTTP {response.status}: {url}",
                        provider="url_source",
                        retryable=500 <= response.status < 600,
                    )
                return await response.read()
        except Exception as e:
            operation_logger.error("download_failed", error=str(e))
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                message=f"url_source download failed: {e}",
                provider="url_source",
                retryable=isinstance(e, (ConnectionError, TimeoutError)),
            ) from e

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
