"""YouTube audio scraper using yt-dlp with large-scale channel support."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yt_dlp  # type: ignore

from audiorag.core.logging_config import get_logger
from audiorag.core.models import AudioFile
from audiorag.core.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

if TYPE_CHECKING:
    import structlog

logger = get_logger(__name__)


@dataclass
class VideoInfo:
    """Minimal video information for channel listing."""

    id: str
    title: str
    url: str
    duration: float | None = None
    uploader: str | None = None


class YouTubeScraper:
    """Downloads audio from YouTube videos using yt-dlp.

    Supports large-scale channel scraping with optimizations:
    - Fast channel listing via extract_flat (no full metadata fetch)
    - Download archive to track already processed videos
    - Batch processing for 20k+ video channels
    - Resume capability for interrupted downloads
    - Lazy playlist processing
    """

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        *,
        download_archive: Path | str | None = None,
        concurrent_fragments: int = 3,
        skip_playlist_after_errors: int = 3,
        extract_flat: bool = False,
        playlist_items: str | None = None,
        cookie_file: Path | str | None = None,
        po_token: str | None = None,
        impersonate_client: str | None = "chrome",
        player_clients: list[str] | None = None,
        js_runtime: str | None = "node",
    ) -> None:
        """Initialize YouTubeScraper and validate FFmpeg availability.

        Args:
            retry_config: Retry configuration. Uses default if not provided.
            download_archive: Path to file tracking already downloaded video IDs.
                Essential for large channels (20k+ videos) to avoid re-downloads.
            concurrent_fragments: Number of fragments to download concurrently.
                Higher values speed up downloads but use more bandwidth.
            skip_playlist_after_errors: Skip playlist after N consecutive errors.
                Prevents getting stuck on problematic videos.
            extract_flat: Use flat extraction (fast, minimal metadata).
                Set True for channel listings, False for full downloads.
            playlist_items: Specific items to download (e.g., "1:100" for first 100).
                Useful for batch processing large channels.
            cookie_file: Path to netscape format cookie file for YouTube authentication.
                Required if YouTube blocks downloads (HTTP 403).
            po_token: Proof of Origin token to bypass bot detection (HTTP 403).
            impersonate_client: Client to impersonate (e.g. "chrome", "safari").
                Requires curl_cffi to be installed.
            player_clients: List of YouTube player clients to use.
            js_runtime: JavaScript runtime to use for EJS scripts (e.g. "node", "deno").
        """
        self._logger = logger.bind(provider="youtube_scraper")
        self._retry_config = retry_config or RetryConfig()
        self._download_archive = Path(download_archive) if download_archive else None
        self._concurrent_fragments = concurrent_fragments
        self._skip_playlist_after_errors = skip_playlist_after_errors
        self._extract_flat = extract_flat
        self._playlist_items = playlist_items
        self._cookie_file = Path(cookie_file) if cookie_file else None
        self._po_token = po_token
        self._impersonate_client = impersonate_client
        self._player_clients = player_clients or ["tv", "web", "mweb"]
        self._js_runtime = js_runtime

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

    def _get_base_ydl_opts(self) -> dict[str, Any]:
        """Get base yt-dlp options with large-scale optimizations."""
        opts: dict[str, Any] = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,  # Continue on individual video errors
            "skip_playlist_after_errors": self._skip_playlist_after_errors,
            "extractor_args": {
                "youtube": {
                    "player_client": self._player_clients,
                }
            },
        }

        if self._po_token:
            token = self._po_token
            if "+" not in token:
                # Default to mweb.gvs context if not provided
                token = f"mweb.gvs+{token}"
            opts["extractor_args"]["youtube"]["po_token"] = [token]

        if self._impersonate_client:
            try:
                from yt_dlp.networking.impersonate import ImpersonateTarget

                opts["impersonate"] = ImpersonateTarget.from_str(self._impersonate_client)
            except (ImportError, ValueError):
                opts["impersonate"] = self._impersonate_client

        if self._js_runtime:
            if ":" in self._js_runtime:
                name, path = self._js_runtime.split(":", 1)
                expanded_path = str(Path(path).expanduser())
                opts["js_runtimes"] = {name: {"path": expanded_path}}
            elif self._js_runtime.startswith("/") or self._js_runtime.startswith("~"):
                expanded_path = str(Path(self._js_runtime).expanduser())
                opts["js_runtimes"] = {"deno": {"path": expanded_path}}
            else:
                opts["js_runtimes"] = {self._js_runtime: {}}
            opts["remote_components"] = {"ejs:github"}

        # Archive file to track downloaded videos (essential for large channels)
        if self._download_archive:
            self._download_archive.parent.mkdir(parents=True, exist_ok=True)
            opts["download_archive"] = str(self._download_archive)

        # Fast extraction mode (for channel listings)
        if self._extract_flat:
            opts["extract_flat"] = "in_playlist"
            opts["lazy_playlist"] = True  # Process as received, don't buffer all

        # Specific playlist items (for batch processing)
        if self._playlist_items:
            opts["playlist_items"] = self._playlist_items

        if self._cookie_file and self._cookie_file.exists():
            opts["cookiefile"] = str(self._cookie_file)

        return opts

    async def list_channel_videos(
        self,
        channel_url: str,
        max_videos: int | None = None,
    ) -> list[VideoInfo]:
        """List all videos in a channel/playlist efficiently.

        Uses extract_flat for fast listing without fetching full metadata.
        Essential for channels with 20k+ videos.

        Args:
            channel_url: Channel or playlist URL
            max_videos: Maximum number of videos to list (None for all)

        Returns:
            List of VideoInfo with basic metadata
        """
        operation_logger = self._logger.bind(
            url=channel_url,
            max_videos=max_videos,
            operation="list_channel_videos",
        )
        operation_logger.info("channel_listing_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        def _list_sync() -> list[VideoInfo]:
            opts: dict[str, Any] = {
                "format": "bestaudio/best",
                "quiet": True,
                "no_warnings": True,
                "extract_flat": "in_playlist",  # Fast extraction, no full metadata
                "lazy_playlist": True,  # Process as received
                "skip_download": True,  # Don't download, just list
            }

            if max_videos:
                opts["playlistend"] = max_videos

            videos = []
            with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore[arg-type]
                try:
                    info = ydl.extract_info(channel_url, download=False)
                    if info is None:
                        return videos

                    # Handle both single videos and playlists
                    entries = info.get("entries", [info])

                    for entry in entries:
                        if entry is None:
                            continue

                        video_id = entry.get("id")
                        if not video_id:
                            continue

                        title = entry.get("title") or "Unknown"
                        videos.append(
                            VideoInfo(
                                id=video_id,
                                title=title,
                                url=f"https://www.youtube.com/watch?v={video_id}",
                                duration=entry.get("duration"),
                                uploader=entry.get("uploader"),
                            )
                        )

                except Exception as e:
                    operation_logger.error(
                        "channel_listing_error",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

            return videos

        try:
            videos = await asyncio.to_thread(_list_sync)
            operation_logger.info(
                "channel_listing_completed",
                video_count=len(videos),
            )
            return videos

        except Exception as e:
            operation_logger.error(
                "channel_listing_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def download_batch(
        self,
        video_urls: list[str],
        output_dir: Path,
        audio_format: str = "mp3",
        max_concurrent: int = 3,
    ) -> list[AudioFile]:
        """Download audio from multiple videos with concurrency control.

        Args:
            video_urls: List of YouTube video URLs
            output_dir: Directory to save audio files
            audio_format: Audio format (default: mp3)
            max_concurrent: Maximum concurrent downloads

        Returns:
            List of successfully downloaded AudioFile objects
        """
        operation_logger = self._logger.bind(
            video_count=len(video_urls),
            output_dir=str(output_dir),
            audio_format=audio_format,
            max_concurrent=max_concurrent,
            operation="download_batch",
        )
        operation_logger.info("batch_download_started")

        output_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[AudioFile] = []
        failed_count = 0

        async def download_one(url: str) -> AudioFile | None:
            async with semaphore:
                try:
                    return await self.download(url, output_dir, audio_format)
                except Exception as e:
                    nonlocal failed_count
                    failed_count += 1
                    operation_logger.warning(
                        "batch_download_item_failed",
                        url=url,
                        error=str(e),
                    )
                    return None

        # Download all videos with controlled concurrency
        tasks = [download_one(url) for url in video_urls]
        download_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in download_results:
            if isinstance(result, AudioFile):
                results.append(result)
            # Exceptions are already logged

        operation_logger.info(
            "batch_download_completed",
            success_count=len(results),
            failed_count=failed_count,
        )

        return results

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
                video_title=result.title,
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

        ydl_opts = self._get_base_ydl_opts()
        ydl_opts.update(
            {
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": audio_format,
                    }
                ],
                "outtmpl": output_template,
            }
        )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
                operation_logger.debug("extracting_video_info")
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError(f"Failed to extract video info from {url}")

                video_id = info["id"]
                raw_title = info.get("title")
                video_title = raw_title if isinstance(raw_title, str) else "Unknown"
                duration = info.get("duration")

                audio_path = output_dir / f"{video_id}.{audio_format}"

                if not audio_path.exists():
                    raise RuntimeError(f"Downloaded file not found at {audio_path}")

                return AudioFile(
                    path=audio_path,
                    source_url=url,
                    title=video_title,
                    duration=duration,
                )

        except Exception as e:
            operation_logger.error(
                "download_sync_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Failed to download audio from {url}: {e}") from e

    @classmethod
    async def scrape_channel(
        cls,
        channel_url: str,
        output_dir: Path,
        audio_format: str = "mp3",
        *,
        batch_size: int = 100,
        max_concurrent: int = 3,
        download_archive: Path | str | None = None,
        max_videos: int | None = None,
    ) -> list[AudioFile]:
        """High-level helper to scrape an entire channel with batch processing.

        This is the recommended method for large channels (20k+ videos).

        Args:
            channel_url: Channel URL
            output_dir: Directory to save audio files
            audio_format: Audio format
            batch_size: Number of videos to download per batch
            max_concurrent: Maximum concurrent downloads within a batch
            download_archive: Path to download archive file
            max_videos: Maximum total videos to download (None for all)

        Returns:
            List of successfully downloaded AudioFile objects
        """
        scraper = cls(
            download_archive=download_archive,
            concurrent_fragments=3,
        )

        # Step 1: Fast channel listing
        videos = await scraper.list_channel_videos(channel_url, max_videos)

        if not videos:
            logger.warning("no_videos_found", channel_url=channel_url)
            return []

        # Step 2: Batch download
        all_results: list[AudioFile] = []
        total_batches = (len(videos) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(videos))
            batch = videos[start_idx:end_idx]

            logger.info(
                "processing_batch",
                batch_number=batch_idx + 1,
                total_batches=total_batches,
                batch_size=len(batch),
            )

            batch_urls = [v.url for v in batch]
            results = await scraper.download_batch(
                batch_urls,
                output_dir,
                audio_format,
                max_concurrent=max_concurrent,
            )

            all_results.extend(results)

            # Small delay between batches to be nice to YouTube
            if batch_idx < total_batches - 1:
                await asyncio.sleep(1)

        logger.info(
            "channel_scrape_completed",
            total_videos=len(videos),
            successful_downloads=len(all_results),
            failed_downloads=len(videos) - len(all_results),
        )

        return all_results
