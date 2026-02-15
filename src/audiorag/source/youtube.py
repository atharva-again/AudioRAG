"""YouTube audio scraper using yt-dlp - minimal implementation."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from audiorag.core.exceptions import ProviderError
from audiorag.core.logging_config import get_logger
from audiorag.core.models import AudioFile, SourceMetadata

logger = get_logger(__name__)


@dataclass
class VideoInfo:
    id: str
    title: str
    url: str
    duration: float | None = None
    uploader: str | None = None


class YouTubeSource:
    """Downloads audio from YouTube using yt-dlp.

    Minimal async wrapper. All yt-dlp options passed directly to YoutubeDL.
    """

    def __init__(
        self,
        *,
        ydl_opts: dict[str, Any] | None = None,
        download_archive: Path | str | None = None,
    ) -> None:
        self._logger = logger.bind(provider="youtube")
        self._ydl_opts = ydl_opts or {}
        self._download_archive = Path(download_archive) if download_archive else None

    def _ensure_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg"):
            raise ProviderError(
                message="FFmpeg not found. Install it to download audio.",
                provider="youtube",
                retryable=False,
            )

    def _get_opts(
        self,
        output_dir: Path | None = None,
        audio_format: str = "mp3",
        metadata_only: bool = False,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {}

        if self._download_archive:
            self._download_archive.parent.mkdir(parents=True, exist_ok=True)
            opts["download_archive"] = str(self._download_archive)

        if not metadata_only:
            opts.update(self._ydl_opts)

        if metadata_only:
            opts["skip_download"] = True
        else:
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": audio_format}]
            opts["outtmpl"] = str(output_dir / "%(id)s.%(ext)s")

        return opts

    async def get_metadata(self, url: str) -> SourceMetadata:
        self._ensure_ffmpeg()

        def _sync() -> SourceMetadata:
            import yt_dlp

            opts = self._get_opts(metadata_only=True)
            with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise ProviderError(
                        message=f"Failed to extract metadata from {url}",
                        provider="youtube",
                        retryable=False,
                    )
                return SourceMetadata(
                    duration=info.get("duration"),
                    title=info.get("title"),
                    raw=info,
                )

        return await asyncio.to_thread(_sync)

    async def download(
        self,
        url: str,
        output_dir: Path,
        audio_format: str = "mp3",
        metadata: SourceMetadata | None = None,
    ) -> AudioFile:
        self._ensure_ffmpeg()

        def _sync() -> AudioFile:
            import yt_dlp

            opts = self._get_opts(output_dir, audio_format)
            with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise ProviderError(
                        message=f"Failed to extract video info from {url}",
                        provider="youtube",
                        retryable=False,
                    )

                video_id = info["id"]
                title = info.get("title") or "Unknown"
                duration = info.get("duration")

                audio_path = output_dir / f"{video_id}.{audio_format}"
                if not audio_path.exists():
                    actual = list(output_dir.glob(f"{video_id}.*"))
                    audio_path = actual[0] if actual else None

                if not audio_path or not audio_path.exists():
                    raise ProviderError(
                        message=f"Downloaded file not found for {url}",
                        provider="youtube",
                        retryable=False,
                    )

                return AudioFile(
                    path=audio_path,
                    source_url=url,
                    title=title,
                    duration=duration,
                )

        return await asyncio.to_thread(_sync)

    async def download_batch(
        self,
        urls: list[str],
        output_dir: Path,
        audio_format: str = "mp3",
        max_concurrent: int = 3,
    ) -> list[AudioFile]:
        output_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[AudioFile] = []
        failed = 0

        async def download_one(url: str) -> AudioFile | None:
            nonlocal failed
            async with semaphore:
                try:
                    return await self.download(url, output_dir, audio_format)
                except Exception:
                    failed += 1
                    return None

        for result in await asyncio.gather(*[download_one(url) for url in urls]):
            if isinstance(result, AudioFile):
                results.append(result)

        self._logger.info(
            "batch_complete",
            total=len(urls),
            succeeded=len(results),
            failed=failed,
        )
        return results

    async def list_channel_videos(
        self,
        channel_url: str,
        max_videos: int | None = None,
    ) -> list[VideoInfo]:
        self._ensure_ffmpeg()

        def _sync() -> list[VideoInfo]:
            import yt_dlp

            opts = self._get_opts(metadata_only=True)
            if max_videos:
                opts["playlistend"] = max_videos

            videos = []
            with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                if info is None:
                    return []

                for entry in info.get("entries", [info]):
                    if entry and entry.get("id"):
                        videos.append(
                            VideoInfo(
                                id=entry["id"],
                                title=entry.get("title") or "Unknown",
                                url=f"https://www.youtube.com/watch?v={entry['id']}",
                                duration=entry.get("duration"),
                                uploader=entry.get("uploader"),
                            )
                        )
            return videos

        videos = await asyncio.to_thread(_sync)
        self._logger.info("listed_videos", count=len(videos), url=channel_url)
        return videos

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
        scraper = cls(download_archive=download_archive)

        videos = await scraper.list_channel_videos(channel_url, max_videos)
        if not videos:
            return []

        all_results = []
        for i in range(0, len(videos), batch_size):
            batch = videos[i : i + batch_size]
            urls = [v.url for v in batch]
            results = await scraper.download_batch(urls, output_dir, audio_format, max_concurrent)
            all_results.extend(results)
            if i + batch_size < len(videos):
                await asyncio.sleep(1)

        return all_results
