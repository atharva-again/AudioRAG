"""Source discovery and expansion for batch indexing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from audiorag.core.logging_config import get_logger
from audiorag.source.url import AUDIO_EXTENSIONS

if TYPE_CHECKING:
    from audiorag.core import AudioRAGConfig

logger = get_logger(__name__)


async def discover_sources(inputs: list[str], config: AudioRAGConfig | None = None) -> list[str]:
    """Expand input URLs and paths into individual indexable sources.

    Handles:
    - YouTube video URLs
    - YouTube playlist/channel URLs (expanded to video URLs)
    - Local audio file paths
    - Local directories (expanded recursively to audio file paths)
    - Direct HTTP audio URLs

    Args:
        inputs: List of URLs or paths to expand.
        config: Optional configuration for YouTubeSource.

    Returns:
        List of expanded source URLs/paths.
    """
    expanded_sources: list[str] = []

    for item in inputs:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                logger.info("expanding_directory", path=str(path))
                files = []
                for ext in AUDIO_EXTENSIONS:
                    files.extend(path.glob(f"**/*{ext}"))
                unique_files = sorted({str(f.absolute()) for f in files})
                expanded_sources.extend(unique_files)
                logger.debug("directory_expanded", count=len(unique_files))
            else:
                expanded_sources.append(str(path.absolute()))
            continue

        if "youtube.com" in item or "youtu.be" in item:
            logger.info("checking_youtube_source", url=item)
            try:
                from audiorag.source.youtube import YouTubeSource

                if config:
                    scraper = YouTubeSource(
                        download_archive=config.youtube_download_archive,
                        cookie_file=config.youtube_cookie_file,
                        po_token=config.youtube_po_token,
                        impersonate_client=config.youtube_impersonate,
                        player_clients=config.youtube_player_clients,
                        js_runtime=config.js_runtime,
                    )
                else:
                    scraper = YouTubeSource()
                videos = await scraper.list_channel_videos(item)

                if videos:
                    video_urls = [v.url for v in videos]
                    expanded_sources.extend(video_urls)
                    logger.debug("youtube_source_expanded", count=len(video_urls))
                else:
                    expanded_sources.append(item)
            except ImportError:
                logger.warning("youtube_source_not_available_for_expansion", url=item)
                expanded_sources.append(item)
            except Exception as e:
                logger.warning("youtube_expansion_failed", url=item, error=str(e))
                expanded_sources.append(item)
            continue

        expanded_sources.append(item)

    seen = set()
    unique_sources = []
    for s in expanded_sources:
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)

    return unique_sources
