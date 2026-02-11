"""Source discovery and expansion for batch indexing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from audiorag.core.logging_config import get_logger
from audiorag.source.url import AUDIO_EXTENSIONS

if TYPE_CHECKING:
    from audiorag.core import AudioRAGConfig

logger = get_logger(__name__)


def _expand_directory(path: Path) -> list[str]:
    """Expand a directory to a list of audio file paths."""
    logger.info("expanding_directory", path=str(path))
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(path.glob(f"**/*{ext}"))
    unique_files = sorted({str(f.absolute()) for f in files})
    logger.debug("directory_expanded", count=len(unique_files))
    return unique_files


async def _expand_youtube_source(item: str, config: AudioRAGConfig | None) -> list[str]:
    """Expand a YouTube URL to individual video URLs."""
    logger.info("checking_youtube_source", url=item)
    try:
        from audiorag.source.youtube import YouTubeSource

        scraper = (
            YouTubeSource(
                download_archive=config.youtube_download_archive,
                cookie_file=config.youtube_cookie_file,
                po_token=config.youtube_po_token,
                impersonate_client=config.youtube_impersonate,
                player_clients=config.youtube_player_clients,
                js_runtime=config.js_runtime,
            )
            if config
            else YouTubeSource()
        )
        videos = await scraper.list_channel_videos(item)

        if videos:
            video_urls = [v.url for v in videos]
            logger.debug("youtube_source_expanded", count=len(video_urls))
            return video_urls
    except ImportError:
        logger.warning("youtube_source_not_available_for_expansion", url=item)
    except Exception as e:
        logger.warning("youtube_expansion_failed", url=item, error=str(e))
    return [item]


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
                expanded_sources.extend(_expand_directory(path))
            else:
                expanded_sources.append(str(path.absolute()))
            continue

        if "youtube.com" in item or "youtu.be" in item:
            expanded_sources.extend(await _expand_youtube_source(item, config))
            continue

        expanded_sources.append(item)

    seen = set()
    unique_sources: list[str] = []
    for s in expanded_sources:
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)

    return unique_sources
