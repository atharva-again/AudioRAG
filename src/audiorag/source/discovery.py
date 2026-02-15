"""Source discovery and expansion for batch indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from audiorag.core.exceptions import DiscoveryError
from audiorag.core.logging_config import get_logger
from audiorag.source.url import AUDIO_EXTENSIONS
from audiorag.source.ydl_utils import build_ydl_opts

if TYPE_CHECKING:
    from audiorag.core import AudioRAGConfig
    from audiorag.core.models import SourceMetadata

logger = get_logger(__name__)


@dataclass
class DiscoveredSource:
    """A discovered source with optional pre-fetched metadata.

    Attributes:
        url: The source URL or file path.
        metadata: Pre-fetched metadata from discovery, if available.
    """

    url: str
    metadata: SourceMetadata | None = None


def _is_youtube_collection(url: str) -> bool:
    """Check if URL is a YouTube playlist or channel.

    Args:
        url: The URL to check.

    Returns:
        True if the URL represents a collection (playlist/channel).
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()

    # Playlist URLs: /playlist or list= parameter
    if "/playlist" in path or "list=" in query:
        return True

    # Channel URLs: /channel/, /c/, /user/, or /@handle
    return "/channel/" in path or "/c/" in path or "/user/" in path or "/@" in path


def _expand_directory(path: Path) -> list[str]:
    """Expand a directory to a list of audio file paths."""
    logger.info("expanding_directory", path=str(path))
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(path.glob(f"**/*{ext}"))
    unique_files = sorted({str(f.absolute()) for f in files})
    logger.debug("directory_expanded", count=len(unique_files))
    return unique_files


async def _expand_youtube_source(
    item: str, config: AudioRAGConfig | None
) -> list[DiscoveredSource]:
    """Expand a YouTube URL to individual video URLs with metadata."""
    logger.info("checking_youtube_source", url=item)
    try:
        from audiorag.core.models import SourceMetadata
        from audiorag.source.youtube import YouTubeSource

        ydl_opts = build_ydl_opts(config) if config else None

        scraper = YouTubeSource(
            download_archive=config.youtube_download_archive if config else None,
            ydl_opts=ydl_opts,
        )
        videos = await scraper.list_channel_videos(item)

        if videos:
            discovered = [
                DiscoveredSource(
                    url=v.url,
                    metadata=SourceMetadata(
                        duration=v.duration,
                        title=v.title,
                        raw=None,
                    ),
                )
                for v in videos
            ]
            logger.debug("youtube_source_expanded", count=len(discovered))
            return discovered

        # Raise error for collections that failed to expand
        if _is_youtube_collection(item):
            raise DiscoveryError(
                f"Failed to expand YouTube source: no videos found for {item}",
                url=item,
            )
    except ImportError:
        logger.warning("youtube_source_not_available_for_expansion", url=item)
    except DiscoveryError:
        raise
    except Exception as e:
        logger.warning("youtube_expansion_failed", url=item, error=str(e))
    return [DiscoveredSource(url=item)]


async def discover_sources(
    inputs: list[str], config: AudioRAGConfig | None = None
) -> list[DiscoveredSource]:
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
        List of discovered sources with optional metadata.
    """
    expanded_sources: list[DiscoveredSource] = []

    for item in inputs:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                files = _expand_directory(path)
                expanded_sources.extend(DiscoveredSource(url=f) for f in files)
            else:
                expanded_sources.append(DiscoveredSource(url=str(path.absolute())))
            continue

        if "youtube.com" in item or "youtu.be" in item:
            expanded_sources.extend(await _expand_youtube_source(item, config))
            continue

        expanded_sources.append(DiscoveredSource(url=item))

    seen: set[str] = set()
    unique_sources: list[DiscoveredSource] = []
    for s in expanded_sources:
        if s.url not in seen:
            unique_sources.append(s)
            seen.add(s.url)

    return unique_sources


async def discover_source_urls(
    inputs: list[str], config: AudioRAGConfig | None = None
) -> list[str]:
    """Expand input URLs and paths into individual indexable source URLs.

    Convenience wrapper that returns only URLs without metadata.

    Args:
        inputs: List of URLs or paths to expand.
        config: Optional configuration for YouTubeSource.

    Returns:
        List of expanded source URLs/paths.
    """
    discovered = await discover_sources(inputs, config)
    return [s.url for s in discovered]
