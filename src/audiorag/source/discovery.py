"""Source discovery and expansion for batch indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from audiorag.core.logging_config import get_logger
from audiorag.source.local import AUDIO_EXTENSIONS

if TYPE_CHECKING:
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


def _expand_directory(path: Path) -> list[str]:
    """Expand a directory to a list of audio file paths."""
    logger.info("expanding_directory", path=str(path))
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(path.glob(f"**/*{ext}"))
    unique_files = sorted({str(f.absolute()) for f in files})
    logger.debug("directory_expanded", count=len(unique_files))
    return unique_files


async def discover_sources(inputs: list[str]) -> list[DiscoveredSource]:
    """Expand input paths into individual indexable sources.

    Handles:
    - Local audio file paths
    - Local directories (expanded recursively to audio file paths)

    Args:
        inputs: List of paths to expand.

    Returns:
        List of discovered sources.
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

        # For non-existent paths, pass through as-is (will fail later if invalid)
        expanded_sources.append(DiscoveredSource(url=item))

    seen: set[str] = set()
    unique_sources: list[DiscoveredSource] = []
    for s in expanded_sources:
        if s.url not in seen:
            unique_sources.append(s)
            seen.add(s.url)

    return unique_sources


async def discover_source_urls(inputs: list[str]) -> list[str]:
    """Expand input paths into individual indexable source URLs.

    Convenience wrapper that returns only URLs without metadata.

    Args:
        inputs: List of paths to expand.

    Returns:
        List of expanded source paths.
    """
    discovered = await discover_sources(inputs)
    return [s.url for s in discovered]
