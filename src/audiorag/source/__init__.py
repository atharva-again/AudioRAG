"""Audio source providers."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import with clear error messages for missing dependencies."""
    if name == "YouTubeSource":
        try:
            from audiorag.source.youtube import YouTubeSource

            return YouTubeSource
        except ImportError:
            raise ImportError(
                "YouTubeSource requires 'yt-dlp'. Install with: pip install audiorag[youtube]"
            ) from None
    if name == "LocalSource":
        try:
            from audiorag.source.local import LocalSource

            return LocalSource
        except ImportError:
            raise ImportError(
                "LocalSource requires 'audiorag'. Install with: pip install audiorag"
            ) from None
    if name == "URLSource":
        try:
            from audiorag.source.url import URLSource

            return URLSource
        except ImportError:
            raise ImportError(
                "URLSource requires 'audiorag'. Install with: pip install audiorag"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LocalSource",
    "URLSource",
    "YouTubeSource",
]
