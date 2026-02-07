"""Chunking strategies."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import with clear error messages for missing dependencies."""
    if name == "TimeBasedChunker":
        try:
            from audiorag.chunk.time_based import TimeBasedChunker

            return TimeBasedChunker
        except ImportError:
            raise ImportError(
                "TimeBasedChunker requires 'audiorag'. Install with: pip install audiorag"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TimeBasedChunker",
]
