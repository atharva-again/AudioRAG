"""Reranking providers."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import with clear error messages for missing dependencies."""
    if name == "CohereReranker":
        try:
            from audiorag.rerank.cohere import CohereReranker

            return CohereReranker
        except ImportError:
            raise ImportError(
                "CohereReranker requires 'cohere'. Install with: pip install audiorag[cohere]"
            ) from None
    if name == "PassthroughReranker":
        try:
            from audiorag.rerank.passthrough import PassthroughReranker

            return PassthroughReranker
        except ImportError:
            raise ImportError(
                "PassthroughReranker requires 'audiorag'. Install with: pip install audiorag"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CohereReranker",
    "PassthroughReranker",
]
