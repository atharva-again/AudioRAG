"""Embedding providers."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import with clear error messages for missing dependencies."""
    if name == "OpenAIEmbedder":
        try:
            from audiorag.embed.openai import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider
        except ImportError:
            raise ImportError(
                "OpenAIEmbedder requires 'openai'. Install with: uv pip install audiorag[openai]"
            ) from None
    if name == "VoyageEmbedder":
        try:
            from audiorag.embed.voyage import VoyageEmbeddingProvider

            return VoyageEmbeddingProvider
        except ImportError:
            raise ImportError(
                "VoyageEmbedder requires 'voyageai'. Install with: uv pip install audiorag[voyage]"
            ) from None
    if name == "CohereEmbedder":
        try:
            from audiorag.embed.cohere import CohereEmbeddingProvider

            return CohereEmbeddingProvider
        except ImportError:
            raise ImportError(
                "CohereEmbedder requires 'cohere'. Install with: uv pip install audiorag[cohere]"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CohereEmbedder",
    "OpenAIEmbedder",
    "VoyageEmbedder",
]
