"""AudioRAG package.

A modular RAG (Retrieval Augmented Generation) library for audio content.

This package provides composable components for:
- Ingesting audio (download, transcribe, chunk, embed, store)
- Querying indexed content (embed query, search, generate)

Usage:
    # Full pipeline (backward compatible)
    from audiorag import AudioRAGPipeline, AudioRAGConfig
    pipeline = AudioRAGPipeline(config)
    await pipeline.index("https://youtube.com/...")
    result = await pipeline.query("What is discussed?")

    # Modular imports (new)
    from audiorag.embed import OpenAIEmbeddingProvider
    from audiorag.store import ChromaDBVectorStore
    from audiorag.pipeline import AudioRAGPipeline

"""

from __future__ import annotations

# Backward compatible exports - full pipeline
from audiorag.core import (
    AudioRAGConfig,
    BatchIndexResult,
    QueryResult,
    RetryConfig,
    Source,
    configure_logging,
    get_logger,
)
from audiorag.pipeline import AudioRAGPipeline

__version__ = "0.5.0"

__all__ = [
    "AudioRAGConfig",
    "AudioRAGPipeline",
    "BatchIndexResult",
    "QueryResult",
    "RetryConfig",
    "Source",
    "__version__",
    "configure_logging",
    "get_logger",
]
