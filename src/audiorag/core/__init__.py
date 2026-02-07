"""Core AudioRAG components.

This module contains the base classes, protocols, models, and configuration
that are shared across all AudioRAG functionality.
"""

from __future__ import annotations

from audiorag.core.config import AudioRAGConfig
from audiorag.core.exceptions import (
    AudioRAGError,
    ConfigurationError,
    PipelineError,
    ProviderError,
    StateError,
)
from audiorag.core.logging_config import configure_logging, get_logger
from audiorag.core.models import (
    AudioFile,
    ChunkMetadata,
    IndexingStatus,
    QueryResult,
    Source,
    TranscriptionSegment,
)
from audiorag.core.protocols import (
    AudioSourceProvider,
    EmbeddingProvider,
    GenerationProvider,
    RerankerProvider,
    STTProvider,
    VectorStoreProvider,
)
from audiorag.core.retry_config import RetryConfig, create_retry_decorator
from audiorag.core.state import StateManager

__all__ = [
    # Config
    "AudioRAGConfig",
    "RetryConfig",
    "create_retry_decorator",
    # Models
    "AudioFile",
    "ChunkMetadata",
    "IndexingStatus",
    "QueryResult",
    "Source",
    "TranscriptionSegment",
    # Protocols
    "AudioSourceProvider",
    "EmbeddingProvider",
    "GenerationProvider",
    "RerankerProvider",
    "STTProvider",
    "VectorStoreProvider",
    # Exceptions
    "AudioRAGError",
    "ConfigurationError",
    "PipelineError",
    "ProviderError",
    "StateError",
    # Logging
    "configure_logging",
    "get_logger",
    # State
    "StateManager",
]
