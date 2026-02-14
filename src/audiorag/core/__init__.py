"""Core AudioRAG components.

This module contains the base classes, protocols, models, and configuration
that are shared across all AudioRAG functionality.
"""

from __future__ import annotations

from audiorag.core.budget import BudgetGovernor, BudgetLimits
from audiorag.core.config import AudioRAGConfig
from audiorag.core.doctor import (
    DependencyCheck,
    DoctorResult,
    check_dependencies,
)
from audiorag.core.exceptions import (
    AudioRAGError,
    BudgetExceededError,
    ConfigurationError,
    PipelineError,
    ProviderError,
    StateError,
)
from audiorag.core.logging_config import configure_logging, get_logger
from audiorag.core.models import (
    AudioFile,
    BatchIndexFailure,
    BatchIndexResult,
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
    VerifiableVectorStoreProvider,
)
from audiorag.core.retry_config import RetryConfig, create_retry_decorator
from audiorag.core.state import StateManager

__all__ = [
    "AudioFile",
    "AudioRAGConfig",
    "AudioRAGError",
    "AudioSourceProvider",
    "BatchIndexFailure",
    "BatchIndexResult",
    "BudgetExceededError",
    "BudgetGovernor",
    "BudgetLimits",
    "ChunkMetadata",
    "ConfigurationError",
    "DependencyCheck",
    "DoctorResult",
    "EmbeddingProvider",
    "GenerationProvider",
    "IndexingStatus",
    "PipelineError",
    "ProviderError",
    "QueryResult",
    "RerankerProvider",
    "RetryConfig",
    "STTProvider",
    "Source",
    "StateError",
    "StateManager",
    "TranscriptionSegment",
    "VectorStoreProvider",
    "VerifiableVectorStoreProvider",
    "check_dependencies",
    "configure_logging",
    "create_retry_decorator",
    "get_logger",
]
