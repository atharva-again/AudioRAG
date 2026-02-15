"""Structured exception hierarchy for AudioRAG.

This module provides a comprehensive exception hierarchy for the AudioRAG codebase,
replacing generic RuntimeError and Exception usage with specific, informative
exception types that enable better error handling and debugging.

Exception Hierarchy:
    AudioRAGError (base)
    ├── PipelineError
    ├── ProviderError
    ├── ConfigurationError
    └── StateError

Usage:
    from audiorag.exceptions import PipelineError, ProviderError

    try:
        await pipeline.index(url)
    except PipelineError as e:
        logger.error(f"Pipeline failed at stage {e.stage}: {e}")
    except ProviderError as e:
        if e.retryable:
            await retry_operation()
"""

from __future__ import annotations


class AudioRAGError(Exception):
    """Base exception class for all AudioRAG errors.

        All exceptions in the AudioRAG codebase should inherit from this class
    to enable catching all AudioRAG-specific errors in a single except block.

    Example:
            try:
                await pipeline.index(url)
            except AudioRAGError as e:
                # Catches any AudioRAG-specific error
                logger.error(f"AudioRAG operation failed: {e}")
    """

    pass


class PipelineError(AudioRAGError):
    """Exception raised when pipeline execution fails.

    This exception provides context about which pipeline stage failed
    and which source was being processed, enabling better debugging
    and error reporting.

    Args:
        message: Human-readable error message.
        stage: The pipeline stage that failed (e.g., "download", "transcribe").
        source_url: The URL of the source being processed, if available.

    Attributes:
        stage: The pipeline stage that failed.
        source_url: The source URL being processed.

    Example:
        raise PipelineError(
            "Failed to process audio: file not found",
            stage="download",
            source_url="file:///path/to/audio.mp3"
        )
    """

    def __init__(self, message: str, stage: str, source_url: str | None = None) -> None:
        """Initialize PipelineError with context."""
        super().__init__(message)
        self.stage = stage
        self.source_url = source_url


class ProviderError(AudioRAGError):
    """Exception raised when an external provider fails.

        This exception wraps errors from external services (OpenAI, Cohere, etc.)
    and indicates whether the error is retryable, enabling automatic retry logic.

    Args:
            message: Human-readable error message.
            provider: The name of the provider that failed (e.g., "openai", "cohere").
            retryable: Whether the error is transient and can be retried.
                Defaults to False.

    Attributes:
            provider: The name of the failed provider.
            retryable: Whether the error can be retried.

    Example:
            raise ProviderError(
                "OpenAI API rate limit exceeded",
                provider="openai",
                retryable=True
            )
    """

    def __init__(self, message: str, provider: str, retryable: bool = False) -> None:
        """Initialize ProviderError with context."""
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ConfigurationError(AudioRAGError):
    """Exception raised when configuration validation fails.

    This exception indicates that the application configuration is invalid
    or missing required values, preventing proper operation.

    Args:
        message: Human-readable error message describing the configuration issue.

    Example:
        raise ConfigurationError(
            "OPENAI_API_KEY is required but not set. "
            "Please set the AUDIORAG_OPENAI_API_KEY environment variable."
        )
    """

    def __init__(self, message: str) -> None:
        """Initialize ConfigurationError."""
        super().__init__(message)


class StateError(AudioRAGError):
    """Exception raised when database or state management fails.

    This exception wraps SQLite and state management errors, providing
    context about database operations that failed.

    Args:
        message: Human-readable error message describing the state error.

    Example:
        raise StateError(
            "Failed to store chunks: database is locked"
        )
    """

    def __init__(self, message: str) -> None:
        """Initialize StateError."""
        super().__init__(message)


class DiscoveryError(AudioRAGError):
    """Exception raised when source discovery or expansion fails.

    This exception indicates that a source path could not be expanded
    into individual audio files, preventing indexing.

    Args:
        message: Human-readable error message describing the discovery failure.
        url: The path that failed to expand.

    Attributes:
        url: The source path that could not be expanded.

    Example:
        raise DiscoveryError(
            "Failed to expand directory: no audio files found",
            url="/path/to/empty/directory"
        )
    """

    def __init__(self, message: str, url: str | None = None) -> None:
        """Initialize DiscoveryError with context."""
        super().__init__(message)
        self.url = url


class BudgetExceededError(ProviderError):
    def __init__(
        self,
        *,
        provider: str,
        metric: str,
        limit: int,
        current: int,
        requested: int,
        window_seconds: int,
    ) -> None:
        message = (
            f"Budget exceeded for provider='{provider}', metric='{metric}': "
            f"limit={limit}, current={current}, requested={requested}, "
            f"window_seconds={window_seconds}"
        )
        super().__init__(message=message, provider=provider, retryable=False)
        self.metric = metric
        self.limit = limit
        self.current = current
        self.requested = requested
        self.window_seconds = window_seconds


__all__ = [
    "AudioRAGError",
    "BudgetExceededError",
    "ConfigurationError",
    "DiscoveryError",
    "PipelineError",
    "ProviderError",
    "StateError",
]
