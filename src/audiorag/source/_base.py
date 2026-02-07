"""Base audio source mixin for audio source providers."""

from __future__ import annotations

from typing import Any

from audiorag.core.exceptions import ProviderError
from audiorag.core.logging_config import get_logger
from audiorag.core.retry_config import RetryConfig, create_retry_decorator

logger = get_logger(__name__)


class AudioSourceMixin:
    """Mixin providing common functionality for audio source providers.

    Subclasses must set:
    - _provider_name: str
    - _retryable_exceptions: tuple[type[Exception], ...]
    """

    _provider_name: str = "audio_source"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        *,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the audio source.

        Args:
            retry_config: Retry configuration. Uses default if not provided.
        """
        self._retry_config = retry_config or RetryConfig()
        self._logger = logger.bind(provider=self._provider_name)

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for provider API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=self._retryable_exceptions,
        )

    async def _wrap_error(self, e: Exception, operation: str) -> ProviderError:
        """Wrap provider errors with structured exception.

        Args:
            e: Original exception.
            operation: Name of the operation that failed.

        Returns:
            ProviderError with context.
        """
        return ProviderError(
            message=f"{self._provider_name} {operation} failed: {e}",
            provider=self._provider_name,
            retryable=isinstance(e, self._retryable_exceptions),
        )
