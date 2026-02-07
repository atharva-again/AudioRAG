"""Base transcriber class for STT providers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from audiorag.core.exceptions import ProviderError
from audiorag.core.logging_config import get_logger
from audiorag.core.models import TranscriptionSegment
from audiorag.core.retry_config import RetryConfig, create_retry_decorator

logger = get_logger(__name__)


@runtime_checkable
class BaseTranscriber(Protocol):
    """Protocol for speech-to-text providers.

    All STT providers must implement this protocol.
    """

    async def transcribe(
        self, audio_path: str, language: str | None = None
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file.
            language: Optional language code (e.g., "en" for English).

        Returns:
            List of transcription segments with timing and text.
        """
        ...


class TranscriberMixin:
    """Mixin providing common functionality for STT providers.

    Subclasses must implement:
    - _provider_name: str
    - _retryable_exceptions: tuple[type[Exception], ...]
    - _transcribe_impl: async method doing the actual API call
    """

    _provider_name: str = "transcriber"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the transcriber.

        Args:
            api_key: Provider API key. If None, uses environment variable.
            model: STT model to use.
            retry_config: Retry configuration. Uses default if not provided.
        """
        self._model = model
        self._retry_config = retry_config or RetryConfig()
        self._logger = logger.bind(provider=self._provider_name, model=model)
        self._api_key = api_key

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name.

        Args:
            value: New model name.
        """
        self._model = value
        self._logger = self._logger.bind(model=value)

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
