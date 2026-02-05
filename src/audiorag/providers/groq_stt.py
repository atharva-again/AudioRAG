"""Groq Speech-to-Text provider implementation using Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from groq import APIError, AsyncGroq, RateLimitError  # type: ignore

from audiorag.logging_config import get_logger
from audiorag.models import TranscriptionSegment
from audiorag.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class GroqSTTProvider:
    """Speech-to-Text provider using Groq's Whisper API.

    Groq provides fast inference for Whisper Large v3 model through
    their OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-large-v3",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize Groq STT provider.

        Args:
            api_key: Groq API key. If None, uses GROQ_API_KEY environment variable.
            model: Model to use for transcription. Defaults to "whisper-large-v3".
            retry_config: Retry configuration. Uses default if not provided.
        """
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self._logger = logger.bind(provider="groq_stt", model=model)
        self._retry_config = retry_config or RetryConfig()

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Groq API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(
                RateLimitError,
                APIError,
                ConnectionError,
            ),
        )

    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file using Groq Whisper API.

        Args:
            audio_path: Path to the audio file to transcribe.
            language: Optional language code (e.g., "en" for English).

        Returns:
            List of TranscriptionSegment objects with timing and text.
        """
        operation_logger = self._logger.bind(
            audio_path=str(audio_path),
            language=language,
            operation="transcribe",
        )
        operation_logger.debug("transcription_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _transcribe_with_retry() -> Any:
            with open(audio_path, "rb") as audio_file:
                return await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    language=language,
                )

        try:
            response = await _transcribe_with_retry()

            # Extract segments from response
            segments = []
            if hasattr(response, "segments") and response.segments:
                for segment in response.segments:
                    segments.append(
                        TranscriptionSegment(
                            start_time=segment.start,
                            end_time=segment.end,
                            text=segment.text,
                        )
                    )

            operation_logger.info(
                "transcription_completed",
                segments_count=len(segments),
                duration_seconds=segments[-1].end_time if segments else 0,
            )
            return segments

        except Exception as e:
            operation_logger.error(
                "transcription_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
