"""OpenAI Speech-to-Text provider using Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError  # type: ignore

from audiorag.core.logging_config import get_logger
from audiorag.core.models import TranscriptionSegment
from audiorag.transcribe._base import TranscriberMixin

logger = get_logger(__name__)


class OpenAITranscriber(TranscriberMixin):
    """Speech-to-Text provider using OpenAI's Whisper API."""

    MODEL_WHISPER_1 = "whisper-1"

    _provider_name: str = "openai_stt"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        RateLimitError,
        APIError,
        APITimeoutError,
        ConnectionError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "whisper-1",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize OpenAI STT provider."""
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        self.client = AsyncOpenAI(api_key=api_key)

    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file using OpenAI Whisper API."""
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
            segments = self._extract_segments(response)
            operation_logger.info(
                "transcription_completed",
                segments_count=len(segments),
                duration_seconds=segments[-1].end_time if segments else 0,
            )
            return segments
        except Exception as e:
            raise await self._wrap_error(e, "transcribe")

    def _extract_segments(self, response: Any) -> list[TranscriptionSegment]:
        """Extract segments from OpenAI response."""
        segments = []
        if hasattr(response, "segments") and response.segments:
            for segment in response.segments:
                if isinstance(segment, dict):
                    start = segment.get("start", 0.0)
                    end = segment.get("end", 0.0)
                    text = segment.get("text", "")
                else:
                    start = getattr(segment, "start", 0.0)
                    end = getattr(segment, "end", 0.0)
                    text = getattr(segment, "text", "")

                segments.append(
                    TranscriptionSegment(
                        start_time=start,
                        end_time=end,
                        text=text,
                    )
                )
        return segments
