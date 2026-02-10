"""AssemblyAI Speech-to-Text provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.core.models import TranscriptionSegment
from audiorag.transcribe._base import TranscriberMixin

logger = get_logger(__name__)


class AssemblyAITranscriber(TranscriberMixin):
    """Speech-to-Text provider using AssemblyAI's API."""

    MODEL_BEST = "best"
    MODEL_NANO = "nano"
    MODEL_UNIVERSAL = "universal"

    MAX_WORDS_PER_SEGMENT = 20

    _provider_name: str = "assemblyai_stt"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "best",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize AssemblyAI STT provider."""
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        import assemblyai as aai  # type: ignore[import]

        aai.settings.api_key = api_key
        self._aai = aai
        self._transcriber = aai.Transcriber()

    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file using AssemblyAI API."""
        operation_logger = self._logger.bind(
            audio_path=str(audio_path),
            language=language,
            operation="transcribe",
        )
        operation_logger.debug("transcription_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _transcribe_with_retry() -> Any:
            return self._transcriber.transcribe(audio_path)

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
        """Extract segments from AssemblyAI response."""
        segments = []
        if response.utterances:
            for utterance in response.utterances:
                segments.append(
                    TranscriptionSegment(
                        start_time=utterance.start / 1000,
                        end_time=utterance.end / 1000,
                        text=utterance.text,
                    )
                )
        elif hasattr(response, "words") and response.words:
            current_text = []
            current_start = None
            current_end = None

            for word in response.words:
                if current_start is None:
                    current_start = word.start / 1000
                current_end = word.end / 1000
                current_text.append(word.text)

                if (
                    word.text.endswith((".", "!", "?"))
                    or len(current_text) >= self.MAX_WORDS_PER_SEGMENT
                ):
                    segments.append(
                        TranscriptionSegment(
                            start_time=current_start,
                            end_time=current_end,
                            text=" ".join(current_text),
                        )
                    )
                    current_text = []
                    current_start = None

            if current_text:
                segments.append(
                    TranscriptionSegment(
                        start_time=current_start or 0,
                        end_time=current_end or 0,
                        text=" ".join(current_text),
                    )
                )
        return segments
