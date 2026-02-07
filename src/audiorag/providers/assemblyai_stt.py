"""AssemblyAI Speech-to-Text provider implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.core.models import TranscriptionSegment
from audiorag.core.retry_config import (
    RetryConfig,
    create_retry_decorator,
)

logger = get_logger(__name__)


class AssemblyAISTTProvider:
    """Speech-to-Text provider using AssemblyAI's API.

    AssemblyAI is a top competitor in the STT space in 2026, offering
    high accuracy, speaker diarization, and various audio intelligence features.

    Available models:
        - "best" (default): Best accuracy for most use cases
        - "nano": Fastest, lowest cost option
        - "universal": Balanced accuracy and speed
    """

    MODEL_BEST = "best"
    MODEL_NANO = "nano"
    MODEL_UNIVERSAL = "universal"

    # Maximum words per segment before breaking
    MAX_WORDS_PER_SEGMENT = 20

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "best",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize AssemblyAI STT provider.

        Args:
            api_key: AssemblyAI API key. If None, uses ASSEMBLYAI_API_KEY environment variable.
            model: Model to use for transcription. Options: "best", "nano", "universal".
            retry_config: Retry configuration. Uses default if not provided.
        """
        # Lazy import to avoid ModuleNotFoundError when optional dep not installed
        import assemblyai as aai  # noqa: PLC0415 # type: ignore

        self._aai = aai
        aai.settings.api_key = api_key
        self._model = model
        self._logger = logger.bind(provider="assemblyai_stt", model=model)
        self._retry_config = retry_config or RetryConfig()
        self._transcriber = aai.Transcriber()

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name - switch models instantly.

        Args:
            value: Model name. Options: "best", "nano", "universal".
        """
        self._model = value
        self._logger = logger.bind(provider="assemblyai_stt", model=value)

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for AssemblyAI API calls."""
        return create_retry_decorator(
            config=self._retry_config,
            exception_types=(
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ),
        )

    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file using AssemblyAI API.

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
            # Map model string to SpeechModel enum
            model_map = {
                "best": self._aai.SpeechModel.best,
                "nano": self._aai.SpeechModel.nano,
                "universal": getattr(
                    self._aai.SpeechModel, "universal", self._aai.SpeechModel.best
                ),
            }
            speech_model = model_map.get(self._model, self._aai.SpeechModel.best)

            config = self._aai.TranscriptionConfig(
                speech_model=speech_model,
                language_code=language,
                punctuate=True,
                format_text=True,
                speaker_labels=True,
            )

            return self._transcriber.transcribe(
                str(audio_path),
                config=config,
            )

        try:
            transcript = await _transcribe_with_retry()

            # Extract segments from response
            segments = []
            if transcript.status == self._aai.TranscriptStatus.completed:
                if transcript.utterances:
                    # Use speaker-labeled utterances if available
                    for utterance in transcript.utterances:
                        segments.append(
                            TranscriptionSegment(
                                start_time=utterance.start / 1000,  # Convert ms to seconds
                                end_time=utterance.end / 1000,
                                text=utterance.text,
                            )
                        )
                elif transcript.words:
                    # Fallback to word-level segmentation
                    current_text = []
                    current_start = None
                    current_end = None

                    for word in transcript.words:
                        if current_start is None:
                            current_start = word.start / 1000
                        current_end = word.end / 1000
                        current_text.append(word.text)

                        # Break at sentence boundaries or max word threshold
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

                    # Add remaining words
                    if current_text:
                        segments.append(
                            TranscriptionSegment(
                                start_time=current_start or 0,
                                end_time=current_end or 0,
                                text=" ".join(current_text),
                            )
                        )
                else:
                    # Fallback to full transcript
                    segments.append(
                        TranscriptionSegment(
                            start_time=0,
                            end_time=transcript.audio_duration or 0,
                            text=transcript.text or "",
                        )
                    )
            else:
                raise RuntimeError(f"Transcription failed with status: {transcript.status}")

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
