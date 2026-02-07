"""Deepgram Speech-to-Text provider implementation."""

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


class DeepgramSTTProvider:
    """Speech-to-Text provider using Deepgram's Nova-2 API.

    Deepgram is one of the most popular STT providers in 2026, known for
    high accuracy, low latency, and support for multiple languages.

    Available models:
        - "nova-2" (default): Latest Nova model with best accuracy
        - "nova-2-general": General purpose model
        - "nova-2-meeting": Optimized for meetings
        - "nova-2-phonecall": Optimized for phone calls
        - "nova-2-voicemail": Optimized for voicemail
        - "nova-2-finance": Optimized for finance
        - "nova-2-conversationalai": Optimized for conversational AI
        - "nova-2-video": Optimized for video content
        - "nova-2-medical": Optimized for medical terminology
        - "nova-1": Previous generation Nova model
        - "enhanced": Enhanced model (English only)
        - "base": Base model (lowest cost)
    """

    MODEL_NOVA_2 = "nova-2"
    MODEL_NOVA_2_GENERAL = "nova-2-general"
    MODEL_NOVA_2_MEETING = "nova-2-meeting"
    MODEL_NOVA_2_PHONECALL = "nova-2-phonecall"
    MODEL_NOVA_2_VOICEMAIL = "nova-2-voicemail"
    MODEL_NOVA_2_FINANCE = "nova-2-finance"
    MODEL_NOVA_2_CONVERSATIONAL = "nova-2-conversationalai"
    MODEL_NOVA_2_VIDEO = "nova-2-video"
    MODEL_NOVA_2_MEDICAL = "nova-2-medical"
    MODEL_NOVA_1 = "nova-1"
    MODEL_ENHANCED = "enhanced"
    MODEL_BASE = "base"

    # Maximum words per segment before breaking
    MAX_WORDS_PER_SEGMENT = 20

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-2",
        retry_config: RetryConfig | None = None,
    ) -> Any:
        """Initialize Deepgram STT provider.

        Args:
            api_key: Deepgram API key. If None, uses DEEPGRAM_API_KEY environment variable.
            model: Model to use for transcription. Defaults to "nova-2".
                Options: "nova-2", "nova-2-general", "nova-2-meeting", "nova-2-phonecall",
                "nova-2-voicemail", "nova-2-finance", "nova-2-conversationalai",
                "nova-2-video", "nova-2-medical", "nova-1", "enhanced", "base".
            retry_config: Retry configuration. Uses default if not provided.
        """
        # Lazy import to avoid ModuleNotFoundError when optional dep not installed
        from deepgram import DeepgramClient  # noqa: PLC0415 # type: ignore

        self.client = DeepgramClient(api_key)
        self._model = model
        self._logger = logger.bind(provider="deepgram_stt", model=model)
        self._retry_config = retry_config or RetryConfig()

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name - switch models instantly.

        Args:
            value: Model name. Any valid Deepgram model string.
        """
        self._model = value
        self._logger = logger.bind(provider="deepgram_stt", model=value)

    def _get_retry_decorator(self) -> Any:
        """Get retry decorator configured for Deepgram API calls."""
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
        """Transcribe audio file using Deepgram API.

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
            from deepgram import PrerecordedOptions  # noqa: PLC0415 # type: ignore

            with open(audio_path, "rb") as audio_file:
                buffer_data = audio_file.read()

            options = PrerecordedOptions(
                model=self.model,
                language=language,
                punctuate=True,
                paragraphs=True,
                diarize=True,
            )

            return self.client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": buffer_data}, options
            )

        try:
            response = await _transcribe_with_retry()

            # Extract segments from response
            segments = []
            if response.results and response.results.channels:
                channel = response.results.channels[0]
                if channel.alternatives:
                    alternative = channel.alternatives[0]

                    # Get paragraphs for better segmentation
                    if alternative.paragraphs and alternative.paragraphs.paragraphs:
                        for paragraph in alternative.paragraphs.paragraphs:
                            for sentence in paragraph.sentences:
                                segments.append(
                                    TranscriptionSegment(
                                        start_time=sentence.start,
                                        end_time=sentence.end,
                                        text=sentence.text,
                                    )
                                )
                    elif alternative.words:
                        # Fallback to word-level segmentation if no paragraphs
                        current_text = []
                        current_start = None
                        current_end = None

                        for word in alternative.words:
                            if current_start is None:
                                current_start = word.start
                            current_end = word.end
                            current_text.append(word.word)

                            # Break at sentence boundaries or max word threshold
                            if (
                                word.word.endswith((".", "!", "?"))
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
                                end_time=response.results.channels[0].alternatives[0].words[-1].end
                                if response.results.channels[0].alternatives[0].words
                                else 0,
                                text=alternative.transcript,
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
