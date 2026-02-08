"""Deepgram Speech-to-Text provider using Nova-2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audiorag.core.logging_config import get_logger
from audiorag.core.models import TranscriptionSegment
from audiorag.transcribe._base import TranscriberMixin

logger = get_logger(__name__)


class DeepgramTranscriber(TranscriberMixin):
    """Speech-to-Text provider using Deepgram's Nova-2 API."""

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

    MAX_WORDS_PER_SEGMENT = 20

    _provider_name: str = "deepgram_stt"
    _retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        RuntimeError,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "nova-2",
        retry_config: Any | None = None,
    ) -> None:
        """Initialize Deepgram STT provider."""
        super().__init__(api_key=api_key, model=model, retry_config=retry_config)
        from deepgram import DeepgramClient  # type: ignore[import]

        self.client = DeepgramClient(api_key)

    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file using Deepgram API."""
        operation_logger = self._logger.bind(
            audio_path=str(audio_path),
            language=language,
            operation="transcribe",
        )
        operation_logger.debug("transcription_started")

        retry_decorator = self._get_retry_decorator()

        @retry_decorator
        async def _transcribe_with_retry() -> Any:
            from deepgram import PrerecordedOptions  # type: ignore[import]

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
        """Extract segments from Deepgram response."""
        segments = []
        if response.results and response.results.channels:
            channel = response.results.channels[0]
            if channel.alternatives:
                alternative = channel.alternatives[0]

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
                    segments = self._segment_by_words(alternative.words)
                else:
                    segments.append(
                        TranscriptionSegment(
                            start_time=0,
                            end_time=response.results.channels[0].alternatives[0].words[-1].end
                            if response.results.channels[0].alternatives[0].words
                            else 0,
                            text=alternative.transcript,
                        )
                    )
        return segments

    def _segment_by_words(self, words: list[Any]) -> list[TranscriptionSegment]:
        """Segment words into groups based on boundaries."""
        segments = []
        current_text = []
        current_start = None
        current_end = None

        for word in words:
            if current_start is None:
                current_start = word.start
            current_end = word.end
            current_text.append(word.word)

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

        if current_text:
            segments.append(
                TranscriptionSegment(
                    start_time=current_start or 0,
                    end_time=current_end or 0,
                    text=" ".join(current_text),
                )
            )
        return segments
