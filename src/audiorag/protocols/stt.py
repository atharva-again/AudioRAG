from pathlib import Path
from typing import Protocol, runtime_checkable

from audiorag.models import TranscriptionSegment


@runtime_checkable
class STTProvider(Protocol):
    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptionSegment]: ...
