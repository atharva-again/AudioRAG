from pathlib import Path
from typing import Protocol, runtime_checkable

from audiorag.models import AudioFile


@runtime_checkable
class AudioSourceProvider(Protocol):
    async def download(
        self, url: str, output_dir: Path, audio_format: str = "mp3"
    ) -> AudioFile: ...
