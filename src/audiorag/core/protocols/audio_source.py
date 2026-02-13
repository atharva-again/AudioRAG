from pathlib import Path
from typing import Protocol, runtime_checkable

from audiorag.core.models import AudioFile, SourceMetadata


@runtime_checkable
class AudioSourceProvider(Protocol):
    async def download(
        self,
        url: str,
        output_dir: Path,
        audio_format: str = "mp3",
        metadata: SourceMetadata | None = None,
    ) -> AudioFile: ...

    async def get_metadata(self, url: str) -> SourceMetadata | None: ...
