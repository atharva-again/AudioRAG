"""Tests for source discovery functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from audiorag.core.exceptions import DiscoveryError


class TestDiscoveryError:
    """Test DiscoveryError exception class."""

    def test_discovery_error_creation(self) -> None:
        """Test DiscoveryError can be created with message and URL."""
        error = DiscoveryError("Failed to expand directory", url="/path/to/dir")
        assert str(error) == "Failed to expand directory"
        assert error.url == "/path/to/dir"

    def test_discovery_error_without_url(self) -> None:
        """Test DiscoveryError works without URL."""
        error = DiscoveryError("Generic discovery failure")
        assert str(error) == "Generic discovery failure"
        assert error.url is None

    def test_discovery_error_inheritance(self) -> None:
        """Test DiscoveryError inherits from AudioRAGError."""
        from audiorag.core.exceptions import AudioRAGError

        error = DiscoveryError("Test")
        assert isinstance(error, AudioRAGError)


class TestDiscoverSources:
    """Test discover_sources function."""

    @pytest.mark.asyncio
    async def test_expands_directory(self, tmp_path: Path) -> None:
        """Test directory expansion finds audio files."""
        from audiorag.source.discovery import discover_sources

        # Create test audio files
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "test1.mp3").write_text("fake audio")
        (audio_dir / "test2.wav").write_text("fake audio")

        result = await discover_sources([str(audio_dir)])

        assert len(result) == 2
        urls = [r.url for r in result]
        assert all("test1.mp3" in url or "test2.wav" in url for url in urls)

    @pytest.mark.asyncio
    async def test_deduplicates_sources(self) -> None:
        """Test duplicate sources are removed."""
        from audiorag.source.discovery import discover_sources

        result = await discover_sources(
            ["/path/to/audio.mp3", "/path/to/audio.mp3"],
        )

        assert len(result) == 1
        assert result[0].url == "/path/to/audio.mp3"

    @pytest.mark.asyncio
    async def test_preserves_file_paths(self) -> None:
        """Test file paths are preserved as-is."""
        from audiorag.source.discovery import discover_sources

        result = await discover_sources(
            ["/path/to/audio.mp3", "/another/path/song.wav"],
        )

        assert len(result) == 2
        urls = [r.url for r in result]
        assert "/path/to/audio.mp3" in urls
        assert "/another/path/song.wav" in urls
