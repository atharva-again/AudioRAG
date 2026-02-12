"""Tests for source discovery and YouTube source functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from audiorag.core.exceptions import DiscoveryError


class TestDiscoveryError:
    """Test DiscoveryError exception class."""

    def test_discovery_error_creation(self) -> None:
        """Test DiscoveryError can be created with message and URL."""
        error = DiscoveryError("Failed to expand playlist", url="https://example.com")
        assert str(error) == "Failed to expand playlist"
        assert error.url == "https://example.com"

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


class TestIsYouTubeCollection:
    """Test _is_youtube_collection helper function."""

    def test_playlist_urls(self) -> None:
        """Test detection of playlist URLs."""
        from audiorag.source.discovery import _is_youtube_collection

        playlist_urls = [
            "https://www.youtube.com/playlist?list=PLxxxxxxxxxx",
            "https://youtube.com/playlist?list=PLxxxxxxxxxx",
            "https://www.youtube.com/watch?v=abc123&list=PLxxxxxxxxxx",
        ]
        for url in playlist_urls:
            assert _is_youtube_collection(url) is True, f"Failed for {url}"

    def test_channel_urls(self) -> None:
        """Test detection of channel URLs."""
        from audiorag.source.discovery import _is_youtube_collection

        channel_urls = [
            "https://www.youtube.com/channel/UCxxxxxxxxxx",
            "https://youtube.com/channel/UCxxxxxxxxxx",
            "https://www.youtube.com/c/ChannelName",
            "https://www.youtube.com/user/username",
            "https://www.youtube.com/@handle",
            "https://youtube.com/@handle",
        ]
        for url in channel_urls:
            assert _is_youtube_collection(url) is True, f"Failed for {url}"

    def test_video_urls_not_collections(self) -> None:
        """Test that single video URLs are not detected as collections."""
        from audiorag.source.discovery import _is_youtube_collection

        video_urls = [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "https://youtube.com/watch?v=xyz789",
        ]
        for url in video_urls:
            assert _is_youtube_collection(url) is False, f"Failed for {url}"

    def test_video_with_playlist_in_title(self) -> None:
        """Test videos with 'playlist' in title are not treated as playlists."""
        from audiorag.source.discovery import _is_youtube_collection

        # Video URL with "playlist" in the path (but not /playlist)
        url = "https://www.youtube.com/watch?v=abc123&title=my+playlist+video"
        assert _is_youtube_collection(url) is False


class TestExpandYouTubeSource:
    """Test _expand_youtube_source function."""

    @pytest.mark.asyncio
    async def test_raises_discovery_error_for_empty_playlist(self) -> None:
        """Test that DiscoveryError is raised when playlist returns no videos."""
        from audiorag.source.discovery import _expand_youtube_source

        mock_scraper = MagicMock()
        mock_scraper.list_channel_videos = AsyncMock(return_value=[])

        with patch("audiorag.source.youtube.YouTubeSource", return_value=mock_scraper):
            with pytest.raises(DiscoveryError) as exc_info:
                await _expand_youtube_source(
                    "https://www.youtube.com/playlist?list=PLtest",
                    None,
                )

        assert "Failed to expand YouTube source" in str(exc_info.value)
        assert exc_info.value.url == "https://www.youtube.com/playlist?list=PLtest"

    @pytest.mark.asyncio
    async def test_raises_discovery_error_for_empty_channel(self) -> None:
        """Test that DiscoveryError is raised when channel returns no videos."""
        from audiorag.source.discovery import _expand_youtube_source

        mock_scraper = MagicMock()
        mock_scraper.list_channel_videos = AsyncMock(return_value=[])

        with patch("audiorag.source.youtube.YouTubeSource", return_value=mock_scraper):
            with pytest.raises(DiscoveryError) as exc_info:
                await _expand_youtube_source(
                    "https://www.youtube.com/@testchannel",
                    None,
                )

        assert "Failed to expand YouTube source" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_original_url_for_single_video(self) -> None:
        """Test that single video URLs return themselves on empty result."""
        from audiorag.source.discovery import _expand_youtube_source

        mock_scraper = MagicMock()
        mock_scraper.list_channel_videos = AsyncMock(return_value=[])

        with patch("audiorag.source.youtube.YouTubeSource", return_value=mock_scraper):
            result = await _expand_youtube_source(
                "https://www.youtube.com/watch?v=abc123",
                None,
            )

        assert result == ["https://www.youtube.com/watch?v=abc123"]

    @pytest.mark.asyncio
    async def test_returns_expanded_videos_on_success(self) -> None:
        """Test successful expansion returns video URLs."""
        from audiorag.source.discovery import _expand_youtube_source

        mock_video = MagicMock()
        mock_video.url = "https://www.youtube.com/watch?v=video1"

        mock_scraper = MagicMock()
        mock_scraper.list_channel_videos = AsyncMock(return_value=[mock_video])

        with patch("audiorag.source.youtube.YouTubeSource", return_value=mock_scraper):
            result = await _expand_youtube_source(
                "https://www.youtube.com/playlist?list=PLtest",
                None,
            )

        assert result == ["https://www.youtube.com/watch?v=video1"]


class TestYouTubeSourcePlayerClients:
    """Test YouTubeSource player_clients handling."""

    def test_player_clients_uses_default_when_none(self) -> None:
        """Test default clients are used when None is passed."""
        from audiorag.source.youtube import YouTubeSource

        source = YouTubeSource(player_clients=None)
        assert source._player_clients == ["tv", "web", "mweb"]

    def test_player_clients_accepts_empty_list(self) -> None:
        """Test empty list is preserved and not overridden."""
        from audiorag.source.youtube import YouTubeSource

        source = YouTubeSource(player_clients=[])
        assert source._player_clients == []

    def test_player_clients_accepts_custom_list(self) -> None:
        """Test custom client list is preserved."""
        from audiorag.source.youtube import YouTubeSource

        custom_clients = ["android", "ios"]
        source = YouTubeSource(player_clients=custom_clients)
        assert source._player_clients == custom_clients

    def test_player_clients_passed_to_extractor_args(self) -> None:
        """Test player_clients are correctly passed to yt-dlp options."""
        from audiorag.source.youtube import YouTubeSource

        custom_clients = ["tv", "android"]
        source = YouTubeSource(player_clients=custom_clients)
        opts = source._get_base_ydl_opts()

        assert opts["extractor_args"]["youtube"]["player_client"] == custom_clients

    def test_empty_player_clients_passed_to_extractor_args(self) -> None:
        """Test empty list is passed to yt-dlp options."""
        from audiorag.source.youtube import YouTubeSource

        source = YouTubeSource(player_clients=[])
        opts = source._get_base_ydl_opts()

        assert opts["extractor_args"]["youtube"]["player_client"] == []


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

        result = await discover_sources([str(audio_dir)], None)

        assert len(result) == 2
        assert all("test1.mp3" in r or "test2.wav" in r for r in result)

    @pytest.mark.asyncio
    async def test_deduplicates_sources(self) -> None:
        """Test duplicate sources are removed."""
        from audiorag.source.discovery import discover_sources

        result = await discover_sources(
            ["https://example.com/audio.mp3", "https://example.com/audio.mp3"],
            None,
        )

        assert len(result) == 1
        assert result[0] == "https://example.com/audio.mp3"

    @pytest.mark.asyncio
    async def test_propagates_discovery_error(self) -> None:
        """Test DiscoveryError is propagated from playlist expansion."""
        from audiorag.source.discovery import discover_sources

        mock_scraper = MagicMock()
        mock_scraper.list_channel_videos = AsyncMock(return_value=[])

        with patch("audiorag.source.youtube.YouTubeSource", return_value=mock_scraper):
            with pytest.raises(DiscoveryError):
                await discover_sources(
                    ["https://www.youtube.com/playlist?list=PLtest"],
                    None,
                )

    @pytest.mark.asyncio
    async def test_preserves_non_youtube_urls(self) -> None:
        """Test non-YouTube URLs are preserved as-is."""
        from audiorag.source.discovery import discover_sources

        result = await discover_sources(
            ["https://example.com/audio.mp3", "https://another.com/song.wav"],
            None,
        )

        assert len(result) == 2
        assert "https://example.com/audio.mp3" in result
        assert "https://another.com/song.wav" in result
