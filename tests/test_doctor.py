"""Tests for doctor module dependency verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from audiorag.core.config import AudioRAGConfig


@dataclass
class MockConfig:
    """Mock config for testing."""

    js_runtime: str | None = "deno"

    @property
    def advanced(self):
        return MockAdvanced(self.js_runtime)


@dataclass
class MockAdvanced:
    """Mock advanced config for testing."""

    js_runtime: str | None = "deno"


class TestCheckDependencies:
    """Tests for check_dependencies function."""

    def test_all_dependencies_found(self) -> None:
        """Test when all dependencies are available in PATH."""
        from audiorag.core.doctor import check_dependencies, DependencyCheck, DoctorResult

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda name: f"/usr/bin/{name}"
            result = check_dependencies()

        assert isinstance(result, DoctorResult)
        assert len(result.checks) == 4
        assert all(check.available for check in result.checks)
        assert result.all_ok is True

        check_names = [check.name for check in result.checks]
        assert "ffmpeg" in check_names
        assert "ffprobe" in check_names
        assert "yt-dlp" in check_names
        assert "deno" in check_names

    def test_ffmpeg_missing(self) -> None:
        """Test when ffmpeg is not available."""
        from audiorag.core.doctor import check_dependencies

        def mock_which(name: str) -> str | None:
            if name == "ffmpeg":
                return None
            return f"/usr/bin/{name}"

        with patch("shutil.which", side_effect=mock_which):
            result = check_dependencies()

        ffmpeg_check = next(c for c in result.checks if c.name == "ffmpeg")
        assert ffmpeg_check.available is False
        assert ffmpeg_check.path is None
        assert result.all_ok is False

    def test_ffprobe_missing(self) -> None:
        """Test when ffprobe is not available."""
        from audiorag.core.doctor import check_dependencies

        def mock_which(name: str) -> str | None:
            if name == "ffprobe":
                return None
            return f"/usr/bin/{name}"

        with patch("shutil.which", side_effect=mock_which):
            result = check_dependencies()

        ffprobe_check = next(c for c in result.checks if c.name == "ffprobe")
        assert ffprobe_check.available is False
        assert ffprobe_check.path is None
        assert result.all_ok is False

    def test_yt_dlp_missing(self) -> None:
        """Test when yt-dlp is not available."""
        from audiorag.core.doctor import check_dependencies

        def mock_which(name: str) -> str | None:
            if name == "yt-dlp":
                return None
            return f"/usr/bin/{name}"

        with patch("shutil.which", side_effect=mock_which):
            result = check_dependencies()

        ytdlp_check = next(c for c in result.checks if c.name == "yt-dlp")
        assert ytdlp_check.available is False
        assert ytdlp_check.path is None
        assert result.all_ok is False

    def test_js_runtime_missing(self) -> None:
        """Test when configured js_runtime is not available."""
        from audiorag.core.doctor import check_dependencies

        def mock_which(name: str) -> str | None:
            if name == "deno":
                return None
            return f"/usr/bin/{name}"

        with patch("shutil.which", side_effect=mock_which):
            result = check_dependencies()

        deno_check = next(c for c in result.checks if c.name == "deno")
        assert deno_check.available is False
        assert deno_check.path is None
        assert result.all_ok is False

    def test_js_runtime_none_configured(self) -> None:
        """Test when js_runtime is None - should skip check."""
        from audiorag.core.doctor import check_dependencies

        config = MockConfig(js_runtime=None)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda name: f"/usr/bin/{name}"
            result = check_dependencies(config)

        check_names = [check.name for check in result.checks]
        assert "deno" not in check_names
        assert "node" not in check_names
        assert "bun" not in check_names
        assert len(result.checks) == 3  # Only ffmpeg, ffprobe, yt-dlp
        assert result.all_ok is True

    def test_returns_structured_result(self) -> None:
        """Test that function returns properly structured DoctorResult."""
        from audiorag.core.doctor import check_dependencies, DependencyCheck, DoctorResult

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda name: f"/usr/bin/{name}"
            result = check_dependencies()

        assert isinstance(result, DoctorResult)
        assert isinstance(result.checks, list)
        assert all(isinstance(check, DependencyCheck) for check in result.checks)
        assert all(hasattr(check, "name") for check in result.checks)
        assert all(hasattr(check, "available") for check in result.checks)
        assert all(hasattr(check, "path") for check in result.checks)
        assert all(hasattr(check, "required") for check in result.checks)
        assert hasattr(result, "all_ok")
        assert isinstance(result.all_ok, bool)

    def test_custom_js_runtime_checked(self) -> None:
        """Test that custom js_runtime from config is checked, not default."""
        from audiorag.core.doctor import check_dependencies

        config = MockConfig(js_runtime="node")

        checked_tools: list[str] = []

        def mock_which(name: str) -> str | None:
            checked_tools.append(name)
            return f"/usr/bin/{name}"

        with patch("shutil.which", side_effect=mock_which):
            result = check_dependencies(config)

        assert "node" in checked_tools
        assert "deno" not in checked_tools
        assert "bun" not in checked_tools

        node_check = next(c for c in result.checks if c.name == "node")
        assert node_check.available is True

    def test_dependency_check_required_field(self) -> None:
        """Test that required field is present in DependencyCheck."""
        from audiorag.core.doctor import check_dependencies

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda name: f"/usr/bin/{name}"
            result = check_dependencies()

        for check in result.checks:
            assert isinstance(check.required, bool)


class TestDependencyCheck:
    """Tests for DependencyCheck dataclass."""

    def test_dependency_check_creation(self) -> None:
        """Test DependencyCheck can be created with all fields."""
        from audiorag.core.doctor import DependencyCheck

        check = DependencyCheck(
            name="ffmpeg",
            available=True,
            path="/usr/bin/ffmpeg",
            required=True,
        )

        assert check.name == "ffmpeg"
        assert check.available is True
        assert check.path == "/usr/bin/ffmpeg"
        assert check.required is True

    def test_dependency_check_missing_path(self) -> None:
        """Test DependencyCheck with missing tool."""
        from audiorag.core.doctor import DependencyCheck

        check = DependencyCheck(
            name="missing_tool",
            available=False,
            path=None,
            required=True,
        )

        assert check.available is False
        assert check.path is None


class TestDoctorResult:
    """Tests for DoctorResult dataclass."""

    def test_all_ok_true_when_all_available(self) -> None:
        """Test all_ok is True when all dependencies are available."""
        from audiorag.core.doctor import DependencyCheck, DoctorResult

        checks = [
            DependencyCheck(name="ffmpeg", available=True, path="/usr/bin/ffmpeg", required=True),
            DependencyCheck(name="ffprobe", available=True, path="/usr/bin/ffprobe", required=True),
        ]
        result = DoctorResult(checks=checks)

        assert result.all_ok is True

    def test_all_ok_false_when_one_missing(self) -> None:
        """Test all_ok is False when any dependency is missing."""
        from audiorag.core.doctor import DependencyCheck, DoctorResult

        checks = [
            DependencyCheck(name="ffmpeg", available=True, path="/usr/bin/ffmpeg", required=True),
            DependencyCheck(name="missing", available=False, path=None, required=True),
        ]
        result = DoctorResult(checks=checks)

        assert result.all_ok is False

    def test_all_ok_true_when_empty(self) -> None:
        """Test all_ok is True when checks list is empty."""
        from audiorag.core.doctor import DoctorResult

        result = DoctorResult(checks=[])

        assert result.all_ok is True


class TestDoctorCLI:
    """Tests for doctor CLI command."""

    def test_doctor_command_registered(self) -> None:
        """Test that doctor command is registered in argparse."""
        from audiorag.cli import main
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Test --help includes doctor command
            with pytest.raises(SystemExit) as exc_info:
                sys.argv = ["audiorag", "--help"]
                main()
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

        assert "doctor" in output.lower()
        assert exc_info.value.code == 0

    def test_doctor_exit_code_0_all_found(self) -> None:
        """Test doctor command exits 0 when all deps found."""
        from audiorag.cli import doctor_cmd

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda name: f"/usr/bin/{name}"

            # Should not raise SystemExit(1)
            try:
                doctor_cmd()
            except SystemExit as e:
                assert e.code == 0

    def test_doctor_exit_code_1_missing_dep(self) -> None:
        """Test doctor command exits 1 when dependency missing."""
        from audiorag.cli import doctor_cmd

        def mock_which(name: str) -> str | None:
            if name == "ffmpeg":
                return None
            return f"/usr/bin/{name}"

        with patch("shutil.which", side_effect=mock_which):
            with pytest.raises(SystemExit) as exc_info:
                doctor_cmd()

            assert exc_info.value.code == 1
