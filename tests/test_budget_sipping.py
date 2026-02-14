"""Tests for budget sipping functionality."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiorag.core.budget import BudgetGovernor, BudgetLimits
from audiorag.core.exceptions import BudgetExceededError
from audiorag.pipeline import AudioRAGPipeline, TranscribeStage


@dataclass
class FakeClock:
    """Deterministic clock for testing."""

    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class MockConfig:
    """Mock config for testing."""

    def __init__(
        self,
        budget_enabled: bool = False,
        budget_sip_max_retries: int = 60,
        budget_sip_wait_seconds: float = 5.0,
        stt_provider: str = "openai",
    ) -> None:
        self.budget_enabled = budget_enabled
        self.budget_sip_max_retries = budget_sip_max_retries
        self.budget_sip_wait_seconds = budget_sip_wait_seconds
        self.stt_provider = stt_provider
        self.database_path = ":memory:"
        self.work_dir = Path("/tmp/test")
        self.chunk_duration_seconds = 30
        self.cleanup_audio = False


def create_mock_pipeline_with_budget(
    tmp_path: Path,
    budget_enabled: bool = True,
    audio_limit: int = 100,
    sip_max_retries: int = 60,
) -> tuple[AudioRAGPipeline, BudgetGovernor, FakeClock]:
    """Create a mock pipeline with budget governor."""
    clock = FakeClock()
    db_path = tmp_path / "budget.db"

    budget_governor = BudgetGovernor(
        enabled=budget_enabled,
        global_limits=BudgetLimits(audio_seconds_per_hour=audio_limit),
        provider_overrides={},
        now=clock,
        db_path=db_path,
    )

    config = MagicMock()
    config.budget_enabled = budget_enabled
    config.budget_sip_max_retries = sip_max_retries
    config.budget_sip_wait_seconds = 0.1  # Fast for tests
    config.stt_provider = "openai"
    config.database_path = str(tmp_path / "test.db")
    config.work_dir = tmp_path
    config.chunk_duration_seconds = 30
    config.cleanup_audio = False
    config.log_level = "INFO"
    config.log_format = "colored"
    config.log_timestamps = True
    config.retrieval_top_k = 10
    config.rerank_top_n = 3

    pipeline = MagicMock()
    pipeline._budget_governor = budget_governor
    pipeline._config = config

    return pipeline, budget_governor, clock


class TestBudgetSipping:
    """Tests for budget sipping functionality."""

    @pytest.mark.asyncio
    async def test_sipping_disabled_when_budget_disabled(self, tmp_path: Path) -> None:
        """When budget is disabled, sipping should not trigger."""
        pipeline, _, _ = create_mock_pipeline_with_budget(
            tmp_path, budget_enabled=False, audio_limit=100
        )

        stage = TranscribeStage()
        ctx = MagicMock()
        ctx.config = pipeline._config
        ctx.logger = MagicMock()

        # Should not raise even with large audio
        await stage._sip_reserve(pipeline, "openai", 1000, ctx)

    @pytest.mark.asyncio
    async def test_sipping_no_wait_when_budget_sufficient(self, tmp_path: Path) -> None:
        """When budget is sufficient, no sipping should occur."""
        pipeline, _, _ = create_mock_pipeline_with_budget(
            tmp_path, budget_enabled=True, audio_limit=1000
        )

        stage = TranscribeStage()
        ctx = MagicMock()
        ctx.config = pipeline._config
        ctx.logger = MagicMock()

        # Should complete without sleep
        with patch("asyncio.sleep") as mock_sleep:
            await stage._sip_reserve(pipeline, "openai", 100, ctx)
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_sipping_waits_on_budget_exceeded(self, tmp_path: Path) -> None:
        """Sipping should wait when budget is exceeded."""
        pipeline, governor, clock = create_mock_pipeline_with_budget(
            tmp_path, budget_enabled=True, audio_limit=100, sip_max_retries=3
        )

        # Use up all budget
        governor.reserve(provider="openai", audio_seconds=100)

        stage = TranscribeStage()
        ctx = MagicMock()
        ctx.config = pipeline._config
        ctx.logger = MagicMock()

        sleep_calls = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # Advance clock to reset budget window
            clock.advance(3601)  # Past 1 hour window

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await stage._sip_reserve(pipeline, "openai", 50, ctx)

        # Should have slept at least once
        assert len(sleep_calls) >= 1
        ctx.logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_sipping_gives_up_after_max_retries(self, tmp_path: Path) -> None:
        """Sipping should raise BudgetExceededError after max retries."""
        pipeline, governor, _ = create_mock_pipeline_with_budget(
            tmp_path, budget_enabled=True, audio_limit=100, sip_max_retries=2
        )

        # Use up all budget
        governor.reserve(provider="openai", audio_seconds=100)

        stage = TranscribeStage()
        ctx = MagicMock()
        ctx.config = pipeline._config
        ctx.config.budget_sip_wait_seconds = 0.01  # Fast
        ctx.logger = MagicMock()

        with pytest.raises(BudgetExceededError):
            with patch("asyncio.sleep"):
                await stage._sip_reserve(pipeline, "openai", 50, ctx)

    @pytest.mark.asyncio
    async def test_sipping_respects_cancellation(self, tmp_path: Path) -> None:
        """Sipping should respect asyncio.CancelledError."""
        pipeline, governor, _ = create_mock_pipeline_with_budget(
            tmp_path, budget_enabled=True, audio_limit=100, sip_max_retries=10
        )

        # Use up all budget
        governor.reserve(provider="openai", audio_seconds=100)

        stage = TranscribeStage()
        ctx = MagicMock()
        ctx.config = pipeline._config
        ctx.logger = MagicMock()

        async def mock_sleep_raise_cancelled(_seconds: float) -> None:
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            with patch("asyncio.sleep", side_effect=mock_sleep_raise_cancelled):
                await stage._sip_reserve(pipeline, "openai", 50, ctx)

    @pytest.mark.asyncio
    async def test_sipping_logs_waiting_state(self, tmp_path: Path) -> None:
        """Sipping should log waiting state with retry count."""
        pipeline, governor, clock = create_mock_pipeline_with_budget(
            tmp_path, budget_enabled=True, audio_limit=100, sip_max_retries=2
        )

        # Use up all budget
        governor.reserve(provider="openai", audio_seconds=100)

        stage = TranscribeStage()
        ctx = MagicMock()
        ctx.config = pipeline._config
        ctx.logger = MagicMock()

        async def mock_sleep_advance(_seconds: float) -> None:
            clock.advance(3601)  # Reset budget window

        with patch("asyncio.sleep", side_effect=mock_sleep_advance):
            await stage._sip_reserve(pipeline, "openai", 50, ctx)

        # Should have logged waiting
        log_calls = [
            call for call in ctx.logger.info.call_args_list if "budget_sip_waiting" in str(call)
        ]
        assert len(log_calls) >= 1
