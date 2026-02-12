from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from audiorag.core.exceptions import ConfigurationError


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _budget_module():
    try:
        return importlib.import_module("audiorag.core.budget")
    except ModuleNotFoundError as exc:
        raise AssertionError("audiorag.core.budget module is missing") from exc


def create_governor(
    *,
    clock: FakeClock,
    enabled: bool = True,
    global_limits: Any | None = None,
    overrides: dict[str, Any] | None = None,
    token_chars_per_token: int = 4,
    db_path: str | Path | None = None,
) -> Any:
    module = _budget_module()
    return module.BudgetGovernor(
        enabled=enabled,
        global_limits=global_limits,
        provider_overrides=overrides or {},
        token_chars_per_token=token_chars_per_token,
        now=clock,
        db_path=db_path,
    )


class TestBudgetGovernorDefaults:
    def test_disabled_governor_allows_any_usage(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            enabled=False,
            global_limits=module.BudgetLimits(rpm=1, tpm=1, audio_seconds_per_hour=1),
        )

        governor.reserve(provider="openai", requests=10, tokens=10_000, audio_seconds=10_000)

    def test_no_limits_configured_allows_any_usage(self) -> None:
        clock = FakeClock()
        governor = create_governor(clock=clock, global_limits=None)

        governor.reserve(provider="openai", requests=10, tokens=10_000, audio_seconds=10_000)


class TestBudgetGovernorRateLimits:
    def test_rpm_blocks_second_request_within_window(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(clock=clock, global_limits=module.BudgetLimits(rpm=1))

        governor.reserve(provider="openai", requests=1)

        module = _budget_module()
        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="openai", requests=1)

    def test_rpm_resets_after_window(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(clock=clock, global_limits=module.BudgetLimits(rpm=1))

        governor.reserve(provider="openai", requests=1)

        clock.advance(61)

        governor.reserve(provider="openai", requests=1)

    def test_tpm_uses_explicit_token_count(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(clock=clock, global_limits=module.BudgetLimits(tpm=10))

        governor.reserve(provider="openai", tokens=8)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="openai", tokens=3)

    def test_tpm_estimates_from_text_chars_when_tokens_missing(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(clock=clock, global_limits=module.BudgetLimits(tpm=10))

        governor.reserve(provider="openai", text_chars=40)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="openai", text_chars=4)

    def test_audio_seconds_per_hour_blocks_when_exceeded(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=300),
        )

        governor.reserve(provider="deepgram", audio_seconds=200)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="deepgram", audio_seconds=150)

    def test_audio_seconds_resets_after_window(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=300),
        )

        governor.reserve(provider="deepgram", audio_seconds=250)
        clock.advance(3601)
        governor.reserve(provider="deepgram", audio_seconds=250)

    def test_provider_overrides_take_precedence(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(rpm=10),
            overrides={"openai": module.BudgetLimits(rpm=1)},
        )

        governor.reserve(provider="openai", requests=1)
        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="openai", requests=1)

        governor.reserve(provider="cohere", requests=1)
        governor.reserve(provider="cohere", requests=1)

    @pytest.mark.parametrize(
        "provider",
        [
            "openai",
            "cohere",
            "voyage",
            "deepgram",
            "assemblyai",
            "groq",
            "anthropic",
            "gemini",
        ],
    )
    def test_unknown_provider_uses_global_limits(self, provider: str) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(clock=clock, global_limits=module.BudgetLimits(rpm=2))

        governor.reserve(provider=provider, requests=1)
        governor.reserve(provider=provider, requests=1)


class TestBudgetGovernorConfigIntegration:
    def test_from_config_creates_disabled_governor(self, mock_config) -> None:
        mock_config.budget_enabled = False

        module = _budget_module()
        governor = module.BudgetGovernor.from_config(mock_config)
        governor.reserve(provider="openai", requests=100)

    def test_from_config_uses_global_limits(self, mock_config) -> None:
        mock_config.budget_enabled = True
        mock_config.budget_rpm = 1
        mock_config.budget_tpm = 10
        mock_config.budget_audio_seconds_per_hour = 300
        mock_config.budget_token_chars_per_token = 4
        mock_config.budget_provider_overrides = {}

        module = _budget_module()
        governor = module.BudgetGovernor.from_config(mock_config)

        governor.reserve(provider="openai", requests=1)
        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="openai", requests=1)


class TestBudgetGovernorDurability:
    def test_limits_persist_across_instances(self, tmp_path: Path) -> None:
        clock = FakeClock()
        module = _budget_module()
        db_path = tmp_path / "budget.db"

        governor_1 = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(rpm=1),
            db_path=db_path,
        )
        governor_1.reserve(provider="openai", requests=1)

        governor_2 = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(rpm=1),
            db_path=db_path,
        )
        with pytest.raises(module.BudgetExceededError):
            governor_2.reserve(provider="openai", requests=1)

    def test_persistent_reserve_is_atomic_across_metrics(self, tmp_path: Path) -> None:
        clock = FakeClock()
        module = _budget_module()
        db_path = tmp_path / "budget.db"

        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(rpm=1, tpm=10),
            db_path=db_path,
        )
        governor.reserve(provider="openai", requests=1, tokens=8)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="openai", requests=1, tokens=1)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="openai", requests=1)


class TestBudgetGovernorRelease:
    def test_release_allows_subsequent_reservation(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=100),
        )

        governor.reserve(provider="deepgram", audio_seconds=90)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="deepgram", audio_seconds=20)

        governor.release(provider="deepgram", audio_seconds=50)

        governor.reserve(provider="deepgram", audio_seconds=40)

    def test_release_with_disabled_governor_is_noop(self) -> None:
        clock = FakeClock()
        governor = create_governor(clock=clock, enabled=False)

        governor.release(provider="openai", audio_seconds=100)

    def test_release_zero_seconds_is_noop(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=100),
        )

        governor.reserve(provider="deepgram", audio_seconds=100)

        governor.release(provider="deepgram", audio_seconds=0)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="deepgram", audio_seconds=1)

    def test_release_negative_seconds_is_noop(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=100),
        )

        governor.reserve(provider="deepgram", audio_seconds=100)

        governor.release(provider="deepgram", audio_seconds=-10)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="deepgram", audio_seconds=1)

    def test_release_persists_across_instances(self, tmp_path: Path) -> None:
        clock = FakeClock()
        module = _budget_module()
        db_path = tmp_path / "budget.db"

        governor_1 = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=100),
            db_path=db_path,
        )
        governor_1.reserve(provider="deepgram", audio_seconds=90)

        governor_1.release(provider="deepgram", audio_seconds=50)

        governor_2 = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=100),
            db_path=db_path,
        )
        governor_2.reserve(provider="deepgram", audio_seconds=50)

    def test_multiple_releases_accumulate(self) -> None:
        clock = FakeClock()
        module = _budget_module()
        governor = create_governor(
            clock=clock,
            global_limits=module.BudgetLimits(audio_seconds_per_hour=100),
        )

        governor.reserve(provider="deepgram", audio_seconds=100)

        with pytest.raises(module.BudgetExceededError):
            governor.reserve(provider="deepgram", audio_seconds=1)

        governor.release(provider="deepgram", audio_seconds=20)
        governor.release(provider="deepgram", audio_seconds=30)

        governor.reserve(provider="deepgram", audio_seconds=45)


class TestBudgetGovernorValidation:
    @pytest.mark.parametrize("rpm", [-1, 0])
    def test_rpm_must_be_positive(self, rpm: int) -> None:
        module = _budget_module()
        with pytest.raises(ConfigurationError):
            module.BudgetLimits(rpm=rpm)

    @pytest.mark.parametrize("tpm", [-10, 0])
    def test_tpm_must_be_positive(self, tpm: int) -> None:
        module = _budget_module()
        with pytest.raises(ConfigurationError):
            module.BudgetLimits(tpm=tpm)

    @pytest.mark.parametrize("audio_seconds", [-5, 0])
    def test_audio_seconds_per_hour_must_be_positive(self, audio_seconds: int) -> None:
        module = _budget_module()
        with pytest.raises(ConfigurationError):
            module.BudgetLimits(audio_seconds_per_hour=audio_seconds)
