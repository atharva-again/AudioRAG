from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from audiorag.core.exceptions import BudgetExceededError, ConfigurationError

if TYPE_CHECKING:
    from audiorag.core.protocols.budget_store import BudgetStore


@dataclass(slots=True)
class BudgetLimits:
    rpm: int | None = None
    tpm: int | None = None
    audio_seconds_per_hour: int | None = None

    def __post_init__(self) -> None:
        if self.rpm is not None and self.rpm <= 0:
            raise ConfigurationError("budget rpm must be > 0")
        if self.tpm is not None and self.tpm <= 0:
            raise ConfigurationError("budget tpm must be > 0")
        if self.audio_seconds_per_hour is not None and self.audio_seconds_per_hour <= 0:
            raise ConfigurationError("budget audio_seconds_per_hour must be > 0")


class BudgetGovernor:
    def __init__(
        self,
        *,
        enabled: bool,
        global_limits: BudgetLimits | None,
        provider_overrides: dict[str, BudgetLimits],
        token_chars_per_token: int = 4,
        now: Callable[[], float] | None = None,
        db_path: str | Path | None = None,
        store: BudgetStore | None = None,
    ) -> None:
        if token_chars_per_token <= 0:
            raise ConfigurationError("budget token_chars_per_token must be > 0")

        self._enabled = enabled
        self._global_limits = global_limits
        self._provider_overrides = {k.lower(): v for k, v in provider_overrides.items()}
        self._token_chars_per_token = token_chars_per_token
        self._now = now or time.time
        self._db_path = Path(db_path) if db_path is not None else None
        self._rpm_windows: dict[str, deque[tuple[float, int]]] = {}
        self._tpm_windows: dict[str, deque[tuple[float, int]]] = {}
        self._audio_windows: dict[str, deque[tuple[float, int]]] = {}
        self._store: BudgetStore | None = None

        # Store parameter takes precedence over db_path
        if store is not None:
            self._store = store
        elif self._db_path is not None and self._enabled:
            # Initialize SQLite store for backward compatibility
            from audiorag.core.budget_store_sqlite import SqliteBudgetStore

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._store = SqliteBudgetStore(self._db_path)

    @classmethod
    def from_config(cls, config: Any, store: BudgetStore | None = None) -> BudgetGovernor:
        enabled = bool(getattr(config, "budget_enabled", False))
        rpm = getattr(config, "budget_rpm", None)
        tpm = getattr(config, "budget_tpm", None)
        audio = getattr(config, "budget_audio_seconds_per_hour", None)
        chars_per_token = int(getattr(config, "budget_token_chars_per_token", 4))
        raw_overrides = getattr(config, "budget_provider_overrides", {}) or {}

        global_limits: BudgetLimits | None = None
        if rpm is not None or tpm is not None or audio is not None:
            global_limits = BudgetLimits(rpm=rpm, tpm=tpm, audio_seconds_per_hour=audio)

        parsed_overrides: dict[str, BudgetLimits] = {}
        if isinstance(raw_overrides, dict):
            for provider, values in raw_overrides.items():
                if not isinstance(provider, str) or not isinstance(values, dict):
                    continue
                parsed_overrides[provider.lower()] = BudgetLimits(
                    rpm=values.get("rpm"),
                    tpm=values.get("tpm"),
                    audio_seconds_per_hour=values.get("audio_seconds_per_hour"),
                )

        return cls(
            enabled=enabled,
            global_limits=global_limits,
            provider_overrides=parsed_overrides,
            token_chars_per_token=chars_per_token,
            db_path=getattr(config, "database_path", None),
            store=store,
        )

    def reserve(
        self,
        *,
        provider: str,
        requests: int = 0,
        tokens: int | None = None,
        text_chars: int | None = None,
        audio_seconds: int | float = 0,
    ) -> None:
        if not self._enabled:
            return

        provider_key = provider.lower()
        limits = self._provider_overrides.get(provider_key, self._global_limits)
        if limits is None:
            return

        token_units = self._resolve_tokens(tokens=tokens, text_chars=text_chars)
        request_units = max(0, requests)
        # Allow negative audio units for compensating releases
        audio_units = math.ceil(audio_seconds)
        now = self._now()

        limit_entries: list[tuple[str, int, int, int]] = []

        if limits.rpm is not None and request_units > 0:
            limit_entries.append(("rpm", request_units, limits.rpm, 60))

        if limits.tpm is not None and token_units > 0:
            limit_entries.append(("tpm", token_units, limits.tpm, 60))

        if limits.audio_seconds_per_hour is not None and audio_units != 0:
            limit_entries.append(
                (
                    "audio_seconds_per_hour",
                    audio_units,
                    limits.audio_seconds_per_hour,
                    3600,
                )
            )

        if not limit_entries:
            return

        if self._store is not None:
            self._reserve_persistent(provider_key, limit_entries, now)
        else:
            self._reserve_in_memory(provider_key, limit_entries, now)

    def release(
        self,
        *,
        provider: str,
        audio_seconds: int | float,
    ) -> None:
        """Undo a reservation by recording a compensating entry.

        Useful for failed downloads or over-estimated durations.
        """
        if not self._enabled or audio_seconds <= 0:
            return

        # Record a negative entry to correct the sum
        self.reserve(provider=provider, audio_seconds=-audio_seconds)

    def _reserve_in_memory(
        self,
        provider: str,
        limit_entries: list[tuple[str, int, int, int]],
        now: float,
    ) -> None:
        metric_to_bucket = {
            "rpm": self._rpm_windows,
            "tpm": self._tpm_windows,
            "audio_seconds_per_hour": self._audio_windows,
        }

        for metric, requested, limit, window_seconds in limit_entries:
            self._assert_within_limit(
                provider=provider,
                metric=metric,
                limit=limit,
                requested=requested,
                window_seconds=window_seconds,
                now=now,
                buckets=metric_to_bucket[metric],
            )

        for metric, requested, _limit, _window_seconds in limit_entries:
            self._record(
                provider=provider,
                value=requested,
                now=now,
                buckets=metric_to_bucket[metric],
            )

    def _reserve_persistent(
        self,
        provider: str,
        limit_entries: list[tuple[str, int, int, int]],
        now: float,
    ) -> None:
        """Reserve budget using the configured store."""
        if self._store is None:
            return

        # Check current usage for each metric
        for metric, requested, limit, window_seconds in limit_entries:
            current = self._store.check_usage(provider, metric, window_seconds, now)
            if current + requested > limit:
                raise BudgetExceededError(
                    provider=provider,
                    metric=metric,
                    limit=limit,
                    current=current,
                    requested=requested,
                    window_seconds=window_seconds,
                )

        # Record usage for each metric
        for metric, requested, _limit, _window_seconds in limit_entries:
            self._store.record_usage(provider, metric, requested, now)

    def _resolve_tokens(self, *, tokens: int | None, text_chars: int | None) -> int:
        if tokens is not None:
            return max(0, tokens)
        if text_chars is None:
            return 0
        if text_chars <= 0:
            return 0
        return math.ceil(text_chars / self._token_chars_per_token)

    def _assert_within_limit(
        self,
        *,
        provider: str,
        metric: str,
        limit: int,
        requested: int,
        window_seconds: int,
        now: float,
        buckets: dict[str, deque[tuple[float, int]]],
    ) -> None:
        current = self._sum_recent(
            provider=provider, window_seconds=window_seconds, now=now, buckets=buckets
        )
        if current + requested > limit:
            raise BudgetExceededError(
                provider=provider,
                metric=metric,
                limit=limit,
                current=current,
                requested=requested,
                window_seconds=window_seconds,
            )

    def _sum_recent(
        self,
        *,
        provider: str,
        window_seconds: int,
        now: float,
        buckets: dict[str, deque[tuple[float, int]]],
    ) -> int:
        events = buckets.get(provider)
        if events is None:
            return 0

        cutoff = now - window_seconds
        while events and events[0][0] <= cutoff:
            events.popleft()
        return sum(value for _timestamp, value in events)

    def _record(
        self,
        *,
        provider: str,
        value: int,
        now: float,
        buckets: dict[str, deque[tuple[float, int]]],
    ) -> None:
        events = buckets.get(provider)
        if events is None:
            events = deque()
            buckets[provider] = events
        events.append((now, value))
