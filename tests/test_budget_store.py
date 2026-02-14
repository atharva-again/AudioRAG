from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from audiorag.core.budget_store_sqlite import SqliteBudgetStore
from audiorag.core.exceptions import BudgetExceededError
from audiorag.core.protocols import BudgetStore


class FakeClock:
    """Deterministic clock for testing."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestBudgetStoreProtocol:
    """Tests for BudgetStore protocol definition."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """BudgetStore should be runtime_checkable."""
        assert hasattr(BudgetStore, "__subclasshook__")

    def test_sqlite_store_satisfies_protocol(self, tmp_path: Path) -> None:
        """SqliteBudgetStore should be an instance of BudgetStore."""
        store = SqliteBudgetStore(tmp_path / "test.db")
        assert isinstance(store, BudgetStore)


class TestSqliteBudgetStore:
    """Tests for SqliteBudgetStore implementation."""

    def test_table_auto_created(self, tmp_path: Path) -> None:
        """Store should create table on initialization."""
        db_path = tmp_path / "test.db"
        store = SqliteBudgetStore(db_path)

        # Verify table exists by recording usage
        store.record_usage("openai", "rpm", 1, now=0.0)
        usage = store.check_usage("openai", "rpm", window_seconds=60, now=0.0)
        assert usage == 1

    def test_record_and_check_usage(self, tmp_path: Path) -> None:
        """Should record usage and return correct sum."""
        store = SqliteBudgetStore(tmp_path / "test.db")

        store.record_usage("openai", "rpm", 10, now=0.0)
        store.record_usage("openai", "rpm", 20, now=30.0)
        store.record_usage("openai", "rpm", 30, now=59.0)

        usage = store.check_usage("openai", "rpm", window_seconds=60, now=60.0)
        assert usage == 60

    def test_check_usage_empty_returns_zero(self, tmp_path: Path) -> None:
        """Should return 0 for empty/no usage."""
        store = SqliteBudgetStore(tmp_path / "test.db")

        usage = store.check_usage("openai", "rpm", window_seconds=60, now=0.0)
        assert usage == 0

    def test_cleanup_removes_old_entries(self, tmp_path: Path) -> None:
        """Should remove entries older than cutoff."""
        store = SqliteBudgetStore(tmp_path / "test.db")

        # Record usage at various times
        store.record_usage("openai", "rpm", 100, now=0.0)
        store.record_usage("openai", "rpm", 200, now=100.0)
        store.record_usage("openai", "rpm", 300, now=200.0)

        # Check with 300s window - all should be present
        usage = store.check_usage("openai", "rpm", window_seconds=300, now=250.0)
        assert usage == 600

        # Cleanup old entries (older than 150)
        store.cleanup("openai", "rpm", cutoff_time=150.0)

        # After cleanup, only the 300 entry remains
        usage = store.check_usage("openai", "rpm", window_seconds=300, now=250.0)
        assert usage == 300

    def test_multiple_providers_isolated(self, tmp_path: Path) -> None:
        """Usage for different providers should be isolated."""
        store = SqliteBudgetStore(tmp_path / "test.db")

        store.record_usage("openai", "rpm", 100, now=0.0)
        store.record_usage("deepgram", "rpm", 200, now=0.0)
        store.record_usage("openai", "audio_seconds_per_hour", 300, now=0.0)

        assert store.check_usage("openai", "rpm", window_seconds=60, now=0.0) == 100
        assert store.check_usage("deepgram", "rpm", window_seconds=60, now=0.0) == 200
        assert (
            store.check_usage("openai", "audio_seconds_per_hour", window_seconds=3600, now=0.0)
            == 300
        )

    def test_multiple_metrics_isolated(self, tmp_path: Path) -> None:
        """Usage for different metrics should be isolated."""
        store = SqliteBudgetStore(tmp_path / "test.db")

        store.record_usage("openai", "rpm", 10, now=0.0)
        store.record_usage("openai", "tpm", 100, now=0.0)

        assert store.check_usage("openai", "rpm", window_seconds=60, now=0.0) == 10
        assert store.check_usage("openai", "tpm", window_seconds=60, now=0.0) == 100

    def test_negative_units_for_release(self, tmp_path: Path) -> None:
        """Negative units should reduce the sum (release pattern)."""
        store = SqliteBudgetStore(tmp_path / "test.db")

        store.record_usage("deepgram", "audio_seconds_per_hour", 100, now=0.0)
        store.record_usage("deepgram", "audio_seconds_per_hour", -50, now=0.0)

        usage = store.check_usage(
            "deepgram", "audio_seconds_per_hour", window_seconds=3600, now=0.0
        )
        assert usage == 50

    def test_check_usage_auto_cleans_expired(self, tmp_path: Path) -> None:
        """check_usage should automatically clean expired entries."""
        store = SqliteBudgetStore(tmp_path / "test.db")

        # Record at time 0
        store.record_usage("openai", "rpm", 100, now=0.0)

        # Check at time 30 - should be present
        usage = store.check_usage("openai", "rpm", window_seconds=60, now=30.0)
        assert usage == 100

        # Check at time 70 (expired from 60s window) - should auto-cleanup
        usage = store.check_usage("openai", "rpm", window_seconds=60, now=70.0)
        assert usage == 0

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """Data should persist across store instances."""
        db_path = tmp_path / "test.db"

        store1 = SqliteBudgetStore(db_path)
        store1.record_usage("openai", "rpm", 50, now=0.0)

        store2 = SqliteBudgetStore(db_path)
        usage = store2.check_usage("openai", "rpm", window_seconds=60, now=0.0)
        assert usage == 50


class InMemoryBudgetStore:
    """Simple in-memory store for testing custom store injection."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[tuple[float, int]]] = {}

    def check_usage(self, provider: str, metric: str, window_seconds: int, now: float) -> int:
        cutoff = now - window_seconds
        key = (provider, metric)
        if key not in self._data:
            return 0
        self._data[key] = [(t, u) for t, u in self._data[key] if t > cutoff]
        return sum(u for _t, u in self._data[key])

    def record_usage(self, provider: str, metric: str, units: int, now: float) -> None:
        key = (provider, metric)
        if key not in self._data:
            self._data[key] = []
        self._data[key].append((now, units))

    def cleanup(self, provider: str, metric: str, cutoff_time: float) -> None:
        key = (provider, metric)
        if key in self._data:
            self._data[key] = [(t, u) for t, u in self._data[key] if t > cutoff_time]


class TestGovernorWithCustomStore:
    """Tests for BudgetGovernor with custom store injection."""

    def test_governor_uses_custom_store(self) -> None:
        """Should use custom store when provided."""
        from audiorag.core.budget import BudgetGovernor, BudgetLimits

        custom_store = InMemoryBudgetStore()
        clock = FakeClock()

        governor = BudgetGovernor(
            enabled=True,
            global_limits=BudgetLimits(rpm=1),
            provider_overrides={},
            now=clock,
            store=custom_store,
        )

        governor.reserve(provider="openai", requests=1)

        # Verify store was used
        usage = custom_store.check_usage("openai", "rpm", window_seconds=60, now=0.0)
        assert usage == 1

        # Second request should fail
        with pytest.raises(BudgetExceededError):
            governor.reserve(provider="openai", requests=1)

    def test_governor_db_path_creates_sqlite_store(self, tmp_path: Path) -> None:
        """db_path should create SqliteBudgetStore for backward compat."""
        from audiorag.core.budget import BudgetGovernor, BudgetLimits

        db_path = tmp_path / "budget.db"
        clock = FakeClock()

        governor = BudgetGovernor(
            enabled=True,
            global_limits=BudgetLimits(rpm=1),
            provider_overrides={},
            now=clock,
            db_path=db_path,
        )

        governor.reserve(provider="openai", requests=1)

        # Verify SQLite file was created
        assert db_path.exists()

    def test_governor_store_takes_precedence_over_db_path(self, tmp_path: Path) -> None:
        """Custom store should take precedence over db_path."""
        from audiorag.core.budget import BudgetGovernor, BudgetLimits

        custom_store = InMemoryBudgetStore()
        db_path = tmp_path / "budget.db"
        clock = FakeClock()

        governor = BudgetGovernor(
            enabled=True,
            global_limits=BudgetLimits(rpm=1),
            provider_overrides={},
            now=clock,
            db_path=db_path,
            store=custom_store,
        )

        governor.reserve(provider="openai", requests=1)

        # Verify custom store was used (not SQLite)
        assert not db_path.exists()
        usage = custom_store.check_usage("openai", "rpm", window_seconds=60, now=0.0)
        assert usage == 1

    def test_governor_no_store_no_db_path_uses_memory(self) -> None:
        """Should use in-memory storage when no store or db_path provided."""
        from audiorag.core.budget import BudgetGovernor, BudgetLimits

        clock = FakeClock()

        governor = BudgetGovernor(
            enabled=True,
            global_limits=BudgetLimits(rpm=1),
            provider_overrides={},
            now=clock,
        )

        # Should work without any persistence
        governor.reserve(provider="openai", requests=1)

        # Second request should fail within window
        with pytest.raises(BudgetExceededError):
            governor.reserve(provider="openai", requests=1)

    def test_governor_from_config_with_custom_store(self) -> None:
        """from_config should accept optional store parameter."""
        from audiorag.core.budget import BudgetGovernor

        class MockConfig:
            budget_enabled: ClassVar[bool] = True
            budget_rpm: ClassVar[int] = 1
            budget_tpm: ClassVar[None] = None
            budget_audio_seconds_per_hour: ClassVar[None] = None
            budget_token_chars_per_token: ClassVar[int] = 4
            budget_provider_overrides: ClassVar[dict] = {}
            database_path: ClassVar[None] = None

        custom_store = InMemoryBudgetStore()
        governor = BudgetGovernor.from_config(MockConfig(), store=custom_store)

        governor.reserve(provider="openai", requests=1)

        usage = custom_store.check_usage("openai", "rpm", window_seconds=60, now=0.0)
        assert usage == 1
