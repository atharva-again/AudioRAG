"""SQLite implementation of the BudgetStore protocol."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from audiorag.core.exceptions import ConfigurationError
from audiorag.core.protocols.budget_store import BudgetStore

PathLike = str | Path


class SqliteBudgetStore:
    """SQLite-backed implementation of BudgetStore.

    Stores budget events in a SQLite database with WAL mode enabled
    for better concurrency. Automatically creates the required table
    and index on initialization.

    Args:
        db_path: Path to the SQLite database file.

    Example:
        store = SqliteBudgetStore("./budget.db")
        store.record_usage("openai", "rpm", 1, now=time.time())
        usage = store.check_usage("openai", "rpm", window_seconds=60, now=time.time())
    """

    def __init__(self, db_path: PathLike) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_table()

    def _initialize_table(self) -> None:
        """Create the budget_events table and index if they don't exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS budget_events ("
                "provider TEXT NOT NULL,"
                "metric TEXT NOT NULL,"
                "units INTEGER NOT NULL,"
                "event_time REAL NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_budget_events_metric_time "
                "ON budget_events(provider, metric, event_time)"
            )
            conn.commit()

    def check_usage(self, provider: str, metric: str, window_seconds: int, now: float) -> int:
        """Return total usage for provider/metric within the time window.

        Automatically cleans up expired entries before calculating.

        Args:
            provider: The provider name (e.g., "openai").
            metric: The metric name (e.g., "rpm", "audio_seconds_per_hour").
            window_seconds: The time window in seconds.
            now: The current timestamp.

        Returns:
            Total usage units within the window.
        """
        cutoff = now - window_seconds
        self.cleanup(provider, metric, cutoff)

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT COALESCE(SUM(units), 0) FROM budget_events "
                "WHERE provider = ? AND metric = ?",
                (provider, metric),
            )
            row = cursor.fetchone()
            return int(row[0] if row and row[0] is not None else 0)

    def record_usage(self, provider: str, metric: str, units: int, now: float) -> None:
        """Record usage units for a provider/metric.

        Args:
            provider: The provider name.
            metric: The metric name.
            units: Usage units (can be negative for releases).
            now: Timestamp for the event.

        Raises:
            ConfigurationError: If database operation fails.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO budget_events (provider, metric, units, event_time) "
                    "VALUES (?, ?, ?, ?)",
                    (provider, metric, units, now),
                )
                conn.commit()
        except sqlite3.DatabaseError as exc:
            raise ConfigurationError(f"budget store database failure: {exc}") from exc

    def cleanup(self, provider: str, metric: str, cutoff_time: float) -> None:
        """Remove entries before the cutoff time.

        Args:
            provider: The provider name.
            metric: The metric name.
            cutoff_time: Entries before this time are removed (exclusive).
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM budget_events WHERE provider = ? AND metric = ? AND event_time < ?",
                (provider, metric, cutoff_time),
            )
            conn.commit()


# Verify protocol conformance at runtime
assert issubclass(SqliteBudgetStore, BudgetStore)
