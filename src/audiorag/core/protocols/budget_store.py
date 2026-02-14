"""BudgetStore protocol for pluggable budget state management.

This protocol defines the interface for budget storage backends,
enabling custom implementations (external databases, Redis, etc.)
to integrate with the BudgetGovernor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BudgetStore(Protocol):
    """Protocol for budget storage backends.

    Implementations must provide synchronous methods for recording
    and checking usage within time windows. All methods accept an
    explicit ``now`` parameter to enable deterministic testing.

    Example:
        class RedisBudgetStore:
            def check_usage(
                self, provider: str, metric: str, window_seconds: int, now: float
            ) -> int:
                # Implementation
                pass

            def record_usage(
                self, provider: str, metric: str, units: int, now: float
            ) -> None:
                # Implementation
                pass

            def cleanup(self, provider: str, metric: str, cutoff_time: float) -> None:
                # Implementation
                pass

        store = RedisBudgetStore()
        assert isinstance(store, BudgetStore)  # True
    """

    def check_usage(self, provider: str, metric: str, window_seconds: int, now: float) -> int:
        """Return the total usage for a provider/metric within the window.

        Implementations should clean up expired entries before calculating.

        Args:
            provider: The provider name (e.g., "openai", "deepgram").
            metric: The metric name (e.g., "rpm", "tpm", "audio_seconds_per_hour").
            window_seconds: The time window in seconds.
            now: The current timestamp (for deterministic testing).

        Returns:
            The total usage units within the window.
        """
        ...

    def record_usage(self, provider: str, metric: str, units: int, now: float) -> None:
        """Record usage units for a provider/metric at the given time.

        Args:
            provider: The provider name.
            metric: The metric name.
            units: The usage units to record (can be negative for releases).
            now: The timestamp for the usage event.
        """
        ...

    def cleanup(self, provider: str, metric: str, cutoff_time: float) -> None:
        """Remove usage entries at or before the cutoff time.

        Args:
            provider: The provider name.
            metric: The metric name.
            cutoff_time: Entries at or before this timestamp are removed.
        """
        ...
