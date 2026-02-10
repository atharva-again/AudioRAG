"""Retry configuration for AudioRAG providers.

This module provides centralized retry logic with exponential backoff
for all external API calls. Supports configurable retry parameters
and integrates with structured logging.
"""

from __future__ import annotations

from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity import (
    retry as tenacity_retry,
)

from audiorag.core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        exponential_multiplier: Multiplier for exponential backoff
    """

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait_seconds: float = 4.0,
        max_wait_seconds: float = 60.0,
        exponential_multiplier: float = 1.0,
    ):
        self.max_attempts = max_attempts
        self.min_wait_seconds = min_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.exponential_multiplier = exponential_multiplier


def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts with structured context.

    Args:
        retry_state: Current state of the retry call
    """
    fn_name = retry_state.fn.__name__ if retry_state.fn else "unknown"
    exception = retry_state.outcome.exception() if retry_state.outcome else None

    logger.warning(
        "retry_attempt",
        function=fn_name,
        attempt=retry_state.attempt_number,
        max_attempts=retry_state.retry_object.stop.max_attempt_number,  # type: ignore[possibly-missing-attribute]
        wait_seconds=retry_state.next_action.sleep if retry_state.next_action else 0,
        error=str(exception) if exception else None,
        error_type=type(exception).__name__ if exception else None,
    )


def create_retry_decorator(
    config: RetryConfig,
    exception_types: tuple[type[Exception], ...],
    operation_name: str | None = None,  # noqa: ARG001
) -> Any:
    """Create a retry decorator with the specified configuration.

    Args:
        config: Retry configuration
        exception_types: Tuple of exception types to retry on
        operation_name: Optional name for the operation (for logging)

    Returns:
        Configured retry decorator
    """
    return tenacity_retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential(
            multiplier=config.exponential_multiplier,
            min=config.min_wait_seconds,
            max=config.max_wait_seconds,
        ),
        retry=retry_if_exception_type(exception_types),
        before_sleep=log_retry_attempt,
        reraise=True,
    )


# Predefined retry configurations for different use cases
RETRY_CONFIG_DEFAULT = RetryConfig(
    max_attempts=3,
    min_wait_seconds=4.0,
    max_wait_seconds=60.0,
    exponential_multiplier=1.0,
)

RETRY_CONFIG_AGGRESSIVE = RetryConfig(
    max_attempts=5,
    min_wait_seconds=1.0,
    max_wait_seconds=30.0,
    exponential_multiplier=0.5,
)

RETRY_CONFIG_CONSERVATIVE = RetryConfig(
    max_attempts=3,
    min_wait_seconds=10.0,
    max_wait_seconds=120.0,
    exponential_multiplier=2.0,
)


OPENAI_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)

COHERE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)

CHROMADB_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)

YOUTUBE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)
