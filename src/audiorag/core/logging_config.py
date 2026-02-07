"""Logging configuration for AudioRAG using structlog.

This module provides structured logging with console output for local development.
Supports context binding, stage timing, and configurable log levels.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import structlog


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "colored",
    log_timestamps: bool = True,
) -> None:
    """Configure structlog with pretty console output for local development.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format - "colored" for dev, "plain" for redirection
        log_timestamps: Whether to include timestamps
    """
    shared_processors: list[structlog.types.Processor] = [
        # Filter by log level
        structlog.stdlib.filter_by_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add log level
        structlog.processors.add_log_level,
        # Add timestamp if enabled
        structlog.processors.TimeStamper(fmt="iso" if log_timestamps else None),
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exception info
        structlog.processors.format_exc_info,
    ]

    # Configure renderer based on format preference
    if log_format == "colored":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level.upper()),
        force=True,
    )

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        BoundLogger with structured logging capabilities
    """
    return structlog.get_logger(name)


class Timer:
    """Context manager for timing operations and logging duration.

    Example:
        with Timer(logger, "download", url=url) as timer:
            result = await download_audio(url)
            timer.complete(chunks=len(result))
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        operation: str,
        **context: Any,
    ) -> None:
        """Initialize timer.

        Args:
            logger: Structlog logger instance
            operation: Name of the operation being timed
            **context: Additional context to bind to log messages
        """
        self.logger = logger.bind(operation=operation, **context)
        self.operation = operation
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> Timer:
        """Start timing."""
        self.start_time = time.monotonic()
        self.logger.info(f"{self.operation}_started")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Log completion or failure."""
        self.end_time = time.monotonic()
        duration_ms = (self.end_time - self.start_time) * 1000

        if exc_type is None:
            self.logger.info(
                f"{self.operation}_completed",
                duration_ms=round(duration_ms, 2),
            )
        else:
            self.logger.error(
                f"{self.operation}_failed",
                duration_ms=round(duration_ms, 2),
                error_type=exc_type.__name__ if exc_type else None,
                error=str(exc_val) if exc_val else None,
            )

    def complete(self, **extra_context: Any) -> None:
        """Mark operation as complete with additional context.

        Args:
            **extra_context: Additional fields to include in completion log
        """
        self.end_time = time.monotonic()
        duration_ms = (self.end_time - self.start_time) * 1000

        self.logger.info(
            f"{self.operation}_completed",
            duration_ms=round(duration_ms, 2),
            **extra_context,
        )
