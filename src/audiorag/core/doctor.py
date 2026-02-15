"""Dependency verification module for AudioRAG.

This module provides functionality to check if required system dependencies
are available in the system PATH.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audiorag.core.config import AudioRAGConfig


@dataclass
class DependencyCheck:
    """Result of checking a single dependency.

    Attributes:
        name: Name of the dependency (e.g., "ffmpeg", "ffprobe")
        available: Whether the dependency was found in PATH
        path: Full path to the executable if found, None otherwise
        required: Whether this dependency is required for operation
    """

    name: str
    available: bool
    path: str | None
    required: bool = True


@dataclass
class DoctorResult:
    """Result of running dependency checks.

    Attributes:
        checks: List of individual dependency check results
    """

    checks: list[DependencyCheck] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """Return True if all required dependencies are available."""
        return all(check.available or not check.required for check in self.checks)


def check_dependencies(
    config: AudioRAGConfig | None = None,  # noqa: ARG001
) -> DoctorResult:
    """Check if required system dependencies are available.

    Checks the following dependencies:
    - ffmpeg: Required for audio processing
    - ffprobe: Required for audio metadata extraction

    Args:
        config: AudioRAG configuration (not used, kept for API compatibility).

    Returns:
        DoctorResult containing all dependency check results
    """
    # Default required binaries
    binaries = [
        ("ffmpeg", True),
        ("ffprobe", True),
    ]

    # Check each binary
    checks: list[DependencyCheck] = []
    for name, required in binaries:
        path = shutil.which(name)
        checks.append(
            DependencyCheck(
                name=name,
                available=path is not None,
                path=path,
                required=required,
            )
        )

    return DoctorResult(checks=checks)
