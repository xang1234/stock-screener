"""Detector interfaces and shared detector-level data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.models import PatternCandidate


@dataclass(frozen=True)
class PatternDetectorInput:
    """Detector input payload independent from scanner infrastructure."""

    symbol: str
    timeframe: str
    daily_bars: int
    weekly_bars: int
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternDetectorResult:
    """Detector output with candidate and diagnostics."""

    detector_name: str
    candidate: PatternCandidate | None
    passed_checks: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class PatternDetector(abc.ABC):
    """Abstract detector contract for analysis-layer pattern modules."""

    name: str

    @abc.abstractmethod
    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        """Return a candidate or an explicit no-candidate result."""
        raise NotImplementedError
