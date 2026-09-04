"""Closed result types produced by one options candidate analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.domain.options_analytics.metrics.aggregate import ChainMetrics
from app.domain.options_analytics.models import (
    ChainObservation,
    HistoryReadiness,
    OptionCandidate,
)


@dataclass(frozen=True)
class AnalysisContext:
    as_of_date: date
    market: str
    risk_free_rate: float | None
    run_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnavailableCandidateAnalysis:
    candidate: OptionCandidate
    reason_codes: tuple[str, ...]
    evidence: dict[str, Any]
    assumptions: dict[str, Any]
    warnings: tuple[str, ...]
    retry_count: int


@dataclass(frozen=True)
class AvailableCandidateAnalysis:
    candidate: OptionCandidate
    observation: ChainObservation
    metrics: ChainMetrics
    core_valid: bool
    metric_values: dict[str, float | int | None]
    strike_points: tuple[dict[str, Any], ...]
    evidence: dict[str, Any]
    assumptions: dict[str, Any]
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    retry_count: int
    history_readiness: HistoryReadiness


CandidateAnalysis = AvailableCandidateAnalysis | UnavailableCandidateAnalysis


__all__ = [
    "AnalysisContext",
    "AvailableCandidateAnalysis",
    "CandidateAnalysis",
    "UnavailableCandidateAnalysis",
]
