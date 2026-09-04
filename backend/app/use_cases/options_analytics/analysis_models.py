"""Closed result types produced by one options candidate analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.domain.options_analytics.history import HistoricalObservation
from app.domain.options_analytics.metrics.aggregate import ChainMetrics
from app.domain.options_analytics.metrics.history import HistoricalMetrics
from app.domain.options_analytics.models import (
    ChainObservation,
    HistoryReadiness,
    OptionCandidate,
)


@dataclass(frozen=True)
class OptionsMetricValues:
    max_pain: float | None = None
    net_gex: float | None = None
    gamma_flip: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    atm_iv: float | None = None
    skew_25_delta: float | None = None
    realized_volatility: float | None = None
    vrp: float | None = None
    activity_intensity: float | None = None
    call_open_interest: int | None = None
    put_open_interest: int | None = None
    call_volume: int | None = None
    put_volume: int | None = None
    call_put_volume_ratio: float | None = None
    volume_oi_ratio: float | None = None
    near_spot_volume_concentration: float | None = None


@dataclass(frozen=True)
class OptionsStrikePoint:
    strike: float
    call_open_interest: int | None = None
    put_open_interest: int | None = None
    call_volume: int | None = None
    put_volume: int | None = None
    call_iv: float | None = None
    put_iv: float | None = None
    estimated_call_gex: float | None = None
    estimated_put_gex: float | None = None


@dataclass(frozen=True)
class AnalysisContext:
    as_of_date: date
    market: str
    risk_free_rate: float | None
    run_warnings: tuple[str, ...] = ()
    historical_observations: tuple[HistoricalObservation, ...] = ()


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
    metric_values: OptionsMetricValues
    historical_metrics: HistoricalMetrics
    strike_points: tuple[OptionsStrikePoint, ...]
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
    "OptionsMetricValues",
    "OptionsStrikePoint",
    "UnavailableCandidateAnalysis",
]
