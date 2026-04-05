"""Schemas for deterministic signal validation responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ValidationSourceKind(str, Enum):
    """Supported validation signal sources."""

    SCAN_PICK = "scan_pick"
    THEME_ALERT = "theme_alert"


class ValidationFreshness(BaseModel):
    """Freshness metadata for validation responses."""

    latest_feature_as_of_date: str | None = None
    latest_theme_alert_at: str | None = None
    price_cache_period: str = "2y"


class ValidationHorizonSummary(BaseModel):
    """Aggregate stats for one validation horizon."""

    horizon_sessions: int
    sample_size: int
    positive_rate: float | None = None
    avg_return_pct: float | None = None
    median_return_pct: float | None = None
    avg_mfe_pct: float | None = None
    avg_mae_pct: float | None = None
    skipped_missing_history: int = 0


class ValidationEvent(BaseModel):
    """One validation event with computed outcomes."""

    source_kind: ValidationSourceKind
    source_ref: str
    event_at: str
    entry_at: str | None = None
    entry_price: float | None = None
    return_1s_pct: float | None = None
    return_5s_pct: float | None = None
    mfe_5s_pct: float | None = None
    mae_5s_pct: float | None = None
    attributes: dict[str, Any]


class ValidationFailureCluster(BaseModel):
    """Aggregate of losing events by one deterministic bucket."""

    cluster_key: str
    label: str
    sample_size: int
    avg_return_pct: float
    median_return_pct: float


class ValidationSourceBreakdown(BaseModel):
    """Per-source validation section used by symbol responses."""

    source_kind: ValidationSourceKind
    horizons: list[ValidationHorizonSummary]
    recent_events: list[ValidationEvent]
    failure_clusters: list[ValidationFailureCluster]
    degraded_reasons: list[str]


class ValidationOverviewResponse(BaseModel):
    """Aggregate validation payload for the overview page."""

    source_kind: ValidationSourceKind
    lookback_days: int
    horizons: list[ValidationHorizonSummary]
    recent_events: list[ValidationEvent]
    failure_clusters: list[ValidationFailureCluster]
    freshness: ValidationFreshness
    degraded_reasons: list[str]


class StockValidationResponse(BaseModel):
    """Validation payload for one symbol across all supported sources."""

    symbol: str
    lookback_days: int
    source_breakdown: list[ValidationSourceBreakdown]
    recent_events: list[ValidationEvent]
    failure_clusters: list[ValidationFailureCluster]
    freshness: ValidationFreshness
    degraded_reasons: list[str]
