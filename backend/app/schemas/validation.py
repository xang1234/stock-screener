"""Schemas for deterministic signal validation responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ValidationSourceKind(str, Enum):
    """Supported validation signal sources."""

    SCAN_PICK = "scan_pick"
    THEME_ALERT = "theme_alert"
    REPLAY_PICK = "replay_pick"


class ValidationFreshness(BaseModel):
    """Freshness metadata for validation responses."""

    latest_feature_as_of_date: str | None = None
    latest_theme_alert_at: str | None = None
    latest_replay_as_of_date: str | None = None
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
    avg_excess_return_pct: float | None = None
    median_excess_return_pct: float | None = None
    # Events with a benchmark bar on both dates; the excess figures average over these.
    excess_sample_size: int = 0
    avg_stopped_return_pct: float | None = None
    median_stopped_return_pct: float | None = None
    # Share of the sample whose stop triggered within the horizon.
    stopped_out_rate: float | None = None
    # Closed windows the cache cannot score: no history, a missing session or no entry price.
    skipped_missing_history: int = 0
    # Windows whose last session falls after the last completed one.
    pending_windows: int = 0
    suppressed_overlapping: int = 0


class ValidationEvent(BaseModel):
    """One validation event with computed outcomes."""

    source_kind: ValidationSourceKind
    source_ref: str
    event_at: str
    entry_at: str | None = None
    entry_price: float | None = None
    return_1s_pct: float | None = None
    return_5s_pct: float | None = None
    return_20s_pct: float | None = None
    return_60s_pct: float | None = None
    mfe_5s_pct: float | None = None
    mae_5s_pct: float | None = None
    excess_return_1s_pct: float | None = None
    excess_return_5s_pct: float | None = None
    # False for a repeat inside an earlier counted event's 5-session window.
    counted: bool = True
    # The horizons whose summaries count this event; a repeat at one horizon can count at another.
    counted_horizons: list[int]
    attributes: dict[str, Any]


class ValidationGroup(BaseModel):
    """Horizon stats for the events sharing one value of one dimension, such as Stage 2."""

    dimension: str
    label: str
    value: str
    horizons: list[ValidationHorizonSummary]


class ValidationSourceBreakdown(BaseModel):
    """Per-source validation section used by symbol responses."""

    source_kind: ValidationSourceKind
    horizons: list[ValidationHorizonSummary]
    recent_events: list[ValidationEvent]
    degraded_reasons: list[str]


class ValidationOverviewResponse(BaseModel):
    """Aggregate validation payload for the overview page."""

    source_kind: ValidationSourceKind
    lookback_days: int
    market: str
    benchmark_symbol: str | None = None
    stop_loss_pct: float
    horizons: list[ValidationHorizonSummary]
    recent_events: list[ValidationEvent]
    groups: list[ValidationGroup]
    freshness: ValidationFreshness
    degraded_reasons: list[str]


class StockValidationResponse(BaseModel):
    """Validation payload for one symbol across all supported sources."""

    symbol: str
    lookback_days: int
    stop_loss_pct: float
    source_breakdown: list[ValidationSourceBreakdown]
    recent_events: list[ValidationEvent]
    freshness: ValidationFreshness
    degraded_reasons: list[str]
