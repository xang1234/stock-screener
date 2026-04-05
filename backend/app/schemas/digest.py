"""Schemas for the daily digest API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from .validation import ValidationHorizonSummary, ValidationSourceKind


class DigestFreshness(BaseModel):
    """Freshness metadata for digest payloads."""

    latest_feature_as_of_date: Optional[str] = None
    latest_feature_published_at: Optional[str] = None
    latest_breadth_date: Optional[str] = None
    latest_theme_metrics_date: Optional[str] = None
    latest_theme_alert_at: Optional[str] = None
    validation_lookback_days: int = 90


class DigestBreadthMetrics(BaseModel):
    """Core breadth metrics used for digest stance classification."""

    up_4pct: Optional[int] = None
    down_4pct: Optional[int] = None
    ratio_5day: Optional[float] = None
    ratio_10day: Optional[float] = None
    total_stocks_scanned: Optional[int] = None


class DigestMarketSection(BaseModel):
    """Market stance section."""

    stance: str
    summary: str
    breadth_metrics: DigestBreadthMetrics


class DigestLeaderItem(BaseModel):
    """One top leader from the latest published feature run."""

    symbol: str
    name: Optional[str] = None
    composite_score: Optional[float] = None
    rating: Optional[str] = None
    industry_group: Optional[str] = None
    reason_summary: str


class DigestThemeItem(BaseModel):
    """One ranked theme row."""

    theme_id: int
    display_name: str
    category: Optional[str] = None
    momentum_score: Optional[float] = None
    mention_velocity: Optional[float] = None
    basket_return_1m: Optional[float] = None
    status: Optional[str] = None


class DigestThemeAlertItem(BaseModel):
    """One recent theme alert row."""

    alert_id: int
    alert_type: str
    severity: Optional[str] = None
    triggered_at: str
    theme: Optional[str] = None
    title: str
    related_tickers: list[str]


class DigestThemeSection(BaseModel):
    """Theme leaders, laggards, and recent alerts."""

    leaders: list[DigestThemeItem]
    laggards: list[DigestThemeItem]
    recent_alerts: list[DigestThemeAlertItem]


class DigestValidationSourceSnapshot(BaseModel):
    """Condensed validation snapshot for one source kind."""

    source_kind: ValidationSourceKind
    horizons: list[ValidationHorizonSummary]
    degraded_reasons: list[str]


class DigestValidationSection(BaseModel):
    """Validation snapshot section for the digest."""

    lookback_days: int
    scan_pick: DigestValidationSourceSnapshot
    theme_alert: DigestValidationSourceSnapshot


class DigestWatchlistHighlight(BaseModel):
    """Overlap summary for one user watchlist."""

    watchlist_id: int
    watchlist_name: str
    matched_symbols: list[str]
    alert_symbols: list[str]
    notes: str


class DigestRiskNote(BaseModel):
    """One deterministic risk note."""

    kind: str
    message: str
    severity: str


class DailyDigestResponse(BaseModel):
    """Full daily digest payload."""

    as_of_date: str
    freshness: DigestFreshness
    market: DigestMarketSection
    leaders: list[DigestLeaderItem]
    themes: DigestThemeSection
    validation: DigestValidationSection
    watchlists: list[DigestWatchlistHighlight]
    risks: list[DigestRiskNote]
    degraded_reasons: list[str]
