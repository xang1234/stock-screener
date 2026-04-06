"""Schemas for app-wide strategy profiles."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .app_runtime import ScanDefaultsResponse


class StrategyProfileDigestConfig(BaseModel):
    """Digest selection and presentation preferences."""

    leader_min_composite_score: float = 0.0
    leader_limit: int = 5
    leader_sort: str = "composite_score"
    theme_sort: str = "momentum_score"
    section_order: list[str] = Field(default_factory=list)
    weak_validation_positive_rate_floor: float = 0.5
    weak_validation_avg_return_floor: float = 0.0


class StrategyProfileStewardshipConfig(BaseModel):
    """Watchlist stewardship thresholds and ordering."""

    exit_score_max: float = 55.0
    exit_rating_max: int = 2
    exit_score_delta_max: float = -12.0
    exit_stage_delta_max: float = -2.0
    defense_earnings_exit_window_days: int = 5
    deteriorating_score_delta_max: float = -5.0
    deteriorating_rs_delta_max: float = -8.0
    strengthening_score_delta_min: float = 5.0
    strengthening_rs_delta_min: float = 8.0
    status_priority: list[str] = Field(default_factory=list)


class StrategyProfileStockActionConfig(BaseModel):
    """Stock workspace action-planning preferences."""

    offense_sizing_guidance: str = "full"
    balanced_sizing_guidance: str = "half"
    defense_sizing_guidance: str = "probe"
    earnings_caution_days: int = 14
    earnings_imminent_days: int = 5
    preferred_setups: list[str] = Field(default_factory=list)
    strengths_title: str = "Top Strengths"
    weaknesses_title: str = "Top Weaknesses"
    summary_emphasis: str = "balanced"


class StrategyProfileSummary(BaseModel):
    """Profile metadata."""

    profile: str
    label: str
    description: str


class StrategyProfileDetail(StrategyProfileSummary):
    """Full strategy-profile behavior contract."""

    scan_defaults: ScanDefaultsResponse
    digest: StrategyProfileDigestConfig
    stewardship: StrategyProfileStewardshipConfig
    stock_action: StrategyProfileStockActionConfig


class StrategyProfileListResponse(BaseModel):
    """List response for all strategy profiles."""

    profiles: list[StrategyProfileDetail]
