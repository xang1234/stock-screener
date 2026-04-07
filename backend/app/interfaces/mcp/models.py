"""Pydantic models for the Hermes Market Copilot MCP server."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolFact(BaseModel):
    """A machine-readable fact in a tool response."""

    key: str
    value: Any
    source: str
    as_of: str | None = None


class ToolCitation(BaseModel):
    """A stable internal reference that supports a tool response."""

    source: str
    label: str
    reference: str
    as_of: str | None = None


class ToolFreshness(BaseModel):
    """Freshness metadata for a tool response."""

    generated_at: str
    as_of_date: str | None = None
    sources: dict[str, str | None] = Field(default_factory=dict)


class ToolEnvelope(BaseModel):
    """Shared result envelope returned by every MCP tool."""

    model_config = ConfigDict(extra="allow")

    summary: str
    facts: list[ToolFact] = Field(default_factory=list)
    citations: list[ToolCitation] = Field(default_factory=list)
    freshness: ToolFreshness
    next_actions: list[str] = Field(default_factory=list)


class MarketOverviewArgs(BaseModel):
    """Arguments for market_overview."""

    as_of_date: date | None = None


class CompareFeatureRunsArgs(BaseModel):
    """Arguments for compare_feature_runs."""

    run_a: int | None = Field(None, ge=1)
    run_b: int | None = Field(None, ge=1)
    limit: int = Field(25, ge=1, le=100)


class CandidateFilters(BaseModel):
    """Supported filters for find_candidates."""

    min_score: float | None = Field(None, ge=0, le=100)
    min_rs_rating: float | None = Field(None, ge=0, le=100)
    min_eps_growth_qq: float | None = None
    min_sales_growth_qq: float | None = None
    min_market_cap: float | None = Field(None, ge=0)
    max_market_cap: float | None = Field(None, ge=0)
    min_price: float | None = Field(None, gt=0)
    max_price: float | None = Field(None, gt=0)
    min_volume: float | None = Field(None, ge=0)
    stage: int | None = Field(None, ge=1, le=4)
    rating: str | list[str] | None = None
    sector: str | None = None
    industry_group: str | None = None
    text_query: str | None = None
    sort_field: str = "composite_score"
    sort_order: Literal["asc", "desc"] = "desc"

    @field_validator("rating", mode="before")
    @classmethod
    def normalize_rating(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, str):
            value = value.strip()
        return value


class FindCandidatesArgs(BaseModel):
    """Arguments for find_candidates."""

    filters: CandidateFilters = Field(default_factory=CandidateFilters)
    universe: str | None = Field(
        None,
        description="Optional scope. Omit or use 'all' for the latest published run, or pass a watchlist name to restrict the search.",
    )
    limit: int = Field(25, ge=1, le=100)


class ExplainSymbolArgs(BaseModel):
    """Arguments for explain_symbol."""

    symbol: str = Field(..., min_length=1, max_length=10)
    depth: Literal["brief", "full"] = "brief"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class WatchlistSnapshotArgs(BaseModel):
    """Arguments for watchlist_snapshot."""

    watchlist: str = Field(..., min_length=1, max_length=100)


class ThemeStateArgs(BaseModel):
    """Arguments for theme_state."""

    theme_name: str | None = None
    limit: int = Field(10, ge=1, le=25)


class TaskStatusArgs(BaseModel):
    """Arguments for task_status."""

    task_name: str | None = None


class WatchlistAddArgs(BaseModel):
    """Arguments for watchlist_add."""

    watchlist: str = Field(..., min_length=1, max_length=100)
    symbols: list[str] = Field(..., min_length=1, max_length=50)
    reason: str | None = Field(None, max_length=500)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for symbol in value:
            clean = symbol.strip().upper()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        if not normalized:
            raise ValueError("symbols must include at least one non-empty ticker")
        return normalized


class GroupRankingsArgs(BaseModel):
    """Arguments for group_rankings."""

    limit: int = Field(20, ge=1, le=197)
    period: Literal["1w", "1m", "3m", "6m"] = "1w"


class StockLookupArgs(BaseModel):
    """Arguments for stock_lookup."""

    symbol: str = Field(..., min_length=1, max_length=10)
    include_technicals: bool = False

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class BreadthSnapshotArgs(BaseModel):
    """Arguments for breadth_snapshot."""

    days: int = Field(5, ge=1, le=30)


class DailyDigestArgs(BaseModel):
    """Arguments for daily_digest."""

    as_of_date: date | None = None
