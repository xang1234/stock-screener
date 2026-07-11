"""DB-independent contract and automation plan for rolling static RRG state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.markets.catalog import MarketCatalog, get_market_catalog


StaticRRGHistorySchemaVersion = Literal["static-rrg-history-v3"]
STATIC_RRG_HISTORY_SCHEMA_VERSION: StaticRRGHistorySchemaVersion = (
    "static-rrg-history-v3"
)
STATIC_RRG_HISTORY_RETENTION_WEEKS = 60


class StaticRRGHistoryBundleError(ValueError):
    """Raised when rolling RRG state cannot be read or validated."""


class StaticRRGGroupPoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    industry_group: str = Field(min_length=1)
    rank: int = Field(ge=1)
    avg_rs_rating: float
    num_stocks: int = Field(ge=0)

    @field_validator("industry_group")
    @classmethod
    def normalize_group(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("industry_group is required")
        return normalized


class StaticRRGWeek(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_date: date
    groups: tuple[StaticRRGGroupPoint, ...]

    @model_validator(mode="after")
    def validate_week(self) -> "StaticRRGWeek":
        names = [group.industry_group for group in self.groups]
        if not names or len(names) != len(set(names)):
            raise ValueError("weekly groups must be non-empty and unique")
        return self


class StaticRRGHistoryState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StaticRRGHistorySchemaVersion
    market: str
    weeks: tuple[StaticRRGWeek, ...]

    @field_validator("market")
    @classmethod
    def normalize_market(cls, value: str) -> str:
        return normalize_static_rrg_market(value)

    @model_validator(mode="after")
    def validate_history(self) -> "StaticRRGHistoryState":
        source_dates = [week.source_date for week in self.weeks]
        if not source_dates:
            raise ValueError("weeks must not be empty")
        week_keys = [static_rrg_week_start(source_date) for source_date in source_dates]
        if source_dates != sorted(source_dates) or len(week_keys) != len(set(week_keys)):
            raise ValueError("weeks must be unique and ordered")
        return self

    @property
    def through_date(self) -> date:
        return self.weeks[-1].source_date


@dataclass(frozen=True)
class StaticRRGHistoryPlan:
    """Canonical paths and capability decision for one market's rolling state."""

    enabled: bool
    market: str
    asset_name: str
    source_path: Path
    output_path: Path

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "enabled": self.enabled,
            "market": self.market,
            "asset_name": self.asset_name,
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
        }


def normalize_static_rrg_market(market: str) -> str:
    normalized = str(market or "").strip().upper()
    if not normalized:
        raise StaticRRGHistoryBundleError("RRG history market is required.")
    return normalized


def static_rrg_asset_name(market: str) -> str:
    return f"rrg-history-{normalize_static_rrg_market(market).lower()}.json.gz"


def static_rrg_week_start(value: date) -> date:
    return value - timedelta(days=(value.weekday() + 1) % 7)


def build_static_rrg_history_plan(
    *,
    market: str,
    directory: Path,
    market_catalog: MarketCatalog | None = None,
) -> StaticRRGHistoryPlan:
    catalog = market_catalog or get_market_catalog()
    normalized_market = normalize_static_rrg_market(market)
    asset_name = static_rrg_asset_name(normalized_market)
    return StaticRRGHistoryPlan(
        enabled=bool(catalog.rrg_scopes_for_market(normalized_market)),
        market=normalized_market,
        asset_name=asset_name,
        source_path=directory / asset_name,
        output_path=directory / "current" / asset_name,
    )


__all__ = [
    "STATIC_RRG_HISTORY_RETENTION_WEEKS",
    "STATIC_RRG_HISTORY_SCHEMA_VERSION",
    "StaticRRGGroupPoint",
    "StaticRRGHistoryBundleError",
    "StaticRRGHistoryPlan",
    "StaticRRGHistoryState",
    "StaticRRGWeek",
    "build_static_rrg_history_plan",
    "normalize_static_rrg_market",
    "static_rrg_asset_name",
    "static_rrg_week_start",
]
