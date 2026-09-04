"""Contract for rolling static breadth-contributor display metadata."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.domain.markets.catalog import MarketCatalog, get_market_catalog


StaticBreadthContributorMetadataSchemaVersion = Literal[
    "static-breadth-contributor-metadata-v1"
]
STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION: (
    StaticBreadthContributorMetadataSchemaVersion
) = "static-breadth-contributor-metadata-v1"
STATIC_BREADTH_CONTRIBUTOR_METADATA_RETENTION_DATES = 20
NO_GROUP_LABEL = "No Group"


class StaticBreadthContributorMetadataBundleError(ValueError):
    """Raised when rolling contributor metadata cannot be trusted."""


def _normalized_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_static_breadth_contributor_metadata_market(market: str) -> str:
    normalized = str(market or "").strip().upper()
    if not normalized:
        raise StaticBreadthContributorMetadataBundleError(
            "Breadth contributor metadata market is required."
        )
    return normalized


class FrozenBreadthContributorMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    company_name: str | None = None
    ibd_industry_group: str = NO_GROUP_LABEL

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = str(value or "").strip().upper()
        if not normalized:
            raise ValueError("symbol is required")
        return normalized

    @field_validator("company_name", mode="before")
    @classmethod
    def normalize_company_name(cls, value: object) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError("company_name must be a string or null")
        return _normalized_text(value)

    @field_validator("ibd_industry_group", mode="before")
    @classmethod
    def normalize_group(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("ibd_industry_group must be a string")
        return _normalized_text(value) or NO_GROUP_LABEL


class FrozenBreadthContributorSession(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    date: date
    contributors: tuple[FrozenBreadthContributorMetadata, ...]

    @model_validator(mode="after")
    def validate_contributors(self) -> "FrozenBreadthContributorSession":
        symbols = [item.symbol for item in self.contributors]
        if len(symbols) != len(set(symbols)):
            raise ValueError("session contributor symbols must be unique")
        if symbols != sorted(symbols):
            raise ValueError("session contributor symbols must be sorted")
        return self


class StaticBreadthContributorMetadataState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StaticBreadthContributorMetadataSchemaVersion
    market: str
    generated_at: datetime
    sessions: tuple[FrozenBreadthContributorSession, ...] = Field(
        max_length=STATIC_BREADTH_CONTRIBUTOR_METADATA_RETENTION_DATES
    )

    @field_validator("market")
    @classmethod
    def normalize_market(cls, value: str) -> str:
        return normalize_static_breadth_contributor_metadata_market(value)

    @model_validator(mode="after")
    def validate_sessions(self) -> "StaticBreadthContributorMetadataState":
        dates = [item.date for item in self.sessions]
        if len(dates) != len(set(dates)):
            raise ValueError("metadata session dates must be unique")
        if dates != sorted(dates, reverse=True):
            raise ValueError("metadata sessions must be newest-first")
        return self


@dataclass(frozen=True)
class StaticBreadthContributorMetadataPlan:
    enabled: bool
    market: str
    asset_name: str
    previous_asset_name: str
    source_path: Path
    previous_path: Path
    output_path: Path

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "enabled": self.enabled,
            "market": self.market,
            "asset_name": self.asset_name,
            "previous_asset_name": self.previous_asset_name,
            "source_path": str(self.source_path),
            "previous_path": str(self.previous_path),
            "output_path": str(self.output_path),
        }


def static_breadth_contributor_metadata_asset_name(market: str) -> str:
    normalized = normalize_static_breadth_contributor_metadata_market(market)
    return f"breadth-contributor-metadata-{normalized.lower()}.json.gz"


def static_breadth_contributor_metadata_previous_asset_name(market: str) -> str:
    normalized = normalize_static_breadth_contributor_metadata_market(market)
    return f"breadth-contributor-metadata-{normalized.lower()}.previous.json.gz"


def build_static_breadth_contributor_metadata_plan(
    *,
    market: str,
    directory: Path,
    market_catalog: MarketCatalog | None = None,
) -> StaticBreadthContributorMetadataPlan:
    catalog = market_catalog or get_market_catalog()
    normalized = normalize_static_breadth_contributor_metadata_market(market)
    asset_name = static_breadth_contributor_metadata_asset_name(normalized)
    previous_asset_name = static_breadth_contributor_metadata_previous_asset_name(
        normalized
    )
    return StaticBreadthContributorMetadataPlan(
        enabled=bool(catalog.get(normalized).capabilities.breadth),
        market=normalized,
        asset_name=asset_name,
        previous_asset_name=previous_asset_name,
        source_path=directory / asset_name,
        previous_path=directory / previous_asset_name,
        output_path=directory / "current" / asset_name,
    )


def write_static_breadth_contributor_metadata(
    path: Path,
    state: StaticBreadthContributorMetadataState,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        state.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            archive.write(encoded)


def read_static_breadth_contributor_metadata(
    path: Path,
    *,
    expected_market: str,
) -> StaticBreadthContributorMetadataState:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        state = StaticBreadthContributorMetadataState.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise StaticBreadthContributorMetadataBundleError(
            f"Unable to read breadth contributor metadata state: {exc}"
        ) from exc
    normalized_market = normalize_static_breadth_contributor_metadata_market(
        expected_market
    )
    if state.market != normalized_market:
        raise StaticBreadthContributorMetadataBundleError(
            f"Breadth contributor metadata market {state.market} does not match "
            f"expected market {normalized_market}."
        )
    return state


__all__ = [
    "STATIC_BREADTH_CONTRIBUTOR_METADATA_RETENTION_DATES",
    "STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION",
    "FrozenBreadthContributorMetadata",
    "FrozenBreadthContributorSession",
    "StaticBreadthContributorMetadataBundleError",
    "StaticBreadthContributorMetadataPlan",
    "StaticBreadthContributorMetadataState",
    "build_static_breadth_contributor_metadata_plan",
    "normalize_static_breadth_contributor_metadata_market",
    "read_static_breadth_contributor_metadata",
    "static_breadth_contributor_metadata_asset_name",
    "static_breadth_contributor_metadata_previous_asset_name",
    "write_static_breadth_contributor_metadata",
]
