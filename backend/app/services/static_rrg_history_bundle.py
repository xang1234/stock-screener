"""Typed weekly history state for static-site RRG builds."""

from __future__ import annotations

import gzip
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.domain.markets.catalog import MarketCatalog, get_market_catalog
from app.models.industry import IBDGroupRank
from app.services.market_taxonomy_service import get_market_taxonomy_service
from app.services.rrg_history_provider import RRGHistoryResult
from app.services.rrg_service import RRGService


STATIC_RRG_HISTORY_SCHEMA_VERSION = "static-rrg-history-v3"
STATIC_RRG_HISTORY_RETENTION_WEEKS = 60


class StaticRRGHistoryBundleError(ValueError):
    """Raised when rolling RRG state cannot be read or validated."""


class StaticRRGHistoryUnavailableError(RuntimeError):
    """Raised when current weekly RRG state cannot be built."""


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

    schema_version: Literal["static-rrg-history-v3"]
    market: str
    weeks: tuple[StaticRRGWeek, ...]

    @field_validator("market")
    @classmethod
    def normalize_market(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("market is required")
        return normalized

    @model_validator(mode="after")
    def validate_history(self) -> "StaticRRGHistoryState":
        source_dates = [week.source_date for week in self.weeks]
        if not source_dates:
            raise ValueError("weeks must not be empty")
        week_keys = [_week_start(source_date) for source_date in source_dates]
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


@dataclass(frozen=True)
class StaticRRGHistoryPreparation:
    """Result of loading and advancing one market's rolling state."""

    plan: StaticRRGHistoryPlan
    state: StaticRRGHistoryState | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticRRGHistoryBundleService:
    """Merge persisted weekly RRG state with freshly calculated group ranks."""

    retention_weeks: int = STATIC_RRG_HISTORY_RETENTION_WEEKS
    market_catalog: MarketCatalog = field(default_factory=get_market_catalog)

    @staticmethod
    def asset_name(market: str) -> str:
        return f"rrg-history-{_normalize_market(market).lower()}.json.gz"

    def enabled_for_market(self, market: str) -> bool:
        return bool(self.market_catalog.rrg_scopes_for_market(_normalize_market(market)))

    def plan(self, *, market: str, directory: Path) -> StaticRRGHistoryPlan:
        normalized_market = _normalize_market(market)
        asset_name = self.asset_name(normalized_market)
        return StaticRRGHistoryPlan(
            enabled=self.enabled_for_market(normalized_market),
            market=normalized_market,
            asset_name=asset_name,
            source_path=directory / asset_name,
            output_path=directory / "current" / asset_name,
        )

    def prepare(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        directory: Path,
    ) -> StaticRRGHistoryPreparation:
        """Load prior state when valid, then advance it from current DB rows."""
        plan = self.plan(market=market, directory=directory)
        if not plan.enabled:
            return StaticRRGHistoryPreparation(plan=plan, state=None)
        previous = None
        warnings: list[str] = []
        if plan.source_path.exists():
            try:
                previous = self.load(plan.source_path, expected_market=market)
            except StaticRRGHistoryBundleError as exc:
                warnings.append(
                    f"Rolling RRG history was invalid and will be bootstrapped: {exc}"
                )
        try:
            state = self.build(
                db,
                market=market,
                through_date=through_date,
                previous=previous,
            )
        except (StaticRRGHistoryBundleError, StaticRRGHistoryUnavailableError) as exc:
            warnings.append(f"Rolling RRG history was not advanced: {exc}")
            state = None
        return StaticRRGHistoryPreparation(
            plan=plan,
            state=state,
            warnings=tuple(warnings),
        )

    def load(self, input_path: Path, *, expected_market: str) -> StaticRRGHistoryState:
        try:
            payload = _read_payload(input_path)
            state = StaticRRGHistoryState.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise StaticRRGHistoryBundleError(
                f"Unable to load RRG history bundle {input_path}: {exc}"
            ) from exc
        normalized_market = _normalize_market(expected_market)
        if state.market != normalized_market:
            raise StaticRRGHistoryBundleError(
                f"RRG history bundle market {state.market} does not match {normalized_market}."
            )
        return state

    def build(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        previous: StaticRRGHistoryState | None = None,
    ) -> StaticRRGHistoryState:
        normalized_market = _normalize_market(market)
        if not self.enabled_for_market(normalized_market):
            raise StaticRRGHistoryUnavailableError(
                f"RRG is not enabled for market {normalized_market}."
            )
        if previous is not None and previous.market != normalized_market:
            raise StaticRRGHistoryBundleError(
                f"RRG history bundle market {previous.market} does not match {normalized_market}."
            )
        if not inspect(db.get_bind()).has_table(IBDGroupRank.__tablename__):
            raise StaticRRGHistoryUnavailableError(
                f"RRG source table {IBDGroupRank.__tablename__} is unavailable."
            )

        cutoff = through_date - timedelta(weeks=max(1, int(self.retention_weeks)))
        rows = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == normalized_market,
                IBDGroupRank.date >= cutoff,
                IBDGroupRank.date <= through_date,
            )
            .order_by(IBDGroupRank.date.asc(), IBDGroupRank.rank.asc())
            .all()
        )
        weeks = {
            _week_start(week.source_date): week
            for week in (previous.weeks if previous is not None else ())
            if week.source_date >= cutoff
        }
        try:
            weeks.update(_weekly_snapshots(rows))
            ordered = tuple(weeks[key] for key in sorted(weeks))
            if not ordered or ordered[-1].source_date != through_date:
                raise StaticRRGHistoryUnavailableError(
                    f"No current group-rank history is available for {normalized_market} "
                    f"on {through_date.isoformat()}."
                )
            return StaticRRGHistoryState(
                schema_version=STATIC_RRG_HISTORY_SCHEMA_VERSION,
                market=normalized_market,
                weeks=ordered,
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise StaticRRGHistoryBundleError(
                f"Unable to build RRG history for {normalized_market}: {exc}"
            ) from exc

    def write(self, state: StaticRRGHistoryState, output_path: Path) -> dict[str, Any]:
        temp_path: Path | None = None
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
            ) as temp:
                temp_path = Path(temp.name)
            _write_payload(temp_path, state.model_dump(mode="json"))
            temp_path.replace(output_path)
        except (OSError, TypeError, ValueError) as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise StaticRRGHistoryBundleError(
                f"Unable to write RRG history bundle {output_path}: {exc}"
            ) from exc
        return {
            "path": str(output_path),
            "market": state.market,
            "through_date": state.through_date.isoformat(),
            "weeks": len(state.weeks),
            "groups": sum(len(week.groups) for week in state.weeks),
        }

    def persist(
        self,
        preparation: StaticRRGHistoryPreparation,
        *,
        exported_as_of_date: date,
    ) -> dict[str, Any] | None:
        """Persist prepared state only when it matches the published artifact."""
        state = preparation.state
        if state is None or state.through_date != exported_as_of_date:
            return None
        return self.write(state, preparation.plan.output_path)


@dataclass(frozen=True)
class StaticRRGHistoryProvider:
    """Serve a validated weekly state through the shared RRG provider contract."""

    state: StaticRRGHistoryState

    def get_all_groups_history(
        self,
        _db: Any,
        *,
        market: str,
        days: int,
        as_of_date: date | None = None,
    ) -> RRGHistoryResult:
        normalized_market = _normalize_market(market)
        if normalized_market != self.state.market:
            return None, {}, {}
        target_date = as_of_date or self.state.through_date
        cutoff = target_date - timedelta(days=days)
        weeks = [
            week
            for week in self.state.weeks
            if cutoff <= week.source_date <= target_date
        ]
        if not weeks:
            return None, {}, {}
        latest = weeks[-1]
        meta = {
            group.industry_group: {
                "industry_group": group.industry_group,
                "rank": group.rank,
                "avg_rs_rating": group.avg_rs_rating,
                "num_stocks": group.num_stocks,
            }
            for group in latest.groups
        }
        series: dict[str, list[tuple[date, float, int]]] = defaultdict(list)
        for week in weeks:
            for group in week.groups:
                series[group.industry_group].append(
                    (week.source_date, group.avg_rs_rating, group.num_stocks)
                )
        return latest.source_date.isoformat(), meta, dict(series)


def build_static_rrg_service(state: StaticRRGHistoryState) -> RRGService:
    """Compose the static adapter once, outside the payload builder."""
    return RRGService(
        history_provider=StaticRRGHistoryProvider(state),
        taxonomy_service=get_market_taxonomy_service(),
    )


def _weekly_snapshots(rows: list[IBDGroupRank]) -> dict[date, StaticRRGWeek]:
    rows_by_date: dict[date, list[IBDGroupRank]] = defaultdict(list)
    for row in rows:
        rows_by_date[row.date].append(row)
    latest_date_by_week: dict[date, date] = {}
    for row_date in rows_by_date:
        week_start = _week_start(row_date)
        latest_date_by_week[week_start] = max(
            row_date,
            latest_date_by_week.get(week_start, row_date),
        )
    return {
        week_start: StaticRRGWeek(
            source_date=source_date,
            groups=tuple(
                StaticRRGGroupPoint(
                    industry_group=row.industry_group,
                    rank=row.rank,
                    avg_rs_rating=row.avg_rs_rating,
                    num_stocks=int(row.num_stocks or 0),
                )
                for row in sorted(rows_by_date[source_date], key=lambda item: item.rank)
            ),
        )
        for week_start, source_date in latest_date_by_week.items()
    }


def _week_start(value: date) -> date:
    return value - timedelta(days=(value.weekday() + 1) % 7)


def _normalize_market(market: str) -> str:
    normalized = str(market or "").strip().upper()
    if not normalized:
        raise StaticRRGHistoryBundleError("RRG history market is required.")
    return normalized


def _read_payload(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))


__all__ = [
    "STATIC_RRG_HISTORY_RETENTION_WEEKS",
    "STATIC_RRG_HISTORY_SCHEMA_VERSION",
    "StaticRRGGroupPoint",
    "StaticRRGHistoryBundleError",
    "StaticRRGHistoryBundleService",
    "StaticRRGHistoryPlan",
    "StaticRRGHistoryProvider",
    "StaticRRGHistoryPreparation",
    "StaticRRGHistoryState",
    "StaticRRGHistoryUnavailableError",
    "StaticRRGWeek",
    "build_static_rrg_service",
]
