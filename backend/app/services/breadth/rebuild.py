"""Shadow rebuild, validation, and atomic cutover for the current breadth revision."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import date
from types import MappingProxyType
from typing import Any

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.models.breadth_contributor import (
    MarketBreadthContributor,
    MarketBreadthContributorSnapshot,
)
from app.models.market_breadth import MarketBreadth
from app.services.breadth_backfill import (
    BreadthBackfillExecutor,
    BreadthBackfillPlan,
    BreadthEligibleUniverse,
)
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    DerivedDataExecutionPolicy,
    DerivedDataTargetKind,
)
from app.services.point_in_time_universe_service import PointInTimeUniverseService

from .contributors import (
    BREADTH_CONTRIBUTOR_SIGNALS,
    CONTRIBUTOR_RETENTION_SESSIONS,
    CONTRIBUTOR_SCHEMA_ID,
    BreadthContributorContractError,
    contributor_calculation_signature,
    parse_contributor_rows,
    reconcile_contributor_aggregate,
    reconcile_contributor_counts,
)
from .types import (
    BREADTH_SHARE_FIELDS,
    CURRENT_BREADTH_CALCULATION_REVISION,
    BreadthContributorSnapshotResult,
    BreadthDailyResult,
    share_pct,
)

TARGET_TABLE = "market_breadth"
STAGING_TABLE = "market_breadth_rebuild"
MANIFEST_TABLE = "market_breadth_rebuild_manifest"
CONTRIBUTOR_SNAPSHOT_TARGET_TABLE = MarketBreadthContributorSnapshot.__tablename__
CONTRIBUTOR_TARGET_TABLE = MarketBreadthContributor.__tablename__
CONTRIBUTOR_SNAPSHOT_STAGING_TABLE = "market_breadth_contributor_snapshots_rebuild"
CONTRIBUTOR_STAGING_TABLE = "market_breadth_contributors_rebuild"
_EXCLUDED_COPY_COLUMNS = {"id", "created_at"}


def _copy_columns() -> tuple[str, ...]:
    return tuple(
        column.name
        for column in MarketBreadth.__table__.columns
        if column.name not in _EXCLUDED_COPY_COLUMNS
    )


class StagingBreadthPersistence:
    """Breadth persistence adapter that writes only to the shadow table."""

    def __init__(self, rebuild: BreadthRebuildService) -> None:
        self._rebuild = rebuild

    def upsert_many(
        self,
        results: Iterable[BreadthDailyResult],
        *,
        contributor_snapshots_by_date: Mapping[date, BreadthContributorSnapshotResult]
        | None = None,
        duration_seconds_by_date: Mapping[date, float] | None = None,
    ) -> tuple[()]:
        if contributor_snapshots_by_date is None:
            raise ValueError("Breadth rebuild requires canonical contributor snapshots")
        self._rebuild.stage_results(
            results,
            contributor_snapshots_by_date=contributor_snapshots_by_date,
            duration_seconds_by_date=duration_seconds_by_date,
        )
        return ()


class BreadthRebuildService:
    def __init__(
        self,
        db: Session,
        *,
        price_cache=None,
        universe_service: PointInTimeUniverseService | None = None,
        calendar_service=None,
        required_markets: Iterable[str] | None = None,
    ) -> None:
        if required_markets is None:
            from app.domain.markets.catalog import get_market_catalog

            required_markets = get_market_catalog().market_codes_with_capability(
                "breadth"
            )
        self.db = db
        self._price_cache = price_cache
        self._universe_service = universe_service or PointInTimeUniverseService()
        self._calendar_service = calendar_service
        self._required_markets = frozenset(
            str(market).upper() for market in required_markets
        )

    @property
    def dialect_name(self) -> str:
        return self.db.get_bind().dialect.name

    def _has_table(self, table_name: str) -> bool:
        """Inspect through the session connection to avoid cross-connection locks."""
        return inspect(self.db.connection()).has_table(table_name)

    def recreate_staging(self) -> None:
        self.db.execute(text(f"DROP TABLE IF EXISTS {CONTRIBUTOR_STAGING_TABLE}"))
        self.db.execute(
            text(f"DROP TABLE IF EXISTS {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE}")
        )
        self.db.execute(text(f"DROP TABLE IF EXISTS {STAGING_TABLE}"))
        self.db.execute(text(f"DROP TABLE IF EXISTS {MANIFEST_TABLE}"))
        if self.dialect_name == "postgresql":
            self.db.execute(
                text(
                    f"CREATE TABLE {STAGING_TABLE} "
                    f"(LIKE {TARGET_TABLE} INCLUDING DEFAULTS "
                    "INCLUDING GENERATED INCLUDING IDENTITY)"
                )
            )
        else:
            self.db.execute(
                text(
                    f"CREATE TABLE {STAGING_TABLE} AS "
                    f"SELECT * FROM {TARGET_TABLE} WHERE 1 = 0"
                )
            )
        self.db.execute(
            text(
                f"CREATE UNIQUE INDEX uix_breadth_rebuild_date_market "
                f"ON {STAGING_TABLE} (date, market)"
            )
        )
        self.db.execute(
            text(
                f"CREATE TABLE {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE} AS "
                "SELECT market, date, calculation_revision, schema_id "
                f"FROM {CONTRIBUTOR_SNAPSHOT_TARGET_TABLE} WHERE 1 = 0"
            )
        )
        self.db.execute(
            text(
                "CREATE UNIQUE INDEX uix_breadth_contributor_snapshot_rebuild_market_date "
                f"ON {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE} (market, date)"
            )
        )
        self.db.execute(
            text(
                f"CREATE TABLE {CONTRIBUTOR_STAGING_TABLE} AS "
                "SELECT snapshots.market, snapshots.date, contributors.symbol, "
                "contributors.company_name, contributors.ibd_industry_group, "
                "contributors.daily_change_pct, contributors.signals_json "
                f"FROM {CONTRIBUTOR_TARGET_TABLE} AS contributors "
                f"JOIN {CONTRIBUTOR_SNAPSHOT_TARGET_TABLE} AS snapshots "
                "ON snapshots.id = contributors.snapshot_id WHERE 1 = 0"
            )
        )
        self.db.execute(
            text(
                "CREATE UNIQUE INDEX uix_breadth_contributor_rebuild_market_date_symbol "
                f"ON {CONTRIBUTOR_STAGING_TABLE} (market, date, symbol)"
            )
        )
        self.db.execute(
            text(
                f"CREATE TABLE {MANIFEST_TABLE} ("
                "market VARCHAR(8) PRIMARY KEY, "
                "expected_dates_json TEXT NOT NULL, "
                "full_market_set BOOLEAN NOT NULL)"
            )
        )
        self.db.commit()

    def record_build_manifest(
        self,
        expected_dates_by_market: Mapping[str, Iterable[date]],
        *,
        full_market_set: bool,
    ) -> None:
        if not self._has_table(MANIFEST_TABLE):
            raise RuntimeError("Breadth rebuild manifest table does not exist")
        self.db.execute(text(f"DELETE FROM {MANIFEST_TABLE}"))
        for raw_market, raw_dates in sorted(expected_dates_by_market.items()):
            market = raw_market.upper()
            dates = tuple(sorted(set(raw_dates)))
            self.db.execute(
                text(
                    f"INSERT INTO {MANIFEST_TABLE} "
                    "(market, expected_dates_json, full_market_set) "
                    "VALUES (:market, :expected_dates_json, :full_market_set)"
                ),
                {
                    "market": market,
                    "expected_dates_json": json.dumps(
                        [value.isoformat() for value in dates]
                    ),
                    "full_market_set": bool(full_market_set),
                },
            )
        self.db.commit()

    def stage_results(
        self,
        results: Iterable[BreadthDailyResult],
        *,
        contributor_snapshots_by_date: Mapping[date, BreadthContributorSnapshotResult],
        duration_seconds_by_date: Mapping[date, float] | None = None,
    ) -> int:
        if not self._has_table(STAGING_TABLE):
            raise RuntimeError("Breadth rebuild staging table does not exist")
        if not self._has_table(
            CONTRIBUTOR_SNAPSHOT_STAGING_TABLE
        ) or not self._has_table(CONTRIBUTOR_STAGING_TABLE):
            raise RuntimeError(
                "Breadth contributor rebuild staging tables do not exist"
            )
        ordered_results = tuple(results)
        expected_dates = {result.calculation_date for result in ordered_results}
        if set(contributor_snapshots_by_date) != expected_dates:
            raise ValueError(
                "Contributor snapshot dates must match staged result dates"
            )
        columns = _copy_columns()
        placeholders = ", ".join(f":{column}" for column in columns)
        column_sql = ", ".join(columns)
        inserted = 0
        for result in ordered_results:
            if result.calculation_revision != CURRENT_BREADTH_CALCULATION_REVISION:
                raise ValueError(
                    "Staging accepts only results from the current canonical "
                    f"breadth revision ({CURRENT_BREADTH_CALCULATION_REVISION})"
                )
            snapshot = contributor_snapshots_by_date[result.calculation_date]
            reconcile_contributor_counts(snapshot, result)
            values = result.to_record_mapping()
            values["contributor_calculation_signature"] = (
                contributor_calculation_signature(snapshot.contributors)
            )
            values["calculation_duration_seconds"] = (
                duration_seconds_by_date.get(result.calculation_date)
                if duration_seconds_by_date is not None
                else None
            )
            self.db.execute(
                text(
                    f"DELETE FROM {STAGING_TABLE} "
                    "WHERE market = :market AND date = :date"
                ),
                {"market": result.market, "date": result.calculation_date},
            )
            self.db.execute(
                text(
                    f"INSERT INTO {STAGING_TABLE} ({column_sql}) "
                    f"VALUES ({placeholders})"
                ),
                {column: values.get(column) for column in columns},
            )
            identity = {
                "market": result.market,
                "date": result.calculation_date,
            }
            self.db.execute(
                text(
                    f"DELETE FROM {CONTRIBUTOR_STAGING_TABLE} "
                    "WHERE market = :market AND date = :date"
                ),
                identity,
            )
            self.db.execute(
                text(
                    f"DELETE FROM {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE} "
                    "WHERE market = :market AND date = :date"
                ),
                identity,
            )
            self.db.execute(
                text(
                    f"INSERT INTO {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE} "
                    "(market, date, calculation_revision, schema_id) "
                    "VALUES (:market, :date, :calculation_revision, :schema_id)"
                ),
                {
                    **identity,
                    "calculation_revision": snapshot.calculation_revision,
                    "schema_id": snapshot.schema_id,
                },
            )
            signals_placeholder = (
                "CAST(:signals_json AS JSON)"
                if self.dialect_name == "postgresql"
                else ":signals_json"
            )
            for contributor in snapshot.contributors:
                self.db.execute(
                    text(
                        f"INSERT INTO {CONTRIBUTOR_STAGING_TABLE} "
                        "(market, date, symbol, company_name, ibd_industry_group, "
                        "daily_change_pct, signals_json) VALUES "
                        "(:market, :date, :symbol, :company_name, "
                        ":ibd_industry_group, :daily_change_pct, "
                        f"{signals_placeholder})"
                    ),
                    {
                        **identity,
                        "symbol": contributor.symbol,
                        "company_name": contributor.company_name,
                        "ibd_industry_group": contributor.ibd_industry_group,
                        "daily_change_pct": contributor.daily_change_pct,
                        "signals_json": json.dumps(dict(contributor.signals)),
                    },
                )
            inserted += 1
        self._prune_staged_contributors()
        self.db.commit()
        return inserted

    def _prune_staged_contributors(self) -> None:
        rows = self.db.execute(
            text(
                f"SELECT market, date FROM {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE} "
                "ORDER BY market, date DESC"
            )
        ).mappings()
        seen_by_market: dict[str, int] = {}
        for row in rows:
            market = str(row["market"])
            seen_by_market[market] = seen_by_market.get(market, 0) + 1
            if seen_by_market[market] <= CONTRIBUTOR_RETENTION_SESSIONS:
                continue
            identity = {"market": market, "date": row["date"]}
            self.db.execute(
                text(
                    f"DELETE FROM {CONTRIBUTOR_STAGING_TABLE} "
                    "WHERE market = :market AND date = :date"
                ),
                identity,
            )
            self.db.execute(
                text(
                    f"DELETE FROM {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE} "
                    "WHERE market = :market AND date = :date"
                ),
                identity,
            )

    def build(
        self,
        *,
        markets: tuple[str, ...],
        start_date: date,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        from app.wiring.bootstrap import get_market_calendar_service, get_price_cache

        calendar = self._calendar_service or get_market_calendar_service()
        price_cache = self._price_cache or get_price_cache()
        normalized_markets = tuple(
            dict.fromkeys(raw_market.upper() for raw_market in markets)
        )
        dates_by_market: dict[str, tuple[date, ...]] = {}
        for market in normalized_markets:
            market_end = end_date or calendar.market_now(market).date()
            dates_by_market[market] = tuple(
                value
                for value in pd.date_range(start=start_date, end=market_end).date
                if calendar.is_trading_day(market, value)
            )
        self.recreate_staging()
        self.record_build_manifest(
            dates_by_market,
            full_market_set=set(normalized_markets) == self._required_markets,
        )
        reports: dict[str, Any] = {}
        for market in normalized_markets:
            dates = dates_by_market[market]
            universes: dict[date, BreadthEligibleUniverse] = {}
            for calculation_date in dates:
                snapshot = self._universe_service.resolve(
                    self.db,
                    market=market,
                    as_of_date=calculation_date,
                )
                universes[calculation_date] = BreadthEligibleUniverse(
                    calculation_date=calculation_date,
                    symbols=snapshot.symbols,
                    eligibility_signature=snapshot.universe_hash,
                )
            calculator = BreadthCalculatorService(
                self.db,
                price_cache,
                market=market,
            )
            calculator.persistence = StagingBreadthPersistence(self)
            plan = BreadthBackfillPlan(
                dates=dates,
                universes=MappingProxyType(universes),
            )
            reports[market] = (
                BreadthBackfillExecutor(calculator)
                .execute(
                    plan,
                    policy=DerivedDataExecutionPolicy(
                        mode=DerivedDataExecutionMode.STRICT_CACHE_ONLY,
                        target_kind=DerivedDataTargetKind.HISTORICAL,
                    ),
                    exclude_unsupported_price_symbols=True,
                    required_as_of_date=dates[-1],
                    require_complete_cache_coverage=True,
                )
                .to_legacy_dict()
            )
        return {
            "markets": reports,
            "processed": sum(value["processed"] for value in reports.values()),
        }

    def validate(
        self,
        *,
        expected_dates_by_market: Mapping[str, Iterable[date]] | None = None,
    ) -> dict[str, Any]:
        if not self._has_table(STAGING_TABLE):
            return {
                "valid": False,
                "errors": ["staging_table_missing"],
                "row_count": 0,
            }
        if not self._has_table(MANIFEST_TABLE):
            return {
                "valid": False,
                "errors": ["staging_manifest_missing"],
                "row_count": 0,
            }
        rows = (
            self.db.execute(
                text(f"SELECT * FROM {STAGING_TABLE} ORDER BY market, date")
            )
            .mappings()
            .all()
        )
        aggregate_by_key: dict[tuple[str, date], Mapping[str, Any]] = {}
        errors: list[str] = []
        manifest_rows = (
            self.db.execute(text(f"SELECT * FROM {MANIFEST_TABLE} ORDER BY market"))
            .mappings()
            .all()
        )
        manifest_dates_by_market: dict[str, set[date]] = {}
        full_market_flags: set[bool] = set()
        for manifest_row in manifest_rows:
            market = str(manifest_row["market"]).upper()
            try:
                raw_dates = json.loads(manifest_row["expected_dates_json"])
                manifest_dates_by_market[market] = {
                    date.fromisoformat(value) for value in raw_dates
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                errors.append(f"invalid_manifest_dates:{market}")
            full_market_flags.add(bool(manifest_row["full_market_set"]))
        if not manifest_rows:
            errors.append("staging_manifest_empty")
        if full_market_flags != {True}:
            errors.append("partial_market_set")
        manifest_markets = set(manifest_dates_by_market)
        errors.extend(
            f"missing_manifest_market:{market}"
            for market in sorted(self._required_markets - manifest_markets)
        )
        errors.extend(
            f"unexpected_manifest_market:{market}"
            for market in sorted(manifest_markets - self._required_markets)
        )
        errors.extend(
            f"empty_manifest_market:{market}"
            for market in sorted(self._required_markets & manifest_markets)
            if not manifest_dates_by_market[market]
        )
        seen: set[tuple[str, date]] = set()
        dates_by_market: dict[str, set[date]] = {}
        for row in rows:
            row_date = (
                date.fromisoformat(row["date"])
                if isinstance(row["date"], str)
                else row["date"]
            )
            key = (str(row["market"]), row_date)
            if key in seen:
                errors.append(f"duplicate:{key[0]}:{key[1]}")
            seen.add(key)
            aggregate_by_key[key] = row
            dates_by_market.setdefault(key[0], set()).add(row_date)
            if row["calculation_revision"] != CURRENT_BREADTH_CALCULATION_REVISION:
                errors.append(f"wrong_revision:{key[0]}:{key[1]}")
            if (
                not row["eligibility_signature"]
                or not row["stockbee_eligibility_signature"]
                or not row["contributor_calculation_signature"]
            ):
                errors.append(f"missing_signature:{key[0]}:{key[1]}")
            broad = row["broad_universe_count"]
            if broad is None or broad < 0:
                errors.append(f"invalid_broad_universe:{key[0]}:{key[1]}")
            eligible_names = (
                "advance_decline_eligible_count",
                "stockbee_daily_eligible_count",
                "stockbee_month_eligible_count",
                "stockbee_34day_eligible_count",
                "stockbee_quarter_eligible_count",
                "t2108_eligible_count",
                "high_low_52week_eligible_count",
                "atr_extension_eligible_count",
                "sma200_eligible_count",
                "off_high_eligible_count",
                "sma20_eligible_count",
                "sma50_eligible_count",
                "trend_template_entry_eligible_count",
            )
            if broad is not None:
                for eligible_name in eligible_names:
                    eligible = row[eligible_name]
                    if eligible is None or not 0 <= eligible <= broad:
                        errors.append(
                            f"invalid_eligibility:{eligible_name}:{key[0]}:{key[1]}"
                        )
            ad_eligible = row["advance_decline_eligible_count"]
            ad_total = sum(
                int(row[name] or 0)
                for name in ("advancing_count", "declining_count", "unchanged_count")
            )
            if ad_eligible is None or ad_total != ad_eligible:
                errors.append(f"advance_decline_mismatch:{key[0]}:{key[1]}")
            pairs = (
                ("stocks_up_4pct", "stockbee_daily_eligible_count"),
                ("stocks_down_4pct", "stockbee_daily_eligible_count"),
                ("stocks_up_25pct_month", "stockbee_month_eligible_count"),
                ("stocks_down_25pct_month", "stockbee_month_eligible_count"),
                ("stocks_up_13pct_34days", "stockbee_34day_eligible_count"),
                ("stocks_down_13pct_34days", "stockbee_34day_eligible_count"),
                ("stocks_up_25pct_quarter", "stockbee_quarter_eligible_count"),
                ("stocks_down_25pct_quarter", "stockbee_quarter_eligible_count"),
                ("up_4pct_near_high_count", "stocks_up_4pct"),
                ("trend_template_entry_count", "trend_template_entry_eligible_count"),
                *(
                    (count_name, eligible_name)
                    for _, count_name, eligible_name in BREADTH_SHARE_FIELDS
                ),
            )
            for count_name, eligible_name in pairs:
                count = row[count_name]
                eligible = row[eligible_name]
                if count is None or eligible is None or not 0 <= count <= eligible:
                    errors.append(
                        f"count_exceeds_eligibility:{count_name}:{key[0]}:{key[1]}"
                    )
            for ratio_name in ("ratio_5day", "ratio_10day"):
                value = row[ratio_name]
                if value is not None and not math.isfinite(float(value)):
                    errors.append(f"non_finite:{ratio_name}:{key[0]}:{key[1]}")
            for pct_name, count_name, eligible_name in BREADTH_SHARE_FIELDS:
                if row[pct_name] != share_pct(row[count_name], row[eligible_name]):
                    errors.append(f"pct_reconciliation:{pct_name}:{key[0]}:{key[1]}")
            mutually_exclusive_pairs = (
                (
                    "stocks_up_4pct",
                    "stocks_down_4pct",
                    "stockbee_daily_eligible_count",
                ),
                (
                    "stocks_up_25pct_month",
                    "stocks_down_25pct_month",
                    "stockbee_month_eligible_count",
                ),
            )
            for up_name, down_name, eligible_name in mutually_exclusive_pairs:
                if int(row[up_name] or 0) + int(row[down_name] or 0) > int(
                    row[eligible_name] or 0
                ):
                    errors.append(
                        f"pair_exceeds_eligibility:{up_name}:{key[0]}:{key[1]}"
                    )
            for count_name, eligible_name in (
                ("new_high_52week_count", "high_low_52week_eligible_count"),
                ("new_low_52week_count", "high_low_52week_eligible_count"),
                ("atr_10x_extension_count", "atr_extension_eligible_count"),
            ):
                if int(row[count_name] or 0) > int(row[eligible_name] or 0):
                    errors.append(
                        f"context_exceeds_eligibility:{count_name}:{key[0]}:{key[1]}"
                    )

        snapshot_rows = (
            self.db.execute(
                text(
                    f"SELECT * FROM {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE} "
                    "ORDER BY market, date"
                )
            )
            .mappings()
            .all()
        )
        snapshot_keys: set[tuple[str, date]] = set()
        for row in snapshot_rows:
            row_date = (
                date.fromisoformat(row["date"])
                if isinstance(row["date"], str)
                else row["date"]
            )
            key = (str(row["market"]), row_date)
            if key in snapshot_keys:
                errors.append(f"duplicate_contributor_snapshot:{key[0]}:{key[1]}")
            snapshot_keys.add(key)
            if row["schema_id"] != CONTRIBUTOR_SCHEMA_ID:
                errors.append(f"wrong_contributor_schema:{key[0]}:{key[1]}")
            if row["calculation_revision"] != CURRENT_BREADTH_CALCULATION_REVISION:
                errors.append(f"wrong_contributor_revision:{key[0]}:{key[1]}")

        expected_snapshot_keys = {
            (market, calculation_date)
            for market, calculation_dates in dates_by_market.items()
            for calculation_date in sorted(calculation_dates, reverse=True)[
                :CONTRIBUTOR_RETENTION_SESSIONS
            ]
        }
        for market, calculation_date in sorted(expected_snapshot_keys - snapshot_keys):
            errors.append(
                f"missing_contributor_snapshot:{market}:{calculation_date.isoformat()}"
            )
        for market, calculation_date in sorted(snapshot_keys - expected_snapshot_keys):
            errors.append(
                f"unexpected_contributor_snapshot:{market}:{calculation_date.isoformat()}"
            )

        contributor_rows = (
            self.db.execute(
                text(
                    f"SELECT * FROM {CONTRIBUTOR_STAGING_TABLE} "
                    "ORDER BY market, date, symbol"
                )
            )
            .mappings()
            .all()
        )
        contributor_rows_by_key: dict[tuple[str, date], list[dict[str, Any]]] = {}
        for row in contributor_rows:
            row_date = (
                date.fromisoformat(row["date"])
                if isinstance(row["date"], str)
                else row["date"]
            )
            key = (str(row["market"]), row_date)
            raw_signals = row["signals_json"]
            if isinstance(raw_signals, str):
                try:
                    raw_signals = json.loads(raw_signals)
                except json.JSONDecodeError:
                    raw_signals = None
            contributor_rows_by_key.setdefault(key, []).append(
                {
                    "symbol": row["symbol"],
                    "company_name": row["company_name"],
                    "ibd_industry_group": row["ibd_industry_group"],
                    "daily_change_pct": row["daily_change_pct"],
                    "signals": raw_signals,
                }
            )

        for key in snapshot_keys:
            aggregate = aggregate_by_key.get(key)
            if aggregate is None:
                errors.append(f"contributor_without_aggregate:{key[0]}:{key[1]}")
                continue
            try:
                parsed = parse_contributor_rows(contributor_rows_by_key.get(key, []))
                reconcile_contributor_aggregate(
                    parsed,
                    {
                        definition.aggregate_field: aggregate[
                            definition.aggregate_field
                        ]
                        for definition in BREADTH_CONTRIBUTOR_SIGNALS.values()
                    },
                )
            except (BreadthContributorContractError, TypeError, ValueError) as exc:
                errors.append(f"invalid_contributors:{key[0]}:{key[1]}:{exc}")

        effective_expected = dict(manifest_dates_by_market)
        for market, expected_dates in (expected_dates_by_market or {}).items():
            normalized_market = market.upper()
            supplied = set(expected_dates)
            if effective_expected.get(normalized_market) != supplied:
                errors.append(f"manifest_mismatch:{normalized_market}")
            effective_expected[normalized_market] = supplied
        for market, expected_dates in effective_expected.items():
            actual_dates = dates_by_market.get(market, set())
            missing = set(expected_dates) - actual_dates
            errors.extend(
                f"missing_date:{market}:{value.isoformat()}"
                for value in sorted(missing)
            )
            unexpected = actual_dates - set(expected_dates)
            errors.extend(
                f"unexpected_date:{market}:{value.isoformat()}"
                for value in sorted(unexpected)
            )
        for market in sorted(set(dates_by_market) - set(effective_expected)):
            errors.append(f"unexpected_market:{market}")

        return {
            "valid": not errors and bool(rows),
            "errors": errors,
            "row_count": len(rows),
            "markets": {
                market: {
                    "row_count": len(values),
                    "start_date": min(values).isoformat() if values else None,
                    "end_date": max(values).isoformat() if values else None,
                }
                for market, values in dates_by_market.items()
            },
            "calculation_revision": CURRENT_BREADTH_CALCULATION_REVISION,
            "formula_contract": {
                "signals": "adjusted_ohlc",
                "liquidity": "raw_close_local_x_volume_adtv20",
                "market_policy": "fixed_market_calibrated_thresholds",
                "currency_mismatch": "stockbee_ineligible_context_preserved",
                "ratios": "today_inclusive",
            },
        }

    def activate(self) -> dict[str, Any]:
        columns = _copy_columns()
        column_sql = ", ".join(columns)
        self.db.rollback()
        with self.db.begin():
            if self.dialect_name == "postgresql":
                self.db.execute(
                    text(
                        "LOCK TABLE "
                        f"{TARGET_TABLE}, {CONTRIBUTOR_SNAPSHOT_TARGET_TABLE}, "
                        f"{CONTRIBUTOR_TARGET_TABLE} IN ACCESS EXCLUSIVE MODE"
                    )
                )
                self.db.execute(
                    text(
                        f"LOCK TABLE {STAGING_TABLE}, {MANIFEST_TABLE}, "
                        f"{CONTRIBUTOR_SNAPSHOT_STAGING_TABLE}, "
                        f"{CONTRIBUTOR_STAGING_TABLE} "
                        "IN ACCESS EXCLUSIVE MODE"
                    )
                )
            report = self.validate()
            if not report["valid"]:
                raise RuntimeError(
                    "Cannot activate invalid breadth rebuild staging data"
                )
            self.db.execute(text(f"DELETE FROM {CONTRIBUTOR_TARGET_TABLE}"))
            self.db.execute(text(f"DELETE FROM {CONTRIBUTOR_SNAPSHOT_TARGET_TABLE}"))
            self.db.execute(text(f"DELETE FROM {TARGET_TABLE}"))
            self.db.execute(
                text(
                    f"INSERT INTO {TARGET_TABLE} ({column_sql}) "
                    f"SELECT {column_sql} FROM {STAGING_TABLE}"
                )
            )
            self.db.execute(
                text(
                    f"INSERT INTO {CONTRIBUTOR_SNAPSHOT_TARGET_TABLE} "
                    "(market, date, calculation_revision, schema_id) "
                    "SELECT market, date, calculation_revision, schema_id "
                    f"FROM {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE}"
                )
            )
            self.db.execute(
                text(
                    f"INSERT INTO {CONTRIBUTOR_TARGET_TABLE} "
                    "(snapshot_id, symbol, company_name, ibd_industry_group, "
                    "daily_change_pct, signals_json) "
                    "SELECT snapshots.id, staged.symbol, staged.company_name, "
                    "staged.ibd_industry_group, staged.daily_change_pct, "
                    "staged.signals_json "
                    f"FROM {CONTRIBUTOR_STAGING_TABLE} AS staged "
                    f"JOIN {CONTRIBUTOR_SNAPSHOT_TARGET_TABLE} AS snapshots "
                    "ON snapshots.market = staged.market AND snapshots.date = staged.date"
                )
            )
            inserted = int(
                self.db.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")).scalar()
                or 0
            )
            wrong_revision = int(
                self.db.execute(
                    text(
                        f"SELECT COUNT(*) FROM {TARGET_TABLE} "
                        "WHERE calculation_revision != :revision "
                        "OR calculation_revision IS NULL"
                    ),
                    {"revision": CURRENT_BREADTH_CALCULATION_REVISION},
                ).scalar()
                or 0
            )
            if inserted != report["row_count"] or wrong_revision:
                raise RuntimeError("Breadth activation verification failed")
            staged_snapshot_count = int(
                self.db.execute(
                    text(f"SELECT COUNT(*) FROM {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE}")
                ).scalar()
                or 0
            )
            activated_snapshot_count = int(
                self.db.execute(
                    text(f"SELECT COUNT(*) FROM {CONTRIBUTOR_SNAPSHOT_TARGET_TABLE}")
                ).scalar()
                or 0
            )
            staged_contributor_count = int(
                self.db.execute(
                    text(f"SELECT COUNT(*) FROM {CONTRIBUTOR_STAGING_TABLE}")
                ).scalar()
                or 0
            )
            activated_contributor_count = int(
                self.db.execute(
                    text(f"SELECT COUNT(*) FROM {CONTRIBUTOR_TARGET_TABLE}")
                ).scalar()
                or 0
            )
            if (
                staged_snapshot_count != activated_snapshot_count
                or staged_contributor_count != activated_contributor_count
            ):
                raise RuntimeError("Breadth contributor activation verification failed")
        return {
            "activated": inserted,
            "calculation_revision": CURRENT_BREADTH_CALCULATION_REVISION,
        }

    def cleanup(self) -> None:
        self.db.execute(text(f"DROP TABLE IF EXISTS {CONTRIBUTOR_STAGING_TABLE}"))
        self.db.execute(
            text(f"DROP TABLE IF EXISTS {CONTRIBUTOR_SNAPSHOT_STAGING_TABLE}")
        )
        self.db.execute(text(f"DROP TABLE IF EXISTS {STAGING_TABLE}"))
        self.db.execute(text(f"DROP TABLE IF EXISTS {MANIFEST_TABLE}"))
        self.db.commit()
