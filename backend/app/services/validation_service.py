"""Deterministic validation service for published signal sources."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from statistics import median
from typing import Any

import pandas as pd
from sqlalchemy import and_, case, func, or_
from sqlalchemy.orm import Session

from app.domain.feature_store.models import INT_TO_RATING
from app.domain.markets.symbol_suffixes import market_symbol_suffix_registry
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.infra.db.portability import json_text
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster
from app.schemas.validation import (
    StockValidationResponse,
    ValidationEvent,
    ValidationFailureCluster,
    ValidationFreshness,
    ValidationHorizonSummary,
    ValidationOverviewResponse,
    ValidationSourceBreakdown,
    ValidationSourceKind,
)
from app.services.benchmark_registry_service import benchmark_registry
from app.services.economic_theme_read_service import EconomicThemeReader
from app.services.price_cache_service import PriceCacheService
from app.utils.market_hours import (
    EASTERN,
    MARKET_OPEN_TIME,
    eastern_day_bounds_utc,
    is_trading_day,
    to_eastern_date,
)

SCAN_PICK_TOP_N = 10
RECENT_EVENTS_LIMIT = 25
FAILURE_CLUSTERS_LIMIT = 5
PRICE_CACHE_PERIOD = "2y"
THEME_ALERT_TYPES = ("breakout", "velocity_spike")
DEFAULT_MARKET = "US"

def _scored_by(field: str):
    """Whether the row carries ``field``; rows before 20261013_0066 hold it only in details_json."""
    return or_(
        getattr(StockFeatureDaily, field).is_not(None),
        and_(
            StockFeatureDaily.hot_fields_stored.is_(None),
            json_text(StockFeatureDaily.details_json, (field,)).is_not(None),
        ),
    )


# Runs before 2026-04-26 carry only universe.market; a run with neither counts as US.
RUN_MARKET = func.coalesce(
    json_text(FeatureRun.config_json, ("market",)),
    json_text(FeatureRun.config_json, ("universe", "market")),
    DEFAULT_MARKET,
)


# A pick is a stock both core methods scored. Rows only the IPO screener ran
# (minervini/canslim NULL) and funds otherwise top the composite ranking.
# is_common_stock alone misses ETFs: finviz marks them common stock.
SCAN_PICK_ELIGIBILITY = and_(
    _scored_by("minervini_score"),
    _scored_by("canslim_score"),
    StockUniverse.is_common_stock.is_not(False),
    StockUniverse.industry.is_distinct_from("Exchange Traded Fund"),
)


@dataclass(frozen=True)
class RawValidationEvent:
    """One raw validation event before market outcomes are attached."""

    symbol: str
    source_kind: ValidationSourceKind
    source_ref: str
    event_at: date | datetime
    attributes: dict[str, Any]


@dataclass(frozen=True)
class EvaluatedValidationEvent:
    """One validation event after deterministic price-outcome calculation."""

    raw: RawValidationEvent
    entry_at: date | None
    entry_price: float | None
    return_1s_pct: float | None
    return_5s_pct: float | None
    mfe_5s_pct: float | None
    mae_5s_pct: float | None
    missing_horizons: frozenset[int]
    exit_at: date | None = None
    excess_return_1s_pct: float | None = None
    excess_return_5s_pct: float | None = None

    def to_response(self) -> ValidationEvent:
        return ValidationEvent(
            source_kind=self.raw.source_kind,
            source_ref=self.raw.source_ref,
            event_at=_serialize_temporal(self.raw.event_at),
            entry_at=self.entry_at.isoformat() if self.entry_at else None,
            entry_price=_round_or_none(self.entry_price),
            return_1s_pct=_round_or_none(self.return_1s_pct),
            return_5s_pct=_round_or_none(self.return_5s_pct),
            mfe_5s_pct=_round_or_none(self.mfe_5s_pct),
            mae_5s_pct=_round_or_none(self.mae_5s_pct),
            excess_return_1s_pct=_round_or_none(self.excess_return_1s_pct),
            excess_return_5s_pct=_round_or_none(self.excess_return_5s_pct),
            attributes=self.raw.attributes,
        )


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _serialize_temporal(value: date | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return value.isoformat()


def _event_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _normalize_event_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return EASTERN.localize(value)
    return value.astimezone(EASTERN)


def _safe_upper_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


def _normalize_price_history(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized.sort_index()

    required_columns = {"Open", "High", "Low", "Close"}
    if not required_columns.issubset(set(normalized.columns)):
        return None
    return normalized


def _excess_return_pct(
    event_return_pct: float | None,
    benchmark: pd.DataFrame | None,
    entry_day: pd.Timestamp,
    close_day: pd.Timestamp | None,
) -> float | None:
    if event_return_pct is None or benchmark is None:
        return None
    if entry_day not in benchmark.index or close_day not in benchmark.index:
        return None
    benchmark_open = float(benchmark.at[entry_day, "Open"])
    if benchmark_open <= 0:
        return None
    benchmark_return_pct = (float(benchmark.at[close_day, "Close"]) / benchmark_open - 1.0) * 100.0
    return event_return_pct - benchmark_return_pct


def drop_overlapping_windows(
    events: list[EvaluatedValidationEvent],
) -> tuple[list[EvaluatedValidationEvent], int]:
    """Keep one event per (source, symbol) window, in input order; also return the suppressed count.

    An event counts only when it enters strictly after the previous kept event's
    fifth session. A kept event whose window is still open blocks every later one.
    """
    suppressed: set[int] = set()
    blocked_through: dict[tuple[ValidationSourceKind, str], date] = {}
    for index, event in sorted(
        enumerate(events),
        key=lambda pair: (
            pair[1].raw.source_kind,
            pair[1].raw.symbol,
            pair[1].entry_at or date.min,
            _serialize_temporal(pair[1].raw.event_at),
            pair[1].raw.source_ref,
        ),
    ):
        if event.entry_at is None:
            continue
        key = (event.raw.source_kind, event.raw.symbol)
        if event.entry_at <= blocked_through.get(key, date.min):
            suppressed.add(index)
            continue
        blocked_through[key] = event.exit_at or date.max
    kept = [event for index, event in enumerate(events) if index not in suppressed]
    return kept, len(suppressed)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _symbol_markets(db: Session, symbols: set[str]) -> dict[str, str]:
    """Each symbol's listed market, else the one its suffix names, else US."""
    listed = dict(
        db.query(StockUniverse.symbol, StockUniverse.market)
        .filter(StockUniverse.symbol.in_(symbols))
        .all()
    )
    return {
        symbol: listed.get(symbol)
        or market_symbol_suffix_registry.market_for_symbol(symbol)
        or DEFAULT_MARKET
        for symbol in symbols
    }


class ScanPickValidationSource:
    """Extract one validation event per top scan pick in a published run."""

    def collect(
        self,
        db: Session,
        *,
        cutoff_date: date,
        market: str,
        until_date: date | None = None,
        symbol: str | None = None,
    ) -> tuple[list[RawValidationEvent], list[str]]:
        rank_expr = func.row_number().over(
            partition_by=StockFeatureDaily.run_id,
            order_by=(
                case((StockFeatureDaily.composite_score.is_(None), 1), else_=0),
                StockFeatureDaily.composite_score.desc(),
                StockFeatureDaily.symbol.asc(),
            ),
        ).label("run_rank")
        ranked_rows = (
            db.query(
                FeatureRun.id.label("run_id"),
                FeatureRun.as_of_date.label("as_of_date"),
                StockFeatureDaily.symbol.label("symbol"),
                StockFeatureDaily.composite_score.label("composite_score"),
                StockFeatureDaily.overall_rating.label("overall_rating"),
                StockFeatureDaily.details_json.label("details_json"),
                rank_expr,
            )
            .join(StockFeatureDaily, StockFeatureDaily.run_id == FeatureRun.id)
            # Outer: the universe sync drops delisted symbols, and their past picks still count.
            .outerjoin(StockUniverse, StockUniverse.symbol == StockFeatureDaily.symbol)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.as_of_date >= cutoff_date,
                RUN_MARKET == market,
                SCAN_PICK_ELIGIBILITY,
            )
            .subquery()
        )

        query = (
            db.query(ranked_rows)
            .filter(ranked_rows.c.run_rank <= SCAN_PICK_TOP_N)
            .order_by(
                ranked_rows.c.as_of_date.desc(),
                ranked_rows.c.run_id.desc(),
                ranked_rows.c.run_rank.asc(),
            )
        )
        if symbol:
            query = query.filter(ranked_rows.c.symbol == symbol)
        if until_date is not None:
            query = query.filter(ranked_rows.c.as_of_date <= until_date)

        rows = query.all()
        if not rows:
            return [], ["no_matching_scan_picks"]

        events: list[RawValidationEvent] = []
        for row in rows:
            details = row.details_json or {}
            events.append(
                RawValidationEvent(
                    symbol=row.symbol,
                    source_kind=ValidationSourceKind.SCAN_PICK,
                    source_ref=f"run:{row.run_id}",
                    event_at=row.as_of_date,
                    attributes={
                        "symbol": row.symbol,
                        "run_id": row.run_id,
                        "run_as_of_date": row.as_of_date.isoformat(),
                        "composite_score": _round_or_none(row.composite_score),
                        "rating": INT_TO_RATING.get(row.overall_rating, details.get("rating")),
                        "stage": details.get("stage"),
                        "ibd_industry_group": details.get("ibd_industry_group"),
                    },
                )
            )
        return events, []


class ThemeAlertValidationSource:
    """Extract one validation event per related ticker in supported theme alerts."""

    def collect(
        self,
        db: Session,
        *,
        cutoff_date: date,
        market: str,
        until_date: date | None = None,
        symbol: str | None = None,
    ) -> tuple[list[RawValidationEvent], list[str]]:
        if market != "US":
            # A legacy US source: same-day entry is timed against the US session.
            return [], ["theme_alerts_us_only"]
        if EconomicThemeReader(db).source_name == "economic":
            # Theme alerts are legacy-only; economic mode serves none, as /themes/alerts does (#475).
            return [], ["economic_theme_alerts_unavailable"]
        cutoff_datetime, _ = eastern_day_bounds_utc(cutoff_date)
        query = (
            db.query(ThemeAlert, ThemeCluster.display_name)
            .outerjoin(ThemeCluster, ThemeCluster.id == ThemeAlert.theme_cluster_id)
            .filter(
                ThemeAlert.alert_type.in_(THEME_ALERT_TYPES),
                ThemeAlert.triggered_at >= cutoff_datetime,
            )
        )
        if until_date is not None:
            _, until_datetime = eastern_day_bounds_utc(until_date)
            query = query.filter(ThemeAlert.triggered_at < until_datetime)
        rows = query.order_by(ThemeAlert.triggered_at.desc(), ThemeAlert.id.desc()).all()

        events: list[RawValidationEvent] = []
        for alert, theme_name in rows:
            if alert.triggered_at is None or to_eastern_date(alert.triggered_at) < cutoff_date:
                continue
            tickers = alert.related_tickers if isinstance(alert.related_tickers, list) else []
            for ticker in tickers:
                normalized_symbol = _safe_upper_symbol(ticker)
                if normalized_symbol is None:
                    continue
                if symbol and normalized_symbol != symbol:
                    continue
                events.append(
                    RawValidationEvent(
                        symbol=normalized_symbol,
                        source_kind=ValidationSourceKind.THEME_ALERT,
                        source_ref=f"alert:{alert.id}",
                        event_at=alert.triggered_at,
                        attributes={
                            "symbol": normalized_symbol,
                            "alert_id": alert.id,
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "theme": theme_name,
                            "theme_cluster_id": alert.theme_cluster_id,
                            "title": alert.title,
                        },
                    )
                )

        markets = _symbol_markets(db, {event.symbol for event in events})
        events = [event for event in events if markets[event.symbol] == market]
        if not events:
            return [], ["no_matching_theme_alerts"]
        return events, []


class PriceOutcomeCalculator:
    """Compute the fixed validation horizons from cached OHLCV data."""

    def __init__(self, price_cache: PriceCacheService) -> None:
        self._price_cache = price_cache

    def evaluate_many(
        self,
        events: list[RawValidationEvent],
        *,
        benchmark_candidates: Sequence[str] = (),
    ) -> tuple[list[EvaluatedValidationEvent], list[str], str | None]:
        """Evaluate events; also return degraded reasons and the benchmark used."""
        histories = self._price_cache.get_many_cached_only(
            sorted({event.symbol for event in events} | set(benchmark_candidates)),
            period=PRICE_CACHE_PERIOD,
        )
        cached_benchmarks = [
            (candidate, candidate_history)
            for candidate in benchmark_candidates
            if (candidate_history := _normalize_price_history(histories.get(candidate))) is not None
        ]
        # A primary whose feed stopped keeps its old rows; the freshest wins, ties in registry order.
        benchmark_symbol, benchmark = max(
            cached_benchmarks, key=lambda pair: pair[1].index[-1], default=(None, None)
        )
        degraded_reasons: list[str] = []
        evaluations: list[EvaluatedValidationEvent] = []

        for event in events:
            history = _normalize_price_history(histories.get(event.symbol))
            if history is None:
                degraded_reasons.append("missing_price_cache")
            evaluation = self._evaluate_one(event, history, benchmark)
            if benchmark_candidates and (
                (evaluation.return_1s_pct is not None and evaluation.excess_return_1s_pct is None)
                or (evaluation.return_5s_pct is not None and evaluation.excess_return_5s_pct is None)
            ):
                degraded_reasons.append("missing_benchmark_prices")
            evaluations.append(evaluation)

        return evaluations, _dedupe(degraded_reasons), benchmark_symbol

    def _evaluate_one(
        self,
        event: RawValidationEvent,
        history: pd.DataFrame | None,
        benchmark: pd.DataFrame | None,
    ) -> EvaluatedValidationEvent:
        if history is None or history.empty:
            return EvaluatedValidationEvent(
                raw=event,
                entry_at=None,
                entry_price=None,
                return_1s_pct=None,
                return_5s_pct=None,
                mfe_5s_pct=None,
                mae_5s_pct=None,
                missing_horizons=frozenset({1, 5}),
            )

        if isinstance(event.event_at, datetime):
            event_dt = _normalize_event_datetime(event.event_at)
            event_day = pd.Timestamp(event_dt.date())
            same_day_entry_allowed = (
                is_trading_day(event_dt.date())
                and event_dt.time() < MARKET_OPEN_TIME
                and event_day in history.index
            )
            if same_day_entry_allowed:
                trading_days = history.index[history.index >= event_day]
            else:
                trading_days = history.index[history.index > event_day]
        else:
            event_day = pd.Timestamp(event.event_at)
            trading_days = history.index[history.index > event_day]
        if len(trading_days) == 0:
            return EvaluatedValidationEvent(
                raw=event,
                entry_at=None,
                entry_price=None,
                return_1s_pct=None,
                return_5s_pct=None,
                mfe_5s_pct=None,
                mae_5s_pct=None,
                missing_horizons=frozenset({1, 5}),
            )

        entry_day = trading_days[0]
        exit_day = trading_days[4] if len(trading_days) >= 5 else None
        exit_at = exit_day.date() if exit_day is not None else None
        entry_row = history.loc[entry_day]
        entry_price = float(entry_row["Open"])
        if entry_price <= 0:
            return EvaluatedValidationEvent(
                raw=event,
                entry_at=entry_day.date(),
                entry_price=None,
                return_1s_pct=None,
                return_5s_pct=None,
                mfe_5s_pct=None,
                mae_5s_pct=None,
                missing_horizons=frozenset({1, 5}),
                exit_at=exit_at,
            )
        close_1s = float(entry_row["Close"])
        return_1s_pct = ((close_1s / entry_price) - 1.0) * 100.0

        missing_horizons: set[int] = set()
        return_5s_pct: float | None = None
        mfe_5s_pct: float | None = None
        mae_5s_pct: float | None = None
        if exit_day is not None:
            window = history.loc[trading_days[:5]]
            return_5s_pct = ((float(window.iloc[-1]["Close"]) / entry_price) - 1.0) * 100.0
            mfe_5s_pct = ((float(window["High"].max()) / entry_price) - 1.0) * 100.0
            mae_5s_pct = ((float(window["Low"].min()) / entry_price) - 1.0) * 100.0
        else:
            missing_horizons.add(5)

        return EvaluatedValidationEvent(
            raw=event,
            entry_at=entry_day.date(),
            entry_price=entry_price,
            return_1s_pct=return_1s_pct,
            return_5s_pct=return_5s_pct,
            mfe_5s_pct=mfe_5s_pct,
            mae_5s_pct=mae_5s_pct,
            missing_horizons=frozenset(missing_horizons),
            exit_at=exit_at,
            excess_return_1s_pct=_excess_return_pct(return_1s_pct, benchmark, entry_day, entry_day),
            excess_return_5s_pct=_excess_return_pct(return_5s_pct, benchmark, entry_day, exit_day),
        )


class FailureClusterBuilder:
    """Bucket losing events into deterministic failure clusters."""

    def build(self, events: list[EvaluatedValidationEvent]) -> list[ValidationFailureCluster]:
        bucket_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        labels: dict[tuple[str, str], str] = {}

        for event in events:
            if event.return_5s_pct is None or event.return_5s_pct >= 0:
                continue
            for cluster_key, label in self._cluster_labels(event):
                bucket_values[cluster_key].append(event.return_5s_pct)
                labels[cluster_key] = label

        ranked_buckets = sorted(
            bucket_values.items(),
            key=lambda item: (-len(item[1]), sum(item[1]) / len(item[1]), item[0][0], item[0][1]),
        )[:FAILURE_CLUSTERS_LIMIT]

        return [
            ValidationFailureCluster(
                cluster_key=f"{key[0]}:{key[1]}",
                label=labels[key],
                sample_size=len(values),
                avg_return_pct=round(sum(values) / len(values), 4),
                median_return_pct=round(float(median(values)), 4),
            )
            for key, values in ranked_buckets
        ]

    def _cluster_labels(
        self,
        event: EvaluatedValidationEvent,
    ) -> list[tuple[tuple[str, str], str]]:
        attrs = event.raw.attributes
        if event.raw.source_kind == ValidationSourceKind.SCAN_PICK:
            mapping = {
                "rating": attrs.get("rating"),
                "stage": attrs.get("stage"),
                "ibd_industry_group": attrs.get("ibd_industry_group"),
            }
        else:
            mapping = {
                "alert_type": attrs.get("alert_type"),
                "severity": attrs.get("severity"),
                "theme": attrs.get("theme"),
            }

        clusters: list[tuple[tuple[str, str], str]] = []
        for field, value in mapping.items():
            if value is None or value == "":
                continue
            normalized = str(value)
            clusters.append(((field, normalized), f"{field.replace('_', ' ').title()}: {normalized}"))
        return clusters


class ValidationService:
    """Compose signal sources, price outcomes, and deterministic summaries."""

    def __init__(
        self,
        *,
        scan_pick_source: ScanPickValidationSource | None = None,
        theme_alert_source: ThemeAlertValidationSource | None = None,
        outcome_calculator: PriceOutcomeCalculator | None = None,
        failure_cluster_builder: FailureClusterBuilder | None = None,
    ) -> None:
        from app.wiring.bootstrap import get_price_cache

        price_cache = get_price_cache()
        self._scan_pick_source = scan_pick_source or ScanPickValidationSource()
        self._theme_alert_source = theme_alert_source or ThemeAlertValidationSource()
        self._outcome_calculator = outcome_calculator or PriceOutcomeCalculator(price_cache)
        self._failure_cluster_builder = failure_cluster_builder or FailureClusterBuilder()

    def get_overview(
        self,
        db: Session,
        *,
        source_kind: ValidationSourceKind,
        lookback_days: int,
        as_of_date: date | None = None,
        market: str = DEFAULT_MARKET,
    ) -> ValidationOverviewResponse:
        market = market.strip().upper()
        effective_as_of_date = self._effective_as_of_date(as_of_date)
        payload = self._build_source_payload(
            db,
            source_kind=source_kind,
            lookback_days=lookback_days,
            as_of_date=effective_as_of_date,
            market=market,
        )
        return ValidationOverviewResponse(
            source_kind=source_kind,
            lookback_days=lookback_days,
            market=market,
            benchmark_symbol=payload["benchmark_symbol"],
            horizons=payload["horizons"],
            recent_events=payload["recent_events"],
            failure_clusters=payload["failure_clusters"],
            freshness=self._build_freshness(db, as_of_date=effective_as_of_date, market=market),
            degraded_reasons=payload["degraded_reasons"],
        )

    def get_stock_validation(
        self,
        db: Session,
        *,
        symbol: str,
        lookback_days: int,
        as_of_date: date | None = None,
    ) -> StockValidationResponse:
        symbol = symbol.upper()
        market = _symbol_markets(db, {symbol})[symbol]
        effective_as_of_date = self._effective_as_of_date(as_of_date)
        source_breakdown: list[ValidationSourceBreakdown] = []
        merged_events: list[ValidationEvent] = []
        merged_failure_clusters: list[ValidationFailureCluster] = []
        degraded_reasons: list[str] = []

        for source_kind in (ValidationSourceKind.SCAN_PICK, ValidationSourceKind.THEME_ALERT):
            payload = self._build_source_payload(
                db,
                source_kind=source_kind,
                lookback_days=lookback_days,
                as_of_date=effective_as_of_date,
                market=market,
                symbol=symbol,
            )
            source_breakdown.append(
                ValidationSourceBreakdown(
                    source_kind=source_kind,
                    horizons=payload["horizons"],
                    recent_events=payload["recent_events"],
                    failure_clusters=payload["failure_clusters"],
                    degraded_reasons=payload["degraded_reasons"],
                )
            )
            merged_events.extend(payload["recent_events"])
            merged_failure_clusters.extend(payload["failure_clusters"])
            degraded_reasons.extend(payload["degraded_reasons"])

        merged_events.sort(key=lambda item: (item.event_at, item.source_ref), reverse=True)
        merged_failure_clusters.sort(
            key=lambda item: (-item.sample_size, item.avg_return_pct, item.cluster_key)
        )

        return StockValidationResponse(
            symbol=symbol,
            lookback_days=lookback_days,
            source_breakdown=source_breakdown,
            recent_events=merged_events[:RECENT_EVENTS_LIMIT],
            failure_clusters=merged_failure_clusters[:FAILURE_CLUSTERS_LIMIT],
            freshness=self._build_freshness(db, as_of_date=effective_as_of_date, market=market),
            degraded_reasons=_dedupe(degraded_reasons),
        )

    def _effective_as_of_date(self, as_of_date: date | None) -> date:
        if as_of_date is not None:
            return as_of_date
        return to_eastern_date(datetime.now(UTC))

    def _build_source_payload(
        self,
        db: Session,
        *,
        source_kind: ValidationSourceKind,
        lookback_days: int,
        market: str,
        as_of_date: date | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        effective_as_of_date = self._effective_as_of_date(as_of_date)
        cutoff_date = effective_as_of_date - timedelta(days=lookback_days)
        source = (
            self._scan_pick_source
            if source_kind == ValidationSourceKind.SCAN_PICK
            else self._theme_alert_source
        )
        raw_events, degraded_reasons = source.collect(
            db,
            cutoff_date=cutoff_date,
            market=market,
            until_date=effective_as_of_date,
            symbol=symbol,
        )
        evaluated_events, calculator_degraded, benchmark_symbol = (
            self._outcome_calculator.evaluate_many(
                raw_events,
                benchmark_candidates=benchmark_registry.get_candidate_symbols(market),
            )
        )
        degraded_reasons.extend(calculator_degraded)
        kept_events, suppressed_overlapping = drop_overlapping_windows(evaluated_events)

        recent_events = [
            event.to_response()
            for event in sorted(
                kept_events,
                key=lambda item: (_serialize_temporal(item.raw.event_at), item.raw.source_ref),
                reverse=True,
            )[:RECENT_EVENTS_LIMIT]
        ]

        return {
            "horizons": [
                self._summarize_horizon(
                    kept_events,
                    horizon_sessions=horizon_sessions,
                    suppressed_overlapping=suppressed_overlapping,
                )
                for horizon_sessions in (1, 5)
            ],
            "recent_events": recent_events,
            "failure_clusters": self._failure_cluster_builder.build(kept_events),
            "degraded_reasons": _dedupe(degraded_reasons),
            "benchmark_symbol": benchmark_symbol,
        }

    def _summarize_horizon(
        self,
        events: list[EvaluatedValidationEvent],
        *,
        horizon_sessions: int,
        suppressed_overlapping: int,
    ) -> ValidationHorizonSummary:
        returns: list[float] = []
        mfe_values: list[float] = []
        mae_values: list[float] = []
        excess_values: list[float] = []
        skipped_missing_history = 0

        for event in events:
            if horizon_sessions in event.missing_horizons or event.entry_price is None:
                skipped_missing_history += 1
                continue

            if horizon_sessions == 1:
                if event.return_1s_pct is None:
                    skipped_missing_history += 1
                    continue
                returns.append(event.return_1s_pct)
                if event.excess_return_1s_pct is not None:
                    excess_values.append(event.excess_return_1s_pct)
                continue

            if event.return_5s_pct is None:
                skipped_missing_history += 1
                continue
            returns.append(event.return_5s_pct)
            if event.excess_return_5s_pct is not None:
                excess_values.append(event.excess_return_5s_pct)
            if event.mfe_5s_pct is not None:
                mfe_values.append(event.mfe_5s_pct)
            if event.mae_5s_pct is not None:
                mae_values.append(event.mae_5s_pct)

        sample_size = len(returns)
        positive_rate = None
        avg_return_pct = None
        median_return_pct = None
        avg_mfe_pct = None
        avg_mae_pct = None
        if sample_size:
            positive_rate = round(sum(1 for value in returns if value > 0) / sample_size, 4)
            avg_return_pct = round(sum(returns) / sample_size, 4)
            median_return_pct = round(float(median(returns)), 4)
            if mfe_values:
                avg_mfe_pct = round(sum(mfe_values) / len(mfe_values), 4)
            if mae_values:
                avg_mae_pct = round(sum(mae_values) / len(mae_values), 4)

        return ValidationHorizonSummary(
            horizon_sessions=horizon_sessions,
            sample_size=sample_size,
            positive_rate=positive_rate,
            avg_return_pct=avg_return_pct,
            median_return_pct=median_return_pct,
            avg_mfe_pct=avg_mfe_pct,
            avg_mae_pct=avg_mae_pct,
            avg_excess_return_pct=(
                round(sum(excess_values) / len(excess_values), 4) if excess_values else None
            ),
            median_excess_return_pct=(
                round(float(median(excess_values)), 4) if excess_values else None
            ),
            skipped_missing_history=skipped_missing_history,
            suppressed_overlapping=suppressed_overlapping,
        )

    def _build_freshness(
        self, db: Session, *, market: str, as_of_date: date | None = None
    ) -> ValidationFreshness:
        feature_query = db.query(func.max(FeatureRun.as_of_date)).filter(
            FeatureRun.status == "published", RUN_MARKET == market
        )
        if as_of_date is not None:
            feature_query = feature_query.filter(FeatureRun.as_of_date <= as_of_date)
        latest_feature_as_of_date = feature_query.scalar()

        latest_theme_alert_at = None
        if market == "US" and EconomicThemeReader(db).source_name != "economic":
            alert_query = db.query(func.max(ThemeAlert.triggered_at)).filter(
                ThemeAlert.alert_type.in_(THEME_ALERT_TYPES)
            )
            if as_of_date is not None:
                _, cutoff = eastern_day_bounds_utc(as_of_date)
                alert_query = alert_query.filter(ThemeAlert.triggered_at < cutoff)
            latest_theme_alert_at = alert_query.scalar()
        return ValidationFreshness(
            latest_feature_as_of_date=(
                latest_feature_as_of_date.isoformat() if latest_feature_as_of_date else None
            ),
            latest_theme_alert_at=(
                _serialize_temporal(latest_theme_alert_at) if latest_theme_alert_at else None
            ),
            price_cache_period=PRICE_CACHE_PERIOD,
        )
