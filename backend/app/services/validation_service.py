"""Deterministic validation service for published signal sources."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from functools import partial
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import and_, case, exists, func
from sqlalchemy.orm import Session

from app.domain.feature_store.models import INT_TO_RATING, RATING_TO_INT
from app.domain.markets.symbol_suffixes import market_symbol_suffix_registry
from app.domain.scanning.opportunity_state.model import ACTION_STATES
from app.infra.db.models.backtest_replay import DEFINITION_VERSION, BacktestReplayPick
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.infra.db.portability import json_text
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster
from app.schemas.validation import (
    StockValidationResponse,
    ValidationEvent,
    ValidationFreshness,
    ValidationGroup,
    ValidationHorizonSummary,
    ValidationOverviewResponse,
    ValidationSourceBreakdown,
    ValidationSourceKind,
)
from app.services.benchmark_registry_service import benchmark_registry
from app.services.economic_theme_read_service import EconomicThemeReader
from app.services.market_calendar_service import (
    CalendarCoverageExpired,
    MarketCalendarService,
)
from app.services.price_cache_service import PriceCacheService
from app.utils.market_hours import (
    EASTERN,
    MARKET_OPEN_TIME,
    eastern_day_bounds_utc,
    to_eastern_date,
)

SCAN_PICK_TOP_N = 10
RECENT_EVENTS_LIMIT = 25
PRICE_CACHE_PERIOD = "2y"
THEME_ALERT_TYPES = ("breakout", "velocity_spike")
DEFAULT_MARKET = "US"
HORIZONS = (1, 5, 20, 60)
STOP_LOSS_PCT = 7.0
# Events this far before the cutoff join the overlap rule, so a pick before it still blocks
# the repeats inside its window. ponytail: one longest-horizon window at 7 calendar days per
# 5 sessions, plus 21 days for exchange holidays (TW Lunar New Year). Counting is exact only
# while no symbol's chain of overlapping picks reaches further back: an older pick decides
# whether the next one counts, and so whether that one blocks a pick in the lookback. Upgrade
# path: replay every earlier pick of the window's symbols, which means ranking every earlier run.
# Replay picks are stored already, so they count over all of them instead.
OVERLAP_LOOKBEHIND = timedelta(days=math.ceil(max(HORIZONS) * 7 / 5) + 21)
# Calendar days that hold the 50 sessions before an entry that extension reads, with the
# same holiday margin.
EXTENSION_LOOKBEHIND = timedelta(days=math.ceil(50 * 7 / 5) + 21)
NO_EVENTS_REASONS = {
    ValidationSourceKind.SCAN_PICK: "no_matching_scan_picks",
    ValidationSourceKind.THEME_ALERT: "no_matching_theme_alerts",
    ValidationSourceKind.REPLAY_PICK: "no_matching_replay_picks",
}
# The dimensions each source's events are grouped by, in display order. Industry group and
# theme are left out: they split the picks into too many groups to mean anything.
GROUP_DIMENSIONS = {
    ValidationSourceKind.SCAN_PICK: (
        "rating",
        "stage",
        "action_state",
        "setup_pattern",
        "extension_band",
    ),
    ValidationSourceKind.REPLAY_PICK: ("stage", "rs_band", "extension_band"),
    ValidationSourceKind.THEME_ALERT: ("alert_type", "severity", "extension_band"),
}
# Labels that title-casing the dimension name gets wrong.
GROUP_LABELS = {"rs_band": "RS Band", "extension_band": "Extension vs 50DMA"}
EXTENSION_BANDS = ("below 50DMA", "0-10%", "10-20%", "20-30%", "30%+")
# Where each band after the first starts, in percent above the 50-day average.
EXTENSION_BAND_EDGES = (0.0, 10.0, 20.0, 30.0)
# The value order of each dimension that has one, such as ratings best first.
GROUP_VALUE_ORDER = {
    "rating": tuple(sorted(RATING_TO_INT, key=RATING_TO_INT.__getitem__, reverse=True)),
    "rs_band": tuple(f"{low}-{low + 9}" for low in range(90, -1, -10)),
    "action_state": tuple(state.value for state in ACTION_STATES),
    "extension_band": EXTENSION_BANDS,
}


# Runs before 2026-04-26 carry only universe.market; a run with neither counts as US.
RUN_MARKET = func.coalesce(
    json_text(FeatureRun.config_json, ("market",)),
    json_text(FeatureRun.config_json, ("universe", "market")),
    DEFAULT_MARKET,
)


# is_common_stock alone misses ETFs: finviz marks them common stock. A symbol the
# universe no longer lists passes, so delisted symbols' past picks still count.
NOT_A_FUND = and_(
    StockUniverse.is_common_stock.is_not(False),
    StockUniverse.industry.is_distinct_from("Exchange Traded Fund"),
)


# A pick is a stock both core methods scored. Rows only the IPO screener ran
# (minervini/canslim NULL) and funds otherwise top the composite ranking.
# Rows written before 20261013_0066 lack the typed scores and never qualify; reading
# details_json instead costs a full JSON parse per row on Postgres.
SCAN_PICK_ELIGIBILITY = and_(
    StockFeatureDaily.minervini_score.is_not(None),
    StockFeatureDaily.canslim_score.is_not(None),
    NOT_A_FUND,
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
class HorizonOutcome:
    """An event's result over its first ``h`` sessions, entered at the first session's open."""

    return_pct: float
    mfe_pct: float
    mae_pct: float
    stopped_return_pct: float
    stopped_out: bool
    excess_return_pct: float | None = None


@dataclass(frozen=True)
class EvaluatedValidationEvent:
    """One validation event after deterministic price-outcome calculation."""

    raw: RawValidationEvent
    entry_at: date | None
    entry_price: float | None
    # Keyed by horizon: the market session each window that has closed ends on.
    exits: dict[int, date] = field(default_factory=dict)
    # The closed windows whose every session is cached.
    outcomes: dict[int, HorizonOutcome] = field(default_factory=dict)
    # Horizons at which this event repeats inside an earlier counted event's window.
    suppressed: frozenset[int] = frozenset()
    history_missing: bool = False
    pct_above_50dma: float | None = None

    def to_response(self) -> ValidationEvent:
        return ValidationEvent(
            source_kind=self.raw.source_kind,
            source_ref=self.raw.source_ref,
            event_at=_serialize_temporal(self.raw.event_at),
            entry_at=self.entry_at.isoformat() if self.entry_at else None,
            entry_price=_round_or_none(self.entry_price),
            return_1s_pct=self._outcome_pct(1, "return_pct"),
            return_5s_pct=self._outcome_pct(5, "return_pct"),
            return_20s_pct=self._outcome_pct(20, "return_pct"),
            return_60s_pct=self._outcome_pct(60, "return_pct"),
            mfe_5s_pct=self._outcome_pct(5, "mfe_pct"),
            mae_5s_pct=self._outcome_pct(5, "mae_pct"),
            excess_return_1s_pct=self._outcome_pct(1, "excess_return_pct"),
            excess_return_5s_pct=self._outcome_pct(5, "excess_return_pct"),
            counted=5 not in self.suppressed,
            counted_horizons=[horizon for horizon in HORIZONS if horizon not in self.suppressed],
            attributes=self.raw.attributes,
        )

    def _outcome_pct(self, horizon: int, name: str) -> float | None:
        outcome = self.outcomes.get(horizon)
        return _round_or_none(getattr(outcome, name)) if outcome else None


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _median(values: list[float]) -> float | None:
    return round(float(median(values)), 4) if values else None


def _serialize_temporal(value: date | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return value.isoformat()


def _event_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return to_eastern_date(value)
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


def _normalize_price_history(
    df: pd.DataFrame | None, through: date | None = None
) -> pd.DataFrame | None:
    """Daily OHLC bars by exchange date, dropping any bar after ``through``."""
    if df is None or df.empty:
        return None

    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized.sort_index()
    if through is not None:
        normalized = normalized[normalized.index <= pd.Timestamp(through)]

    required_columns = {"Open", "High", "Low", "Close"}
    if normalized.empty or not required_columns.issubset(set(normalized.columns)):
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
    benchmark_close = float(benchmark.at[close_day, "Close"])
    if not (benchmark_open > 0 and benchmark_close > 0):
        return None
    benchmark_return_pct = (benchmark_close / benchmark_open - 1.0) * 100.0
    return event_return_pct - benchmark_return_pct


def mark_overlapping_windows(
    events: list[EvaluatedValidationEvent],
) -> list[EvaluatedValidationEvent]:
    """Return ``events`` in input order, each marking the horizons at which it is a repeat.

    At a horizon of h sessions, an event counts only when it enters strictly after the
    previous counted event's h-th market session for the same source and symbol, cached or
    not. A counted event whose h-session window is still open blocks every later one; an
    event with no entry price blocks nothing. Every event counts at 1 session, where the
    window is the entry day alone.
    """
    entered = sorted(
        (pair for pair in enumerate(events) if pair[1].entry_price is not None),
        key=lambda pair: (
            pair[1].raw.source_kind,
            pair[1].raw.symbol,
            pair[1].entry_at,
            _serialize_temporal(pair[1].raw.event_at),
            pair[1].raw.source_ref,
        ),
    )
    suppressed: dict[int, set[int]] = defaultdict(set)
    for horizon in HORIZONS:
        if horizon == 1:
            continue
        blocked_through: dict[tuple[ValidationSourceKind, str], date] = {}
        for index, event in entered:
            key = (event.raw.source_kind, event.raw.symbol)
            if event.entry_at <= blocked_through.get(key, date.min):
                suppressed[index].add(horizon)
                continue
            blocked_through[key] = event.exits.get(horizon, date.max)
    return [
        replace(event, suppressed=frozenset(suppressed[index])) if index in suppressed else event
        for index, event in enumerate(events)
    ]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _outcome_degraded_reasons(
    events: list[EvaluatedValidationEvent], *, benchmarked: bool
) -> list[str]:
    """Reasons from the outcomes the summaries use: each horizon's non-repeat events."""
    reasons: list[str] = []
    if any(event.history_missing for event in events):
        reasons.append("missing_price_cache")
    windows = [
        (event, horizon)
        for event in events
        for horizon in event.exits
        if horizon not in event.suppressed
    ]
    if any(
        not event.history_missing and horizon not in event.outcomes for event, horizon in windows
    ):
        reasons.append("incomplete_price_history")
    if benchmarked and any(
        event.outcomes[horizon].excess_return_pct is None
        for event, horizon in windows
        if horizon in event.outcomes
    ):
        reasons.append("missing_benchmark_prices")
    return reasons


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
        in_window = [
            FeatureRun.status == "published",
            FeatureRun.as_of_date >= cutoff_date,
            RUN_MARKET == market,
        ]
        if until_date is not None:
            in_window.append(FeatureRun.as_of_date <= until_date)
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
            .filter(*in_window, SCAN_PICK_ELIGIBILITY)
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

        events: list[RawValidationEvent] = []
        for row in query.all():
            details = row.details_json or {}
            setup = details.get("setup_engine")
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
                        "action_state": details.get("action_state"),
                        "setup_pattern": (
                            setup.get("pattern_primary") if isinstance(setup, dict) else None
                        ),
                    },
                )
            )
        legacy_rows = exists().where(
            StockFeatureDaily.run_id == FeatureRun.id,
            StockFeatureDaily.hot_fields_stored.is_(None),
        )
        has_legacy_runs = db.query(exists().where(*in_window, legacy_rows)).scalar()
        return events, ["legacy_runs_without_typed_scores"] if has_legacy_runs else []


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
        return [event for event in events if markets[event.symbol] == market], []


def _rs_band(rating: int) -> str:
    low = rating // 10 * 10
    return f"{low}-{low + 9}"


class ReplayPickValidationSource:
    """Extract one validation event per stored replay pick of the current definition."""

    def collect(
        self,
        db: Session,
        *,
        cutoff_date: date,
        market: str,
        until_date: date | None = None,
        symbol: str | None = None,
    ) -> tuple[list[RawValidationEvent], list[str]]:
        query = db.query(BacktestReplayPick).filter(
            BacktestReplayPick.market == market,
            BacktestReplayPick.definition_version == DEFINITION_VERSION,
            BacktestReplayPick.as_of_date >= cutoff_date,
        )
        if until_date is not None:
            query = query.filter(BacktestReplayPick.as_of_date <= until_date)
        if symbol:
            query = query.filter(BacktestReplayPick.symbol == symbol)
        rows = query.order_by(
            BacktestReplayPick.as_of_date.desc(), BacktestReplayPick.rank.asc()
        ).all()
        return [
            RawValidationEvent(
                symbol=row.symbol,
                source_kind=ValidationSourceKind.REPLAY_PICK,
                source_ref=f"replay:{market}:{row.as_of_date.isoformat()}",
                event_at=row.as_of_date,
                attributes={
                    "symbol": row.symbol,
                    "rank": row.rank,
                    "minervini_score": _round_or_none(row.minervini_score),
                    "rs_rating": row.rs_rating,
                    "rs_band": _rs_band(row.rs_rating),
                    "stage": row.stage,
                },
            )
            for row in rows
        ], []


class PriceOutcomeCalculator:
    """Compute the fixed validation horizons from cached OHLCV data."""

    def __init__(self, price_cache: PriceCacheService) -> None:
        self._price_cache = price_cache

    def evaluate_many(
        self,
        events: list[RawValidationEvent],
        *,
        sessions: Sequence[date],
        benchmark_candidates: Sequence[str] = (),
    ) -> tuple[list[EvaluatedValidationEvent], str | None]:
        """Evaluate events over the market's ``sessions``, ignoring bars after the last one;
        also return the benchmark used."""
        through = max(sessions, default=None)
        histories = {
            symbol: _normalize_price_history(frame, through)
            for symbol, frame in self._price_cache.get_many_cached_only(
                sorted({event.symbol for event in events} | set(benchmark_candidates)),
                period=PRICE_CACHE_PERIOD,
            ).items()
        }
        calendar = pd.DatetimeIndex(sessions)
        cached_benchmarks = [
            (candidate, histories[candidate])
            for candidate in benchmark_candidates
            if histories.get(candidate) is not None
        ]
        # A primary whose feed stopped keeps its old rows; the freshest wins, ties in registry order.
        benchmark_symbol, benchmark = max(
            cached_benchmarks, key=lambda pair: pair[1].index[-1], default=(None, None)
        )
        evaluations = [
            self._evaluate_one(event, histories.get(event.symbol), benchmark, calendar)
            for event in events
        ]
        return evaluations, benchmark_symbol

    def _evaluate_one(
        self,
        event: RawValidationEvent,
        history: pd.DataFrame | None,
        benchmark: pd.DataFrame | None,
        calendar: pd.DatetimeIndex,
    ) -> EvaluatedValidationEvent:
        # Extension reads backward from the full calendar: a pick on a session this symbol's
        # refresh has not reached is missing a bar, which the trim below would hide.
        before_entry = calendar[: _entry_session_index(event.event_at, calendar)][-50:]
        if history is not None and len(calendar) > 1 and history.index[-1] == calendar[-2]:
            # The calendar counts a session completed before the after-close refresh reaches
            # this symbol's batch; its window past that session is pending, not missing history.
            calendar = calendar[:-1]
        start = _entry_session_index(event.event_at, calendar)
        sessions = calendar[start : start + max(HORIZONS)]
        exits = {
            horizon: sessions[horizon - 1].date() for horizon in HORIZONS if horizon <= len(sessions)
        }
        if history is None:
            return EvaluatedValidationEvent(
                raw=event, entry_at=None, entry_price=None, exits=exits, history_missing=True
            )
        evaluation = EvaluatedValidationEvent(
            raw=event,
            entry_at=None,
            entry_price=None,
            exits=exits,
            pct_above_50dma=_pct_above_50dma(history, before_entry),
        )
        if sessions.empty:
            return evaluation

        # Score only the sessions before the first one the cache lacks: past a gap, every
        # later bar would stand in for an earlier session.
        cached = np.logical_and.accumulate(sessions.isin(history.index))
        window = history.loc[sessions[cached]]
        entry_at = sessions[0].date()
        entry_price = float(window["Open"].iloc[0]) if cached[0] else math.nan
        if not entry_price > 0:
            return replace(evaluation, entry_at=entry_at)
        return replace(
            evaluation,
            entry_at=entry_at,
            entry_price=entry_price,
            outcomes=_horizon_outcomes(window, entry_price, benchmark),
        )


def _pct_above_50dma(history: pd.DataFrame, before_entry: pd.DatetimeIndex) -> float | None:
    """The last close of the 50 sessions ``before_entry`` against their mean close.

    Sessions from the entry on are left out: they were unknown when the position opened. A
    session the cache lacks leaves the extension unknown rather than averaging across the gap.
    """
    positions = history.index.get_indexer(before_entry)
    if len(positions) < 50 or (positions < 0).any():
        return None
    closes = history["Close"].to_numpy(dtype=float)[positions]
    if not (closes > 0).all():
        return None
    # Rounded so float noise cannot drop an exact band edge, such as 20%, into the band below.
    return round((closes[-1] / closes.mean() - 1.0) * 100.0, 6)


def _entry_session_index(event_at: date | datetime, calendar: pd.DatetimeIndex) -> int:
    """Position in ``calendar`` of the session whose open the event enters at: its own day
    for an alert before that session opens, else the next session."""
    if isinstance(event_at, datetime):
        event_dt = _normalize_event_datetime(event_at)
        side = "left" if event_dt.time() < MARKET_OPEN_TIME else "right"
        return int(calendar.searchsorted(pd.Timestamp(event_dt.date()), side=side))
    return int(calendar.searchsorted(pd.Timestamp(event_at), side="right"))


def _horizon_outcomes(
    window: pd.DataFrame, entry_price: float, benchmark: pd.DataFrame | None
) -> dict[int, HorizonOutcome]:
    """Outcomes for every horizon ``window`` covers, its first bar being the entry session."""

    def pct(price: float) -> float:
        return (price / entry_price - 1.0) * 100.0

    def prices(column: str) -> np.ndarray:
        # A zero or missing price is a bad tick: it neither triggers the stop nor sets an extreme.
        values = window[column].to_numpy(dtype=float)
        return np.where(values > 0, values, np.nan)

    opens, lows, closes = prices("Open"), prices("Low"), prices("Close")
    # Seeded with the entry price, so a bad tick on the entry bar still leaves an extreme.
    peaks = np.fmax.accumulate(np.fmax(prices("High"), entry_price))
    troughs = np.fmin.accumulate(np.fmin(lows, entry_price))
    stop_price = entry_price * (100.0 - STOP_LOSS_PCT) / 100.0
    # The entry session opens at the entry price, above the stop, so only its low can trigger.
    gapped = opens <= stop_price
    hit = gapped | (lows <= stop_price)
    stop_day = int(hit.argmax()) if hit.any() else len(window)
    gapped_through = stop_day < len(window) and gapped[stop_day]
    stop_return_pct = pct(opens[stop_day]) if gapped_through else -STOP_LOSS_PCT

    outcomes: dict[int, HorizonOutcome] = {}
    for horizon in HORIZONS:
        last = horizon - 1
        # A bad closing tick leaves the window unscored rather than a -100% return.
        if horizon > len(window) or np.isnan(closes[last]):
            continue
        return_pct = pct(closes[last])
        stopped_out = stop_day <= last
        outcomes[horizon] = HorizonOutcome(
            return_pct=return_pct,
            mfe_pct=pct(peaks[last]),
            mae_pct=pct(troughs[last]),
            stopped_return_pct=stop_return_pct if stopped_out else return_pct,
            stopped_out=stopped_out,
            excess_return_pct=_excess_return_pct(
                return_pct, benchmark, window.index[0], window.index[last]
            ),
        )
    return outcomes


def _group_values(event: EvaluatedValidationEvent) -> dict[str, Any]:
    """The event's value for every dimension: its attributes, and the band of its extension."""
    pct = event.pct_above_50dma
    band = None if pct is None else EXTENSION_BANDS[bisect_right(EXTENSION_BAND_EDGES, pct)]
    return {**event.raw.attributes, "extension_band": band}


def _group_order(dimension: str, value: str) -> tuple[int, float, str]:
    """A value's place in its dimension's ``GROUP_VALUE_ORDER``. Values that order lacks, and
    values of dimensions without one, come after it, numbers numerically so Stage 2 comes
    before Stage 10, then words alphabetically."""
    order = GROUP_VALUE_ORDER.get(dimension, ())
    if value in order:
        return (0, order.index(value), value)
    try:
        return (1, float(value), value)
    except ValueError:
        return (2, 0.0, value)


class ValidationService:
    """Compose signal sources, price outcomes, and deterministic summaries."""

    def __init__(
        self,
        *,
        scan_pick_source: ScanPickValidationSource | None = None,
        theme_alert_source: ThemeAlertValidationSource | None = None,
        replay_pick_source: ReplayPickValidationSource | None = None,
        outcome_calculator: PriceOutcomeCalculator | None = None,
        market_calendar: MarketCalendarService | None = None,
    ) -> None:
        from app.wiring.bootstrap import get_market_calendar_service, get_price_cache

        price_cache = get_price_cache()
        self._market_calendar = market_calendar or get_market_calendar_service()
        self._sources = {
            ValidationSourceKind.SCAN_PICK: scan_pick_source or ScanPickValidationSource(),
            ValidationSourceKind.THEME_ALERT: theme_alert_source or ThemeAlertValidationSource(),
            ValidationSourceKind.REPLAY_PICK: replay_pick_source or ReplayPickValidationSource(),
        }
        self._outcome_calculator = outcome_calculator or PriceOutcomeCalculator(price_cache)

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
            stop_loss_pct=STOP_LOSS_PCT,
            horizons=payload["horizons"],
            recent_events=payload["recent_events"],
            groups=self._build_groups(payload["events"], source_kind),
            freshness=self._build_freshness(
                db,
                as_of_date=effective_as_of_date,
                market=market,
                include_replay=source_kind == ValidationSourceKind.REPLAY_PICK,
            ),
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
                    degraded_reasons=payload["degraded_reasons"],
                )
            )
            merged_events.extend(payload["recent_events"])
            degraded_reasons.extend(payload["degraded_reasons"])

        merged_events.sort(key=lambda item: (item.event_at, item.source_ref), reverse=True)

        return StockValidationResponse(
            symbol=symbol,
            lookback_days=lookback_days,
            stop_loss_pct=STOP_LOSS_PCT,
            source_breakdown=source_breakdown,
            recent_events=merged_events[:RECENT_EVENTS_LIMIT],
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
        collect_from = (
            date.min
            if source_kind == ValidationSourceKind.REPLAY_PICK
            else cutoff_date - OVERLAP_LOOKBEHIND
        )
        raw_events, degraded_reasons = self._sources[source_kind].collect(
            db,
            cutoff_date=collect_from,
            market=market,
            until_date=effective_as_of_date,
            symbol=symbol,
        )
        calendar_reasons: list[str] = []
        try:
            through = self._market_calendar.last_completed_trading_day(market)
        except CalendarCoverageExpired as exc:
            through = exc.verified_through
            calendar_reasons.append("calendar_coverage_expired")
        benchmark_candidates = benchmark_registry.get_candidate_symbols(market)
        evaluated_events, benchmark_symbol = self._outcome_calculator.evaluate_many(
            raw_events,
            benchmark_candidates=benchmark_candidates,
            # Ending at ``through``, which never passes verified coverage, drops today's partial
            # bar while the session is open. A bar fetched mid-session stays partial until the
            # after-close refresh, which can run after this cut-off moves.
            # Starting no later than the cutoff keeps the session before ``through`` in the
            # calendar even when every event falls on ``through``. The earlier sessions serve
            # only extension, which reads the 50 before each entry, since windows look forward.
            sessions=self._market_calendar.trading_days(
                market,
                min([cutoff_date, *(_event_date(event.event_at) for event in raw_events)])
                - EXTENSION_LOOKBEHIND,
                through,
            ),
        )
        events = [
            event
            for event in mark_overlapping_windows(evaluated_events)
            if _event_date(event.raw.event_at) >= cutoff_date
        ]
        if not events and not degraded_reasons:
            degraded_reasons.append(NO_EVENTS_REASONS[source_kind])
        degraded_reasons.extend(calendar_reasons)
        degraded_reasons.extend(
            _outcome_degraded_reasons(events, benchmarked=bool(benchmark_candidates))
        )

        recent_events = [
            event.to_response()
            for event in sorted(
                events,
                key=lambda item: (_serialize_temporal(item.raw.event_at), item.raw.source_ref),
                reverse=True,
            )[:RECENT_EVENTS_LIMIT]
        ]

        return {
            "horizons": self._summarize_horizons(events),
            "recent_events": recent_events,
            "events": events,
            "degraded_reasons": _dedupe(degraded_reasons),
            "benchmark_symbol": benchmark_symbol,
        }

    def _build_groups(
        self, events: list[EvaluatedValidationEvent], source_kind: ValidationSourceKind
    ) -> list[ValidationGroup]:
        groups: list[ValidationGroup] = []
        values = [_group_values(event) for event in events]
        for dimension in GROUP_DIMENSIONS[source_kind]:
            members: dict[str, list[EvaluatedValidationEvent]] = defaultdict(list)
            for event, event_values in zip(events, values):
                value = event_values.get(dimension)
                if value is not None and value != "":
                    members[str(value)].append(event)
            groups.extend(
                ValidationGroup(
                    dimension=dimension,
                    label=GROUP_LABELS.get(dimension) or dimension.replace("_", " ").title(),
                    value=value,
                    horizons=self._summarize_horizons(members[value]),
                )
                for value in sorted(members, key=partial(_group_order, dimension))
            )
        return groups

    def _summarize_horizons(
        self, events: list[EvaluatedValidationEvent]
    ) -> list[ValidationHorizonSummary]:
        return [
            self._summarize_horizon(events, horizon_sessions=horizon_sessions)
            for horizon_sessions in HORIZONS
        ]

    def _summarize_horizon(
        self,
        events: list[EvaluatedValidationEvent],
        *,
        horizon_sessions: int,
    ) -> ValidationHorizonSummary:
        counted = [event for event in events if horizon_sessions not in event.suppressed]
        outcomes = [
            event.outcomes[horizon_sessions]
            for event in counted
            if horizon_sessions in event.outcomes
        ]
        returns = [outcome.return_pct for outcome in outcomes]
        stopped_returns = [outcome.stopped_return_pct for outcome in outcomes]
        excess_values = [
            outcome.excess_return_pct
            for outcome in outcomes
            if outcome.excess_return_pct is not None
        ]
        pending = sum(horizon_sessions not in event.exits for event in counted)
        return ValidationHorizonSummary(
            horizon_sessions=horizon_sessions,
            sample_size=len(outcomes),
            positive_rate=_mean([float(value > 0) for value in returns]),
            avg_return_pct=_mean(returns),
            median_return_pct=_median(returns),
            avg_mfe_pct=_mean([outcome.mfe_pct for outcome in outcomes]),
            avg_mae_pct=_mean([outcome.mae_pct for outcome in outcomes]),
            avg_excess_return_pct=_mean(excess_values),
            median_excess_return_pct=_median(excess_values),
            excess_sample_size=len(excess_values),
            avg_stopped_return_pct=_mean(stopped_returns),
            median_stopped_return_pct=_median(stopped_returns),
            stopped_out_rate=_mean([float(outcome.stopped_out) for outcome in outcomes]),
            skipped_missing_history=len(counted) - len(outcomes) - pending,
            pending_windows=pending,
            suppressed_overlapping=len(events) - len(counted),
        )

    def _build_freshness(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date | None = None,
        include_replay: bool = False,
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

        latest_replay_as_of_date = None
        if include_replay:
            replay_query = db.query(func.max(BacktestReplayPick.as_of_date)).filter(
                BacktestReplayPick.market == market,
                BacktestReplayPick.definition_version == DEFINITION_VERSION,
            )
            if as_of_date is not None:
                replay_query = replay_query.filter(BacktestReplayPick.as_of_date <= as_of_date)
            latest_replay_as_of_date = replay_query.scalar()
        return ValidationFreshness(
            latest_feature_as_of_date=(
                latest_feature_as_of_date.isoformat() if latest_feature_as_of_date else None
            ),
            latest_theme_alert_at=(
                _serialize_temporal(latest_theme_alert_at) if latest_theme_alert_at else None
            ),
            latest_replay_as_of_date=(
                latest_replay_as_of_date.isoformat() if latest_replay_as_of_date else None
            ),
            price_cache_period=PRICE_CACHE_PERIOD,
        )
