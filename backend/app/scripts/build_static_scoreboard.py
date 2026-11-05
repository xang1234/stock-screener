"""Score the ledger's published matches against stored prices for one market artifact.

Runs as its own process after the export, so a crash or timeout cannot lose the
market. Anything short of success leaves manifest.market.json as it was, which
advertises nothing.

Mid-session: reads stock_prices only and never past the export's as_of_date, so it
cannot see a partial bar and cannot reach a provider.
"""

from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.stock import StockPrice
from app.schemas.validation import ValidationSourceKind
from app.scripts.scoreboard_ledger import (
    LedgerDay,
    Origin,
    format_timestamp,
    load_for_build,
)
from app.services.benchmark_registry_service import benchmark_registry
from app.services.market_calendar_service import MarketCalendarService
from app.services.price_row_normalization import (
    STOCK_PRICE_ROW_COLUMNS,
    normalize_price_frame,
    stock_price_frames_by_symbol,
)
from app.services.static_market_artifact_contract import (
    STATIC_MARKET_METADATA_FILENAME,
    STATIC_SCOREBOARD_SCHEMA_VERSION,
    read_static_market_manifest,
)
from app.services.validation_service import (
    CachedPriceSource,
    EvaluatedValidationEvent,
    PriceOutcomeCalculator,
    RawValidationEvent,
    mark_overlapping_windows,
    summarize_horizon,
)

SCOREBOARD_HORIZONS = (5, 20, 60)
THIN_SAMPLE_SIZE = 20
THIN_SIGNAL_DATES = 5
# ponytail: a year bounds time and memory and stays inside the 730 days of bundled
# prices. Ceiling: older picks leave the page, while their days stay in the ledger, so
# a longer history only needs this raised alongside the bundle.
SCORED_WINDOW_DAYS = 365
PRICE_QUERY_CHUNK = 250


class StockPriceTable:
    """``stock_prices`` between two dates, for the calculator's price source.

    No provider and no 50-bar minimum: PriceCacheService would report a recent IPO
    as missing history. The date bounds replace ``period``.
    """

    def __init__(self, db: Session, *, start: date, through: date) -> None:
        self._db = db
        self._start = start
        self._through = through

    def get_many_cached_only(self, symbols: list[str], period: str = "") -> dict[str, pd.DataFrame | None]:
        frames: dict[str, pd.DataFrame | None] = {}
        for offset in range(0, len(symbols), PRICE_QUERY_CHUNK):
            chunk = symbols[offset : offset + PRICE_QUERY_CHUNK]
            rows = (
                self._db.query(*(getattr(StockPrice, column) for column in STOCK_PRICE_ROW_COLUMNS))
                .filter(
                    StockPrice.symbol.in_(chunk),
                    StockPrice.date >= self._start,
                    StockPrice.date <= self._through,
                )
                .order_by(StockPrice.symbol, StockPrice.date)
                .all()
            )
            by_symbol = stock_price_frames_by_symbol(rows, include_adj_close=False)
            frames.update((symbol, normalize_price_frame(by_symbol.get(symbol))) for symbol in chunk)
        return frames


def entry_anchor(published_at: datetime, as_of: date, market: str, calendar: MarketCalendarService) -> date:
    """The date whose next session is the first to open strictly after ``published_at``.

    The calculator enters at the first session after this date, so it need not be a
    session itself. Never before ``as_of``: the matches rest on that day's close.
    """
    day = published_at.astimezone(calendar.market_timezone(market)).date()
    opened = calendar.session_open(market, day)
    return max(as_of, day if opened is not None and opened <= published_at else day - timedelta(days=1))


class CohortKey(NamedTuple):
    preset: str
    definition: str
    origin: Origin


@dataclass(frozen=True)
class CohortEvent:
    as_of_date: date
    anchor: date
    evaluated: EvaluatedValidationEvent


def _horizon_block(events: list[CohortEvent], horizon: int, sessions: list[date]) -> dict[str, Any]:
    summary = summarize_horizon([event.evaluated for event in events], horizon_sessions=horizon)
    counted = [event for event in events if horizon not in event.evaluated.suppressed]
    closed = [event for event in counted if horizon in event.evaluated.outcomes]
    signal_dates = len({event.as_of_date for event in closed})
    excess = [
        event.evaluated.outcomes[horizon].excess_return_pct
        for event in closed
        if event.evaluated.outcomes[horizon].excess_return_pct is not None
    ]
    # A symbol whose refresh lags one session stays pending with every session elapsed;
    # its window is still short of the last bar.
    open_sessions = [
        min(len(sessions) - bisect_right(sessions, event.anchor), horizon - 1)
        for event in counted
        if horizon not in event.evaluated.exits
    ]
    return {
        "horizon_sessions": horizon,
        "sample_size": summary.sample_size,
        "signal_dates": signal_dates,
        "pending_windows": summary.pending_windows,
        "oldest_open_sessions": max(open_sessions, default=None),
        "thin": summary.sample_size < THIN_SAMPLE_SIZE or signal_dates < THIN_SIGNAL_DATES,
        "positive_rate": summary.positive_rate,
        "beat_benchmark_rate": round(sum(value > 0 for value in excess) / len(excess), 4) if excess else None,
        "median_return_pct": summary.median_return_pct,
        "median_excess_return_pct": summary.median_excess_return_pct,
        "excess_sample_size": summary.excess_sample_size,
        "skipped_missing_history": summary.skipped_missing_history,
        "suppressed_overlapping": summary.suppressed_overlapping,
    }


def _scored_from(as_of_date: date) -> date:
    return as_of_date - timedelta(days=SCORED_WINDOW_DAYS)


def build_scoreboard(
    ledger: dict[date, LedgerDay],
    *,
    market: str,
    as_of_date: date,
    preset_screens: list[dict[str, Any]],
    prices: CachedPriceSource,
    calendar: MarketCalendarService,
    generated_at: datetime,
) -> dict[str, Any]:
    days = [ledger[day] for day in sorted(ledger)]
    first_scored = bisect_left([day.as_of_date for day in days], _scored_from(as_of_date))
    scored = days[first_scored:]
    anchors = {
        day.as_of_date: entry_anchor(day.published_at, day.as_of_date, market, calendar) for day in scored
    }
    sessions = calendar.trading_days(market, min(*anchors.values(), as_of_date), as_of_date)

    cohort_dates: dict[CohortKey, list[date]] = defaultdict(list)
    history: dict[str, list[tuple[CohortKey, date, date, str]]] = defaultdict(list)
    # The day just before the window shows what was already on each list when it opens.
    last_matches = (
        {preset: set(match.symbols) for preset, match in days[first_scored - 1].presets.items()}
        if first_scored
        else {}
    )
    for day in scored:
        anchor = anchors[day.as_of_date]
        for preset, match in day.presets.items():
            key = CohortKey(preset, match.definition, day.origin)
            cohort_dates[key].append(day.as_of_date)
            # A pick is a symbol entering the screen. On a preset's first day nothing
            # can be told apart from what was already on the list, so it yields none.
            if preset in last_matches:
                history[preset].extend(
                    (key, day.as_of_date, anchor, symbol)
                    for symbol in match.symbols
                    if symbol not in last_matches[preset]
                )
            last_matches[preset] = set(match.symbols)

    # A symbol's outcome depends only on its entry, so presets share one evaluation.
    unique = sorted({(symbol, anchor) for rows in history.values() for _, _, anchor, symbol in rows})
    evaluations, benchmark = PriceOutcomeCalculator(prices).evaluate_many(
        [
            RawValidationEvent(symbol, ValidationSourceKind.SCAN_PICK, "", anchor, {})
            for symbol, anchor in unique
        ],
        sessions=sessions,
        benchmark_candidates=benchmark_registry.get_candidate_symbols(market),
    )
    evaluated = dict(zip(unique, evaluations))

    current = {preset["id"]: preset for preset in preset_screens}
    recorded = {key.preset for key in cohort_dates}
    ordered = [preset for preset in current if preset in recorded] + sorted(recorded - set(current))

    cohorts = []
    for preset in ordered:
        rows = history[preset]
        # Marked over the preset's whole history, before segmenting, so a definition
        # or era change cannot reset a symbol's window.
        marked = mark_overlapping_windows([evaluated[(symbol, anchor)] for _, _, anchor, symbol in rows])
        events: dict[CohortKey, list[CohortEvent]] = defaultdict(list)
        for (key, as_of, anchor, _symbol), event in zip(rows, marked):
            events[key].append(CohortEvent(as_of, anchor, event))
        keys = sorted(
            (key for key in cohort_dates if key.preset == preset),
            key=lambda key: (list(Origin).index(key.origin), cohort_dates[key][0]),
        )
        shown = current.get(preset, {})
        for key in keys:
            cohorts.append(
                {
                    "preset": preset,
                    "name": shown.get("short_name") or shown.get("name") or preset,
                    "retired": preset not in current,
                    "definition": key.definition,
                    "origin": key.origin.value,
                    "first_date": cohort_dates[key][0].isoformat(),
                    "last_date": cohort_dates[key][-1].isoformat(),
                    "events": len(events[key]),
                    "signal_dates": len({event.as_of_date for event in events[key]}),
                    "horizons": [_horizon_block(events[key], horizon, sessions) for horizon in SCOREBOARD_HORIZONS],
                }
            )

    recording_since, ledger_through = days[0].as_of_date, days[-1].as_of_date
    recorded_sessions = calendar.trading_days(market, recording_since, ledger_through)
    return {
        "schema_version": STATIC_SCOREBOARD_SCHEMA_VERSION,
        "market": market,
        "as_of_date": as_of_date.isoformat(),
        "generated_at": format_timestamp(generated_at),
        "benchmark": benchmark,
        "recording_since": recording_since.isoformat(),
        "scored_since": scored[0].as_of_date.isoformat(),
        "ledger_through": ledger_through.isoformat(),
        "missing_sessions": sum(session not in ledger for session in recorded_sessions),
        "cohorts": cohorts,
    }


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_scoreboard(
    *,
    market: str,
    market_dir: Path,
    ledger_dir: Path,
    db: Session,
    calendar: MarketCalendarService,
    now: datetime,
) -> bool:
    """Write and advertise the asset; False when the market has no ledger to score."""
    ledger = load_for_build(ledger_dir, market)
    manifest_path = market_dir / STATIC_MARKET_METADATA_FILENAME
    manifest = read_static_market_manifest(manifest_path, expected_market=market)
    entry = manifest["entry"]
    as_of_date = date.fromisoformat(entry["as_of_date"])
    if not ledger or max(ledger) < _scored_from(as_of_date):
        print(f"::warning::No usable {market} ledger; the scoreboard is omitted.", flush=True)
        return False
    scan = json.loads((market_dir / "scan" / "manifest.json").read_text(encoding="utf-8"))

    payload = build_scoreboard(
        ledger,
        market=market,
        as_of_date=as_of_date,
        preset_screens=scan["preset_screens"],
        prices=StockPriceTable(db, start=_scored_from(as_of_date), through=as_of_date),
        calendar=calendar,
        generated_at=now,
    )
    target = market_dir / "scoreboard.json"
    _write_atomic(target, json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    entry["assets"]["scoreboard"] = {"path": f"markets/{market.lower()}/scoreboard.json"}
    _write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{market}: scored {len(ledger)} ledger days into {len(payload['cohorts'])} cohorts", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--market-dir", required=True, type=Path)
    parser.add_argument("--ledger-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    with SessionLocal() as db:
        write_scoreboard(
            market=args.market,
            market_dir=args.market_dir,
            ledger_dir=args.ledger_dir,
            db=db,
            calendar=MarketCalendarService(),
            now=datetime.now(timezone.utc),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
