"""Replay Minervini scan picks point-in-time from stored prices and Market RS runs.

A replay pick for session D of a market: among the symbols in the market's
completed Market RS run for D, funds excluded, the Minervini screener scores
prices and benchmark cut at D with RS from that run. The top passing symbols by
Minervini score, ties by symbol, are D's picks. Prices come straight from
stock_prices: the price cache windows back from today and falls back to providers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from time import perf_counter

import pandas as pd
from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    balanced_run_has_current_snapshot_contract,
)
from app.domain.scanning.ports import MarketRsResolution, StockDataProvider
from app.infra.db.models.backtest_replay import DEFINITION_VERSION, BacktestReplayPick
from app.infra.db.models.relative_strength import StockRsSnapshot
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.scanners.base_screener import StockData
from app.scanners.scan_orchestrator import ScanOrchestrator
from app.scanners.screener_registry import screener_registry
from app.services.benchmark_registry_service import benchmark_registry
from app.services.market_rs_reader import SqlMarketRsReader
from app.services.price_cache_service import PERIOD_DAYS
from app.services.price_row_normalization import (
    STOCK_PRICE_ROW_COLUMNS,
    normalize_price_frame,
    stock_price_frames_by_symbol,
)
from app.services.validation_service import NOT_A_FUND

REPLAY_TOP_N = 10
SCREENERS = ["minervini"]
# A live scan reads Minervini's 2y price window back from today; a replay cuts it back from D.
PRICE_WINDOW = timedelta(days=PERIOD_DAYS["2y"])
PRICE_LOAD_CHUNK = 500


class ProviderFetchAttempted(BaseException):
    """The orchestrator asked for data the replay builds itself.

    A BaseException: scan_stock_multi turns every Exception into an error row,
    and a fetch has to stop the replay instead.
    """


class NoFetchStockDataProvider(StockDataProvider):
    def prepare_data(self, symbol, requirements, *, allow_partial=True):
        raise ProviderFetchAttempted(f"prepare_data({symbol})")

    def prepare_data_bulk(self, symbols, requirements, **_):
        raise ProviderFetchAttempted(f"prepare_data_bulk({', '.join(symbols)})")

    def apply_market_rs_resolution(self, results, resolution):
        raise ProviderFetchAttempted(f"apply_market_rs_resolution({', '.join(results)})")


@dataclass(frozen=True)
class ReplayPick:
    rank: int
    symbol: str
    minervini_score: float
    rs_rating: int
    stage: int | None


@dataclass(frozen=True)
class SessionReplay:
    as_of_date: date
    market_rs_run_id: int
    scored: int
    passed: int
    errors: int
    picks: tuple[ReplayPick, ...]
    seconds: float
    # A failed session writes nothing and reports no picks, so its stored picks stay.
    failure: str | None


def select_picks(results: Iterable[Mapping]) -> tuple[ReplayPick, ...]:
    """The top passing results by Minervini score, ties by symbol, ranked from 1."""
    passing = sorted(
        (result for result in results if result.get("minervini_passes") is True),
        key=lambda result: (-result["minervini_score"], result["symbol"]),
    )
    return tuple(
        ReplayPick(
            rank=rank,
            symbol=result["symbol"],
            minervini_score=float(result["minervini_score"]),
            rs_rating=int(result["rs_rating"]),
            stage=result.get("stage"),
        )
        for rank, result in enumerate(passing[:REPLAY_TOP_N], start=1)
    )


def minervini_orchestrator() -> ScanOrchestrator:
    # No Market RS reader: it would serve the latest run to a StockData without RS.
    return ScanOrchestrator(data_provider=NoFetchStockDataProvider(), registry=screener_registry)


def replay_market(
    session_factory: Callable[[], Session],
    market: str,
    *,
    start: date | None = None,
    end: date | None = None,
    write: bool = True,
    skip_existing: bool = False,
    orchestrator: ScanOrchestrator | None = None,
) -> Iterator[SessionReplay]:
    """Replay every session in [start, end] with a completed Market RS run, oldest first.

    Writing replaces one session's picks per transaction, so a rerun converges and an
    interrupted run resumes by redoing sessions or, with ``skip_existing``, skipping
    the ones that have picks. A session where any symbol errored writes nothing.
    """
    orchestrator = orchestrator or minervini_orchestrator()
    reader = SqlMarketRsReader(session_factory)
    db = session_factory()
    try:
        runs = _runs_to_replay(db, market, start=start, end=end, skip_existing=skip_existing)
        if not runs:
            return
        symbols = _eligible_symbols(db, [run_id for _, run_id in runs])
        benchmarks = benchmark_registry.get_candidate_symbols(market)
        frames = _load_frames(
            db, sorted({*symbols, *benchmarks}), start=runs[0][0] - PRICE_WINDOW, end=runs[-1][0]
        )
        # Scoring takes hours; don't hold the connection idle in a transaction meanwhile.
        db.rollback()
        for as_of_date, run_id in runs:
            started = perf_counter()
            benchmark_symbol = next(
                (
                    symbol
                    for symbol in benchmarks
                    if symbol in frames and pd.Timestamp(as_of_date) in frames[symbol].index
                ),
                None,
            )
            if benchmark_symbol is None:
                yield SessionReplay(
                    as_of_date=as_of_date,
                    market_rs_run_id=run_id,
                    scored=0,
                    passed=0,
                    errors=0,
                    picks=(),
                    seconds=perf_counter() - started,
                    failure=f"no benchmark bar ({', '.join(benchmarks)})",
                )
                continue
            resolution = reader.get(
                market=market, symbols=symbols, as_of_date=as_of_date, run_id=run_id
            )
            results = list(_score_session(orchestrator, resolution, frames, benchmark_symbol))
            # A screener's own error leaves the row's overall rating; only minervini_rating says so.
            errors = sum(
                "Error" in (result.get("rating"), result.get("minervini_rating"))
                for result in results
            )
            picks = () if errors else select_picks(results)
            if write and not errors:
                _replace_picks(db, resolution, picks)
            yield SessionReplay(
                as_of_date=as_of_date,
                market_rs_run_id=run_id,
                scored=len(results),
                passed=sum(result.get("minervini_passes") is True for result in results),
                errors=errors,
                picks=picks,
                seconds=perf_counter() - started,
                failure="scan errors" if errors else None,
            )
    finally:
        db.close()


def _runs_to_replay(
    db: Session, market: str, *, start: date | None, end: date | None, skip_existing: bool
) -> list[tuple[date, int]]:
    runs = MarketRsRunRepository().list_completed_runs(
        db,
        market=market,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        start_date=start or date.min,
        through_date=end or date.max,
    )
    replayed = (
        {
            as_of_date
            for (as_of_date,) in db.query(BacktestReplayPick.as_of_date)
            .filter(
                BacktestReplayPick.market == market.upper(),
                BacktestReplayPick.definition_version == DEFINITION_VERSION,
            )
            .distinct()
        }
        if skip_existing
        else set()
    )
    return sorted(
        (run.as_of_date, run.id)
        for run in runs
        if balanced_run_has_current_snapshot_contract(run) and run.as_of_date not in replayed
    )


def _eligible_symbols(db: Session, run_ids: list[int]) -> list[str]:
    """Every non-fund symbol rated in any of the runs."""
    return sorted(
        symbol
        for (symbol,) in db.query(StockRsSnapshot.symbol)
        .outerjoin(StockUniverse, StockUniverse.symbol == StockRsSnapshot.symbol)
        .filter(StockRsSnapshot.run_id.in_(run_ids), NOT_A_FUND)
        .distinct()
    )


def _load_frames(
    db: Session, symbols: list[str], *, start: date, end: date
) -> dict[str, pd.DataFrame]:
    """Each symbol's stored bars in [start, end], built as the price cache builds them."""
    frames: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(symbols), PRICE_LOAD_CHUNK):
        rows = (
            db.query(*(getattr(StockPrice, column) for column in STOCK_PRICE_ROW_COLUMNS))
            .filter(
                StockPrice.symbol.in_(symbols[offset : offset + PRICE_LOAD_CHUNK]),
                StockPrice.date >= start,
                StockPrice.date <= end,
            )
            .order_by(StockPrice.symbol, StockPrice.date)
            .all()
        )
        for symbol, frame in stock_price_frames_by_symbol(rows, include_adj_close=True).items():
            normalized = normalize_price_frame(frame)
            if normalized is not None:
                frames[symbol] = normalized
    return frames


def _score_session(
    orchestrator: ScanOrchestrator,
    resolution: MarketRsResolution,
    frames: Mapping[str, pd.DataFrame],
    benchmark_symbol: str,
) -> Iterator[dict]:
    session = pd.Timestamp(resolution.as_of_date)
    window = slice(session - PRICE_WINDOW, session)
    benchmark = frames[benchmark_symbol].loc[window]
    for symbol in sorted(resolution.ratings_by_symbol):
        if symbol not in frames:
            continue
        stock = StockData(
            symbol=symbol,
            price_data=frames[symbol].loc[window],
            benchmark_data=benchmark,
            benchmark_symbol=benchmark_symbol,
            market=resolution.market,
            rs_source=resolution.stock_source(symbol),
        )
        yield orchestrator.scan_stock_multi(
            symbol, SCREENERS, pre_fetched_data=stock, market_rs_resolution=resolution
        )


def _replace_picks(
    db: Session, resolution: MarketRsResolution, picks: tuple[ReplayPick, ...]
) -> None:
    db.query(BacktestReplayPick).filter(
        BacktestReplayPick.market == resolution.market,
        BacktestReplayPick.as_of_date == resolution.as_of_date,
        BacktestReplayPick.definition_version == DEFINITION_VERSION,
    ).delete(synchronize_session=False)
    db.add_all(
        BacktestReplayPick(
            market=resolution.market,
            as_of_date=resolution.as_of_date,
            rank=pick.rank,
            symbol=pick.symbol,
            minervini_score=pick.minervini_score,
            rs_rating=pick.rs_rating,
            stage=pick.stage,
            definition_version=DEFINITION_VERSION,
            market_rs_run_id=resolution.run_id,
        )
        for pick in picks
    )
    db.commit()
