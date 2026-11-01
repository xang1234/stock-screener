from __future__ import annotations

import re
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, SessionLocal
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    BALANCED_RS_PRICE_BASIS,
    BALANCED_RS_SNAPSHOT_SCHEMA_VERSION,
)
from app.infra.db.models.backtest_replay import DEFINITION_VERSION, BacktestReplayPick
from app.infra.db.models.relative_strength import (
    MarketRsRun,
    StockRsPlacementSnapshot,
    StockRsSnapshot,
)
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.scanners.base_screener import DataRequirements, StockData
from app.scripts import replay_scan_picks
from app.services.backtest_replay_service import (
    ProviderFetchAttempted,
    ReplayPick,
    minervini_orchestrator,
    replay_market,
    select_picks,
)

TABLES = [
    MarketRsRun.__table__,
    StockRsSnapshot.__table__,
    StockRsPlacementSnapshot.__table__,
    StockUniverse.__table__,
    StockPrice.__table__,
    BacktestReplayPick.__table__,
]
SESSIONS = pd.bdate_range("2025-01-01", periods=320)
D1 = SESSIONS[300].date()
D2 = SESSIONS[305].date()


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine, tables=TABLES)
    yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.drop_all(engine, tables=TABLES)


def _add_run(db, as_of: date, ratings: dict[str, int], *, market="US") -> int:
    run = MarketRsRun(
        market=market,
        as_of_date=as_of,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        status="completed",
        benchmark_symbol="SPY",
        benchmark_as_of_date=as_of,
        universe_hash="a" * 64,
        expected_symbol_count=len(ratings),
        eligible_symbol_count=len(ratings),
        excluded_symbol_count=0,
        diagnostics_json={
            "price_basis": BALANCED_RS_PRICE_BASIS,
            "rs_snapshot_schema_version": BALANCED_RS_SNAPSHOT_SCHEMA_VERSION,
        },
        completed_at=datetime.now(UTC),
        rows=[
            StockRsSnapshot(
                symbol=symbol,
                overall_rs=rating,
                **{f"rs_{h}": rating for h in ("1d", "1w", "1m", "3m", "6m", "9m", "12m")},
                weighted_composite=float(rating),
                **{
                    f"excess_return_{h}": 0.1
                    for h in ("1d", "1w", "1m", "3m", "6m", "9m", "12m")
                },
            )
            for symbol, rating in ratings.items()
        ],
    )
    db.add(run)
    db.commit()
    return run.id


def _add_prices(db, symbol: str, closes, sessions=SESSIONS) -> None:
    db.add_all(
        StockPrice(
            symbol=symbol,
            date=day.date(),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            adj_close=close,
            volume=1_000_000,
        )
        for day, close in zip(sessions, closes)
    )
    db.commit()


def _uptrend(daily_growth: float) -> list[float]:
    return [20.0 * (1 + daily_growth) ** index for index in range(len(SESSIONS))]


def _passing(symbol: str, score: float, *, rs: int = 90) -> dict:
    return {"symbol": symbol, "minervini_passes": True, "minervini_score": score, "rs_rating": rs, "stage": 2}


class _ScoringStub:
    """Scores from a table; records what each call was handed."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls = []

    def scan_stock_multi(self, symbol, screener_names, *, pre_fetched_data, market_rs_resolution, **_):
        self.calls.append(
            (
                symbol,
                pre_fetched_data.price_data.index[-1].date(),
                pre_fetched_data.benchmark_data.index[-1].date(),
                pre_fetched_data.rs_source.run_id,
                market_rs_resolution.run_id,
            )
        )
        if symbol not in self.scores:
            return {"symbol": symbol, "rating": "Pass", "minervini_passes": False, "minervini_score": 99.0}
        return _passing(symbol, self.scores[symbol], rs=pre_fetched_data.rs_source.ratings["rs_rating"])


def _stored_picks(session_factory, as_of: date) -> list[tuple]:
    db = session_factory()
    try:
        return [
            (row.rank, row.symbol, row.minervini_score, row.rs_rating, row.market_rs_run_id)
            for row in db.query(BacktestReplayPick)
            .filter_by(market="US", as_of_date=as_of, definition_version=DEFINITION_VERSION)
            .order_by(BacktestReplayPick.rank)
        ]
    finally:
        db.close()


def test_select_picks_takes_the_top_ten_passing_symbols_by_score_ties_by_symbol():
    results = [
        _passing("KKK", 90.0),
        _passing("BBB", 95.0),
        _passing("AAA", 95.0),
        *(_passing(symbol, 80.0 - index) for index, symbol in enumerate("CDEFGHIJ")),
        {"symbol": "FAIL", "minervini_passes": False, "minervini_score": 99.0, "rs_rating": 95, "stage": 2},
        {"symbol": "NORS", "rating": "Insufficient Data", "rs_rating": None},
    ]

    assert select_picks(results) == (
        ReplayPick(rank=1, symbol="AAA", minervini_score=95.0, rs_rating=90, stage=2),
        ReplayPick(rank=2, symbol="BBB", minervini_score=95.0, rs_rating=90, stage=2),
        ReplayPick(rank=3, symbol="KKK", minervini_score=90.0, rs_rating=90, stage=2),
        *(
            ReplayPick(rank=4 + index, symbol=symbol, minervini_score=80.0 - index, rs_rating=90, stage=2)
            for index, symbol in enumerate("CDEFGHI")
        ),
    )


def test_each_session_scores_its_own_runs_non_fund_symbols_on_frames_cut_at_that_session(session_factory):
    db = session_factory()
    db.add_all(
        [
            StockUniverse(symbol="AAA", market="US", industry="Semiconductors", is_common_stock=True),
            StockUniverse(symbol="ETFX", market="US", industry="Exchange Traded Fund", is_common_stock=True),
            StockUniverse(symbol="PREF", market="US", industry="Banks", is_common_stock=False),
        ]
    )
    db.commit()
    d1_run = _add_run(db, D1, {"AAA": 90, "ETFX": 95, "PREF": 95, "GONE": 80})
    d2_run = _add_run(db, D2, {"AAA": 70, "NEWB": 85})
    for symbol in ("AAA", "ETFX", "PREF", "GONE", "NEWB", "SPY"):
        _add_prices(db, symbol, _uptrend(0.002))
    db.close()
    stub = _ScoringStub({"AAA": 80.0, "GONE": 75.0, "NEWB": 70.0})

    replays = list(replay_market(session_factory, "US", orchestrator=stub))

    assert sorted(stub.calls) == [
        ("AAA", D1, D1, d1_run, d1_run),
        ("AAA", D2, D2, d2_run, d2_run),
        ("GONE", D1, D1, d1_run, d1_run),
        ("NEWB", D2, D2, d2_run, d2_run),
    ]
    assert [(r.as_of_date, r.market_rs_run_id, r.scored, r.passed) for r in replays] == [
        (D1, d1_run, 2, 2),
        (D2, d2_run, 2, 2),
    ]
    assert _stored_picks(session_factory, D2) == [
        (1, "AAA", 80.0, 70, d2_run),
        (2, "NEWB", 70.0, 85, d2_run),
    ]


def test_a_rerun_replaces_a_sessions_picks_and_leaves_exactly_ten(session_factory):
    db = session_factory()
    symbols = [f"S{index:02d}" for index in range(12)]
    run_id = _add_run(db, D1, dict.fromkeys(symbols, 90))
    for symbol in [*symbols, "SPY"]:
        _add_prices(db, symbol, _uptrend(0.002))
    db.close()

    list(replay_market(session_factory, "US", orchestrator=_ScoringStub({s: 50.0 + i for i, s in enumerate(symbols)})))
    list(replay_market(session_factory, "US", orchestrator=_ScoringStub({s: 50.0 - i for i, s in enumerate(symbols)})))

    assert _stored_picks(session_factory, D1) == [
        (rank, f"S{rank - 1:02d}", 51.0 - rank, 90, run_id) for rank in range(1, 11)
    ]


def test_a_dry_run_reports_picks_and_writes_nothing(session_factory):
    db = session_factory()
    run_id = _add_run(db, D1, {"AAA": 90})
    _add_prices(db, "AAA", _uptrend(0.002))
    _add_prices(db, "SPY", _uptrend(0.001))
    db.close()

    replays = list(replay_market(session_factory, "US", write=False, orchestrator=_ScoringStub({"AAA": 80.0})))

    assert [(r.as_of_date, r.market_rs_run_id, r.picks) for r in replays] == [
        (D1, run_id, (ReplayPick(rank=1, symbol="AAA", minervini_score=80.0, rs_rating=90, stage=2),)),
    ]
    assert _stored_picks(session_factory, D1) == []


def test_skip_existing_replays_only_sessions_without_picks_and_start_end_bound_the_window(session_factory):
    db = session_factory()
    for as_of in (SESSIONS[290].date(), D1, D2):
        _add_run(db, as_of, {"AAA": 90})
    _add_prices(db, "AAA", _uptrend(0.002))
    _add_prices(db, "SPY", _uptrend(0.001))
    db.close()
    list(replay_market(session_factory, "US", start=D1, end=D1, orchestrator=_ScoringStub({"AAA": 80.0})))

    replays = list(
        replay_market(
            session_factory, "US", start=D1, skip_existing=True, orchestrator=_ScoringStub({"AAA": 80.0})
        )
    )

    assert [r.as_of_date for r in replays] == [D2]


def test_the_minervini_replay_sees_no_bar_after_its_session_and_never_fetches(session_factory):
    db = session_factory()
    crash = _uptrend(0.004)
    crash[303:] = [close * 0.4 for close in crash[303:]]
    _add_run(db, D1, {"AAA": 95, "BBB": 85, "CCC": 75})
    _add_run(db, D2, {"AAA": 95, "BBB": 85, "CCC": 75})
    _add_prices(db, "AAA", crash)
    _add_prices(db, "BBB", _uptrend(0.003))
    _add_prices(db, "CCC", list(reversed(_uptrend(0.003))))
    _add_prices(db, "SPY", _uptrend(0.001))
    db.close()

    replays = list(replay_market(session_factory, "US"))

    assert [(r.as_of_date, [p.symbol for p in r.picks], r.errors) for r in replays] == [
        (D1, ["AAA", "BBB"], 0),
        (D2, ["BBB"], 0),
    ]


def test_a_session_takes_the_first_benchmark_candidate_with_a_bar_on_that_session(session_factory):
    db = session_factory()
    run_id = _add_run(db, D1, {"AAA": 90})
    _add_prices(db, "AAA", _uptrend(0.002))
    _add_prices(db, "SPY", _uptrend(0.001)[:200])
    _add_prices(db, "IVV", _uptrend(0.001))
    db.close()
    stub = _ScoringStub({"AAA": 80.0})

    list(replay_market(session_factory, "US", orchestrator=stub))

    assert stub.calls == [("AAA", D1, D1, run_id, run_id)]


@pytest.mark.parametrize("benchmark_bars", [200, 0], ids=["stale", "missing"])
def test_a_session_without_a_benchmark_bar_fails_before_scoring_and_keeps_its_picks(
    session_factory, benchmark_bars
):
    db = session_factory()
    run_id = _add_run(db, D1, {"AAA": 90})
    _add_prices(db, "AAA", _uptrend(0.002))
    _add_prices(db, "SPY", _uptrend(0.001)[:benchmark_bars])
    db.add(
        BacktestReplayPick(
            market="US",
            as_of_date=D1,
            rank=1,
            symbol="OLD",
            minervini_score=70.0,
            rs_rating=80,
            stage=2,
            definition_version=DEFINITION_VERSION,
            market_rs_run_id=run_id,
        )
    )
    db.commit()
    db.close()
    stub = _ScoringStub({"AAA": 80.0})

    replays = list(replay_market(session_factory, "US", orchestrator=stub))

    assert (stub.calls, [(r.as_of_date, r.scored, r.picks, r.failure) for r in replays]) == (
        [],
        [(D1, 0, (), "no benchmark bar (SPY, IVV)")],
    )
    assert _stored_picks(session_factory, D1) == [(1, "OLD", 70.0, 80, run_id)]


def test_a_fetch_attempt_escapes_the_orchestrator_instead_of_becoming_an_error_row():
    with pytest.raises(ProviderFetchAttempted, match="AAA"):
        minervini_orchestrator().scan_stock_multi(
            "AAA", ["minervini"], pre_merged_requirements=DataRequirements()
        )


def _seed_two_leaders() -> int:
    db = SessionLocal()
    try:
        run_id = _add_run(db, D1, {"AAA": 95, "BBB": 85})
        _add_prices(db, "AAA", _uptrend(0.004))
        _add_prices(db, "BBB", _uptrend(0.003))
        _add_prices(db, "SPY", _uptrend(0.001))
        return run_id
    finally:
        db.close()


def _stored_symbols() -> list[str]:
    db = SessionLocal()
    try:
        return [row.symbol for row in db.query(BacktestReplayPick).order_by(BacktestReplayPick.rank)]
    finally:
        db.close()


def test_the_cli_dry_run_prints_each_sessions_picks_and_writes_nothing(capsys):
    run_id = _seed_two_leaders()

    exit_code = replay_scan_picks.main(["--market", "us", "--dry-run"])

    lines = capsys.readouterr().out.splitlines()
    assert (exit_code, lines[0], lines[2]) == (
        0,
        "Replaying US: dry run, nothing written",
        "US: 1 sessions, 2 picks, 0 failed sessions",
    )
    assert re.fullmatch(
        rf"{D1} run={run_id} scored=2 passed=2 errors=0 \d+\.\d\ds AAA:\d+\.\d BBB:\d+\.\d", lines[1]
    )
    assert _stored_symbols() == []


def test_the_cli_writes_picks_without_dry_run():
    _seed_two_leaders()

    assert replay_scan_picks.main(["--market", "US", "--start", D1.isoformat()]) == 0
    assert _stored_symbols() == ["AAA", "BBB"]


def _fail_minervini_for(monkeypatch, *symbols: str) -> None:
    """Raise inside MinerviniScanner.scan_stock's try block, which turns it into an "Error" sub-result."""
    has_sufficient_data = StockData.has_sufficient_data

    def failing(self, min_days=100):
        if self.symbol in symbols:
            raise RuntimeError(f"scoring bug on {self.symbol}")
        return has_sufficient_data(self, min_days)

    monkeypatch.setattr(StockData, "has_sufficient_data", failing)


def test_a_symbol_too_short_to_rate_is_skipped_without_failing_the_session(session_factory):
    db = session_factory()
    run_id = _add_run(db, D1, {"AAA": 95, "SHORT": 90})
    _add_prices(db, "AAA", _uptrend(0.004))
    _add_prices(db, "SHORT", _uptrend(0.004)[201:301], sessions=SESSIONS[201:301])
    _add_prices(db, "SPY", _uptrend(0.001))
    db.close()

    replays = list(replay_market(session_factory, "US"))

    assert [(r.scored, r.passed, r.errors, r.failure) for r in replays] == [(2, 1, 0, None)]
    assert [row[:2] + row[3:] for row in _stored_picks(session_factory, D1)] == [(1, "AAA", 95, run_id)]


def test_a_session_where_every_symbol_errors_keeps_its_stored_picks(session_factory, monkeypatch):
    db = session_factory()
    _add_run(db, D1, {"AAA": 95, "BBB": 85})
    _add_prices(db, "AAA", _uptrend(0.004))
    _add_prices(db, "BBB", _uptrend(0.003))
    _add_prices(db, "SPY", _uptrend(0.001))
    db.close()
    list(replay_market(session_factory, "US"))
    stored = _stored_picks(session_factory, D1)
    _fail_minervini_for(monkeypatch, "AAA", "BBB")

    replays = list(replay_market(session_factory, "US"))

    assert [(r.scored, r.errors, r.picks, r.failure) for r in replays] == [(2, 2, (), "scan errors")]
    assert ([row[1] for row in stored], _stored_picks(session_factory, D1)) == (["AAA", "BBB"], stored)


def test_the_cli_keeps_a_sessions_picks_and_exits_1_when_one_symbol_errors(monkeypatch, capsys):
    run_id = _seed_two_leaders()
    assert replay_scan_picks.main(["--market", "US"]) == 0
    _fail_minervini_for(monkeypatch, "AAA")

    exit_code = replay_scan_picks.main(["--market", "US"])

    lines = capsys.readouterr().out.splitlines()
    assert (exit_code, _stored_symbols(), lines[-1]) == (
        1,
        ["AAA", "BBB"],
        "US: 1 sessions, 0 picks, 1 failed sessions",
    )
    assert re.fullmatch(
        rf"{D1} run={run_id} scored=2 passed=1 errors=1 \d+\.\d\ds FAILED: scan errors", lines[-2]
    )
