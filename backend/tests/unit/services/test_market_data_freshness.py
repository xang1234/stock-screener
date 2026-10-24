"""Unit tests for the symbol-scoped freshness check service.

Uses an in-memory session stub; no real DB required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _fresh_benchmarks(monkeypatch):
    """Every market's benchmark is current unless a test says otherwise."""
    from app.services import market_data_freshness

    monkeypatch.setattr(
        market_data_freshness,
        "_latest_benchmark_bar",
        lambda market: (("BENCH",), date.max),
    )


@dataclass
class _Row:
    symbol: str
    market: str
    last_date: date | None


def _stale_tail_policy():
    from app.services.market_data_freshness import ScanFreshnessPolicy

    return ScanFreshnessPolicy.allowing_stale_tail()


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return self

    def outerjoin(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def group_by(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows

    def close(self):
        pass


def _patch_session(rows):
    return patch(
        "app.services.market_data_freshness.SessionLocal",
        return_value=_FakeSession(rows),
    )


def _rows(symbols, *, market="US", last_date=date(2026, 6, 18)):
    return [_Row(symbol=s, market=market, last_date=last_date) for s in symbols]


def _patch_calendar(last_completed_by_market):
    def _last_completed(market, now=None):
        return last_completed_by_market[market]

    fake_calendar = SimpleNamespace(last_completed_trading_day=_last_completed)
    return patch(
        "app.services.market_data_freshness.get_market_calendar_service",
        return_value=fake_calendar,
    )


def _patch_refresh_state(last_refreshed_by_market):
    def _state(_session, market):
        value = last_refreshed_by_market.get(market)
        if value is None:
            return None
        return {
            "market": market,
            "status": "completed",
            "last_refreshed_trading_day": value.isoformat(),
        }

    return patch(
        "app.services.market_data_freshness.get_market_refresh_state",
        side_effect=_state,
    )


def _patch_refresh_state_payload(state_by_market):
    def _state(_session, market):
        return state_by_market.get(market)

    return patch(
        "app.services.market_data_freshness.get_market_refresh_state",
        side_effect=_state,
    )


def test_fresh_universe_returns_none():
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
        _Row(symbol="MSFT", market="US", last_date=date(2026, 4, 23)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 4, 23)}),
        _patch_refresh_state({"US": date(2026, 4, 23)}),
    ):
        assert check_symbol_freshness(["AAPL", "MSFT"]) is None


def test_missing_refresh_state_falls_back_to_symbol_dates():
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
        _Row(symbol="MSFT", market="US", last_date=date(2026, 4, 23)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 4, 23)}),
        _patch_refresh_state({}),
    ):
        assert check_symbol_freshness(["AAPL", "MSFT"]) is None


def test_stale_market_returns_detail():
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 22)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 4, 23)}),
        _patch_refresh_state({"US": date(2026, 4, 23)}),
    ):
        detail = check_symbol_freshness(["AAPL"])

    assert detail is not None
    assert detail["code"] == "market_data_stale"
    assert detail["stale_markets"][0]["market"] == "US"
    assert detail["stale_markets"][0]["oldest_last_cached_date"] == "2026-04-22"


def test_unresolved_symbols_flagged_even_if_covered_symbols_are_fresh():
    """Round 4 Codex P2: symbols requested by the scan but missing from
    stock_universe must be treated as stale — the freshness query silently
    drops them, so we need to compare requested-vs-resolved explicitly.
    """
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 4, 23)}),
        _patch_refresh_state({"US": date(2026, 4, 23)}),
    ):
        detail = check_symbol_freshness(["AAPL", "UNKNOWN_FAKE", "MISSING"])

    assert detail is not None
    assert detail["code"] == "market_data_stale"
    assert detail["unresolved_symbols"] == ["MISSING", "UNKNOWN_FAKE"]
    assert "unknown symbols" in detail["message"]


def test_uncovered_symbols_in_known_market_flag_stale():
    """Symbol exists in stock_universe but has no stock_prices rows at all."""
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
        _Row(symbol="NEWIPO", market="US", last_date=None),  # no cached prices
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 4, 23)}),
        _patch_refresh_state({"US": date(2026, 4, 23)}),
    ):
        detail = check_symbol_freshness(["AAPL", "NEWIPO"])

    assert detail is not None
    us = detail["stale_markets"][0]
    assert us["market"] == "US"
    assert us["uncovered_symbols"] == 1
    assert us["covered_symbols"] == 1


def test_empty_symbol_list_returns_none():
    from app.services.market_data_freshness import check_symbol_freshness

    assert check_symbol_freshness([]) is None
    assert check_symbol_freshness(["", None]) is None


def test_calendar_failure_fails_closed():
    """Round 6 CodeRabbit + Codex P1 (same finding): if
    last_completed_trading_day raises (unsupported market code, calendar
    backend outage, etc.) the gate must fail-closed — treat the market as
    stale rather than silently letting the scan through.
    """
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [
        _Row(symbol="AAPL", market="US", last_date=date(2026, 4, 23)),
    ]

    def _raising_calendar(market, now=None):
        raise RuntimeError("calendar backend is down")

    with _patch_session(rows), patch(
        "app.services.market_data_freshness.get_market_calendar_service",
        return_value=SimpleNamespace(last_completed_trading_day=_raising_calendar),
    ):
        detail = check_symbol_freshness(["AAPL"])

    assert detail is not None
    assert detail["code"] == "market_data_stale"
    assert detail["stale_markets"][0]["market"] == "US"
    assert detail["stale_markets"][0]["reason"] == "calendar_unavailable"
    assert detail["stale_markets"][0]["expected_date"] is None
    assert "calendar unavailable" in detail["message"]


def test_degraded_broad_scan_omits_small_tail_at_99_percent():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    fresh_symbols = [f"FRESH{i:03d}" for i in range(99)]
    symbols = [*fresh_symbols, "STALE001"]
    rows = [
        *_rows(fresh_symbols),
        *_rows(["STALE001"], last_date=date(2026, 5, 13)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        decision = evaluate_symbol_freshness(symbols, policy=_stale_tail_policy())

    assert decision.blocking_detail is None
    assert decision.symbols_to_scan == tuple(fresh_symbols)
    warning = decision.warnings[0].to_dict()
    assert warning["code"] == "market_data_stale_tail_omitted"
    assert warning["omitted_count"] == 1
    assert warning["total_symbols"] == 100
    assert warning["freshness_rate"] == 0.99
    assert warning["omitted_symbols"] == ["STALE001"]


def test_degraded_multi_market_warning_names_only_markets_with_omitted_symbols():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    fresh_us_symbols = [f"FRESH{i:03d}" for i in range(99)]
    symbols = [*fresh_us_symbols, "STALE001", "0700.HK"]
    rows = [
        *_rows(fresh_us_symbols),
        *_rows(["STALE001"], last_date=date(2026, 5, 13)),
        *_rows(["0700.HK"], market="HK", last_date=date(2026, 6, 17)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18), "HK": date(2026, 6, 17)}),
        _patch_refresh_state({"US": date(2026, 6, 18), "HK": date(2026, 6, 17)}),
    ):
        decision = evaluate_symbol_freshness(symbols, policy=_stale_tail_policy())

    warning = decision.warnings[0].to_dict()
    assert warning["markets"] == ["US"]
    assert warning["expected_dates"] == {"US": "2026-06-18"}
    assert warning["oldest_last_cached_dates"] == {"US": "2026-05-13"}


def test_degraded_policy_is_explicit_value_object():
    from app.services.market_data_freshness import (
        ScanFreshnessPolicy,
        evaluate_symbol_freshness,
    )

    fresh_symbols = [f"FRESH{i:03d}" for i in range(99)]
    symbols = [*fresh_symbols, "STALE001"]
    rows = [
        *_rows(fresh_symbols),
        *_rows(["STALE001"], last_date=date(2026, 5, 13)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        decision = evaluate_symbol_freshness(
            symbols,
            policy=ScanFreshnessPolicy.allowing_stale_tail(),
        )

    assert decision.blocking_detail is None
    assert decision.symbols_to_scan == tuple(fresh_symbols)
    assert decision.warnings[0].omitted_symbols == ("STALE001",)


def test_evaluate_symbol_freshness_exposes_policy_not_boolean_mode():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    signature = inspect.signature(evaluate_symbol_freshness)

    assert "policy" in signature.parameters
    assert "allow_stale_tail" not in signature.parameters


def test_degraded_broad_scan_caps_omissions_at_100():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    fresh_symbols = [f"FRESH{i:05d}" for i in range(10_000)]
    stale_symbols = [f"STALE{i:03d}" for i in range(1, 102)]
    symbols = [*fresh_symbols, *stale_symbols]
    rows = [
        *_rows(fresh_symbols),
        *_rows(stale_symbols, last_date=date(2026, 5, 13)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        decision = evaluate_symbol_freshness(symbols, policy=_stale_tail_policy())

    assert decision.blocking_detail is not None
    assert decision.blocking_detail["code"] == "market_data_stale"
    us = decision.blocking_detail["stale_markets"][0]
    assert us["stale_symbol_count"] == 101
    assert us["sample_stale_symbols"][:3] == [
        "STALE001",
        "STALE002",
        "STALE003",
    ]


def test_degraded_broad_scan_requires_99_percent_freshness():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    fresh_symbols = [f"FRESH{i:03d}" for i in range(98)]
    symbols = [*fresh_symbols, "STALE001"]
    rows = [
        *_rows(fresh_symbols),
        *_rows(["STALE001"], last_date=date(2026, 5, 13)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        decision = evaluate_symbol_freshness(symbols, policy=_stale_tail_policy())

    assert decision.blocking_detail is not None
    assert decision.blocking_detail["code"] == "market_data_stale"
    assert decision.warnings == ()


def test_degraded_decision_preserves_resolver_order_after_omission():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    tail = [f"FRESH{i:03d}" for i in range(97)]
    symbols = ["MSFT", "STALE001", "AAPL", *tail]
    rows = [
        *_rows(["MSFT", "AAPL", *tail]),
        *_rows(["STALE001"], last_date=date(2026, 5, 13)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        decision = evaluate_symbol_freshness(symbols, policy=_stale_tail_policy())

    assert decision.blocking_detail is None
    assert decision.symbols_to_scan[:3] == ("MSFT", "AAPL", "FRESH000")


def test_strict_checker_still_blocks_stale_tail():
    from app.services.market_data_freshness import check_symbol_freshness

    fresh_symbols = [f"FRESH{i:03d}" for i in range(99)]
    symbols = [*fresh_symbols, "STALE001"]
    rows = [
        *_rows(fresh_symbols),
        *_rows(["STALE001"], last_date=date(2026, 5, 13)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        detail = check_symbol_freshness(symbols)

    assert detail is not None
    assert detail["code"] == "market_data_stale"


def test_degraded_policy_requires_completed_refresh_state():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    fresh_symbols = [f"FRESH{i:03d}" for i in range(99)]
    symbols = [*fresh_symbols, "STALE001"]
    rows = [
        *_rows(fresh_symbols),
        *_rows(["STALE001"], last_date=date(2026, 5, 13)),
    ]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state_payload({
            "US": {
                "market": "US",
                "status": "running",
                "last_refreshed_trading_day": "2026-06-18",
            }
        }),
    ):
        decision = evaluate_symbol_freshness(symbols, policy=_stale_tail_policy())

    assert decision.blocking_detail is not None
    assert decision.blocking_detail["code"] == "market_data_stale"
    assert decision.blocking_detail["stale_markets"][0]["reason"] == "refresh_state_missing"
    assert decision.warnings == ()


def _patch_benchmark(latest_by_market):
    """Benchmark candidates and newest cached bar per market (None = no rows)."""
    return patch(
        "app.services.market_data_freshness._latest_benchmark_bar",
        side_effect=lambda market: latest_by_market[market],
    )


@pytest.mark.parametrize("stale_tail", [False, True])
def test_stale_benchmark_blocks_a_market_whose_stocks_are_fresh(stale_tail):
    """#455: the scan's benchmark lookup fetches from the provider when no cached
    benchmark reaches the last completed session, so the gate must refuse."""
    from app.services.market_data_freshness import (
        ScanFreshnessPolicy,
        evaluate_symbol_freshness,
    )

    symbols = [f"FRESH{i:03d}" for i in range(100)]
    with (
        _patch_session(_rows(symbols)),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
        _patch_benchmark({"US": (("SPY",), date(2026, 6, 17))}),
    ):
        decision = evaluate_symbol_freshness(
            symbols,
            policy=(
                ScanFreshnessPolicy.allowing_stale_tail()
                if stale_tail
                else ScanFreshnessPolicy.strict()
            ),
        )

    detail = decision.blocking_detail
    assert detail is not None and decision.warnings == ()
    us = detail["stale_markets"][0]
    assert us["reason"] == "benchmark_stale"
    assert us["benchmark"] == {
        "symbols": ["SPY"],
        "last_cached_date": "2026-06-17",
        "expected_date": "2026-06-18",
    }
    assert "benchmark SPY last: 2026-06-17, expected: 2026-06-18" in detail["message"]
    assert "oldest stock: 2026-06-18" in detail["message"]


def test_missing_benchmark_rows_block_the_market():
    from app.services.market_data_freshness import check_symbol_freshness

    with (
        _patch_session(_rows(["AAPL"])),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
        _patch_benchmark({"US": (("SPY", "VOO"), None)}),
    ):
        detail = check_symbol_freshness(["AAPL"])

    us = detail["stale_markets"][0]
    assert us["reason"] == "benchmark_stale"
    assert us["benchmark"]["last_cached_date"] is None
    assert "benchmark SPY/VOO" in detail["message"]


def test_benchmark_check_only_blocks_its_own_market():
    from app.services.market_data_freshness import check_symbol_freshness

    rows = [*_rows(["AAPL"]), *_rows(["0700.HK"], market="HK", last_date=date(2026, 6, 18))]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18), "HK": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18), "HK": date(2026, 6, 18)}),
        _patch_benchmark({
            "US": (("SPY",), date(2026, 6, 18)),
            "HK": (("^HSI",), date(2026, 6, 16)),
        }),
    ):
        detail = check_symbol_freshness(["AAPL", "0700.HK"])

    assert [m["market"] for m in detail["stale_markets"]] == ["HK"]


def test_market_without_a_configured_benchmark_is_not_blocked():
    from app.services.market_data_freshness import check_symbol_freshness

    with (
        _patch_session(_rows(["AAPL"])),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
        _patch_benchmark({"US": ((), None)}),
    ):
        assert check_symbol_freshness(["AAPL"]) is None


def _benchmark_reader(bars_by_symbol):
    """The real benchmark reader over fake ``stock_prices`` rows per symbol."""
    from app.services.benchmark_cache_service import BenchmarkCacheService

    class _Session:
        def __init__(self):
            self._symbol = None

        def query(self, *args):
            return self

        def filter(self, symbol_clause, *args):
            self._symbol = symbol_clause.right.value
            return self

        def order_by(self, *args):
            return self

        def all(self):
            return bars_by_symbol.get(self._symbol, [])

        def close(self):
            pass

    return BenchmarkCacheService(redis_client=object(), session_factory=_Session)


def _bars(count, last, *, nan_last=False):
    days =[last - timedelta(days=offset) for offset in range(count)][::-1]
    return [
        SimpleNamespace(
            date=day,
            open=100.0,
            high=101.0,
            low=99.0,
            close=float("nan") if nan_last and day == last else 100.5,
            volume=1_000_000,
        )
        for day in days
    ]


def _real_latest_benchmark_bar(monkeypatch, bars_by_symbol):
    from app.services import market_data_freshness

    monkeypatch.undo()  # the real helper, not the autouse stand-in
    monkeypatch.setattr(
        market_data_freshness,
        "get_benchmark_cache",
        lambda: _benchmark_reader(bars_by_symbol),
    )
    return market_data_freshness._latest_benchmark_bar


def test_latest_benchmark_bar_takes_the_newest_usable_candidate(monkeypatch):
    from app.services.benchmark_registry_service import benchmark_registry

    primary, fallback = benchmark_registry.get_candidate_symbols("US")
    latest_bar = _real_latest_benchmark_bar(
        monkeypatch,
        {primary: _bars(200, date.today() - timedelta(days=3)), fallback: _bars(200, date.today())},
    )

    candidates, latest = latest_bar("US")

    assert candidates == (primary, fallback)
    assert latest == date.today()


def test_latest_benchmark_bar_skips_a_latest_bar_the_reader_drops(monkeypatch):
    """Codex review on #458: a current-dated row with a non-finite close is
    dropped by the reader, so it must not count as fresh."""
    from app.services.benchmark_registry_service import benchmark_registry

    candidates = benchmark_registry.get_candidate_symbols("US")
    latest_bar = _real_latest_benchmark_bar(
        monkeypatch, {candidates[0]: _bars(200, date.today(), nan_last=True)}
    )

    assert latest_bar("US")[1] == date.today() - timedelta(days=1)


def test_latest_benchmark_bar_ignores_a_candidate_the_reader_rejects(monkeypatch):
    """Fewer than 100 rows: the reader returns nothing and the scan would fetch."""
    from app.services.benchmark_registry_service import benchmark_registry

    candidates = benchmark_registry.get_candidate_symbols("US")
    latest_bar = _real_latest_benchmark_bar(
        monkeypatch, {candidates[0]: _bars(99, date.today())}
    )

    assert latest_bar("US")[1] is None


def test_latest_benchmark_bar_treats_an_unknown_market_as_unconfigured(monkeypatch):
    latest_bar = _real_latest_benchmark_bar(monkeypatch, {})

    assert latest_bar("ZZ") == ((), None)


def _custom_policy():
    from app.services.market_data_freshness import ScanFreshnessPolicy

    return ScanFreshnessPolicy.for_universe_type("custom")


def _evaluate_us_custom(fresh, stale, *, policy=None, refresh_state=None, calendar=None):
    from app.services.market_data_freshness import evaluate_symbol_freshness

    rows = [*_rows(fresh), *_rows(stale, last_date=date(2026, 5, 13))]
    with (
        _patch_session(rows),
        calendar or _patch_calendar({"US": date(2026, 6, 18)}),
        refresh_state or _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        return evaluate_symbol_freshness([*fresh, *stale], policy=policy or _custom_policy())


@pytest.mark.parametrize(
    ("fresh", "stale", "message"),
    [
        (
            [f"FRESH{i}" for i in range(9)],
            ["HALTED"],
            "Omitted 1 stale symbol from this scan (90.00% fresh): HALTED",
        ),
        (["NVDA"], ["HALTED"], "Omitted 1 stale symbol from this scan (50.00% fresh): HALTED"),
        (
            [f"FRESH{i}" for i in range(8)],
            ["HALTED", "SUSPENDED"],
            "Omitted 2 stale symbols from this scan (80.00% fresh): HALTED, SUSPENDED",
        ),
    ],
    ids=["10_with_1_stale", "2_with_1_stale", "10_with_2_stale"],
)
def test_custom_scan_omits_stale_symbols_while_one_fresh_symbol_remains(fresh, stale, message):
    decision = _evaluate_us_custom(fresh, stale)

    assert decision.blocking_detail is None
    assert decision.symbols_to_scan == tuple(fresh)
    assert [warning.message for warning in decision.warnings] == [message]
    assert decision.warnings[0].omitted_symbols == tuple(stale)


def test_custom_scan_stores_at_most_200_omitted_symbols_and_names_20():
    stale = [f"STALE{i:03d}" for i in range(250)]

    decision = _evaluate_us_custom(["NVDA"], stale)

    [warning] = decision.warnings
    assert decision.symbols_to_scan == ("NVDA",)
    assert warning.omitted_count == 250
    assert warning.omitted_symbols == tuple(stale[:200])
    assert warning.message == (
        "Omitted 250 stale symbols from this scan (0.40% fresh): "
        "STALE000, STALE001, STALE002, STALE003, STALE004, STALE005, STALE006, "
        "STALE007, STALE008, STALE009, STALE010, STALE011, STALE012, STALE013, "
        "STALE014, STALE015, STALE016, STALE017, STALE018, STALE019, ..."
    )


def test_custom_scan_with_every_symbol_stale_still_blocks():
    decision = _evaluate_us_custom([], ["HALTED", "SUSPENDED"])

    assert decision.warnings == ()
    assert decision.blocking_detail["code"] == "market_data_stale"
    assert decision.blocking_detail["stale_markets"][0]["sample_stale_symbols"] == [
        "HALTED",
        "SUSPENDED",
    ]


@pytest.mark.parametrize("universe_type", ["test", "market", "all"])
def test_non_custom_scans_keep_their_thresholds_for_a_small_list(universe_type):
    from app.services.market_data_freshness import ScanFreshnessPolicy

    decision = _evaluate_us_custom(
        [f"FRESH{i}" for i in range(9)],
        ["HALTED"],
        policy=ScanFreshnessPolicy.for_universe_type(universe_type),
    )

    assert decision.warnings == ()
    assert decision.blocking_detail["code"] == "market_data_stale"


@pytest.mark.parametrize(
    ("patches", "reason"),
    [
        (
            {"refresh_state": _patch_refresh_state_payload({
                "US": {"market": "US", "status": "running", "last_refreshed_trading_day": "2026-06-18"}
            })},
            "refresh_state_missing",
        ),
        ({"refresh_state": _patch_refresh_state({"US": date(2026, 6, 17)})}, "refresh_state_stale"),
        (
            {"calendar": patch(
                "app.services.market_data_freshness.get_market_calendar_service",
                return_value=SimpleNamespace(
                    last_completed_trading_day=lambda market, now=None: 1 / 0
                ),
            )},
            "calendar_unavailable",
        ),
    ],
    ids=["refresh_state_missing", "refresh_state_stale", "calendar_unavailable"],
)
def test_custom_scan_still_hard_blocks_on_market_state(patches, reason):
    decision = _evaluate_us_custom([f"FRESH{i}" for i in range(9)], ["HALTED"], **patches)

    assert decision.warnings == ()
    assert decision.blocking_detail["stale_markets"][0]["reason"] == reason


def test_custom_scan_still_hard_blocks_on_a_stale_benchmark():
    with _patch_benchmark({"US": (("SPY",), date(2026, 6, 17))}):
        decision = _evaluate_us_custom([f"FRESH{i}" for i in range(9)], ["HALTED"])

    assert decision.warnings == ()
    assert decision.blocking_detail["stale_markets"][0]["reason"] == "benchmark_stale"


def test_custom_scan_with_inactive_symbols_allowed_blocks_on_unresolved_symbols():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    rows = [*_rows([f"FRESH{i}" for i in range(8)]), *_rows(["HALTED"], last_date=date(2026, 5, 13))]
    with (
        _patch_session(rows),
        _patch_calendar({"US": date(2026, 6, 18)}),
        _patch_refresh_state({"US": date(2026, 6, 18)}),
    ):
        decision = evaluate_symbol_freshness(
            [*(f"FRESH{i}" for i in range(8)), "HALTED", "UNLISTED"],
            policy=_custom_policy(),
        )

    assert decision.warnings == ()
    assert decision.blocking_detail["unresolved_symbols"] == ["UNLISTED"]
