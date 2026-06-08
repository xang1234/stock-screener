from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from app.models.stock import StockPrice


def _calendar(day: date):
    return SimpleNamespace(last_completed_trading_day=lambda _market: day)


def test_price_history_coverage_splits_fresh_stale_and_no_history(universe_session):
    from app.services.price_history_coverage import classify_price_history

    universe_session.add_all(
        [
            StockPrice(symbol="0700.HK", date=date(2026, 6, 5), close=100),
            StockPrice(symbol="0005.HK", date=date(2026, 6, 8), close=50),
        ]
    )
    universe_session.commit()

    coverage = classify_price_history(
        universe_session,
        symbols=["0700.HK", "0005.HK", "9999.HK"],
        as_of_date=date(2026, 6, 8),
    )

    assert coverage.fresh == ("0005.HK",)
    assert coverage.stale == ("0700.HK",)
    assert coverage.no_history == ("9999.HK",)


def test_bootstrap_plan_uses_stale_top_up_and_full_bootstrap_for_no_history(universe_session):
    from app.services.price_refresh_planning import (
        NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        STALE_PRICE_TOP_UP_PERIOD,
        plan_price_refresh,
    )

    universe_session.add_all(
        [
            StockPrice(symbol="0700.HK", date=date(2026, 6, 5), close=100),
            StockPrice(symbol="0005.HK", date=date(2026, 6, 8), close=50),
        ]
    )
    universe_session.commit()

    plan = plan_price_refresh(
        universe_session,
        all_symbols=["0700.HK", "9999.HK"],
        mode="bootstrap",
        effective_market="HK",
        market_calendar_service=_calendar(date(2026, 6, 8)),
        github_sync={
            "status": "success",
            "as_of_date": "2026-06-05",
            "source_revision": "daily_prices_hk:20260605090000",
            "stale_reason": "behind expected session",
        },
    )

    assert plan.source == "github+live"
    assert plan.github_seed_used is True
    assert plan.symbols == ("0700.HK", "9999.HK")
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        ("stale", ("0700.HK",), STALE_PRICE_TOP_UP_PERIOD),
        ("no_history", ("9999.HK",), NO_HISTORY_PRICE_BOOTSTRAP_PERIOD),
    ]


def test_full_mode_stays_full_even_when_github_sync_result_is_available(universe_session):
    from app.services.price_refresh_planning import (
        NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        plan_price_refresh,
    )

    plan = plan_price_refresh(
        universe_session,
        all_symbols=["0700.HK", "9999.HK"],
        mode="full",
        effective_market="HK",
        market_calendar_service=_calendar(date(2026, 6, 8)),
        github_sync={"status": "success", "as_of_date": "2026-06-08"},
    )

    assert plan.source == "live"
    assert plan.github_seed_used is False
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        ("full", ("0700.HK", "9999.HK"), NO_HISTORY_PRICE_BOOTSTRAP_PERIOD)
    ]


def test_current_github_bundle_classifies_history_without_a_second_missing_symbol_api(universe_session):
    from app.services.price_refresh_planning import (
        NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        STALE_PRICE_TOP_UP_PERIOD,
        plan_price_refresh,
    )

    universe_session.add_all(
        [
            StockPrice(symbol="0700.HK", date=date(2026, 6, 5), close=100),
            StockPrice(symbol="0005.HK", date=date(2026, 6, 8), close=50),
        ]
    )
    universe_session.commit()

    plan = plan_price_refresh(
        universe_session,
        all_symbols=["0700.HK", "0005.HK", "9999.HK"],
        mode="bootstrap",
        effective_market="HK",
        market_calendar_service=_calendar(date(2026, 6, 8)),
        github_sync={
            "status": "success",
            "as_of_date": "2026-06-08",
            "source_revision": "daily_prices_hk:20260608090000",
        },
    )

    assert plan.source == "github+live"
    assert plan.github_seed_used is True
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        ("stale", ("0700.HK",), STALE_PRICE_TOP_UP_PERIOD),
        ("no_history", ("9999.HK",), NO_HISTORY_PRICE_BOOTSTRAP_PERIOD),
    ]


def test_current_github_bundle_accepts_datetime_as_of_date(universe_session):
    from app.services.price_refresh_planning import plan_price_refresh

    universe_session.add(StockPrice(symbol="0700.HK", date=date(2026, 6, 8), close=100))
    universe_session.commit()

    plan = plan_price_refresh(
        universe_session,
        all_symbols=["0700.HK"],
        mode="bootstrap",
        effective_market="HK",
        market_calendar_service=_calendar(date(2026, 6, 8)),
        github_sync={
            "status": "success",
            "as_of_date": datetime(2026, 6, 8, 9, 0),
            "source_revision": "daily_prices_hk:20260608090000",
        },
    )

    assert plan.source == "github"
    assert plan.github_seed_used is True
    assert plan.completion_message == "GitHub daily price bundle is current - no live fetch needed"


def test_failed_github_sync_is_live_top_up_not_github_live(universe_session):
    from app.services.price_refresh_planning import plan_price_refresh

    universe_session.add(StockPrice(symbol="0700.HK", date=date(2026, 6, 5), close=100))
    universe_session.commit()

    plan = plan_price_refresh(
        universe_session,
        all_symbols=["0700.HK"],
        mode="delta",
        effective_market="HK",
        market_calendar_service=_calendar(date(2026, 6, 8)),
        github_sync={"status": "missing", "reason": "not found"},
    )

    assert plan.source == "live"
    assert plan.github_seed_used is False
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        ("stale", ("0700.HK",), "7d"),
    ]
