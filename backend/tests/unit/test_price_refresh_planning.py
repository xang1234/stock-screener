from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from app.models.stock import StockPrice


def _calendar(day: date):
    return SimpleNamespace(last_completed_trading_day=lambda _market: day)


def _seed(payload):
    from app.services.price_refresh_planning import GitHubSeedOutcome

    return GitHubSeedOutcome.from_mapping(payload)


def _planning_input(**overrides):
    from app.services.price_history_coverage import PriceHistoryCoverage
    from app.services.price_refresh_planning import PriceRefreshPlanningInput

    values = {
        "all_symbols": ("0700.HK",),
        "mode": "bootstrap",
        "effective_market": "HK",
        "target_as_of": date(2026, 6, 8),
        "coverage": PriceHistoryCoverage(stale=("0700.HK",)),
    }
    values.update(overrides)
    return PriceRefreshPlanningInput(**values)


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


def test_price_history_coverage_can_require_positive_latest_volume(universe_session):
    from app.services.price_history_coverage import classify_price_history

    universe_session.add_all(
        [
            StockPrice(symbol="GOOD", date=date(2026, 6, 8), close=100, volume=1000),
            StockPrice(symbol="ZERO", date=date(2026, 6, 8), close=50, volume=0),
            StockPrice(symbol="MISSING", date=date(2026, 6, 8), close=25, volume=None),
            StockPrice(symbol="OLD", date=date(2026, 6, 5), close=10, volume=1000),
        ]
    )
    universe_session.commit()

    coverage = classify_price_history(
        universe_session,
        symbols=["GOOD", "ZERO", "MISSING", "OLD", "NONE"],
        as_of_date=date(2026, 6, 8),
        require_positive_volume=True,
    )

    assert coverage.fresh == ("GOOD",)
    assert coverage.stale == ("ZERO", "MISSING", "OLD")
    assert coverage.no_history == ("NONE",)


def test_price_history_coverage_can_require_positive_volume_for_selected_symbols(universe_session):
    from app.services.price_history_coverage import classify_price_history

    universe_session.add_all(
        [
            StockPrice(symbol="STOCK", date=date(2026, 6, 1), close=100, volume=1000),
            StockPrice(symbol="STOCK", date=date(2026, 6, 8), close=101, volume=0),
            StockPrice(symbol="BENCH", date=date(2026, 6, 8), close=50, volume=0),
            StockPrice(symbol="FX", date=date(2026, 6, 8), close=1.3, volume=None),
            StockPrice(symbol="GOOD", date=date(2026, 6, 8), close=25, volume=1000),
        ]
    )
    universe_session.commit()

    coverage = classify_price_history(
        universe_session,
        symbols=["STOCK", "BENCH", "FX", "GOOD", "NONE"],
        as_of_date=date(2026, 6, 8),
        symbols_requiring_positive_volume=["STOCK"],
    )

    assert coverage.fresh == ("BENCH", "FX", "GOOD")
    assert coverage.stale == ("STOCK",)
    assert coverage.no_history == ("NONE",)


def test_bootstrap_plan_uses_stale_top_up_and_full_bootstrap_for_no_history(universe_session):
    from app.services.price_history_coverage import PriceHistoryCoverage
    from app.services.price_refresh_planning import (
        GitHubSeedOutcome,
        PriceRefreshJobKind,
        PriceRefreshSource,
        NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        STALE_PRICE_TOP_UP_PERIOD,
        plan_price_refresh_from_input,
    )

    plan = plan_price_refresh_from_input(_planning_input(
        all_symbols=["0700.HK", "9999.HK"],
        github_seed=_seed({
            "status": "success",
            "as_of_date": "2026-06-05",
            "source_revision": "daily_prices_hk:20260605090000",
            "stale_reason": "behind expected session",
        }),
        coverage=PriceHistoryCoverage(stale=("0700.HK",), no_history=("9999.HK",)),
    ))

    assert plan.source is PriceRefreshSource.GITHUB_AND_LIVE
    assert plan.github_seed_used is True
    assert isinstance(plan.github_seed, GitHubSeedOutcome)
    assert plan.github_seed.status.value == "success"
    assert plan.symbols == ("0700.HK", "9999.HK")
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        (PriceRefreshJobKind.STALE, ("0700.HK",), STALE_PRICE_TOP_UP_PERIOD),
        (PriceRefreshJobKind.NO_HISTORY, ("9999.HK",), NO_HISTORY_PRICE_BOOTSTRAP_PERIOD),
    ]
    assert plan.coverage_summary is not None
    assert plan.coverage_summary.universe_total == 2
    assert plan.coverage_summary.already_fresh == 0
    assert plan.coverage_summary.live_top_up_total == 2
    assert plan.coverage_summary.universe_total_by_market == {"HK": 2}


def test_price_refresh_plan_excludes_unsupported_yahoo_symbols_from_live_jobs():
    from app.services.price_history_coverage import PriceHistoryCoverage
    from app.services.price_refresh_planning import (
        PriceRefreshJobKind,
        plan_price_refresh_from_input,
    )

    plan = plan_price_refresh_from_input(_planning_input(
        all_symbols=["0335.T", "335A.T"],
        effective_market="JP",
        coverage=PriceHistoryCoverage(no_history=("0335.T", "335A.T")),
    ))

    assert plan.all_symbols == ("0335.T", "335A.T")
    assert plan.symbols == ("335A.T",)
    assert plan.unsupported_symbols == ("0335.T",)
    assert plan.coverage_summary is not None
    assert plan.coverage_summary.universe_total == 2
    assert plan.coverage_summary.live_top_up_total == 1
    assert plan.coverage_summary.unsupported_top_up_total == 1
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        (PriceRefreshJobKind.NO_HISTORY, ("335A.T",), "2y"),
    ]


def test_current_github_bundle_reports_unsupported_only_top_up_as_terminal_gap():
    from app.services.price_history_coverage import PriceHistoryCoverage
    from app.services.price_refresh_planning import (
        PriceRefreshSource,
        plan_price_refresh_from_input,
    )

    plan = plan_price_refresh_from_input(_planning_input(
        all_symbols=["0335.T"],
        effective_market="JP",
        github_seed=_seed({
            "status": "success",
            "as_of_date": "2026-06-08",
            "source_revision": "daily_prices_jp:20260608090000",
        }),
        coverage=PriceHistoryCoverage(no_history=("0335.T",)),
    ))

    assert plan.source is PriceRefreshSource.GITHUB
    assert plan.symbols == ()
    assert plan.unsupported_symbols == ("0335.T",)
    assert plan.completion_message == (
        "GitHub daily price bundle synced; unsupported symbols could not be live-refreshed"
    )
    assert plan.coverage_summary is not None
    assert plan.coverage_summary.already_fresh == 0
    assert plan.coverage_summary.unsupported_top_up_total == 1


def test_full_mode_stays_full_even_when_github_sync_result_is_available(universe_session):
    from app.services.price_refresh_planning import (
        NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        PriceRefreshJobKind,
        PriceRefreshSource,
        plan_price_refresh_from_input,
    )

    plan = plan_price_refresh_from_input(_planning_input(
        all_symbols=["0700.HK", "9999.HK"],
        mode="full",
        github_seed=_seed({"status": "success", "as_of_date": "2026-06-08"}),
        coverage=None,
    ))

    assert plan.source is PriceRefreshSource.LIVE
    assert plan.github_seed_used is False
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        (PriceRefreshJobKind.FULL, ("0700.HK", "9999.HK"), NO_HISTORY_PRICE_BOOTSTRAP_PERIOD)
    ]


def test_current_github_bundle_classifies_history_without_a_second_missing_symbol_api(universe_session):
    from app.services.price_history_coverage import PriceHistoryCoverage
    from app.services.price_refresh_planning import (
        NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        PriceRefreshJobKind,
        PriceRefreshSource,
        STALE_PRICE_TOP_UP_PERIOD,
        plan_price_refresh_from_input,
    )

    plan = plan_price_refresh_from_input(_planning_input(
        all_symbols=["0700.HK", "0005.HK", "9999.HK"],
        github_seed=_seed({
            "status": "success",
            "as_of_date": "2026-06-08",
            "source_revision": "daily_prices_hk:20260608090000",
        }),
        coverage=PriceHistoryCoverage(
            fresh=("0005.HK",),
            stale=("0700.HK",),
            no_history=("9999.HK",),
        ),
    ))

    assert plan.source is PriceRefreshSource.GITHUB_AND_LIVE
    assert plan.github_seed_used is True
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        (PriceRefreshJobKind.STALE, ("0700.HK",), STALE_PRICE_TOP_UP_PERIOD),
        (PriceRefreshJobKind.NO_HISTORY, ("9999.HK",), NO_HISTORY_PRICE_BOOTSTRAP_PERIOD),
    ]
    assert plan.coverage_summary is not None
    assert plan.coverage_summary.universe_total == 3
    assert plan.coverage_summary.already_fresh == 1
    assert plan.coverage_summary.live_top_up_total == 2
    assert plan.coverage_summary.universe_total_by_market == {"HK": 3}


def test_current_github_bundle_accepts_datetime_as_of_date(universe_session):
    from app.services.price_history_coverage import PriceHistoryCoverage
    from app.services.price_refresh_planning import PriceRefreshSource, plan_price_refresh_from_input

    plan = plan_price_refresh_from_input(_planning_input(
        all_symbols=["0700.HK"],
        github_seed=_seed({
            "status": "success",
            "as_of_date": datetime(2026, 6, 8, 9, 0),
            "source_revision": "daily_prices_hk:20260608090000",
        }),
        coverage=PriceHistoryCoverage(fresh=("0700.HK",)),
    ))

    assert plan.source is PriceRefreshSource.GITHUB
    assert plan.github_seed_used is True
    assert plan.completion_message == "GitHub daily price bundle is current - no live fetch needed"


def test_failed_github_sync_is_live_top_up_not_github_live(universe_session):
    from app.services.price_refresh_planning import (
        PriceRefreshJobKind,
        PriceRefreshSource,
        plan_price_refresh_from_input,
    )

    plan = plan_price_refresh_from_input(_planning_input(
        all_symbols=["0700.HK"],
        mode="delta",
        github_seed=_seed({"status": "missing", "reason": "not found"}),
    ))

    assert plan.source is PriceRefreshSource.LIVE
    assert plan.github_seed_used is False
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        (PriceRefreshJobKind.STALE, ("0700.HK",), "7d"),
    ]


def test_github_seed_and_plan_do_not_expose_mapping_compatibility_surface():
    from app.services.price_refresh_planning import (
        GitHubSeedOutcome,
        GitHubSeedStatus,
        PriceRefreshPlan,
    )

    seed = GitHubSeedOutcome(status=GitHubSeedStatus.SUCCESS)
    plan = PriceRefreshPlan(symbols=(), github_seed=seed)

    assert not hasattr(seed, "get")
    assert "__getitem__" not in GitHubSeedOutcome.__dict__
    assert "github_sync" not in PriceRefreshPlan.__dict__


def test_price_refresh_plan_can_be_built_from_precomputed_inputs_without_database():
    from app.services.price_history_coverage import PriceHistoryCoverage
    from app.services.price_refresh_planning import (
        PriceRefreshJobKind,
        PriceRefreshPlanningInput,
        PriceRefreshSource,
        plan_price_refresh_from_input,
    )

    plan = plan_price_refresh_from_input(
        PriceRefreshPlanningInput(
            all_symbols=("0700.HK", "0005.HK", "9999.HK"),
            mode="bootstrap",
            effective_market="HK",
            github_seed=_seed({
                "status": "success",
                "as_of_date": "2026-06-08",
                "source_revision": "daily_prices_hk:20260608090000",
            }),
            coverage=PriceHistoryCoverage(
                fresh=("0005.HK",),
                stale=("0700.HK",),
                no_history=("9999.HK",),
            ),
        )
    )

    assert plan.source is PriceRefreshSource.GITHUB_AND_LIVE
    assert plan.github_seed_used is True
    assert [(job.kind, job.symbols, job.period) for job in plan.jobs] == [
        (PriceRefreshJobKind.STALE, ("0700.HK",), "7d"),
        (PriceRefreshJobKind.NO_HISTORY, ("9999.HK",), "2y"),
    ]


def test_build_market_price_refresh_plan_owns_universe_and_github_seed(universe_session):
    from app.models.stock_universe import StockUniverse
    from app.services.price_refresh_plan_builder import build_market_price_refresh_plan
    from app.services.price_refresh_planning import (
        PriceRefreshSource,
    )

    universe_session.add_all(
        [
            StockUniverse(symbol="0700.HK", market="HK", market_cap=500),
            StockUniverse(symbol="0005.HK", market="HK", market_cap=100),
            StockUniverse(symbol="7203.T", market="JP", market_cap=300),
        ]
    )
    universe_session.add(StockPrice(symbol="0700.HK", date=date(2026, 6, 8), close=100))
    universe_session.commit()

    sync_calls = []

    plan = build_market_price_refresh_plan(
        universe_session,
        mode="bootstrap",
        market="hk",
        effective_market="HK",
        normalize_market=lambda market: str(market).upper(),
        market_calendar_service=_calendar(date(2026, 6, 8)),
        sync_github_seed=lambda db, *, market, allow_stale: (
            sync_calls.append((db, market, allow_stale))
            or {"status": "success", "as_of_date": "2026-06-08"}
        ),
    )

    assert sync_calls == [(universe_session, "HK", True)]
    # Universe symbols first, then the HK key-market instruments needed by
    # the Daily Snapshot cards (0700.HK deduplicated against the universe).
    assert plan.all_symbols == (
        "0700.HK", "0005.HK", "^HSI", "2800.HK", "3690.HK", "0941.HK",
    )
    assert plan.symbol_markets == {symbol: "HK" for symbol in plan.all_symbols}
    assert plan.source is PriceRefreshSource.GITHUB_AND_LIVE
    assert plan.symbols == ("0005.HK", "^HSI", "2800.HK", "3690.HK", "0941.HK")


def test_split_supported_price_symbols_reuses_provider_no_data_policy():
    from app.domain.providers.price_symbol_support import split_supported_price_symbols

    supported, unsupported = split_supported_price_symbols(
        ["7203.T", "0123.T", "BAD-W", "AAPL"]
    )

    assert supported == ["7203.T", "AAPL"]
    assert unsupported == ["0123.T", "BAD-W"]


def test_legacy_symbol_support_imports_reexport_domain_policy():
    from app.domain.providers.price_symbol_support import split_supported_price_symbols
    from app.services.price_symbol_validation import (
        split_supported_price_symbols as service_split_supported_price_symbols,
    )
    from app.utils.symbol_support import (
        split_supported_price_symbols as utils_split_supported_price_symbols,
    )

    assert service_split_supported_price_symbols is split_supported_price_symbols
    assert utils_split_supported_price_symbols is split_supported_price_symbols


def test_bootstrap_price_readiness_uses_price_refresh_universe_and_support_policy(
    universe_session,
):
    from app.models.stock_universe import StockUniverse
    from app.services.bootstrap_price_readiness import evaluate_bootstrap_price_readiness

    universe_session.add_all(
        [
            StockUniverse(symbol="7203.T", market="JP", market_cap=500),
            StockUniverse(symbol="0123.T", market="JP", market_cap=400),
            StockUniverse(symbol="JP-W", market="JP", market_cap=300),
            StockUniverse(symbol="AAPL", market="US", market_cap=200),
        ]
    )
    universe_session.add(StockPrice(symbol="7203.T", date=date(2026, 6, 8), close=100))
    universe_session.commit()

    report = evaluate_bootstrap_price_readiness(
        universe_session,
        market="jp",
        as_of_date=date(2026, 6, 8),
    )

    assert report["market"] == "JP"
    assert report["eligible"] is True
    assert report["price_total_symbols"] == 1
    assert report["price_covered_symbols"] == 1
    assert report["unsupported_skipped_count"] == 2
    assert report["unsupported_symbols_preview"] == ["0123.T", "JP-W"]
