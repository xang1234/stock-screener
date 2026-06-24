from __future__ import annotations

from collections import Counter
from datetime import date


def _github_seed():
    from app.services.price_refresh_planning import GitHubSeedOutcome

    seed = GitHubSeedOutcome.from_mapping(
        {
            "status": "up_to_date",
            "as_of_date": "2026-06-23",
            "source_revision": "daily_prices_us:20260624002036",
        }
    )
    assert seed is not None
    return seed


def test_seeded_live_accounting_reports_universe_coverage_not_live_batch_size():
    from app.services.price_refresh_accounting import account_live_refresh
    from app.services.price_refresh_execution import PriceRefreshExecutionSummary
    from app.services.price_refresh_planning import (
        PriceRefreshCoverageSummary,
        PriceRefreshJob,
        PriceRefreshJobKind,
        PriceRefreshMode,
        PriceRefreshPlan,
        PriceRefreshSource,
    )

    top_up_symbols = tuple(f"TOP{i}" for i in range(84))
    plan = PriceRefreshPlan(
        symbols=top_up_symbols,
        jobs=(
            PriceRefreshJob(
                kind=PriceRefreshJobKind.STALE,
                symbols=top_up_symbols,
                period="7d",
            ),
        ),
        all_symbols=tuple(f"SYM{i}" for i in range(9983)),
        symbol_markets={symbol: "US" for symbol in top_up_symbols},
        github_seed=_github_seed(),
        github_seed_used=True,
        coverage_summary=PriceRefreshCoverageSummary(
            universe_total=9983,
            already_fresh=9899,
            stale=84,
            no_history=0,
            live_top_up_total=84,
            universe_total_by_market={"US": 9983},
            already_fresh_by_market={"US": 9899},
            live_top_up_total_by_market={"US": 84},
        ),
    )
    execution = PriceRefreshExecutionSummary(
        refreshed=50,
        failed=34,
        failed_symbols=[f"FAIL{i}" for i in range(34)],
        failure_kinds={},
        refreshed_by_market=Counter({"US": 50}),
        failed_by_market=Counter({"US": 34}),
        processed=84,
        total=84,
    )

    accounting = account_live_refresh(
        plan,
        execution,
        effective_market="US",
        last_completed_trading_day=lambda _market: date(2026, 6, 23),
    )

    assert accounting.status == "completed"
    assert accounting.source is PriceRefreshSource.GITHUB_AND_LIVE
    assert accounting.refreshed == 9949
    assert accounting.failed == 34
    assert accounting.total == 9983
    assert accounting.coverage_refreshed == 9949
    assert accounting.coverage_failed == 34
    assert accounting.coverage_total == 9983
    assert accounting.coverage_success_rate == 9949 / 9983
    assert accounting.already_fresh == 9899
    assert accounting.live_top_up_refreshed == 50
    assert accounting.live_top_up_failed == 34
    assert accounting.live_top_up_total == 84
    assert accounting.unsupported_top_up_total == 0
    assert accounting.market_success_rates == {"US": (date(2026, 6, 23), 9949 / 9983)}

    outcome = accounting.to_outcome(mode=PriceRefreshMode.DELTA)
    result = outcome.to_task_result()
    assert result["refreshed"] == 9949
    assert result["failed"] == 34
    assert result["total"] == 9983
    assert result["live_top_up_total"] == 84

    finalization = accounting.to_finalization()
    assert finalization.metadata_status == "completed"
    assert finalization.metadata_refreshed == 9949
    assert finalization.metadata_total == 9983


def test_terminal_github_accounting_reports_bundle_count_in_outcome_and_finalization():
    from app.services.price_refresh_accounting import account_terminal_refresh
    from app.services.price_refresh_planning import (
        PriceRefreshMode,
        PriceRefreshPlan,
        PriceRefreshSource,
    )

    plan = PriceRefreshPlan(
        symbols=(),
        all_symbols=("AAPL", "MSFT"),
        github_seed=_github_seed(),
        github_seed_used=True,
        completion_message="GitHub daily price bundle is current - no live fetch needed",
    )

    accounting = account_terminal_refresh(
        plan,
        mode=PriceRefreshMode.DELTA,
        effective_market="US",
        last_completed_trading_day=lambda _market: date(2026, 6, 22),
    )

    assert accounting.source is PriceRefreshSource.GITHUB
    assert accounting.refreshed == 2
    assert accounting.failed == 0
    assert accounting.total == 2
    assert accounting.coverage_refreshed == 2
    assert accounting.coverage_failed == 0
    assert accounting.coverage_total == 2
    assert accounting.live_top_up_total == 0
    assert accounting.market_success_rates == {"US": (date(2026, 6, 23), 1.0)}

    outcome = accounting.to_outcome(mode=PriceRefreshMode.DELTA)
    assert outcome.refreshed == 2
    assert outcome.total == 2

    finalization = accounting.to_finalization()
    assert finalization.metadata_refreshed == 2
    assert finalization.metadata_total == 2


def test_live_only_accounting_keeps_live_batch_as_denominator():
    from app.services.price_refresh_accounting import account_live_refresh
    from app.services.price_refresh_execution import PriceRefreshExecutionSummary
    from app.services.price_refresh_planning import (
        PriceRefreshJob,
        PriceRefreshJobKind,
        PriceRefreshPlan,
        PriceRefreshSource,
    )

    plan = PriceRefreshPlan(
        symbols=("AAPL", "MSFT", "TSLA"),
        jobs=(
            PriceRefreshJob(
                kind=PriceRefreshJobKind.STALE,
                symbols=("AAPL", "MSFT", "TSLA"),
                period="7d",
            ),
        ),
        all_symbols=("AAPL", "MSFT", "TSLA"),
        symbol_markets={"AAPL": "US", "MSFT": "US", "TSLA": "US"},
    )
    execution = PriceRefreshExecutionSummary(
        refreshed=2,
        failed=1,
        failed_symbols=["TSLA"],
        refreshed_by_market=Counter({"US": 2}),
        failed_by_market=Counter({"US": 1}),
        processed=3,
        total=3,
    )

    accounting = account_live_refresh(
        plan,
        execution,
        effective_market="US",
        last_completed_trading_day=lambda _market: date(2026, 6, 23),
    )

    assert accounting.status == "partial"
    assert accounting.source is PriceRefreshSource.LIVE
    assert accounting.refreshed == 2
    assert accounting.failed == 1
    assert accounting.total == 3
    assert accounting.coverage_total is None
    assert accounting.live_top_up_total is None
    assert accounting.market_success_rates == {}


def test_github_seeded_live_accounting_without_coverage_preserves_source():
    from app.services.price_refresh_accounting import account_live_refresh
    from app.services.price_refresh_execution import PriceRefreshExecutionSummary
    from app.services.price_refresh_planning import (
        PriceRefreshJob,
        PriceRefreshJobKind,
        PriceRefreshPlan,
        PriceRefreshSource,
    )

    plan = PriceRefreshPlan(
        symbols=("AAPL", "MSFT"),
        jobs=(
            PriceRefreshJob(
                kind=PriceRefreshJobKind.STALE,
                symbols=("AAPL", "MSFT"),
                period="7d",
            ),
        ),
        all_symbols=("AAPL", "MSFT"),
        github_seed=_github_seed(),
        github_seed_used=True,
    )
    execution = PriceRefreshExecutionSummary(
        refreshed=2,
        failed=0,
        failed_symbols=[],
        refreshed_by_market=Counter({"US": 2}),
        processed=2,
        total=2,
    )

    accounting = account_live_refresh(
        plan,
        execution,
        effective_market="US",
        last_completed_trading_day=lambda _market: date(2026, 6, 23),
    )

    assert accounting.source is PriceRefreshSource.GITHUB_AND_LIVE
    assert accounting.refreshed == 2
    assert accounting.total == 2
    assert accounting.coverage_total is None
