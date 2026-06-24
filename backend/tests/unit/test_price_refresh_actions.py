from __future__ import annotations


def test_terminal_completion_helper_returns_none_when_plan_has_symbols():
    from app.services import price_refresh_actions as module
    from app.services.price_refresh_actions import build_terminal_completion
    from app.services.price_refresh_planning import PriceRefreshMode, PriceRefreshPlan

    completion = build_terminal_completion(
        mode=PriceRefreshMode.BOOTSTRAP,
        effective_market="JP",
        plan=PriceRefreshPlan(symbols=("7203.T",)),
        last_completed_trading_day=lambda market: (_ for _ in ()).throw(
            AssertionError(f"unexpected calendar lookup for {market}")
        ),
    )

    assert not hasattr(module, "PriceRefreshActionFactory")
    assert completion is None


def test_terminal_completion_helper_returns_terminal_completion():
    from datetime import date

    from app.services.price_refresh_actions import build_terminal_completion
    from app.services.price_refresh_planning import (
        GitHubSeedOutcome,
        PriceRefreshMode,
        PriceRefreshPlan,
        PriceRefreshSource,
    )

    github_seed = GitHubSeedOutcome.from_mapping(
        {
            "status": "success",
            "as_of_date": "2026-06-08",
            "source_revision": "daily_prices_jp:20260608120000",
        }
    )
    assert github_seed is not None
    plan = PriceRefreshPlan(
        symbols=(),
        all_symbols=("7203.T", "9984.T"),
        github_seed=github_seed,
        github_seed_used=True,
        completion_message="GitHub daily price bundle is current - no live fetch needed",
    )

    completion = build_terminal_completion(
        mode=PriceRefreshMode.BOOTSTRAP,
        effective_market="JP",
        plan=plan,
        last_completed_trading_day=lambda market: date(2026, 6, 7),
    )

    assert completion is not None
    assert completion.outcome.source is PriceRefreshSource.GITHUB
    assert completion.outcome.github_seed is github_seed
    assert completion.outcome.message == plan.completion_message
    assert completion.outcome.refreshed == 2
    assert completion.outcome.failed == 0
    assert completion.outcome.total == 2
    assert completion.outcome.coverage_refreshed == 2
    assert completion.outcome.coverage_failed == 0
    assert completion.outcome.coverage_total == 2
    assert completion.outcome.live_top_up_refreshed == 0
    assert completion.outcome.live_top_up_failed == 0
    assert completion.outcome.live_top_up_total == 0
    assert completion.finalization.metadata_refreshed == 2
    assert completion.finalization.metadata_total == 2
    assert completion.finalization.market_success_rates == {
        "JP": (date(2026, 6, 8), 1.0),
    }
