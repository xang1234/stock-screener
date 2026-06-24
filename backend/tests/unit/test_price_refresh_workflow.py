from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_price_refresh_outcome_serializes_typed_github_metadata():
    from app.services.price_refresh_planning import (
        GitHubSeedOutcome,
        PriceRefreshMode,
        PriceRefreshSource,
    )
    from app.services.price_refresh_workflow import PriceRefreshOutcome

    outcome = PriceRefreshOutcome(
        status="completed",
        mode=PriceRefreshMode.BOOTSTRAP,
        source=PriceRefreshSource.GITHUB,
        message="GitHub daily price bundle is current - no live fetch needed",
        refreshed=0,
        failed=0,
        total=0,
        completed_at=datetime(2026, 6, 8, 12, 0, 0),
        github_seed=GitHubSeedOutcome.from_mapping(
            {
                "status": "success",
                "as_of_date": "2026-06-08",
                "source_revision": "daily_prices_jp:20260608120000",
            }
        ),
    )

    assert outcome.to_task_result() == {
        "status": "completed",
        "source": "github",
        "github_sync_status": "success",
        "source_revision": "daily_prices_jp:20260608120000",
        "refreshed": 0,
        "failed": 0,
        "total": 0,
        "message": "GitHub daily price bundle is current - no live fetch needed",
        "mode": "bootstrap",
        "completed_at": "2026-06-08T12:00:00",
    }


def test_github_only_terminal_completion_does_not_warm_benchmarks():
    from app.services.price_refresh_activity import PriceRefreshActivityReporter
    from app.services.price_refresh_live_runner import PriceRefreshRetryScheduler
    from app.services.price_refresh_planning import (
        GitHubSeedOutcome,
        PriceRefreshPlan,
    )
    from app.services.price_refresh_workflow import (
        PriceRefreshMarketGateway,
        PriceRefreshWorkflow,
        PriceRefreshWorkflowDependencies,
    )

    db = MagicMock()
    price_cache = MagicMock()
    activity_reporter = MagicMock(spec=PriceRefreshActivityReporter)
    live_runner = MagicMock()
    retry_scheduler = MagicMock(spec=PriceRefreshRetryScheduler)
    warm_benchmarks = MagicMock(return_value={"status": "ok"})
    github_seed = GitHubSeedOutcome.from_mapping(
        {
            "status": "success",
            "as_of_date": "2026-06-08",
            "source_revision": "daily_prices_jp:20260608090000",
        }
    )
    build_refresh_plan = MagicMock(
        return_value=PriceRefreshPlan(
            symbols=(),
            jobs=(),
            all_symbols=("7203.T", "6758.T"),
            github_seed=github_seed,
            github_seed_used=True,
            completion_message="GitHub daily price bundle is current - no live fetch needed",
        )
    )
    gateway = PriceRefreshMarketGateway(
        normalize_market=lambda market: str(market).upper(),
        market_tag=lambda market: f"[{market}]",
        log_extra=lambda market: {"market": market},
        get_eastern_now=lambda: SimpleNamespace(
            weekday=lambda: 1,
            hour=12,
            date=lambda: date(2026, 6, 9),
        ),
        is_trading_day=lambda _day: True,
        format_market_status=lambda: "open",
        is_market_enabled_now=lambda _market: True,
    )
    workflow = PriceRefreshWorkflow(
        PriceRefreshWorkflowDependencies(
            session_factory=lambda: db,
            price_cache_factory=lambda: price_cache,
            bulk_fetcher_factory=MagicMock(),
            warm_benchmarks=warm_benchmarks,
            build_refresh_plan=build_refresh_plan,
            last_completed_trading_day=lambda _market: date(2026, 6, 8),
            activity_reporter=activity_reporter,
            live_runner=live_runner,
            retry_scheduler=retry_scheduler,
            market_gateway=gateway,
            raise_if_transient_database_error=MagicMock(),
            safe_rollback=MagicMock(),
        )
    )
    task = SimpleNamespace(
        name="app.tasks.cache_tasks.smart_refresh_cache",
        request=SimpleNamespace(id="task-jp"),
        update_state=MagicMock(),
    )

    result = workflow.run(task=task, mode="bootstrap", market="JP", activity_lifecycle="bootstrap")

    assert result["status"] == "completed"
    assert result["source"] == "github"
    assert result["refreshed"] == 2
    assert result["failed"] == 0
    assert result["total"] == 2
    assert result["coverage_refreshed"] == 2
    assert result["coverage_failed"] == 0
    assert result["coverage_total"] == 2
    assert result["live_top_up_refreshed"] == 0
    assert result["live_top_up_failed"] == 0
    assert result["live_top_up_total"] == 0
    warm_benchmarks.assert_not_called()
    live_runner.run.assert_not_called()
    activity_reporter.finalize_success.assert_called_once()
    db.close.assert_called_once()
