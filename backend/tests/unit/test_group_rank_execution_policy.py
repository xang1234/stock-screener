from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

from app.services.group_rank_cache_policy import GroupRankCacheRequirement
from app.services.group_rank_models import (
    GroupRankCalculationResult,
    GroupRankPrefetchStats,
    GroupRanking,
)


def _prefetch_stats(
    *,
    misses: int = 0,
) -> GroupRankPrefetchStats:
    target = 100
    return GroupRankPrefetchStats(
        target_symbols=target,
        symbols_with_prices=target - misses,
        cache_miss_symbols=misses,
        cache_miss_symbols_sample=(("MISS",) if misses else ()),
        cache_coverage_ratio=(target - misses) / target,
        benchmark_available=True,
        benchmark_cached=True,
        benchmark_symbol="SPY",
        benchmark_role="primary",
        market="US",
        cache_only=True,
        skipped_unsupported_symbols=0,
    )


def _group_calculation(
    *,
    misses: int = 0,
) -> GroupRankCalculationResult:
    return GroupRankCalculationResult(
        rankings=(
            GroupRanking(
                industry_group="Software",
                date=date(2026, 3, 20),
                rank=1,
                avg_rs_rating=90.0,
                median_rs_rating=89.0,
                weighted_avg_rs_rating=91.0,
                rs_std_dev=2.0,
                num_stocks=3,
                num_stocks_rs_above_80=2,
                top_symbol="AAA",
                top_rs_rating=96.0,
            ),
        ),
        prefetch_stats=_prefetch_stats(misses=misses),
    )


def _patch_serialized_lock(monkeypatch):
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    fake_coordination = MagicMock()
    fake_coordination.acquire_market_workload.return_value = (
        True,
        False,
    )
    fake_coordination.release_market_workload.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: fake_coordination,
    )


def _patch_calendar_service(
    monkeypatch,
    now: datetime,
    *,
    is_trading_day: bool = True,
):
    fake = MagicMock()
    fake.is_trading_day.return_value = is_trading_day
    fake.market_now.return_value = now
    fake.last_completed_trading_day.return_value = now.date()
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.get_market_calendar_service",
        lambda: fake,
    )
    return fake


def _setup_daily_task(
    monkeypatch,
    *,
    now: datetime,
    warmup_metadata,
):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.price_cache.get_warmup_metadata.return_value = (
        warmup_metadata
    )
    fake_service.calculate_group_rankings.return_value = (
        _group_calculation()
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )
    monkeypatch.setattr(
        snapshot_module,
        "safe_publish_groups_bootstrap",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, now)
    return module, fake_db, fake_service


def test_same_day_group_rankings_require_complete_warmup(monkeypatch):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 3, 20, 17, 40),
        warmup_metadata={
            "status": "partial",
            "count": 9000,
            "total": 10000,
            "completed_at": datetime.now().isoformat(),
        },
    )

    result = module.calculate_daily_group_rankings.run()

    assert "warmup not complete" in result["error"].lower()
    fake_service.calculate_group_rankings.assert_not_called()


def test_partial_warmup_is_accepted_above_market_threshold(monkeypatch):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 6, 10, 17, 40),
        warmup_metadata={
            "status": "partial",
            "count": 1081,
            "total": 1969,
            "completed_at": datetime.now().isoformat(),
        },
    )

    result = module.calculate_daily_group_rankings.run(market="TW")

    assert result["groups_ranked"] == 1
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs["policy"].mode.value == "auto"
    assert call_kwargs["policy"].cache_only is True
    assert call_kwargs["cache_requirement"] == (
        GroupRankCacheRequirement.minimum(
            0.50,
            reason="partial_warmup",
        )
    )


def test_stale_partial_warmup_is_rejected(monkeypatch):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 6, 10, 17, 40),
        warmup_metadata={
            "status": "partial",
            "count": 1081,
            "total": 1969,
            "completed_at": "2020-01-01T00:00:00",
        },
    )

    result = module.calculate_daily_group_rankings.run(market="TW")

    assert "stale" in result["error"].lower()
    fake_service.calculate_group_rankings.assert_not_called()


def test_in_process_same_day_bypass_keeps_strict_cache_requirement(
    monkeypatch,
):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 3, 20, 17, 40),
        warmup_metadata=None,
    )

    with module.allow_same_day_group_rank_warmup_bypass():
        result = module.calculate_daily_group_rankings.run()

    assert result["groups_ranked"] == 1
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs["policy"].mode.value == "auto"
    assert call_kwargs["policy"].requires_warmup_metadata is False
    assert call_kwargs["cache_requirement"] == (
        GroupRankCacheRequirement.strict()
    )


def test_manual_historical_group_rankings_allow_provider_reads(monkeypatch):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 3, 20, 17, 40),
        warmup_metadata=None,
    )

    result = module.calculate_daily_group_rankings.run("2026-03-19")

    assert result["cache_only"] is False
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs["policy"].mode.value == "auto"
    assert call_kwargs["policy"].cache_only is False
    assert call_kwargs["cache_requirement"] == (
        GroupRankCacheRequirement.disabled()
    )


def test_manual_strict_cache_only_remains_supported(monkeypatch):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 4, 3, 0, 30),
        warmup_metadata=None,
    )

    result = module.calculate_daily_group_rankings.run(
        "2026-04-02",
        force_cache_only=True,
    )

    assert result["cache_only"] is True
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs["policy"].mode.value == "strict_cache_only"
    assert call_kwargs["cache_requirement"] == (
        GroupRankCacheRequirement.strict()
    )


def test_guarded_historical_group_rankings_use_tolerant_cache_only_policy(
    monkeypatch,
):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 3, 20, 17, 40),
        warmup_metadata=None,
    )
    fake_service.calculate_group_rankings.return_value = (
        _group_calculation(misses=30)
    )

    result = module.calculate_daily_group_rankings.run(
        "2026-03-19",
        market="US",
        refresh_guarded_cache_only=True,
    )

    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
    assert result["prefetch_stats"]["cache_miss_symbols"] == 30
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs["policy"].mode.value == "refresh_guarded"
    assert call_kwargs["cache_requirement"] == (
        GroupRankCacheRequirement.disabled()
    )


def test_strict_legacy_flag_wins_over_guarded_tolerance(monkeypatch):
    module, _, fake_service = _setup_daily_task(
        monkeypatch,
        now=datetime(2026, 3, 20, 17, 40),
        warmup_metadata=None,
    )

    module.calculate_daily_group_rankings.run(
        "2026-03-19",
        force_cache_only=True,
        refresh_guarded_cache_only=True,
    )

    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs["policy"].mode.value == "strict_cache_only"
    assert call_kwargs["cache_requirement"] == (
        GroupRankCacheRequirement.strict()
    )


def test_guarded_group_wrapper_propagates_policy_to_gapfill_and_target(
    monkeypatch,
):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = [date(2026, 3, 18)]
    fake_service.fill_gaps_optimized.return_value = {
        "total_dates": 1,
        "processed": 0,
        "errors": 1,
        "prefetch_stats": {"cache_miss_symbols": 4},
    }
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )
    monkeypatch.setattr(
        module.settings,
        "group_rank_gapfill_enabled",
        True,
    )
    monkeypatch.setattr(
        "app.services.ibd_industry_service.IBDIndustryService.get_all_groups",
        lambda db, market: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        datetime(2026, 3, 20, 17, 40),
    )
    target_call = MagicMock(
        return_value={
            "date": "2026-03-19",
            "groups_ranked": 1,
            "cache_only": True,
            "cache_policy": "refresh_guarded",
        }
    )
    monkeypatch.setattr(
        module,
        "_calculate_daily_group_rankings_in_process",
        target_call,
    )

    result = module.calculate_daily_group_rankings_with_gapfill.run(
        market="US",
        calculation_date="2026-03-19",
        execution_policy="refresh_guarded",
    )

    fill_kwargs = fake_service.fill_gaps_optimized.call_args.kwargs
    assert fill_kwargs["market"] == "US"
    assert fill_kwargs["policy"].mode.value == "refresh_guarded"
    assert fill_kwargs["policy"] is fill_kwargs["policy"].for_gap_fill()
    target_call.assert_called_once_with(
        market="US",
        activity_lifecycle="daily_refresh",
        calculation_date="2026-03-19",
        execution_policy="refresh_guarded",
    )
    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
