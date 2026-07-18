from datetime import date
from unittest.mock import MagicMock, Mock

import pytest

from app.services.daily_group_rank_runner import (
    DailyGroupRankDependencies,
    DailyGroupRankRequest,
    GroupRankWarmupIncomplete,
    NoGroupRankingsCalculated,
    run_daily_group_rankings,
)
from app.services.derived_data_execution_policy import (
    resolve_derived_data_execution_policy,
)
from app.services.group_rank_cache_policy import (
    GroupRankCacheRequirement,
)
from app.services.group_rank_models import (
    GroupRankCalculationResult,
    GroupRankPrefetchStats,
    GroupRanking,
)
from app.services.group_rank_warmup_policy import (
    SameDayGroupRankWarmupDecision,
)


DAY = date(2026, 3, 20)


def _policy(mode: str, *, target: date = DAY):
    return resolve_derived_data_execution_policy(
        execution_policy=mode,
        target_date=target,
        current_date=DAY,
    )


def _stats() -> GroupRankPrefetchStats:
    return GroupRankPrefetchStats(
        target_symbols=3,
        symbols_with_prices=3,
        cache_miss_symbols=0,
        cache_miss_symbols_sample=(),
        cache_coverage_ratio=1.0,
        benchmark_available=True,
        benchmark_cached=True,
        benchmark_symbol="SPY",
        benchmark_role="primary",
        market="US",
        cache_only=True,
        skipped_unsupported_symbols=0,
    )


def _ranking() -> GroupRanking:
    return GroupRanking(
        industry_group="Software",
        date=DAY,
        rank=1,
        avg_rs_rating=90.0,
        median_rs_rating=89.0,
        weighted_avg_rs_rating=91.0,
        rs_std_dev=2.0,
        num_stocks=3,
        num_stocks_rs_above_80=2,
        top_symbol="AAA",
        top_rs_rating=96.0,
    )


def _calculation(*, rankings=None) -> GroupRankCalculationResult:
    return GroupRankCalculationResult(
        rankings=(
            (_ranking(),)
            if rankings is None
            else tuple(rankings)
        ),
        prefetch_stats=_stats(),
    )


def _dependencies(*, service=None) -> DailyGroupRankDependencies:
    service = service or Mock()
    service.calculate_group_rankings.return_value = _calculation()
    return DailyGroupRankDependencies(
        service=service,
        bump_epoch=Mock(),
        publish_snapshot=Mock(),
        repair_current_us_metadata=Mock(
            return_value={"repaired": 2}
        ),
    )


def _request(policy=None, **overrides) -> DailyGroupRankRequest:
    values = {
        "calculation_date": DAY,
        "current_date": DAY,
        "market": "US",
        "activity_lifecycle": "daily_refresh",
        "policy": policy or _policy("refresh_guarded"),
    }
    values.update(overrides)
    return DailyGroupRankRequest(**values)


def test_runner_returns_compatible_success_outcome():
    dependencies = _dependencies()
    request = _request()

    outcome = run_daily_group_rankings(
        MagicMock(),
        request,
        dependencies,
    )

    result = outcome.to_task_result(request.policy)
    assert result["groups_ranked"] == 1
    assert result["top_group"] == "Software"
    assert result["cache_policy"] == "refresh_guarded"
    assert result["metadata_repair"] == {"repaired": 2}
    dependencies.bump_epoch.assert_called_once_with("US")
    dependencies.publish_snapshot.assert_called_once_with()
    dependencies.repair_current_us_metadata.assert_called_once_with(DAY)


def test_strict_warmup_failure_prevents_calculation(monkeypatch):
    import app.services.daily_group_rank_runner as module

    service = Mock()
    dependencies = _dependencies(service=service)
    monkeypatch.setattr(
        module,
        "evaluate_same_day_group_rank_warmup",
        lambda *_args, **_kwargs: SameDayGroupRankWarmupDecision(
            error="warmup incomplete",
            cache_requirement=GroupRankCacheRequirement.disabled(),
        ),
    )

    with pytest.raises(
        GroupRankWarmupIncomplete,
        match="warmup incomplete",
    ):
        run_daily_group_rankings(
            MagicMock(),
            _request(policy=_policy("auto")),
            dependencies,
        )

    service.calculate_group_rankings.assert_not_called()
    dependencies.bump_epoch.assert_not_called()


def test_no_groups_failure_carries_prefetch_stats():
    dependencies = _dependencies()
    dependencies.service.calculate_group_rankings.return_value = (
        _calculation(rankings=())
    )

    with pytest.raises(NoGroupRankingsCalculated) as exc_info:
        run_daily_group_rankings(
            MagicMock(),
            _request(),
            dependencies,
        )

    assert exc_info.value.prefetch_stats == _stats()
    assert exc_info.value.duration_seconds >= 0
    dependencies.bump_epoch.assert_not_called()
    dependencies.publish_snapshot.assert_not_called()


def test_transient_service_error_propagates_unchanged():
    failure = ConnectionError("network down")
    dependencies = _dependencies()
    dependencies.service.calculate_group_rankings.side_effect = failure

    with pytest.raises(ConnectionError) as exc_info:
        run_daily_group_rankings(
            MagicMock(),
            _request(),
            dependencies,
        )

    assert exc_info.value is failure
    dependencies.bump_epoch.assert_not_called()


def test_snapshot_failure_is_best_effort():
    dependencies = _dependencies()
    dependencies.publish_snapshot.side_effect = RuntimeError(
        "snapshot unavailable"
    )

    outcome = run_daily_group_rankings(
        MagicMock(),
        _request(),
        dependencies,
    )

    assert outcome.rankings == (_ranking(),)
    dependencies.bump_epoch.assert_called_once_with("US")
