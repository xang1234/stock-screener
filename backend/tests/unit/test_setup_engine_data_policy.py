"""Tests for Setup Engine data sufficiency and degradation policy."""

from app.analysis.patterns.policy import (
    SetupEngineDataRequirements,
    evaluate_setup_engine_data_policy,
    policy_failed_checks,
    policy_invalidation_flags,
)


def test_policy_returns_ok_when_requirements_met():
    result = evaluate_setup_engine_data_policy(
        daily_bars=300,
        weekly_bars=80,
        benchmark_bars=260,
        current_week_sessions=5,
    )
    assert result["status"] == "ok"
    assert result["failed_reasons"] == []
    assert result["degradation_reasons"] == []


def test_policy_degrades_missing_benchmark_when_allowed():
    result = evaluate_setup_engine_data_policy(
        daily_bars=300,
        weekly_bars=80,
        benchmark_bars=0,
        current_week_sessions=5,
    )
    assert result["status"] == "degraded"
    assert "missing_or_short_benchmark_history" in result["degradation_reasons"]


def test_policy_marks_incomplete_week_without_lookahead():
    result = evaluate_setup_engine_data_policy(
        daily_bars=300,
        weekly_bars=80,
        benchmark_bars=260,
        current_week_sessions=2,
    )
    assert result["status"] == "degraded"
    assert result["requires_weekly_exclude_current"] is True


def test_policy_insufficient_daily_history_is_explicit():
    result = evaluate_setup_engine_data_policy(
        daily_bars=100,
        weekly_bars=80,
        benchmark_bars=260,
        current_week_sessions=5,
    )
    assert result["status"] == "insufficient"
    assert "insufficient_daily_bars" in result["failed_reasons"]
    assert "insufficient_data" in policy_failed_checks(result)


def test_policy_benchmark_can_be_required_strictly():
    req = SetupEngineDataRequirements(
        allow_degraded_without_benchmark=False,
    )
    result = evaluate_setup_engine_data_policy(
        daily_bars=300,
        weekly_bars=80,
        benchmark_bars=0,
        current_week_sessions=5,
        requirements=req,
    )
    assert result["status"] == "insufficient"
    assert "insufficient_benchmark_bars" in result["failed_reasons"]
    assert "data_policy:insufficient" in policy_invalidation_flags(result)
