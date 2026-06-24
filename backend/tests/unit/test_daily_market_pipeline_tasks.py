"""Daily per-market pipeline orchestration tests."""

from __future__ import annotations

from datetime import date

import pytest


class _FakeSignature:
    def __init__(self, task: str, *, args=None, kwargs=None):
        self.task = task
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.queue = None

    def set(self, queue=None, **_kwargs):
        self.queue = queue
        return self


class _FakeTask:
    def __init__(self, task: str):
        self.task = task

    def si(self, *args, **kwargs):
        return _FakeSignature(self.task, args=args, kwargs=kwargs)


def test_daily_market_pipeline_orders_refresh_compute_and_scan(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        "app.tasks.cache_tasks.smart_refresh_cache",
        _FakeTask("app.tasks.cache_tasks.smart_refresh_cache"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        _FakeTask("app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_market_exposure",
        _FakeTask("app.tasks.breadth_tasks.calculate_market_exposure"),
    )
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        _FakeTask("app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        _FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )

    signatures = module._build_daily_market_pipeline_signatures("hk", date(2026, 3, 16))

    assert [signature.task for signature in signatures] == [
        "app.tasks.cache_tasks.smart_refresh_cache",
        "app.tasks.daily_market_pipeline_tasks.guard_price_refresh",
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        "app.tasks.daily_market_pipeline_tasks.guard_breadth_result",
        "app.tasks.breadth_tasks.calculate_market_exposure",
        "app.tasks.daily_market_pipeline_tasks.guard_exposure_result",
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        "app.tasks.daily_market_pipeline_tasks.guard_group_result",
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        "app.tasks.daily_market_pipeline_tasks.guard_snapshot_result",
    ]
    assert signatures[0].kwargs == {"mode": "delta", "market": "HK"}
    assert signatures[-2].kwargs == {
        "market": "HK",
        "as_of_date_str": "2026-03-16",
        "universe_name": "market:HK",
        "publish_pointer_key": "latest_published_market:HK",
        "static_daily_mode": True,
    }


def test_queue_daily_market_pipeline_skips_disabled_market(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: False,
    )

    result = module.queue_daily_market_pipeline.run("HK")

    assert result["status"] == "skipped"
    assert result["reason"] == "market HK is disabled in local runtime preferences"


def test_guard_price_refresh_fails_lock_contention_skip_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "already_running",
        "skipped": True,
        "reason": "data_fetch_lock_active",
    }

    with pytest.raises(RuntimeError, match="Daily price refresh failed for HK"):
        module.guard_price_refresh.run(result, market="HK")


def test_guard_price_refresh_accepts_high_coverage_partial_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 900,
        "failed": 100,
        "total": 1000,
    }

    assert module.guard_price_refresh.run(result, market="HK") == {
        "status": "ok",
        "market": "HK",
        "stage": "prices",
    }


def test_guard_price_refresh_uses_coverage_denominator_when_present():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 50,
        "failed": 34,
        "total": 84,
        "coverage_refreshed": 9949,
        "coverage_failed": 34,
        "coverage_total": 9983,
        "live_top_up_refreshed": 50,
        "live_top_up_failed": 34,
        "live_top_up_total": 84,
    }

    assert module.guard_price_refresh.run(result, market="US") == {
        "status": "ok",
        "market": "US",
        "stage": "prices",
    }


def test_guard_price_refresh_falls_back_when_coverage_values_are_null():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 900,
        "failed": 100,
        "total": 1000,
        "coverage_refreshed": None,
        "coverage_total": None,
    }

    assert module.guard_price_refresh.run(result, market="HK") == {
        "status": "ok",
        "market": "HK",
        "stage": "prices",
    }


def test_guard_price_refresh_fails_low_coverage_partial_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "partial",
        "refreshed": 899,
        "failed": 101,
        "total": 1000,
    }

    with pytest.raises(RuntimeError, match="Daily price refresh failed for HK"):
        module.guard_price_refresh.run(result, market="HK")


def test_guard_snapshot_result_accepts_already_published_scan_result():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "skipped",
        "reason": "already_published",
        "existing_run_id": 123,
        "auto_scan_id": 456,
    }

    assert module.guard_snapshot_result.run(result, market="HK") == {
        "status": "ok",
        "market": "HK",
        "stage": "scan",
        "auto_scan_id": 456,
    }


def test_guard_group_result_accepts_no_taxonomy_market_skip():
    from app.tasks import daily_market_pipeline_tasks as module

    result = {
        "status": "skipped",
        "reason": "no_taxonomy_for_market",
        "market": "JP",
    }

    assert module.guard_group_result.run(result, market="JP") == {
        "status": "ok",
        "market": "JP",
        "stage": "groups",
    }


def test_guard_exposure_result_skips_without_aborting_pipeline():
    # Exposure is a non-critical leaf: a failed/missing exposure result must NOT
    # raise (which would abort groups + snapshot); it returns a skipped status.
    from app.tasks import daily_market_pipeline_tasks as module

    result = {"error": "benchmark_not_current", "market": "US", "date": "2026-06-19"}
    assert module.guard_exposure_result.run(result, market="US") == {
        "status": "skipped",
        "market": "US",
        "stage": "exposure",
    }
