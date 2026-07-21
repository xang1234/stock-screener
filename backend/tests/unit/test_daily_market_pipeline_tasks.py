"""Daily per-market pipeline orchestration tests."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from types import SimpleNamespace

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
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        _FakeTask("app.tasks.market_rs_tasks.calculate_market_rs_snapshot"),
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
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        "app.tasks.daily_market_pipeline_tasks.guard_market_rs_result",
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
    assert signatures[2].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "formula_version": "balanced-horizon-percentile-v2",
    }
    assert signatures[4].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "execution_policy": "refresh_guarded",
    }
    assert signatures[8].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "execution_policy": "refresh_guarded",
    }
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


def test_queue_daily_market_pipeline_skips_until_local_bootstrap_ready(monkeypatch):
    from app.tasks import daily_market_pipeline_tasks as module

    class _FakeCalendar:
        def market_now(self, _market):
            return datetime(2026, 3, 16, 10, 0)

        def is_trading_day(self, _market, _today):
            return True

        def last_completed_trading_day(self, _market):
            return date(2026, 3, 16)

    chain_called = False

    class _FakeChain:
        def apply_async(self):
            nonlocal chain_called
            chain_called = True
            return SimpleNamespace(id="queued-task")

    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_bootstrap_status",
        lambda _db: SimpleNamespace(
            bootstrap_required=True,
            bootstrap_state="not_started",
            primary_market="US",
            enabled_markets=["US"],
        ),
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(module, "_market_pipeline_active", lambda _market: None)
    monkeypatch.setattr(module, "MarketCalendarService", lambda: _FakeCalendar())
    monkeypatch.setattr(module, "_build_daily_market_pipeline_signatures", lambda *_args: [])
    monkeypatch.setattr(module, "chain", lambda *_signatures: _FakeChain())

    result = module.queue_daily_market_pipeline.run("US")

    assert result["status"] == "skipped"
    assert result["reason"] == "local_runtime_bootstrap_not_ready"
    assert result["bootstrap_state"] == "not_started"
    assert result["bootstrap_required"] is True
    assert chain_called is False


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


def test_guard_market_rs_blocks_failed_result_when_balanced_is_active(monkeypatch):
    from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        module,
        "_active_formula_for_market",
        lambda _market: BALANCED_RS_FORMULA_VERSION,
    )

    with pytest.raises(RuntimeError, match="Canonical Market RS failed for US"):
        module.guard_market_rs_result.run(
            {
                "status": "failed",
                "market": "US",
                "as_of_date": "2026-04-10",
                "formula_version": BALANCED_RS_FORMULA_VERSION,
                "reason_code": "benchmark_anchor_missing",
            },
            market="US",
            calculation_date="2026-04-10",
        )


def test_guard_market_rs_allows_failed_shadow_when_legacy_is_active(monkeypatch):
    from app.domain.relative_strength import (
        BALANCED_RS_FORMULA_VERSION,
        LEGACY_RS_FORMULA_VERSION,
    )
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        module,
        "_active_formula_for_market",
        lambda _market: LEGACY_RS_FORMULA_VERSION,
    )

    assert module.guard_market_rs_result.run(
        {
            "status": "failed",
            "market": "US",
            "as_of_date": "2026-04-10",
            "formula_version": BALANCED_RS_FORMULA_VERSION,
            "reason_code": "benchmark_anchor_missing",
        },
        market="US",
        calculation_date="2026-04-10",
    ) == {
        "status": "skipped",
        "market": "US",
        "stage": "market_rs_shadow",
        "as_of_date": "2026-04-10",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": None,
    }


def test_guard_market_rs_accepts_exact_completed_balanced_run(monkeypatch):
    from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
    from app.tasks import daily_market_pipeline_tasks as module

    monkeypatch.setattr(
        module,
        "_active_formula_for_market",
        lambda _market: BALANCED_RS_FORMULA_VERSION,
    )

    assert module.guard_market_rs_result.run(
        {
            "status": "completed",
            "market": "US",
            "as_of_date": "2026-04-10",
            "formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 42,
            "eligible_symbol_count": 5000,
        },
        market="US",
        calculation_date="2026-04-10",
    ) == {
        "status": "ok",
        "market": "US",
        "stage": "market_rs",
        "as_of_date": "2026-04-10",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 42,
    }
