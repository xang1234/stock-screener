"""Tests for the static-site export CLI bootstrap behavior."""

from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.scripts.export_static_site as export_script
import app.tasks.fundamentals_tasks as fundamentals_tasks
import app.tasks.market_rs_tasks as market_rs_tasks
import app.tasks.universe_tasks as universe_tasks
from app.domain.markets import market_registry
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.interfaces.tasks import feature_store_tasks
from app.services.group_rank_history_backfill_service import (
    DEFAULT_CALENDAR_DAY_GROUP_RANK_HISTORY_LOOKBACK_DAYS,
    GroupRankHistoryBackfillResult,
    GroupRankHistoryBackfillStatus,
)

_REAL_ENSURE_GROUP_RANK_HISTORY = export_script._ensure_group_rank_history


def test_direct_module_execution_binds_breadth_wrapper_before_main():
    module_path = Path(export_script.__file__)

    def stop_at_main(frame, event, _arg):
        if (
            event == "call"
            and frame.f_code.co_name == "main"
            and Path(frame.f_code.co_filename) == module_path
        ):
            assert "_ensure_breadth_history" in frame.f_globals
            raise SystemExit(0)

    previous_profile = sys.getprofile()
    sys.setprofile(stop_at_main)
    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(module_path), run_name="__main__")
    finally:
        sys.setprofile(previous_profile)

    assert exc_info.value.code == 0


def _backfill_result(
    *,
    status: GroupRankHistoryBackfillStatus,
    market: str,
    as_of_date: date,
    error: str | None = None,
) -> GroupRankHistoryBackfillResult:
    return GroupRankHistoryBackfillResult(
        status=status,
        market=market,
        as_of_date=as_of_date,
        lookback_start_date=(
            as_of_date
            - timedelta(days=DEFAULT_CALENDAR_DAY_GROUP_RANK_HISTORY_LOOKBACK_DAYS)
        ),
        errors=1 if status is GroupRankHistoryBackfillStatus.ERRORED else 0,
        error=error,
    )


def _stub_ready_group_rank_history(monkeypatch) -> None:
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market, formula_version: _backfill_result(
            status=GroupRankHistoryBackfillStatus.SKIPPED,
            market=market,
            as_of_date=as_of_date,
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: {"status": "skipped"},
    )


def test_static_export_markets_match_market_registry():
    assert export_script.STATIC_EXPORT_MARKETS == market_registry.supported_market_codes()


def _stub_static_market_exposure(monkeypatch):
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda *, as_of_date, market: {
            "market": market,
            "date": as_of_date.isoformat(),
            "exposure_score": 50.0,
        },
    )


@pytest.fixture(autouse=True)
def _stub_static_breadth_history(monkeypatch):
    monkeypatch.setattr(
        export_script,
        "_ensure_breadth_history",
        lambda *, as_of_date, market=export_script.STATIC_DEFAULT_MARKET, **_kwargs: {
            "status": "completed",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
        },
    )


def test_static_breadth_history_stub_preserves_default_market():
    result = export_script._ensure_breadth_history(as_of_date=date(2026, 7, 31))

    assert result["market"] == export_script.STATIC_DEFAULT_MARKET


def test_ensure_group_rank_history_uses_static_bootstrap_coordinator(
    monkeypatch,
):
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        _REAL_ENSURE_GROUP_RANK_HISTORY,
    )
    as_of_date = date(2026, 7, 24)
    static_coordinator = object()
    expected_calendar_service = object()
    expected = _backfill_result(
        status=GroupRankHistoryBackfillStatus.COMPLETED,
        market="US",
        as_of_date=as_of_date,
    )

    def build_static_coordinator(*, calendar_service):
        assert calendar_service is expected_calendar_service
        return static_coordinator

    monkeypatch.setattr(
        export_script,
        "build_static_group_snapshot_coordinator",
        build_static_coordinator,
        raising=False,
    )
    monkeypatch.setattr(
        export_script,
        "get_market_calendar_service",
        lambda: expected_calendar_service,
    )
    class RecordingBackfillService:
        def __init__(
            self,
            *,
            session_factory,
            calendar_service,
            group_snapshot_coordinator,
        ):
            assert session_factory is export_script.SessionLocal
            assert calendar_service is expected_calendar_service
            assert group_snapshot_coordinator is static_coordinator

        def backfill(self, *, as_of_date, market, formula_version):
            assert as_of_date == date(2026, 7, 24)
            assert market == "US"
            assert formula_version == BALANCED_RS_FORMULA_VERSION
            return expected

    monkeypatch.setattr(
        export_script,
        "GroupRankHistoryBackfillService",
        RecordingBackfillService,
    )

    result = export_script._ensure_group_rank_history(
        as_of_date=as_of_date,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert result is expected


def test_ensure_group_rank_history_preserves_runtime_coordinator_for_legacy(
    monkeypatch,
):
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        _REAL_ENSURE_GROUP_RANK_HISTORY,
    )
    as_of_date = date(2026, 7, 24)
    runtime_coordinator = object()
    expected_calendar_service = object()
    expected = _backfill_result(
        status=GroupRankHistoryBackfillStatus.COMPLETED,
        market="US",
        as_of_date=as_of_date,
    )

    monkeypatch.setattr(
        export_script,
        "build_static_group_snapshot_coordinator",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy history must use the runtime coordinator")
        ),
    )
    monkeypatch.setattr(
        export_script,
        "get_group_rank_snapshot_coordinator",
        lambda: runtime_coordinator,
        raising=False,
    )
    monkeypatch.setattr(
        export_script,
        "get_market_calendar_service",
        lambda: expected_calendar_service,
    )

    class RecordingBackfillService:
        def __init__(
            self,
            *,
            session_factory,
            calendar_service,
            group_snapshot_coordinator,
        ):
            assert session_factory is export_script.SessionLocal
            assert calendar_service is expected_calendar_service
            assert group_snapshot_coordinator is runtime_coordinator

        def backfill(self, *, as_of_date, market, formula_version):
            assert as_of_date == date(2026, 7, 24)
            assert market == "US"
            assert formula_version == LEGACY_RS_FORMULA_VERSION
            return expected

    monkeypatch.setattr(
        export_script,
        "GroupRankHistoryBackfillService",
        RecordingBackfillService,
    )

    result = export_script._ensure_group_rank_history(
        as_of_date=as_of_date,
        market="US",
        formula_version=LEGACY_RS_FORMULA_VERSION,
    )

    assert result is expected


@pytest.fixture(autouse=True)
def _stub_balanced_static_rs_preparation(monkeypatch):
    """Keep legacy refresh tests focused; parity coverage restores the real helper."""
    real_helper = getattr(export_script, "_prepare_balanced_static_rs", None)
    monkeypatch.setattr(
        export_script,
        "_prepare_balanced_static_rs",
        lambda *, market, as_of_date: {
            "status": "completed",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 1,
        },
        raising=False,
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market, formula_version: _backfill_result(
            status=GroupRankHistoryBackfillStatus.SKIPPED,
            market=market,
            as_of_date=as_of_date,
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: {"status": "skipped"},
    )
    return real_helper


def _restore_real_balanced_static_rs(monkeypatch, real_helper) -> None:
    if real_helper is not None:
        monkeypatch.setattr(export_script, "_prepare_balanced_static_rs", real_helper)


def _stub_available_static_rs_benchmark(
    monkeypatch,
    *,
    benchmark_symbol: str,
    candidate_symbols: tuple[str, ...] | None = None,
) -> None:
    candidates = candidate_symbols or (benchmark_symbol,)

    class FakeBenchmarkCache:
        def resolve_benchmark_bundle(self, **_kwargs):
            return SimpleNamespace(
                bundle=SimpleNamespace(
                    benchmark_symbol=benchmark_symbol,
                    candidate_symbols=candidates,
                ),
                error_payload=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("available benchmark must not build an error")
                ),
            )

    monkeypatch.setattr(export_script, "get_benchmark_cache", lambda: FakeBenchmarkCache())


def _stub_market_rs_snapshot_task(monkeypatch, run) -> None:
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(run=run),
    )


def _stub_balanced_rs_formula_activation(monkeypatch, events: list[str]) -> None:
    @contextmanager
    def fake_session():
        yield SimpleNamespace(commit=lambda: events.append("formula_commit"))

    class FakeMarketRsRepository:
        def activate_formula(self, _db, *, market, formula_version):
            events.append(f"activate:{market}:{formula_version}")

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "MarketRsRunRepository",
        FakeMarketRsRepository,
        raising=False,
    )


def test_run_daily_refresh_activates_balanced_rs_before_static_consumers(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    events: list[str] = []

    @contextmanager
    def fake_session():
        yield SimpleNamespace(commit=lambda: events.append("formula_commit"))

    class FakeMarketRsRepository:
        def activate_formula(self, _db, *, market, formula_version):
            assert market == "US"
            assert formula_version == BALANCED_RS_FORMULA_VERSION
            events.append("formula_activate")

    _restore_real_balanced_static_rs(monkeypatch, _stub_balanced_static_rs_preparation)
    _stub_available_static_rs_benchmark(monkeypatch, benchmark_symbol="SPY")
    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "MarketRsRunRepository",
        FakeMarketRsRepository,
        raising=False,
    )
    _stub_market_rs_snapshot_task(
        monkeypatch,
        lambda **kwargs: events.append("market_rs_snapshot")
        or {
            "status": "completed",
            "market": kwargs["market"],
            "as_of_date": kwargs["calculation_date"],
            "formula_version": kwargs["formula_version"],
            "market_rs_run_id": 42,
            "eligible_symbol_count": 500,
        },
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 17),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda **_kwargs: events.append("prices") or {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda **_kwargs: events.append("exposure") or {"exposure_score": 50.0},
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **_kwargs: events.append("feature_snapshot")
            or {"status": "published", "run_id": 77}
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market, formula_version: events.append("group_history")
        or _backfill_result(
            status=GroupRankHistoryBackfillStatus.COMPLETED,
            market=market,
            as_of_date=as_of_date,
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda _db, csv_path=None: 1,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(
        market="US",
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
    )

    assert warnings == []
    assert events == [
        "prices",
        "market_rs_snapshot",
        "formula_activate",
        "formula_commit",
        "exposure",
        "feature_snapshot",
        "group_history",
    ]
    assert results["market_rs"]["US"] == {
        "status": "completed",
        "market": "US",
        "as_of_date": "2026-07-17",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 42,
        "eligible_symbol_count": 500,
    }


def test_prepare_balanced_static_rs_force_hydrates_required_benchmark_anchor(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    calls: list[dict[str, object]] = []
    events: list[str] = []

    class FakeBenchmarkCache:
        def resolve_benchmark_bundle(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                bundle=SimpleNamespace(benchmark_symbol="ES3.SI"),
                error_payload=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("successful benchmark hydration must not build an error")
                ),
            )

    @contextmanager
    def fake_session():
        yield SimpleNamespace(commit=lambda: events.append("formula_commit"))

    class FakeMarketRsRepository:
        def activate_formula(self, _db, *, market, formula_version):
            events.append(f"activate:{market}:{formula_version}")

    real_helper = _stub_balanced_static_rs_preparation
    if real_helper is not None:
        monkeypatch.setattr(export_script, "_prepare_balanced_static_rs", real_helper)
    monkeypatch.setattr(export_script, "get_benchmark_cache", lambda: FakeBenchmarkCache())
    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "MarketRsRunRepository",
        FakeMarketRsRepository,
        raising=False,
    )
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: {
                "status": "completed",
                "market": kwargs["market"],
                "as_of_date": kwargs["calculation_date"],
                "formula_version": kwargs["formula_version"],
                "market_rs_run_id": 42,
            }
        ),
    )

    result = export_script._prepare_balanced_static_rs(
        market="SG",
        as_of_date=date(2026, 7, 24),
    )

    assert calls == [
        {
            "market": "SG",
            "period": "2y",
            "force_refresh": True,
            "fallback_policy": export_script.BenchmarkFallbackPolicy.ALLOW,
            "required_as_of_date": date(2026, 7, 24),
        }
    ]
    assert events == [
        f"activate:SG:{BALANCED_RS_FORMULA_VERSION}",
        "formula_commit",
    ]
    assert result["market_rs_run_id"] == 42


def test_prepare_balanced_static_rs_retries_stale_benchmark_anchor_before_market_rs(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    calls: list[dict[str, object]] = []
    events: list[str] = []

    class FakeStaleResolution:
        bundle = None

        @staticmethod
        def error_payload(*, market, as_of_date):
            return {
                "error": "benchmark_not_current",
                "market": market,
                "date": as_of_date.isoformat(),
            }

    class FakeBenchmarkCache:
        def resolve_benchmark_bundle(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return FakeStaleResolution()
            return SimpleNamespace(
                bundle=SimpleNamespace(benchmark_symbol="ES3.SI"),
                error_payload=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("successful benchmark retry must not build an error")
                ),
            )

    @contextmanager
    def fake_session():
        yield SimpleNamespace(commit=lambda: events.append("formula_commit"))

    class FakeMarketRsRepository:
        def activate_formula(self, _db, *, market, formula_version):
            events.append(f"activate:{market}:{formula_version}")

    real_helper = _stub_balanced_static_rs_preparation
    if real_helper is not None:
        monkeypatch.setattr(export_script, "_prepare_balanced_static_rs", real_helper)
    monkeypatch.setattr(export_script, "get_benchmark_cache", lambda: FakeBenchmarkCache())
    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "MarketRsRunRepository",
        FakeMarketRsRepository,
        raising=False,
    )
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: events.append("market_rs_snapshot")
            or {
                "status": "completed",
                "market": kwargs["market"],
                "as_of_date": kwargs["calculation_date"],
                "formula_version": kwargs["formula_version"],
                "market_rs_run_id": 42,
            }
        ),
    )

    result = export_script._prepare_balanced_static_rs(
        market="SG",
        as_of_date=date(2026, 7, 24),
    )

    assert calls == [
        {
            "market": "SG",
            "period": "2y",
            "force_refresh": True,
            "fallback_policy": export_script.BenchmarkFallbackPolicy.ALLOW,
            "required_as_of_date": date(2026, 7, 24),
        },
        {
            "market": "SG",
            "period": "2y",
            "force_refresh": True,
            "fallback_policy": export_script.BenchmarkFallbackPolicy.ALLOW,
            "required_as_of_date": date(2026, 7, 24),
        },
    ]
    assert events == [
        "market_rs_snapshot",
        f"activate:SG:{BALANCED_RS_FORMULA_VERSION}",
        "formula_commit",
    ]
    assert result["market_rs_run_id"] == 42


def test_prepare_balanced_static_rs_uses_cached_resolution_after_forced_hydration_misses(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    calls: list[dict[str, object]] = []
    events: list[str] = []

    class FakeForcedMiss:
        bundle = None

        @staticmethod
        def error_payload(*, market, as_of_date):
            return {
                "error": "benchmark_not_current",
                "market": market,
                "date": as_of_date.isoformat(),
            }

    class FakeBenchmarkCache:
        def resolve_benchmark_bundle(self, **kwargs):
            calls.append(kwargs)
            if kwargs["force_refresh"]:
                return FakeForcedMiss()
            return SimpleNamespace(
                bundle=SimpleNamespace(
                    benchmark_symbol="ES3.SI",
                    candidate_symbols=("^STI", "ES3.SI"),
                ),
                error_payload=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("successful cached fallback must not build an error")
                ),
            )

    @contextmanager
    def fake_session():
        yield SimpleNamespace(commit=lambda: events.append("formula_commit"))

    class FakeMarketRsRepository:
        def activate_formula(self, _db, *, market, formula_version):
            events.append(f"activate:{market}:{formula_version}")

    real_helper = _stub_balanced_static_rs_preparation
    if real_helper is not None:
        monkeypatch.setattr(export_script, "_prepare_balanced_static_rs", real_helper)
    monkeypatch.setattr(export_script, "get_benchmark_cache", lambda: FakeBenchmarkCache())
    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "MarketRsRunRepository",
        FakeMarketRsRepository,
        raising=False,
    )
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: events.append("market_rs_snapshot")
            or {
                "status": "completed",
                "market": kwargs["market"],
                "as_of_date": kwargs["calculation_date"],
                "formula_version": kwargs["formula_version"],
                "market_rs_run_id": 42,
            }
        ),
    )

    result = export_script._prepare_balanced_static_rs(
        market="SG",
        as_of_date=date(2026, 7, 24),
    )

    assert [call["force_refresh"] for call in calls] == [True, True, False]
    assert events == [
        "market_rs_snapshot",
        f"activate:SG:{BALANCED_RS_FORMULA_VERSION}",
        "formula_commit",
    ]
    assert result["market_rs_run_id"] == 42


def test_prepare_balanced_static_rs_reports_resolution_exception_without_market_rs(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    calls: list[dict[str, object]] = []

    class FakeBenchmarkCache:
        def resolve_benchmark_bundle(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("provider unavailable")

    real_helper = _stub_balanced_static_rs_preparation
    if real_helper is not None:
        monkeypatch.setattr(export_script, "_prepare_balanced_static_rs", real_helper)
    monkeypatch.setattr(export_script, "get_benchmark_cache", lambda: FakeBenchmarkCache())
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(
            run=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("balanced RS must not run after benchmark resolution errors")
            )
        ),
    )

    result = export_script._prepare_balanced_static_rs(
        market="SG",
        as_of_date=date(2026, 7, 24),
    )

    assert [call["force_refresh"] for call in calls] == [True, True, False]
    assert result == {
        "status": "failed",
        "market": "SG",
        "as_of_date": "2026-07-24",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "reason_code": "benchmark_adjusted_anchor_missing",
        "diagnostics": {
            "error": "benchmark_resolution_exception",
            "market": "SG",
            "date": "2026-07-24",
            "error_type": "RuntimeError",
            "error_message": "provider unavailable",
        },
        "market_rs_run_id": None,
    }


def test_prepare_balanced_static_rs_hydrates_fallback_after_anchor_gap(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    hydrated_symbols: list[dict[str, object]] = []
    market_rs_calls: list[dict[str, object]] = []
    events: list[str] = []

    class FakeBenchmarkCache:
        def resolve_benchmark_bundle(self, **_kwargs):
            return SimpleNamespace(
                bundle=SimpleNamespace(
                    benchmark_symbol="^STI",
                    candidate_symbols=("^STI", "ES3.SI"),
                ),
                error_payload=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("successful benchmark hydration must not build an error")
                ),
            )

        def fetch_and_cache_benchmark(self, **kwargs):
            hydrated_symbols.append(kwargs)
            return SimpleNamespace()

    @contextmanager
    def fake_session():
        yield SimpleNamespace(commit=lambda: events.append("formula_commit"))

    class FakeMarketRsRepository:
        def activate_formula(self, _db, *, market, formula_version):
            events.append(f"activate:{market}:{formula_version}")

    def fake_market_rs_run(**kwargs):
        market_rs_calls.append(kwargs)
        if len(market_rs_calls) == 1:
            return {
                "status": "failed",
                "market": kwargs["market"],
                "as_of_date": kwargs["calculation_date"],
                "formula_version": kwargs["formula_version"],
                "reason_code": "benchmark_adjusted_anchor_missing",
                "diagnostics": {
                    "missing_anchor_dates": {
                        "^STI": ["2025-07-24"],
                        "ES3.SI": ["2025-07-24"],
                    }
                },
            }
        return {
            "status": "completed",
            "market": kwargs["market"],
            "as_of_date": kwargs["calculation_date"],
            "formula_version": kwargs["formula_version"],
            "market_rs_run_id": 42,
        }

    real_helper = _stub_balanced_static_rs_preparation
    if real_helper is not None:
        monkeypatch.setattr(export_script, "_prepare_balanced_static_rs", real_helper)
    monkeypatch.setattr(export_script, "get_benchmark_cache", lambda: FakeBenchmarkCache())
    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "MarketRsRunRepository",
        FakeMarketRsRepository,
        raising=False,
    )
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(run=fake_market_rs_run),
    )

    result = export_script._prepare_balanced_static_rs(
        market="SG",
        as_of_date=date(2026, 7, 24),
    )

    assert hydrated_symbols == [
        {
            "benchmark_symbol": "ES3.SI",
            "market": "SG",
            "period": "2y",
            "required_as_of_date": date(2026, 7, 24),
        }
    ]
    assert len(market_rs_calls) == 2
    assert events == [
        f"activate:SG:{BALANCED_RS_FORMULA_VERSION}",
        "formula_commit",
    ]
    assert result["market_rs_run_id"] == 42


def test_prepare_balanced_static_rs_reports_price_coverage_gap_without_activation(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    events: list[str] = []

    _restore_real_balanced_static_rs(monkeypatch, _stub_balanced_static_rs_preparation)
    _stub_available_static_rs_benchmark(monkeypatch, benchmark_symbol="^GDAXI")
    _stub_balanced_rs_formula_activation(monkeypatch, events)
    _stub_market_rs_snapshot_task(
        monkeypatch,
        lambda **kwargs: {
            "status": "failed",
            "market": kwargs["market"],
            "as_of_date": kwargs["calculation_date"],
            "formula_version": kwargs["formula_version"],
            "reason_code": "current_adjusted_price_coverage_below_threshold",
            "diagnostics": {
                "current_price_coverage": 0.8434065934065934,
                "minimum_current_price_coverage": 0.88,
                "current_prices_available": 1228,
                "expected_symbol_count": 1456,
            },
        },
    )

    result = export_script._prepare_balanced_static_rs(
        market="DE",
        as_of_date=date(2026, 7, 27),
    )

    assert events == []
    assert result == {
        "status": "failed",
        "market": "DE",
        "as_of_date": "2026-07-27",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "reason_code": "current_adjusted_price_coverage_below_threshold",
        "diagnostics": {
            "current_price_coverage": 0.8434065934065934,
            "minimum_current_price_coverage": 0.88,
            "current_prices_available": 1228,
            "expected_symbol_count": 1456,
        },
        "market_rs_run_id": None,
    }


def test_prepare_balanced_static_rs_still_raises_unexpected_failed_result(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    _restore_real_balanced_static_rs(monkeypatch, _stub_balanced_static_rs_preparation)
    _stub_available_static_rs_benchmark(monkeypatch, benchmark_symbol="^GDAXI")
    _stub_market_rs_snapshot_task(
        monkeypatch,
        lambda **kwargs: {
            "status": "failed",
            "market": kwargs["market"],
            "as_of_date": kwargs["calculation_date"],
            "formula_version": kwargs["formula_version"],
            "reason_code": "calculation_failed",
            "diagnostics": {"error": "database invariant failed"},
        },
    )

    with pytest.raises(RuntimeError, match="Balanced Market RS preparation failed"):
        export_script._prepare_balanced_static_rs(
            market="DE",
            as_of_date=date(2026, 7, 27),
        )


def test_prepare_balanced_static_rs_reports_missing_required_benchmark_anchor(
    monkeypatch,
    _stub_balanced_static_rs_preparation,
):
    class FakeResolution:
        bundle = None

        @staticmethod
        def error_payload(*, market, as_of_date):
            return {
                "error": "benchmark_not_current",
                "market": market,
                "date": as_of_date.isoformat(),
                "benchmark_candidates": [
                    {
                        "symbol": "^STI",
                        "role": "primary",
                        "source": "fetch",
                        "status": "stale_required_date",
                        "latest_date": "2026-07-23",
                    },
                    {
                        "symbol": "ES3.SI",
                        "role": "fallback",
                        "source": "fetch",
                        "status": "stale_required_date",
                        "latest_date": "2026-07-23",
                    },
                ],
            }

    class FakeBenchmarkCache:
        def resolve_benchmark_bundle(self, **_kwargs):
            return FakeResolution()

    real_helper = _stub_balanced_static_rs_preparation
    if real_helper is not None:
        monkeypatch.setattr(export_script, "_prepare_balanced_static_rs", real_helper)
    monkeypatch.setattr(export_script, "get_benchmark_cache", lambda: FakeBenchmarkCache())
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(
            run=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("balanced RS must not run without a current benchmark anchor")
            )
        ),
    )

    result = export_script._prepare_balanced_static_rs(
        market="SG",
        as_of_date=date(2026, 7, 24),
    )

    assert result == {
        "status": "failed",
        "market": "SG",
        "as_of_date": "2026-07-24",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "reason_code": "benchmark_adjusted_anchor_missing",
        "diagnostics": {
            "error": "benchmark_not_current",
            "market": "SG",
            "date": "2026-07-24",
            "benchmark_candidates": [
                {
                    "symbol": "^STI",
                    "role": "primary",
                    "source": "fetch",
                    "status": "stale_required_date",
                    "latest_date": "2026-07-23",
                },
                {
                    "symbol": "ES3.SI",
                    "role": "fallback",
                    "source": "fetch",
                    "status": "stale_required_date",
                    "latest_date": "2026-07-23",
                },
            ],
        },
        "market_rs_run_id": None,
    }


def test_run_daily_refresh_skips_snapshot_when_market_rs_benchmark_anchor_missing(
    monkeypatch,
):
    calls: list[str] = []

    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 24),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append(f"price:{market}") or {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script,
        "_prepare_static_rs_formula",
        lambda *, market, as_of_date, formula_version: calls.append(f"market-rs:{market}")
        or {
            "status": "failed",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": formula_version,
            "reason_code": "benchmark_adjusted_anchor_missing",
            "diagnostics": {
                "error": "benchmark_not_current",
                "benchmark_candidates": [
                    {
                        "symbol": "^STI",
                        "role": "primary",
                        "source": "fetch",
                        "status": "stale_required_date",
                        "latest_date": "2026-07-23",
                    }
                ],
            },
            "market_rs_run_id": None,
        },
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("group-rank history must not run without market RS")
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("market exposure must not run without market RS")
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("feature snapshot must not run without market RS")
            )
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata enrichment must not run without a snapshot")
        ),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda _db, csv_path=None: 1,
    )

    results, warnings = export_script._run_daily_refresh(
        market="SG",
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
    )

    assert calls == ["price:SG", "market-rs:SG"]
    assert results["feature_snapshots"]["SG"] == {
        "status": "skipped",
        "reason": "market_rs_not_ready",
        "market": "SG",
        "as_of_date": "2026-07-24",
        "failure_diagnostics": {
            "reason_code": "benchmark_adjusted_anchor_missing",
            "diagnostics": {
                "error": "benchmark_not_current",
                "benchmark_candidates": [
                    {
                        "symbol": "^STI",
                        "role": "primary",
                        "source": "fetch",
                        "status": "stale_required_date",
                        "latest_date": "2026-07-23",
                    }
                ],
            },
        },
        "warnings": [
            "Static export market SG Market RS not ready for 2026-07-24: "
            "benchmark_adjusted_anchor_missing."
        ],
    }
    assert results["group_rank_history_backfill"]["SG"]["status"] == "skipped"
    assert results["group_rank_history_backfill"]["SG"]["reason"] == "market_rs_not_ready"
    assert "error" not in results["group_rank_history_backfill"]["SG"]
    assert results["market_exposure"]["SG"]["status"] == "skipped"
    assert results["ibd_metadata_refresh"]["SG"]["reason"] == "snapshot_not_ready"
    assert (
        "Static export market SG Market RS not ready for 2026-07-24: "
        "benchmark_adjusted_anchor_missing."
    ) in warnings


def test_run_daily_refresh_continues_eligible_market_when_peer_market_rs_is_not_ready(
    monkeypatch,
):
    calls: list[str] = []

    monkeypatch.setattr(export_script, "STATIC_EXPORT_MARKETS", ("DE", "HK"))
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 24),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append(f"price:{market}")
        or {"status": "completed"},
    )

    def prepare_market_rs(*, market, as_of_date, formula_version):
        calls.append(f"market-rs:{market}")
        if market == "DE":
            return {
                "status": "failed",
                "market": market,
                "as_of_date": as_of_date.isoformat(),
                "formula_version": formula_version,
                "reason_code": "current_adjusted_price_coverage_below_threshold",
                "diagnostics": {
                    "current_price_coverage": 0.84,
                    "minimum_current_price_coverage": 0.88,
                },
                "market_rs_run_id": None,
            }
        return {
            "status": "completed",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": formula_version,
            "market_rs_run_id": 42,
        }

    monkeypatch.setattr(export_script, "_prepare_static_rs_formula", prepare_market_rs)

    def compute_exposure(*, as_of_date, market):
        if market == "DE":
            raise AssertionError("DE exposure must be skipped when Market RS is not ready")
        calls.append(f"exposure:{market}")
        return {"status": "completed"}

    monkeypatch.setattr(export_script, "_compute_static_market_exposure", compute_exposure)
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append(f"snapshot:{kwargs['market']}")
            or {"status": "published", "run_id": 77, "market": kwargs["market"]}
        ),
    )

    def ensure_group_history(*, as_of_date, market, formula_version):
        if market == "DE":
            raise AssertionError("DE group-rank history must be skipped")
        calls.append(f"group:{market}")
        return _backfill_result(
            status=GroupRankHistoryBackfillStatus.COMPLETED,
            market=market,
            as_of_date=as_of_date,
        )

    monkeypatch.setattr(export_script, "_ensure_group_rank_history", ensure_group_history)
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **kwargs: calls.append(f"metadata:{kwargs['feature_run_id']}")
        or {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda _db, csv_path=None: 1,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
    )

    assert calls == [
        "price:DE",
        "price:HK",
        "market-rs:DE",
        "market-rs:HK",
        "exposure:HK",
        "snapshot:HK",
        "group:HK",
        "metadata:77",
    ]
    assert results["feature_snapshots"]["DE"]["reason"] == "market_rs_not_ready"
    assert results["feature_snapshots"]["HK"] == {
        "status": "published",
        "run_id": 77,
        "market": "HK",
    }
    assert results["group_rank_history_backfill"]["DE"]["reason"] == "market_rs_not_ready"
    assert results["group_rank_history_backfill"]["HK"]["status"] == "completed"
    assert (
        "Static export market DE Market RS not ready for 2026-07-24: "
        "current_adjusted_price_coverage_below_threshold."
    ) in warnings


def test_run_daily_refresh_raises_hard_market_rs_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 24),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script,
        "_prepare_static_rs_formula",
        lambda *, market, as_of_date, formula_version: {
            "status": "failed",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": formula_version,
            "reason_code": "calculation_failed",
            "diagnostics": {"error": "database invariant failed"},
        },
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("hard Market RS failures must not fall through to exposure")
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("hard Market RS failures must not build feature snapshots")
            )
        ),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda _db, csv_path=None: 1,
    )

    with pytest.raises(RuntimeError, match="Static Market RS failed hard for DE"):
        export_script._run_daily_refresh(
            market="DE",
            skip_universe_refresh=True,
            skip_fundamentals_refresh=True,
        )


def test_run_daily_refresh_reports_every_hard_market_rs_failure(
    monkeypatch,
):
    monkeypatch.setattr(export_script, "STATIC_EXPORT_MARKETS", ("DE", "HK"))
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 24),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script,
        "_prepare_static_rs_formula",
        lambda *, market, as_of_date, formula_version: {
            "status": "failed",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": formula_version,
            "reason_code": "calculation_failed",
            "diagnostics": {"error": f"{market} invariant failed"},
        },
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("hard Market RS failures must not fall through to exposure")
        ),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda _db, csv_path=None: 1,
    )

    with pytest.raises(RuntimeError) as exc_info:
        export_script._run_daily_refresh(
            skip_universe_refresh=True,
            skip_fundamentals_refresh=True,
        )

    message = str(exc_info.value)
    assert "Static Market RS failed hard for DE on 2026-07-24" in message
    assert "Static Market RS failed hard for HK on 2026-07-24" in message


def test_run_daily_refresh_treats_explicit_legacy_rs_selection_as_ready(
    monkeypatch,
):
    calls: list[str] = []

    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 24),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append(f"price:{market}")
        or {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script,
        "_prepare_static_rs_formula",
        lambda *, market, as_of_date, formula_version: calls.append(f"market-rs:{market}")
        or {
            "status": "selected",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": formula_version,
            "market_rs_run_id": None,
        },
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market, formula_version: calls.append(f"group:{market}")
        or _backfill_result(
            status=GroupRankHistoryBackfillStatus.COMPLETED,
            market=market,
            as_of_date=as_of_date,
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda *, as_of_date, market: calls.append(f"exposure:{market}")
        or {"status": "completed"},
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append(f"snapshot:{kwargs['market']}")
            or {"status": "published", "run_id": 77}
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda _db, csv_path=None: 1,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(
        market="US",
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        rs_formula_version=LEGACY_RS_FORMULA_VERSION,
    )

    assert warnings == []
    assert calls == [
        "price:US",
        "market-rs:US",
        "exposure:US",
        "snapshot:US",
        "group:US",
    ]
    assert results["feature_snapshots"]["US"]["run_id"] == 77


def test_prepare_static_rs_formula_supports_explicit_legacy_rollback(monkeypatch):
    events: list[str] = []

    @contextmanager
    def fake_session():
        yield SimpleNamespace(commit=lambda: events.append("formula_commit"))

    class FakeMarketRsRepository:
        def activate_formula(self, _db, *, market, formula_version):
            events.append(f"activate:{market}:{formula_version}")

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "MarketRsRunRepository",
        FakeMarketRsRepository,
        raising=False,
    )
    monkeypatch.setattr(
        market_rs_tasks,
        "calculate_market_rs_snapshot",
        SimpleNamespace(
            run=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("legacy static rollback must not calculate balanced RS")
            )
        ),
    )

    result = export_script._prepare_static_rs_formula(
        market="US",
        as_of_date=date(2026, 7, 17),
        formula_version=LEGACY_RS_FORMULA_VERSION,
    )

    assert events == [
        f"activate:US:{LEGACY_RS_FORMULA_VERSION}",
        "formula_commit",
    ]
    assert result == {
        "status": "selected",
        "market": "US",
        "as_of_date": "2026-07-17",
        "formula_version": LEGACY_RS_FORMULA_VERSION,
        "market_rs_run_id": None,
    }


def test_run_daily_refresh_bootstraps_universe_before_other_tasks(monkeypatch):
    calls: list[str] = []

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append("feature_snapshot")
            or {"run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda *, pointer_key, run_id: calls.append(f"pointer:{pointer_key}:{run_id}"),
    )
    _stub_ready_group_rank_history(monkeypatch)
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    expected_markets = list(export_script.STATIC_EXPORT_MARKETS)

    assert warnings == []
    assert calls == [
        "universe_refresh",
        "fundamentals_refresh",
        *(["price_refresh"] * len(expected_markets)),
        *(["feature_snapshot"] * len(expected_markets)),
        "pointer:latest_published:77",
    ]
    assert results["universe_refresh"]["task"] == "universe_refresh"
    assert results["ibd_seed_refresh"]["loaded"] == 10105
    assert set(results["feature_snapshots"]) == set(expected_markets)
    for market in expected_markets:
        assert results["feature_snapshots"][market]["kwargs"] == {
            "as_of_date_str": "2026-04-02",
            "static_daily_mode": True,
            "universe_name": f"market:{market.lower()}",
            "market": market,
            "publish_pointer_key": f"latest_published_market:{market}",
            "ignore_runtime_market_gate": True,
            "rs_formula_version_override": BALANCED_RS_FORMULA_VERSION,
            "skip_ibd_metadata_enrichment": True,
        }
    assert set(results["price_refresh"]) == set(expected_markets)
    assert results["default_market_pointer"] == {
        "market": "US",
        "pointer_key": "latest_published",
        "run_id": 77,
    }


def test_run_daily_refresh_uses_resolved_tracked_ibd_csv_path(monkeypatch, tmp_path):
    calls: list[str] = []
    resolved_csv = tmp_path / "data" / "IBD_industry_group.csv"
    resolved_csv.parent.mkdir(parents=True, exist_ok=True)
    resolved_csv.write_text("AAPL,Software\n", encoding="utf-8")

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    load_calls: list[tuple[object, Path | None]] = []

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append("feature_snapshot")
            or {"run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(export_script, "_tracked_ibd_csv_path", lambda: resolved_csv)
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: load_calls.append((db, csv_path)) or 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_ready_group_rank_history(monkeypatch)
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert warnings == []
    assert load_calls
    assert load_calls[0][1] == resolved_csv
    assert results["ibd_seed_refresh"] == {
        "csv_path": str(resolved_csv),
        "loaded": 10105,
    }


def test_run_daily_refresh_computes_market_exposure_before_snapshot(monkeypatch):
    calls: list[str] = []

    @contextmanager
    def fake_session():
        yield object()

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 6, 25),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append(f"price:{market}:{as_of_date.isoformat()}")
        or {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda *, as_of_date, market: calls.append(f"exposure:{market}:{as_of_date.isoformat()}")
        or {"market": market, "date": as_of_date.isoformat(), "exposure_score": 42.0},
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append(f"snapshot:{kwargs['market']}:{kwargs['as_of_date_str']}")
            or {"run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    _stub_ready_group_rank_history(monkeypatch)
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        market="US",
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
    )

    assert warnings == []
    assert calls == [
        "price:US:2026-06-25",
        "exposure:US:2026-06-25",
        "snapshot:US:2026-06-25",
    ]
    assert results["market_exposure"] == {
        "US": {"market": "US", "date": "2026-06-25", "exposure_score": 42.0}
    }


def test_run_daily_refresh_skips_unsupported_breadth_and_builds_snapshot(
    monkeypatch,
):
    calls: list[str] = []

    @contextmanager
    def fake_session():
        yield object()

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 6, 25),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append(
            f"price:{market}:{as_of_date.isoformat()}"
        )
        or {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_breadth_history",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported market must not calculate breadth")
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported market must not calculate exposure")
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append(
                f"snapshot:{kwargs['market']}:{kwargs['as_of_date_str']}"
            )
            or {"status": "published", "run_id": 77}
        ),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    _stub_ready_group_rank_history(monkeypatch)
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(
        market="SG",
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
    )

    assert warnings == []
    assert calls == ["price:SG:2026-06-25", "snapshot:SG:2026-06-25"]
    assert results["breadth_history"]["SG"] == {
        "status": "skipped",
        "reason": "market_breadth_unsupported",
        "market": "SG",
        "as_of_date": "2026-06-25",
    }
    assert results["market_exposure"]["SG"] == {
        "status": "skipped",
        "reason": "market_breadth_unsupported",
        "market": "SG",
        "date": "2026-06-25",
    }
    assert results["feature_snapshots"]["SG"] == {
        "status": "published",
        "run_id": 77,
    }


def test_run_daily_refresh_skips_snapshot_when_market_exposure_errors(monkeypatch):
    calls: list[str] = []

    @contextmanager
    def fake_session():
        yield object()

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 6, 25),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append(f"price:{market}:{as_of_date.isoformat()}")
        or {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_compute_static_market_exposure",
        lambda *, as_of_date, market: calls.append(f"exposure:{market}:{as_of_date.isoformat()}")
        or {"market": market, "date": as_of_date.isoformat(), "error": "no_benchmark_data"},
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("snapshot should not publish after exposure failure")
            )
        ),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda **kwargs: calls.append(
            f"group:{kwargs['market']}:{kwargs['as_of_date'].isoformat()}"
        ) or _backfill_result(
            status=GroupRankHistoryBackfillStatus.COMPLETED,
            market=kwargs["market"],
            as_of_date=kwargs["as_of_date"],
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata enrichment should not run without a snapshot")
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("default pointer should not update without a snapshot")
        ),
    )

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        market="US",
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
    )

    assert calls == [
        "price:US:2026-06-25",
        "exposure:US:2026-06-25",
    ]
    assert results["feature_snapshots"] == {
        "US": {
            "status": "skipped",
            "reason": "market_exposure_not_ready",
            "market": "US",
            "as_of_date": "2026-06-25",
            "failure_diagnostics": {
                "date": "2026-06-25",
                "error": "no_benchmark_data",
            },
            "warnings": [
                "Static export market US exposure not stored for 2026-06-25: no_benchmark_data."
            ],
        }
    }
    assert results["group_rank_history_backfill"]["US"] == {
        "status": "skipped",
        "market": "US",
        "as_of_date": "2026-06-25",
        "lookback_start_date": "2025-12-20",
        "missing_dates": 0,
        "processed": 0,
        "errors": 0,
        "reason": "snapshot_not_ready",
    }
    assert results["ibd_metadata_refresh"]["US"]["reason"] == "snapshot_not_ready"
    assert "Static export market US exposure not stored for 2026-06-25: no_benchmark_data." in warnings


def test_run_daily_refresh_can_hydrate_imported_snapshot_without_live_fundamentals(monkeypatch):
    calls: list[str] = []

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    @contextmanager
    def fake_session():
        yield "db-session"

    hydrate_calls: list[tuple[object, bool]] = []

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append("feature_snapshot")
            or {"run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            hydrate_all_published_snapshots=lambda db, allow_yahoo_hydration=False: hydrate_calls.append((db, allow_yahoo_hydration))
            or {"task": "fundamentals_hydrate"},
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda *, pointer_key, run_id: calls.append(f"pointer:{pointer_key}:{run_id}"),
    )
    _stub_ready_group_rank_history(monkeypatch)
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        build_mode=export_script.STATIC_BUILD_MODE_FULL,
        hydrate_published_snapshot=True,
    )

    assert warnings == []
    assert calls == [
        *(["price_refresh"] * len(export_script.STATIC_EXPORT_MARKETS)),
        *(["feature_snapshot"] * len(export_script.STATIC_EXPORT_MARKETS)),
        "pointer:latest_published:77",
    ]
    assert hydrate_calls == [("db-session", False)]
    assert "universe_refresh" not in results
    assert "fundamentals_refresh" not in results
    assert results["fundamentals_hydrate"]["task"] == "fundamentals_hydrate"


def test_run_daily_refresh_price_delta_mode_skips_snapshot_hydration(monkeypatch):
    calls: list[str] = []

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    @contextmanager
    def fake_session():
        yield "db-session"

    hydrate_calls: list[tuple[object, bool]] = []

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append("feature_snapshot")
            or {"run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            hydrate_all_published_snapshots=lambda db, allow_yahoo_hydration=False: hydrate_calls.append((db, allow_yahoo_hydration))
            or {"task": "fundamentals_hydrate"},
        ),
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_ready_group_rank_history(monkeypatch)
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        build_mode=export_script.STATIC_BUILD_MODE_PRICE_DELTA,
        hydrate_published_snapshot=True,
    )

    assert warnings == []
    assert hydrate_calls == []
    assert "fundamentals_hydrate" not in results


def test_run_daily_refresh_warns_when_default_market_run_id_is_missing(monkeypatch):
    calls: list[str] = []

    def build_snapshot(**kwargs):
        calls.append(kwargs["market"])
        if kwargs["market"] == export_script.STATIC_DEFAULT_MARKET:
            return {"status": "completed"}
        return {"run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda: {"task": "universe_refresh"}))
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda **_kwargs: calls.append("pointer"),
    )
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert calls == list(export_script.STATIC_EXPORT_MARKETS)
    assert "default_market_pointer" not in results
    assert warnings == ["No US feature snapshot produced a run id; 'latest_published' was not updated."]


def test_run_daily_refresh_does_not_repoint_default_pointer_for_unpublished_us_run(monkeypatch):
    pointer_calls: list[dict] = []

    def build_snapshot(**kwargs):
        if kwargs["market"] == export_script.STATIC_DEFAULT_MARKET:
            return {"status": "failed", "run_id": 91}
        return {"status": "published", "run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda: {"task": "universe_refresh"}))
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda **kwargs: pointer_calls.append(kwargs),
    )
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert pointer_calls == []
    assert "default_market_pointer" not in results
    assert warnings == ["US feature snapshot returned status 'failed'; 'latest_published' was not updated."]


def test_run_daily_refresh_disables_serialized_lock_during_export(monkeypatch):
    calls: list[tuple[str, bool, bool]] = []
    state = {"fetch_lock_disabled": False, "workload_disabled": False}
    events: list[str] = []

    def make_disable(name: str, state_key: str):
        @contextmanager
        def _ctx():
            events.append(f"enter:{name}")
            state[state_key] = True
            try:
                yield
            finally:
                state[state_key] = False
                events.append(f"exit:{name}")

        return _ctx

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, state["fetch_lock_disabled"], state["workload_disabled"]))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(
        export_script,
        "disable_serialized_data_fetch_lock",
        make_disable("fetch", "fetch_lock_disabled"),
    )
    monkeypatch.setattr(
        export_script,
        "disable_serialized_market_workload",
        make_disable("workload", "workload_disabled"),
    )
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert events == ["enter:fetch", "enter:workload", "exit:workload", "exit:fetch"]
    assert all(fetch_disabled and workload_disabled for _, fetch_disabled, workload_disabled in calls)


def test_run_daily_refresh_limits_work_to_selected_market(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, kwargs))
            return {"task": name, "kwargs": kwargs, "run_id": 77}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        market="HK",
    )

    assert warnings == []
    assert results["price_refresh"]["market"] == "HK"
    assert set(results["feature_snapshots"]) == {"HK"}
    assert calls == [
        ("universe_refresh", {"market": "HK"}),
        ("fundamentals_refresh", {"market": "HK"}),
        (
            "feature_snapshot",
            {
                "as_of_date_str": "2026-04-02",
                "static_daily_mode": True,
                "universe_name": "market:hk",
                "market": "HK",
                "publish_pointer_key": "latest_published_market:HK",
                "ignore_runtime_market_gate": True,
                "rs_formula_version_override": BALANCED_RS_FORMULA_VERSION,
                "skip_ibd_metadata_enrichment": True,
            },
        ),
    ]


def test_run_daily_refresh_uses_per_market_trading_date_for_in(monkeypatch):
    """Regression: ``build_daily_snapshot`` for IN must receive IN's latest
    completed trading date, not NYSE's. Before the per-market calendar fix,
    running ``--market IN`` on a day NSE was closed but NYSE was open caused
    the snapshot to be silently skipped with ``reason='not_trading_day'``."""

    snapshot_calls: list[dict] = []

    def build_snapshot_run(**kwargs):
        snapshot_calls.append(kwargs)
        return {"status": "published", "run_id": 91}

    # Simulate the failure scenario: NYSE traded on Apr 2 (US holiday for
    # IN), so IN's latest completed session is Apr 1. The previous bug used
    # Apr 2 for IN's snapshot, hitting the not_trading_day guard.
    market_dates = {"US": date(2026, 4, 2), "IN": date(2026, 4, 1)}

    monkeypatch.setattr(
        universe_tasks,
        "refresh_stock_universe",
        SimpleNamespace(run=lambda **_kwargs: {"task": "universe_refresh"}),
    )
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda **_kwargs: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot_run),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {
            "task": "price_refresh",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
        },
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda selected_market: market_dates[selected_market],
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    results, _warnings = export_script._run_daily_refresh(market="IN")  # noqa: SLF001

    assert results["price_refresh"]["as_of_date"] == "2026-04-01"
    assert snapshot_calls == [
        {
            "as_of_date_str": "2026-04-01",
            "static_daily_mode": True,
            "universe_name": "market:in",
            "market": "IN",
            "publish_pointer_key": "latest_published_market:IN",
            "ignore_runtime_market_gate": True,
            "rs_formula_version_override": BALANCED_RS_FORMULA_VERSION,
            "skip_ibd_metadata_enrichment": True,
        }
    ]


def test_run_daily_refresh_uses_static_daily_mode_and_group_rank_bypass(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, kwargs))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    feature_calls = [call for call in calls if call[0] == "feature_snapshot"]
    assert len(feature_calls) == len(export_script.STATIC_EXPORT_MARKETS)
    assert feature_calls[0][1] == {
        "as_of_date_str": "2026-04-02",
        "static_daily_mode": True,
        "universe_name": "market:us",
        "market": "US",
        "publish_pointer_key": "latest_published_market:US",
        "ignore_runtime_market_gate": True,
        "rs_formula_version_override": BALANCED_RS_FORMULA_VERSION,
        "skip_ibd_metadata_enrichment": True,
    }


def test_run_daily_refresh_builds_snapshot_before_group_rank_backfill_and_reenriches(monkeypatch):
    """``build_daily_snapshot`` hydrates broad historical prices in static CI.
    The group-rank history backfill must run after that hydration step, then
    re-run metadata enrichment so static export reads up-to-date
    ``ibd_group_rank`` values from ``details_json``."""

    events: list[str] = []

    def make_task(name: str):
        def run(**kwargs):
            events.append(name)
            return {"task": name, "kwargs": kwargs, "run_id": 77, "status": "published"}

        return SimpleNamespace(run=run)

    enrich_calls: list[dict] = []

    def fake_enrich(*, feature_run_id, ranking_date, **_kwargs):
        events.append(f"enrich:{feature_run_id}")
        enrich_calls.append({"feature_run_id": feature_run_id, "ranking_date": ranking_date})
        return {"run_id": feature_run_id, "updated_rows": 3, "missing_rank_rows": 0}

    group_rank_calls: list[dict] = []

    def fake_ensure_group_rank_history(*, as_of_date, market, formula_version):
        events.append(f"group_rank:{market}")
        group_rank_calls.append({
            "as_of_date": as_of_date,
            "market": market,
            "formula_version": formula_version,
        })
        return _backfill_result(
            status=GroupRankHistoryBackfillStatus.COMPLETED,
            market=market,
            as_of_date=as_of_date,
        )

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        fake_enrich,
    )
    monkeypatch.setattr(export_script, "_ensure_group_rank_history", fake_ensure_group_rank_history)
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh(market="US")  # noqa: SLF001 - intentional unit test coverage

    assert warnings == []
    # Order matters: the group-rank backfill needs the historical prices
    # hydrated by build_daily_snapshot, and the re-enrich step must run after
    # the backfill so it can repair the inner enrichment's missing ranks.
    assert events == [
        "universe_refresh",
        "fundamentals_refresh",
        "feature_snapshot",
        "group_rank:US",
        "enrich:77",
    ]
    assert group_rank_calls == [{
        "as_of_date": date(2026, 4, 2),
        "market": "US",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
    }]
    assert enrich_calls == [{"feature_run_id": 77, "ranking_date": date(2026, 4, 2)}]
    assert results["ibd_metadata_refresh"]["US"] == {
        "run_id": 77,
        "updated_rows": 3,
        "missing_rank_rows": 0,
    }


def test_run_daily_refresh_skips_reenrich_when_group_rank_backfill_errored(monkeypatch):
    """If ``_ensure_group_rank_history`` fails, the IBDGroupRank table is
    still missing ``as_of_date`` rows. Re-enriching in that state would
    overwrite previously valid ``ibd_group_rank`` values with ``None``
    (most damaging when ``build_daily_snapshot`` returned ``already_published``
    and the existing run carries ranks from an earlier successful refresh).
    The driver must skip re-enrich when the backfill did not succeed."""

    enrich_calls: list[dict] = []

    def fake_enrich(**kwargs):
        enrich_calls.append(kwargs)
        return {"updated_rows": 99}

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda **_kwargs: {"task": "universe_refresh"}))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", SimpleNamespace(run=lambda **_kwargs: {"task": "fundamentals_refresh"}))
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: {
                "status": "skipped",
                "reason": "already_published",
                "existing_run_id": 77,
                "kwargs": kwargs,
            }
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        fake_enrich,
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market, formula_version: _backfill_result(
            status=GroupRankHistoryBackfillStatus.ERRORED,
            market=market,
            as_of_date=as_of_date,
            error="Failed to fetch SPY benchmark data",
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    results, _warnings = export_script._run_daily_refresh(market="US")  # noqa: SLF001 - intentional unit test coverage

    assert enrich_calls == []
    assert results["ibd_metadata_refresh"]["US"] == {
        "status": "skipped",
        "market": "US",
        "reason": "group_rank_backfill_errored",
    }


def test_run_daily_refresh_quarantines_fresh_deferred_snapshot_when_group_rank_backfill_errored(
    monkeypatch,
):
    enrich_calls: list[dict] = []

    def fake_enrich(**kwargs):
        enrich_calls.append(kwargs)
        return {"updated_rows": 99}

    monkeypatch.setattr(
        universe_tasks,
        "refresh_stock_universe",
        SimpleNamespace(run=lambda **_kwargs: {"task": "universe_refresh"}),
    )
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda **_kwargs: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: {
                "status": "published",
                "run_id": 77,
                "metadata_refresh": {
                    "status": "skipped",
                    "reason": "deferred",
                },
                "kwargs": kwargs,
            }
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        fake_enrich,
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market, formula_version: _backfill_result(
            status=GroupRankHistoryBackfillStatus.ERRORED,
            market=market,
            as_of_date=as_of_date,
            error="Failed to fetch SPY benchmark data",
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    results, warnings = export_script._run_daily_refresh(market="US")  # noqa: SLF001 - intentional unit test coverage

    snapshot = results["feature_snapshots"]["US"]
    assert snapshot["status"] == "quarantined"
    assert snapshot["reason"] == "group_rank_backfill_not_ready"
    assert snapshot["run_id"] == 77
    assert snapshot["failure_diagnostics"]["group_rank_history_backfill"]["status"] == "errored"
    assert snapshot["failure_diagnostics"]["group_rank_history_backfill"]["error"] == (
        "Failed to fetch SPY benchmark data"
    )
    assert (
        "Static export market US group-rank history backfill not ready for 2026-04-02: errored."
        in warnings
    )
    assert enrich_calls == []
    assert results["ibd_metadata_refresh"]["US"] == {
        "status": "skipped",
        "market": "US",
        "reason": "snapshot_not_ready",
    }


def test_run_daily_refresh_skips_reenrich_when_snapshot_not_ready(monkeypatch):
    enrich_calls: list[dict] = []

    def fake_enrich(**kwargs):
        enrich_calls.append(kwargs)
        return {"updated_rows": 0}

    def build_snapshot(**kwargs):
        if kwargs["market"] == "US":
            return {"status": "failed", "run_id": 91}
        return {"status": "published", "run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        fake_enrich,
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market, formula_version: _backfill_result(
            status=GroupRankHistoryBackfillStatus.COMPLETED,
            market=market,
            as_of_date=as_of_date,
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda: {"task": "universe_refresh"}))
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    results, _warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    # US returned status "failed" → not snapshot_ready → re-enrich must skip it
    assert results["ibd_metadata_refresh"]["US"]["status"] == "skipped"
    assert results["ibd_metadata_refresh"]["US"]["reason"] == "snapshot_not_ready"
    # The other markets returned no status but a run_id, so they ARE re-enriched.
    other_markets = [m for m in export_script.STATIC_EXPORT_MARKETS if m != "US"]
    assert all(
        results["ibd_metadata_refresh"][m]["updated_rows"] == 0
        for m in other_markets
    )
    assert {call["feature_run_id"] for call in enrich_calls} == {77}


def test_run_daily_refresh_warns_when_non_default_market_snapshot_is_not_publish_ready(monkeypatch):
    def build_snapshot(**kwargs):
        if kwargs["market"] == "HK":
            return {
                "status": "skipped",
                "reason": "market HK is disabled in local runtime preferences",
                "market": "HK",
            }
        return {"status": "published", "run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda: {"task": "universe_refresh"}))
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)
    _stub_static_market_exposure(monkeypatch)

    _results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert "Static export market HK snapshot returned status 'skipped' (market HK is disabled in local runtime preferences)." in warnings


def test_resolve_latest_completed_trading_date_uses_market_calendar(monkeypatch):
    """Per-market resolution: each market hits its own calendar instead of
    NYSE's. Regression test for the IN-skipped-on-US-trading-day bug."""

    calls: list[str] = []
    market_dates = {
        "US": date(2026, 4, 2),
        "IN": date(2026, 4, 3),
        "HK": date(2026, 4, 1),
    }

    def last_completed_trading_day(market: str) -> date:
        calls.append(market)
        return market_dates[market]

    monkeypatch.setattr(
        export_script,
        "get_market_calendar_service",
        lambda: SimpleNamespace(last_completed_trading_day=last_completed_trading_day),
    )

    assert export_script._resolve_latest_completed_trading_date("IN") == date(2026, 4, 3)
    assert export_script._resolve_latest_completed_trading_date("US") == date(2026, 4, 2)
    assert export_script._resolve_latest_completed_trading_date("HK") == date(2026, 4, 1)
    assert calls == ["IN", "US", "HK"]


def test_main_rejects_market_in_combine_mode(monkeypatch, tmp_path):
    combine_calls: list[tuple[object, object, bool]] = []

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        lambda artifacts_dir, output_dir, *, clean=True: combine_calls.append((artifacts_dir, output_dir, clean)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--combine-artifacts-dir",
            str(tmp_path / "artifacts"),
            "--market",
            "HK",
        ],
    )

    with pytest.raises(SystemExit, match="--combine-artifacts-dir cannot be used together with --market"):
        export_script.main()

    assert combine_calls == []


def test_main_rejects_fallback_artifacts_without_combine_mode(monkeypatch, tmp_path):
    combine_calls: list[tuple[object, object, object, bool]] = []

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        lambda artifacts_dir, output_dir, *, fallback_artifacts_dir=None, clean=True: combine_calls.append(
            (artifacts_dir, output_dir, fallback_artifacts_dir, clean)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--fallback-artifacts-dir",
            str(tmp_path / "fallback"),
        ],
    )

    with pytest.raises(SystemExit, match="--fallback-artifacts-dir requires --combine-artifacts-dir"):
        export_script.main()

    assert combine_calls == []


def test_main_passes_fallback_artifacts_dir_to_combine(monkeypatch, tmp_path):
    combine_calls: list[tuple[object, object, object, bool, object, object, object]] = []
    output_dir = tmp_path / "out"
    artifacts_dir = tmp_path / "artifacts"
    fallback_dir = tmp_path / "fallback"

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        lambda artifacts_dir, output_dir, *, fallback_artifacts_dir=None, clean=True,
        rs_formula_version_overrides=None,
        fallback_rs_formula_version_overrides=None,
        optional_markets=(): combine_calls.append(
            (
                artifacts_dir,
                output_dir,
                fallback_artifacts_dir,
                clean,
                rs_formula_version_overrides,
                fallback_rs_formula_version_overrides,
                optional_markets,
            )
        )
        or SimpleNamespace(
            output_dir=output_dir,
            generated_at="2026-04-05T22:00:00Z",
            as_of_date="2026-04-05",
            warnings=(),
            manifest={},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--combine-artifacts-dir",
            str(artifacts_dir),
            "--fallback-artifacts-dir",
            str(fallback_dir),
            "--no-clean",
        ],
    )

    assert export_script.main() == 0

    assert combine_calls == [(
        artifacts_dir,
        output_dir,
        fallback_dir,
        False,
        {
            market: BALANCED_RS_FORMULA_VERSION
            for market in export_script.STATIC_EXPORT_MARKETS
        },
        {},
        export_script.OPTIONAL_STATIC_MARKETS,
    )]


def test_main_passes_independent_options_roots_to_combine(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "markets"
    options_dir = tmp_path / "options-current"
    fallback_options_dir = tmp_path / "options-fallback"
    captured = {}

    def combine(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_dir=output_dir,
            generated_at="2026-09-04T22:00:00Z",
            as_of_date="2026-09-04",
            warnings=(),
            manifest={},
        )

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        combine,
    )

    assert export_script.main(
        [
            "--output-dir",
            str(output_dir),
            "--combine-artifacts-dir",
            str(artifacts_dir),
            "--options-artifacts-dir",
            str(options_dir),
            "--fallback-options-artifacts-dir",
            str(fallback_options_dir),
        ]
    ) == 0
    assert captured["options_artifacts_dir"] == options_dir
    assert captured["fallback_options_artifacts_dir"] == fallback_options_dir


def test_main_combines_with_independent_per_market_rs_formula_policy(
    monkeypatch,
    tmp_path,
):
    captured: list[dict[str, object]] = []
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        lambda _artifacts_dir, _output_dir, **kwargs: captured.append(kwargs)
        or SimpleNamespace(
            output_dir=output_dir,
            generated_at="2026-04-05T22:00:00Z",
            as_of_date="2026-04-05",
            warnings=(),
            manifest={},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--combine-artifacts-dir",
            str(tmp_path / "artifacts"),
            "--rs-formula-overrides-json",
            '{"HK":"legacy-linear-v1"}',
        ],
    )

    assert export_script.main() == 0
    current_policy = captured[0]["rs_formula_version_overrides"]
    assert isinstance(current_policy, dict)
    assert current_policy["HK"] == LEGACY_RS_FORMULA_VERSION
    assert current_policy["US"] == BALANCED_RS_FORMULA_VERSION
    assert captured[0]["fallback_rs_formula_version_overrides"] == {
        "HK": LEGACY_RS_FORMULA_VERSION
    }


def test_main_rejects_global_rs_formula_override_when_combining(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--combine-artifacts-dir",
            str(tmp_path / "artifacts"),
            "--rs-formula-version",
            LEGACY_RS_FORMULA_VERSION,
        ],
    )

    with pytest.raises(SystemExit, match="single-market exports"):
        export_script.main()


def test_main_passes_formula_override_to_direct_market_export(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, object] = {}

    class FakeExportService:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, output_dir, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_dir=output_dir,
                generated_at="2026-04-05T22:00:00Z",
                as_of_date="2026-04-05",
                warnings=(),
            )

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(export_script, "StaticSiteExportService", FakeExportService)

    result = export_script.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--market",
            "HK",
            "--rs-formula-version",
            LEGACY_RS_FORMULA_VERSION,
        ]
    )

    assert result == 0
    assert captured["rs_formula_version_overrides"] == {
        **{
            market: BALANCED_RS_FORMULA_VERSION
            for market in export_script.STATIC_EXPORT_MARKETS
        },
        "HK": LEGACY_RS_FORMULA_VERSION,
    }
