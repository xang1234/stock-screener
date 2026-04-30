"""Task-level regression tests for feature store Celery entrypoints."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest
from celery.exceptions import Retry, SoftTimeLimitExceeded
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.config import settings
from app.interfaces.tasks.feature_store_tasks import (
    BOOTSTRAP_SCAN_COVERAGE_MAX_RETRIES,
    FEATURE_SNAPSHOT_TRANSIENT_MAX_RETRIES,
    FEATURE_SNAPSHOT_TOTAL_MAX_RETRIES,
    _bootstrap_coverage_total,
    _create_auto_scan_for_published_run,
    _enrich_feature_run_with_ibd_metadata,
    _fail_stale_feature_runs,
    _repair_current_us_group_metadata,
    _upsert_feature_run_pointer,
    build_daily_snapshot,
)
from app.domain.scanning.defaults import (
    get_bootstrap_scan_profile,
    get_default_scan_profile,
)
from app.schemas.universe import UniverseType
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
from app.models.scan_result import Scan
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.services.market_taxonomy_service import MarketTaxonomyEntry


class _FakeTask:
    request = SimpleNamespace(id="task-123")


class _FakeUseCase:
    def __init__(self) -> None:
        self.received_cmd = None

    def execute(self, *, uow, cmd, progress, cancel):
        self.received_cmd = cmd
        return SimpleNamespace(
            run_id=11,
            status="published",
            total_symbols=2,
            processed_symbols=2,
            failed_symbols=0,
            skipped_symbols=0,
            row_count=2,
            duration_seconds=12.5,
            dq_passed=True,
            warnings=(),
        )


_TASK_BODY = inspect.unwrap(build_daily_snapshot.run)


class _NonSkippingUoW:
    def __init__(self) -> None:
        self.universe = SimpleNamespace(resolve_symbols=lambda _universe_def: ["AAPL", "MSFT"])
        self.feature_runs = SimpleNamespace(
            find_latest_published_exact=lambda **_kwargs: None,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_build_daily_snapshot_normalizes_default_active_universe():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.interfaces.tasks.feature_store_tasks.date"
    ) as mock_date, patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        mock_date.today.return_value = date(2026, 3, 16)

        result = _TASK_BODY(_FakeTask())

    assert result["status"] == "published"
    assert result["row_count"] == 2
    assert result["duration_seconds"] == 12.5
    assert fake_use_case.received_cmd is not None
    assert fake_use_case.received_cmd.universe_def.type == UniverseType.MARKET
    assert fake_use_case.received_cmd.universe_def.market.value == "US"


def test_bootstrap_coverage_total_preserves_zero_price_total():
    assert _bootstrap_coverage_total(
        {"price_total_symbols": 0, "fundamentals_total_symbols": 7}
    ) == 0
    assert _bootstrap_coverage_total(
        {"price_total_symbols": None, "fundamentals_total_symbols": 7}
    ) == 7


def test_build_daily_snapshot_autoretry_budget_covers_bootstrap_polling():
    assert (
        FEATURE_SNAPSHOT_TOTAL_MAX_RETRIES
        == BOOTSTRAP_SCAN_COVERAGE_MAX_RETRIES
        + FEATURE_SNAPSHOT_TRANSIENT_MAX_RETRIES
    )
    assert build_daily_snapshot.max_retries == FEATURE_SNAPSHOT_TOTAL_MAX_RETRIES


def test_build_daily_snapshot_never_passes_legacy_dict_shape():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")

    assert isinstance(fake_use_case.received_cmd.universe_def, dict) is False


def test_build_daily_snapshot_uses_default_scan_profile_when_not_provided():
    fake_use_case = _FakeUseCase()
    defaults = get_default_scan_profile("US")

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")

    assert fake_use_case.received_cmd.screener_names == defaults["screeners"]
    assert fake_use_case.received_cmd.criteria == defaults["criteria"]
    assert fake_use_case.received_cmd.composite_method == defaults["composite_method"]


def test_build_daily_snapshot_bootstrap_uses_lightweight_scan_profile():
    fake_use_case = _FakeUseCase()
    bootstrap_defaults = get_bootstrap_scan_profile("US")

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            activity_lifecycle="bootstrap",
            bootstrap_cache_only_if_covered=True,
            bootstrap_coverage_report={"eligible": True},
        )

    assert fake_use_case.received_cmd.screener_names == bootstrap_defaults["screeners"]
    assert "setup_engine" not in fake_use_case.received_cmd.screener_names
    assert fake_use_case.received_cmd.criteria == bootstrap_defaults["criteria"]
    assert fake_use_case.received_cmd.composite_method == bootstrap_defaults["composite_method"]


def test_build_daily_snapshot_defaults_to_market_profile_and_pointer():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.interfaces.tasks.feature_store_tasks._is_market_trading_day",
        return_value=True,
    ), patch(
        "app.services.market_calendar_service.MarketCalendarService.last_completed_trading_day",
        return_value=date(2026, 3, 16),
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        return_value=True,
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._upsert_feature_run_pointer",
    ) as mock_upsert:
        result = _TASK_BODY(_FakeTask(), market="hk")

    assert result["status"] == "published"
    assert result["as_of_date"] == "2026-03-16"
    assert fake_use_case.received_cmd.market == "HK"
    assert fake_use_case.received_cmd.universe_def.type == UniverseType.MARKET
    assert fake_use_case.received_cmd.universe_def.market.value == "HK"
    assert fake_use_case.received_cmd.publish_pointer_key == "latest_published_market:HK"
    mock_upsert.assert_not_called()


def test_build_daily_snapshot_skip_if_published_requires_exact_signature_match():
    lookup_calls = []

    class _CheckUoW:
        def __init__(self) -> None:
            self.universe = SimpleNamespace(resolve_symbols=lambda _universe_def: ["AAPL", "MSFT"])
            self.feature_runs = SimpleNamespace(
                find_latest_published_exact=lambda **kwargs: (
                    lookup_calls.append(kwargs),
                    SimpleNamespace(
                        id=41,
                        as_of_date=date(2026, 3, 16),
                    ),
                )[1]
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.hash_scan_signature",
        return_value="same-input",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.hash_universe_symbols",
        return_value="same-universe",
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        return_value=_CheckUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
    ) as mock_auto_scan:
        result = _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")

    assert result["status"] == "skipped"
    assert result["reason"] == "already_published"
    assert result["existing_run_id"] == 41
    assert fake_use_case.received_cmd is None
    mock_auto_scan.assert_called_once_with(
        feature_run_id=41,
        universe_name=get_default_scan_profile("US")["universe"],
        screeners=get_default_scan_profile("US")["screeners"],
        criteria=get_default_scan_profile("US")["criteria"],
        composite_method=get_default_scan_profile("US")["composite_method"],
    )
    assert lookup_calls == [{
        "input_hash": "same-input",
        "universe_hash": "same-universe",
        "as_of_date": date(2026, 3, 16),
    }]


def test_build_daily_snapshot_bootstrap_skip_if_published_uses_cache_only_supported_hash():
    lookup_calls = []
    hash_calls = []

    class _CheckUoW:
        def __init__(self) -> None:
            self.universe = SimpleNamespace(resolve_symbols=lambda _universe_def: ["AAPL", "BAD-WT", "MSFT"])
            self.feature_runs = SimpleNamespace(
                find_latest_published_exact=lambda **kwargs: (
                    lookup_calls.append(kwargs),
                    SimpleNamespace(
                        id=51,
                        as_of_date=date(2026, 3, 16),
                    ),
                )[1] if kwargs["universe_hash"] == "AAPL,MSFT" else None
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_use_case = _FakeUseCase()

    def _hash_universe(symbols):
        hash_calls.append(list(symbols))
        return ",".join(symbols)

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.hash_scan_signature",
        return_value="same-input",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.hash_universe_symbols",
        side_effect=_hash_universe,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        return_value=_CheckUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            activity_lifecycle="bootstrap",
            bootstrap_cache_only_if_covered=True,
            bootstrap_coverage_report={"eligible": True},
        )

    assert result["status"] == "skipped"
    assert result["reason"] == "already_published"
    assert result["existing_run_id"] == 51
    assert result["skipped_symbols"] == 1
    assert fake_use_case.received_cmd is None
    assert hash_calls == [["AAPL", "MSFT"]]
    assert lookup_calls == [{
        "input_hash": "same-input",
        "universe_hash": "AAPL,MSFT",
        "as_of_date": date(2026, 3, 16),
    }]


def test_build_daily_snapshot_skip_if_published_repairs_requested_pointer():
    class _CheckUoW:
        def __init__(self) -> None:
            self.universe = SimpleNamespace(resolve_symbols=lambda _universe_def: ["0700.HK"])
            self.feature_runs = SimpleNamespace(
                find_latest_published_exact=lambda **_kwargs: SimpleNamespace(
                    id=84,
                    as_of_date=date(2026, 3, 16),
                )
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        return_value=_CheckUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        return_value=True,
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._upsert_feature_run_pointer",
    ) as mock_upsert:
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            universe_name="market:hk",
            market="HK",
            publish_pointer_key="latest_published_market:HK",
        )

    assert result["status"] == "skipped"
    assert result["existing_run_id"] == 84
    mock_upsert.assert_called_once_with(
        session_factory=ANY,
        pointer_key="latest_published_market:HK",
        run_id=84,
    )


def test_upsert_feature_run_pointer_retries_after_integrity_error():
    pointer = SimpleNamespace(run_id=10)
    query_results = [None, pointer]

    class _FakeSession:
        def __init__(self) -> None:
            self.added = []
            self.commit_calls = 0
            self.rollback_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def query(self, _model):
            return self

        def filter(self, *_args, **_kwargs):
            return self

        def first(self):
            return query_results.pop(0)

        def add(self, obj):
            self.added.append(obj)

        def commit(self):
            self.commit_calls += 1
            if self.commit_calls == 1:
                raise IntegrityError("insert", {}, Exception("duplicate"))

        def rollback(self):
            self.rollback_calls += 1

    fake_session = _FakeSession()

    _upsert_feature_run_pointer(
        session_factory=lambda: fake_session,
        pointer_key="latest_published_market:HK",
        run_id=84,
    )

    assert fake_session.rollback_calls == 1
    assert fake_session.commit_calls == 2
    assert pointer.run_id == 84


def test_build_daily_snapshot_reraises_soft_time_limit():
    class _TimeoutUseCase:
        def execute(self, **kwargs):
            raise SoftTimeLimitExceeded()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=_TimeoutUseCase(),
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        with pytest.raises(SoftTimeLimitExceeded):
            _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")

    assert build_daily_snapshot.soft_time_limit == 10800


def test_build_daily_snapshot_reraises_soft_time_limit_when_failure_activity_publish_breaks():
    class _TimeoutUseCase:
        def execute(self, **kwargs):
            raise SoftTimeLimitExceeded()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=_TimeoutUseCase(),
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.mark_market_activity_failed",
        side_effect=RuntimeError("activity store unavailable"),
    ):
        with pytest.raises(SoftTimeLimitExceeded):
            _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")


def test_build_daily_snapshot_creates_auto_scan_after_publish():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ) as mock_auto_scan:
        _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")

    mock_auto_scan.assert_called_once_with(
        feature_run_id=11,
        universe_name=get_default_scan_profile("US")["universe"],
        screeners=get_default_scan_profile("US")["screeners"],
        criteria=get_default_scan_profile("US")["criteria"],
        composite_method=get_default_scan_profile("US")["composite_method"],
    )


def test_build_daily_snapshot_static_daily_mode_requires_bulk_prefetch():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.utils.parallelism.os.cpu_count",
        return_value=4,
    ), patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            static_daily_mode=True,
        )

    assert result["status"] == "published"
    assert fake_use_case.received_cmd.require_bulk_prefetch is True
    assert fake_use_case.received_cmd.exclude_unsupported_price_symbols is True
    assert fake_use_case.received_cmd.batch_only_prices is True
    assert fake_use_case.received_cmd.batch_only_fundamentals is True
    assert (
        fake_use_case.received_cmd.static_chunk_size
        == settings.static_snapshot_chunk_size
    )
    assert (
        fake_use_case.received_cmd.static_parallel_workers
        == 2
    )


def test_build_daily_snapshot_static_daily_mode_keeps_setting_as_upper_bound_on_larger_hosts():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.utils.parallelism.os.cpu_count",
        return_value=16,
    ), patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            static_daily_mode=True,
        )

    assert (
        fake_use_case.received_cmd.static_parallel_workers
        == settings.static_snapshot_parallel_workers
    )


def test_build_daily_snapshot_static_daily_mode_uses_null_progress_sink():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        side_effect=AssertionError("CeleryProgressSink should not be used in static mode"),
    ), patch(
        "app.domain.scanning.ports.NullProgressSink",
        return_value=object(),
    ) as mock_null_progress, patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            static_daily_mode=True,
        )

    assert result["status"] == "published"
    mock_null_progress.assert_called_once_with()


def test_build_daily_snapshot_bootstrap_gate_pass_wires_cache_only_without_null_progress():
    fake_use_case = _FakeUseCase()
    celery_progress = object()

    with patch(
        "app.utils.parallelism.os.cpu_count",
        return_value=4,
    ), patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=celery_progress,
    ) as mock_celery_progress, patch(
        "app.domain.scanning.ports.NullProgressSink",
        side_effect=AssertionError("bootstrap gate must keep normal progress"),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            activity_lifecycle="bootstrap",
            bootstrap_cache_only_if_covered=True,
            bootstrap_coverage_report={
                "eligible": True,
                "price_coverage_ratio": 1.0,
                "fundamentals_coverage_ratio": 1.0,
            },
        )

    assert result["status"] == "published"
    assert fake_use_case.received_cmd.bootstrap_cache_only_if_covered is True
    assert not hasattr(fake_use_case.received_cmd, "bootstrap_coverage_threshold")
    assert fake_use_case.received_cmd.bootstrap_coverage_report["eligible"] is True
    assert fake_use_case.received_cmd.static_chunk_size == settings.static_snapshot_chunk_size
    assert fake_use_case.received_cmd.static_parallel_workers == 2
    assert fake_use_case.received_cmd.batch_only_prices is False
    assert fake_use_case.received_cmd.require_bulk_prefetch is False
    mock_celery_progress.assert_called_once()


def test_build_daily_snapshot_bootstrap_gate_fail_retries_without_live_fallback():
    fake_use_case = _FakeUseCase()
    retry_calls = []
    retry_kwargs = {
        "as_of_date_str": "2026-03-16",
        "activity_lifecycle": "bootstrap",
        "bootstrap_cache_only_if_covered": True,
        "bootstrap_coverage_retry_count": 2,
        "bootstrap_coverage_report": {
            "eligible": False,
            "price_coverage_ratio": 0.9,
            "fundamentals_coverage_ratio": 1.0,
            "price_missing_symbols": 1,
            "fundamentals_missing_symbols": 0,
        },
    }

    class _RetryTask:
        request = SimpleNamespace(id="task-123", retries=2, kwargs=retry_kwargs)

        def retry(self, *, exc=None, countdown=None, max_retries=None, kwargs=None):
            retry_calls.append(
                {
                    "exc": exc,
                    "countdown": countdown,
                    "max_retries": max_retries,
                    "kwargs": kwargs,
                }
            )
            raise Retry(message=str(exc))

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        side_effect=AssertionError("auto scan must not publish while waiting for coverage"),
    ):
        with pytest.raises(Retry):
            _TASK_BODY(
                _RetryTask(),
                **retry_kwargs,
            )

    assert fake_use_case.received_cmd is None
    assert retry_calls == [
        {
            "exc": ANY,
            "countdown": 30,
            "max_retries": FEATURE_SNAPSHOT_TOTAL_MAX_RETRIES,
            "kwargs": {
                **retry_kwargs,
                "bootstrap_coverage_report": None,
                "bootstrap_coverage_retry_count": 3,
            },
        }
    ]
    assert "waiting_for_bootstrap_cache_coverage:US" in str(retry_calls[0]["exc"])


def test_build_daily_snapshot_bootstrap_gate_exhaustion_fails_with_coverage_report():
    fake_use_case = _FakeUseCase()
    failed_activity = []

    class _RetryExhaustedTask:
        request = SimpleNamespace(id="task-123", retries=120)

        def retry(self, **_kwargs):
            raise AssertionError("retry must not be called after retry budget is exhausted")

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.mark_market_activity_failed",
        lambda _db, **kwargs: failed_activity.append(kwargs),
    ):
        with pytest.raises(RuntimeError, match="bootstrap cache coverage exhausted"):
            _TASK_BODY(
                _RetryExhaustedTask(),
                as_of_date_str="2026-03-16",
                activity_lifecycle="bootstrap",
                bootstrap_cache_only_if_covered=True,
                bootstrap_coverage_retry_count=BOOTSTRAP_SCAN_COVERAGE_MAX_RETRIES,
                bootstrap_coverage_report={
                    "eligible": False,
                    "price_coverage_ratio": 0.9,
                    "fundamentals_coverage_ratio": 1.0,
                    "price_missing_symbols": 1,
                    "fundamentals_missing_symbols": 0,
                },
            )

    assert fake_use_case.received_cmd is None
    assert failed_activity
    assert failed_activity[0]["stage_key"] == "scan"
    assert failed_activity[0]["lifecycle"] == "bootstrap"
    assert "waiting_for_bootstrap_cache_coverage:US" in failed_activity[0]["message"]
    assert "price_missing=1" in failed_activity[0]["message"]


def test_build_daily_snapshot_does_not_expose_bootstrap_threshold_override():
    assert "bootstrap_coverage_threshold" not in inspect.signature(_TASK_BODY).parameters


def test_enrich_feature_run_with_ibd_metadata_updates_details_json():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            IBDIndustryGroup.__table__,
            IBDGroupRank.__table__,
        ],
    )
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    with session_factory() as db:
        db.add(
            FeatureRun(
                id=21,
                as_of_date=date(2026, 4, 2),
                run_type="daily_snapshot",
                status="published",
            )
        )
        db.add_all(
            [
                StockFeatureDaily(
                    run_id=21,
                    symbol="NVDA",
                    as_of_date=date(2026, 4, 2),
                    composite_score=95.0,
                    overall_rating=5,
                    passes_count=4,
                    details_json={"symbol": "NVDA"},
                ),
                StockFeatureDaily(
                    run_id=21,
                    symbol="MSFT",
                    as_of_date=date(2026, 4, 2),
                    composite_score=88.0,
                    overall_rating=4,
                    passes_count=3,
                    details_json={"symbol": "MSFT", "ibd_group_rank": 99},
                ),
            ]
        )
        db.add_all(
            [
                IBDIndustryGroup(symbol="NVDA", industry_group="Semiconductors"),
                IBDIndustryGroup(symbol="MSFT", industry_group="Software"),
                IBDGroupRank(industry_group="Semiconductors", date=date(2026, 4, 2), rank=1, avg_rs_rating=95.0),
            ]
        )
        db.commit()

    stats = _enrich_feature_run_with_ibd_metadata(
        feature_run_id=21,
        ranking_date=date(2026, 4, 2),
        session_factory=session_factory,
    )

    with session_factory() as db:
        nvda = db.query(StockFeatureDaily).filter_by(run_id=21, symbol="NVDA").one()
        msft = db.query(StockFeatureDaily).filter_by(run_id=21, symbol="MSFT").one()

    assert stats["updated_rows"] == 2
    assert stats["missing_industry_rows"] == 0
    assert stats["missing_rank_rows"] == 1
    assert nvda.details_json["ibd_industry_group"] == "Semiconductors"
    assert nvda.details_json["ibd_group_rank"] == 1
    assert msft.details_json["ibd_industry_group"] == "Software"
    assert msft.details_json["ibd_group_rank"] is None

    engine.dispose()


def test_enrich_feature_run_with_ibd_metadata_batches_feature_rows():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            IBDIndustryGroup.__table__,
            IBDGroupRank.__table__,
        ],
    )
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    with session_factory() as db:
        db.add(
            FeatureRun(
                id=24,
                as_of_date=date(2026, 4, 2),
                run_type="daily_snapshot",
                status="published",
            )
        )
        feature_rows = [
            StockFeatureDaily(
                run_id=24,
                symbol=f"SYM{index:03d}",
                as_of_date=date(2026, 4, 2),
                composite_score=95.0,
                overall_rating=5,
                passes_count=4,
                details_json={"symbol": f"SYM{index:03d}"},
            )
            for index in range(501)
        ]
        db.add_all(feature_rows)
        db.add_all(
            IBDIndustryGroup(symbol=row.symbol, industry_group="Software")
            for row in feature_rows
        )
        db.add(
            IBDGroupRank(
                industry_group="Software",
                date=date(2026, 4, 2),
                rank=3,
                avg_rs_rating=90.0,
            )
        )
        db.commit()

    statements: list[str] = []

    def collect_sql(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", collect_sql)
    try:
        stats = _enrich_feature_run_with_ibd_metadata(
            feature_run_id=24,
            ranking_date=date(2026, 4, 2),
            session_factory=session_factory,
        )
    finally:
        event.remove(engine, "before_cursor_execute", collect_sql)

    feature_row_selects = [
        statement
        for statement in statements
        if statement.lstrip().lower().startswith("select")
        and "stock_feature_daily" in statement.lower()
        and "join" not in statement.lower()
        and "count(" not in statement.lower()
    ]

    assert stats["total_rows"] == 501
    assert stats["updated_rows"] == 501
    assert len(feature_row_selects) >= 2, "\n---\n".join(statements)
    assert all("LIMIT" in statement.upper() for statement in feature_row_selects)

    engine.dispose()


def test_enrich_feature_run_with_ibd_metadata_uses_market_taxonomy_for_non_us_runs():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
        ],
    )
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    with session_factory() as db:
        db.add(
            FeatureRun(
                id=22,
                as_of_date=date(2026, 4, 2),
                run_type="daily_snapshot",
                status="published",
                config_json={"universe": {"market": "HK"}},
            )
        )
        db.add(
            StockFeatureDaily(
                run_id=22,
                symbol="0700.HK",
                as_of_date=date(2026, 4, 2),
                composite_score=95.0,
                overall_rating=5,
                passes_count=4,
                details_json={"symbol": "0700.HK", "rs_rating": 98.0},
            )
        )
        db.commit()

    class _FakeTaxonomyService:
        def get(self, symbol, *, market=None, exchange=None):  # noqa: ARG002
            if symbol == "0700.HK" and market == "HK":
                return MarketTaxonomyEntry(
                    market="HK",
                    symbol="0700.HK",
                    industry_group="Internet Services",
                    themes=("AI Infrastructure", "Cloud"),
                )
            return None

    class _FakeMarketGroupRankingService:
        @staticmethod
        def compute_group_rankings_from_serialized_rows(rows, *, ranking_date):  # noqa: ARG004
            assert rows[0]["ibd_industry_group"] == "Internet Services"
            return [
                {
                    "industry_group": "Internet Services",
                    "rank": 2,
                }
            ]

    stats = _enrich_feature_run_with_ibd_metadata(
        feature_run_id=22,
        ranking_date=date(2026, 4, 2),
        session_factory=session_factory,
        taxonomy_service=_FakeTaxonomyService(),
        market_group_ranking_service=_FakeMarketGroupRankingService(),
    )

    with session_factory() as db:
        row = db.query(StockFeatureDaily).filter_by(run_id=22, symbol="0700.HK").one()

    assert stats["updated_rows"] == 1
    assert stats["missing_industry_rows"] == 0
    assert stats["missing_rank_rows"] == 0
    assert row.details_json["ibd_industry_group"] == "Internet Services"
    assert row.details_json["ibd_group_rank"] == 2
    assert row.details_json["market_themes"] == ["AI Infrastructure", "Cloud"]

    engine.dispose()


def test_enrich_feature_run_with_ibd_metadata_overrides_non_us_sector():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as db:
        db.add(
            FeatureRun(
                id=23,
                as_of_date=date(2026, 4, 2),
                run_type="daily_snapshot",
                status="published",
                config_json={"universe": {"market": "JP"}},
            )
        )
        db.add(
            StockFeatureDaily(
                run_id=23,
                symbol="7203.T",
                as_of_date=date(2026, 4, 2),
                composite_score=91.0,
                overall_rating=5,
                passes_count=4,
                details_json={
                    "symbol": "7203.T",
                    "gics_sector": "Consumer Discretionary",
                },
            )
        )
        db.commit()

    class _FakeTaxonomyService:
        def get(self, symbol, *, market=None, exchange=None):  # noqa: ARG002
            if symbol == "7203.T" and market == "JP":
                return MarketTaxonomyEntry(
                    market="JP",
                    symbol="7203.T",
                    industry_group="Transportation Equipment",
                    sector="Manufacturing",
                    industry="Automobiles",
                    themes=("Automation",),
                )
            return None

    class _FakeMarketGroupRankingService:
        @staticmethod
        def compute_group_rankings_from_serialized_rows(rows, *, ranking_date):  # noqa: ARG004
            assert rows[0]["ibd_industry_group"] == "Transportation Equipment"
            return [
                {
                    "industry_group": "Transportation Equipment",
                    "rank": 1,
                }
            ]

    _enrich_feature_run_with_ibd_metadata(
        feature_run_id=23,
        ranking_date=date(2026, 4, 2),
        session_factory=session_factory,
        taxonomy_service=_FakeTaxonomyService(),
        market_group_ranking_service=_FakeMarketGroupRankingService(),
    )

    with session_factory() as db:
        row = db.query(StockFeatureDaily).filter_by(run_id=23, symbol="7203.T").one()

    assert row.details_json["gics_sector"] == "Manufacturing"
    assert row.details_json["gics_industry"] == "Automobiles"
    assert row.details_json["ibd_industry_group"] == "Transportation Equipment"
    assert row.details_json["market_themes"] == ["Automation"]

    engine.dispose()
def test_repair_current_us_group_metadata_updates_latest_published_run_and_republishes_scan_snapshot():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            FeatureRunPointer.__table__,
            StockFeatureDaily.__table__,
            IBDIndustryGroup.__table__,
            IBDGroupRank.__table__,
            Scan.__table__,
        ],
    )
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    with session_factory() as db:
        db.add(
            FeatureRun(
                id=31,
                as_of_date=date(2026, 4, 2),
                run_type="daily_snapshot",
                status="published",
                config_json={"universe": {"market": "US"}},
            )
        )
        db.add(FeatureRunPointer(key="latest_published_market:US", run_id=31))
        db.add(
            StockFeatureDaily(
                run_id=31,
                symbol="NVDA",
                as_of_date=date(2026, 4, 2),
                composite_score=99.0,
                overall_rating=5,
                passes_count=4,
                details_json={"symbol": "NVDA"},
            )
        )
        db.add(
            Scan(
                scan_id="scan-us-1",
                status="completed",
                feature_run_id=31,
                universe_market="US",
            )
        )
        db.add(IBDIndustryGroup(symbol="NVDA", industry_group="Semiconductors"))
        db.add(
            IBDGroupRank(
                industry_group="Semiconductors",
                date=date(2026, 4, 2),
                rank=1,
                avg_rs_rating=95.0,
            )
        )
        db.commit()

    with patch(
        "app.services.ui_snapshot_service.safe_publish_scan_bootstrap",
        return_value=None,
    ) as mock_publish:
        stats = _repair_current_us_group_metadata(
            ranking_date=date(2026, 4, 2),
            session_factory=session_factory,
        )

    with session_factory() as db:
        nvda = db.query(StockFeatureDaily).filter_by(run_id=31, symbol="NVDA").one()

    assert stats["feature_run"]["run_id"] == 31
    assert stats["feature_run"]["updated_rows"] == 1
    assert nvda.details_json["ibd_industry_group"] == "Semiconductors"
    assert nvda.details_json["ibd_group_rank"] == 1
    assert mock_publish.call_args_list[0].args == ("scan-us-1",)
    assert mock_publish.call_args_list[1].args == ()

    engine.dispose()


def test_create_auto_scan_uses_saved_run_universe_count_for_total_stocks():
    created_scan = {}

    class _CountQuery:
        def filter(self, *_args):
            return self

        def scalar(self):
            return 2

    class _AutoScanUoW:
        def __init__(self) -> None:
            self.session = SimpleNamespace(query=lambda *_args, **_kwargs: _CountQuery())
            self.scans = SimpleNamespace(
                get_by_idempotency_key=lambda _key: None,
                create=lambda **kwargs: created_scan.update(kwargs) or SimpleNamespace(
                    scan_id="scan-123",
                    universe_key="all",
                ),
            )
            self.universe = SimpleNamespace(
                resolve_symbols=lambda _universe_def: ["AAPL", "MZYX-U", "MSFT"],
            )
            self.feature_runs = SimpleNamespace(
                get_run=lambda _run_id: SimpleNamespace(
                    published_at=datetime(2026, 3, 16, 21, 30, tzinfo=timezone.utc),
                    stats=SimpleNamespace(passed_symbols=1),
                ),
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def commit(self):
            return None

    universe_def = SimpleNamespace(
        label=lambda: "All Active Stocks",
        key=lambda: "all",
        type=SimpleNamespace(value="all"),
        exchange=None,
        index=None,
        symbols=None,
    )

    with patch(
        "app.database.SessionLocal",
        return_value=SimpleNamespace(close=lambda: None),
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _AutoScanUoW(),
    ), patch(
        "app.services.universe_resolver.normalize_universe_definition",
        return_value=universe_def,
    ), patch(
        "app.services.scan_execution.cleanup_old_scans",
        return_value=None,
    ), patch(
        "app.services.ui_snapshot_service.safe_publish_scan_bootstrap",
        return_value=None,
    ):
        scan_id = _create_auto_scan_for_published_run(
            feature_run_id=41,
            universe_name="all",
            screeners=["minervini"],
            criteria={},
            composite_method="weighted_average",
        )

    assert scan_id == "scan-123"
    assert created_scan["total_stocks"] == 2


def test_create_auto_scan_skips_resolving_live_universe_when_run_universe_count_exists():
    created_scan = {}
    resolve_calls = []

    class _CountQuery:
        def filter(self, *_args):
            return self

        def scalar(self):
            return 3

    class _AutoScanUoW:
        def __init__(self) -> None:
            self.session = SimpleNamespace(query=lambda *_args, **_kwargs: _CountQuery())
            self.scans = SimpleNamespace(
                get_by_idempotency_key=lambda _key: None,
                create=lambda **kwargs: created_scan.update(kwargs) or SimpleNamespace(
                    scan_id="scan-456",
                    universe_key="all",
                ),
            )
            self.universe = SimpleNamespace(
                resolve_symbols=lambda _universe_def: resolve_calls.append(_universe_def) or ["AAPL", "MSFT"]
            )
            self.feature_runs = SimpleNamespace(
                get_run=lambda _run_id: SimpleNamespace(
                    published_at=datetime(2026, 3, 16, 21, 30, tzinfo=timezone.utc),
                    stats=SimpleNamespace(passed_symbols=2),
                )
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def commit(self):
            return None

    with patch("app.database.SessionLocal"), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        return_value=_AutoScanUoW(),
    ), patch(
        "app.services.ui_snapshot_service.safe_publish_scan_bootstrap"
    ), patch(
        "app.services.scan_execution.cleanup_old_scans"
    ):
        scan_id = _create_auto_scan_for_published_run(
            feature_run_id=9,
            universe_name="all",
            screeners=["minervini"],
            criteria={},
            composite_method="average",
        )

    assert scan_id == "scan-456"
    assert created_scan["total_stocks"] == 3
    assert resolve_calls == []
    assert created_scan["passed_stocks"] == 2


def test_build_daily_snapshot_includes_cleaned_stale_runs_metadata():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._fail_stale_feature_runs",
        return_value=3,
    ):
        result = _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")

    assert result["cleaned_stale_runs"] == 3


def test_build_daily_snapshot_enriches_published_run_metadata_after_publish():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._enrich_feature_run_with_ibd_metadata",
        return_value={"updated_rows": 2},
    ) as mock_enrich:
        result = _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16")

    assert result["metadata_refresh"] == {"updated_rows": 2}
    mock_enrich.assert_called_once_with(
        feature_run_id=11,
        ranking_date=date(2026, 3, 16),
    )


def test_fail_stale_feature_runs_uses_utc_aware_cutoff():
    created_at = datetime(2026, 3, 30, 0, 0, tzinfo=timezone.utc)
    captured = {
        "cutoff": None,
        "stats": None,
        "warnings": None,
    }

    class _FakeStaleRunQuery:
        def __init__(self) -> None:
            self._criteria = ()

        def filter(self, *criteria):
            self._criteria = criteria
            captured["cutoff"] = criteria[1].right.value
            return self

        def order_by(self, *_args):
            return self

        def all(self):
            return [(17, created_at)]

    class _FakeCountQuery:
        def __init__(self, value: int) -> None:
            self._value = value

        def filter(self, *_args):
            return self

        def scalar(self):
            return self._value

    class _FakeSession:
        def __init__(self) -> None:
            self._queries = iter(
                [
                    _FakeStaleRunQuery(),
                    _FakeCountQuery(3),
                    _FakeCountQuery(1),
                ]
            )

        def query(self, *_entities):
            return next(self._queries)

    class _FakeFeatureRuns:
        def mark_failed(self, run_id, stats, warnings=()):
            captured["stats"] = (run_id, stats)
            captured["warnings"] = warnings

    class _FakeUoW:
        def __init__(self) -> None:
            self.session = _FakeSession()
            self.feature_runs = _FakeFeatureRuns()
            self.committed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def commit(self):
            self.committed = True

    fake_uow = _FakeUoW()

    with patch(
        "app.infra.db.uow.SqlUnitOfWork",
        return_value=fake_uow,
    ):
        cleaned = _fail_stale_feature_runs(
            session_factory=object(),
            stale_after_minutes=60,
        )

    assert cleaned == 1
    assert fake_uow.committed is True
    assert captured["cutoff"] is not None
    assert captured["cutoff"].tzinfo == timezone.utc
    assert captured["stats"] is not None
    run_id, stats = captured["stats"]
    assert run_id == 17
    assert stats.total_symbols == 3
    assert stats.processed_symbols == 1
    assert stats.failed_symbols == 0
    assert stats.duration_seconds > 0
    assert captured["warnings"] is not None


def test_build_daily_snapshot_publishes_market_activity():
    fake_use_case = _FakeUseCase()
    started = []
    completed = []

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        return_value=True,
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.mark_market_activity_started",
        side_effect=lambda *args, **kwargs: started.append(kwargs),
    ), patch(
        "app.interfaces.tasks.feature_store_tasks.mark_market_activity_completed",
        side_effect=lambda *args, **kwargs: completed.append(kwargs),
    ):
        result = _TASK_BODY(_FakeTask(), as_of_date_str="2026-03-16", market="US")

    assert result["status"] == "published"
    assert started[0]["stage_key"] == "scan"
    assert started[0]["lifecycle"] == "daily_refresh"
    assert started[0]["message"] == "Running market scan"
    assert completed[0]["stage_key"] == "scan"
    assert completed[0]["message"] == "Market scan ready"


def test_build_daily_snapshot_skips_disabled_market_by_default():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        return_value=False,
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            universe_name="market:hk",
            market="HK",
        )

    assert result["status"] == "skipped"
    assert result["market"] == "HK"
    assert result["reason"] == "market HK is disabled in local runtime preferences"
    assert fake_use_case.received_cmd is None


def test_build_daily_snapshot_allows_disabled_market_when_runtime_gate_is_ignored():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        return_value=False,
    ), patch(
        "app.interfaces.tasks.feature_store_tasks._create_auto_scan_for_published_run",
        return_value="auto-scan-001",
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            universe_name="market:hk",
            market="HK",
            static_daily_mode=True,
            ignore_runtime_market_gate=True,
        )

    assert result["status"] == "published"
    assert fake_use_case.received_cmd is not None


def test_build_daily_snapshot_does_not_ignore_market_gate_outside_static_mode():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ), patch(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        return_value=False,
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            universe_name="market:hk",
            market="HK",
            ignore_runtime_market_gate=True,
        )

    assert result["status"] == "skipped"
    assert result["market"] == "HK"
    assert result["reason"] == "market HK is disabled in local runtime preferences"
    assert fake_use_case.received_cmd is None


def test_build_daily_snapshot_uses_market_calendar_for_non_us_market():
    fake_use_case = _FakeUseCase()

    with patch(
        "app.interfaces.tasks.feature_store_tasks._is_market_trading_day",
        return_value=False,
    ) as mock_is_trading_day, patch(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        return_value=True,
    ), patch(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        return_value=fake_use_case,
    ), patch(
        "app.database.SessionLocal"
    ), patch(
        "app.infra.db.uow.SqlUnitOfWork",
        side_effect=lambda *_args, **_kwargs: _NonSkippingUoW(),
    ), patch(
        "app.infra.tasks.progress_sink.CeleryProgressSink",
        return_value=object(),
    ), patch(
        "app.domain.scanning.ports.NeverCancelledToken",
        return_value=object(),
    ):
        result = _TASK_BODY(
            _FakeTask(),
            as_of_date_str="2026-03-16",
            universe_name="market:hk",
            market="HK",
        )

    assert result["status"] == "skipped"
    assert result["reason"] == "not_trading_day"
    mock_is_trading_day.assert_called_once_with(date(2026, 3, 16), market="HK")
    assert fake_use_case.received_cmd is None
