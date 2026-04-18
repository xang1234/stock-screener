"""Task-level regression tests for feature store Celery entrypoints."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.config import settings
from app.interfaces.tasks.feature_store_tasks import (
    _create_auto_scan_for_published_run,
    _enrich_feature_run_with_ibd_metadata,
    _fail_stale_feature_runs,
    _upsert_feature_run_pointer,
    build_daily_snapshot,
)
from app.domain.scanning.defaults import get_default_scan_profile
from app.schemas.universe import UniverseType
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.industry import IBDGroupRank, IBDIndustryGroup


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
    assert fake_use_case.received_cmd.universe_def.type == UniverseType.ALL


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
    defaults = get_default_scan_profile()

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
        universe_name=get_default_scan_profile()["universe"],
        screeners=get_default_scan_profile()["screeners"],
        criteria=get_default_scan_profile()["criteria"],
        composite_method=get_default_scan_profile()["composite_method"],
    )
    assert lookup_calls == [{
        "input_hash": "same-input",
        "universe_hash": "same-universe",
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
        universe_name=get_default_scan_profile()["universe"],
        screeners=get_default_scan_profile()["screeners"],
        criteria=get_default_scan_profile()["criteria"],
        composite_method=get_default_scan_profile()["composite_method"],
    )


def test_build_daily_snapshot_static_daily_mode_requires_bulk_prefetch():
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
    assert started[0]["stage_key"] == "snapshot"
    assert started[0]["lifecycle"] == "daily_refresh"
    assert completed[0]["stage_key"] == "snapshot"


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
            ignore_runtime_market_gate=True,
        )

    assert result["status"] == "published"
    assert fake_use_case.received_cmd is not None
