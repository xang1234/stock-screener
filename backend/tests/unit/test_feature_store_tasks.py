"""Task-level regression tests for feature store Celery entrypoints."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded

from app.interfaces.tasks.feature_store_tasks import (
    _fail_stale_feature_runs,
    build_daily_snapshot,
)
from app.domain.scanning.defaults import get_default_scan_profile
from app.schemas.universe import UniverseType


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
            row_count=2,
            duration_seconds=12.5,
            dq_passed=True,
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
