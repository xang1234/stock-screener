"""Guarded balanced Market RS rollout tests."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.market_rs_inputs import MarketRsInputUnavailable
from app.services.market_rs_static_artifact_validator import (
    MarketRsStaticArtifactValidator,
)
from app.services.market_rs_rollout_service import (
    ActivationValidationReport,
    MarketRsActivationRejected,
    MarketRsRolloutService,
)


def _service(
    *,
    calendar=None,
    loader=None,
    repository=None,
    snapshot=None,
    groups=None,
    feature_factory=None,
):
    kwargs = {}
    if feature_factory is not None:
        kwargs["feature_run_repository_factory"] = feature_factory
    return MarketRsRolloutService(
        calendar_service=calendar or MagicMock(),
        input_loader=loader or MagicMock(),
        market_rs_snapshot_service=snapshot or MagicMock(),
        market_rs_repository=repository or MagicMock(),
        canonical_group_service=groups or MagicMock(),
        **kwargs,
    )


def test_candidate_dates_start_at_first_probe_with_two_eligible_stocks():
    sessions = [date(2026, 4, 8), date(2026, 4, 9), date(2026, 4, 10)]
    calendar = MagicMock()
    calendar.trading_days.return_value = sessions
    loader = MagicMock()
    loader.load.side_effect = [
        MarketRsInputUnavailable(
            "too early",
            reason_code="session_anchors_unavailable",
            diagnostics={},
        ),
        SimpleNamespace(excess_returns_by_symbol={"AAA": {}, "BBB": {}}),
    ]
    service = _service(calendar=calendar, loader=loader)
    service.backfill_service._earliest_available_price_date = (  # type: ignore[method-assign]
        lambda _db, _market: sessions[0]
    )

    assert service.earliest_backfillable_date(
        MagicMock(),
        market="us",
        through_date=sessions[-1],
    ) == sessions[1]
    assert service.candidate_dates(
        MagicMock(),
        market="US",
        through_date=sessions[-1],
        first_valid_date=sessions[1],
    ) == (sessions[1], sessions[2])


def test_earliest_backfillable_date_does_not_hide_unexpected_loader_errors():
    session = date(2026, 4, 10)
    calendar = MagicMock()
    calendar.trading_days.return_value = [session]
    loader = MagicMock()
    loader.load.side_effect = RuntimeError("database connection lost")
    service = _service(calendar=calendar, loader=loader)
    service.backfill_service._earliest_available_price_date = (  # type: ignore[method-assign]
        lambda _db, _market: session
    )

    with pytest.raises(RuntimeError, match="database connection lost"):
        service.earliest_backfillable_date(
            MagicMock(),
            market="US",
            through_date=session,
        )


def test_backfill_resumes_completed_stock_run_and_reports_all_failures(monkeypatch):
    dates = (date(2026, 4, 8), date(2026, 4, 9), date(2026, 4, 10))
    completed_run = SimpleNamespace(id=10, eligible_symbol_count=2)
    retried_run = SimpleNamespace(id=11, eligible_symbol_count=2)
    absent_run = SimpleNamespace(id=12, eligible_symbol_count=2)
    repository = MagicMock()
    repository.get_completed_exact.side_effect = [completed_run, None, None]
    snapshot = MagicMock()
    snapshot.calculate.side_effect = [completed_run, retried_run, absent_run]
    groups = MagicMock()
    groups.calculate_and_store.side_effect = [
        [{"market_rs_run_id": 10}],
        RuntimeError("group aggregation failed"),
        [{"market_rs_run_id": 12}, {"market_rs_run_id": 12}],
    ]
    service = _service(repository=repository, snapshot=snapshot, groups=groups)
    monkeypatch.setattr(
        service.backfill_service,
        "earliest_backfillable_date",
        lambda *a, **k: dates[0],
    )
    monkeypatch.setattr(
        service.backfill_service,
        "candidate_dates",
        lambda *a, **k: dates,
    )

    report = service.backfill(MagicMock(), market="US", through_date=dates[-1])

    assert report.candidate_count == 3
    assert report.completed_count == 2
    assert report.failed_count == 1
    assert report.latest_run_id == 12
    assert [item.as_of_date for item in report.results] == list(dates)
    assert report.results[1].reason_code == "group_calculation_failed"
    assert [call.kwargs["as_of_date"] for call in snapshot.calculate.call_args_list] == [
        dates[0],
        dates[1],
        dates[2],
    ]
    assert all(
        call.kwargs["rebuild_incompatible"] is True
        for call in snapshot.calculate.call_args_list
    )


def test_rejected_activation_rolls_back_without_moving_either_pointer(tmp_path):
    repository = MagicMock()
    feature_repository = MagicMock()
    service = _service(repository=repository)
    db = MagicMock()
    validation = ActivationValidationReport(
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=date(2026, 4, 10),
        first_valid_date=date(2026, 4, 8),
        candidate_count=3,
        latest_market_rs_run_id=42,
        latest_universe_hash="universe-a",
        feature_run_id=99,
        feature_universe_hash="feature-a",
        static_bundle_sha256="bundle-a",
        errors=("candidate trading-date gap",),
    )

    with pytest.raises(MarketRsActivationRejected, match="candidate trading-date gap"):
        service.activate(
            db,
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            feature_run_id=99,
            validation=validation,
            static_staging_dir=tmp_path,
        )

    repository.activate_formula.assert_not_called()
    feature_repository.repoint_published.assert_not_called()
    db.commit.assert_not_called()


def test_activation_rejects_manifest_changed_after_validation(tmp_path):
    repository = MagicMock()
    service = _service(repository=repository)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version":"static-site-v3"}', encoding="utf-8")
    validated_hash = MarketRsStaticArtifactValidator.bundle_fingerprint(
        tmp_path,
        market="US",
    ).sha256
    manifest_path.write_text('{"schema_version":"changed"}', encoding="utf-8")
    validation = ActivationValidationReport(
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=date(2026, 4, 10),
        first_valid_date=date(2026, 4, 8),
        candidate_count=3,
        latest_market_rs_run_id=42,
        latest_universe_hash="universe-a",
        feature_run_id=99,
        feature_universe_hash="feature-a",
        static_bundle_sha256=validated_hash,
        errors=(),
    )

    with pytest.raises(MarketRsActivationRejected, match="changed after validation"):
        service.activate(
            MagicMock(),
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            feature_run_id=99,
            validation=validation,
            static_staging_dir=tmp_path,
        )

    repository.activate_formula.assert_not_called()


def test_activation_rejects_market_bundle_file_changed_after_validation(
    tmp_path,
    monkeypatch,
):
    repository = MagicMock()
    service = _service(repository=repository)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version":"static-site-v3"}', encoding="utf-8")
    groups_path = tmp_path / "markets" / "us" / "groups.json"
    groups_path.parent.mkdir(parents=True)
    groups_path.write_text('{"rankings":[]}', encoding="utf-8")
    bundle_hash = MarketRsStaticArtifactValidator.bundle_fingerprint(
        tmp_path,
        market="US",
    ).sha256
    validation = ActivationValidationReport(
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=date(2026, 4, 10),
        first_valid_date=date(2026, 4, 8),
        candidate_count=3,
        latest_market_rs_run_id=42,
        latest_universe_hash="universe-a",
        feature_run_id=99,
        feature_universe_hash="feature-a",
        static_bundle_sha256=bundle_hash,
        errors=(),
    )
    monkeypatch.setattr(service.validator, "revalidate_static", lambda *a, **k: ())
    groups_path.write_text('{"rankings":[{"rank":1}]}', encoding="utf-8")

    with pytest.raises(MarketRsActivationRejected, match="bundle changed"):
        service.activate(
            MagicMock(),
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            feature_run_id=99,
            validation=validation,
            static_staging_dir=tmp_path,
        )

    repository.activate_formula.assert_not_called()


def test_validation_collects_feature_and_static_errors_without_short_circuiting(
    monkeypatch,
    tmp_path,
):
    through_date = date(2026, 4, 10)
    run = SimpleNamespace(
        id=42,
        universe_hash="universe-a",
        eligible_symbol_count=2,
        rows=[],
    )
    feature_repository = MagicMock()
    feature_repository.get_run.return_value = SimpleNamespace(
        id=99,
        status=SimpleNamespace(value="completed"),
        as_of_date=date(2026, 4, 9),
        universe_hash="feature-a",
        config={},
    )
    service = _service(feature_factory=lambda _db: feature_repository)
    monkeypatch.setattr(
        service.backfill_service,
        "earliest_backfillable_date",
        lambda *args, **kwargs: through_date,
    )
    monkeypatch.setattr(
        service.backfill_service,
        "candidate_dates",
        lambda *args, **kwargs: (through_date,),
    )
    monkeypatch.setattr(
        service.validator,
        "_validate_run_and_groups",
        lambda *args, **kwargs: run,
    )

    validation = service.validate_activation(
        MagicMock(),
        market="US",
        through_date=through_date,
        feature_run_id=99,
        static_staging_dir=tmp_path / "missing-stage",
    )

    assert validation.ok is False
    assert any("not published for the activation date" in error for error in validation.errors)
    assert any("rs_formula_version" in error for error in validation.errors)
    assert any("Missing staged static-site-v3 manifest" in error for error in validation.errors)


def test_successful_activation_revalidates_then_commits_both_pointers(
    monkeypatch, tmp_path
):
    events: list[str] = []
    repository = MagicMock()
    repository.get_completed_exact.return_value = SimpleNamespace(
        id=42,
        universe_hash="universe-a",
        eligible_symbol_count=2,
    )
    repository.activate_formula.side_effect = lambda *a, **k: events.append("market")
    feature_repository = MagicMock()
    feature_repository.get_run.return_value = SimpleNamespace(
        id=99,
        status=SimpleNamespace(value="published"),
        universe_hash="feature-a",
        as_of_date=date(2026, 4, 10),
        config={
            "market": "US",
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 42,
            "rs_as_of_date": "2026-04-10",
            "rs_universe_size": 2,
        },
    )
    feature_repository.repoint_published.side_effect = (
        lambda *a, **k: events.append("feature")
    )
    service = _service(
        repository=repository,
        feature_factory=lambda _db: feature_repository,
    )
    db = MagicMock()
    db.commit.side_effect = lambda: events.append("commit")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version":"static-site-v3"}', encoding="utf-8")
    bundle_hash = MarketRsStaticArtifactValidator.bundle_fingerprint(
        tmp_path,
        market="US",
    ).sha256
    revalidate = MagicMock(return_value=())
    monkeypatch.setattr(service.validator, "revalidate_static", revalidate)
    validation = ActivationValidationReport(
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=date(2026, 4, 10),
        first_valid_date=date(2026, 4, 8),
        candidate_count=3,
        latest_market_rs_run_id=42,
        latest_universe_hash="universe-a",
        feature_run_id=99,
        feature_universe_hash="feature-a",
        static_bundle_sha256=bundle_hash,
        errors=(),
    )

    service.activate(
        db,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        feature_run_id=99,
        validation=validation,
        static_staging_dir=tmp_path,
    )

    assert events == ["market", "feature", "commit"]
    feature_repository.repoint_published.assert_called_once_with(
        99,
        pointer_key="latest_published_market:US",
    )
    revalidate.assert_called_once()
