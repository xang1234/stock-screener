"""Atomic Market RS + Feature pointer activation integration coverage."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from unittest.mock import MagicMock

import pytest

from app.domain.feature_store.models import RunStatus
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.models.relative_strength import MarketRsFormulaPointer, MarketRsRun
from app.infra.db.repositories.feature_run_repo import SqlFeatureRunRepository
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.market_rs_rollout_service import (
    ActivationValidationReport,
    MarketRsRolloutService,
)


def _seed_activation_candidates(db_session):
    through_date = date(2026, 4, 10)
    db_session.add(
        MarketRsFormulaPointer(
            market="US",
            formula_version=LEGACY_RS_FORMULA_VERSION,
        )
    )
    rs_run = MarketRsRun(
        market="US",
        as_of_date=through_date,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        status="completed",
        benchmark_symbol="SPY",
        benchmark_as_of_date=through_date,
        universe_hash="universe-a",
        expected_symbol_count=0,
        eligible_symbol_count=0,
        excluded_symbol_count=0,
        diagnostics_json={"price_basis": "adj_close_only"},
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(rs_run)
    db_session.flush()
    old_feature = FeatureRun(
        as_of_date=date(2026, 4, 9),
        run_type="daily_snapshot",
        status=RunStatus.PUBLISHED.value,
        universe_hash="legacy-feature",
        config_json={"market": "US", "rs_formula_version": LEGACY_RS_FORMULA_VERSION},
        published_at=datetime.now(timezone.utc),
    )
    candidate = FeatureRun(
        as_of_date=through_date,
        run_type="daily_snapshot",
        status=RunStatus.PUBLISHED.value,
        universe_hash="feature-a",
        config_json={
            "market": "US",
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": rs_run.id,
            "rs_as_of_date": through_date.isoformat(),
        },
        published_at=datetime.now(timezone.utc),
    )
    db_session.add_all([old_feature, candidate])
    db_session.flush()
    db_session.add(
        FeatureRunPointer(
            key="latest_published_market:US",
            run_id=old_feature.id,
        )
    )
    db_session.commit()
    return rs_run.id, candidate.id, old_feature.id


def _validation(
    rs_run_id: int,
    feature_run_id: int,
    manifest_hash: str,
) -> ActivationValidationReport:
    return ActivationValidationReport(
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=date(2026, 4, 10),
        first_valid_date=date(2026, 4, 10),
        candidate_count=1,
        latest_market_rs_run_id=rs_run_id,
        latest_universe_hash="universe-a",
        feature_run_id=feature_run_id,
        feature_universe_hash="feature-a",
        static_manifest_sha256=manifest_hash,
        errors=(),
    )


def _service(repository, feature_factory):
    return MarketRsRolloutService(
        calendar_service=MagicMock(),
        input_loader=MagicMock(),
        market_rs_snapshot_service=MagicMock(),
        market_rs_repository=repository,
        canonical_group_service=MagicMock(),
        feature_run_repository_factory=feature_factory,
    )


def test_activation_switches_market_and_feature_pointers_in_one_commit(
    db_session, tmp_path, monkeypatch
):
    rs_run_id, candidate_id, _old_id = _seed_activation_candidates(db_session)
    service = _service(MarketRsRunRepository(), SqlFeatureRunRepository)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version":"static-site-v3"}', encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(service, "revalidate_static", lambda *args, **kwargs: ())

    service.activate(
        db_session,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        feature_run_id=candidate_id,
        validation=_validation(rs_run_id, candidate_id, manifest_hash),
        static_staging_dir=tmp_path,
    )

    db_session.expire_all()
    assert db_session.get(MarketRsFormulaPointer, "US").formula_version == (
        BALANCED_RS_FORMULA_VERSION
    )
    assert db_session.get(
        FeatureRunPointer,
        "latest_published_market:US",
    ).run_id == candidate_id


def test_failure_after_market_pointer_flush_rolls_back_both_pointers(
    db_session, tmp_path, monkeypatch
):
    rs_run_id, candidate_id, old_id = _seed_activation_candidates(db_session)

    class _FailingFeatureRepository(SqlFeatureRunRepository):
        def repoint_published(self, run_id, pointer_key="latest_published"):
            raise RuntimeError("pointer write failed")

    service = _service(MarketRsRunRepository(), _FailingFeatureRepository)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version":"static-site-v3"}', encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(service, "revalidate_static", lambda *args, **kwargs: ())

    with pytest.raises(RuntimeError, match="pointer write failed"):
        service.activate(
            db_session,
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            feature_run_id=candidate_id,
            validation=_validation(rs_run_id, candidate_id, manifest_hash),
            static_staging_dir=tmp_path,
        )

    db_session.expire_all()
    assert db_session.get(MarketRsFormulaPointer, "US").formula_version == (
        LEGACY_RS_FORMULA_VERSION
    )
    assert db_session.get(
        FeatureRunPointer,
        "latest_published_market:US",
    ).run_id == old_id
