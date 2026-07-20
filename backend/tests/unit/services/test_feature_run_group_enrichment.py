from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.industry import IBDIndustryGroup
from app.services.feature_run_group_enrichment import (
    FeatureRunGroupEnrichmentService,
)
from app.services.group_rank_snapshot_reader import GroupSnapshotUnavailable


AS_OF = date(2026, 4, 10)


class _Taxonomy:
    def get(self, *args, **kwargs):
        raise AssertionError("US enrichment must use the stored IBD taxonomy")


def _tables(engine):
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            IBDIndustryGroup.__table__,
        ],
    )


def _row(*, run_id: int, details: dict) -> StockFeatureDaily:
    return StockFeatureDaily(
        run_id=run_id,
        symbol="AAA",
        as_of_date=AS_OF,
        composite_score=90.0,
        overall_rating=5,
        passes_count=4,
        details_json=details,
    )


def test_enrichment_reads_the_feature_runs_exact_formula_snapshot():
    engine = create_engine("sqlite:///:memory:")
    _tables(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add_all([
            FeatureRun(
                id=31,
                as_of_date=AS_OF,
                run_type="daily_snapshot",
                status="published",
                config_json={
                    "market": "US",
                    "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
                    "market_rs_run_id": 42,
                    "rs_as_of_date": AS_OF.isoformat(),
                    "rs_universe_size": 1,
                },
            ),
            _row(run_id=31, details={"ibd_group_rank": 99}),
            IBDIndustryGroup(symbol="AAA", industry_group="Software"),
        ])
        db.commit()

    publications = []

    class _Reader:
        def load_publication(self, db, *, publication, include_top_symbol_names):
            publications.append(publication)
            return [{"industry_group": "Software", "rank": 7}]

    stats = FeatureRunGroupEnrichmentService(
        session_factory=factory,
        taxonomy_service=_Taxonomy(),
        snapshot_reader=_Reader(),
        batch_size=10,
    ).enrich(feature_run_id=31, ranking_date=AS_OF)

    with factory() as db:
        stored = db.query(StockFeatureDaily).filter_by(run_id=31, symbol="AAA").one()
    assert publications[0].snapshot.formula_version == BALANCED_RS_FORMULA_VERSION
    assert publications[0].market_rs_run_id == 42
    assert publications[0].universe_size == 1
    assert stored.details_json["ibd_group_rank"] == 7
    assert stats["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    engine.dispose()


def test_missing_exact_snapshot_rolls_back_without_erasing_previous_rank():
    rollback_calls = []

    class _TrackingSession(Session):
        def rollback(self):
            rollback_calls.append(True)
            return super().rollback()

    engine = create_engine("sqlite:///:memory:")
    _tables(engine)
    factory = sessionmaker(bind=engine, class_=_TrackingSession)
    with factory() as db:
        db.add_all([
            FeatureRun(
                id=32,
                as_of_date=AS_OF,
                run_type="daily_snapshot",
                status="published",
                config_json={"market": "US"},
            ),
            _row(
                run_id=32,
                details={
                    "ibd_industry_group": "Software",
                    "ibd_group_rank": 4,
                    "ibd_group_rank_date": "2026-04-09",
                },
            ),
            IBDIndustryGroup(symbol="AAA", industry_group="Software"),
        ])
        db.commit()

    class _MissingReader:
        def load_publication(self, db, *, publication, include_top_symbol_names):
            assert publication.snapshot.formula_version == LEGACY_RS_FORMULA_VERSION
            raise GroupSnapshotUnavailable(publication.snapshot)

    service = FeatureRunGroupEnrichmentService(
        session_factory=factory,
        taxonomy_service=_Taxonomy(),
        snapshot_reader=_MissingReader(),
        batch_size=10,
    )
    with pytest.raises(GroupSnapshotUnavailable):
        service.enrich(feature_run_id=32, ranking_date=AS_OF)

    with factory() as db:
        stored = db.query(StockFeatureDaily).filter_by(run_id=32, symbol="AAA").one()
    assert rollback_calls
    assert stored.details_json["ibd_group_rank"] == 4
    assert stored.details_json["ibd_group_rank_date"] == "2026-04-09"
    engine.dispose()


def test_group_less_market_enrichment_is_a_noop():
    engine = create_engine("sqlite:///:memory:")
    _tables(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add_all(
            [
                FeatureRun(
                    id=33,
                    as_of_date=AS_OF,
                    run_type="daily_snapshot",
                    status="published",
                    config_json={"market": "DE"},
                ),
                _row(
                    run_id=33,
                    details={
                        "ibd_industry_group": "Software",
                        "ibd_group_rank": 4,
                    },
                ),
            ]
        )
        db.commit()

    class _UnexpectedReader:
        @staticmethod
        def load_publication(*_args, **_kwargs):
            raise AssertionError("group snapshot must not be loaded")

    stats = FeatureRunGroupEnrichmentService(
        session_factory=factory,
        taxonomy_service=_Taxonomy(),
        snapshot_reader=_UnexpectedReader(),
        batch_size=10,
    ).enrich(feature_run_id=33, ranking_date=AS_OF)

    with factory() as db:
        stored = db.query(StockFeatureDaily).filter_by(run_id=33, symbol="AAA").one()
    assert stats["status"] == "skipped"
    assert stats["reason"] == "group_rankings_not_supported"
    assert stats["total_rows"] == 1
    assert stats["updated_rows"] == 0
    assert stored.details_json["ibd_group_rank"] == 4
    engine.dispose()
