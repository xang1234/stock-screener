from dataclasses import replace
from datetime import date, timedelta
from types import MappingProxyType

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.breadth_contributor import (
    MarketBreadthContributor,
    MarketBreadthContributorSnapshot,
)
from app.models.market_breadth import MarketBreadth
from app.services.breadth.contributors import contributor_calculation_signature
from app.services.breadth.persistence import BreadthPersistence
from app.services.breadth.types import (
    BreadthContributor,
    BreadthContributorSnapshotResult,
    BreadthDailyResult,
    BreadthEligibilityCounts,
    BreadthIndicatorValues,
)


def _database():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            MarketBreadth.__table__,
            MarketBreadthContributorSnapshot.__table__,
            MarketBreadthContributor.__table__,
        ],
    )
    return engine, sessionmaker(bind=engine)()


def _result(*, advancing: int = 8) -> BreadthDailyResult:
    return BreadthDailyResult(
        market="US",
        calculation_date=date(2026, 8, 21),
        values=BreadthIndicatorValues(
            stocks_up_4pct=3,
            stocks_down_4pct=1,
            advancing_count=advancing,
            declining_count=2,
            t2108_count=7,
            t2108_pct=70.0,
        ),
        eligibility=BreadthEligibilityCounts(
            advance_decline_eligible_count=10,
            stockbee_daily_eligible_count=9,
            t2108_eligible_count=10,
        ),
        broad_universe_count=12,
        eligibility_signature="a" * 64,
        stockbee_eligibility_signature="b" * 64,
    )


def _snapshot(
    calculation_date: date = date(2026, 8, 21),
) -> BreadthContributorSnapshotResult:
    return BreadthContributorSnapshotResult(
        market="US",
        calculation_date=calculation_date,
        calculation_revision=3,
        schema_id="breadth-contributors-v1",
        contributors=(
            BreadthContributor(
                symbol="AAA",
                company_name="Alpha",
                ibd_industry_group="Group A",
                daily_change_pct=8.0,
                signals=MappingProxyType({"up_4pct": 8.0}),
            ),
            BreadthContributor(
                symbol="BBB",
                company_name="Beta",
                ibd_industry_group="Group A",
                daily_change_pct=7.0,
                signals=MappingProxyType({"up_4pct": 7.0}),
            ),
            BreadthContributor(
                symbol="CCC",
                company_name="Gamma",
                ibd_industry_group="No Group",
                daily_change_pct=6.0,
                signals=MappingProxyType({"up_4pct": 6.0}),
            ),
            BreadthContributor(
                symbol="DDD",
                company_name="Delta",
                ibd_industry_group="Group D",
                daily_change_pct=-5.0,
                signals=MappingProxyType({"down_4pct": -5.0}),
            ),
        ),
    )


def test_persistence_upserts_every_current_field_in_one_market_partition():
    _engine, db = _database()
    persistence = BreadthPersistence(db)

    persistence.upsert_daily(_result(), duration_seconds=1.25)
    persistence.upsert_daily(_result(advancing=9), duration_seconds=0.75)

    rows = db.query(MarketBreadth).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.advancing_count == 9
    assert row.advance_decline_eligible_count == 10
    assert row.broad_universe_count == 12
    assert row.total_stocks_scanned == 12
    assert row.stockbee_eligibility_signature == "b" * 64
    assert row.calculation_revision == 3
    assert row.calculation_duration_seconds == 0.75


def test_persistence_rolls_back_aggregate_when_contributor_counts_disagree():
    """Catches partial aggregate writes when contributor validation fails."""
    _engine, db = _database()
    persistence = BreadthPersistence(db)
    persistence.upsert_daily(
        _result(advancing=8),
        contributor_snapshot=_snapshot(),
        duration_seconds=1.0,
    )
    mismatched = replace(_snapshot(), contributors=_snapshot().contributors[1:])

    with pytest.raises(ValueError, match="stocks_up_4pct"):
        persistence.upsert_daily(
            _result(advancing=9),
            contributor_snapshot=mismatched,
            duration_seconds=0.5,
        )

    assert db.query(MarketBreadth).one().advancing_count == 8
    assert db.query(MarketBreadthContributorSnapshot).count() == 1
    assert db.query(MarketBreadthContributor).count() == 4


def test_persistence_stores_contributor_provenance_with_aggregate():
    _engine, db = _database()
    snapshot = _snapshot()

    BreadthPersistence(db).upsert_daily(
        _result(),
        contributor_snapshot=snapshot,
        duration_seconds=1.0,
    )

    assert (
        db.query(MarketBreadth).one().contributor_calculation_signature
        == contributor_calculation_signature(snapshot.contributors)
    )


def test_daily_upsert_deletes_stale_snapshot_when_fresh_snapshot_is_missing():
    """Catches serving stale drilldown rows after aggregate-only recalculation."""
    _engine, db = _database()
    persistence = BreadthPersistence(db)
    persistence.upsert_daily(
        _result(advancing=8),
        contributor_snapshot=_snapshot(),
        duration_seconds=1.0,
    )

    persistence.upsert_daily(
        _result(advancing=9),
        contributor_snapshot=None,
        duration_seconds=0.5,
    )

    assert db.query(MarketBreadth).one().advancing_count == 9
    assert db.query(MarketBreadthContributorSnapshot).count() == 0
    assert db.query(MarketBreadthContributor).count() == 0
    assert db.query(MarketBreadth).one().contributor_calculation_signature is None


def test_bulk_upsert_deletes_stale_snapshot_when_fresh_snapshots_are_missing():
    """Catches normal backfills retaining stale contributor documents."""
    _engine, db = _database()
    persistence = BreadthPersistence(db)
    persistence.upsert_daily(
        _result(advancing=8),
        contributor_snapshot=_snapshot(),
        duration_seconds=1.0,
    )

    persistence.upsert_many(
        (_result(advancing=9),),
        contributor_snapshots_by_date=None,
        duration_seconds_by_date={date(2026, 8, 21): 0.5},
    )

    assert db.query(MarketBreadth).one().advancing_count == 9
    assert db.query(MarketBreadthContributorSnapshot).count() == 0
    assert db.query(MarketBreadthContributor).count() == 0


def test_contributor_retention_keeps_twenty_dates_and_all_aggregate_history():
    """Catches pruning aggregate history or retaining stale contributor dates."""
    _engine, db = _database()
    persistence = BreadthPersistence(db)
    first_date = date(2026, 7, 1)

    for offset in range(21):
        calculation_date = first_date + timedelta(days=offset)
        persistence.upsert_daily(
            replace(_result(), calculation_date=calculation_date),
            contributor_snapshot=_snapshot(calculation_date),
            duration_seconds=0.1,
        )

    assert db.query(MarketBreadth).count() == 21
    assert db.query(MarketBreadthContributorSnapshot).count() == 20
    retained_dates = {
        row.date for row in db.query(MarketBreadthContributorSnapshot).all()
    }
    assert first_date not in retained_dates
    assert first_date + timedelta(days=20) in retained_dates


def test_contributor_only_replacement_rejects_equal_count_identity_drift():
    _engine, db = _database()
    persistence = BreadthPersistence(db)
    original = _snapshot()
    persistence.upsert_daily(
        _result(),
        contributor_snapshot=original,
        duration_seconds=0.1,
    )
    changed = replace(
        original,
        contributors=(
            replace(original.contributors[0], symbol="ZZZ"),
            *original.contributors[1:],
        ),
    )

    with pytest.raises(ValueError, match="provenance"):
        persistence.replace_contributor_snapshots(
            (changed,),
            expected_aggregates={changed.calculation_date: _result()},
        )

    stored_symbols = {
        row.symbol for row in db.query(MarketBreadthContributor).all()
    }
    assert stored_symbols == {"AAA", "BBB", "CCC", "DDD"}


def test_persistence_stores_participation_fields():
    _engine, db = _database()
    result = _result()
    result = replace(
        result,
        values=replace(
            result.values,
            above_sma200_count=4,
            above_sma200_pct=40.0,
            down_20pct_from_high_count=3,
            down_20pct_from_high_pct=30.0,
            up_4pct_near_high_count=2,
            above_sma20_count=6,
            above_sma20_pct=60.0,
            above_sma50_count=5,
            above_sma50_pct=50.0,
            trend_template_entry_count=1,
        ),
        eligibility=replace(
            result.eligibility,
            sma200_eligible_count=10,
            off_high_eligible_count=10,
            sma20_eligible_count=10,
            sma50_eligible_count=10,
            trend_template_entry_eligible_count=8,
        ),
    )

    BreadthPersistence(db).upsert_daily(result, duration_seconds=None)

    row = db.query(MarketBreadth).one()
    assert (row.above_sma200_pct, row.down_20pct_from_high_pct) == (40.0, 30.0)
    assert (row.sma200_eligible_count, row.off_high_eligible_count) == (10, 10)
    assert row.up_4pct_near_high_count == 2
    assert (row.above_sma20_pct, row.above_sma50_pct) == (60.0, 50.0)
    assert (row.sma20_eligible_count, row.sma50_eligible_count) == (10, 10)
    assert (row.trend_template_entry_count, row.trend_template_entry_eligible_count) == (1, 8)
