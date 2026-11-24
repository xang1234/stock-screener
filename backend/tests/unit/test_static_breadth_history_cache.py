from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.market_breadth import MarketBreadth
from app.scripts.static_breadth_history_cache import (
    RECOMPUTED_TAIL_SESSIONS,
    export_bundle,
    import_bundle,
)
from app.services.breadth import CURRENT_BREADTH_CALCULATION_REVISION

START = date(2026, 3, 2)


@pytest.fixture
def make_session():
    def factory():
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine, tables=[MarketBreadth.__table__])
        return sessionmaker(bind=engine)()

    return factory


def _row(day: int, *, market="US", revision=CURRENT_BREADTH_CALCULATION_REVISION):
    return MarketBreadth(
        market=market,
        date=START + timedelta(days=day),
        stocks_up_4pct=day,
        stocks_down_4pct=1,
        ratio_5day=1.5,
        eligibility_signature=f"sig-{day}",
        calculation_revision=revision,
    )


def test_round_trip_restores_all_but_the_recomputed_tail(make_session):
    source = make_session()
    source.add_all([_row(day) for day in range(30)])
    source.add_all([_row(0, market="JP"), _row(30, revision=CURRENT_BREADTH_CALCULATION_REVISION - 1)])
    source.commit()
    bundle = export_bundle(source, market="US")

    target = make_session()
    target.add(_row(0))  # already present: left untouched
    target.commit()
    result = import_bundle(target, bundle, market="US")

    restored = target.query(MarketBreadth).order_by(MarketBreadth.date).all()
    assert len(bundle["rows"]) == 30  # other markets and stale revisions stay out
    assert result == {
        "status": "restored",
        "restored_dates": 30 - RECOMPUTED_TAIL_SESSIONS - 1,
        "dropped_tail_dates": RECOMPUTED_TAIL_SESSIONS,
    }
    assert [row.date for row in restored] == [
        START + timedelta(days=day) for day in range(30 - RECOMPUTED_TAIL_SESSIONS)
    ]
    assert (restored[5].stocks_up_4pct, restored[5].ratio_5day, restored[5].eligibility_signature) == (
        5,
        1.5,
        "sig-5",
    )


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"market": "JP"}, "incompatible_bundle"),
        # v1 bundles hold 52-week-high counts from before the missing-High fix.
        ({"schema": "static-breadth-history-v1"}, "incompatible_bundle"),
        ({"calculation_revision": CURRENT_BREADTH_CALCULATION_REVISION + 1}, "calculation_revision_changed"),
    ],
)
def test_incompatible_bundles_restore_nothing(make_session, patch, reason):
    source = make_session()
    source.add_all([_row(day) for day in range(30)])
    source.commit()
    bundle = export_bundle(source, market="US") | patch

    target = make_session()
    assert import_bundle(target, bundle, market="US") == {"status": "skipped", "reason": reason}
    assert target.query(MarketBreadth).count() == 0


@pytest.mark.parametrize(
    "added_column", ["above_sma200_pct", "above_sma20_pct", "trend_template_entry_count"]
)
def test_bundle_from_before_a_column_was_added_restores_nothing(
    make_session, added_column
):
    source = make_session()
    source.add_all([_row(day) for day in range(30)])
    source.commit()
    bundle = export_bundle(source, market="US")
    for row in bundle["rows"]:
        del row[added_column]

    target = make_session()
    assert import_bundle(target, bundle, market="US") == {
        "status": "skipped",
        "reason": "columns_changed",
    }
    assert target.query(MarketBreadth).count() == 0
