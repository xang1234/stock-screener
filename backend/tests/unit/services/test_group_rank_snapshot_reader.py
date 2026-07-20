from datetime import date

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
    RsPublicationIdentity,
)
from app.infra.db.models.relative_strength import MarketRsRun
from app.models.industry import IBDGroupRank
from app.services.group_rank_snapshot_reader import (
    GroupRankSnapshotReader,
    GroupSnapshotIntegrityError,
)


AS_OF = date(2026, 4, 10)


def _run(db_session, *, run_id=41, as_of_date=AS_OF):
    row = MarketRsRun(
        id=run_id,
        market="US",
        as_of_date=as_of_date,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        status="completed",
        benchmark_symbol="SPY",
        benchmark_as_of_date=as_of_date,
        universe_hash="reader-test",
        expected_symbol_count=3,
        eligible_symbol_count=3,
        excluded_symbol_count=0,
        diagnostics_json={"price_basis": "adj_close_only"},
    )
    db_session.add(row)
    db_session.flush()
    return row


def _rank(db_session, *, formula, rank, run_id=None, group="Software"):
    db_session.add(
        IBDGroupRank(
            market="US",
            industry_group=group,
            date=AS_OF,
            rank=rank,
            avg_rs_rating=88.0,
            num_stocks=3,
            num_stocks_rs_above_80=2,
            top_symbol="AAA",
            top_rs_rating=99.0,
            rs_formula_version=formula,
            market_rs_run_id=run_id,
        )
    )


def test_identity_normalizes_market_and_rejects_blank_formula():
    identity = GroupSnapshotIdentity(" hk ", AS_OF, BALANCED_RS_FORMULA_VERSION)
    assert identity.market == "HK"
    with pytest.raises(ValueError, match="formula_version"):
        GroupSnapshotIdentity("US", AS_OF, " ")


def test_load_exact_never_crosses_formula(db_session):
    run = _run(db_session)
    _rank(db_session, formula=BALANCED_RS_FORMULA_VERSION, rank=1, run_id=run.id)
    _rank(db_session, formula=LEGACY_RS_FORMULA_VERSION, rank=9, group="Legacy")
    db_session.commit()

    rows = GroupRankSnapshotReader().load_exact(
        db_session,
        identity=GroupSnapshotIdentity("US", AS_OF, BALANCED_RS_FORMULA_VERSION),
    )

    assert [row["industry_group"] for row in rows] == ["Software"]
    assert rows[0]["market_rs_run_id"] == run.id


def test_balanced_rows_must_share_the_exact_completed_run(db_session):
    first = _run(db_session, run_id=41)
    _run(db_session, run_id=42, as_of_date=date(2026, 4, 9))
    _rank(db_session, formula=BALANCED_RS_FORMULA_VERSION, rank=1, run_id=first.id)
    _rank(
        db_session,
        formula=BALANCED_RS_FORMULA_VERSION,
        rank=2,
        run_id=42,
        group="Hardware",
    )
    db_session.commit()

    with pytest.raises(GroupSnapshotIntegrityError, match="Market RS run"):
        GroupRankSnapshotReader().load_exact(
            db_session,
            identity=GroupSnapshotIdentity(
                "US", AS_OF, BALANCED_RS_FORMULA_VERSION
            ),
        )


def test_load_publication_rejects_a_different_market_rs_run(db_session):
    run = _run(db_session, run_id=41)
    _rank(db_session, formula=BALANCED_RS_FORMULA_VERSION, rank=1, run_id=run.id)
    db_session.commit()

    expected = RsPublicationIdentity(
        snapshot=GroupSnapshotIdentity(
            "US",
            AS_OF,
            BALANCED_RS_FORMULA_VERSION,
        ),
        market_rs_run_id=42,
        universe_size=3,
    )

    with pytest.raises(GroupSnapshotIntegrityError, match="expected Market RS run"):
        GroupRankSnapshotReader().load_publication(
            db_session,
            publication=expected,
        )


def test_available_dates_is_formula_scoped(db_session):
    _rank(db_session, formula=LEGACY_RS_FORMULA_VERSION, rank=1)
    db_session.commit()
    assert GroupRankSnapshotReader().available_dates(
        db_session,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=AS_OF,
    ) == ()
