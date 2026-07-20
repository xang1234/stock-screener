from __future__ import annotations

from datetime import date

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.models.relative_strength import MarketRsRun, StockRsSnapshot
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.stock_universe import StockUniverse
from app.services.canonical_group_ranking_service import CanonicalGroupRankingService
from app.services.ibd_industry_service import IBDIndustryService


AS_OF = date(2026, 4, 10)


def _seed_run(db, rows):
    run = MarketRsRun(
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        status="completed",
        benchmark_symbol="SPY",
        benchmark_as_of_date=AS_OF,
        universe_hash="canonical-test",
        expected_symbol_count=len(rows),
        eligible_symbol_count=len(rows),
        excluded_symbol_count=0,
        diagnostics_json={},
    )
    db.add(run)
    db.flush()
    for symbol, overall, one_month, three_month, cap in rows:
        db.add(
            StockRsSnapshot(
                run_id=run.id,
                symbol=symbol,
                overall_rs=overall,
                rs_1m=one_month,
                rs_3m=three_month,
                rs_6m=50,
                rs_9m=50,
                rs_12m=50,
                weighted_composite=float(overall),
                excess_return_1m=0.1,
                excess_return_3m=0.1,
                excess_return_6m=0.1,
                excess_return_9m=0.1,
                excess_return_12m=0.1,
            )
        )
        db.add(
            StockUniverse(
                symbol=symbol,
                name=f"{symbol} Inc.",
                market="US",
                exchange="NASDAQ",
                market_cap=cap,
                is_active=True,
                status="active",
            )
        )
    db.flush()
    return run


def _map(db, group, *symbols):
    for symbol in symbols:
        db.add(
            IBDIndustryGroup(
                symbol=symbol,
                industry_group=group,
                market="US",
                source="manual",
            )
        )
    db.flush()


def test_canonical_group_means_share_one_eligible_constituent_set(db_session):
    run = _seed_run(
        db_session,
        [
            ("AAA", 99, 10, 20, 100.0),
            ("BBB", 80, 20, 40, 200.0),
            ("CCC", 50, 30, 60, 300.0),
            ("DDD", 70, 90, 90, 400.0),
            ("EEE", 60, 80, 80, 500.0),
        ],
    )
    _map(db_session, "Leaders", "AAA", "BBB", "CCC", "ABSENT")
    _map(db_session, "Too Small", "DDD", "EEE")

    rankings = CanonicalGroupRankingService().calculate_and_store(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert [row["industry_group"] for row in rankings] == ["Leaders"]
    leaders = rankings[0]
    assert leaders["avg_rs_rating"] == pytest.approx((99 + 80 + 50) / 3)
    assert leaders["avg_rs_rating_1m"] == pytest.approx((10 + 20 + 30) / 3)
    assert leaders["avg_rs_rating_3m"] == pytest.approx((20 + 40 + 60) / 3)
    assert leaders["num_stocks"] == 3
    assert leaders["market_rs_run_id"] == run.id
    assert leaders["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION

    stored = db_session.query(IBDGroupRank).one()
    assert stored.avg_rs_rating == pytest.approx(leaders["avg_rs_rating"])
    assert stored.avg_rs_rating_1m == pytest.approx(leaders["avg_rs_rating_1m"])
    assert stored.avg_rs_rating_3m == pytest.approx(leaders["avg_rs_rating_3m"])


def test_canonical_main_rank_uses_only_unrounded_equal_weight_overall_rs(db_session):
    _seed_run(
        db_session,
        [
            ("A1", 90, 10, 10, 1.0),
            ("A2", 80, 10, 10, 1.0),
            ("A3", 70, 10, 10, 10_000.0),
            ("B1", 79, 99, 99, 10_000.0),
            ("B2", 79, 99, 99, 10_000.0),
            ("B3", 79, 99, 99, 10_000.0),
            ("C1", 80, 50, 50, 100.0),
            ("C2", 80, 50, 50, 100.0),
            ("C3", 80, 50, 50, 100.0),
        ],
    )
    _map(db_session, "Alpha", "A1", "A2", "A3")
    _map(db_session, "Beta", "B1", "B2", "B3")
    _map(db_session, "Charlie", "C1", "C2", "C3")

    rankings = CanonicalGroupRankingService().calculate_and_store(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert [row["industry_group"] for row in rankings] == [
        "Alpha",
        "Charlie",
        "Beta",
    ]
    assert rankings[0]["avg_rs_rating"] == rankings[1]["avg_rs_rating"] == 80.0
    assert rankings[0]["weighted_avg_rs_rating"] < rankings[1]["weighted_avg_rs_rating"]


def test_canonical_top_stock_ties_use_1m_then_cap_then_symbol(db_session):
    _seed_run(
        db_session,
        [
            ("ZZZ", 90, 50, 50, 1_000.0),
            ("BBB", 90, 60, 50, 10.0),
            ("AAA", 90, 60, 50, 10.0),
        ],
    )
    _map(db_session, "Tie Group", "ZZZ", "BBB", "AAA")

    [ranking] = CanonicalGroupRankingService().calculate_and_store(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert ranking["top_symbol"] == "AAA"


def test_canonical_group_aggregation_does_not_query_membership_per_group(
    db_session,
    monkeypatch,
):
    _seed_run(
        db_session,
        [
            ("AAA", 90, 50, 50, 100.0),
            ("BBB", 80, 50, 50, 100.0),
            ("CCC", 70, 50, 50, 100.0),
        ],
    )
    _map(db_session, "Bulk", "AAA", "BBB", "CCC")

    def reject_per_group_lookup(*args, **kwargs):
        raise AssertionError("canonical aggregation must bulk-load memberships")

    monkeypatch.setattr(
        IBDIndustryService,
        "get_group_symbols",
        reject_per_group_lookup,
    )

    [ranking] = CanonicalGroupRankingService().calculate_and_store(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert ranking["industry_group"] == "Bulk"


def test_canonical_and_legacy_group_rows_coexist(db_session):
    _seed_run(
        db_session,
        [
            ("AAA", 90, 50, 50, 100.0),
            ("BBB", 80, 50, 50, 100.0),
            ("CCC", 70, 50, 50, 100.0),
        ],
    )
    _map(db_session, "Coexist", "AAA", "BBB", "CCC")
    db_session.add(
        IBDGroupRank(
            market="US",
            industry_group="Coexist",
            date=AS_OF,
            rank=1,
            avg_rs_rating=95.0,
            num_stocks=3,
            num_stocks_rs_above_80=3,
            top_symbol="AAA",
            top_rs_rating=99.0,
            rs_formula_version=LEGACY_RS_FORMULA_VERSION,
        )
    )
    db_session.commit()

    CanonicalGroupRankingService().calculate_and_store(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    rows = db_session.query(IBDGroupRank).order_by(IBDGroupRank.rs_formula_version).all()
    assert {row.rs_formula_version for row in rows} == {
        LEGACY_RS_FORMULA_VERSION,
        BALANCED_RS_FORMULA_VERSION,
    }
