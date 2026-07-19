"""Acceptance proof for canonical stock, Group, static, and RRG parity."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.orm import sessionmaker

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    HORIZON_WEIGHTS,
    calculate_balanced_rs,
)
from app.infra.db.models.relative_strength import MarketRsFormulaPointer
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.industry import IBDIndustryGroup
from app.models.stock_universe import StockUniverse
from app.services.canonical_group_ranking_service import CanonicalGroupRankingService
from app.services.market_rs_reader import SqlMarketRsReader
from app.services.rrg_service import RRGParams, compute_group_rrg
from app.services.static_groups_payload_builder import (
    StaticGroupsSnapshot,
    build_static_groups_payload,
)

from .golden_fixtures import (
    CANONICAL_GROUP_RS_FIELDS,
    CANONICAL_STOCK_RS_FIELDS,
)


AS_OF = date(2026, 4, 10)
SYMBOLS = ("OLD", "A1", "A2", "B1", "B2", "B3")


def _returns(*, old_twelve_month: float = 10.0):
    return {
        "OLD": {
            "1m": -0.50,
            "3m": -0.40,
            "6m": 2.0,
            "9m": 5.0,
            "12m": old_twelve_month,
        },
        "A1": {"1m": 0.30, "3m": 0.35, "6m": 0.40, "9m": 0.45, "12m": 0.50},
        "A2": {"1m": 0.20, "3m": 0.25, "6m": 0.30, "9m": 0.35, "12m": 0.40},
        "B1": {"1m": 0.10, "3m": 0.15, "6m": 0.20, "9m": 0.25, "12m": 0.30},
        "B2": {"1m": 0.05, "3m": 0.10, "6m": 0.15, "9m": 0.20, "12m": 0.25},
        "B3": {"1m": 0.00, "3m": 0.05, "6m": 0.10, "9m": 0.15, "12m": 0.20},
    }


def _seed_canonical_snapshot(db_session):
    scores = calculate_balanced_rs(_returns())
    repository = MarketRsRunRepository()
    run = repository.start_or_restart(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        benchmark_symbol="SPY",
        benchmark_as_of_date=AS_OF,
        universe_hash="parity-universe",
        expected_symbol_count=len(scores),
    )
    repository.replace_rows(db_session, run, scores)
    repository.mark_completed(
        run,
        excluded_symbol_count=0,
        diagnostics={"price_basis": "adj_close_only"},
    )
    db_session.commit()

    for index, symbol in enumerate(SYMBOLS):
        db_session.add(
            StockUniverse(
                symbol=symbol,
                market="US",
                exchange="NASDAQ",
                market_cap=float((index + 1) * 100),
                is_active=True,
                status="active",
            )
        )
        db_session.add(
            IBDIndustryGroup(
                symbol=symbol,
                industry_group="Alpha" if index < 3 else "Beta",
                market="US",
                source="manual",
            )
        )
    db_session.add(
        MarketRsFormulaPointer(
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )
    )
    db_session.commit()
    return run


def _stock_payload(resolution, symbol: str) -> dict:
    ratings = resolution.ratings_by_symbol[symbol]
    return {
        **ratings,
        "rs_formula_version": resolution.formula_version,
        "market_rs_run_id": resolution.run_id,
        "rs_universe_size": resolution.universe_size,
    }


def test_stock_subset_consumers_and_static_projection_use_identical_market_percentiles(
    db_session,
):
    run = _seed_canonical_snapshot(db_session)
    reader = SqlMarketRsReader(sessionmaker(bind=db_session.get_bind()))
    full_market = reader.get(market="US", symbols=SYMBOLS, as_of_date=AS_OF)
    index_subset = reader.get(market="US", symbols=("OLD", "A1", "B1"), as_of_date=AS_OF)
    watchlist = reader.get(market="US", symbols=("OLD", "B3"), as_of_date=AS_OF)

    for symbol in ("OLD", "A1", "B1", "B3"):
        expected = _stock_payload(full_market, symbol)
        consumer = watchlist if symbol in {"OLD", "B3"} else index_subset
        actual = _stock_payload(consumer, symbol)
        assert {field: actual[field] for field in CANONICAL_STOCK_RS_FIELDS} == {
            field: expected[field] for field in CANONICAL_STOCK_RS_FIELDS
        }

    assert full_market.run_id == run.id
    assert full_market.universe_size == len(SYMBOLS)
    assert full_market.ratings_by_symbol["OLD"]["rs_rating_1m"] < 50
    assert full_market.ratings_by_symbol["OLD"]["rs_rating_3m"] < 50


def test_group_live_and_static_payloads_match_and_main_rank_uses_overall_only(
    db_session,
):
    run = _seed_canonical_snapshot(db_session)
    rankings = CanonicalGroupRankingService().calculate_and_store(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    serialized_rankings = [
        {
            **row,
            "date": (
                row["date"].isoformat()
                if isinstance(row.get("date"), date)
                else row.get("date")
            ),
        }
        for row in rankings
    ]
    static = build_static_groups_payload(
        StaticGroupsSnapshot(
            date=AS_OF.isoformat(),
            rankings=serialized_rankings,
            movers={"period": "1m", "gainers": [], "losers": []},
            group_details={},
            market="US",
            rs_formula_version=BALANCED_RS_FORMULA_VERSION,
            market_rs_run_id=run.id,
            rs_as_of_date=AS_OF.isoformat(),
            rs_universe_size=len(SYMBOLS),
        ),
        generated_at="2026-04-10T23:00:00Z",
        schema_version="static-site-v3",
    )
    static_rows = static["payload"]["rankings"]["rankings"]

    for live, offline in zip(rankings, static_rows, strict=True):
        for field in CANONICAL_GROUP_RS_FIELDS:
            if isinstance(live[field], float):
                assert offline[field] == pytest.approx(live[field])
            else:
                assert offline[field] == live[field]
    assert [row["industry_group"] for row in rankings] == [
        row["industry_group"]
        for row in sorted(
            rankings,
            key=lambda row: (-row["avg_rs_rating"], row["industry_group"]),
        )
    ]


def test_extreme_raw_twelve_month_magnitude_is_bounded_by_horizon_percentiles():
    thousand_percent = calculate_balanced_rs(_returns(old_twelve_month=10.0))["OLD"]
    ten_thousand_percent = calculate_balanced_rs(_returns(old_twelve_month=100.0))["OLD"]

    assert ten_thousand_percent.rs_12m == thousand_percent.rs_12m
    assert ten_thousand_percent.weighted_composite == pytest.approx(
        thousand_percent.weighted_composite
    )
    assert ten_thousand_percent.overall_rs == thousand_percent.overall_rs
    assert HORIZON_WEIGHTS["1m"] + HORIZON_WEIGHTS["3m"] == pytest.approx(0.50)


def test_rrg_transform_is_identical_for_live_and_static_versioned_history():
    series = [
        (date(2025, 1, 1) + timedelta(days=index), 45.0 + index * 0.05)
        for index in range(220)
    ]
    live = compute_group_rrg(series, RRGParams())
    static = compute_group_rrg(tuple(series), RRGParams())

    assert live == static
    assert live is not None
