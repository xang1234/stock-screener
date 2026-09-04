from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.infra.query.options_candidate_source import SqlOptionsCandidateSource
from app.models.stock import StockFundamental, StockPrice


def test_candidate_source_uses_pinned_run_and_domain_caps_with_complete_inputs() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            StockFundamental.__table__,
            StockPrice.__table__,
        ],
    )
    session = sessionmaker(bind=engine)()
    as_of = date(2026, 9, 4)
    session.add(
        FeatureRun(
            id=7,
            as_of_date=as_of,
            run_type="daily_snapshot",
            status="published",
        )
    )
    rows = []
    fundamentals = []
    for index in range(45):
        symbol = f"C{index:02}"
        rows.append(
            StockFeatureDaily(
                run_id=7,
                symbol=symbol,
                as_of_date=as_of,
                composite_score=200 - index,
                details_json={
                    "current_price": 100 + index,
                    "avg_dollar_volume": 200_000_000,
                    "dividend_yield": 1.0,
                    "rs_rating": 70,
                    "ibd_group_rank": 80,
                },
            )
        )
        fundamentals.append(StockFundamental(symbol=symbol, adv_usd=200_000_000, dividend_yield=0.01))
    for index in range(45):
        symbol = f"L{index:02}"
        rows.append(
            StockFeatureDaily(
                run_id=7,
                symbol=symbol,
                as_of_date=as_of,
                composite_score=100 - index,
                details_json={
                    "current_price": 50 + index,
                    "avg_dollar_volume": 200_000_000,
                    "dividend_yield": 2.0,
                    "rs_rating": 90,
                    "ibd_group_rank": 10,
                },
            )
        )
        fundamentals.append(StockFundamental(symbol=symbol, adv_usd=200_000_000, dividend_yield=0.02))
    rows.append(
        StockFeatureDaily(
            run_id=7,
            symbol="EXACT",
            as_of_date=as_of,
            composite_score=999,
            details_json={
                "current_price": 10,
                "avg_dollar_volume": 100_000_000,
                "rs_rating": 99,
                "ibd_group_rank": 1,
            },
        )
    )
    fundamentals.append(StockFundamental(symbol="EXACT", adv_usd=100_000_000))
    for offset in range(21):
        session.add(
            StockPrice(
                symbol="C00",
                date=as_of - timedelta(days=20 - offset),
                close=100 + offset,
                volume=1_000_000,
            )
        )
    session.add_all(rows + fundamentals)
    session.commit()

    result = SqlOptionsCandidateSource(session).read(7)

    assert result.source_feature_run_id == 7
    assert result.as_of_date == as_of
    assert len(result.current_candidates) == 80
    assert "EXACT" not in {row.symbol for row in result.current_candidates}
    assert sum(row.candidate_rank is not None for row in result.current_candidates) == 40
    assert sum(row.leader_rank is not None for row in result.current_candidates) == 40
    first = result.current_candidates[0]
    assert first.symbol == "C00"
    assert first.spot_price == 100
    assert first.dividend_yield == 0.01
    assert first.price_closes == tuple(float(value) for value in range(100, 121))

    session.close()
    engine.dispose()


def test_candidate_source_preserves_both_ranks_for_overlap() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            StockFundamental.__table__,
            StockPrice.__table__,
        ],
    )
    session = sessionmaker(bind=engine)()
    session.add(FeatureRun(id=8, as_of_date=date(2026, 9, 4), run_type="daily_snapshot", status="published"))
    session.add(
        StockFeatureDaily(
            run_id=8,
            symbol="aapl",
            as_of_date=date(2026, 9, 4),
            composite_score=99,
            details_json={
                "current_price": 200,
                "avg_dollar_volume": 500_000_000,
                "rs_rating": 95,
                "ibd_group_rank": 2,
            },
        )
    )
    session.add(StockFundamental(symbol="aapl", adv_usd=500_000_000))
    session.commit()

    result = SqlOptionsCandidateSource(session).read(8)

    assert len(result.current_candidates) == 1
    assert result.current_candidates[0].symbol == "AAPL"
    assert result.current_candidates[0].candidate_rank == 1
    assert result.current_candidates[0].leader_rank == 1
    session.close()
    engine.dispose()


def test_current_liquidity_uses_feature_run_snapshot_not_mutable_fundamentals() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            StockFundamental.__table__,
            StockPrice.__table__,
        ],
    )
    session = sessionmaker(bind=engine)()
    as_of = date(2026, 9, 4)
    session.add(
        FeatureRun(
            id=9,
            as_of_date=as_of,
            run_type="daily_snapshot",
            status="published",
        )
    )
    session.add(
        StockFeatureDaily(
            run_id=9,
            symbol="AAPL",
            as_of_date=as_of,
            composite_score=99,
            details_json={
                "current_price": 200,
                "avg_dollar_volume": 150_000_000,
                "rs_rating": 95,
                "ibd_group_rank": 2,
            },
        )
    )
    session.add(StockFundamental(symbol="AAPL", adv_usd=50_000_000))
    session.commit()

    result = SqlOptionsCandidateSource(session).read(9)

    assert [row.symbol for row in result.current_candidates] == ["AAPL"]
    assert result.current_candidates[0].daily_dollar_volume == 150_000_000
    session.close()
    engine.dispose()


def test_continuity_inputs_ignore_mutable_fundamentals_and_use_latest_pinned_close() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[StockFundamental.__table__, StockPrice.__table__],
    )
    session = sessionmaker(bind=engine)()
    as_of = date(2026, 9, 4)
    session.add(
        StockFundamental(
            symbol="aapl",
            adv_usd=250_000_000,
            dividend_yield=0.012,
        )
    )
    session.add_all(
        [
            StockPrice(symbol="aapl", date=as_of - timedelta(days=1), close=199, volume=1),
            StockPrice(symbol="aapl", date=as_of, close=201, volume=1),
            StockPrice(symbol="aapl", date=as_of + timedelta(days=1), close=999, volume=1),
        ]
    )
    session.commit()

    inputs = SqlOptionsCandidateSource(session).read_continuity_inputs(
        ["AAPL", "MISSING"], as_of
    )

    assert set(inputs) == {"AAPL"}
    assert inputs["AAPL"].spot_price == 201
    assert inputs["AAPL"].price_closes == (199.0, 201.0)
    assert inputs["AAPL"].daily_dollar_volume is None
    assert inputs["AAPL"].dividend_yield is None
    session.close()
    engine.dispose()
