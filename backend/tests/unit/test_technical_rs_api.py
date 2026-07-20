from __future__ import annotations

from datetime import date

import pytest
from fastapi import HTTPException

from app.api.v1 import technical
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.domain.scanning.ports import MarketRsResolution
from app.models.stock_universe import StockUniverse
from app.services.market_rs_reader import CanonicalMarketRsUnavailable


class _Reader:
    def __init__(self, ratings_by_symbol):
        self.ratings_by_symbol = ratings_by_symbol
        self.calls: list[dict] = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        return MarketRsResolution.canonical(
            market="US",
            as_of_date=date(2026, 7, 17),
            formula_version=BALANCED_RS_FORMULA_VERSION,
            run_id=42,
            universe_size=5000,
            ratings_by_symbol=self.ratings_by_symbol,
        )


def _add_stock(db, symbol="TEST", market="US"):
    db.add(
        StockUniverse(
            symbol=symbol,
            market=market,
            exchange="NASDAQ",
            currency="USD",
            timezone="America/New_York",
            is_active=True,
            status="active",
        )
    )
    db.commit()


@pytest.mark.asyncio
async def test_balanced_technical_rs_returns_canonical_snapshot_without_live_fetch(
    universe_session,
    monkeypatch,
):
    _add_stock(universe_session)
    reader = _Reader(
        {
            "TEST": {
                "rs_rating": 87,
                "rs_rating_1m": 42,
                "rs_rating_3m": 91,
                "rs_rating_12m": 98,
            }
        }
    )
    monkeypatch.setattr(
        technical,
        "get_yfinance_service",
        lambda: (_ for _ in ()).throw(AssertionError("live prices must not be fetched")),
    )

    response = await technical.get_rs_rating(
        "test",
        db=universe_session,
        market_rs_reader=reader,
    )

    assert response == {
        "symbol": "TEST",
        "market": "US",
        "rs_rating": 87,
        "rs_rating_1m": 42,
        "rs_rating_3m": 91,
        "rs_rating_12m": 98,
        "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 42,
        "rs_as_of_date": "2026-07-17",
        "rs_universe_size": 5000,
    }
    assert reader.calls == [
        {
            "market": "US",
            "symbols": ("TEST",),
            "as_of_date": None,
            "formula_version": None,
        }
    ]


@pytest.mark.asyncio
async def test_balanced_technical_rs_ineligible_stock_returns_not_enough_history(
    universe_session,
    monkeypatch,
):
    _add_stock(universe_session)
    reader = _Reader({})
    monkeypatch.setattr(
        technical,
        "get_yfinance_service",
        lambda: (_ for _ in ()).throw(AssertionError("live prices must not be fetched")),
    )

    response = await technical.get_rs_rating(
        "TEST",
        db=universe_session,
        market_rs_reader=reader,
    )

    assert response["error"] == "Not enough history to calculate RS rating"
    assert response["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert response["market_rs_run_id"] == 42
    assert response["rs_as_of_date"] == "2026-07-17"


@pytest.mark.asyncio
async def test_technical_rs_returns_service_unavailable_when_publication_is_missing(
    universe_session,
):
    _add_stock(universe_session)

    class _UnavailableReader:
        @staticmethod
        def get(**_kwargs):
            raise CanonicalMarketRsUnavailable(
                "Canonical Market RS is unavailable for US at latest"
            )

    with pytest.raises(HTTPException) as exc_info:
        await technical.get_rs_rating(
            "TEST",
            db=universe_session,
            market_rs_reader=_UnavailableReader(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == (
        "Canonical Market RS is unavailable for US at latest"
    )
