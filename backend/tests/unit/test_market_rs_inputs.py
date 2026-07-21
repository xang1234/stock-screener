from datetime import date

import pytest

from app.config import settings
from app.models.stock import StockPrice
from app.services.market_rs_inputs import (
    MarketRsInputLoader,
    MarketRsInputUnavailable,
)
from app.services.point_in_time_universe_service import (
    PointInTimeUniverse,
    PointInTimeUniverseUnavailable,
)


ANCHORS = {
    0: date(2026, 4, 10),
    21: date(2026, 3, 10),
    63: date(2026, 1, 9),
    126: date(2025, 10, 10),
    189: date(2025, 7, 11),
    252: date(2025, 4, 10),
}


class _UniverseStub:
    def __init__(self, symbols):
        self.symbols = tuple(symbols)

    def resolve(self, _db, *, market, as_of_date):
        return PointInTimeUniverse(
            market=market,
            as_of_date=as_of_date,
            symbols=self.symbols,
            universe_hash="u" * 64,
        )


class _CalendarStub:
    @staticmethod
    def session_anchors(_market, _as_of_date, *, offsets):
        assert set(offsets) == {21, 63, 126, 189, 252}
        return dict(ANCHORS)


class _BenchmarkRegistryStub:
    def __init__(self, candidates=("SPY",)):
        self.candidates = list(candidates)

    @staticmethod
    def normalize_market(market):
        return market.upper()

    def get_candidate_symbols(self, _market):
        return list(self.candidates)

    def get_primary_symbol(self, _market):
        return self.candidates[0]


def _loader(symbols, *, candidates=("SPY",)):
    return MarketRsInputLoader(
        point_in_time_universe=_UniverseStub(symbols),
        market_calendar=_CalendarStub(),
        benchmark_registry=_BenchmarkRegistryStub(candidates),
    )


def _price(symbol, offset, *, adjusted, close=None):
    return StockPrice(
        symbol=symbol,
        date=ANCHORS[offset],
        adj_close=adjusted,
        close=adjusted if close is None else close,
    )


def _complete_rows(symbol, values, *, close_multiplier=1.0):
    return [
        _price(
            symbol,
            offset,
            adjusted=values[offset],
            close=values[offset] * close_multiplier,
        )
        for offset in ANCHORS
    ]


def test_load_uses_adjusted_prices_and_excludes_only_insufficient_history(db_session):
    db_session.add_all(
        [
            *_complete_rows(
                "AAA",
                {0: 120.0, 21: 100.0, 63: 90.0, 126: 80.0, 189: 70.0, 252: 60.0},
                close_multiplier=10.0,
            ),
            *_complete_rows(
                "BBB",
                {0: 90.0, 21: 80.0, 63: 75.0, 126: 70.0, 189: 65.0, 252: 60.0},
            ),
            *[
                _price("YOUNG", offset, adjusted=50.0)
                for offset in ANCHORS
                if offset != 252
            ],
            *_complete_rows(
                "SPY",
                {0: 110.0, 21: 100.0, 63: 100.0, 126: 100.0, 189: 100.0, 252: 100.0},
            ),
        ]
    )
    db_session.commit()

    inputs = _loader(("AAA", "BBB", "YOUNG")).load(
        db_session, market="US", as_of_date=ANCHORS[0]
    )

    assert inputs.benchmark_symbol == "SPY"
    assert inputs.expected_symbols == ("AAA", "BBB", "YOUNG")
    assert set(inputs.excess_returns_by_symbol) == {"AAA", "BBB"}
    assert inputs.exclusions == {
        "YOUNG": "missing_adjusted_252_session_anchor"
    }
    assert inputs.current_price_coverage == pytest.approx(1.0)
    assert inputs.excess_returns_by_symbol["AAA"]["1m"] == pytest.approx(
        (120.0 / 100.0 - 1.0) - (110.0 / 100.0 - 1.0)
    )


def test_load_chooses_first_complete_benchmark_candidate(db_session):
    db_session.add_all(
        [
            *_complete_rows("AAA", {offset: 100.0 + offset for offset in ANCHORS}),
            *[
                _price("SPY", offset, adjusted=100.0)
                for offset in ANCHORS
                if offset != 252
            ],
            *_complete_rows("VTI", {offset: 100.0 for offset in ANCHORS}),
        ]
    )
    db_session.commit()

    inputs = _loader(("AAA",), candidates=("SPY", "VTI")).load(
        db_session, market="US", as_of_date=ANCHORS[0]
    )

    assert inputs.benchmark_symbol == "VTI"


def test_load_fails_when_no_benchmark_has_every_exact_anchor(db_session):
    db_session.add_all(
        [
            *_complete_rows("AAA", {offset: 100.0 for offset in ANCHORS}),
            *[
                _price("SPY", offset, adjusted=100.0)
                for offset in ANCHORS
                if offset != 252
            ],
        ]
    )
    db_session.commit()

    with pytest.raises(MarketRsInputUnavailable) as exc_info:
        _loader(("AAA",)).load(
            db_session, market="US", as_of_date=ANCHORS[0]
        )

    assert exc_info.value.reason_code == "benchmark_adjusted_anchor_missing"
    assert exc_info.value.benchmark_symbol == "SPY"
    assert exc_info.value.expected_symbol_count == 1


def test_load_fails_when_current_price_coverage_is_below_ninety_percent(db_session):
    symbols = tuple(f"S{index}" for index in range(10))
    db_session.add_all(
        [
            *[
                _price(symbol, 0, adjusted=100.0)
                for symbol in symbols[:8]
            ],
            *_complete_rows("SPY", {offset: 100.0 for offset in ANCHORS}),
        ]
    )
    db_session.commit()

    with pytest.raises(MarketRsInputUnavailable) as exc_info:
        _loader(symbols).load(
            db_session, market="US", as_of_date=ANCHORS[0]
        )

    assert (
        exc_info.value.reason_code
        == "current_adjusted_price_coverage_below_threshold"
    )
    assert exc_info.value.diagnostics["current_price_coverage"] == pytest.approx(0.8)


def test_load_allows_ca_current_price_coverage_matching_configured_policy(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings, "market_rs_min_current_price_coverage_ca", 0.70)
    symbols = tuple(f"S{index}" for index in range(20))
    db_session.add_all(
        [
            *[
                _price(symbol, 0, adjusted=100.0)
                for symbol in symbols[:15]
            ],
            *_complete_rows("^GSPTSE", {offset: 100.0 for offset in ANCHORS}),
        ]
    )
    db_session.commit()

    inputs = _loader(symbols, candidates=("^GSPTSE",)).load(
        db_session, market="CA", as_of_date=ANCHORS[0]
    )

    assert inputs.current_price_coverage == pytest.approx(0.75)
    assert set(inputs.exclusions) == set(symbols)


def test_load_uses_configured_market_specific_current_price_threshold(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings, "market_rs_min_current_price_coverage_ca", 0.80)
    symbols = tuple(f"S{index}" for index in range(20))
    db_session.add_all(
        [
            *[
                _price(symbol, 0, adjusted=100.0)
                for symbol in symbols[:15]
            ],
            *_complete_rows("^GSPTSE", {offset: 100.0 for offset in ANCHORS}),
        ]
    )
    db_session.commit()

    with pytest.raises(MarketRsInputUnavailable) as exc_info:
        _loader(symbols, candidates=("^GSPTSE",)).load(
            db_session, market="CA", as_of_date=ANCHORS[0]
        )

    assert (
        exc_info.value.reason_code
        == "current_adjusted_price_coverage_below_threshold"
    )
    assert exc_info.value.diagnostics["current_price_coverage"] == pytest.approx(0.75)
    assert exc_info.value.diagnostics["minimum_current_price_coverage"] == pytest.approx(0.80)


def test_load_translates_unavailable_historical_universe_to_input_failure(db_session):
    class _UnavailableUniverse:
        @staticmethod
        def resolve(_db, *, market, as_of_date):
            raise PointInTimeUniverseUnavailable(
                f"{market} historical universe for {as_of_date} is incomplete"
            )

    loader = MarketRsInputLoader(
        point_in_time_universe=_UnavailableUniverse(),
        market_calendar=_CalendarStub(),
        benchmark_registry=_BenchmarkRegistryStub(),
    )

    with pytest.raises(MarketRsInputUnavailable) as exc_info:
        loader.load(db_session, market="US", as_of_date=ANCHORS[0])

    assert exc_info.value.reason_code == "point_in_time_universe_unavailable"
    assert exc_info.value.diagnostics == {
        "error": "US historical universe for 2026-04-10 is incomplete"
    }
