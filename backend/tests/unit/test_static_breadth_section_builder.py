from datetime import date

import pandas as pd
import pytest
from app.services.breadth.types import BreadthUniverseMember, BreadthUniverseSnapshot
from app.services.point_in_time_universe_service import (
    PointInTimeUniverse,
    PointInTimeUniverseMember,
    hash_point_in_time_universe_symbols,
)
from app.services.static_breadth_section_builder import (
    StaticBreadthEngineInputFactory,
    StaticBreadthSectionBuilder,
)
from app.services.static_site_errors import StaticSiteSectionUnavailableError


def _price_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Adj Close": 100.0,
            "Volume": 1_000_000,
        },
        index=index,
    )


def test_static_inputs_retain_only_the_local_price_feature_window():
    recent_index = pd.bdate_range(end="2026-08-21", periods=400)
    full_index = pd.DatetimeIndex([pd.Timestamp("2020-01-02"), *recent_index])
    canonical_dates = list(recent_index[-120:].date)

    inputs = StaticBreadthEngineInputFactory().build(
        market="HK",
        canonical_dates=canonical_dates,
        price_data={"0700.HK": _price_frame(full_index)},
        currencies_by_symbol={"0700.HK": "HKD"},
    )

    retained_index = inputs.request.prices_by_symbol["0700.HK"].index
    assert len(retained_index) == 372
    assert retained_index[0] == recent_index[28]
    assert pd.Timestamp("2020-01-02") not in retained_index
    assert inputs.request.market_policy.market == "HK"
    assert inputs.request.market_policy.currency == "HKD"
    assert not hasattr(inputs.request, "fx_by_currency")


def test_static_inputs_keep_each_dates_point_in_time_universe():
    first_date = date(2026, 8, 20)
    second_date = date(2026, 8, 21)
    prices = {
        "OLD": _price_frame(pd.bdate_range(end=second_date, periods=30)),
        "NEW": _price_frame(pd.bdate_range(end=second_date, periods=30)),
    }
    universes = {
        first_date: BreadthUniverseSnapshot(
            calculation_date=first_date,
            members=(BreadthUniverseMember("OLD", "USD"),),
            broad_signature="first",
        ),
        second_date: BreadthUniverseSnapshot(
            calculation_date=second_date,
            members=(
                BreadthUniverseMember("OLD", "USD"),
                BreadthUniverseMember("NEW", "USD"),
            ),
            broad_signature="second",
        ),
    }

    inputs = StaticBreadthEngineInputFactory().build(
        market="US",
        canonical_dates=[first_date, second_date],
        price_data=prices,
        universes_by_date=universes,
    )

    assert tuple(
        member.symbol
        for member in inputs.request.universes_by_date[first_date].members
    ) == ("OLD",)
    assert tuple(
        member.symbol
        for member in inputs.request.universes_by_date[second_date].members
    ) == ("OLD", "NEW")


def test_static_builder_resolves_and_passes_a_universe_for_each_history_date():
    first_date = date(2026, 8, 20)
    second_date = date(2026, 8, 21)
    price_frame = _price_frame(
        pd.DatetimeIndex([pd.Timestamp(first_date), pd.Timestamp(second_date)])
    )

    class _Resolver:
        def resolve(self, _db, *, market, as_of_date):
            symbols = (
                ("OLD",)
                if as_of_date == first_date
                else ("OLD", "NEW")
            )
            return PointInTimeUniverse(
                market=market,
                as_of_date=as_of_date,
                symbols=symbols,
                universe_hash=hash_point_in_time_universe_symbols(symbols),
                members=tuple(
                    PointInTimeUniverseMember(symbol=symbol, currency="USD")
                    for symbol in symbols
                ),
            )

    class _PriceCache:
        def get_cached_only(self, symbol, *, period):
            assert (symbol, period) == ("SPY", "1y")
            return price_frame

        def get_many_cached_only_fresh(
            self,
            symbols,
            *,
            period,
            required_as_of_date,
            minimum_rows,
        ):
            assert period == "2y"
            assert required_as_of_date == second_date
            assert minimum_rows == 1
            return {symbol: price_frame for symbol in symbols}

    class _BenchmarkCache:
        def get_benchmark_candidates(self, market):
            assert market == "US"
            return ("SPY",)

        def get_benchmark_symbol(self, market):
            return "SPY"

    captured = {}

    class _InputFactory:
        def build(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("captured inputs")

    builder = StaticBreadthSectionBuilder(
        ui_snapshot_service=object(),
        price_cache=_PriceCache(),
        benchmark_cache=_BenchmarkCache(),
        engine_input_factory=_InputFactory(),
        universe_resolver=_Resolver(),
    )

    with pytest.raises(RuntimeError, match="captured inputs"):
        builder.build(
            generated_at="2026-08-21T22:00:00Z",
            expected_as_of_date=second_date,
            market="US",
            serialized_rows=[{"symbol": "OLD"}, {"symbol": "NEW"}],
            db=object(),
        )

    universes = captured["universes_by_date"]
    assert tuple(member.symbol for member in universes[first_date].members) == (
        "OLD",
    )
    assert tuple(member.symbol for member in universes[second_date].members) == (
        "NEW",
        "OLD",
    )


def test_static_builder_keeps_short_history_when_a_formula_can_use_it():
    calculation_date = date(2026, 8, 28)
    index = pd.bdate_range(end=calculation_date, periods=34)
    stock_prices = _price_frame(index)
    stock_prices.loc[index[-1], ["Open", "High", "Close", "Adj Close"]] = (
        105.0,
        106.0,
        105.0,
        105.0,
    )
    stock_prices.loc[index[-1], "Volume"] = 2_000_000
    benchmark_prices = _price_frame(index)

    class _Resolver:
        def resolve(self, _db, *, market, as_of_date):
            return PointInTimeUniverse(
                market=market,
                as_of_date=as_of_date,
                symbols=("NEW.TO",),
                universe_hash=hash_point_in_time_universe_symbols(("NEW.TO",)),
                members=(
                    PointInTimeUniverseMember(symbol="NEW.TO", currency="CAD"),
                ),
            )

    class _PriceCache:
        def get_cached_only(self, symbol, *, period):
            assert period == "1y"
            return benchmark_prices

        def get_many_cached_only(self, symbols, *, period):
            assert period == "2y"
            return {symbol: None for symbol in symbols}

        def get_many_cached_only_fresh(
            self,
            symbols,
            *,
            period,
            required_as_of_date,
            minimum_rows,
        ):
            assert period == "2y"
            assert required_as_of_date == calculation_date
            return {
                symbol: stock_prices if minimum_rows <= len(stock_prices) else None
                for symbol in symbols
            }

    class _BenchmarkCache:
        def get_benchmark_candidates(self, market):
            assert market == "CA"
            return ("^GSPTSE",)

        def get_benchmark_symbol(self, market):
            return "^GSPTSE"

    payload = StaticBreadthSectionBuilder(
        ui_snapshot_service=object(),
        price_cache=_PriceCache(),
        benchmark_cache=_BenchmarkCache(),
        universe_resolver=_Resolver(),
    ).build(
        generated_at="2026-08-29T22:00:00Z",
        expected_as_of_date=calculation_date,
        market="CA",
        serialized_rows=[{"symbol": "NEW.TO"}],
        db=object(),
    )

    assert payload["payload"]["current"]["stocks_up_4pct"] == 1
    assert calculation_date.isoformat() in payload[
        "_contributor_calculation_signatures"
    ]


def test_static_builder_keeps_departed_member_through_last_membership_date():
    benchmark_index = pd.bdate_range(end="2026-08-28", periods=35)
    departure_date = benchmark_index[-2].date()
    calculation_date = benchmark_index[-1].date()
    departed_prices = _price_frame(benchmark_index[:-1])
    departed_prices.loc[
        benchmark_index[-2], ["Open", "High", "Close", "Adj Close"]
    ] = (105.0, 106.0, 105.0, 105.0)
    departed_prices.loc[benchmark_index[-2], "Volume"] = 2_000_000
    current_prices = _price_frame(benchmark_index)
    benchmark_prices = _price_frame(benchmark_index)

    class _Resolver:
        def resolve(self, _db, *, market, as_of_date):
            symbols = (
                ("CURRENT.TO", "DEPARTED.TO")
                if as_of_date <= departure_date
                else ("CURRENT.TO",)
            )
            return PointInTimeUniverse(
                market=market,
                as_of_date=as_of_date,
                symbols=symbols,
                universe_hash=hash_point_in_time_universe_symbols(symbols),
                members=tuple(
                    PointInTimeUniverseMember(symbol=symbol, currency="CAD")
                    for symbol in symbols
                ),
            )

    class _PriceCache:
        def get_cached_only(self, symbol, *, period):
            assert period == "1y"
            return benchmark_prices

        def get_many_cached_only_fresh(
            self,
            symbols,
            *,
            period,
            required_as_of_date,
            minimum_rows,
        ):
            assert period == "2y"
            assert minimum_rows == 1
            return {
                symbol: (
                    departed_prices
                    if symbol == "DEPARTED.TO"
                    and required_as_of_date <= departure_date
                    else current_prices
                    if symbol == "CURRENT.TO"
                    else None
                )
                for symbol in symbols
            }

    class _BenchmarkCache:
        def get_benchmark_candidates(self, market):
            assert market == "CA"
            return ("^GSPTSE",)

        def get_benchmark_symbol(self, market):
            return "^GSPTSE"

    payload = StaticBreadthSectionBuilder(
        ui_snapshot_service=object(),
        price_cache=_PriceCache(),
        benchmark_cache=_BenchmarkCache(),
        universe_resolver=_Resolver(),
    ).build(
        generated_at="2026-08-29T22:00:00Z",
        expected_as_of_date=calculation_date,
        market="CA",
        serialized_rows=[{"symbol": "CURRENT.TO"}],
        db=object(),
    )

    departure_row = next(
        row
        for row in payload["payload"]["history_90d"]
        if row["date"] == departure_date.isoformat()
    )
    assert departure_row["stocks_up_4pct"] == 1


class _EmptyUniverseResolver:
    def resolve(self, _db, *, market, as_of_date):
        return PointInTimeUniverse(
            market=market,
            as_of_date=as_of_date,
            symbols=(),
            universe_hash="empty",
        )


def test_static_builder_does_not_fall_back_when_database_universe_is_empty():
    builder = StaticBreadthSectionBuilder(
        ui_snapshot_service=object(),
        price_cache=object(),
        benchmark_cache=object(),
        universe_resolver=_EmptyUniverseResolver(),
    )

    with pytest.raises(
        StaticSiteSectionUnavailableError,
        match="No common-stock universe is available",
    ):
        builder.build(
            generated_at="2026-08-21T22:00:00Z",
            expected_as_of_date=date(2026, 8, 21),
            market="US",
            serialized_rows=[{"symbol": "ETF"}],
            db=object(),
        )
