from datetime import date

import pandas as pd
import pytest
from app.services.breadth.engine import BreadthEngine, BreadthEngineRequest
from app.services.breadth.market_policy import get_breadth_market_policy
from app.services.breadth.types import (
    BreadthUniverseMember,
    BreadthUniverseSnapshot,
)
from app.services.point_in_time_universe_service import (
    hash_point_in_time_universe_symbols,
)
from app.services.static_breadth_eligibility import (
    static_breadth_eligibility_signature,
)


def _prices(index, closes, *, final_volume=110_000):
    close = pd.Series(closes, index=index, dtype=float)
    volume = pd.Series(100_000, index=index, dtype=float)
    volume.iloc[-1] = final_volume
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=index,
    )


def test_engine_tracks_metric_specific_eligibility_for_mixed_history():
    veteran_index = pd.bdate_range(end="2026-08-21", periods=252)
    ipo_index = veteran_index[-2:]
    calculation_date = date(2026, 8, 21)
    members = (
        BreadthUniverseMember("IPO", "USD"),
        BreadthUniverseMember("VETERAN", "USD"),
    )
    snapshot = BreadthUniverseSnapshot(
        calculation_date=calculation_date,
        members=members,
        broad_signature=hash_point_in_time_universe_symbols(("IPO", "VETERAN")),
    )

    result = BreadthEngine().calculate(
        BreadthEngineRequest(
            market="US",
            dates=(calculation_date,),
            universes_by_date={calculation_date: snapshot},
            prices_by_symbol={
                "VETERAN": _prices(veteran_index, [100.0] * 251 + [104.0]),
                "IPO": _prices(ipo_index, [10.0, 11.0]),
            },
            market_policy=get_breadth_market_policy("US"),
        )
    )[calculation_date]

    assert result.broad_universe_count == 2
    assert result.eligibility.advance_decline_eligible_count == 2
    assert result.values.advancing_count == 2
    assert result.values.declining_count == 0
    assert result.values.unchanged_count == 0
    assert result.eligibility.stockbee_daily_eligible_count == 1
    assert result.values.stocks_up_4pct == 1
    assert result.eligibility.stockbee_quarter_eligible_count == 1
    assert result.eligibility.high_low_52week_eligible_count == 1
    assert result.values.new_high_52week_count == 1
    assert result.eligibility.t2108_eligible_count == 1
    assert result.values.t2108_count == 1
    assert result.values.t2108_pct == pytest.approx(100.0)
    assert result.stockbee_eligibility_signature == (
        hash_point_in_time_universe_symbols(("VETERAN",))
    )
    assert result.calculation_revision == 3

    record = result.to_record_mapping()
    assert record["total_stocks_scanned"] == record["broad_universe_count"] == 2


def test_engine_persists_the_canonical_broad_eligibility_signature():
    index = pd.bdate_range(end="2026-08-21", periods=2)
    calculation_date = date(2026, 8, 21)
    snapshot = BreadthUniverseSnapshot(
        calculation_date=calculation_date,
        members=(BreadthUniverseMember("AAA", "USD"),),
        broad_signature=hash_point_in_time_universe_symbols(("AAA",)),
    )

    result = BreadthEngine().calculate(
        BreadthEngineRequest(
            market="US",
            dates=(calculation_date,),
            universes_by_date={calculation_date: snapshot},
            prices_by_symbol={"AAA": _prices(index, [100.0, 101.0])},
            market_policy=get_breadth_market_policy("US"),
        )
    )[calculation_date]

    assert result.eligibility_signature == static_breadth_eligibility_signature(
        ("AAA",)
    )


def test_stockbee_signature_tracks_liquidity_when_daily_history_is_ineligible():
    index = pd.bdate_range(end="2026-08-21", periods=21)
    calculation_date = date(2026, 8, 21)
    prices = _prices(index, [10.0] * len(index))
    prices.loc[index[-2], "Adj Close"] = float("nan")
    snapshot = BreadthUniverseSnapshot(
        calculation_date=calculation_date,
        members=(BreadthUniverseMember("LIQUID", "USD"),),
        broad_signature=hash_point_in_time_universe_symbols(("LIQUID",)),
    )

    result = BreadthEngine().calculate(
        BreadthEngineRequest(
            market="US",
            dates=(calculation_date,),
            universes_by_date={calculation_date: snapshot},
            prices_by_symbol={"LIQUID": prices},
            market_policy=get_breadth_market_policy("US"),
        )
    )[calculation_date]

    assert result.eligibility.stockbee_daily_eligible_count == 0
    assert result.stockbee_eligibility_signature == (
        hash_point_in_time_universe_symbols(("LIQUID",))
    )


def test_engine_isolates_a_malformed_symbol_frame():
    index = pd.bdate_range(end="2026-08-21", periods=252)
    calculation_date = date(2026, 8, 21)
    malformed_index = index.append(
        pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=1)])
    )
    members = (
        BreadthUniverseMember("BAD", "USD"),
        BreadthUniverseMember("GOOD", "USD"),
    )
    snapshot = BreadthUniverseSnapshot(
        calculation_date=calculation_date,
        members=members,
        broad_signature=hash_point_in_time_universe_symbols(("BAD", "GOOD")),
    )

    result = BreadthEngine().calculate(
        BreadthEngineRequest(
            market="US",
            dates=(calculation_date,),
            universes_by_date={calculation_date: snapshot},
            prices_by_symbol={
                "BAD": _prices(malformed_index, [100.0] * len(malformed_index)),
                "GOOD": _prices(index, [100.0] * 251 + [104.0]),
            },
            market_policy=get_breadth_market_policy("US"),
        )
    )[calculation_date]

    assert result.broad_universe_count == 2
    assert result.eligibility.advance_decline_eligible_count == 1
    assert result.values.advancing_count == 1


def test_engine_currency_mismatch_preserves_context_eligibility() -> None:
    index = pd.bdate_range(end="2026-08-21", periods=252)
    calculation_date = date(2026, 8, 21)
    snapshot = BreadthUniverseSnapshot(
        calculation_date=calculation_date,
        members=(BreadthUniverseMember("CROSS", "USD"),),
        broad_signature=hash_point_in_time_universe_symbols(("CROSS",)),
    )

    result = BreadthEngine().calculate(
        BreadthEngineRequest(
            market="CA",
            dates=(calculation_date,),
            universes_by_date={calculation_date: snapshot},
            prices_by_symbol={
                "CROSS": _prices(index, [100.0] * 251 + [104.0]),
            },
            market_policy=get_breadth_market_policy("CA"),
        )
    )[calculation_date]

    assert result.eligibility.stockbee_daily_eligible_count == 0
    assert result.eligibility.stockbee_month_eligible_count == 0
    assert result.eligibility.stockbee_34day_eligible_count == 0
    assert result.eligibility.stockbee_quarter_eligible_count == 0
    assert result.eligibility.advance_decline_eligible_count == 1
    assert result.eligibility.t2108_eligible_count == 1
    assert result.eligibility.high_low_52week_eligible_count == 1
    assert result.eligibility.atr_extension_eligible_count == 1


def test_engine_rejects_a_policy_for_another_market() -> None:
    with pytest.raises(ValueError, match="policy market CA does not match request market US"):
        BreadthEngine().calculate(
            BreadthEngineRequest(
                market="US",
                dates=(),
                universes_by_date={},
                prices_by_symbol={},
                market_policy=get_breadth_market_policy("CA"),
            )
        )


def _random_walk(index, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    close = pd.Series(
        50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(index)))), index=index
    )
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.01, len(index))),
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(200_000, 2_000_000, len(index)).astype(float),
        },
        index=index,
    )


def _streaming_fixture():
    index = pd.bdate_range(end="2026-08-21", periods=320)
    dates = tuple(value.date() for value in index[-3:])
    symbols = tuple(f"S{number}" for number in range(7))
    members = tuple(BreadthUniverseMember(symbol, "USD") for symbol in symbols)
    universes = {
        calculation_date: BreadthUniverseSnapshot(
            calculation_date=calculation_date,
            members=members,
            broad_signature=hash_point_in_time_universe_symbols(symbols),
        )
        for calculation_date in dates
    }
    prices = {
        symbol: _random_walk(index, seed) for seed, symbol in enumerate(symbols)
    }
    return dates, universes, prices


def test_accumulator_batches_match_a_single_whole_universe_call():
    dates, universes, prices = _streaming_fixture()
    engine = BreadthEngine()
    whole = engine.calculate_with_contributors(
        BreadthEngineRequest(
            market="US",
            dates=dates,
            universes_by_date=universes,
            prices_by_symbol=prices,
            market_policy=get_breadth_market_policy("US"),
        )
    )

    accumulator = engine.accumulator(
        market="US",
        dates=dates,
        universes_by_date=universes,
        market_policy=get_breadth_market_policy("US"),
    )
    symbols = list(reversed(prices))
    for offset in range(0, len(symbols), 2):
        accumulator.add_prices(
            {symbol: prices[symbol] for symbol in symbols[offset : offset + 2]}
        )

    assert accumulator.finish() == whole
    assert any(
        snapshot.contributors for snapshot in whole.contributor_snapshots.values()
    )


def test_accumulator_rejects_a_symbol_supplied_twice():
    dates, universes, prices = _streaming_fixture()
    accumulator = BreadthEngine().accumulator(
        market="US",
        dates=dates,
        universes_by_date=universes,
        market_policy=get_breadth_market_policy("US"),
    )
    accumulator.add_prices({"S0": prices["S0"]})

    with pytest.raises(ValueError, match="supplied twice"):
        accumulator.add_prices({"S0": prices["S0"]})


def test_accumulator_finish_on_a_subset_matches_a_request_for_that_subset():
    dates, universes, prices = _streaming_fixture()
    engine = BreadthEngine()
    accumulator = engine.accumulator(
        market="US",
        dates=dates,
        universes_by_date=universes,
        market_policy=get_breadth_market_policy("US"),
    )
    accumulator.add_prices(prices)
    subset = dates[1:]

    assert accumulator.finish(dates=subset) == engine.calculate_with_contributors(
        BreadthEngineRequest(
            market="US",
            dates=subset,
            universes_by_date={value: universes[value] for value in subset},
            prices_by_symbol=prices,
            market_policy=get_breadth_market_policy("US"),
        )
    )
    with pytest.raises(ValueError, match="accumulated dates"):
        accumulator.finish(dates=(date(2020, 1, 2),))


def _single_date_result(prices_by_symbol):
    calculation_date = date(2026, 8, 21)
    snapshot = BreadthUniverseSnapshot(
        calculation_date=calculation_date,
        members=tuple(BreadthUniverseMember(s, "USD") for s in sorted(prices_by_symbol)),
        broad_signature=hash_point_in_time_universe_symbols(tuple(sorted(prices_by_symbol))),
    )
    return BreadthEngine().calculate(
        BreadthEngineRequest(
            market="US",
            dates=(calculation_date,),
            universes_by_date={calculation_date: snapshot},
            prices_by_symbol=prices_by_symbol,
            market_policy=get_breadth_market_policy("US"),
        )
    )[calculation_date]


def test_engine_counts_participation_against_its_own_denominators():
    index = pd.bdate_range(end="2026-08-21", periods=252)
    result = _single_date_result(
        {
            "LEADER": _prices(index, [100.0] * 251 + [104.0]),
            "LAGGARD": _prices(index, [200.0] + [100.0] * 250 + [104.0]),
            "DECLINER": _prices(index, [100.0] * 251 + [90.0]),
            "IPO": _prices(index[-2:], [10.0, 11.0]),
        }
    )

    assert result.eligibility.sma200_eligible_count == 3
    assert result.values.above_sma200_count == 2
    assert result.values.above_sma200_pct == pytest.approx(66.67)
    assert result.eligibility.off_high_eligible_count == 4
    assert result.values.down_20pct_from_high_count == 1
    assert result.values.down_20pct_from_high_pct == pytest.approx(25.0)
    assert result.values.stocks_up_4pct == 2
    assert result.values.up_4pct_near_high_count == 1

    record = result.to_record_mapping()
    assert record["above_sma200_pct"] == pytest.approx(66.67)
    assert record["off_high_eligible_count"] == 4


def test_engine_reports_no_participation_pct_without_eligible_stocks():
    index = pd.bdate_range(end="2026-08-21", periods=2)
    result = _single_date_result({"IPO": _prices(index, [10.0, 11.0])})

    assert result.eligibility.sma200_eligible_count == 0
    assert result.values.above_sma200_pct is None
    assert result.eligibility.off_high_eligible_count == 1
    assert result.values.down_20pct_from_high_pct == 0.0
