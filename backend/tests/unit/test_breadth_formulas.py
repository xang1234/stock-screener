import pandas as pd
import pytest

from app.services.breadth.formulas import (
    prepare_feature_frame,
    signal_flags_at as calculate_signal_flags,
)
from app.services.breadth.market_policy import get_breadth_market_policy
from app.services.breadth.types import BreadthFormulaPolicy

CALCULATION_DATE = pd.Timestamp("2026-08-21")
US_POLICY = get_breadth_market_policy("US")


def signal_flags_at(
    feature_frame,
    calculation_date,
    formula_policy,
    market_policy=US_POLICY,
    *,
    stockbee_currency_matches=True,
):
    return calculate_signal_flags(
        feature_frame,
        calculation_date,
        formula_policy,
        market_policy,
        stockbee_currency_matches=stockbee_currency_matches,
    )


def _feature_row(**overrides):
    values = {
        "adjusted_close": 104.0,
        "prior_adjusted_close": 100.0,
        "daily_return": 0.04,
        "volume": 100_000.0,
        "prior_volume": 99_999.0,
        "adtv20_local": 250_000.0,
        "adjusted_close_20": 100.0,
        "raw_close_local_20": 5.0,
        "month_return": 0.04,
        "low_34": 92.0,
        "high_34": 104.0,
        "low_65": 83.0,
        "high_65": 104.0,
        "sma40": 100.0,
        "sma50": 100.0,
        "atr14": 1.0,
        "adjusted_high": 105.0,
        "adjusted_low": 103.0,
        "previous_251_high": 104.0,
        "previous_251_low": 90.0,
        "sma200": 100.0,
        "high_252": 105.0,
    }
    values.update(overrides)
    return pd.DataFrame([values], index=[CALCULATION_DATE])


def test_prepare_feature_frame_separates_adjusted_signals_from_raw_liquidity():
    index = pd.to_datetime(["2026-08-20", "2026-08-21"])
    prices = pd.DataFrame(
        {
            "Open": [99.0, 49.5],
            "High": [102.0, 51.0],
            "Low": [98.0, 49.0],
            "Close": [100.0, 50.0],
            "Adj Close": [50.0, 50.0],
            "Volume": [200_000, 220_000],
        },
        index=index,
    )
    result = prepare_feature_frame(prices)

    assert result.iloc[0].adjusted_high == pytest.approx(51.0)
    assert result.iloc[0].raw_close_local == pytest.approx(100.0)
    assert result.iloc[0].traded_value_local == pytest.approx(20_000_000.0)
    assert "fx_to_usd" not in result
    assert "adtv20_usd" not in result


def test_prepare_feature_frame_normalizes_timezone_aware_daily_sessions():
    index = pd.bdate_range("2026-07-01", periods=40, tz="America/New_York")
    prices = pd.DataFrame(
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
    features = prepare_feature_frame(prices)
    signals = signal_flags_at(
        features,
        index[-1].date(),
        BreadthFormulaPolicy(),
    )

    assert features.index.tz is None
    assert signals.eligibility.advance_decline is True


def test_prepare_feature_frame_rejects_a_missing_session():
    index = pd.DatetimeIndex([pd.Timestamp("2026-03-19"), pd.NaT])
    prices = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.0, 101.0],
            "Adj Close": [100.0, 101.0],
            "Volume": [1_000_000, 1_000_000],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="valid sessions"):
        prepare_feature_frame(prices)


def test_prepare_feature_frame_initializes_and_smooths_wilder_atr():
    index = pd.bdate_range("2026-07-01", periods=15)
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Adj Close": 100.0,
            "Volume": 100_000,
        },
        index=index,
    )

    result = prepare_feature_frame(prices)

    assert result.atr14.iloc[:13].isna().all()
    assert result.atr14.iloc[13] == pytest.approx(2.0)
    assert result.atr14.iloc[14] == pytest.approx(2.0)


def test_prepare_feature_frame_restarts_wilder_atr_after_initial_gap():
    index = pd.bdate_range("2026-07-01", periods=18)
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Adj Close": 100.0,
            "Volume": 100_000,
        },
        index=index,
    )
    prices.loc[index[0], "High"] = float("nan")

    result = prepare_feature_frame(
        prices,
        atr_period=14,
    )

    assert result.atr14.iloc[:14].isna().all()
    assert result.atr14.iloc[14] == pytest.approx(2.0)
    assert result.atr14.iloc[15] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("overrides", "expected_up", "expected_down"),
    [
        ({"daily_return": 0.04}, True, False),
        ({"daily_return": 0.039999}, False, False),
        ({"daily_return": -0.04}, False, True),
        ({"daily_return": -0.039999}, False, False),
        ({"daily_return": 0.04, "volume": 99_999.0}, False, False),
        (
            {"daily_return": 0.04, "volume": 100_000.0, "prior_volume": 100_000.0},
            False,
            False,
        ),
        ({"daily_return": 0.04, "adtv20_local": 249_999.99}, False, False),
    ],
)
def test_daily_stockbee_boundaries(overrides, expected_up, expected_down):
    signals = signal_flags_at(
        _feature_row(**overrides), CALCULATION_DATE.date(), BreadthFormulaPolicy()
    )

    assert signals.up_4pct is expected_up
    assert signals.down_4pct is expected_down


@pytest.mark.parametrize(
    ("month_return", "up_25", "down_25", "up_50", "down_50"),
    [
        (0.25, True, False, False, False),
        (0.249999, False, False, False, False),
        (-0.25, False, True, False, False),
        (0.50, True, False, True, False),
        (-0.50, False, True, False, True),
    ],
)
def test_month_stockbee_boundaries(month_return, up_25, down_25, up_50, down_50):
    signals = signal_flags_at(
        _feature_row(month_return=month_return),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )

    assert signals.up_25pct_month is up_25
    assert signals.down_25pct_month is down_25
    assert signals.up_50pct_month is up_50
    assert signals.down_50pct_month is down_50


def test_month_reference_price_uses_market_local_threshold_inclusively():
    ca_policy = get_breadth_market_policy("CA")
    eligible = signal_flags_at(
        _feature_row(
            adtv20_local=5_000.0,
            raw_close_local_20=0.30,
            month_return=0.25,
        ),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
        ca_policy,
    )
    ineligible = signal_flags_at(
        _feature_row(
            adtv20_local=5_000.0,
            raw_close_local_20=0.299,
            month_return=0.25,
        ),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
        ca_policy,
    )

    assert eligible.eligibility.stockbee_month is True
    assert eligible.up_25pct_month is True
    assert ineligible.eligibility.stockbee_month is False
    assert ineligible.up_25pct_month is False


def test_daily_volume_uses_market_local_share_threshold():
    de_policy = get_breadth_market_policy("DE")
    eligible = signal_flags_at(
        _feature_row(
            adtv20_local=5_000.0,
            volume=300.0,
            prior_volume=299.0,
        ),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
        de_policy,
    )
    ineligible = signal_flags_at(
        _feature_row(
            adtv20_local=5_000.0,
            volume=299.0,
            prior_volume=298.0,
        ),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
        de_policy,
    )

    assert eligible.up_4pct is True
    assert ineligible.up_4pct is False


def test_currency_mismatch_disables_only_stockbee_eligibility():
    signals = signal_flags_at(
        _feature_row(),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
        US_POLICY,
        stockbee_currency_matches=False,
    )

    assert signals.eligibility.stockbee_liquidity is False
    assert signals.eligibility.stockbee_daily is False
    assert signals.eligibility.stockbee_month is False
    assert signals.eligibility.stockbee_34day is False
    assert signals.eligibility.stockbee_quarter is False
    assert signals.eligibility.advance_decline is True
    assert signals.eligibility.t2108 is True
    assert signals.eligibility.high_low_52week is True
    assert signals.eligibility.atr_extension is True


@pytest.mark.parametrize(
    ("overrides", "signal_name", "expected"),
    [
        ({"adjusted_close": 113.0, "low_34": 100.0}, "up_13pct_34days", True),
        ({"adjusted_close": 112.999, "low_34": 100.0}, "up_13pct_34days", False),
        ({"adjusted_close": 87.0, "high_34": 100.0}, "down_13pct_34days", True),
        ({"adjusted_close": 125.0, "low_65": 100.0}, "up_25pct_quarter", True),
        ({"adjusted_close": 75.0, "high_65": 100.0}, "down_25pct_quarter", True),
    ],
)
def test_range_signal_boundaries(overrides, signal_name, expected):
    signals = signal_flags_at(
        _feature_row(**overrides), CALCULATION_DATE.date(), BreadthFormulaPolicy()
    )

    assert getattr(signals, signal_name) is expected


def test_t2108_requires_strictly_above_sma40():
    equal = signal_flags_at(
        _feature_row(adjusted_close=100.0, sma40=100.0),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )
    above = signal_flags_at(
        _feature_row(adjusted_close=100.001, sma40=100.0),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )

    assert equal.eligibility.t2108 is True
    assert equal.t2108_above is False
    assert above.t2108_above is True


def test_52week_record_comparison_is_strict():
    equal = signal_flags_at(
        _feature_row(adjusted_high=104.0, previous_251_high=104.0),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )
    record = signal_flags_at(
        _feature_row(adjusted_high=104.001, previous_251_high=104.0),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )

    assert equal.new_high_52week is False
    assert record.new_high_52week is True


@pytest.mark.parametrize(
    ("atr", "eligible", "extended"),
    [
        (1.1, True, True),
        (1.1001, True, False),
        (0.0, False, False),
        (float("inf"), False, False),
    ],
)
def test_atr_extension_boundaries(atr, eligible, extended):
    signals = signal_flags_at(
        _feature_row(adjusted_close=110.0, sma50=100.0, atr14=atr),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )

    assert signals.eligibility.atr_extension is eligible
    assert signals.atr_10x_extension is extended


def test_signal_lookup_requires_exact_session_and_invalid_prices_are_ineligible():
    missing = signal_flags_at(
        _feature_row(), pd.Timestamp("2026-08-22").date(), BreadthFormulaPolicy()
    )
    invalid = signal_flags_at(
        _feature_row(adjusted_close=float("nan")),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )

    assert missing.eligibility.advance_decline is False
    assert invalid.eligibility.advance_decline is False
    assert invalid.eligibility.stockbee_daily is False


def _flat_prices(sessions, close=100.0, *, end="2026-08-21"):
    index = pd.bdate_range(end=end, periods=sessions)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )


@pytest.mark.parametrize(
    ("overrides", "eligible", "above"),
    [
        ({"adjusted_close": 100.001, "sma200": 100.0}, True, True),
        ({"adjusted_close": 100.0, "sma200": 100.0}, True, False),
        ({"adjusted_close": 90.0, "sma200": 100.0}, True, False),
        ({"sma200": float("nan")}, False, False),
    ],
)
def test_above_sma200_is_strict_and_needs_the_average(overrides, eligible, above):
    signals = signal_flags_at(
        _feature_row(**overrides), CALCULATION_DATE.date(), BreadthFormulaPolicy()
    )

    assert signals.eligibility.sma200 is eligible
    assert signals.above_sma200 is above


@pytest.mark.parametrize(
    ("close", "eligible", "down"),
    [
        (80.0, True, True),
        (80.01, True, False),
        (float("nan"), False, False),
    ],
)
def test_down_20pct_from_high_includes_the_boundary(close, eligible, down):
    signals = signal_flags_at(
        _feature_row(adjusted_close=close, high_252=100.0),
        CALCULATION_DATE.date(),
        BreadthFormulaPolicy(),
    )

    assert signals.eligibility.off_high is eligible
    assert signals.down_20pct_from_high is down


@pytest.mark.parametrize(
    ("overrides", "near"),
    [
        ({"high_252": 104.0 / 0.85}, True),
        ({"high_252": 130.0}, False),
        ({"high_252": 105.0, "daily_return": 0.039999}, False),
    ],
)
def test_up_4pct_near_high_needs_a_4pct_gain_within_15pct_of_the_high(
    overrides, near
):
    signals = signal_flags_at(
        _feature_row(**overrides), CALCULATION_DATE.date(), BreadthFormulaPolicy()
    )

    assert signals.up_4pct_near_high is near


def test_sma200_needs_200_sessions():
    short = prepare_feature_frame(_flat_prices(199))
    full = prepare_feature_frame(_flat_prices(200))

    short_signals = signal_flags_at(short, CALCULATION_DATE.date(), BreadthFormulaPolicy())
    full_signals = signal_flags_at(full, CALCULATION_DATE.date(), BreadthFormulaPolicy())

    assert short_signals.eligibility.sma200 is False
    assert full_signals.eligibility.sma200 is True


def test_recent_ipo_measures_its_drawdown_from_the_listing_high():
    prices = _flat_prices(30, close=38.0)
    prices.iloc[0, prices.columns.get_loc("High")] = 50.0

    signals = signal_flags_at(
        prepare_feature_frame(prices), CALCULATION_DATE.date(), BreadthFormulaPolicy()
    )

    assert signals.eligibility.sma200 is False
    assert signals.eligibility.off_high is True
    assert signals.down_20pct_from_high is True


@pytest.mark.parametrize(("sessions", "down"), [(252, True), (253, False)])
def test_52_week_high_window_is_252_sessions_including_today(sessions, down):
    prices = _flat_prices(sessions)
    prices.iloc[0, prices.columns.get_loc("High")] = 200.0

    signals = signal_flags_at(
        prepare_feature_frame(prices), CALCULATION_DATE.date(), BreadthFormulaPolicy()
    )

    assert signals.down_20pct_from_high is down
