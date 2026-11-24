"""Pure per-symbol market-breadth price formulas."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from types import MappingProxyType

import numpy as np
import pandas as pd

from .contributors import BREADTH_CONTRIBUTOR_SIGNALS
from .types import (
    BreadthFormulaPolicy,
    BreadthMarketPolicy,
    SymbolBreadthEvaluation,
    SymbolBreadthSignals,
    SymbolMetricEligibility,
)

_REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
BREADTH_FEATURE_WARMUP_SESSIONS = 251
DOWN_FROM_HIGH_RATIO = 0.80
NEAR_HIGH_RATIO = 0.85


def _normalized_session_index(values: object) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def validate_price_frame(prices: pd.DataFrame) -> None:
    """Reject structural defects that make per-session features ambiguous."""
    missing = [column for column in _REQUIRED_PRICE_COLUMNS if column not in prices]
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")
    try:
        normalized_index = _normalized_session_index(prices.index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Price frame index must contain valid sessions") from exc
    if normalized_index.isna().any():
        raise ValueError("Price frame index must contain valid sessions")
    if normalized_index.has_duplicates:
        raise ValueError("Price frame index must contain unique sessions")


def prices_for_feature_window(
    prices_by_symbol: Mapping[str, pd.DataFrame],
    calculation_dates: tuple[date, ...],
) -> dict[str, pd.DataFrame]:
    """Retain requested sessions plus the exact 251-session warm-up."""
    if not calculation_dates:
        return {}
    first_date = min(calculation_dates)
    last_date = max(calculation_dates)
    result: dict[str, pd.DataFrame] = {}
    for symbol, history in prices_by_symbol.items():
        ordered = history.sort_index()
        session_dates = [pd.Timestamp(value).date() for value in ordered.index]
        in_scope = [
            position
            for position, session_date in enumerate(session_dates)
            if first_date <= session_date <= last_date
        ]
        if not in_scope:
            continue
        start = max(0, in_scope[0] - BREADTH_FEATURE_WARMUP_SESSIONS)
        result[symbol] = ordered.iloc[start : in_scope[-1] + 1]
    return result


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    output = np.full(len(values), np.nan, dtype=float)
    if period <= 0 or len(values) < period:
        return pd.Series(output, index=values.index, dtype=float)

    # The recursion is sequential, so it stays a loop, but over plain Python
    # floats: per-element Series.iloc reads and writes cost ~60x more. The
    # arithmetic is the same IEEE double math, so results are bit-identical.
    raw = values.to_numpy(dtype=float, na_value=np.nan).tolist()
    previous = math.nan
    for position in range(period - 1, len(raw)):
        current = raw[position]
        if math.isnan(current):
            previous = math.nan
        elif math.isnan(previous):
            # Seeding stays a pandas mean so its summation, and therefore the
            # exact float result, is unchanged. It only runs at the start and
            # after a NaN gap.
            restart = values.iloc[position - period + 1 : position + 1]
            previous = float(restart.mean()) if not restart.isna().any() else math.nan
        else:
            previous = ((previous * (period - 1)) + current) / period
        output[position] = previous
    return pd.Series(output, index=values.index, dtype=float)


def prepare_feature_frame(
    prices: pd.DataFrame,
    *,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Prepare adjusted signals and raw local-market liquidity inputs."""

    validate_price_frame(prices)

    frame = prices.loc[:, _REQUIRED_PRICE_COLUMNS].copy()
    frame.index = _normalized_session_index(frame.index)
    frame = frame.sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")

    raw_close = frame["Close"].where(frame["Close"] > 0)
    adjusted_close = frame["Adj Close"].where(frame["Adj Close"] > 0)
    adjustment_factor = adjusted_close / raw_close

    result = pd.DataFrame(index=frame.index)
    result["raw_open"] = frame["Open"]
    result["raw_high"] = frame["High"]
    result["raw_low"] = frame["Low"]
    result["raw_close"] = raw_close
    result["adjusted_open"] = frame["Open"] * adjustment_factor
    result["adjusted_high"] = frame["High"] * adjustment_factor
    result["adjusted_low"] = frame["Low"] * adjustment_factor
    result["adjusted_close"] = adjusted_close
    result["volume"] = frame["Volume"].where(frame["Volume"] >= 0)
    result["raw_close_local"] = raw_close
    result["traded_value_local"] = result["raw_close_local"] * result["volume"]
    result["adtv20_local"] = (
        result["traded_value_local"].rolling(20, min_periods=20).mean()
    )

    result["prior_adjusted_close"] = adjusted_close.shift(1)
    result["prior_volume"] = result["volume"].shift(1)
    result["daily_return"] = adjusted_close / result["prior_adjusted_close"] - 1.0
    result["adjusted_close_20"] = adjusted_close.shift(20)
    result["raw_close_local_20"] = result["raw_close_local"].shift(20)
    result["month_return"] = adjusted_close / result["adjusted_close_20"] - 1.0

    result["low_34"] = adjusted_close.rolling(34, min_periods=34).min()
    result["high_34"] = adjusted_close.rolling(34, min_periods=34).max()
    result["low_65"] = adjusted_close.rolling(65, min_periods=65).min()
    result["high_65"] = adjusted_close.rolling(65, min_periods=65).max()
    result["sma40"] = adjusted_close.rolling(40, min_periods=40).mean()
    result["sma50"] = adjusted_close.rolling(50, min_periods=50).mean()
    result["sma200"] = adjusted_close.rolling(200, min_periods=200).mean()
    # min_periods=1: a recent listing measures from its listing high.
    # A missing High inside the window makes the high unknown, like previous_251_high.
    high_gaps = result["adjusted_high"].isna().rolling(252, min_periods=1).sum()
    result["high_252"] = (
        result["adjusted_high"].rolling(252, min_periods=1).max().where(high_gaps == 0)
    )

    previous_close = adjusted_close.shift(1)
    true_range = pd.concat(
        (
            result["adjusted_high"] - result["adjusted_low"],
            (result["adjusted_high"] - previous_close).abs(),
            (result["adjusted_low"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1, skipna=True)
    invalid_ohlc = (
        result[["adjusted_high", "adjusted_low", "adjusted_close"]].isna().any(axis=1)
    )
    result["true_range"] = true_range.mask(invalid_ohlc)
    result["atr14"] = _wilder_average(result["true_range"], atr_period)

    result["previous_251_high"] = (
        result["adjusted_high"].shift(1).rolling(251, min_periods=251).max()
    )
    result["previous_251_low"] = (
        result["adjusted_low"].shift(1).rolling(251, min_periods=251).min()
    )
    return result


def _row_at(feature_frame: pd.DataFrame, calculation_date: date) -> pd.Series | None:
    target = pd.Timestamp(calculation_date)
    if target not in feature_frame.index:
        return None
    if feature_frame.index.has_duplicates:
        raise ValueError("Feature frame index must contain unique sessions")
    return feature_frame.loc[target]


def _finite(*values: object) -> bool:
    return all(pd.notna(value) and np.isfinite(float(value)) for value in values)


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or bool(
        np.isclose(value, threshold, rtol=1e-12, atol=1e-12)
    )


def _at_most(value: float, threshold: float) -> bool:
    return value < threshold or bool(
        np.isclose(value, threshold, rtol=1e-12, atol=1e-12)
    )


def evaluate_symbol_at(
    feature_frame: pd.DataFrame,
    calculation_date: date,
    policy: BreadthFormulaPolicy,
    market_policy: BreadthMarketPolicy,
    *,
    stockbee_currency_matches: bool = True,
) -> SymbolBreadthEvaluation:
    row = _row_at(feature_frame, calculation_date)
    if row is None:
        return SymbolBreadthEvaluation(
            signals=SymbolBreadthSignals(eligibility=SymbolMetricEligibility()),
            daily_change_pct=None,
            qualifying_values=MappingProxyType({}),
        )

    advance_decline = _finite(row.adjusted_close, row.prior_adjusted_close)
    liquid = (
        stockbee_currency_matches
        and _finite(row.adtv20_local)
        and float(row.adtv20_local) >= market_policy.min_adtv_local
    )
    daily = advance_decline and liquid and _finite(row.volume, row.prior_volume)
    month = (
        liquid
        and _finite(row.adjusted_close, row.adjusted_close_20, row.raw_close_local_20)
        and float(row.raw_close_local_20)
        >= market_policy.min_month_reference_price_local
    )
    day_34 = liquid and _finite(row.adjusted_close, row.low_34, row.high_34)
    quarter = liquid and _finite(row.adjusted_close, row.low_65, row.high_65)
    t2108 = _finite(row.adjusted_close, row.sma40)
    high_low_52week = _finite(
        row.adjusted_high,
        row.adjusted_low,
        row.previous_251_high,
        row.previous_251_low,
    )
    atr_extension = (
        _finite(row.adjusted_close, row.sma50, row.atr14) and float(row.atr14) > 0
    )
    sma200 = _finite(row.adjusted_close, row.sma200)
    off_high = _finite(row.adjusted_close, row.high_252) and float(row.high_252) > 0

    eligibility = SymbolMetricEligibility(
        advance_decline=advance_decline,
        stockbee_liquidity=liquid,
        stockbee_daily=daily,
        stockbee_month=month,
        stockbee_34day=day_34,
        stockbee_quarter=quarter,
        t2108=t2108,
        high_low_52week=high_low_52week,
        atr_extension=atr_extension,
        sma200=sma200,
        off_high=off_high,
    )

    daily_volume_filter = (
        daily
        and float(row.volume) >= market_policy.min_daily_volume
        and float(row.volume) > float(row.prior_volume)
    )
    daily_return = float(row.daily_return) if _finite(row.daily_return) else np.nan
    up_4pct = daily_volume_filter and _at_least(daily_return, 0.04)
    ratio_to_high = (
        float(row.adjusted_close) / float(row.high_252) if off_high else np.nan
    )
    month_return = float(row.month_return) if _finite(row.month_return) else np.nan
    gain_from_low_34 = (
        float(row.adjusted_close / row.low_34 - 1.0) if day_34 else np.nan
    )
    loss_from_high_34 = (
        float(row.adjusted_close / row.high_34 - 1.0) if day_34 else np.nan
    )
    gain_from_low_65 = (
        float(row.adjusted_close / row.low_65 - 1.0) if quarter else np.nan
    )
    loss_from_high_65 = (
        float(row.adjusted_close / row.high_65 - 1.0) if quarter else np.nan
    )

    atr_10x = False
    extension_ratio = np.nan
    if atr_extension:
        gain_from_sma50_pct = (
            float(row.adjusted_close) / float(row.sma50) - 1.0
        ) * 100.0
        atr_pct = float(row.atr14) / float(row.adjusted_close) * 100.0
        extension_ratio = (
            gain_from_sma50_pct / atr_pct
            if gain_from_sma50_pct > 0 and atr_pct > 0
            else np.nan
        )
        atr_10x = (
            gain_from_sma50_pct > 0
            and atr_pct > 0
            and _at_least(
                extension_ratio,
                policy.atr_extension_threshold,
            )
        )

    signals = SymbolBreadthSignals(
        eligibility=eligibility,
        advancing=advance_decline
        and float(row.adjusted_close) > float(row.prior_adjusted_close),
        declining=advance_decline
        and float(row.adjusted_close) < float(row.prior_adjusted_close),
        unchanged=advance_decline
        and float(row.adjusted_close) == float(row.prior_adjusted_close),
        up_4pct=up_4pct,
        down_4pct=daily_volume_filter and _at_most(daily_return, -0.04),
        up_25pct_quarter=quarter and _at_least(gain_from_low_65, 0.25),
        down_25pct_quarter=quarter and _at_most(loss_from_high_65, -0.25),
        up_25pct_month=month and _at_least(month_return, 0.25),
        down_25pct_month=month and _at_most(month_return, -0.25),
        up_50pct_month=month and _at_least(month_return, 0.50),
        down_50pct_month=month and _at_most(month_return, -0.50),
        up_13pct_34days=day_34 and _at_least(gain_from_low_34, 0.13),
        down_13pct_34days=day_34 and _at_most(loss_from_high_34, -0.13),
        new_high_52week=high_low_52week
        and float(row.adjusted_high) > float(row.previous_251_high),
        new_low_52week=high_low_52week
        and float(row.adjusted_low) < float(row.previous_251_low),
        t2108_above=t2108 and float(row.adjusted_close) > float(row.sma40),
        atr_10x_extension=atr_10x,
        above_sma200=sma200 and float(row.adjusted_close) > float(row.sma200),
        down_20pct_from_high=off_high
        and _at_most(ratio_to_high, DOWN_FROM_HIGH_RATIO),
        up_4pct_near_high=up_4pct
        and off_high
        and _at_least(ratio_to_high, NEAR_HIGH_RATIO),
    )
    candidate_values = {
        "up_4pct": daily_return * 100.0,
        "down_4pct": daily_return * 100.0,
        "up_25pct_quarter": gain_from_low_65 * 100.0,
        "down_25pct_quarter": loss_from_high_65 * 100.0,
        "up_25pct_month": month_return * 100.0,
        "down_25pct_month": month_return * 100.0,
        "up_50pct_month": month_return * 100.0,
        "down_50pct_month": month_return * 100.0,
        "up_13pct_34days": gain_from_low_34 * 100.0,
        "down_13pct_34days": loss_from_high_34 * 100.0,
        "atr_10x_extension": float(extension_ratio),
    }
    qualifying_values = {
        signal_key: candidate_values[signal_key]
        for signal_key, definition in BREADTH_CONTRIBUTOR_SIGNALS.items()
        if getattr(signals, definition.signal_attribute)
    }

    return SymbolBreadthEvaluation(
        signals=signals,
        daily_change_pct=(daily_return * 100.0 if _finite(daily_return) else None),
        qualifying_values=MappingProxyType(qualifying_values),
    )


def signal_flags_at(
    feature_frame: pd.DataFrame,
    calculation_date: date,
    policy: BreadthFormulaPolicy,
    market_policy: BreadthMarketPolicy,
    *,
    stockbee_currency_matches: bool = True,
) -> SymbolBreadthSignals:
    """Compatibility wrapper for callers that only need canonical flags."""
    return evaluate_symbol_at(
        feature_frame,
        calculation_date,
        policy,
        market_policy,
        stockbee_currency_matches=stockbee_currency_matches,
    ).signals
