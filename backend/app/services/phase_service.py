"""Phase classification and Minervini trend template logic (ported from Repo A)."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    if len(prices) < period:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_slope(series: pd.Series, periods: int = 20) -> float:
    if len(series) < periods or series.isna().all():
        return 0.0
    recent = series.iloc[-periods:].dropna()
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent))
    y = recent.values
    if np.std(x) == 0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    avg_price = np.mean(y)
    if avg_price == 0:
        return 0.0
    return (slope / avg_price) * 100


def calculate_distance_from_sma(price: float, sma: float) -> float:
    if sma == 0:
        return 0.0
    return ((price - sma) / sma) * 100


def detect_volatility_contraction(prices: pd.Series, window: int = 20) -> Dict:
    if len(prices) < window * 2:
        return {"is_contracting": False, "contraction_quality": 0.0, "current_volatility": 0.0}
    volatility = prices.rolling(window=window).std()
    if len(volatility.dropna()) < 2:
        return {"is_contracting": False, "contraction_quality": 0.0, "current_volatility": 0.0}
    current_vol = volatility.iloc[-1]
    avg_vol = volatility.iloc[-window * 2: -window].mean()
    contraction_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    is_contracting = contraction_ratio < 0.7
    quality = max(0, min(100, (1 - contraction_ratio) * 100))
    return {
        "is_contracting": is_contracting,
        "contraction_quality": round(quality, 2),
        "current_volatility": round(current_vol, 2),
        "contraction_ratio": round(contraction_ratio, 2),
    }


def calculate_volume_ratio(volumes: pd.Series, period: int = 20) -> float:
    if len(volumes) < period + 1:
        return 1.0
    current = volumes.iloc[-1]
    avg = volumes.iloc[-period - 1: -1].mean()
    if avg == 0:
        return 1.0
    return current / avg


def calculate_relative_strength(stock_prices: pd.Series, spy_prices: pd.Series, period: int = 63) -> pd.Series:
    if len(stock_prices) == 0 or len(spy_prices) == 0:
        return pd.Series([np.nan] * len(stock_prices), index=stock_prices.index)
    stock_prices = stock_prices.copy()
    spy_prices = spy_prices.copy()
    if not isinstance(stock_prices.index, pd.DatetimeIndex):
        return pd.Series([np.nan] * len(stock_prices), index=stock_prices.index)
    if not isinstance(spy_prices.index, pd.DatetimeIndex):
        return pd.Series([np.nan] * len(stock_prices), index=stock_prices.index)
    if stock_prices.index.tz is not None:
        stock_prices.index = stock_prices.index.tz_localize(None)
    if spy_prices.index.tz is not None:
        spy_prices.index = spy_prices.index.tz_localize(None)
    spy_aligned = spy_prices.reindex(stock_prices.index, method="ffill")
    if spy_aligned.isna().all():
        return pd.Series([np.nan] * len(stock_prices), index=stock_prices.index)
    rs = (stock_prices / spy_aligned) * 100
    return rs.ffill()


def calculate_rs_slope(rs_series: pd.Series, periods: int = 15) -> float:
    return calculate_slope(rs_series, periods)


def classify_phase(price_data: pd.DataFrame, current_price: Optional[float] = None) -> Dict:
    """Classify Weinstein stage (1-4) from OHLCV DataFrame."""
    if len(price_data) < 200:
        return {"phase": 0, "phase_name": "Insufficient Data", "confidence": 0.0, "reasons": []}

    close = price_data["Close"]
    high = price_data["High"]
    low = price_data["Low"]
    volume = price_data.get("Volume", pd.Series([], dtype=float))

    if current_price is None:
        current_price = float(close.iloc[-1])

    sma_50 = calculate_sma(close, 50)
    sma_150 = calculate_sma(close, 150)
    sma_200 = calculate_sma(close, 200)

    if sma_50.isna().all() or sma_200.isna().all():
        return {"phase": 0, "phase_name": "Insufficient Data", "confidence": 0.0, "reasons": []}

    sma_50_val = float(sma_50.iloc[-1])
    sma_150_val = float(sma_150.iloc[-1]) if not sma_150.isna().all() else 0.0
    sma_200_val = float(sma_200.iloc[-1])

    if len(close) >= 252:
        week_52_high = float(high.iloc[-252:].max())
        week_52_low = float(low.iloc[-252:].min())
    else:
        week_52_high = float(high.max())
        week_52_low = float(low.min())

    slope_50 = calculate_slope(sma_50, 20)
    slope_200 = calculate_slope(sma_200, 20)
    vol_data = detect_volatility_contraction(close, 20)

    if len(volume) > 20:
        avg_volume = volume.iloc[-20:].mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    else:
        volume_ratio = 1.0

    reasons = []
    confidence = 0.0

    if current_price < sma_50_val and current_price < sma_200_val and sma_50_val < sma_200_val:
        phase, phase_name = 4, "Downtrend"
        reasons.append(f"Price below both 50 SMA and 200 SMA")
        confidence = 70
        if slope_50 < 0 and slope_200 < 0:
            confidence += 20
        if slope_50 < 0:
            confidence += 10
    elif current_price > sma_50_val and sma_50_val > sma_200_val and slope_50 > 0:
        phase, phase_name = 2, "Uptrend/Breakout"
        reasons.append("Price above 50 SMA; 50 SMA > 200 SMA (Golden Cross)")
        confidence = 70
        if slope_200 > 0:
            confidence += 15
        if volume_ratio > 1.2:
            confidence += 15
    elif current_price > sma_50_val and calculate_distance_from_sma(current_price, sma_50_val) > 25:
        phase, phase_name = 3, "Distribution/Top"
        reasons.append(f"Price extended above 50 SMA")
        confidence = 60
    else:
        phase, phase_name = 1, "Base Building"
        reasons.append("Price in consolidation")
        confidence = 50
        if abs(slope_50) < 0.1:
            confidence += 15
        if vol_data["is_contracting"]:
            confidence += 15

    return {
        "phase": phase,
        "phase_name": phase_name,
        "confidence": min(confidence, 100),
        "reasons": reasons,
        "sma_50": round(sma_50_val, 2),
        "sma_150": round(sma_150_val, 2),
        "sma_200": round(sma_200_val, 2),
        "slope_50": round(slope_50, 4),
        "slope_200": round(slope_200, 4),
        "distance_from_50sma": round(calculate_distance_from_sma(current_price, sma_50_val), 2),
        "distance_from_200sma": round(calculate_distance_from_sma(current_price, sma_200_val), 2),
        "week_52_high": round(week_52_high, 2),
        "week_52_low": round(week_52_low, 2),
        "volatility_contraction": vol_data,
        "sma_200_series": sma_200,
    }


def validate_minervini_trend_template(current_price: float, phase_info: Dict, sma_200_series: pd.Series) -> Dict:
    sma_50 = phase_info.get("sma_50", 0)
    sma_150 = phase_info.get("sma_150", 0)
    sma_200 = phase_info.get("sma_200", 0)
    week_52_high = phase_info.get("week_52_high", 0)
    week_52_low = phase_info.get("week_52_low", 0)

    criteria = {}
    passed_count = 0

    c1 = current_price > sma_150 and current_price > sma_200
    criteria["price_above_150_200"] = c1
    passed_count += int(c1)

    c2 = sma_150 > sma_200
    criteria["sma_150_above_200"] = c2
    passed_count += int(c2)

    if len(sma_200_series) >= 20:
        c3 = float(sma_200_series.iloc[-1]) > float(sma_200_series.iloc[-20])
    else:
        c3 = phase_info.get("slope_200", 0) > 0
    criteria["sma_200_rising"] = c3
    passed_count += int(c3)

    c4 = sma_50 > sma_150
    criteria["sma_50_above_150"] = c4
    passed_count += int(c4)

    c5 = current_price > sma_50
    criteria["price_above_50"] = c5
    passed_count += int(c5)

    if week_52_low > 0:
        dist_low = ((current_price - week_52_low) / week_52_low) * 100
        c6 = dist_low >= 30
        criteria["distance_from_52w_low_pct"] = round(dist_low, 1)
    else:
        c6 = False
        criteria["distance_from_52w_low_pct"] = 0
    criteria["price_30pct_above_52w_low"] = c6
    passed_count += int(c6)

    if week_52_high > 0:
        dist_high = ((week_52_high - current_price) / week_52_high) * 100
        c7 = dist_high <= 25
        criteria["distance_from_52w_high_pct"] = round(dist_high, 1)
    else:
        c7 = False
        criteria["distance_from_52w_high_pct"] = 100
    criteria["price_near_52w_high"] = c7
    passed_count += int(c7)

    c8 = phase_info.get("phase") == 2
    criteria["confirmed_stage_2"] = c8
    passed_count += int(c8)

    return {
        "passes_template": passed_count >= 7,
        "criteria_passed": passed_count,
        "criteria_total": 8,
        "template_score": int((passed_count / 8) * 100),
        "criteria_details": criteria,
    }


def detect_vcp_pattern(price_data: pd.DataFrame, current_price: float, phase_info: Dict,
                       min_contractions: int = 2, max_contractions: int = 6) -> Dict:
    if len(price_data) < 60:
        return {
            "is_vcp": False, "vcp_quality": 0, "contractions": [], "contraction_count": 0,
            "base_length_weeks": 0, "breakout_volume_ratio": 0.0, "pattern_details": "Insufficient data",
            "quality_factors": [], "contraction_quality": 0, "volume_quality": 0,
            "near_52w_high": False, "distance_from_52w_high_pct": 100,
        }

    close = price_data["Close"]
    high = price_data["High"]
    low = price_data["Low"]
    volume = price_data.get("Volume", pd.Series([], dtype=float))

    lookback = min(len(price_data), 325)
    base_data = price_data.tail(lookback)
    window = 10

    peaks, troughs = [], []
    bh = base_data["High"].rolling(window=window, center=True).max()
    bl = base_data["Low"].rolling(window=window, center=True).min()

    for i in range(window, len(base_data) - window):
        if base_data["High"].iloc[i] == bh.iloc[i]:
            if (base_data["High"].iloc[i] > base_data["High"].iloc[i - 5: i].max() and
                    base_data["High"].iloc[i] > base_data["High"].iloc[i + 1: i + 6].max()):
                peaks.append({"index": i, "date": base_data.index[i], "price": base_data["High"].iloc[i]})
        if base_data["Low"].iloc[i] == bl.iloc[i]:
            if (base_data["Low"].iloc[i] < base_data["Low"].iloc[i - 5: i].min() and
                    base_data["Low"].iloc[i] < base_data["Low"].iloc[i + 1: i + 6].min()):
                troughs.append({"index": i, "date": base_data.index[i], "price": base_data["Low"].iloc[i]})

    contractions = []
    if len(peaks) >= 2 and len(troughs) >= 2:
        for peak in peaks[:-1]:
            matching = [t for t in troughs if t["index"] > peak["index"]]
            if matching:
                trough = matching[0]
                drawdown_pct = ((peak["price"] - trough["price"]) / peak["price"]) * 100
                if len(volume) > 0:
                    avg_vol_b = volume.iloc[: peak["index"]].tail(20).mean()
                    avg_vol_d = volume.iloc[peak["index"]: trough["index"]].mean()
                    vol_ratio = avg_vol_d / avg_vol_b if avg_vol_b > 0 else 1.0
                else:
                    vol_ratio = 1.0
                try:
                    dur = (trough["date"] - peak["date"]).days
                except Exception:
                    dur = 0
                contractions.append({
                    "number": len(contractions) + 1,
                    "peak_price": peak["price"], "trough_price": trough["price"],
                    "drawdown_pct": round(drawdown_pct, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "duration_days": dur,
                })

    is_contracting = False
    contraction_quality = 0.0
    if len(contractions) >= min_contractions:
        contracting_count = sum(1 for i in range(1, len(contractions))
                                if contractions[i]["drawdown_pct"] < contractions[i - 1]["drawdown_pct"])
        if len(contractions) > 1:
            contraction_quality = (contracting_count / (len(contractions) - 1)) * 100
            is_contracting = contraction_quality >= 50

    volume_quality = 0.0
    if len(contractions) >= 2:
        drying = sum(1 for c in contractions if c["volume_ratio"] < 1.0)
        volume_quality = (drying / len(contractions)) * 100

    base_length_weeks = 0.0
    if len(contractions) > 0:
        try:
            base_length_weeks = (base_data.index[-1] - contractions[0].get("peak_date", base_data.index[0])).days / 7
        except Exception:
            pass

    breakout_volume_ratio = 1.0
    if len(volume) > 20:
        avg_vol_20 = volume.iloc[-21: -1].mean()
        if avg_vol_20 > 0:
            breakout_volume_ratio = volume.iloc[-1] / avg_vol_20

    week_52_high = phase_info.get("week_52_high", current_price)
    dist_52w = ((week_52_high - current_price) / week_52_high * 100) if week_52_high > 0 else 100
    near_52w = dist_52w <= 25

    vcp_quality = 0.0
    quality_factors = []
    if min_contractions <= len(contractions) <= max_contractions:
        cs = min(20, (len(contractions) / max_contractions) * 20)
        vcp_quality += cs
        quality_factors.append(f"{len(contractions)} contractions ({cs:.0f} pts)")
    vcp_quality += (contraction_quality / 100) * 30
    vcp_quality += (volume_quality / 100) * 20
    if 3 <= base_length_weeks <= 65:
        vcp_quality += 10
    if near_52w:
        hs = max(0, 20 - (dist_52w / 25 * 20))
        vcp_quality += hs

    is_vcp = (
        len(contractions) >= min_contractions and
        len(contractions) <= max_contractions and
        is_contracting and
        vcp_quality >= 50
    )

    if len(contractions) >= min_contractions:
        recent = contractions[-4:]
        sizes = [f"{c['drawdown_pct']:.1f}%" for c in recent]
        pattern_details = f"{len(contractions)} contractions: {' → '.join(sizes)}"
    else:
        pattern_details = f"Only {len(contractions)} contraction(s) detected (need {min_contractions}+)"

    return {
        "is_vcp": is_vcp,
        "vcp_quality": round(vcp_quality, 1),
        "contractions": contractions,
        "contraction_count": len(contractions),
        "contraction_quality": round(contraction_quality, 1),
        "volume_quality": round(volume_quality, 1),
        "base_length_weeks": round(base_length_weeks, 1),
        "breakout_volume_ratio": round(breakout_volume_ratio, 2),
        "near_52w_high": near_52w,
        "distance_from_52w_high_pct": round(dist_52w, 1),
        "pattern_details": pattern_details,
        "quality_factors": quality_factors,
    }
