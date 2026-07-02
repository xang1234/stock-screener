"""Buy/sell signal scoring and stop-loss calculation (ported from Repo A)."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .phase_service import (
    calculate_sma,
    calculate_rs_slope,
    calculate_distance_from_sma,
    detect_volatility_contraction,
    calculate_volume_ratio,
    validate_minervini_trend_template,
)

logger = logging.getLogger(__name__)


def find_base_high(prices: pd.Series, window: int = 60) -> Optional[float]:
    if len(prices) < window:
        return None
    return float(prices.iloc[-window:].max())


def find_pivot_high(prices: pd.Series, window: int = 20) -> Optional[float]:
    if len(prices) < window:
        return None
    return float(prices.iloc[-window:].max())


def detect_breakout(price_data: pd.DataFrame, current_price: float, phase_info: Dict,
                    vcp_data: Optional[Dict] = None) -> Dict:
    if phase_info.get("phase") not in [1, 2]:
        return {"is_breakout": False, "breakout_level": None, "breakout_type": None, "volume_confirmed": False}

    close = price_data["Close"]
    volume = price_data.get("Volume", pd.Series([], dtype=float))

    volume_confirmed = False
    volume_ratio = 1.0
    if len(volume) > 20:
        avg_vol = volume.iloc[-21: -1].mean()
        current_vol = volume.iloc[-1]
        if avg_vol > 0:
            volume_ratio = current_vol / avg_vol
        volume_confirmed = volume_ratio >= 1.5

    base_high = find_base_high(close, 60)
    pivot_high = find_pivot_high(close, 20)
    sma_50 = phase_info.get("sma_50")

    is_breakout, breakout_level, breakout_type = False, None, None

    if vcp_data and vcp_data.get("is_vcp") and vcp_data.get("contractions"):
        last_peak = vcp_data["contractions"][-1]["peak_price"]
        if current_price > last_peak:
            is_breakout, breakout_level = True, last_peak
            breakout_type = f"VCP Breakout ({vcp_data['contraction_count']} contractions)"

    if not is_breakout and base_high and current_price > base_high:
        is_breakout, breakout_level, breakout_type = True, base_high, "Base Breakout"
    elif not is_breakout and pivot_high and current_price > pivot_high and (base_high is None or pivot_high < base_high):
        is_breakout, breakout_level, breakout_type = True, pivot_high, "Pivot Breakout"
    elif not is_breakout and sma_50 and current_price > sma_50:
        if len(close) >= 2 and close.iloc[-2] < sma_50 < current_price:
            is_breakout, breakout_level, breakout_type = True, sma_50, "50 SMA Breakout"

    return {
        "is_breakout": is_breakout,
        "breakout_level": round(breakout_level, 2) if breakout_level else None,
        "breakout_type": breakout_type,
        "volume_confirmed": volume_confirmed,
        "volume_ratio": round(volume_ratio, 2),
    }


def calculate_stop_loss(price_data: pd.DataFrame, current_price: float,
                        phase_info: Dict, phase: int) -> float:
    sma_50 = phase_info.get("sma_50", 0)

    if phase == 2:
        recent_low = price_data["Low"].iloc[-10:].min() if len(price_data) >= 10 else price_data["Low"].min()
        swing_low_stop = recent_low * 0.995
        sma_stop = sma_50 * 0.99 if sma_50 > 0 else swing_low_stop
        stop_loss = max(swing_low_stop, sma_stop)
        risk_pct = (current_price - stop_loss) / current_price
        if risk_pct < 0.03:
            stop_loss = current_price * 0.97
        elif risk_pct > 0.10:
            stop_loss = current_price * 0.90
    else:
        base_low = price_data["Low"].iloc[-30:].min() if len(price_data) >= 30 else price_data["Low"].min()
        stop_loss = base_low * 0.99
        risk_pct = (current_price - stop_loss) / current_price
        if risk_pct > 0.10:
            stop_loss = current_price * 0.90

    return stop_loss


def score_buy_signal(ticker: str, price_data: pd.DataFrame, current_price: float,
                     phase_info: Dict, rs_series: pd.Series,
                     fundamentals: Optional[Dict] = None, vcp_data: Optional[Dict] = None) -> Dict:
    phase = phase_info.get("phase", 0)

    if phase != 2:
        return {
            "ticker": ticker, "is_buy": False, "score": 0,
            "reason": f"Not in Phase 2 (currently Phase {phase})", "details": {},
        }

    sma_200_series = phase_info.get("sma_200_series", pd.Series([], dtype=float))
    minervini = validate_minervini_trend_template(current_price, phase_info, sma_200_series)

    if not minervini["passes_template"]:
        return {
            "ticker": ticker, "is_buy": False, "score": 0,
            "reason": f"Fails Minervini template ({minervini['criteria_passed']}/8)",
            "details": {"minervini": minervini},
        }

    score = 0
    details = {}
    reasons = []

    sma_50 = phase_info.get("sma_50", 0)
    sma_200 = phase_info.get("sma_200", 0)
    slope_50 = phase_info.get("slope_50", 0)
    slope_200 = phase_info.get("slope_200", 0)
    distance_50 = phase_info.get("distance_from_50sma", 0)
    distance_200 = phase_info.get("distance_from_200sma", 0)

    trend_score = 0
    distance_component = min(15, max(0, (distance_50 / 15.0 * 10) + (distance_200 / 20.0 * 5)))
    slope_component = min(15, max(0, (slope_50 / 0.08 * 10) + (slope_200 / 0.05 * 5)))
    trend_score += distance_component + slope_component

    breakout_info = detect_breakout(price_data, current_price, phase_info, vcp_data)
    if breakout_info["is_breakout"]:
        trend_score += 10
        details["breakout"] = breakout_info

    if distance_50 > 30:
        trend_score -= 10
    elif distance_50 > 20:
        trend_score -= 5

    score += min(trend_score, 40)
    details["trend_score"] = min(trend_score, 40)

    fundamental_score = 20
    if fundamentals:
        fundamental_score = 0
        eps_yoy = fundamentals.get("eps_yoy_change")
        if eps_yoy is not None:
            fundamental_score += min(15, max(0, ((eps_yoy + 20) / 80.0) * 15))
        else:
            fundamental_score += 7.5
        revenue_yoy = fundamentals.get("revenue_yoy_change")
        if revenue_yoy is not None and revenue_yoy > 0:
            fundamental_score += min(15, (revenue_yoy / 20.0) * 15)
        fundamental_score += 10
    score += fundamental_score
    details["fundamental_score"] = fundamental_score

    volume_score = 5
    if "Volume" in price_data.columns and len(price_data) >= 30:
        recent_prices = price_data["Close"].iloc[-6:]
        recent_volume = price_data["Volume"].iloc[-5:]
        up_vol, down_vol, up_days, down_days = 0, 0, 0, 0
        for i in range(1, len(recent_prices)):
            v = recent_volume.iloc[i - 1]
            if recent_prices.iloc[i] > recent_prices.iloc[i - 1]:
                up_vol += v; up_days += 1
            else:
                down_vol += v; down_days += 1
        avg_up = up_vol / up_days if up_days > 0 else 0
        avg_down = down_vol / down_days if down_days > 0 else 1
        vol_ratio = avg_up / avg_down if avg_down > 0 else 1.0
        volume_score = min(10, max(0, 5 + (vol_ratio - 1.0) * 10))
    score += volume_score
    details["volume_score"] = volume_score

    rs_score = 5
    if len(rs_series) >= 20 and not rs_series.isna().all():
        rs_sl = calculate_rs_slope(rs_series, 20)
        rs_score = min(10, max(0, 5 + (rs_sl * 16.67)))
        details["rs_slope"] = round(rs_sl, 3)
    score += rs_score
    details["rs_score"] = round(rs_score, 2)

    stop_loss = calculate_stop_loss(price_data, current_price, phase_info, phase)
    details["stop_loss"] = stop_loss

    rr_score = 0
    risk_amount = current_price - stop_loss if stop_loss else 0
    reward_target = current_price * 1.30
    reward_amount = reward_target - current_price
    if risk_amount > 0:
        rr_ratio = reward_amount / risk_amount
        rr_score = min(15, ((rr_ratio - 2.0) * 6) + 3) if rr_ratio >= 2.0 else 0
        details["risk_reward_ratio"] = round(rr_ratio, 2)
        details["reward_target"] = round(reward_target, 2)
    score += rr_score
    details["rr_score"] = round(rr_score, 2)

    vcp_bonus = 0
    if vcp_data and vcp_data.get("is_vcp"):
        vcp_quality = vcp_data.get("vcp_quality", 0)
        vcp_bonus = 5 if vcp_quality >= 80 else 3 if vcp_quality >= 60 else 1
    score += vcp_bonus

    final_score = max(0, min(score, 125))
    is_buy = final_score >= 70

    return {
        "ticker": ticker,
        "is_buy": is_buy,
        "score": round(final_score, 1),
        "phase": phase,
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "risk_reward_ratio": details.get("risk_reward_ratio", 0),
        "minervini_criteria_passed": minervini["criteria_passed"],
        "reasons": reasons,
        "details": details,
    }
