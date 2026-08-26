"""Server-side port of the Options Analytics dashboard's "Executive Signal"
weighted-factor scoring (see evaluateMarketFactors/getMarketSignal in
frontend/src/pages/OptionsAnalyticsDashboardPage.jsx).

That scoring only ever existed in the browser, computed per-symbol from a
single live payload. This module re-implements the exact same weights,
vote thresholds, and score bands against a persisted OptionsMetricsSnapshot
row instead, so it can run server-side across an entire universe (for the
Command Center's ranking tables and alert generation) without duplicating
divergent logic in two languages. Keep this in sync with the frontend
version by hand -- there is no shared source of truth between them, so a
change to one's weights/thresholds should be mirrored in the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SignalFactor:
    key: str
    weight: float
    vote: int  # -1, 0, or +1
    sentence: str


@dataclass
class MarketSignal:
    factors: List[SignalFactor] = field(default_factory=list)
    score: float = 0.0
    label: str = "Neutral"  # Buy | Bullish | Neutral | Bearish | Sell


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_market_factors(snapshot: Any) -> List[SignalFactor]:
    """Build the same factor list as the frontend's evaluateMarketFactors(),
    reading from an OptionsMetricsSnapshot ORM row (or any object/dict
    exposing the same attribute names). A factor is included only when its
    inputs are present -- exactly mirroring the frontend's `if (x != null)`
    guards, so a row with sparse fields (e.g. a batch_abbreviated row)
    naturally produces fewer factors rather than treating missing data as
    neutral.
    """

    def get(name: str):
        if isinstance(snapshot, dict):
            return snapshot.get(name)
        return getattr(snapshot, name, None)

    factors: List[SignalFactor] = []

    total_gex = _coerce_float(get("total_gex"))
    if total_gex is not None:
        vote = 1 if total_gex > 0 else -1 if total_gex < 0 else 0
        factors.append(SignalFactor(
            key="gex", weight=2, vote=vote,
            sentence=(
                "Total GEX is positive, indicating a long-gamma regime that may support upward "
                "pressure as option sellers hedge into rising prices." if vote == 1 else
                "Total GEX is negative, indicating a short-gamma regime that may amplify downside "
                "moves as option sellers hedge into falling prices." if vote == -1 else
                "Total GEX is neutral, indicating balanced gamma exposure."
            ),
        ))

    skew = _coerce_float(get("skew"))
    if skew is not None:
        vote = 1 if skew < 0 else -1 if skew > 0 else 0
        factors.append(SignalFactor(
            key="skew", weight=1, vote=vote,
            sentence=(
                "Volatility skew is positive, showing put skew and suggesting demand for downside "
                "protection." if vote == -1 else
                "Volatility skew is negative, showing call skew and suggesting bullish interest in "
                "upside risk." if vote == 1 else
                "Volatility skew is neutral, showing no strong call/put bias."
            ),
        ))

    max_pain = _coerce_float(get("max_pain_distance_pct"))
    if max_pain is not None:
        vote = -1 if max_pain > 0.5 else 1 if max_pain < -0.5 else 0
        factors.append(SignalFactor(
            key="maxPain", weight=0.5, vote=vote,
            sentence=(
                "Price is above max pain, which may reflect heavier call exposure and a mild pull "
                "back toward max pain into expiry." if vote == -1 else
                "Price is below max pain, which may reflect heavier put exposure and a mild pull "
                "back toward max pain into expiry." if vote == 1 else
                "Price is close to max pain, suggesting the options market is relatively balanced "
                "around current levels."
            ),
        ))

    call_premium = _coerce_float(get("call_premium_notional"))
    put_premium = _coerce_float(get("put_premium_notional"))
    if call_premium is not None and put_premium is not None:
        premium_pcr = put_premium / (call_premium or 1)
        vote = 1 if premium_pcr < 0.7 else -1 if premium_pcr > 1.5 else 0
        factors.append(SignalFactor(
            key="premiumPcr", weight=1, vote=vote,
            sentence=(
                "Premium put/call ratio is call-biased -- real dollars are flowing predominantly "
                "into calls today." if vote == 1 else
                "Premium put/call ratio is put-biased -- real dollars are flowing predominantly "
                "into puts today." if vote == -1 else
                "Premium put/call ratio is roughly balanced between calls and puts."
            ),
        ))

    call_oi = _coerce_float(get("total_call_oi"))
    put_oi = _coerce_float(get("total_put_oi"))
    if call_oi is not None and put_oi is not None:
        vote = 1 if call_oi > put_oi * 1.15 else -1 if put_oi > call_oi * 1.15 else 0
        factors.append(SignalFactor(
            key="openInterest", weight=1, vote=vote,
            sentence=(
                "Call open interest exceeds put open interest, suggesting bullish or "
                "resistance-testing positioning." if vote == 1 else
                "Put open interest exceeds call open interest, suggesting protective or bearish "
                "positioning." if vote == -1 else
                "Call and put open interest are roughly balanced."
            ),
        ))

    spot = _coerce_float(get("underlying_price"))
    call_wall = _coerce_float(get("call_wall"))
    put_wall = _coerce_float(get("put_wall"))
    if spot is not None and call_wall is not None and spot >= call_wall:
        factors.append(SignalFactor(
            key="callWallBreak", weight=1, vote=1,
            sentence="Price has pushed above the call wall, suggesting that resistance is no longer holding.",
        ))
    if spot is not None and put_wall is not None and spot <= put_wall:
        factors.append(SignalFactor(
            key="putWallBreak", weight=1, vote=-1,
            sentence="Price has fallen below the put wall, suggesting that support is no longer holding.",
        ))

    return factors


def get_market_signal(factors: List[SignalFactor]) -> MarketSignal:
    """Same score bands as the frontend's getMarketSignal(). Thresholds are
    absolute, not scaled to how many factors fired -- a row with only GEX
    available (weight 2) can still reach Bullish/Bearish on its own, exactly
    matching the frontend's documented behavior."""
    if not factors:
        return MarketSignal(factors=[], score=0.0, label="Neutral")

    score = sum(f.weight * f.vote for f in factors)

    if score >= 3:
        label = "Buy"
    elif score >= 1:
        label = "Bullish"
    elif score <= -3:
        label = "Sell"
    elif score <= -1:
        label = "Bearish"
    else:
        label = "Neutral"

    return MarketSignal(factors=factors, score=score, label=label)


def evaluate_snapshot_signal(snapshot: Any) -> MarketSignal:
    """Convenience wrapper: factors + score + label in one call."""
    factors = evaluate_market_factors(snapshot)
    return get_market_signal(factors)
