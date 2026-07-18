from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

BALANCED_RS_FORMULA_VERSION = "balanced-horizon-percentile-v2"
LEGACY_RS_FORMULA_VERSION = "legacy-linear-v1"
HORIZON_SESSIONS = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252}
HORIZON_WEIGHTS = {"1m": 0.20, "3m": 0.30, "6m": 0.20, "9m": 0.15, "12m": 0.15}
HORIZONS = tuple(HORIZON_SESSIONS)


@dataclass(frozen=True)
class StockRsScore:
    symbol: str
    overall_rs: int
    rs_1m: int
    rs_3m: int
    rs_6m: int
    rs_9m: int
    rs_12m: int
    weighted_composite: float
    excess_return_1m: float
    excess_return_3m: float
    excess_return_6m: float
    excess_return_9m: float
    excess_return_12m: float

    def as_scanner_fields(self) -> dict[str, int]:
        return {
            "rs_rating": self.overall_rs,
            "rs_rating_1m": self.rs_1m,
            "rs_rating_3m": self.rs_3m,
            "rs_rating_12m": self.rs_12m,
        }


def percentile_ratings(values: Mapping[str, float]) -> dict[str, int]:
    if len(values) < 2:
        raise ValueError("percentile ratings require at least two observations")
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("percentile inputs must be finite")

    ordered = sorted((float(value), symbol) for symbol, value in values.items())
    result: dict[str, int] = {}
    index = 0
    count = len(ordered)
    while index < count:
        end = index
        while end + 1 < count and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + (end + 1)) / 2.0
        rating = 1 + math.floor(98.0 * (average_rank - 1.0) / (count - 1) + 0.5)
        for position in range(index, end + 1):
            result[ordered[position][1]] = int(rating)
        index = end + 1
    return result


def calculate_balanced_rs(
    excess_returns_by_symbol: Mapping[str, Mapping[str, float]],
) -> dict[str, StockRsScore]:
    if len(excess_returns_by_symbol) < 2:
        raise ValueError("balanced RS requires at least two eligible stocks")
    expected = set(HORIZONS)
    normalized: dict[str, dict[str, float]] = {}
    for symbol, values in excess_returns_by_symbol.items():
        missing = expected - set(values)
        if missing:
            raise ValueError(f"{symbol} missing horizons: {', '.join(sorted(missing))}")
        normalized[symbol] = {horizon: float(values[horizon]) for horizon in HORIZONS}

    components = {
        horizon: percentile_ratings(
            {symbol: values[horizon] for symbol, values in normalized.items()}
        )
        for horizon in HORIZONS
    }
    composites = {
        symbol: sum(
            HORIZON_WEIGHTS[horizon] * components[horizon][symbol]
            for horizon in HORIZONS
        )
        for symbol in normalized
    }
    overall = percentile_ratings(composites)

    return {
        symbol: StockRsScore(
            symbol=symbol,
            overall_rs=overall[symbol],
            rs_1m=components["1m"][symbol],
            rs_3m=components["3m"][symbol],
            rs_6m=components["6m"][symbol],
            rs_9m=components["9m"][symbol],
            rs_12m=components["12m"][symbol],
            weighted_composite=composites[symbol],
            excess_return_1m=normalized[symbol]["1m"],
            excess_return_3m=normalized[symbol]["3m"],
            excess_return_6m=normalized[symbol]["6m"],
            excess_return_9m=normalized[symbol]["9m"],
            excess_return_12m=normalized[symbol]["12m"],
        )
        for symbol in normalized
    }
