"""Scan-row metrics available even when full screener history is unavailable."""

from __future__ import annotations

import math

import pandas as pd

from app.domain.scanning.ports import CanonicalStockRsSource
from app.analysis.patterns.rs_line import (
    DEFAULT_BLUE_DOT_RECENT_DAYS,
    DEFAULT_LOOKBACK,
    RsLineLeadershipSnapshot,
    rs_line_leadership_snapshot,
)
from app.scanners.base_screener import StockData
from app.scanners.criteria.adr_calculator import ADRCalculator
from app.scanners.criteria.price_sparkline import PriceSparklineCalculator
from app.scanners.criteria.relative_strength import RelativeStrengthCalculator
from app.scanners.criteria.rs_resolution import (
    CanonicalStockRsUnavailable,
    resolve_stock_rs,
)
from app.scanners.criteria.rs_sparkline import RSSparklineCalculator


def _calculate_price_change_1d(close_chrono) -> float | None:
    if close_chrono is None or len(close_chrono) < 2:
        return None
    previous = close_chrono.iloc[-2]
    current = close_chrono.iloc[-1]
    if pd.isna(previous) or pd.isna(current):
        return None
    try:
        previous_float = float(previous)
        current_float = float(current)
    except (TypeError, ValueError):
        return None
    if previous_float == 0 or not math.isfinite(previous_float) or not math.isfinite(current_float):
        return None
    return round(((current_float - previous_float) / previous_float) * 100, 2)


def partial_history_metrics(stock_data: StockData) -> dict[str, object]:
    """Calculate row fields that do not require a full scan history."""
    price_data = stock_data.price_data
    precomputed = stock_data.precomputed_scan_context
    close_chrono = (
        precomputed.close_chrono
        if precomputed is not None and precomputed.close_chrono is not None
        else price_data["Close"].reset_index(drop=True)
        if price_data is not None and not price_data.empty and "Close" in price_data.columns
        else None
    )
    close_rev = (
        precomputed.close_rev
        if precomputed is not None and precomputed.close_rev is not None
        else close_chrono[::-1].reset_index(drop=True)
        if close_chrono is not None
        else None
    )
    benchmark_close_chrono = (
        precomputed.benchmark_close_chrono
        if precomputed is not None and precomputed.benchmark_close_chrono is not None
        else stock_data.benchmark_data["Close"].reset_index(drop=True)
        if (
            stock_data.benchmark_data is not None
            and not stock_data.benchmark_data.empty
            and "Close" in stock_data.benchmark_data.columns
        )
        else None
    )
    benchmark_close_rev = (
        precomputed.benchmark_close_rev
        if precomputed is not None and precomputed.benchmark_close_rev is not None
        else benchmark_close_chrono[::-1].reset_index(drop=True)
        if benchmark_close_chrono is not None
        else None
    )

    metrics: dict[str, object] = {
        "rs_rating": None,
        "rs_rating_1m": None,
        "rs_rating_3m": None,
        "rs_rating_12m": None,
        "stage": None,
        "ma_alignment": None,
        "price_sparkline_data": None,
        "price_trend": None,
        "price_change_1d": None,
        "rs_sparkline_data": None,
        "rs_trend": None,
        "adr_percent": None,
        **(
            precomputed.rs_line_leadership
            if precomputed is not None
            else RsLineLeadershipSnapshot.empty()
        ).as_scan_fields(),
    }

    try:
        resolved_rs = resolve_stock_rs(stock_data, lambda: {})
    except CanonicalStockRsUnavailable:
        resolved_rs = {}
    for field in ("rs_rating", "rs_rating_1m", "rs_rating_3m", "rs_rating_12m"):
        if resolved_rs.get(field) is not None:
            metrics[field] = resolved_rs[field]

    if close_chrono is not None:
        price_result = PriceSparklineCalculator().calculate_price_sparkline(close_chrono)
        price_data_result = price_result.get("price_data")
        metrics["price_sparkline_data"] = price_data_result
        metrics["price_trend"] = (
            price_result.get("price_trend") if price_data_result is not None else None
        )
        metrics["price_change_1d"] = (
            price_result.get("price_change_1d")
            if price_result.get("price_change_1d") is not None
            else _calculate_price_change_1d(close_chrono)
        )

    if close_chrono is not None and benchmark_close_chrono is not None:
        rs_result = RSSparklineCalculator().calculate_rs_sparkline(
            close_chrono,
            benchmark_close_chrono,
        )
        rs_data_result = rs_result.get("rs_data")
        metrics["rs_sparkline_data"] = rs_data_result
        metrics["rs_trend"] = (
            rs_result.get("rs_trend") if rs_data_result is not None else None
        )
        if precomputed is None and price_data is not None and stock_data.benchmark_data is not None:
            metrics.update(
                rs_line_leadership_snapshot(
                    price_data["Close"],
                    stock_data.benchmark_data["Close"],
                    lookback=DEFAULT_LOOKBACK,
                    recent_days=DEFAULT_BLUE_DOT_RECENT_DAYS,
                ).as_scan_fields()
            )

    if (
        not resolved_rs
        and not isinstance(stock_data.rs_source, CanonicalStockRsSource)
        and close_rev is not None
        and benchmark_close_rev is not None
        and len(close_rev) >= 21
        and len(benchmark_close_rev) >= 21
    ):
        rs_calc = RelativeStrengthCalculator()
        stock_return = rs_calc.calculate_return(close_rev, 21)
        benchmark_return = rs_calc.calculate_return(benchmark_close_rev, 21)
        if stock_return is not None and benchmark_return is not None:
            metrics["rs_rating_1m"] = rs_calc.calculate_period_rs_rating(
                21,
                close_rev,
                benchmark_close_rev,
                stock_data.rs_universe_performances.get(21)
                if stock_data.rs_universe_performances
                else None,
            )

    if price_data is not None and not price_data.empty:
        # Young IPO rows have little history, so require a fully valid
        # 20-session ADR window instead of the scanner-wide 80% tolerance.
        metrics["adr_percent"] = ADRCalculator().calculate_adr_percent(
            price_data,
            period=20,
            min_valid_rows=20,
        )

    return metrics
