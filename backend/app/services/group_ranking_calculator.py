"""Pure calculation engine for IBD group rankings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
import math
import statistics
from typing import Mapping, Optional, Sequence

import pandas as pd

from ..scanners.criteria.relative_strength import (
    RelativeStrengthCalculator,
)
from .group_rank_models import GroupRankPrefetchData, GroupRanking


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GroupRankingCandidate:
    industry_group: str
    date: date
    avg_rs_rating: float
    median_rs_rating: float | None
    weighted_avg_rs_rating: float | None
    rs_std_dev: float | None
    num_stocks: int
    num_stocks_rs_above_80: int
    top_symbol: str | None
    top_rs_rating: float | None


@dataclass(frozen=True)
class GroupRankingCalculator:
    rs_calculator: RelativeStrengthCalculator

    def calculate_for_date(
        self,
        *,
        prefetch: GroupRankPrefetchData,
        group_names: Sequence[str],
        calculation_date: date,
    ) -> tuple[GroupRanking, ...]:
        by_date = self.calculate_for_dates(
            prefetch=prefetch,
            group_names=group_names,
            calculation_dates=(calculation_date,),
        )
        return by_date.get(calculation_date, ())

    def calculate_for_dates(
        self,
        *,
        prefetch: GroupRankPrefetchData,
        group_names: Sequence[str],
        calculation_dates: Sequence[date],
    ) -> dict[date, tuple[GroupRanking, ...]]:
        rs_by_date = self._calculate_rs_by_symbol_for_dates(
            prefetch,
            list(calculation_dates),
        )
        result: dict[date, tuple[GroupRanking, ...]] = {}
        for calculation_date in calculation_dates:
            metrics = [
                group_metrics
                for group_name in group_names
                if (
                    group_metrics
                    := self._calculate_group_metrics_from_rs(
                        group_name,
                        prefetch.symbols_by_group.get(group_name, ()),
                        rs_by_date.get(calculation_date, {}),
                        prefetch.market_caps,
                        calculation_date,
                    )
                )
            ]
            metrics.sort(
                key=lambda item: item.avg_rs_rating,
                reverse=True,
            )
            result[calculation_date] = tuple(
                GroupRanking(
                    industry_group=item.industry_group,
                    date=item.date,
                    rank=rank,
                    avg_rs_rating=item.avg_rs_rating,
                    median_rs_rating=item.median_rs_rating,
                    weighted_avg_rs_rating=(
                        item.weighted_avg_rs_rating
                    ),
                    rs_std_dev=item.rs_std_dev,
                    num_stocks=item.num_stocks,
                    num_stocks_rs_above_80=(
                        item.num_stocks_rs_above_80
                    ),
                    top_symbol=item.top_symbol,
                    top_rs_rating=item.top_rs_rating,
                )
                for rank, item in enumerate(metrics, start=1)
            )
        return result

    @staticmethod
    def _calculate_pct_above_80(
        count_above_80: Optional[int],
        total_count: Optional[int],
    ) -> Optional[float]:
        if not total_count:
            return None
        count = count_above_80 or 0
        return round((count / total_count) * 100, 1)

    def _calculate_rs_by_symbol_for_dates(
        self,
        prefetch: GroupRankPrefetchData,
        calculation_dates: Sequence[date],
    ) -> dict[date, dict[str, float]]:
        """Precompute RS ratings once per symbol/date for aggregation."""
        ordered_dates = sorted(calculation_dates)
        result: dict[date, dict[str, float]] = {
            calc_date: {}
            for calc_date in ordered_dates
        }
        benchmark_close = self._close_series(
            prefetch.benchmark_prices
        )
        if benchmark_close is None:
            return result

        periods = self.rs_calculator.PERIODS
        max_period = max(periods.keys())
        benchmark_returns = self._period_returns(
            benchmark_close,
            periods.keys(),
        )
        benchmark_positions = self._positions_by_date(
            benchmark_close.index,
            ordered_dates,
        )

        for symbol, prices in prefetch.prices_by_symbol.items():
            close = self._close_series(prices)
            if close is None:
                continue

            symbol_returns = self._period_returns(
                close,
                periods.keys(),
            )
            symbol_positions = self._positions_by_date(
                close.index,
                ordered_dates,
            )
            for calc_date in ordered_dates:
                latest_idx = symbol_positions.get(calc_date, -1)
                if latest_idx < max_period - 1:
                    continue

                benchmark_idx = benchmark_positions.get(
                    calc_date,
                    -1,
                )
                weighted_performance = 0.0
                total_weight = 0.0
                for period, weight in periods.items():
                    if benchmark_idx < period - 1:
                        continue
                    stock_return = self._series_value_at(
                        symbol_returns[period],
                        latest_idx,
                    )
                    benchmark_return = self._series_value_at(
                        benchmark_returns[period],
                        benchmark_idx,
                    )
                    if (
                        stock_return is None
                        or benchmark_return is None
                    ):
                        continue
                    weighted_performance += (
                        stock_return - benchmark_return
                    ) * weight
                    total_weight += weight

                if total_weight == 0:
                    continue

                relative_performance = (
                    weighted_performance / total_weight
                )
                rs_rating = self._scale_relative_performance_to_rs(
                    relative_performance
                )
                if rs_rating > 0:
                    result[calc_date][symbol] = rs_rating

        return result

    @staticmethod
    def _close_series(
        prices: Optional[pd.DataFrame],
    ) -> Optional[pd.Series]:
        if (
            prices is None
            or prices.empty
            or "Close" not in prices.columns
        ):
            return None
        return prices["Close"].sort_index()

    @staticmethod
    def _period_returns(
        close: pd.Series,
        periods,
    ) -> dict[int, pd.Series]:
        return {
            period: close.pct_change(
                periods=period - 1,
                fill_method=None,
            )
            for period in periods
        }

    @staticmethod
    def _positions_by_date(
        index,
        calculation_dates: Sequence[date],
    ) -> dict[date, int]:
        positions: dict[date, int] = {}
        for calc_date in calculation_dates:
            exclusive_end = (
                pd.Timestamp(calc_date) + pd.Timedelta(days=1)
            )
            if (
                isinstance(index, pd.DatetimeIndex)
                and index.tz is not None
                and exclusive_end.tz is None
            ):
                exclusive_end = exclusive_end.tz_localize(index.tz)
            positions[calc_date] = (
                index.searchsorted(exclusive_end, side="left") - 1
            )
        return positions

    @staticmethod
    def _series_value_at(
        series: pd.Series,
        index: int,
    ) -> Optional[float]:
        try:
            value = series.iloc[index]
        except IndexError:
            return None
        if pd.isna(value) or not math.isfinite(float(value)):
            return None
        return float(value)

    @staticmethod
    def _scale_relative_performance_to_rs(
        relative_performance: float,
    ) -> float:
        if relative_performance >= 0:
            rs_rating = min(
                100,
                50 + (relative_performance * 100),
            )
        else:
            rs_rating = max(
                0,
                50 + (relative_performance * 100),
            )
        return round(rs_rating, 2)

    def _calculate_group_metrics_from_rs(
        self,
        group_name: str,
        symbols: Sequence[str],
        rs_by_symbol: Mapping[str, float],
        market_caps: Mapping[str, float],
        calculation_date: date,
    ) -> Optional[_GroupRankingCandidate]:
        if not symbols:
            logger.debug(
                "No validated symbols found for group: %s",
                group_name,
            )
            return None

        rs_ratings = []
        top_symbol = None
        top_rs = -1
        rs_above_80_count = 0
        weighted_sum = 0.0
        weighted_total = 0.0

        for symbol in symbols:
            rs_rating = rs_by_symbol.get(symbol)
            if rs_rating is None or rs_rating <= 0:
                continue

            rs_ratings.append(rs_rating)
            if rs_rating > top_rs:
                top_rs = rs_rating
                top_symbol = symbol
            if rs_rating >= 80:
                rs_above_80_count += 1

            market_cap = market_caps.get(symbol)
            if market_cap:
                weighted_sum += rs_rating * market_cap
                weighted_total += market_cap

        return self._build_group_metrics(
            group_name=group_name,
            calculation_date=calculation_date,
            rs_ratings=rs_ratings,
            top_symbol=top_symbol,
            top_rs=top_rs,
            rs_above_80_count=rs_above_80_count,
            weighted_sum=weighted_sum,
            weighted_total=weighted_total,
        )

    @staticmethod
    def _build_group_metrics(
        *,
        group_name: str,
        calculation_date: date,
        rs_ratings: list[float],
        top_symbol: str | None,
        top_rs: float,
        rs_above_80_count: int,
        weighted_sum: float,
        weighted_total: float,
    ) -> Optional[_GroupRankingCandidate]:
        if len(rs_ratings) < 3:
            logger.debug(
                "Insufficient data for group %s: only %s stocks "
                "with valid RS",
                group_name,
                len(rs_ratings),
            )
            return None

        avg_rs = sum(rs_ratings) / len(rs_ratings)
        median_rs = statistics.median(rs_ratings)
        rs_std_dev = (
            statistics.pstdev(rs_ratings)
            if len(rs_ratings) > 1
            else None
        )
        weighted_avg_rs = (
            weighted_sum / weighted_total
            if weighted_total > 0
            else None
        )

        return _GroupRankingCandidate(
            industry_group=group_name,
            date=calculation_date,
            avg_rs_rating=round(avg_rs, 2),
            median_rs_rating=round(median_rs, 2),
            weighted_avg_rs_rating=(
                round(weighted_avg_rs, 2)
                if weighted_avg_rs is not None
                else None
            ),
            rs_std_dev=(
                round(rs_std_dev, 2)
                if rs_std_dev is not None
                else None
            ),
            num_stocks=len(rs_ratings),
            num_stocks_rs_above_80=rs_above_80_count,
            top_symbol=top_symbol,
            top_rs_rating=(
                round(top_rs, 2)
                if top_rs > 0
                else None
            ),
        )
