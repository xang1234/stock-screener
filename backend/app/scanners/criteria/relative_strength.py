"""
Relative Strength (RS) calculation for stocks.

Implements the Minervini/O'Neil methodology for calculating relative strength
rating (0-100) based on price performance vs benchmark over multiple timeframes.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class RelativeStrengthCalculator:
    """
    Calculate Relative Strength rating for stocks.

    Methodology:
    - Compare stock performance vs benchmark (SPY) over multiple periods
    - Weight recent performance more heavily
    - Assign percentile rank (0-100) vs universe
    """

    # Time periods in trading days and their weights
    PERIODS = {
        63: 0.40,   # Last quarter (3 months): 40% weight
        126: 0.20,  # 2nd quarter: 20% weight
        189: 0.20,  # 3rd quarter: 20% weight
        252: 0.20,  # Full year: 20% weight
    }

    def __init__(self, benchmark: str = "SPY"):
        """
        Initialize RS calculator.

        Args:
            benchmark: Benchmark ticker symbol (default: SPY)
        """
        self.benchmark = benchmark

    def calculate_return(self, prices: pd.Series, period: int) -> Optional[float]:
        """
        Calculate return over a specific period.

        Args:
            prices: Price series (most recent first)
            period: Number of days to look back

        Returns:
            Return as decimal (e.g., 0.15 for 15%) or None if insufficient data
        """
        if len(prices) < period:
            logger.warning(f"Insufficient data: {len(prices)} < {period}")
            return None

        try:
            current_price = prices.iloc[0]
            past_price = prices.iloc[period - 1]

            if past_price == 0 or pd.isna(past_price) or pd.isna(current_price):
                return None

            return (current_price - past_price) / past_price

        except (IndexError, ZeroDivisionError) as e:
            logger.error(f"Error calculating return: {e}")
            return None

    def calculate_weighted_performance(
        self,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series
    ) -> Optional[float]:
        """
        Calculate weighted relative performance vs benchmark.

        Args:
            stock_prices: Stock price series (most recent first)
            benchmark_prices: Benchmark price series (most recent first)

        Returns:
            Weighted relative performance or None if error
        """
        if len(stock_prices) < max(self.PERIODS.keys()):
            logger.warning("Insufficient price history for RS calculation")
            return None

        weighted_performance = 0.0
        total_weight = 0.0

        for period, weight in self.PERIODS.items():
            stock_return = self.calculate_return(stock_prices, period)
            benchmark_return = self.calculate_return(benchmark_prices, period)

            if stock_return is None or benchmark_return is None:
                continue

            # Relative performance = stock return - benchmark return
            relative_perf = stock_return - benchmark_return
            weighted_performance += relative_perf * weight
            total_weight += weight

        if total_weight == 0:
            return None

        # Normalize by actual total weight used
        return weighted_performance / total_weight

    def calculate_rs_rating(
        self,
        stock_symbol: str,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series,
        universe_performances: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate RS rating (0-100) for a stock.

        Args:
            stock_symbol: Stock ticker symbol
            stock_prices: Stock price series
            benchmark_prices: Benchmark price series
            universe_performances: List of performance values for all stocks in universe
                                  (for percentile ranking). If None, returns raw performance.

        Returns:
            Dict with RS rating and performance metrics
        """
        # Calculate weighted relative performance
        rel_performance = self.calculate_weighted_performance(
            stock_prices,
            benchmark_prices
        )

        if rel_performance is None:
            return {
                "rs_rating": 0,
                "relative_performance": None,
                "percentile_rank": None,
            }

        # If universe data provided, calculate percentile rank
        if universe_performances:
            # Count how many stocks have worse performance
            better_count = sum(1 for perf in universe_performances if perf < rel_performance)
            percentile = (better_count / len(universe_performances)) * 100
            rs_rating = round(percentile, 2)
        else:
            # Without universe, use a simple scaling
            # Positive performance > 0 gets score 50-100, negative gets 0-50
            if rel_performance >= 0:
                rs_rating = min(100, 50 + (rel_performance * 100))
            else:
                rs_rating = max(0, 50 + (rel_performance * 100))
            rs_rating = round(rs_rating, 2)
            percentile = None

        return {
            "rs_rating": rs_rating,
            "relative_performance": round(rel_performance * 100, 2),  # As percentage
            "percentile_rank": percentile,
        }

    def calculate_period_returns(
        self,
        prices: pd.Series
    ) -> Dict[int, Optional[float]]:
        """
        Calculate returns for all standard periods.

        Args:
            prices: Price series

        Returns:
            Dict mapping period to return percentage
        """
        returns = {}
        for period in self.PERIODS.keys():
            ret = self.calculate_return(prices, period)
            returns[period] = round(ret * 100, 2) if ret is not None else None

        return returns

    def calculate_period_rs_rating(
        self,
        period: int,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series,
        universe_performances: Optional[List[float]] = None
    ) -> float:
        """
        Calculate RS rating for a single period.

        Args:
            period: Number of trading days (e.g., 21, 63, 252)
            stock_prices: Stock price series (most recent first)
            benchmark_prices: Benchmark price series (most recent first)
            universe_performances: List of relative performance values for universe
                                  (for percentile ranking). If None, uses simple scaling.

        Returns:
            RS rating (0-100)
        """
        # Calculate returns for the period
        stock_return = self.calculate_return(stock_prices, period)
        benchmark_return = self.calculate_return(benchmark_prices, period)

        # If insufficient data, return 0
        if stock_return is None or benchmark_return is None:
            return 0.0

        # Calculate relative performance
        rel_performance = stock_return - benchmark_return

        # If universe data provided, calculate percentile rank
        if universe_performances:
            # Count how many stocks have worse performance
            better_count = sum(1 for perf in universe_performances if perf < rel_performance)
            percentile = (better_count / len(universe_performances)) * 100
            rs_rating = round(percentile, 2)
        else:
            # Without universe, use simple scaling
            # Positive performance > 0 gets score 50-100, negative gets 0-50
            if rel_performance >= 0:
                rs_rating = min(100, 50 + (rel_performance * 100))
            else:
                rs_rating = max(0, 50 + (rel_performance * 100))
            rs_rating = round(rs_rating, 2)

        return rs_rating

    def calculate_all_rs_ratings(
        self,
        stock_symbol: str,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series,
        universe_performances: Optional[Dict[int, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Calculate RS ratings for all periods (weighted + individual periods).

        Args:
            stock_symbol: Stock ticker symbol
            stock_prices: Stock price series (most recent first)
            benchmark_prices: Benchmark price series (most recent first)
            universe_performances: Dict mapping period to list of performance values
                                  {21: [floats], 63: [floats], 252: [floats]}
                                  If None, uses simple scaling for all ratings.

        Returns:
            Dict with all RS ratings:
            {
                'rs_rating': float,      # Existing weighted RS
                'rs_rating_1m': float,   # 1-month RS (21 days)
                'rs_rating_3m': float,   # 3-month RS (63 days)
                'rs_rating_12m': float,  # 12-month RS (252 days)
            }
        """
        # Calculate existing weighted RS rating
        weighted_rs_result = self.calculate_rs_rating(
            stock_symbol,
            stock_prices,
            benchmark_prices,
            universe_performances.get('weighted') if universe_performances else None
        )

        # Calculate individual period RS ratings
        rs_1m = self.calculate_period_rs_rating(
            21,
            stock_prices,
            benchmark_prices,
            universe_performances.get(21) if universe_performances else None
        )

        rs_3m = self.calculate_period_rs_rating(
            63,
            stock_prices,
            benchmark_prices,
            universe_performances.get(63) if universe_performances else None
        )

        rs_12m = self.calculate_period_rs_rating(
            252,
            stock_prices,
            benchmark_prices,
            universe_performances.get(252) if universe_performances else None
        )

        return {
            'rs_rating': weighted_rs_result['rs_rating'],
            'rs_rating_1m': rs_1m,
            'rs_rating_3m': rs_3m,
            'rs_rating_12m': rs_12m,
            # Expose weighted detail so callers don't need a second
            # calculate_rs_rating call to recover these fields.
            'relative_performance': weighted_rs_result.get('relative_performance'),
            'percentile_rank': weighted_rs_result.get('percentile_rank'),
        }


def calculate_simple_rs(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
    period: int = 252
) -> Optional[float]:
    """
    Simple RS calculation for a single period.

    Args:
        stock_prices: Stock price series
        benchmark_prices: Benchmark price series
        period: Period in days (default: 252 = 1 year)

    Returns:
        RS rating (0-100) or None
    """
    calc = RelativeStrengthCalculator()

    if len(stock_prices) < period or len(benchmark_prices) < period:
        return None

    stock_return = calc.calculate_return(stock_prices, period)
    bench_return = calc.calculate_return(benchmark_prices, period)

    if stock_return is None or bench_return is None:
        return None

    rel_perf = stock_return - bench_return

    # Simple mapping: 0% relative = 50, positive adds, negative subtracts
    rs = 50 + (rel_perf * 100)
    return max(0, min(100, rs))
