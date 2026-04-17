"""
Beta and Beta-Adjusted Relative Strength Calculator.

Implements Beta calculation for risk-adjusted strength metrics,
enabling Matt Caruso-style screening that accounts for stock volatility.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BetaCalculator:
    """
    Calculate Beta and Beta-Adjusted RS metrics for stocks.

    Beta measures a stock's volatility relative to the market (SPY).
    Beta-Adjusted RS divides RS by Beta to identify stocks that are
    outperforming relative to their risk profile.
    """

    # Configuration
    BETA_PERIOD = 252  # 1 year of trading days
    MIN_DATA_POINTS = 200  # Minimum days required for reliable beta
    BETA_FLOOR = 0.5  # Prevent extreme values for low-beta stocks
    MAX_BETA_ADJ_RS = 100  # Cap result at 100 for consistency

    def __init__(self):
        """Initialize Beta calculator."""
        pass

    def calculate_beta(
        self,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series,
        period: int = None
    ) -> Optional[float]:
        """
        Calculate rolling Beta (252-day) for a stock.

        Beta = Covariance(stock_returns, benchmark_returns) / Variance(benchmark_returns)

        Args:
            stock_prices: Stock price series (chronological order, oldest first)
            benchmark_prices: Benchmark (SPY) price series (chronological order)
            period: Number of days for calculation (default: BETA_PERIOD)

        Returns:
            Beta value or None if insufficient data
        """
        if period is None:
            period = self.BETA_PERIOD

        # Validate data length
        if len(stock_prices) < self.MIN_DATA_POINTS or len(benchmark_prices) < self.MIN_DATA_POINTS:
            logger.debug(f"Insufficient data for beta: {len(stock_prices)} stock, {len(benchmark_prices)} benchmark")
            return None

        try:
            # Use the most recent 'period' days
            stock_slice = stock_prices.iloc[-period:] if len(stock_prices) >= period else stock_prices
            bench_slice = benchmark_prices.iloc[-period:] if len(benchmark_prices) >= period else benchmark_prices

            # Ensure same length
            min_len = min(len(stock_slice), len(bench_slice))
            stock_slice = stock_slice.iloc[-min_len:]
            bench_slice = bench_slice.iloc[-min_len:]

            # Calculate daily returns
            stock_returns = stock_slice.pct_change(fill_method=None).dropna()
            bench_returns = bench_slice.pct_change(fill_method=None).dropna()

            # Align series (in case of any NaN differences)
            min_len = min(len(stock_returns), len(bench_returns))
            if min_len < self.MIN_DATA_POINTS - 1:
                logger.debug(f"Insufficient return data after alignment: {min_len}")
                return None

            stock_returns = stock_returns.iloc[-min_len:]
            bench_returns = bench_returns.iloc[-min_len:]

            # Calculate covariance and variance
            covariance = np.cov(stock_returns, bench_returns)[0][1]
            variance = np.var(bench_returns, ddof=1)

            if variance == 0 or np.isnan(variance):
                logger.debug("Zero or NaN variance in benchmark returns")
                return None

            beta = covariance / variance

            # Sanity check - beta should be reasonable
            if np.isnan(beta) or np.isinf(beta) or beta < -5 or beta > 10:
                logger.debug(f"Unreasonable beta value: {beta}")
                return None

            return round(beta, 3)

        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return None

    def calculate_beta_adjusted_rs(
        self,
        rs_rating: float,
        beta: float
    ) -> Optional[float]:
        """
        Calculate Beta-Adjusted RS (Treynor-style).

        Beta_Adjusted_RS = RS_Rating / max(Beta, BETA_FLOOR)

        Interpretation:
        - Low-beta stocks with high RS get boosted (harder to outperform)
        - High-beta stocks with high RS get penalized (expected to outperform more)

        Args:
            rs_rating: RS Rating (0-100)
            beta: Stock's beta value

        Returns:
            Beta-adjusted RS (0-100) or None if inputs invalid
        """
        if rs_rating is None or beta is None:
            return None

        try:
            # Apply beta floor to prevent extreme values
            effective_beta = max(beta, self.BETA_FLOOR)

            # Calculate adjusted RS
            adjusted_rs = rs_rating / effective_beta

            # Cap at maximum
            adjusted_rs = min(adjusted_rs, self.MAX_BETA_ADJ_RS)

            return round(adjusted_rs, 2)

        except Exception as e:
            logger.warning(f"Error calculating beta-adjusted RS: {e}")
            return None

    def calculate_all_beta_metrics(
        self,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series,
        rs_rating: float,
        rs_rating_1m: Optional[float] = None,
        rs_rating_3m: Optional[float] = None,
        rs_rating_12m: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate Beta and all Beta-Adjusted RS metrics.

        Args:
            stock_prices: Stock price series (chronological order)
            benchmark_prices: Benchmark (SPY) price series (chronological order)
            rs_rating: Weighted RS Rating (0-100)
            rs_rating_1m: 1-month RS Rating (optional)
            rs_rating_3m: 3-month RS Rating (optional)
            rs_rating_12m: 12-month RS Rating (optional)

        Returns:
            Dict with beta and beta-adjusted RS values:
            {
                'beta': float,
                'beta_adj_rs': float,
                'beta_adj_rs_1m': float,
                'beta_adj_rs_3m': float,
                'beta_adj_rs_12m': float,
            }
        """
        # Calculate beta
        beta = self.calculate_beta(stock_prices, benchmark_prices)

        # Calculate all beta-adjusted RS values
        result = {
            'beta': beta,
            'beta_adj_rs': self.calculate_beta_adjusted_rs(rs_rating, beta) if beta else None,
            'beta_adj_rs_1m': self.calculate_beta_adjusted_rs(rs_rating_1m, beta) if beta and rs_rating_1m else None,
            'beta_adj_rs_3m': self.calculate_beta_adjusted_rs(rs_rating_3m, beta) if beta and rs_rating_3m else None,
            'beta_adj_rs_12m': self.calculate_beta_adjusted_rs(rs_rating_12m, beta) if beta and rs_rating_12m else None,
        }

        return result
