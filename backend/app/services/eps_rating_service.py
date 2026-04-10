"""
EPS Rating Service for IBD-style earnings quality score calculation.

Calculates an EPS Rating (0-99 percentile) that measures earnings quality by combining:
- Long-term growth (5-year CAGR) - 40% weight
- Short-term acceleration (recent 2 quarters YoY growth) - 50% weight
- Acceleration bonus (reward for accelerating growth) - 10% weight

Formula:
    raw_score = 0.40 * CAGR_5yr + 0.50 * avg(Q1_YoY, Q2_YoY) + 0.10 * (Q1_YoY - Q2_YoY)
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EPSRatingService:
    """
    Service for calculating IBD-style EPS Ratings.

    The EPS Rating measures earnings quality by combining:
    - CAGR (5-year compound annual growth rate) - 40% weight
    - Recent quarterly YoY growth - 50% weight
    - Acceleration bonus - 10% weight
    """

    # Weights for composite score
    ALPHA = 0.40  # 5-year CAGR weight
    BETA = 0.50   # Quarterly avg weight
    GAMMA = 0.10  # Acceleration bonus

    # Capping thresholds for extreme values
    MAX_GROWTH_PCT = 500.0   # Cap extreme growth at 500%
    MIN_GROWTH_PCT = -100.0  # Floor at -100%

    def extract_annual_eps_history(self, annual_income_stmt: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Extract annual EPS values from yfinance income_stmt DataFrame.

        Args:
            annual_income_stmt: DataFrame from ticker.income_stmt (annual)

        Returns:
            List of (year, eps_value) tuples sorted by year descending (most recent first)
        """
        if annual_income_stmt is None or annual_income_stmt.empty:
            return []

        try:
            # Find EPS row (Diluted EPS preferred)
            eps_row = None
            for idx in annual_income_stmt.index:
                idx_str = str(idx).lower()
                if 'diluted eps' in idx_str or 'dilutedeps' in idx_str:
                    eps_row = idx
                    break

            if eps_row is None:
                for idx in annual_income_stmt.index:
                    idx_str = str(idx).lower()
                    if 'basic eps' in idx_str or 'basiceps' in idx_str:
                        eps_row = idx
                        break

            if eps_row is None:
                logger.debug("No EPS row found in annual income statement")
                return []

            # Extract EPS values by year
            eps_history = []
            for col in annual_income_stmt.columns:
                try:
                    year = col.year if hasattr(col, 'year') else int(str(col)[:4])
                    eps_value = annual_income_stmt.loc[eps_row, col]

                    if pd.notna(eps_value):
                        eps_history.append((year, float(eps_value)))
                except Exception as e:
                    logger.debug(f"Error extracting EPS for column {col}: {e}")
                    continue

            # Sort by year descending (most recent first)
            eps_history.sort(key=lambda x: x[0], reverse=True)

            return eps_history

        except Exception as e:
            logger.debug(f"Error extracting annual EPS history: {e}")
            return []

    def calculate_5yr_cagr(self, annual_eps: List[Tuple[int, float]]) -> Tuple[Optional[float], int]:
        """
        Calculate 5-year compound annual growth rate from annual EPS data.

        CAGR = ((end_value / start_value) ^ (1/n)) - 1

        Args:
            annual_eps: List of (year, eps_value) tuples sorted by year descending

        Returns:
            Tuple of (cagr_percentage, years_available)
        """
        if not annual_eps or len(annual_eps) < 2:
            return None, 0

        # Get start and end values
        years_available = min(len(annual_eps), 5)

        # Most recent value (end)
        end_value = annual_eps[0][1]

        # Value from n years ago (start)
        start_idx = years_available - 1
        start_value = annual_eps[start_idx][1]

        # Handle edge cases
        if start_value is None or end_value is None:
            return None, 0

        # For negative to positive or very small denominators
        if abs(start_value) < 0.01:
            # If earnings went from near-zero to positive, that's strong growth
            if end_value > 0.5:
                return min(self.MAX_GROWTH_PCT, 100.0), years_available
            return None, years_available

        try:
            # Standard CAGR calculation
            if start_value > 0 and end_value > 0:
                # Both positive: standard CAGR
                ratio = end_value / start_value
                n_years = years_available - 1
                cagr = (pow(ratio, 1.0 / n_years) - 1) * 100
            elif start_value < 0 and end_value > 0:
                # Turnaround: went from loss to profit
                # Calculate improvement rate
                improvement = (end_value - start_value) / abs(start_value)
                cagr = min(improvement * 20, self.MAX_GROWTH_PCT)  # Scale and cap
            elif start_value > 0 and end_value < 0:
                # Deterioration: went from profit to loss
                cagr = max(-100.0, -50.0)  # Significant penalty
            else:
                # Both negative: check if losses are improving
                if abs(end_value) < abs(start_value):
                    # Losses shrinking
                    improvement = (abs(start_value) - abs(end_value)) / abs(start_value)
                    cagr = improvement * 50  # Moderate bonus
                else:
                    # Losses growing
                    cagr = -25.0  # Penalty

            # Cap extreme values
            cagr = max(self.MIN_GROWTH_PCT, min(self.MAX_GROWTH_PCT, cagr))

            return round(cagr, 2), years_available

        except Exception as e:
            logger.debug(f"Error calculating CAGR: {e}")
            return None, 0

    def extract_quarterly_yoy_growth(self, quarterly_income_stmt: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract most recent 2 quarters' YoY EPS growth from quarterly income statement.

        Args:
            quarterly_income_stmt: DataFrame from ticker.quarterly_income_stmt

        Returns:
            Tuple of (q1_yoy_growth, q2_yoy_growth) as percentages
        """
        if quarterly_income_stmt is None or quarterly_income_stmt.empty:
            return None, None

        if quarterly_income_stmt.shape[1] < 5:
            # Need at least 5 quarters for 2 YoY comparisons
            return None, None

        try:
            # Find EPS row
            eps_row = None
            for idx in quarterly_income_stmt.index:
                idx_str = str(idx).lower()
                if 'diluted eps' in idx_str or 'dilutedeps' in idx_str:
                    eps_row = idx
                    break

            if eps_row is None:
                for idx in quarterly_income_stmt.index:
                    idx_str = str(idx).lower()
                    if 'basic eps' in idx_str or 'basiceps' in idx_str:
                        eps_row = idx
                        break

            if eps_row is None:
                return None, None

            # Columns are ordered most recent first
            q1_yoy = None
            q2_yoy = None

            # Q1 YoY: Compare col[0] to col[4] (most recent vs same quarter last year)
            if quarterly_income_stmt.shape[1] >= 5:
                recent_q1 = quarterly_income_stmt.loc[eps_row, quarterly_income_stmt.columns[0]]
                year_ago_q1 = quarterly_income_stmt.loc[eps_row, quarterly_income_stmt.columns[4]]

                if pd.notna(recent_q1) and pd.notna(year_ago_q1) and abs(year_ago_q1) > 0.01:
                    q1_yoy = ((recent_q1 - year_ago_q1) / abs(year_ago_q1)) * 100
                    q1_yoy = max(self.MIN_GROWTH_PCT, min(self.MAX_GROWTH_PCT, q1_yoy))
                    q1_yoy = round(q1_yoy, 2)

            # Q2 YoY: Compare col[1] to col[5] (prior quarter vs same quarter last year)
            if quarterly_income_stmt.shape[1] >= 6:
                recent_q2 = quarterly_income_stmt.loc[eps_row, quarterly_income_stmt.columns[1]]
                year_ago_q2 = quarterly_income_stmt.loc[eps_row, quarterly_income_stmt.columns[5]]

                if pd.notna(recent_q2) and pd.notna(year_ago_q2) and abs(year_ago_q2) > 0.01:
                    q2_yoy = ((recent_q2 - year_ago_q2) / abs(year_ago_q2)) * 100
                    q2_yoy = max(self.MIN_GROWTH_PCT, min(self.MAX_GROWTH_PCT, q2_yoy))
                    q2_yoy = round(q2_yoy, 2)

            return q1_yoy, q2_yoy

        except Exception as e:
            logger.debug(f"Error extracting quarterly YoY growth: {e}")
            return None, None

    def calculate_raw_score(
        self,
        cagr_5yr: Optional[float],
        q1_yoy: Optional[float],
        q2_yoy: Optional[float]
    ) -> Optional[float]:
        """
        Calculate raw composite EPS score before percentile ranking.

        Formula:
            raw_score = ALPHA * CAGR_5yr + BETA * avg(Q1_YoY, Q2_YoY) + GAMMA * (Q1_YoY - Q2_YoY)

        Where:
            ALPHA = 0.40 (5-year CAGR weight)
            BETA = 0.50 (quarterly avg weight)
            GAMMA = 0.10 (acceleration bonus)

        Args:
            cagr_5yr: 5-year CAGR percentage
            q1_yoy: Most recent quarter YoY growth percentage
            q2_yoy: Prior quarter YoY growth percentage

        Returns:
            Raw score (unbounded, will be percentile-ranked later)
        """
        # Need at least quarterly data
        if q1_yoy is None and q2_yoy is None:
            return None

        # Calculate quarterly average
        if q1_yoy is not None and q2_yoy is not None:
            quarterly_avg = (q1_yoy + q2_yoy) / 2
            acceleration = q1_yoy - q2_yoy
        elif q1_yoy is not None:
            quarterly_avg = q1_yoy
            acceleration = 0
        else:
            quarterly_avg = q2_yoy
            acceleration = 0

        # Use CAGR if available, otherwise use quarterly as proxy
        cagr_component = cagr_5yr if cagr_5yr is not None else quarterly_avg

        # Calculate raw score
        raw_score = (
            self.ALPHA * cagr_component +
            self.BETA * quarterly_avg +
            self.GAMMA * acceleration
        )

        return round(raw_score, 2)

    def calculate_eps_rating_data(
        self,
        annual_income_stmt: pd.DataFrame,
        quarterly_income_stmt: pd.DataFrame
    ) -> Dict:
        """
        Calculate all EPS rating components for a single stock.

        Args:
            annual_income_stmt: DataFrame from ticker.income_stmt
            quarterly_income_stmt: DataFrame from ticker.quarterly_income_stmt

        Returns:
            Dict with all EPS rating fields:
            {
                'eps_5yr_cagr': float or None,
                'eps_q1_yoy': float or None,
                'eps_q2_yoy': float or None,
                'eps_raw_score': float or None,
                'eps_years_available': int
            }
        """
        result = {
            'eps_5yr_cagr': None,
            'eps_q1_yoy': None,
            'eps_q2_yoy': None,
            'eps_raw_score': None,
            'eps_years_available': 0
        }

        # Extract annual EPS history and calculate CAGR
        annual_eps = self.extract_annual_eps_history(annual_income_stmt)
        cagr, years = self.calculate_5yr_cagr(annual_eps)
        result['eps_5yr_cagr'] = cagr
        result['eps_years_available'] = years

        # Extract quarterly YoY growth
        q1_yoy, q2_yoy = self.extract_quarterly_yoy_growth(quarterly_income_stmt)
        result['eps_q1_yoy'] = q1_yoy
        result['eps_q2_yoy'] = q2_yoy

        # Calculate raw score
        raw_score = self.calculate_raw_score(cagr, q1_yoy, q2_yoy)
        result['eps_raw_score'] = raw_score

        return result

    def calculate_percentile_ranks(
        self,
        raw_scores: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Calculate percentile ranks (0-99) for all stocks based on raw scores.

        Args:
            raw_scores: Dict mapping symbol to raw_score

        Returns:
            Dict mapping symbol to eps_rating (0-99)
        """
        if not raw_scores:
            return {}

        # Filter out None values
        valid_scores = {
            symbol: score
            for symbol, score in raw_scores.items()
            if score is not None
        }

        if not valid_scores:
            return {}

        # Convert to numpy array for percentile calculation
        symbols = list(valid_scores.keys())
        scores = np.array([valid_scores[s] for s in symbols])

        # Calculate percentile for each score
        # scipy.stats.percentileofscore gives percentile (0-100), we want 0-99
        ratings = {}
        for i, symbol in enumerate(symbols):
            # Count how many scores are below this one
            below = np.sum(scores < scores[i])
            equal = np.sum(scores == scores[i])

            # Percentile: percentage of values that fall below
            # Using 'weak' method: strictly less than
            percentile = (below + 0.5 * (equal - 1)) / len(scores) * 100

            # Convert to 0-99 scale
            eps_rating = min(99, max(0, int(percentile)))
            ratings[symbol] = eps_rating

        return ratings
