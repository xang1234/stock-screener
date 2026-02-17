"""
Base screener abstractions for multi-screener architecture.

Provides the foundation for implementing multiple screening strategies:
- BaseStockScreener: Abstract base class for all screeners
- DataRequirements: Specifies what data a screener needs
- ScreenerResult: Standardized result format from screeners
- StockData: Container for all fetched stock data
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import pandas as pd


@dataclass
class DataRequirements:
    """
    Specifies what data a screener needs to perform its analysis.

    This allows the data preparation layer to fetch all required data
    once and share it across multiple screeners.
    """
    price_period: str = "2y"  # How much price history: "1y", "2y", "5y"
    needs_fundamentals: bool = False  # Basic fundamentals (market cap, P/E, etc.)
    needs_quarterly_growth: bool = False  # Quarterly earnings/revenue growth
    needs_benchmark: bool = False  # SPY benchmark data for RS calculation
    needs_earnings_history: bool = False  # Historical earnings data

    def merge(self, other: 'DataRequirements') -> 'DataRequirements':
        """
        Merge two data requirements, taking the union of needs.

        Args:
            other: Another DataRequirements instance

        Returns:
            New DataRequirements with merged needs
        """
        # Take the longer period
        periods = {"1y": 1, "2y": 2, "5y": 5, "max": 10}
        self_years = periods.get(self.price_period, 2)
        other_years = periods.get(other.price_period, 2)
        merged_period = self.price_period if self_years >= other_years else other.price_period

        return DataRequirements(
            price_period=merged_period,
            needs_fundamentals=self.needs_fundamentals or other.needs_fundamentals,
            needs_quarterly_growth=self.needs_quarterly_growth or other.needs_quarterly_growth,
            needs_benchmark=self.needs_benchmark or other.needs_benchmark,
            needs_earnings_history=self.needs_earnings_history or other.needs_earnings_history
        )

    @classmethod
    def merge_all(cls, requirements_list: list['DataRequirements']) -> 'DataRequirements':
        """Merge a list of data requirements into one, taking the union of needs.

        Args:
            requirements_list: List of DataRequirements to merge

        Returns:
            Single DataRequirements with all needs merged
        """
        if not requirements_list:
            return cls()
        result = requirements_list[0]
        for req in requirements_list[1:]:
            result = result.merge(req)
        return result


@dataclass
class ScreenerResult:
    """
    Standardized result format from a screener.

    All screeners return this format for consistent processing.
    """
    score: float  # Overall score (0-100)
    passes: bool  # Whether stock passes the screener
    rating: str  # Human-readable rating: "Strong Buy", "Buy", "Watch", "Pass"
    breakdown: Dict[str, float]  # Component scores (e.g., {"rs": 85, "stage": 90})
    details: Dict[str, Any]  # Full analysis details
    screener_name: str  # Name of screener that produced this result

    def __post_init__(self):
        """Validate score is in valid range."""
        if not 0 <= self.score <= 100:
            raise ValueError(f"Score must be between 0 and 100, got {self.score}")


@dataclass
class StockData:
    """
    Container for all fetched stock data.

    This is passed to screeners so they don't need to fetch data themselves.
    All data is fetched once by the data preparation layer and shared.
    """
    symbol: str
    price_data: pd.DataFrame  # OHLCV data with DatetimeIndex
    benchmark_data: pd.DataFrame  # SPY benchmark data (same format as price_data)
    fundamentals: Optional[Dict[str, Any]] = None  # Basic fundamentals
    quarterly_growth: Optional[Dict[str, Any]] = None  # Quarterly growth metrics
    earnings_history: Optional[pd.DataFrame] = None  # Historical earnings data

    # Additional metadata
    fetch_errors: Dict[str, str] = field(default_factory=dict)  # Any errors during fetch

    def has_sufficient_data(self, min_days: int = 100) -> bool:
        """
        Check if we have sufficient price data for analysis.

        Args:
            min_days: Minimum number of days required

        Returns:
            True if sufficient data available
        """
        if self.price_data is None or self.price_data.empty:
            return False
        return len(self.price_data) >= min_days

    def get_current_price(self) -> Optional[float]:
        """Get the most recent closing price."""
        if self.price_data is None or self.price_data.empty:
            return None
        return float(self.price_data['Close'].iloc[-1])

    def get_error_summary(self) -> Optional[str]:
        """Get summary of any fetch errors."""
        if not self.fetch_errors:
            return None
        return "; ".join(f"{k}: {v}" for k, v in self.fetch_errors.items())


class BaseStockScreener(ABC):
    """
    Abstract base class for all stock screeners.

    Subclasses must implement:
    - screener_name: Unique identifier for the screener
    - get_data_requirements: What data the screener needs
    - scan_stock: Perform the actual screening logic
    - calculate_rating: Convert score to human-readable rating
    """

    @property
    @abstractmethod
    def screener_name(self) -> str:
        """
        Unique identifier for this screener.

        Examples: "minervini", "canslim", "ipo", "custom"
        """
        pass

    @abstractmethod
    def get_data_requirements(self, criteria: Optional[Dict] = None) -> DataRequirements:
        """
        Specify what data this screener needs.

        Args:
            criteria: Optional criteria that might affect data needs

        Returns:
            DataRequirements object
        """
        pass

    @abstractmethod
    def scan_stock(self, symbol: str, data: StockData, criteria: Optional[Dict] = None) -> ScreenerResult:
        """
        Perform screening analysis on a stock.

        Args:
            symbol: Stock symbol
            data: Pre-fetched stock data
            criteria: Optional screening criteria/parameters

        Returns:
            ScreenerResult with score, rating, and details
        """
        pass

    @abstractmethod
    def calculate_rating(self, score: float, details: Dict) -> str:
        """
        Convert numeric score to human-readable rating.

        Args:
            score: Numeric score (0-100)
            details: Analysis details that might affect rating

        Returns:
            Rating string like "Strong Buy", "Buy", "Watch", "Pass"
        """
        pass

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.screener_name})"

    def __repr__(self) -> str:
        """Developer representation."""
        return self.__str__()
