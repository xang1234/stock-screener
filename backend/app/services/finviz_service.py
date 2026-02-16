"""
finvizfinance Service

Wrapper for finvizfinance library with rate limiting and error handling.
"""
import logging
import time
from typing import Dict, Optional
from finvizfinance.quote import finvizfinance

from .finviz_parser import FinvizParser
from .finviz_validator import FinvizValidator

logger = logging.getLogger(__name__)


class FinvizService:
    """Service for fetching and parsing finvizfinance data"""

    def __init__(self):
        """Initialize FinvizService."""
        self.parser = FinvizParser()
        self.validator = FinvizValidator()

    def _rate_limited_call(self, func, *args, **kwargs):
        """
        Execute a function with rate limiting via Redis-backed distributed limiter.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result from function
        """
        from .rate_limiter import rate_limiter
        from ..config import settings
        rate_limiter.wait("finviz", min_interval_s=settings.finviz_rate_limit_interval)
        return func(*args, **kwargs)

    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data for a symbol from finvizfinance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with fundamental metrics in yfinance-compatible format,
            or None if error occurs
        """
        try:
            logger.info(f"Fetching fundamentals for {symbol} from finvizfinance")
            start_time = time.time()

            # Fetch data with rate limiting
            def fetch():
                stock = finvizfinance(symbol, verbose=0)
                if not stock.flag:  # Stock not found
                    logger.warning(f"Stock {symbol} not found in finvizfinance")
                    return None
                fundament = stock.ticker_fundament(raw=True)
                # Also fetch description
                try:
                    description = stock.ticker_description()
                except Exception:
                    description = None
                return (fundament, description)

            result = self._rate_limited_call(fetch)

            if not result:
                return None

            finviz_data, description = result

            # Parse to normalized format
            normalized_data = self.parser.normalize_fundamentals(finviz_data)

            # Add description if available
            if description:
                normalized_data['description_finviz'] = description

            # Also include growth metrics (EPS Q/Q, Sales Q/Q, etc.)
            growth_data = self.parser.normalize_quarterly_growth(finviz_data)
            if growth_data:
                # Merge growth data into fundamentals (excluding internal fields)
                for key, value in growth_data.items():
                    if not key.startswith('_') and value is not None:
                        normalized_data[key] = value

            elapsed = time.time() - start_time
            logger.info(
                f"Fetched fundamentals for {symbol} from finvizfinance: "
                f"{len(normalized_data)} fields in {elapsed:.2f}s"
            )

            return normalized_data

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol} from finvizfinance: {e}")
            return None

    def get_quarterly_growth(self, symbol: str) -> Optional[Dict]:
        """
        Fetch quarterly and yearly growth metrics for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with growth metrics (eps_growth_qq, sales_growth_qq, etc.),
            or None if error occurs
        """
        try:
            logger.info(f"Fetching quarterly growth for {symbol} from finvizfinance")
            start_time = time.time()

            # Fetch data with rate limiting
            def fetch():
                stock = finvizfinance(symbol, verbose=0)
                if not stock.flag:
                    logger.warning(f"Stock {symbol} not found in finvizfinance")
                    return None
                return stock.ticker_fundament(raw=True)

            finviz_data = self._rate_limited_call(fetch)

            if not finviz_data:
                return None

            # Parse growth metrics
            growth_data = self.parser.normalize_quarterly_growth(finviz_data)

            elapsed = time.time() - start_time
            logger.info(
                f"Fetched quarterly growth for {symbol} from finvizfinance: "
                f"{len(growth_data)} fields in {elapsed:.2f}s"
            )

            return growth_data

        except Exception as e:
            logger.error(f"Error fetching quarterly growth for {symbol} from finvizfinance: {e}")
            return None

    def get_combined_data(self, symbol: str, validate: bool = True) -> Optional[Dict]:
        """
        Fetch both fundamentals and growth metrics in a single call.

        This is more efficient than calling get_fundamentals and get_quarterly_growth
        separately, as it only makes one API call to finvizfinance.

        Args:
            symbol: Stock ticker symbol
            validate: If True, validate data quality before returning

        Returns:
            Dict with keys 'fundamentals' and 'growth', or None if error
        """
        try:
            logger.info(f"Fetching combined data for {symbol} from finvizfinance")
            start_time = time.time()

            # Fetch data with rate limiting
            def fetch():
                stock = finvizfinance(symbol, verbose=0)
                if not stock.flag:
                    logger.warning(f"Stock {symbol} not found in finvizfinance")
                    return None
                return stock.ticker_fundament(raw=True)

            finviz_data = self._rate_limited_call(fetch)

            if not finviz_data:
                return None

            # Parse both fundamentals and growth from same data
            fundamentals = self.parser.normalize_fundamentals(finviz_data)
            growth = self.parser.normalize_quarterly_growth(finviz_data)

            # Validate if requested (range checks produce warnings, not blocking errors)
            if validate:
                is_valid, report = self.validator.validate_all(fundamentals, growth, strict=True)

                if not is_valid:
                    # Log warning but still use the data
                    logger.warning(
                        f"Range validation warnings for {symbol}: {report['errors']}"
                    )
                else:
                    logger.debug(
                        f"Validation passed for {symbol}: "
                        f"fund_completeness={report['completeness']['fundamentals']:.1f}%, "
                        f"growth_completeness={report['completeness']['growth']:.1f}%"
                    )

            elapsed = time.time() - start_time
            logger.info(
                f"Fetched combined data for {symbol}: "
                f"{len(fundamentals)} fundamental fields, "
                f"{len(growth)} growth fields in {elapsed:.2f}s"
            )

            return {
                'fundamentals': fundamentals,
                'growth': growth,
                'data_source': 'finviz',
            }

        except Exception as e:
            logger.error(f"Error fetching combined data for {symbol} from finvizfinance: {e}")
            return None

    # Fields that are ONLY available from finviz (not in yfinance)
    FINVIZ_ONLY_FIELDS = {
        # Short interest data - not in yfinance
        'Short Float': 'short_float',
        'Short Ratio': 'short_ratio',
        'Short Interest': 'short_interest',

        # Insider transactions - yfinance only has ownership %, not transactions
        'Insider Trans': 'insider_transactions',

        # Institutional transactions - yfinance only has ownership %, not transactions
        'Inst Trans': 'institutional_transactions',

        # Forward estimates - yfinance has some, but finviz has more complete data
        'EPS next Y': 'eps_next_y',
        'EPS next 5Y': 'eps_next_5y',
        'EPS next Q': 'eps_next_q',

        # Financial health - not all in yfinance
        'LT Debt/Eq': 'lt_debt_to_equity',
        'ROIC': 'roic',
        'P/C': 'price_to_cash',
        'P/FCF': 'price_to_fcf',

        # Analyst recommendation (more detailed than yfinance)
        'Recom': 'recommendation',

        # Sales growth metrics - finviz has these directly
        'Sales past 5Y': 'sales_past_5y',
    }

    def get_finviz_only_fields(self, symbol: str) -> Optional[Dict]:
        """
        Fetch ONLY the fields that are unique to finviz (not available in yfinance).

        This is used in the hybrid approach where yfinance provides most data
        and finviz supplements with unique fields like short interest.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with finviz-only fields, or None if error
        """
        try:
            logger.debug(f"Fetching finviz-only fields for {symbol}")
            start_time = time.time()

            # Fetch data with rate limiting
            def fetch():
                stock = finvizfinance(symbol, verbose=0)
                if not stock.flag:
                    logger.debug(f"Stock {symbol} not found in finvizfinance")
                    return None, None
                fundament = stock.ticker_fundament(raw=True)
                # Also fetch company description
                try:
                    description = stock.ticker_description()
                except Exception as e:
                    logger.debug(f"Error fetching description for {symbol}: {e}")
                    description = None
                return fundament, description

            finviz_data, description = self._rate_limited_call(fetch)

            if not finviz_data:
                return None

            # Extract only the unique fields
            result = {}

            # Add company description if available
            if description:
                result['description_finviz'] = description
            for finviz_key, our_key in self.FINVIZ_ONLY_FIELDS.items():
                raw_value = finviz_data.get(finviz_key)

                if raw_value is None or raw_value == '-':
                    continue

                # Parse value based on field type
                parsed_value = None

                # Percentages
                if finviz_key in ['Short Float', 'Insider Trans', 'Inst Trans',
                                  'EPS next Y', 'EPS next 5Y', 'Sales past 5Y']:
                    parsed_value = self.parser.parse_percentage(raw_value)

                # Numbers with suffixes
                elif finviz_key in ['Short Interest']:
                    parsed_value = self.parser.parse_number_with_suffix(raw_value)

                # Simple ratios
                elif finviz_key in ['Short Ratio', 'LT Debt/Eq', 'ROIC', 'P/C',
                                    'P/FCF', 'Recom', 'EPS next Q']:
                    parsed_value = self.parser.parse_ratio(raw_value)

                if parsed_value is not None:
                    result[our_key] = parsed_value

            # Mark data source
            result['finviz_data_source'] = 'finviz'

            elapsed = time.time() - start_time
            logger.debug(
                f"Fetched {len(result)} finviz-only fields for {symbol} in {elapsed:.2f}s"
            )

            return result

        except Exception as e:
            logger.warning(f"Error fetching finviz-only fields for {symbol}: {e}")
            return None

    def get_finviz_only_fields_batch(
        self,
        symbols: list,
        max_workers: int = 1,
    ) -> Dict[str, Optional[Dict]]:
        """
        Fetch finviz-only fields for multiple symbols.

        Note: finvizfinance doesn't support batch fetching, so this is sequential
        with rate limiting. However, it only fetches ~15 fields per stock.

        Args:
            symbols: List of ticker symbols
            max_workers: Number of concurrent workers (default 1 for safety)

        Returns:
            Dict mapping symbols to their finviz-only data
        """
        results = {}
        total = len(symbols)

        logger.info(f"Fetching finviz-only fields for {total} symbols")

        for i, symbol in enumerate(symbols):
            if i > 0 and i % 50 == 0:
                logger.info(f"Progress: {i}/{total} symbols ({i/total*100:.1f}%)")

            result = self.get_finviz_only_fields(symbol)
            results[symbol] = result

        success_count = len([r for r in results.values() if r is not None])
        logger.info(f"Finviz-only batch complete: {success_count}/{total} successful")

        return results


# Global instance
finviz_service = FinvizService()
