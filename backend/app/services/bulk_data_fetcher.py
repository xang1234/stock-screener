"""
Bulk Data Fetcher using yfinance.Tickers() for efficient batch operations.

Fetches multiple stocks in a single operation to reduce API overhead
and improve throughput for large-scale scanning operations.
"""
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class BulkDataFetcher:
    """
    Fetch data for multiple stocks efficiently using yfinance.Tickers().

    yfinance.Tickers() is more efficient than individual yf.Ticker() calls
    because it batches API requests internally, reducing overhead and
    potentially getting better rate treatment from Yahoo Finance.
    """

    def fetch_batch_data(
        self,
        symbols: List[str],
        period: str = '2y',
        include_fundamentals: bool = True,
        delay_per_ticker: float = 0.0
    ) -> Dict[str, Dict]:
        """
        Fetch data for multiple symbols in a single efficient operation.

        Args:
            symbols: List of ticker symbols to fetch
            period: Time period for historical data (default: 2y)
            include_fundamentals: Whether to include fundamental data
            delay_per_ticker: Seconds to wait between individual ticker.history()
                calls within the batch (default: 0.0, use settings.yfinance_per_ticker_delay
                for rate-limited contexts)

        Returns:
            Dict mapping symbols to their data:
            {
                'AAPL': {
                    'symbol': 'AAPL',
                    'price_data': pd.DataFrame,
                    'info': dict,
                    'fundamentals': dict,
                    'has_error': False
                },
                ...
            }
        """
        if not symbols:
            return {}

        logger.info(f"Bulk fetching data for {len(symbols)} symbols using yfinance.Tickers()")

        try:
            # Use Tickers for efficient multi-symbol fetching
            # This creates a single request for multiple stocks
            tickers = yf.Tickers(' '.join(symbols))

            results = {}
            for i, symbol in enumerate(symbols):
                try:
                    ticker = tickers.tickers[symbol]

                    # Fetch historical price data
                    hist = ticker.history(period=period)

                    # Optionally fetch fundamental info
                    info = None
                    fundamentals = None
                    if include_fundamentals:
                        try:
                            info = ticker.info
                            fundamentals = self._extract_fundamentals(info)
                        except Exception as e:
                            logger.warning(f"Could not fetch fundamentals for {symbol}: {e}")

                    # Store result
                    results[symbol] = {
                        'symbol': symbol,
                        'price_data': hist,
                        'info': info,
                        'fundamentals': fundamentals,
                        'has_error': False,
                        'error': None
                    }

                    logger.debug(f"✓ {symbol}: {len(hist)} price rows")

                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
                    results[symbol] = {
                        'symbol': symbol,
                        'price_data': None,
                        'info': None,
                        'fundamentals': None,
                        'has_error': True,
                        'error': str(e)
                    }

                # Per-symbol rate limiting within the batch
                if delay_per_ticker > 0 and i < len(symbols) - 1:
                    time.sleep(delay_per_ticker)

            logger.info(
                f"Bulk fetch completed: {len([r for r in results.values() if not r['has_error']])} "
                f"successful, {len([r for r in results.values() if r['has_error']])} failed"
            )
            return results

        except Exception as e:
            logger.error(f"Bulk fetch failed: {e}", exc_info=True)
            # Return error results for all symbols
            return {
                symbol: {
                    'symbol': symbol,
                    'price_data': None,
                    'info': None,
                    'fundamentals': None,
                    'has_error': True,
                    'error': f"Bulk fetch error: {str(e)}"
                }
                for symbol in symbols
            }

    def _extract_fundamentals(self, info: Dict) -> Dict:
        """
        Extract fundamental data from ticker info.

        Args:
            info: Ticker info dictionary from yfinance

        Returns:
            Dict with cleaned fundamental metrics
        """
        if not info:
            return {}

        fundamentals = {
            # Market data
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'shares_float': info.get('floatShares'),

            # Valuation metrics
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'ev_sales': info.get('enterpriseToRevenue'),
            'ev_ebitda': info.get('enterpriseToEbitda'),

            # Profitability metrics
            'eps_current': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            'roe': self._convert_to_percentage(info.get('returnOnEquity')),
            'roa': self._convert_to_percentage(info.get('returnOnAssets')),
            'profit_margin': self._convert_to_percentage(info.get('profitMargins')),
            'operating_margin': self._convert_to_percentage(info.get('operatingMargins')),
            'gross_margin': self._convert_to_percentage(info.get('grossMargins')),

            # Growth metrics (as percentages)
            'revenue_growth': self._convert_to_percentage(info.get('revenueGrowth')),
            'earnings_growth': self._convert_to_percentage(info.get('earningsGrowth')),
            'earnings_quarterly_growth': self._convert_to_percentage(info.get('earningsQuarterlyGrowth')),

            # Revenue
            'revenue_current': info.get('totalRevenue'),
            'net_income': info.get('netIncomeToCommon'),

            # Dividend metrics
            'dividend_ttm': info.get('dividendRate'),
            'dividend_yield': self._convert_to_percentage(info.get('dividendYield')),
            'payout_ratio': self._convert_to_percentage(info.get('payoutRatio')),

            # Financial health
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'debt_to_equity': info.get('debtToEquity'),
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'free_cashflow': info.get('freeCashflow'),
            'operating_cashflow': info.get('operatingCashflow'),

            # Ownership (convert to percentage)
            'institutional_ownership': self._convert_to_percentage(info.get('heldPercentInstitutions')),
            'insider_ownership': self._convert_to_percentage(info.get('heldPercentInsiders')),

            # Stock info
            'beta': info.get('beta'),
            'week_52_high': info.get('fiftyTwoWeekHigh'),
            'week_52_low': info.get('fiftyTwoWeekLow'),
            'avg_volume': info.get('averageVolume'),
            'avg_volume_10d': info.get('averageVolume10days'),
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),

            # Company info
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'employees': info.get('fullTimeEmployees'),
            'country': info.get('country'),

            # Analyst info
            'target_price': info.get('targetMeanPrice'),
            'recommendation': info.get('recommendationKey'),

            # Company description
            'description_yfinance': info.get('longBusinessSummary'),

            # Mark data source
            'data_source': 'yfinance',
            'data_source_timestamp': datetime.utcnow().isoformat(),
        }

        # Remove None values to save space
        return {k: v for k, v in fundamentals.items() if v is not None}

    def _convert_to_percentage(self, value: Optional[float]) -> Optional[float]:
        """Convert decimal to percentage (0.15 -> 15.0)."""
        if value is None:
            return None
        return round(value * 100, 2)

    def _extract_quarterly_growth(self, ticker) -> Dict[str, Optional[float]]:
        """
        Extract quarterly growth metrics from ticker's income statement.

        Calculates EPS Q/Q and Sales Q/Q growth from quarterly data.

        Args:
            ticker: yfinance Ticker object

        Returns:
            Dict with quarterly growth metrics
        """
        result = {
            'eps_growth_qq': None,
            'sales_growth_qq': None,
            'eps_growth_yy': None,
            'sales_growth_yy': None,
            'recent_quarter_date': None,
            'previous_quarter_date': None,
        }

        try:
            quarterly_income = ticker.quarterly_income_stmt

            if quarterly_income is None or quarterly_income.shape[1] < 2:
                return result

            # Columns are most recent first
            recent_col = quarterly_income.columns[0]
            prev_col = quarterly_income.columns[1]

            result['recent_quarter_date'] = str(recent_col)
            result['previous_quarter_date'] = str(prev_col)

            # Find EPS row (Diluted EPS preferred)
            eps_row = None
            for idx in quarterly_income.index:
                idx_str = str(idx).lower()
                if 'diluted eps' in idx_str or 'dilutedeps' in idx_str:
                    eps_row = idx
                    break
            if eps_row is None:
                for idx in quarterly_income.index:
                    idx_str = str(idx).lower()
                    if 'basic eps' in idx_str or 'basiceps' in idx_str:
                        eps_row = idx
                        break

            # Calculate EPS Q/Q growth
            if eps_row is not None:
                recent_eps = quarterly_income.loc[eps_row, recent_col]
                prev_eps = quarterly_income.loc[eps_row, prev_col]

                if prev_eps != 0 and not pd.isna(prev_eps) and not pd.isna(recent_eps):
                    if abs(prev_eps) > 0.05:  # Avoid tiny denominators
                        eps_growth = ((recent_eps - prev_eps) / abs(prev_eps)) * 100
                        result['eps_growth_qq'] = round(float(eps_growth), 2)

            # Find Revenue row
            revenue_row = None
            for idx in quarterly_income.index:
                idx_str = str(idx).lower()
                if 'total revenue' in idx_str or 'totalrevenue' in idx_str:
                    revenue_row = idx
                    break
            if revenue_row is None:
                for idx in quarterly_income.index:
                    idx_str = str(idx).lower()
                    if 'revenue' in idx_str:
                        revenue_row = idx
                        break

            # Calculate Sales Q/Q growth
            if revenue_row is not None:
                recent_rev = quarterly_income.loc[revenue_row, recent_col]
                prev_rev = quarterly_income.loc[revenue_row, prev_col]

                if prev_rev != 0 and not pd.isna(prev_rev) and not pd.isna(recent_rev):
                    sales_growth = ((recent_rev - prev_rev) / abs(prev_rev)) * 100
                    result['sales_growth_qq'] = round(float(sales_growth), 2)

            # Y/Y growth (compare to same quarter last year if available)
            if quarterly_income.shape[1] >= 5:
                year_ago_col = quarterly_income.columns[4]

                if eps_row is not None:
                    recent_eps = quarterly_income.loc[eps_row, recent_col]
                    year_ago_eps = quarterly_income.loc[eps_row, year_ago_col]
                    if year_ago_eps != 0 and not pd.isna(year_ago_eps) and not pd.isna(recent_eps):
                        if abs(year_ago_eps) > 0.05:
                            yy_growth = ((recent_eps - year_ago_eps) / abs(year_ago_eps)) * 100
                            result['eps_growth_yy'] = round(float(yy_growth), 2)

                if revenue_row is not None:
                    recent_rev = quarterly_income.loc[revenue_row, recent_col]
                    year_ago_rev = quarterly_income.loc[revenue_row, year_ago_col]
                    if year_ago_rev != 0 and not pd.isna(year_ago_rev) and not pd.isna(recent_rev):
                        yy_growth = ((recent_rev - year_ago_rev) / abs(year_ago_rev)) * 100
                        result['sales_growth_yy'] = round(float(yy_growth), 2)

        except Exception as e:
            logger.debug(f"Error extracting quarterly growth: {e}")

        return result

    def _extract_eps_rating_data(self, ticker) -> Dict[str, Optional[float]]:
        """
        Extract EPS rating data from ticker's annual and quarterly income statements.

        Uses EPSRatingService to calculate:
        - 5-year CAGR from annual income statement
        - Q1 and Q2 YoY growth from quarterly income statement
        - Raw composite score

        Args:
            ticker: yfinance Ticker object

        Returns:
            Dict with EPS rating components:
            {
                'eps_5yr_cagr': float or None,
                'eps_q1_yoy': float or None,
                'eps_q2_yoy': float or None,
                'eps_raw_score': float or None,
                'eps_years_available': int
            }
        """
        from .eps_rating_service import eps_rating_service

        result = {
            'eps_5yr_cagr': None,
            'eps_q1_yoy': None,
            'eps_q2_yoy': None,
            'eps_raw_score': None,
            'eps_years_available': 0
        }

        try:
            # Get annual income statement (for 5-year CAGR)
            annual_income = ticker.income_stmt

            # Get quarterly income statement (for YoY comparisons)
            quarterly_income = ticker.quarterly_income_stmt

            # Use EPS rating service to calculate all components
            eps_data = eps_rating_service.calculate_eps_rating_data(
                annual_income,
                quarterly_income
            )

            result.update(eps_data)
            logger.debug(f"Extracted EPS rating data: CAGR={result['eps_5yr_cagr']}, Q1={result['eps_q1_yoy']}, Q2={result['eps_q2_yoy']}")

        except Exception as e:
            logger.debug(f"Error extracting EPS rating data: {e}")

        return result

    def fetch_batch_prices(
        self,
        symbols: List[str],
        period: str = '2y',
    ) -> Dict[str, Dict]:
        """
        Fetch price data for multiple symbols using yf.download() (single HTTP request).

        This is significantly faster than fetch_batch_data() which calls ticker.history()
        per symbol with per-ticker delays. yf.download() batches all symbols into one
        request and returns data for all tickers at once.

        Args:
            symbols: List of ticker symbols to fetch
            period: Time period for historical data (default: 2y)

        Returns:
            Dict mapping symbols to their data:
            {
                'AAPL': {
                    'symbol': 'AAPL',
                    'price_data': pd.DataFrame,
                    'info': None,
                    'fundamentals': None,
                    'has_error': False
                },
                ...
            }
        """
        if not symbols:
            return {}

        logger.info(f"Batch fetching prices for {len(symbols)} symbols using yf.download()")

        results = {}

        try:
            # yf.download with group_by='ticker' returns MultiIndex columns: (ticker, OHLCV)
            # threads=False avoids threading issues inside Celery workers
            # progress=False suppresses the tqdm progress bar
            raw = yf.download(
                tickers=symbols,
                period=period,
                group_by='ticker',
                threads=False,
                progress=False,
            )

            if raw is None or raw.empty:
                logger.warning("yf.download returned empty DataFrame")
                for symbol in symbols:
                    results[symbol] = {
                        'symbol': symbol, 'price_data': None, 'info': None,
                        'fundamentals': None, 'has_error': True,
                        'error': 'yf.download returned empty'
                    }
                return results

            # Single symbol: yf.download returns flat columns (Open, High, etc.)
            # Multi-symbol: returns MultiIndex columns (AAPL/Open, AAPL/High, etc.)
            if len(symbols) == 1:
                symbol = symbols[0]
                df = raw.copy()
                if df is not None and not df.empty:
                    results[symbol] = {
                        'symbol': symbol, 'price_data': df, 'info': None,
                        'fundamentals': None, 'has_error': False, 'error': None
                    }
                else:
                    results[symbol] = {
                        'symbol': symbol, 'price_data': None, 'info': None,
                        'fundamentals': None, 'has_error': True, 'error': 'No data returned'
                    }
            else:
                # Multi-symbol: split by ticker
                for symbol in symbols:
                    try:
                        if symbol in raw.columns.get_level_values(0):
                            df = raw[symbol].copy()
                            # Drop rows where all values are NaN (symbol had no data for that date)
                            df = df.dropna(how='all')
                            if not df.empty:
                                results[symbol] = {
                                    'symbol': symbol, 'price_data': df, 'info': None,
                                    'fundamentals': None, 'has_error': False, 'error': None
                                }
                            else:
                                results[symbol] = {
                                    'symbol': symbol, 'price_data': None, 'info': None,
                                    'fundamentals': None, 'has_error': True,
                                    'error': 'No data after filtering NaN rows'
                                }
                        else:
                            results[symbol] = {
                                'symbol': symbol, 'price_data': None, 'info': None,
                                'fundamentals': None, 'has_error': True,
                                'error': 'Symbol not in download results'
                            }
                    except Exception as e:
                        logger.warning(f"Error extracting {symbol} from download: {e}")
                        results[symbol] = {
                            'symbol': symbol, 'price_data': None, 'info': None,
                            'fundamentals': None, 'has_error': True, 'error': str(e)
                        }

            success = len([r for r in results.values() if not r['has_error']])
            failed = len(results) - success
            logger.info(f"Batch price download: {success} successful, {failed} failed")

            # Add missing symbols as errors
            for symbol in symbols:
                if symbol not in results:
                    results[symbol] = {
                        'symbol': symbol, 'price_data': None, 'info': None,
                        'fundamentals': None, 'has_error': True,
                        'error': 'Missing from results'
                    }

            return results

        except Exception as e:
            logger.error(f"yf.download batch failed: {e}", exc_info=True)
            return {
                symbol: {
                    'symbol': symbol, 'price_data': None, 'info': None,
                    'fundamentals': None, 'has_error': True,
                    'error': f"Batch download error: {str(e)}"
                }
                for symbol in symbols
            }

    def fetch_batch_with_cache_check(
        self,
        symbols: List[str],
        cached_data: Dict[str, any],
        period: str = '2y'
    ) -> Dict[str, Dict]:
        """
        Fetch data for symbols, using cached data where available.

        Args:
            symbols: List of symbols to fetch
            cached_data: Dict of already cached symbol data
            period: Time period for historical data

        Returns:
            Combined dict of cached + freshly fetched data
        """
        # Identify which symbols need fetching
        cache_misses = [s for s in symbols if s not in cached_data or cached_data[s] is None]

        if not cache_misses:
            logger.info(f"All {len(symbols)} symbols served from cache")
            return cached_data

        logger.info(
            f"Cache: {len(symbols) - len(cache_misses)} hits, "
            f"{len(cache_misses)} misses - fetching {len(cache_misses)} symbols"
        )

        # Fetch missing symbols in bulk
        fresh_data = self.fetch_batch_data(cache_misses, period=period)

        # Combine cached and fresh data
        combined = {**cached_data}
        combined.update(fresh_data)

        return combined

    def fetch_batch_fundamentals(
        self,
        symbols: List[str],
        batch_size: int = 50,
        include_quarterly: bool = True,
        delay_between_batches: float = 2.0,
        delay_per_ticker: float = 1.5
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals for multiple symbols efficiently.

        This method is optimized for fundamental data fetching only (no price history),
        making it much faster for bulk fundamental updates.

        Args:
            symbols: List of ticker symbols to fetch
            batch_size: Number of symbols per batch (default 50)
            include_quarterly: Whether to fetch quarterly growth data (slower but more complete)
            delay_between_batches: Seconds to wait between batches (default 2.0)
            delay_per_ticker: Seconds to wait between individual ticker info fetches (default 0.2)

        Returns:
            Dict mapping symbols to their fundamental data
        """
        if not symbols:
            return {}

        logger.info(f"Fetching fundamentals for {len(symbols)} symbols (batch_size={batch_size}, delay={delay_per_ticker}s/ticker)")

        all_results = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        consecutive_backoffs = 0  # Track consecutive rate-limited batches

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(symbols))
            batch_symbols = symbols[start_idx:end_idx]

            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_symbols)} symbols)")

            batch_rate_limit_failures = 0

            try:
                # Use yf.Tickers for efficient batch fetching
                tickers = yf.Tickers(' '.join(batch_symbols))

                for i, symbol in enumerate(batch_symbols):
                    try:
                        ticker = tickers.tickers[symbol]

                        # Get info (fundamentals)
                        info = ticker.info
                        fundamentals = self._extract_fundamentals(info)

                        # Optionally get quarterly growth
                        if include_quarterly:
                            quarterly = self._extract_quarterly_growth(ticker)
                            fundamentals.update(quarterly)

                            # Also extract EPS rating data
                            eps_rating_data = self._extract_eps_rating_data(ticker)
                            fundamentals.update(eps_rating_data)

                        all_results[symbol] = fundamentals

                    except Exception as e:
                        error_str = str(e).lower()
                        logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
                        all_results[symbol] = {'has_error': True, 'error': str(e)}
                        if any(ind in error_str for ind in ("rate", "429", "too many", "limit", "throttl")):
                            batch_rate_limit_failures += 1

                    # Rate limit between individual ticker fetches within batch
                    if i < len(batch_symbols) - 1 and delay_per_ticker > 0:
                        from .rate_limiter import rate_limiter
                        rate_limiter.wait("yfinance", min_interval_s=delay_per_ticker)

            except Exception as e:
                logger.error(f"Batch error: {e}")
                for symbol in batch_symbols:
                    all_results[symbol] = {'has_error': True, 'error': str(e)}

            # Batch-level backoff: if >50% of batch hit rate limits, back off
            batch_failure_rate = batch_rate_limit_failures / len(batch_symbols) if batch_symbols else 0
            if batch_failure_rate > 0.5:
                consecutive_backoffs += 1
                backoff_time = min(60 * (2 ** (consecutive_backoffs - 1)), 480)  # 60, 120, 240, 480 max
                logger.warning(
                    f"Rate limited: {batch_rate_limit_failures}/{len(batch_symbols)} symbols in batch {batch_num + 1}. "
                    f"Backing off {backoff_time}s before next batch."
                )
                time.sleep(backoff_time)
            else:
                consecutive_backoffs = 0  # Reset on successful batch
                # Normal rate limit between batches
                if batch_num < total_batches - 1:
                    from .rate_limiter import rate_limiter
                    rate_limiter.wait("yfinance:batch", min_interval_s=delay_between_batches)

        success_count = len([r for r in all_results.values() if not r.get('has_error', False)])
        logger.info(f"Batch fundamentals complete: {success_count}/{len(symbols)} successful")

        return all_results

    def fetch_fundamentals_parallel(
        self,
        symbols: List[str],
        batch_size: int = 50,
        max_workers: int = 3,
        include_quarterly: bool = True,
        delay_per_ticker: float = 1.5
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals using parallel batch processing.

        Splits symbols into batches and processes batches in parallel
        using ThreadPoolExecutor for faster throughput.

        Args:
            symbols: List of ticker symbols
            batch_size: Symbols per batch
            max_workers: Number of parallel workers
            include_quarterly: Whether to include quarterly growth data
            delay_per_ticker: Seconds between individual ticker fetches (default 0.2)

        Returns:
            Dict mapping symbols to fundamental data
        """
        if not symbols:
            return {}

        # Split into batches
        batches = [
            symbols[i:i + batch_size]
            for i in range(0, len(symbols), batch_size)
        ]

        logger.info(f"Parallel fetch: {len(symbols)} symbols in {len(batches)} batches with {max_workers} workers")

        all_results = {}

        def fetch_batch(batch_symbols: List[str]) -> Dict[str, Dict]:
            """Fetch a single batch."""
            batch_results = {}
            try:
                tickers = yf.Tickers(' '.join(batch_symbols))

                for i, symbol in enumerate(batch_symbols):
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info
                        fundamentals = self._extract_fundamentals(info)

                        if include_quarterly:
                            quarterly = self._extract_quarterly_growth(ticker)
                            fundamentals.update(quarterly)

                            # Also extract EPS rating data
                            eps_rating_data = self._extract_eps_rating_data(ticker)
                            fundamentals.update(eps_rating_data)

                        batch_results[symbol] = fundamentals

                    except Exception as e:
                        logger.debug(f"Error for {symbol}: {e}")
                        batch_results[symbol] = {'has_error': True}

                    # Rate limit between individual ticker fetches
                    if i < len(batch_symbols) - 1 and delay_per_ticker > 0:
                        from .rate_limiter import rate_limiter
                        rate_limiter.wait("yfinance", min_interval_s=delay_per_ticker)

            except Exception as e:
                logger.warning(f"Batch error: {e}")
                for s in batch_symbols:
                    batch_results[s] = {'has_error': True}

            return batch_results

        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_batch, batch): batch for batch in batches}

            for future in as_completed(futures):
                batch_result = future.result()
                all_results.update(batch_result)

        success_count = len([r for r in all_results.values() if not r.get('has_error', False)])
        logger.info(f"Parallel fetch complete: {success_count}/{len(symbols)} successful")

        return all_results
