"""
Hybrid Fundamentals Service

Orchestrates multiple data sources for optimal performance:
1. yfinance (batch via yf.Tickers): ~35 fields in bulk
2. Technical calculator: ~12 fields from cached price data
3. finviz (unique fields only): ~15 fields that aren't available elsewhere

Target: Reduce full refresh from ~4 hours to 1-2 hours.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .bulk_data_fetcher import BulkDataFetcher
from .technical_calculator_service import TechnicalCalculatorService
from .finviz_service import FinvizService
from .price_cache_service import PriceCacheService
from .institutional_ownership_service import InstitutionalOwnershipService
from ..database import SessionLocal

logger = logging.getLogger(__name__)


class HybridFundamentalsService:
    """
    Hybrid fundamental data fetching for optimal performance.

    Data Sources:
    1. yfinance (batch via yf.Tickers): ~35 fields
       - Market data: market_cap, shares_outstanding, shares_float
       - Valuation: pe_ratio, forward_pe, peg_ratio, price_to_book, etc.
       - Profitability: profit_margin, operating_margin, roe, roa
       - Financial health: current_ratio, quick_ratio, debt_to_equity
       - Growth: revenue_growth, earnings_growth, eps_growth_qq, sales_growth_qq

    2. Technical Calculator (from cached price data): ~12 fields
       - Indicators: rsi_14, atr_14
       - SMA distances: sma_20, sma_50, sma_200
       - Performance: perf_week, perf_month, perf_quarter, etc.
       - Volatility: volatility_week, volatility_month
       - 52-week: week_52_high, week_52_low, distances

    3. finviz (unique fields only): ~15 fields
       - Short interest: short_float, short_ratio, short_interest
       - Transactions: insider_transactions, institutional_transactions
       - Forward estimates: eps_next_y, eps_next_5y, eps_next_q
       - Financial: lt_debt_to_equity, roic, price_to_cash, price_to_fcf
    """

    def __init__(
        self,
        include_finviz: bool = True,
        yfinance_batch_size: int = 50,
        yfinance_delay_per_ticker: float = 1.5,
        yfinance_delay_between_batches: float = 2.0,
        finviz_rate_limit: float = 0.5
    ):
        """
        Initialize HybridFundamentalsService.

        Args:
            include_finviz: Whether to fetch finviz-only fields (slower but more complete)
            yfinance_batch_size: Symbols per yfinance batch
            yfinance_delay_per_ticker: Seconds between individual yfinance ticker fetches (default 0.2)
            yfinance_delay_between_batches: Seconds between yfinance batches (default 2.0)
            finviz_rate_limit: Seconds between finviz API calls (default 0.5)
        """
        self.include_finviz = include_finviz
        self.yfinance_batch_size = yfinance_batch_size
        self.yfinance_delay_per_ticker = yfinance_delay_per_ticker
        self.yfinance_delay_between_batches = yfinance_delay_between_batches
        self.finviz_rate_limit = finviz_rate_limit

        self.bulk_fetcher = BulkDataFetcher()
        self.technical_calc = TechnicalCalculatorService()
        self.finviz_service = FinvizService()
        self.price_cache = PriceCacheService.get_instance()

    def fetch_fundamentals(
        self,
        symbol: str,
        include_technicals: bool = True,
        include_finviz: bool = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch fundamentals for a single symbol using hybrid approach.

        Args:
            symbol: Stock ticker symbol
            include_technicals: Whether to calculate technical indicators
            include_finviz: Whether to include finviz-only fields (default: use instance setting)

        Returns:
            Dict with merged fundamental data from all sources
        """
        if include_finviz is None:
            include_finviz = self.include_finviz

        result = {}

        # Phase 1: yfinance fundamentals
        yf_data = self.bulk_fetcher.fetch_batch_fundamentals(
            [symbol],
            batch_size=1,
            include_quarterly=True
        )
        if symbol in yf_data and not yf_data[symbol].get('has_error'):
            result.update(yf_data[symbol])

        # Phase 2: Technical calculations from price cache
        if include_technicals:
            price_data = self.price_cache.get_historical_data(symbol, period='2y')
            if price_data is not None:
                technicals = self.technical_calc.calculate_all(price_data)
                result.update(technicals)

        # Phase 3: finviz-only fields
        if include_finviz:
            finviz_data = self.finviz_service.get_finviz_only_fields(symbol)
            if finviz_data:
                result.update(finviz_data)

        # Add metadata
        result['symbol'] = symbol
        result['hybrid_fetch_timestamp'] = datetime.utcnow().isoformat()

        return result if result else None

    def fetch_fundamentals_batch(
        self,
        symbols: List[str],
        include_technicals: bool = True,
        include_finviz: bool = None,
        progress_callback=None
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals for multiple symbols using hybrid approach.

        This is the main method for bulk fundamental updates.

        Args:
            symbols: List of ticker symbols
            include_technicals: Whether to calculate technical indicators
            include_finviz: Whether to include finviz-only fields
            progress_callback: Optional callback for progress updates (called with current, total)

        Returns:
            Dict mapping symbols to their fundamental data
        """
        if not symbols:
            return {}

        if include_finviz is None:
            include_finviz = self.include_finviz

        total = len(symbols)
        logger.info(f"Starting hybrid fetch for {total} symbols")
        start_time = time.time()

        results = {symbol: {} for symbol in symbols}

        # ============================================================
        # Phase 1: Batch fetch yfinance fundamentals (~25 min for 7000)
        # ============================================================
        logger.info("Phase 1: Fetching yfinance fundamentals...")
        phase1_start = time.time()

        yf_data = self.bulk_fetcher.fetch_batch_fundamentals(
            symbols,
            batch_size=self.yfinance_batch_size,
            include_quarterly=True,
            delay_between_batches=self.yfinance_delay_between_batches,
            delay_per_ticker=self.yfinance_delay_per_ticker
        )

        for symbol in symbols:
            if symbol in yf_data and not yf_data[symbol].get('has_error'):
                results[symbol].update(yf_data[symbol])

        phase1_time = time.time() - phase1_start
        yf_success = len([s for s in symbols if s in yf_data and not yf_data.get(s, {}).get('has_error')])
        logger.info(f"Phase 1 complete: {yf_success}/{total} in {phase1_time:.1f}s")

        if progress_callback:
            progress_callback(total * 0.3, total)  # 30% after phase 1

        # ============================================================
        # Phase 2: Calculate technicals from price cache (~10 min)
        # ============================================================
        if include_technicals:
            logger.info("Phase 2: Calculating technical indicators...")
            phase2_start = time.time()

            # Bulk fetch price data from cache
            price_data_dict = self.price_cache.get_many(symbols, period='2y')

            # Calculate technicals for each symbol
            tech_success = 0
            for i, symbol in enumerate(symbols):
                price_data = price_data_dict.get(symbol)
                if price_data is not None and not price_data.empty:
                    technicals = self.technical_calc.calculate_all(price_data)
                    results[symbol].update(technicals)
                    tech_success += 1

                if i > 0 and i % 500 == 0:
                    logger.info(f"Phase 2 progress: {i}/{total}")

            phase2_time = time.time() - phase2_start
            logger.info(f"Phase 2 complete: {tech_success}/{total} in {phase2_time:.1f}s")

            if progress_callback:
                progress_callback(total * 0.5, total)  # 50% after phase 2

        # ============================================================
        # Phase 3: Fetch finviz-only fields (~1-1.5 hours)
        # ============================================================
        if include_finviz:
            logger.info("Phase 3: Fetching finviz-only fields...")
            phase3_start = time.time()

            finviz_success = 0
            for i, symbol in enumerate(symbols):
                finviz_data = self.finviz_service.get_finviz_only_fields(symbol)
                if finviz_data:
                    results[symbol].update(finviz_data)
                    finviz_success += 1

                if i > 0 and i % 50 == 0:
                    elapsed = time.time() - phase3_start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / rate if rate > 0 else 0
                    logger.info(
                        f"Phase 3 progress: {i}/{total} ({i/total*100:.1f}%), "
                        f"ETA: {eta/60:.1f} min"
                    )

                    if progress_callback:
                        # Phase 3 is 50% to 100% of progress
                        phase3_progress = 0.5 + (i / total) * 0.5
                        progress_callback(int(total * phase3_progress), total)

            phase3_time = time.time() - phase3_start
            logger.info(f"Phase 3 complete: {finviz_success}/{total} in {phase3_time:.1f}s")

        # Add metadata to all results
        for symbol in symbols:
            results[symbol]['symbol'] = symbol
            results[symbol]['hybrid_fetch_timestamp'] = datetime.utcnow().isoformat()

        total_time = time.time() - start_time
        logger.info(
            f"Hybrid fetch complete: {total} symbols in {total_time/60:.1f} minutes"
        )

        return results

    def fetch_fundamentals_yfinance_only(
        self,
        symbols: List[str],
        include_technicals: bool = True
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals using only yfinance + technical calculations.

        This is the fast path that skips finviz entirely.
        Use this when finviz-only fields aren't needed.

        Args:
            symbols: List of ticker symbols
            include_technicals: Whether to calculate technical indicators

        Returns:
            Dict mapping symbols to their fundamental data
        """
        return self.fetch_fundamentals_batch(
            symbols,
            include_technicals=include_technicals,
            include_finviz=False
        )

    def fetch_fundamentals_with_parallel_finviz(
        self,
        symbols: List[str],
        include_technicals: bool = True,
        finviz_workers: int = 2,
        progress_callback=None
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals with parallel finviz fetching for faster performance.

        WARNING: Using multiple workers for finviz may trigger rate limiting.
        Use with caution and monitor for errors.

        Args:
            symbols: List of ticker symbols
            include_technicals: Whether to calculate technical indicators
            finviz_workers: Number of parallel workers for finviz (default 2)
            progress_callback: Optional progress callback

        Returns:
            Dict mapping symbols to their fundamental data
        """
        if not symbols:
            return {}

        total = len(symbols)
        logger.info(f"Starting parallel hybrid fetch for {total} symbols")
        start_time = time.time()

        results = {symbol: {} for symbol in symbols}

        # Phase 1: yfinance (already batched)
        logger.info("Phase 1: Fetching yfinance fundamentals...")
        yf_data = self.bulk_fetcher.fetch_fundamentals_parallel(
            symbols,
            batch_size=self.yfinance_batch_size,
            max_workers=3,
            include_quarterly=True,
            delay_per_ticker=self.yfinance_delay_per_ticker
        )

        for symbol in symbols:
            if symbol in yf_data and not yf_data[symbol].get('has_error'):
                results[symbol].update(yf_data[symbol])

        if progress_callback:
            progress_callback(int(total * 0.25), total)

        # Phase 2: Technical calculations
        if include_technicals:
            logger.info("Phase 2: Calculating technical indicators...")
            price_data_dict = self.price_cache.get_many(symbols, period='2y')
            technicals = self.technical_calc.calculate_batch(price_data_dict)
            for symbol, tech_data in technicals.items():
                results[symbol].update(tech_data)

            if progress_callback:
                progress_callback(int(total * 0.4), total)

        # Phase 3: Parallel finviz fetching
        logger.info(f"Phase 3: Parallel finviz fetching ({finviz_workers} workers)...")

        # Split symbols into chunks for workers
        chunk_size = (len(symbols) + finviz_workers - 1) // finviz_workers
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

        def fetch_finviz_chunk(chunk_symbols: List[str]) -> Dict[str, Dict]:
            """Fetch finviz data for a chunk of symbols."""
            chunk_results = {}
            # Create a separate FinvizService instance for thread safety
            finviz = FinvizService()
            for symbol in chunk_symbols:
                data = finviz.get_finviz_only_fields(symbol)
                chunk_results[symbol] = data or {}
            return chunk_results

        with ThreadPoolExecutor(max_workers=finviz_workers) as executor:
            futures = {executor.submit(fetch_finviz_chunk, chunk): chunk for chunk in chunks}

            completed = 0
            for future in as_completed(futures):
                chunk_results = future.result()
                for symbol, data in chunk_results.items():
                    if data:
                        results[symbol].update(data)

                completed += len(futures[future])
                if progress_callback:
                    phase3_progress = 0.4 + (completed / total) * 0.6
                    progress_callback(int(total * phase3_progress), total)

        # Add metadata
        for symbol in symbols:
            results[symbol]['symbol'] = symbol
            results[symbol]['hybrid_fetch_timestamp'] = datetime.utcnow().isoformat()

        total_time = time.time() - start_time
        logger.info(f"Parallel hybrid fetch complete: {total} symbols in {total_time/60:.1f} minutes")

        return results

    def store_all_caches(
        self,
        results: Dict[str, Dict],
        fundamentals_cache
    ) -> Dict[str, int]:
        """
        Store hybrid results in fundamentals cache.

        Quarterly growth fields are now consolidated into fundamentals cache,
        so no separate quarterly cache storage is needed.

        Also updates institutional ownership history (SCD2) for tracking
        ownership changes over time.

        Args:
            results: Dict mapping symbols to their fundamental data
            fundamentals_cache: FundamentalsCacheService instance

        Returns:
            Dict with storage statistics
        """
        stats = {
            'fundamentals_stored': 0,
            'ownership_updated': 0,
            'failed': 0
        }

        for symbol, data in results.items():
            if not data or data.get('has_error'):
                stats['failed'] += 1
                continue

            try:
                # Store in fundamentals cache (includes quarterly growth fields)
                fundamentals_cache.store(symbol, data, data_source='hybrid')
                stats['fundamentals_stored'] += 1

            except Exception as e:
                logger.warning(f"Error storing {symbol}: {e}")
                stats['failed'] += 1

        # Bulk update institutional ownership history (SCD2)
        try:
            db = SessionLocal()
            ownership_service = InstitutionalOwnershipService(db)

            # Convert results dict to list for bulk_update
            fundamentals_list = [
                {**data, 'symbol': symbol}
                for symbol, data in results.items()
                if data and not data.get('has_error')
            ]

            ownership_updated = ownership_service.bulk_update(
                fundamentals_list,
                data_source='hybrid'
            )
            stats['ownership_updated'] = ownership_updated
            logger.info(f"Ownership history: {ownership_updated} records updated")

        except Exception as e:
            logger.warning(f"Error in bulk ownership update: {e}")
        finally:
            db.close()

        return stats


# Global instance with default settings
hybrid_fundamentals_service = HybridFundamentalsService()
