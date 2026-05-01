"""
Hybrid Fundamentals Service

Orchestrates multiple data sources for optimal performance:
1. yfinance (batch via yf.Tickers): ~35 fields in bulk
2. Technical calculator: ~12 fields from cached price data
3. finviz (unique fields only): ~15 fields that aren't available elsewhere

Target: Reduce full refresh from ~4 hours to 1-2 hours.
"""
import logging
from collections.abc import Callable
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from sqlalchemy.orm import Session

from .bulk_data_fetcher import BulkDataFetcher
from .technical_calculator_service import TechnicalCalculatorService
from .finviz_service import FinvizService
from .price_cache_service import PriceCacheService
from .institutional_ownership_service import InstitutionalOwnershipService
from . import provider_routing_policy as routing_policy

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
        finviz_rate_limit: float = 0.5,
        price_cache: PriceCacheService | None = None,
        finviz_service: FinvizService | None = None,
        data_source_service: Any | None = None,
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

        self.technical_calc = TechnicalCalculatorService()
        resolved_rate_limiter = getattr(finviz_service, "_rate_limiter", None)
        if finviz_service is None:
            from app.services.rate_limiter import RedisRateLimiter

            resolved_rate_limiter = RedisRateLimiter()
            finviz_service = FinvizService(rate_limiter=resolved_rate_limiter)
        if price_cache is None:
            from app.database import SessionLocal
            from app.services.redis_pool import get_redis_client

            price_cache = PriceCacheService(
                redis_client=get_redis_client(),
                session_factory=SessionLocal,
            )
        if resolved_rate_limiter is None:
            resolved_rate_limiter = getattr(finviz_service, "_rate_limiter", None)
        if resolved_rate_limiter is None:
            from app.services.rate_limiter import RedisRateLimiter

            resolved_rate_limiter = RedisRateLimiter()
        self._finviz_rate_limiter = resolved_rate_limiter
        self.bulk_fetcher = BulkDataFetcher(rate_limiter=resolved_rate_limiter)

        self.finviz_service = finviz_service
        self.price_cache = price_cache
        self._data_source_service = data_source_service

    def _market_for_symbol(
        self,
        symbol: str,
        market_by_symbol: Optional[Dict[str, str]] = None,
    ) -> str:
        explicit = (market_by_symbol or {}).get(symbol)
        if explicit:
            return str(explicit).strip().upper()
        normalized = str(symbol or "").strip().upper()
        if normalized.endswith((".SS", ".SZ", ".BJ")):
            return routing_policy.MARKET_CN
        return routing_policy.MARKET_US

    def _is_cn_symbol(
        self,
        symbol: str,
        market_by_symbol: Optional[Dict[str, str]] = None,
    ) -> bool:
        return self._market_for_symbol(symbol, market_by_symbol) == routing_policy.MARKET_CN

    def _get_data_source_service(self):
        if self._data_source_service is None:
            from app.wiring.bootstrap import get_data_source_service

            self._data_source_service = get_data_source_service()
        return self._data_source_service

    def _fetch_cn_fundamentals_payload(self, symbol: str) -> Dict[str, Any]:
        """Fetch CN fundamentals through AKShare/BaoStock-aware routing."""
        data_source = self._get_data_source_service()
        combined = data_source.get_combined_data(symbol, market=routing_policy.MARKET_CN)
        if not combined:
            return {}
        merged: Dict[str, Any] = {}
        fundamentals = combined.get("fundamentals") or {}
        growth = combined.get("growth") or {}
        if isinstance(fundamentals, dict):
            merged.update(fundamentals)
        if isinstance(growth, dict):
            for key, value in growth.items():
                if value is not None and key not in merged:
                    merged[key] = value
        return merged

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

        # Phase 1: provider fundamentals. CN symbols must use the
        # AKShare/BaoStock-aware service because Yahoo is not the primary
        # source and Beijing yfinance support is intentionally disabled.
        if self._is_cn_symbol(symbol):
            result.update(self._fetch_cn_fundamentals_payload(symbol))
        else:
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
        progress_callback=None,
        market_by_symbol: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals for multiple symbols using hybrid approach.

        This is the main method for bulk fundamental updates.

        Args:
            symbols: List of ticker symbols
            include_technicals: Whether to calculate technical indicators
            include_finviz: Whether to include finviz-only fields
            progress_callback: Optional callback for progress updates (called with current, total)
            market_by_symbol: Optional mapping ``{symbol: market_code}``.
                When provided, Phase 3 (finviz) skips symbols whose market
                is not covered by finviz per the routing policy (e.g.
                HK/JP/TW). ``None`` or missing keys default to US behaviour,
                preserving legacy callers.

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

        cn_symbols = [s for s in symbols if self._is_cn_symbol(s, market_by_symbol)]
        cn_symbol_set = set(cn_symbols)
        yfinance_symbols = [s for s in symbols if s not in cn_symbol_set]

        if cn_symbols:
            logger.info(
                "Phase 1a: Fetching CN fundamentals via AKShare/BaoStock-aware routing for %d symbols...",
                len(cn_symbols),
            )
            for symbol in cn_symbols:
                try:
                    cn_data = self._fetch_cn_fundamentals_payload(symbol)
                    if cn_data:
                        results[symbol].update(cn_data)
                except Exception as exc:  # pragma: no cover - provider/network variability
                    logger.warning("CN fundamentals fetch failed for %s: %s", symbol, exc)

        # ============================================================
        # Phase 1: Batch fetch yfinance fundamentals (~25 min for 7000)
        # ============================================================
        logger.info("Phase 1: Fetching yfinance fundamentals...")
        phase1_start = time.time()

        yf_data = {}
        if yfinance_symbols:
            yf_data = self.bulk_fetcher.fetch_batch_fundamentals(
                yfinance_symbols,
                batch_size=self.yfinance_batch_size,
                include_quarterly=True,
                delay_between_batches=self.yfinance_delay_between_batches,
                delay_per_ticker=self.yfinance_delay_per_ticker,
                market_by_symbol=market_by_symbol,
                progress_callback=(
                    lambda completed, yf_total: progress_callback(
                        max(1, int(total * 0.3 * (completed / max(yf_total, 1)))),
                        total,
                    )
                    if progress_callback
                    else None
                ),
            )

        for symbol in yfinance_symbols:
            if symbol in yf_data and not yf_data[symbol].get('has_error'):
                results[symbol].update(yf_data[symbol])

        phase1_time = time.time() - phase1_start
        yf_success = len([s for s in yfinance_symbols if s in yf_data and not yf_data.get(s, {}).get('has_error')])
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
            # Filter symbols by routing policy: finviz is US-only, so skip
            # HK/JP/TW symbols entirely rather than making doomed API calls.
            finviz_eligible = [
                s for s in symbols
                if routing_policy.is_supported(
                    self._market_for_symbol(s, market_by_symbol),
                    routing_policy.PROVIDER_FINVIZ,
                )
            ]
            skipped = len(symbols) - len(finviz_eligible)
            if skipped:
                logger.info(
                    "Phase 3: skipping %d/%d symbols excluded from finviz "
                    "by routing policy %s (non-US markets).",
                    skipped, len(symbols), routing_policy.policy_version(),
                )

            logger.info(
                "Phase 3: Fetching finviz-only fields for %d symbols...",
                len(finviz_eligible),
            )
            phase3_start = time.time()

            finviz_total = len(finviz_eligible)
            finviz_success = 0
            # Phase 3 is US-only (per routing policy above); the breaker check
            # inside ``_rate_limited_call`` is the single gate. An additional
            # ``breaker.check()`` here would mutate state — it can promote
            # open→half_open and consume the single probe before the actual
            # call runs, leaving the probe wasted (and, in the no-Redis
            # fallback, permanently wedging the circuit).
            from .provider_circuit_breaker import CircuitOpenError

            for i, symbol in enumerate(finviz_eligible):
                # Resolve the symbol's market for per-market rate budgeting;
                # finviz routing already constrained this to supported markets.
                symbol_market = (market_by_symbol or {}).get(symbol) or "US"
                try:
                    finviz_data = self.finviz_service.get_finviz_only_fields(
                        symbol, market=symbol_market,
                    )
                except CircuitOpenError:
                    logger.warning(
                        "Phase 3 aborted at %d/%d: circuit open for finviz:%s",
                        i, finviz_total, symbol_market.lower(),
                    )
                    break
                if finviz_data:
                    results[symbol].update(finviz_data)
                    finviz_success += 1

                if i > 0 and i % 50 == 0 and finviz_total > 0:
                    elapsed = time.time() - phase3_start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (finviz_total - i) / rate if rate > 0 else 0
                    logger.info(
                        f"Phase 3 progress: {i}/{finviz_total} "
                        f"({i/finviz_total*100:.1f}%), ETA: {eta/60:.1f} min"
                    )

                    if progress_callback:
                        # Phase 3 is 50% to 100% of overall progress; scale
                        # by finviz-eligible count, not total.
                        phase3_progress = 0.5 + (i / finviz_total) * 0.5
                        progress_callback(int(total * phase3_progress), total)

            phase3_time = time.time() - phase3_start
            logger.info(
                f"Phase 3 complete: {finviz_success}/{finviz_total} in "
                f"{phase3_time:.1f}s"
            )
            if progress_callback:
                # Always emit completion even when finviz_eligible is empty.
                progress_callback(total, total)
        elif progress_callback:
            progress_callback(total, total)

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
        finviz_workers: Optional[int] = None,
        progress_callback=None,
        market_by_symbol: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Fetch fundamentals with parallel finviz fetching for faster performance.

        The Redis-backed rate limiter still serializes egress per market, so
        ``finviz_workers > 1`` does not raise the request rate above the
        configured cadence — it only fills idle time during HTTP RTT. The
        provider circuit breaker pauses the market entirely on sustained
        429s.

        Args:
            symbols: List of ticker symbols
            include_technicals: Whether to calculate technical indicators
            finviz_workers: Number of parallel workers for finviz. When
                ``None``, resolved per-market via
                ``RateBudgetPolicy.get_provider_workers``.
            progress_callback: Optional progress callback
            market_by_symbol: Optional mapping ``{symbol: market}`` forwarded
                to cadence-aware quarterly growth extraction and to
                per-market finviz rate budgeting.

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
        cn_symbols = [s for s in symbols if self._is_cn_symbol(s, market_by_symbol)]
        cn_symbol_set = set(cn_symbols)
        yfinance_symbols = [s for s in symbols if s not in cn_symbol_set]
        if cn_symbols:
            logger.info(
                "Phase 1a: Fetching CN fundamentals via AKShare/BaoStock-aware routing for %d symbols...",
                len(cn_symbols),
            )
            for symbol in cn_symbols:
                try:
                    cn_data = self._fetch_cn_fundamentals_payload(symbol)
                    if cn_data:
                        results[symbol].update(cn_data)
                except Exception as exc:  # pragma: no cover - provider/network variability
                    logger.warning("CN fundamentals fetch failed for %s: %s", symbol, exc)

        logger.info("Phase 1: Fetching yfinance fundamentals...")
        yf_data = {}
        if yfinance_symbols:
            yf_data = self.bulk_fetcher.fetch_fundamentals_parallel(
                yfinance_symbols,
                batch_size=self.yfinance_batch_size,
                max_workers=3,
                include_quarterly=True,
                delay_per_ticker=self.yfinance_delay_per_ticker,
                market_by_symbol=market_by_symbol,
            )

        for symbol in yfinance_symbols:
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

        # Phase 3: Parallel finviz fetching, grouped by market so the
        # per-market rate limiter and circuit breaker apply correctly.
        from . import provider_routing_policy as routing_policy
        from .rate_budget_policy import get_rate_budget_policy

        market_by_symbol = market_by_symbol or {}
        eligible_by_market: Dict[str, List[str]] = {}
        for symbol in symbols:
            market = self._market_for_symbol(symbol, market_by_symbol)
            if not routing_policy.is_supported(market, routing_policy.PROVIDER_FINVIZ):
                continue
            eligible_by_market.setdefault(market, []).append(symbol)

        policy = get_rate_budget_policy()
        completed = 0
        for market, market_symbols in eligible_by_market.items():
            workers = (
                finviz_workers
                if finviz_workers is not None
                else policy.get_provider_workers("finviz", market)
            )
            logger.info(
                "Phase 3: finviz for market=%s (%d symbols, workers=%d)",
                market, len(market_symbols), workers,
            )
            market_results = self.finviz_service.get_finviz_only_fields_batch(
                market_symbols,
                max_workers=workers,
                market=market,
            )
            for symbol, data in market_results.items():
                if data:
                    results[symbol].update(data)

            completed += len(market_symbols)
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
        fundamentals_cache,
        *,
        session_factory: Callable[[], Session],
        include_quarterly: bool = True,  # kept for compatibility with existing task call sites
        market_by_symbol: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Store hybrid results in fundamentals cache.

        Quarterly growth fields are now consolidated into fundamentals cache,
        so no separate quarterly cache storage is needed.

        Also updates institutional ownership history (SCD2) for tracking
        ownership changes over time.

        Args:
            results: Dict mapping symbols to their fundamental data
            fundamentals_cache: FundamentalsCacheService instance
            market_by_symbol: Optional {symbol: market_code} map. Passed
                through to ``fundamentals_cache.store`` so T2 completeness
                scoring is market-aware without a per-symbol DB lookup.

        Returns:
            Dict with storage statistics, including nested provider error counts.
        """
        stats = {
            'fundamentals_stored': 0,
            'quarterly_stored': 0,
            'ownership_updated': 0,
            'failed': 0,
            'persisted_symbols': 0,
            'failed_persistence_symbols': 0,
            'provider_error_counts': {},
        }
        markets = market_by_symbol or {}
        persisted_symbols: list[str] = []

        provider_error_counts: dict[str, int] = {}

        def _record_provider_error(data: Dict | None) -> None:
            if not data:
                provider_error_counts["empty_payload"] = provider_error_counts.get("empty_payload", 0) + 1
                return
            error = str(data.get("error") or "").lower()
            if "404" in error or "quote not found" in error:
                key = "yahoo_quote_not_found"
            elif "no price data found" in error or "no data found" in error:
                key = "yahoo_price_missing"
            elif error:
                key = "provider_fetch_error"
            else:
                key = "provider_fetch_error"
            provider_error_counts[key] = provider_error_counts.get(key, 0) + 1

        for symbol, data in results.items():
            if not data or data.get('has_error'):
                stats['failed'] += 1
                _record_provider_error(data)
                continue

            try:
                # Store in fundamentals cache (includes quarterly growth fields)
                persisted = fundamentals_cache.store(
                    symbol, data, data_source='hybrid',
                    market=markets.get(symbol),
                )
                if persisted:
                    stats['fundamentals_stored'] += 1
                    stats['persisted_symbols'] += 1
                    persisted_symbols.append(symbol)
                    if include_quarterly:
                        stats['quarterly_stored'] += 1
                else:
                    stats['failed'] += 1
                    stats['failed_persistence_symbols'] += 1

            except Exception as e:
                logger.warning(f"Error storing {symbol}: {e}")
                stats['failed'] += 1
                stats['failed_persistence_symbols'] += 1

        # Bulk update institutional ownership history (SCD2)
        db = None
        try:
            db = session_factory()
            ownership_service = InstitutionalOwnershipService(db)

            # Convert results dict to list for bulk_update
            fundamentals_list = [
                {**results[symbol], 'symbol': symbol}
                for symbol in persisted_symbols
                if results.get(symbol)
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
            if db is not None:
                db.close()

        stats['provider_error_counts'] = provider_error_counts
        return stats
