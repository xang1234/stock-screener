"""
Scan orchestrator for multi-screener coordination.

Coordinates running multiple screeners on a single stock, with data
fetched once and shared across all screeners. Combines results and
calculates composite scores.
"""
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_screener import BaseStockScreener, ScreenerResult, StockData, DataRequirements
from .screener_registry import ScreenerRegistry

from app.domain.scanning.models import CompositeMethod, ScreenerOutputDomain
from app.domain.scanning.scoring import calculate_composite_score, calculate_overall_rating
from app.domain.scanning.ports import StockDataProvider

logger = logging.getLogger(__name__)


def _to_domain_output(name: str, result: ScreenerResult) -> ScreenerOutputDomain:
    """Map an infrastructure ScreenerResult to a domain ScreenerOutputDomain."""
    return ScreenerOutputDomain(
        screener_name=name,
        score=result.score,
        passes=result.passes,
        rating=result.rating,
        breakdown=result.breakdown,
        details=result.details,
    )


class ScanOrchestrator:
    """
    Orchestrates multi-screener stock analysis.

    Coordinates:
    1. Getting screeners from registry
    2. Merging data requirements
    3. Fetching data once
    4. Running all screeners
    5. Combining results
    """

    def __init__(self, data_provider: StockDataProvider, registry: ScreenerRegistry):
        """Initialize orchestrator with injected dependencies.

        Args:
            data_provider: Port for fetching stock data
            registry: Registry of available screeners
        """
        self._data_provider = data_provider
        self._registry = registry

    def scan_stock_multi(
        self,
        symbol: str,
        screener_names: List[str],
        criteria: Optional[Dict] = None,
        composite_method: str = "weighted_average",
        pre_merged_requirements: Optional[DataRequirements] = None,
        pre_fetched_data: Optional[StockData] = None
    ) -> Dict:
        """
        Run multiple screeners on a single stock.

        Args:
            symbol: Stock symbol
            screener_names: List of screener names to run
            criteria: Optional criteria/parameters for screeners
            composite_method: How to combine scores (weighted_average, maximum, minimum)
            pre_merged_requirements: Optional pre-merged requirements (batch optimization)
            pre_fetched_data: Optional pre-fetched stock data (batch optimization)

        Returns:
            Dict with combined results from all screeners
        """
        try:
            # Parse composite_method string defensively
            try:
                method_enum = CompositeMethod(composite_method)
            except ValueError:
                logger.warning("Unknown composite method '%s', defaulting to weighted_average", composite_method)
                method_enum = CompositeMethod.WEIGHTED_AVERAGE

            # 1. Get screener instances from registry
            try:
                screeners = self._registry.get_multiple(screener_names)
            except ValueError as e:
                logger.error(f"Error getting screeners: {e}")
                return self._error_result(symbol, str(e))

            # 2. Get stock data (use pre-fetched if available)
            if pre_fetched_data:
                # Use pre-fetched data from batch processing (NO rate limiting!)
                stock_data = pre_fetched_data
                logger.debug(f"Using pre-fetched data for {symbol}")
            else:
                # Merge requirements and fetch data (legacy path)
                if pre_merged_requirements:
                    requirements = pre_merged_requirements
                    logger.debug(f"Using pre-merged requirements for {symbol}")
                else:
                    requirements = DataRequirements.merge_all([
                        screener.get_data_requirements(criteria)
                        for screener in screeners.values()
                    ])
                    logger.info(f"Merged data requirements for {symbol}: {requirements}")

                # Fetch data ONCE
                stock_data = self._data_provider.prepare_data(symbol, requirements)

            # 4. Validate data
            if not stock_data.has_sufficient_data():
                return self._insufficient_data_result(
                    symbol,
                    stock_data.get_error_summary() or "Insufficient data"
                )

            # 5. Run all screeners in parallel on the same data
            screener_results: Dict[str, ScreenerResult] = {}

            def run_screener(name: str, screener: BaseStockScreener) -> tuple[str, Optional[ScreenerResult]]:
                """Run a single screener and return (name, result) tuple."""
                try:
                    result = screener.scan_stock(symbol, stock_data, criteria)
                    logger.info(
                        f"{symbol} - {name}: score={result.score:.1f}, "
                        f"passes={result.passes}, rating={result.rating}"
                    )
                    return (name, result)
                except Exception as e:
                    logger.error(f"Error running {name} screener on {symbol}: {e}")
                    return (name, None)

            # Execute screeners in parallel (max 5 workers for 5 screeners)
            with ThreadPoolExecutor(max_workers=min(len(screeners), 5)) as executor:
                # Submit all screener tasks
                futures = {
                    executor.submit(run_screener, name, screener): name
                    for name, screener in screeners.items()
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    name, result = future.result()
                    if result is not None:
                        screener_results[name] = result

            if not screener_results:
                return self._error_result(symbol, "All screeners failed")

            # 6. Calculate composite score using domain functions
            domain_outputs = {name: _to_domain_output(name, r) for name, r in screener_results.items()}
            composite_score = calculate_composite_score(domain_outputs, method_enum)

            # 7. Determine overall rating
            rating_category = calculate_overall_rating(composite_score, domain_outputs)
            overall_rating = rating_category.value

            # 8. Combine results
            combined_result = self._combine_results(
                symbol,
                stock_data,
                screener_results,
                composite_score,
                overall_rating,
                composite_method
            )

            return combined_result

        except Exception as e:
            logger.error(f"Error orchestrating scan for {symbol}: {e}")
            return self._error_result(symbol, str(e))

    def _combine_results(
        self,
        symbol: str,
        stock_data: StockData,
        screener_results: Dict[str, ScreenerResult],
        composite_score: float,
        overall_rating: str,
        composite_method: str
    ) -> Dict:
        """
        Combine all screener results into a single result dict.

        Args:
            symbol: Stock symbol
            stock_data: Stock data
            screener_results: Results from each screener
            composite_score: Combined score
            overall_rating: Overall rating
            composite_method: Method used for combining

        Returns:
            Combined result dict
        """
        # Extract individual scores
        individual_scores = {
            f"{name}_score": result.score
            for name, result in screener_results.items()
        }

        # Extract individual ratings
        individual_ratings = {
            f"{name}_rating": result.rating
            for name, result in screener_results.items()
        }

        # Extract individual passes
        individual_passes = {
            f"{name}_passes": result.passes
            for name, result in screener_results.items()
        }

        # Build breakdown of all screener details
        screener_details = {
            name: {
                "score": result.score,
                "passes": result.passes,
                "rating": result.rating,
                "breakdown": result.breakdown,
                "details": result.details
            }
            for name, result in screener_results.items()
        }

        # Get current price
        current_price = stock_data.get_current_price()

        # Build combined result
        result = {
            "symbol": symbol,
            "composite_score": round(composite_score, 2),
            "rating": overall_rating,
            "current_price": current_price,

            # Individual screener scores
            **individual_scores,

            # Individual ratings
            **individual_ratings,

            # Individual passes
            **individual_passes,

            # Metadata
            "screeners_run": list(screener_results.keys()),
            "composite_method": composite_method,
            "screeners_passed": sum(1 for r in screener_results.values() if r.passes),
            "screeners_total": len(screener_results),

            # Full details
            "details": {
                "screeners": screener_details,
                "data_errors": stock_data.fetch_errors if stock_data.fetch_errors else None
            }
        }

        # Add backward compatibility fields for minervini
        if "minervini" in screener_results:
            minervini_result = screener_results["minervini"]
            result["passes_template"] = minervini_result.passes
            result["minervini_score"] = minervini_result.score

            # Extract common Minervini fields from details if available
            minervini_details = minervini_result.details
            # Core fields
            if "rs_rating" in minervini_details:
                result["rs_rating"] = minervini_details["rs_rating"]
            if "rs_rating_1m" in minervini_details:
                result["rs_rating_1m"] = minervini_details["rs_rating_1m"]
            if "rs_rating_3m" in minervini_details:
                result["rs_rating_3m"] = minervini_details["rs_rating_3m"]
            if "rs_rating_12m" in minervini_details:
                result["rs_rating_12m"] = minervini_details["rs_rating_12m"]
            if "stage" in minervini_details:
                result["stage"] = minervini_details["stage"]
            if "stage_name" in minervini_details:
                result["stage_name"] = minervini_details["stage_name"]
            # Growth metrics
            if "adr_percent" in minervini_details:
                result["adr_percent"] = minervini_details["adr_percent"]
            if "eps_growth_qq" in minervini_details:
                result["eps_growth_qq"] = minervini_details["eps_growth_qq"]
            if "sales_growth_qq" in minervini_details:
                result["sales_growth_qq"] = minervini_details["sales_growth_qq"]
            # Technical indicators
            if "ma_alignment" in minervini_details:
                result["ma_alignment"] = minervini_details["ma_alignment"]
            if "vcp_detected" in minervini_details:
                result["vcp_detected"] = minervini_details["vcp_detected"]
            if "vcp_score" in minervini_details:
                result["vcp_score"] = minervini_details["vcp_score"]
            if "vcp_pivot" in minervini_details:
                result["vcp_pivot"] = minervini_details["vcp_pivot"]
            if "vcp_ready_for_breakout" in minervini_details:
                result["vcp_ready_for_breakout"] = minervini_details["vcp_ready_for_breakout"]
            if "vcp_contraction_ratio" in minervini_details:
                result["vcp_contraction_ratio"] = minervini_details["vcp_contraction_ratio"]
            if "vcp_atr_score" in minervini_details:
                result["vcp_atr_score"] = minervini_details["vcp_atr_score"]
            if "position_52week" in minervini_details:
                result["position_52week"] = minervini_details["position_52week"]
            if "volume_trend" in minervini_details:
                result["volume_trend"] = minervini_details["volume_trend"]
            # RS Sparkline data
            if "rs_sparkline_data" in minervini_details:
                result["rs_sparkline_data"] = minervini_details["rs_sparkline_data"]
            if "rs_trend" in minervini_details:
                result["rs_trend"] = minervini_details["rs_trend"]
            # Price Sparkline data
            if "price_sparkline_data" in minervini_details:
                result["price_sparkline_data"] = minervini_details["price_sparkline_data"]
            if "price_change_1d" in minervini_details:
                result["price_change_1d"] = minervini_details["price_change_1d"]
            if "price_trend" in minervini_details:
                result["price_trend"] = minervini_details["price_trend"]
            # Performance metrics (new technical filters)
            if "perf_week" in minervini_details:
                result["perf_week"] = minervini_details["perf_week"]
            if "perf_month" in minervini_details:
                result["perf_month"] = minervini_details["perf_month"]
            # Qullamaggie extended performance metrics
            if "perf_3m" in minervini_details:
                result["perf_3m"] = minervini_details["perf_3m"]
            if "perf_6m" in minervini_details:
                result["perf_6m"] = minervini_details["perf_6m"]
            # Episodic Pivot metrics
            if "gap_percent" in minervini_details:
                result["gap_percent"] = minervini_details["gap_percent"]
            if "volume_surge" in minervini_details:
                result["volume_surge"] = minervini_details["volume_surge"]
            # EMA distances (new technical filters)
            if "ema_10_distance" in minervini_details:
                result["ema_10_distance"] = minervini_details["ema_10_distance"]
            if "ema_20_distance" in minervini_details:
                result["ema_20_distance"] = minervini_details["ema_20_distance"]
            if "ema_50_distance" in minervini_details:
                result["ema_50_distance"] = minervini_details["ema_50_distance"]
            # 52-week distances (promoted to top-level for indexed columns)
            if "above_52w_low_pct" in minervini_details:
                result["above_52w_low_pct"] = minervini_details["above_52w_low_pct"]
            if "from_52w_high_pct" in minervini_details:
                result["from_52w_high_pct"] = minervini_details["from_52w_high_pct"]
            # Beta and Beta-Adjusted RS metrics
            if "beta" in minervini_details:
                result["beta"] = minervini_details["beta"]
            if "beta_adj_rs" in minervini_details:
                result["beta_adj_rs"] = minervini_details["beta_adj_rs"]
            if "beta_adj_rs_1m" in minervini_details:
                result["beta_adj_rs_1m"] = minervini_details["beta_adj_rs_1m"]
            if "beta_adj_rs_3m" in minervini_details:
                result["beta_adj_rs_3m"] = minervini_details["beta_adj_rs_3m"]
            if "beta_adj_rs_12m" in minervini_details:
                result["beta_adj_rs_12m"] = minervini_details["beta_adj_rs_12m"]

        # Fallback: Extract growth metrics from CANSLIM if not already set
        if "canslim" in screener_results:
            canslim_details = screener_results["canslim"].details
            # EPS growth from CANSLIM's C criteria
            if result.get("eps_growth_qq") is None and "c_current_earnings" in canslim_details:
                c_details = canslim_details["c_current_earnings"]
                if isinstance(c_details, dict) and "eps_growth_qq" in c_details:
                    result["eps_growth_qq"] = c_details["eps_growth_qq"]
            # EPS growth Y/Y from CANSLIM's A criteria
            if result.get("eps_growth_yy") is None and "a_annual_earnings" in canslim_details:
                a_details = canslim_details["a_annual_earnings"]
                if isinstance(a_details, dict) and "eps_growth_yy" in a_details:
                    result["eps_growth_yy"] = a_details["eps_growth_yy"]

        # Final fallback: Extract growth metrics directly from stock_data.quarterly_growth
        if stock_data.quarterly_growth:
            qg = stock_data.quarterly_growth
            if result.get("eps_growth_qq") is None and qg.get("eps_growth_qq") is not None:
                result["eps_growth_qq"] = qg["eps_growth_qq"]
            if result.get("sales_growth_qq") is None and qg.get("sales_growth_qq") is not None:
                result["sales_growth_qq"] = qg["sales_growth_qq"]
            if result.get("eps_growth_yy") is None and qg.get("eps_growth_yy") is not None:
                result["eps_growth_yy"] = qg["eps_growth_yy"]
            if result.get("sales_growth_yy") is None and qg.get("sales_growth_yy") is not None:
                result["sales_growth_yy"] = qg["sales_growth_yy"]

        # Extract market_cap from fundamentals if available
        if stock_data.fundamentals and stock_data.fundamentals.get("market_cap"):
            result["market_cap"] = stock_data.fundamentals["market_cap"]

        # Extract EPS Rating from fundamentals if available
        if stock_data.fundamentals and stock_data.fundamentals.get("eps_rating") is not None:
            result["eps_rating"] = stock_data.fundamentals["eps_rating"]

        # Extract IPO date from fundamentals if available
        if stock_data.fundamentals and stock_data.fundamentals.get("ipo_date"):
            result["ipo_date"] = stock_data.fundamentals["ipo_date"]

        # Calculate average dollar volume (avg_volume × current_price)
        # Primary: Use avg_volume from fundamentals (Finviz data)
        # Fallback: Calculate from price_data
        avg_volume = None
        current_price = result.get("current_price")

        # Try fundamentals first (more reliable avg_volume from Finviz)
        if stock_data.fundamentals and stock_data.fundamentals.get("avg_volume"):
            avg_volume = stock_data.fundamentals["avg_volume"]

        # Fallback: calculate from price_data if fundamentals not available
        if avg_volume is None and stock_data.price_data is not None and len(stock_data.price_data) > 0:
            price_data = stock_data.price_data
            recent_data = price_data.tail(50)
            if len(recent_data) > 0 and "Volume" in recent_data.columns:
                vol_series = recent_data["Volume"].dropna()
                if len(vol_series) > 0:
                    avg_volume = int(vol_series.mean())

        # Calculate dollar volume if we have both avg_volume and price
        if avg_volume and current_price:
            result["avg_dollar_volume"] = int(avg_volume * current_price)

        return result

    def _error_result(self, symbol: str, error: str) -> Dict:
        """Return result for errors."""
        return {
            "symbol": symbol,
            "composite_score": 0,
            "rating": "Error",
            "error": f"Scan error: {error}",
            "current_price": None,
            "screeners_run": [],
        }

    def _insufficient_data_result(self, symbol: str, reason: str) -> Dict:
        """Return result for insufficient data."""
        return {
            "symbol": symbol,
            "composite_score": 0,
            "rating": "Insufficient Data",
            "error": reason,
            "current_price": None,
            "screeners_run": [],
        }
