"""Assembly of persisted/API scan-result projections."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from app.analysis.patterns.config import (
    DEFAULT_SETUP_ENGINE_PARAMETERS,
    SetupEngineParameters,
)
from app.analysis.patterns.rs_line import RsLineLeadershipSnapshot
from app.services.opportunity_state_service import (
    build_data_limited_projection,
    build_opportunity_projection,
)

from .base_screener import ScreenerResult, StockData

logger = logging.getLogger(__name__)

OpportunityProjector = Callable[
    [dict[str, object], StockData, SetupEngineParameters],
    dict[str, object],
]

_MINERVINI_PROMOTED_FIELDS = (
    "rs_rating",
    "rs_rating_1m",
    "rs_rating_3m",
    "rs_rating_12m",
    "stage",
    "stage_name",
    "adr_percent",
    "eps_growth_qq",
    "sales_growth_qq",
    "ma_alignment",
    "vcp_detected",
    "vcp_score",
    "vcp_pivot",
    "vcp_ready_for_breakout",
    "vcp_contraction_ratio",
    "vcp_atr_score",
    "position_52week",
    "volume_trend",
    "rs_sparkline_data",
    "rs_trend",
    "price_sparkline_data",
    "price_change_1d",
    "price_trend",
    "perf_week",
    "perf_month",
    "perf_3m",
    "perf_6m",
    "gap_percent",
    "volume_surge",
    "pocket_pivot",
    "power_trend",
    "ema_10_distance",
    "ema_20_distance",
    "ema_50_distance",
    "above_52w_low_pct",
    "from_52w_high_pct",
    "beta",
    "beta_adj_rs",
    "beta_adj_rs_1m",
    "beta_adj_rs_3m",
    "beta_adj_rs_12m",
)


def market_rs_audit_fields(stock_data: StockData) -> dict[str, object]:
    if stock_data.rs_source is None:
        return {
            "rs_formula_version": None,
            "market_rs_run_id": None,
            "rs_universe_size": None,
        }
    return stock_data.rs_source.audit_fields()


def _history_bar_count(stock_data: StockData) -> int:
    price_data = stock_data.price_data
    if price_data is None or getattr(price_data, "empty", True):
        return 0
    return len(price_data)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


@dataclass(frozen=True)
class ScanResultAssemblyRequest:
    symbol: str
    stock_data: StockData
    screener_results: dict[str, ScreenerResult]
    composite_score: float
    overall_rating: str
    composite_method: str
    applicable_screeners: tuple[str, ...] | None = None
    unavailable_screeners: tuple[str, ...] | None = None
    history_bars: int | None = None
    scan_mode: str = "full"
    data_status: str = "complete"
    is_scannable: bool = True
    ipo_bonus: float = 0.0
    composite_reason: str | None = None
    quality_downgrade_reason: str | None = None
    field_completeness_score: int | None = None
    opportunity_parameters: SetupEngineParameters = DEFAULT_SETUP_ENGINE_PARAMETERS


class ScanResultAssembler:
    """Build one stable scan-result projection from completed screener output."""

    def __init__(
        self,
        *,
        opportunity_projector: OpportunityProjector = build_opportunity_projection,
    ) -> None:
        self._opportunity_projector = opportunity_projector

    def assemble(self, request: ScanResultAssemblyRequest) -> dict[str, object]:
        applicable_screeners = (
            request.applicable_screeners
            if request.applicable_screeners is not None
            else tuple(request.screener_results)
        )
        unavailable_screeners = (
            request.unavailable_screeners
            if request.unavailable_screeners is not None
            else ()
        )
        history_bars = (
            request.history_bars
            if request.history_bars is not None
            else _history_bar_count(request.stock_data)
        )
        result = self._base_result(
            request,
            applicable_screeners=applicable_screeners,
            unavailable_screeners=unavailable_screeners,
            history_bars=history_bars,
        )
        self._promote_minervini(result, request.screener_results)
        self._promote_setup_engine(result, request.screener_results)
        self._promote_canslim_growth(result, request.screener_results)
        self._promote_quarterly_growth(result, request.stock_data)
        self._promote_fundamentals(result, request)
        self._attach_average_dollar_volume(result, request)
        self._attach_opportunity_projection(result, request)
        return result

    @staticmethod
    def _base_result(
        request: ScanResultAssemblyRequest,
        *,
        applicable_screeners: tuple[str, ...],
        unavailable_screeners: tuple[str, ...],
        history_bars: int,
    ) -> dict[str, object]:
        screener_results = request.screener_results
        audit_fields = market_rs_audit_fields(request.stock_data)
        leadership = (
            request.stock_data.precomputed_scan_context.rs_line_leadership
            if request.stock_data.precomputed_scan_context is not None
            else RsLineLeadershipSnapshot.empty()
        )
        result: dict[str, object] = {
            "symbol": request.symbol,
            "composite_score": round(request.composite_score, 2),
            "rating": request.overall_rating,
            "current_price": request.stock_data.get_current_price(),
            **{
                f"{name}_score": output.score
                for name, output in screener_results.items()
            },
            **{
                f"{name}_rating": output.rating
                for name, output in screener_results.items()
            },
            **{
                f"{name}_passes": output.passes
                for name, output in screener_results.items()
            },
            "screeners_run": list(screener_results),
            "composite_method": request.composite_method,
            "screeners_passed": sum(
                1 for output in screener_results.values() if output.passes
            ),
            "screeners_total": len(screener_results),
            "result_status": "ok",
            "data_status": request.data_status,
            "is_scannable": request.is_scannable,
            "scan_mode": request.scan_mode,
            "history_bars": history_bars,
            "applicable_screeners": list(applicable_screeners),
            "unavailable_screeners": list(unavailable_screeners),
            "composite_reason": request.composite_reason,
            "ipo_bonus": request.ipo_bonus,
            **audit_fields,
            "field_completeness_score": request.field_completeness_score,
            "quality_downgrade_reason": request.quality_downgrade_reason,
            **leadership.as_scan_fields(),
            "details": {
                "screeners": {
                    name: {
                        "score": output.score,
                        "passes": output.passes,
                        "rating": output.rating,
                        "breakdown": output.breakdown,
                        "details": output.details,
                    }
                    for name, output in screener_results.items()
                },
                "data_errors": request.stock_data.fetch_errors or None,
                **audit_fields,
            },
        }
        return result

    @staticmethod
    def _promote_minervini(
        result: dict[str, object],
        screener_results: dict[str, ScreenerResult],
    ) -> None:
        output = screener_results.get("minervini")
        if output is None:
            return
        result["passes_template"] = output.passes
        result["minervini_score"] = output.score
        for field in _MINERVINI_PROMOTED_FIELDS:
            if field in output.details:
                result[field] = output.details[field]

    @staticmethod
    def _promote_setup_engine(
        result: dict[str, object],
        screener_results: dict[str, ScreenerResult],
    ) -> None:
        output = screener_results.get("setup_engine")
        if output is None or not isinstance(output.details, dict):
            return
        if "setup_engine" in output.details:
            result["setup_engine"] = dict(output.details["setup_engine"])

    @staticmethod
    def _promote_canslim_growth(
        result: dict[str, object],
        screener_results: dict[str, ScreenerResult],
    ) -> None:
        output = screener_results.get("canslim")
        if output is None:
            return
        current = output.details.get("c_current_earnings")
        if (
            result.get("eps_growth_qq") is None
            and isinstance(current, dict)
            and "eps_growth_qq" in current
        ):
            result["eps_growth_qq"] = current["eps_growth_qq"]
        annual = output.details.get("a_annual_earnings")
        if (
            result.get("eps_growth_yy") is None
            and isinstance(annual, dict)
            and "eps_growth_yy" in annual
        ):
            result["eps_growth_yy"] = annual["eps_growth_yy"]

    @staticmethod
    def _promote_quarterly_growth(
        result: dict[str, object], stock_data: StockData
    ) -> None:
        growth = stock_data.quarterly_growth or {}
        for field in (
            "eps_growth_qq",
            "sales_growth_qq",
            "eps_growth_yy",
            "sales_growth_yy",
        ):
            if result.get(field) is None and growth.get(field) is not None:
                result[field] = growth[field]

    @staticmethod
    def _promote_fundamentals(
        result: dict[str, object], request: ScanResultAssemblyRequest
    ) -> None:
        fundamentals = request.stock_data.fundamentals or {}
        for source, target in (
            ("market_cap", "market_cap"),
            ("market_cap_usd", "market_cap_usd"),
            ("eps_rating", "eps_rating"),
            ("dividend_yield", "dividend_yield"),
        ):
            if fundamentals.get(source) is not None:
                result[target] = fundamentals[source]
        if fundamentals.get("sector"):
            result["gics_sector"] = fundamentals["sector"]
        if fundamentals.get("industry"):
            result["gics_industry"] = fundamentals["industry"]
        if fundamentals.get("ipo_date"):
            result["ipo_date"] = fundamentals["ipo_date"]
            return
        ipo_output = request.screener_results.get("ipo")
        if ipo_output is not None and isinstance(ipo_output.details, dict):
            if ipo_output.details.get("ipo_date"):
                result["ipo_date"] = ipo_output.details["ipo_date"]

    @staticmethod
    def _attach_average_dollar_volume(
        result: dict[str, object], request: ScanResultAssemblyRequest
    ) -> None:
        stock_data = request.stock_data
        fundamentals = stock_data.fundamentals or {}
        avg_volume = fundamentals.get("avg_volume") or None
        if avg_volume is None and stock_data.price_data is not None:
            recent = stock_data.price_data.tail(50)
            if not recent.empty and "Volume" in recent.columns:
                volumes = recent["Volume"].dropna()
                if not volumes.empty:
                    avg_volume = int(volumes.mean())
        avg_volume_value = _finite_float(avg_volume)
        current_price_value = _finite_float(result.get("current_price"))
        if avg_volume_value is not None and current_price_value is not None:
            result["avg_dollar_volume"] = int(avg_volume_value * current_price_value)
            return
        logger.debug(
            "Skipping avg_dollar_volume for %s avg_volume=%s current_price=%s",
            request.symbol,
            avg_volume,
            result.get("current_price"),
        )

    def _attach_opportunity_projection(
        self,
        result: dict[str, object],
        request: ScanResultAssemblyRequest,
    ) -> None:
        try:
            projection = self._opportunity_projector(
                result,
                request.stock_data,
                request.opportunity_parameters,
            )
        except Exception:
            logger.exception(
                "Opportunity policy assembly failed for %s", request.symbol
            )
            projection = build_data_limited_projection(
                result,
                request.stock_data,
                "opportunity_policy_error",
            )
        result.update(projection)
