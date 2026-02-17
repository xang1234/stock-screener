"""SQLAlchemy implementation of ScanResultRepository."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.domain.scanning.filter_spec import QuerySpec
from app.domain.scanning.models import ResultPage, ScanResultItemDomain
from app.domain.scanning.ports import ScanResultRepository
from app.infra.query.scan_result_query import apply_filters, apply_sort_and_paginate
from app.infra.serialization import convert_numpy_types
from app.models.scan_result import ScanResult
from app.models.stock_universe import StockUniverse

logger = logging.getLogger(__name__)


def _map_orchestrator_result(scan_id: str, symbol: str, raw: dict) -> dict:
    """Map a raw orchestrator result dict to ScanResult column values.

    This is a **pure function** — no DB queries, no side-effects.
    IBD/GICS classification lookups are *not* performed here; if needed
    the caller should pre-populate ``ibd_industry_group`` etc. in *raw*.
    """
    # Normalise numpy/pandas types to native Python before extraction.
    # Without this, numpy floats/ints from the ScanOrchestrator would
    # cause JSON serialization failures in the ``details`` column.
    raw = convert_numpy_types(raw)

    r: dict[str, Any] = {}

    r["scan_id"] = scan_id
    r["symbol"] = symbol.upper()

    # Scores
    r["composite_score"] = raw.get("composite_score", raw.get("minervini_score", 0))
    r["minervini_score"] = raw.get("minervini_score")
    r["canslim_score"] = raw.get("canslim_score")
    r["ipo_score"] = raw.get("ipo_score")
    r["custom_score"] = raw.get("custom_score")
    r["volume_breakthrough_score"] = raw.get("volume_breakthrough_score")

    # Rating (backward-compat fallback)
    rating = raw.get("rating")
    if not rating:
        passes = raw.get("passes_template", False)
        cs = r["composite_score"] or 0
        if passes and cs >= 80:
            rating = "Strong Buy"
        elif passes:
            rating = "Buy"
        elif cs >= 60:
            rating = "Watch"
        else:
            rating = "Pass"
    r["rating"] = rating

    # Price / volume / cap
    r["price"] = raw.get("current_price")
    r["volume"] = raw.get("avg_dollar_volume")
    r["market_cap"] = raw.get("market_cap")

    # Indexed technical fields
    r["stage"] = raw.get("stage")
    r["rs_rating"] = raw.get("rs_rating")
    r["rs_rating_1m"] = raw.get("rs_rating_1m")
    r["rs_rating_3m"] = raw.get("rs_rating_3m")
    r["rs_rating_12m"] = raw.get("rs_rating_12m")
    r["eps_growth_qq"] = raw.get("eps_growth_qq")
    r["sales_growth_qq"] = raw.get("sales_growth_qq")
    r["eps_growth_yy"] = raw.get("eps_growth_yy")
    r["sales_growth_yy"] = raw.get("sales_growth_yy")
    r["peg_ratio"] = raw.get("peg_ratio")
    r["adr_percent"] = raw.get("adr_percent")
    r["eps_rating"] = raw.get("eps_rating")
    r["ipo_date"] = raw.get("ipo_date")

    # Beta / Beta-Adjusted RS
    r["beta"] = raw.get("beta")
    r["beta_adj_rs"] = raw.get("beta_adj_rs")
    r["beta_adj_rs_1m"] = raw.get("beta_adj_rs_1m")
    r["beta_adj_rs_3m"] = raw.get("beta_adj_rs_3m")
    r["beta_adj_rs_12m"] = raw.get("beta_adj_rs_12m")

    # Sparklines
    r["rs_sparkline_data"] = raw.get("rs_sparkline_data")
    r["rs_trend"] = raw.get("rs_trend")
    r["price_sparkline_data"] = raw.get("price_sparkline_data")
    r["price_change_1d"] = raw.get("price_change_1d")
    r["price_trend"] = raw.get("price_trend")

    # Performance metrics
    r["perf_week"] = raw.get("perf_week")
    r["perf_month"] = raw.get("perf_month")
    r["perf_3m"] = raw.get("perf_3m")
    r["perf_6m"] = raw.get("perf_6m")

    # Episodic Pivot
    r["gap_percent"] = raw.get("gap_percent")
    r["volume_surge"] = raw.get("volume_surge")

    # EMA distances
    r["ema_10_distance"] = raw.get("ema_10_distance")
    r["ema_20_distance"] = raw.get("ema_20_distance")
    r["ema_50_distance"] = raw.get("ema_50_distance")

    # 52-week distances
    r["week_52_high_distance"] = raw.get("from_52w_high_pct")
    r["week_52_low_distance"] = raw.get("above_52w_low_pct")

    # IBD/GICS (caller may pre-populate in raw)
    r["ibd_industry_group"] = raw.get("ibd_industry_group")
    r["ibd_group_rank"] = raw.get("ibd_group_rank")
    r["gics_sector"] = raw.get("gics_sector")
    r["gics_industry"] = raw.get("gics_industry")

    # Full result dict stored as JSON
    r["details"] = raw

    return r


class SqlScanResultRepository(ScanResultRepository):
    """Persist and retrieve ScanResult rows via SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_insert(self, rows: list[dict]) -> int:
        objects = [ScanResult(**row) for row in rows]
        self._session.bulk_save_objects(objects)
        self._session.flush()
        return len(objects)

    def persist_orchestrator_results(
        self, scan_id: str, results: list[tuple[str, dict]]
    ) -> int:
        rows = [
            _map_orchestrator_result(scan_id, symbol, result)
            for symbol, result in results
        ]
        return self.bulk_insert(rows)

    def count_by_scan_id(self, scan_id: str) -> int:
        return (
            self._session.query(func.count(ScanResult.id))
            .filter(ScanResult.scan_id == scan_id)
            .scalar()
            or 0
        )

    def query(
        self,
        scan_id: str,
        spec: QuerySpec,
        *,
        include_sparklines: bool = True,
    ) -> ResultPage:
        # Base query: LEFT JOIN stock_universe to get company names.
        q = (
            self._session.query(ScanResult, StockUniverse.name)
            .outerjoin(StockUniverse, ScanResult.symbol == StockUniverse.symbol)
            .filter(ScanResult.scan_id == scan_id)
        )

        # Apply domain filters → SQLAlchemy WHERE clauses.
        q = apply_filters(q, spec.filters)

        # Apply sort + pagination (SQL or Python depending on field).
        rows, total, _python_sorted = apply_sort_and_paginate(
            q, spec.sort, spec.page,
        )

        # Map ORM rows to domain objects.
        items = tuple(
            _map_row_to_domain(result, company_name, include_sparklines)
            for result, company_name in rows
        )

        return ResultPage(
            items=items,
            total=total,
            page=spec.page.page,
            per_page=spec.page.per_page,
        )


def _map_row_to_domain(
    result: ScanResult,
    company_name: str | None,
    include_sparklines: bool,
) -> ScanResultItemDomain:
    """Map a ScanResult ORM row to a domain value object."""
    details = result.details or {}

    extended: dict[str, Any] = {
        "company_name": company_name,
        "minervini_score": result.minervini_score,
        "canslim_score": result.canslim_score,
        "ipo_score": result.ipo_score,
        "custom_score": result.custom_score,
        "volume_breakthrough_score": result.volume_breakthrough_score,
        "rs_rating": result.rs_rating,
        "rs_rating_1m": result.rs_rating_1m,
        "rs_rating_3m": result.rs_rating_3m,
        "rs_rating_12m": result.rs_rating_12m,
        "stage": result.stage,
        "stage_name": details.get("stage_name"),
        "volume": result.volume,
        "market_cap": result.market_cap,
        "ma_alignment": details.get("ma_alignment"),
        "vcp_detected": details.get("vcp_detected"),
        "vcp_score": details.get("vcp_score"),
        "vcp_pivot": details.get("vcp_pivot"),
        "vcp_ready_for_breakout": details.get("vcp_ready_for_breakout"),
        "vcp_contraction_ratio": details.get("vcp_contraction_ratio"),
        "vcp_atr_score": details.get("vcp_atr_score"),
        "passes_template": details.get("passes_template", False),
        "adr_percent": result.adr_percent,
        "eps_growth_qq": result.eps_growth_qq,
        "sales_growth_qq": result.sales_growth_qq,
        "eps_growth_yy": result.eps_growth_yy,
        "sales_growth_yy": result.sales_growth_yy,
        "peg_ratio": result.peg_ratio,
        "eps_rating": result.eps_rating,
        "ibd_industry_group": result.ibd_industry_group,
        "ibd_group_rank": result.ibd_group_rank,
        "gics_sector": result.gics_sector,
        "gics_industry": result.gics_industry,
        "rs_sparkline_data": result.rs_sparkline_data if include_sparklines else None,
        "rs_trend": result.rs_trend,
        "price_sparkline_data": result.price_sparkline_data if include_sparklines else None,
        "price_change_1d": result.price_change_1d,
        "price_trend": result.price_trend,
        "ipo_date": result.ipo_date,
        "beta": result.beta,
        "beta_adj_rs": result.beta_adj_rs,
        "beta_adj_rs_1m": result.beta_adj_rs_1m,
        "beta_adj_rs_3m": result.beta_adj_rs_3m,
        "beta_adj_rs_12m": result.beta_adj_rs_12m,
        "perf_week": result.perf_week,
        "perf_month": result.perf_month,
        "perf_3m": result.perf_3m,
        "perf_6m": result.perf_6m,
        "gap_percent": result.gap_percent,
        "volume_surge": result.volume_surge,
        "ema_10_distance": result.ema_10_distance,
        "ema_20_distance": result.ema_20_distance,
        "ema_50_distance": result.ema_50_distance,
        "week_52_high_distance": result.week_52_high_distance,
        "week_52_low_distance": result.week_52_low_distance,
    }

    # Clamp score to 0-100 to satisfy domain invariant; legacy data may exceed.
    raw_score = result.composite_score or 0
    clamped_score = max(0.0, min(100.0, float(raw_score)))

    return ScanResultItemDomain(
        symbol=result.symbol,
        composite_score=clamped_score,
        rating=result.rating or "Pass",
        current_price=result.price,
        screener_outputs={},  # not populated for list queries
        screeners_run=details.get("screeners_run", []),
        composite_method=details.get("composite_method", "weighted_average"),
        screeners_passed=details.get("screeners_passed", 0),
        screeners_total=details.get("screeners_total", 0),
        extended_fields=extended,
    )
