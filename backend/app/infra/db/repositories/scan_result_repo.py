"""SQLAlchemy implementation of ScanResultRepository."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.domain.scanning.filter_spec import QuerySpec
from app.domain.scanning.models import FilterOptions, ResultPage, ScanResultItemDomain
from app.domain.scanning.ports import ScanResultRepository
from app.domain.scanning.filter_spec import FilterSpec, SortSpec
from app.analysis.patterns.report import validate_setup_engine_report_payload
from app.infra.query.scan_result_query import apply_filters, apply_sort_all, apply_sort_and_paginate
from app.infra.serialization import convert_numpy_types
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.scan_result import ScanResult
from app.models.stock import StockFundamental, StockIndustry
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

    setup_engine = raw.get("setup_engine")
    if isinstance(setup_engine, dict):
        validation_errors = validate_setup_engine_report_payload(setup_engine)
        if validation_errors:
            logger.warning(
                "%s setup_engine payload validation failed; dropping payload. errors=%s",
                symbol.upper(),
                validation_errors[:5],
            )
            raw = dict(raw)
            raw.pop("setup_engine", None)
            raw["setup_engine_validation_errors"] = validation_errors

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
        if not results:
            return 0

        symbols = [symbol.upper() for symbol, _ in results]
        enrichment = self._load_symbol_enrichment(symbols)

        rows = [
            _map_orchestrator_result(
                scan_id,
                symbol,
                self._enrich_raw_result(symbol.upper(), result, enrichment),
            )
            for symbol, result in results
        ]
        return self.bulk_insert(rows)

    def _load_symbol_enrichment(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Load enrichment fields used by scan result persistence in bulk."""
        if not symbols:
            return {}

        unique_symbols = sorted(set(symbols))
        enrichment: dict[str, dict[str, Any]] = {
            s: {} for s in unique_symbols
        }

        fundamentals_rows = (
            self._session.query(
                StockFundamental.symbol,
                StockFundamental.eps_rating,
                StockFundamental.ipo_date,
                StockFundamental.sector,
                StockFundamental.industry,
            )
            .filter(StockFundamental.symbol.in_(unique_symbols))
            .all()
        )
        for row in fundamentals_rows:
            d = enrichment.setdefault(row.symbol, {})
            d["eps_rating"] = row.eps_rating
            d["ipo_date"] = row.ipo_date
            d["sector"] = row.sector
            d["industry"] = row.industry

        industry_rows = (
            self._session.query(
                StockIndustry.symbol,
                StockIndustry.sector,
                StockIndustry.industry,
            )
            .filter(StockIndustry.symbol.in_(unique_symbols))
            .all()
        )
        for row in industry_rows:
            d = enrichment.setdefault(row.symbol, {})
            # Only backfill missing values from stock_industry.
            if not d.get("sector"):
                d["sector"] = row.sector
            if not d.get("industry"):
                d["industry"] = row.industry

        ibd_group_rows = (
            self._session.query(
                IBDIndustryGroup.symbol,
                IBDIndustryGroup.industry_group,
            )
            .filter(IBDIndustryGroup.symbol.in_(unique_symbols))
            .all()
        )
        for row in ibd_group_rows:
            d = enrichment.setdefault(row.symbol, {})
            d["ibd_industry_group"] = row.industry_group

        latest_rank_date = self._session.query(func.max(IBDGroupRank.date)).scalar()
        rank_by_group: dict[str, int] = {}
        if latest_rank_date is not None:
            rank_rows = (
                self._session.query(IBDGroupRank.industry_group, IBDGroupRank.rank)
                .filter(IBDGroupRank.date == latest_rank_date)
                .all()
            )
            rank_by_group = {g: rank for g, rank in rank_rows}

        for symbol, d in enrichment.items():
            group_name = d.get("ibd_industry_group")
            if group_name:
                d["ibd_group_rank"] = rank_by_group.get(group_name)

        return enrichment

    def _enrich_raw_result(
        self,
        symbol: str,
        raw: dict[str, Any],
        enrichment: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Backfill missing classification/fundamental keys before persistence."""
        enriched = dict(raw)
        meta = enrichment.get(symbol, {})

        if enriched.get("eps_rating") is None and meta.get("eps_rating") is not None:
            enriched["eps_rating"] = meta["eps_rating"]

        if not enriched.get("ipo_date"):
            if meta.get("ipo_date"):
                enriched["ipo_date"] = _to_iso_date(meta["ipo_date"])
            else:
                ipo_from_screener = (
                    (
                        enriched.get("details", {})
                        .get("screeners", {})
                        .get("ipo", {})
                        .get("details", {})
                        .get("ipo_date")
                    )
                    if isinstance(enriched.get("details"), dict)
                    else None
                )
                if ipo_from_screener:
                    enriched["ipo_date"] = _to_iso_date(ipo_from_screener)

        if not enriched.get("gics_sector") and meta.get("sector"):
            enriched["gics_sector"] = meta["sector"]
        if not enriched.get("gics_industry") and meta.get("industry"):
            enriched["gics_industry"] = meta["industry"]

        if not enriched.get("ibd_industry_group") and meta.get("ibd_industry_group"):
            enriched["ibd_industry_group"] = meta["ibd_industry_group"]
        if (
            enriched.get("ibd_group_rank") is None
            and meta.get("ibd_group_rank") is not None
        ):
            enriched["ibd_group_rank"] = meta["ibd_group_rank"]

        return enriched

    def delete_by_scan_id(self, scan_id: str) -> int:
        count = (
            self._session.query(ScanResult)
            .filter(ScanResult.scan_id == scan_id)
            .delete()
        )
        self._session.flush()
        return count

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

    def query_all(
        self,
        scan_id: str,
        filters: FilterSpec,
        sort: SortSpec,
        *,
        include_sparklines: bool = False,
    ) -> tuple[ScanResultItemDomain, ...]:
        q = (
            self._session.query(ScanResult, StockUniverse.name)
            .outerjoin(StockUniverse, ScanResult.symbol == StockUniverse.symbol)
            .filter(ScanResult.scan_id == scan_id)
        )
        q = apply_filters(q, filters)
        rows = apply_sort_all(q, sort)
        return tuple(
            _map_row_to_domain(result, company_name, include_sparklines)
            for result, company_name in rows
        )

    def get_by_symbol(
        self, scan_id: str, symbol: str
    ) -> ScanResultItemDomain | None:
        """Return a single result by scan_id + symbol, or None."""
        row = (
            self._session.query(ScanResult, StockUniverse.name)
            .outerjoin(StockUniverse, ScanResult.symbol == StockUniverse.symbol)
            .filter(ScanResult.scan_id == scan_id, ScanResult.symbol == symbol)
            .first()
        )
        if row is None:
            return None
        result, company_name = row
        return _map_row_to_domain(result, company_name, include_sparklines=True)

    def get_peers_by_industry(
        self, scan_id: str, ibd_industry_group: str
    ) -> tuple[ScanResultItemDomain, ...]:
        rows = (
            self._session.query(ScanResult, StockUniverse.name)
            .outerjoin(StockUniverse, ScanResult.symbol == StockUniverse.symbol)
            .filter(
                ScanResult.scan_id == scan_id,
                ScanResult.ibd_industry_group == ibd_industry_group,
            )
            .order_by(ScanResult.composite_score.desc())
            .all()
        )
        return tuple(
            _map_row_to_domain(result, company_name, include_sparklines=True)
            for result, company_name in rows
        )

    def get_peers_by_sector(
        self, scan_id: str, gics_sector: str
    ) -> tuple[ScanResultItemDomain, ...]:
        rows = (
            self._session.query(ScanResult, StockUniverse.name)
            .outerjoin(StockUniverse, ScanResult.symbol == StockUniverse.symbol)
            .filter(
                ScanResult.scan_id == scan_id,
                ScanResult.gics_sector == gics_sector,
            )
            .order_by(ScanResult.composite_score.desc())
            .all()
        )
        return tuple(
            _map_row_to_domain(result, company_name, include_sparklines=True)
            for result, company_name in rows
        )

    def get_details_by_symbol(
        self, scan_id: str, symbol: str
    ) -> dict | None:
        """Return the raw details JSON blob for a single result, or None."""
        row = (
            self._session.query(ScanResult.details)
            .filter(ScanResult.scan_id == scan_id, ScanResult.symbol == symbol)
            .first()
        )
        return row[0] if row else None

    def get_filter_options(self, scan_id: str) -> FilterOptions:
        """Query distinct categorical values for filter dropdowns."""

        def _distinct_non_empty(column):
            """Return sorted unique non-null, non-empty values for *column*."""
            rows = (
                self._session.query(column)
                .filter(
                    ScanResult.scan_id == scan_id,
                    column.isnot(None),
                    column != "",
                )
                .distinct()
                .all()
            )
            return tuple(sorted(v for (v,) in rows if v))

        return FilterOptions(
            ibd_industries=_distinct_non_empty(ScanResult.ibd_industry_group),
            gics_sectors=_distinct_non_empty(ScanResult.gics_sector),
            ratings=_distinct_non_empty(ScanResult.rating),
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

    # Setup Engine fields (extracted from details JSON blob)
    se_data = details.get("setup_engine", {}) or {}
    extended["se_setup_score"] = se_data.get("setup_score")
    extended["se_pattern_primary"] = se_data.get("pattern_primary")
    extended["se_distance_to_pivot_pct"] = se_data.get("distance_to_pivot_pct")
    extended["se_in_early_zone"] = se_data.get("in_early_zone")
    extended["se_extended_from_pivot"] = se_data.get("extended_from_pivot")
    extended["se_base_length_weeks"] = se_data.get("base_length_weeks")
    extended["se_base_depth_pct"] = se_data.get("base_depth_pct")
    extended["se_support_tests_count"] = se_data.get("support_tests_count")
    extended["se_tight_closes_count"] = se_data.get("tight_closes_count")
    extended["se_bb_width_pctile_252"] = se_data.get("bb_width_pctile_252")
    extended["se_bb_squeeze"] = se_data.get("bb_squeeze")
    extended["se_volume_vs_50d"] = se_data.get("volume_vs_50d")
    extended["se_up_down_volume_ratio_10d"] = se_data.get("up_down_volume_ratio_10d")
    extended["se_quiet_days_10d"] = se_data.get("quiet_days_10d")
    extended["se_rs_line_new_high"] = se_data.get("rs_line_new_high")
    extended["se_pivot_price"] = se_data.get("pivot_price")
    extended["se_setup_ready"] = se_data.get("setup_ready")

    # Additional SE fields for explain drawer
    extended["se_quality_score"] = se_data.get("quality_score")
    extended["se_readiness_score"] = se_data.get("readiness_score")
    extended["se_pattern_confidence"] = se_data.get("pattern_confidence")
    extended["se_pivot_type"] = se_data.get("pivot_type")
    extended["se_pivot_date"] = se_data.get("pivot_date")
    extended["se_timeframe"] = se_data.get("timeframe")
    extended["se_atr14_pct"] = se_data.get("atr14_pct")
    extended["se_explain"] = se_data.get("explain")
    extended["se_candidates"] = se_data.get("candidates")

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


def _to_iso_date(value: Any) -> str | None:
    """Normalize date-like values to YYYY-MM-DD strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)
