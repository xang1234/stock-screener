"""SQLAlchemy implementation of FeatureStoreRepository."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.query import FilterSpec, PageSpec, QuerySpec, SortSpec
from app.domain.feature_store.models import (
    INT_TO_RATING,
    FeaturePage,
    FeatureRow,
    FeatureRowWrite,
)
from app.domain.feature_store.ports import FeatureStoreRepository
from app.domain.feature_store.quality import DQInputs
from app.domain.scanning.models import FilterOptions, ResultPage, ScanResultItemDomain
from app.infra.db.models.feature_store import (
    FeatureRun,
    FeatureRunPointer,
    FeatureRunUniverseSymbol,
    StockFeatureDaily,
)
from app.infra.query.feature_store_query import (
    apply_filters,
    apply_sort_all,
    apply_sort_and_paginate,
)
from app.models.stock_universe import StockUniverse

_BATCH_SIZE = 500


class SqlFeatureStoreRepository(FeatureStoreRepository):
    """Persist and query per-symbol feature rows via SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_snapshot_rows(
        self,
        run_id: int,
        rows: Sequence[FeatureRowWrite],
    ) -> int:
        if not rows:
            return 0

        count = 0
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            values = [
                {
                    "run_id": run_id,
                    "symbol": row.symbol,
                    "as_of_date": row.as_of_date,
                    "composite_score": row.composite_score,
                    "overall_rating": row.overall_rating,
                    "passes_count": row.passes_count,
                    "details_json": row.details,
                }
                for row in batch
            ]
            stmt = sqlite_insert(StockFeatureDaily).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id", "symbol"],
                set_={
                    "as_of_date": stmt.excluded.as_of_date,
                    "composite_score": stmt.excluded.composite_score,
                    "overall_rating": stmt.excluded.overall_rating,
                    "passes_count": stmt.excluded.passes_count,
                    "details_json": stmt.excluded.details_json,
                },
            )
            self._session.execute(stmt)
            count += len(batch)

        self._session.flush()
        return count

    def save_run_universe_symbols(
        self,
        run_id: int,
        symbols: Sequence[str],
    ) -> None:
        if not symbols:
            return

        for i in range(0, len(symbols), _BATCH_SIZE):
            batch = symbols[i : i + _BATCH_SIZE]
            objects = [
                FeatureRunUniverseSymbol(run_id=run_id, symbol=s) for s in batch
            ]
            self._session.bulk_save_objects(objects)

        self._session.flush()

    def count_by_run_id(self, run_id: int) -> int:
        return (
            self._session.query(func.count(StockFeatureDaily.symbol))
            .filter(StockFeatureDaily.run_id == run_id)
            .scalar()
            or 0
        )

    def query_latest(
        self,
        filters=None,
        sort=None,
        page=None,
    ) -> FeaturePage:
        p = page or PageSpec()
        pointer = (
            self._session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        if pointer is None:
            return FeaturePage(items=(), total=0, page=p.page, per_page=p.per_page)

        return self._query_features(
            pointer.run_id,
            filters or FilterSpec(),
            sort or SortSpec(),
            p,
        )

    def query_run(
        self,
        run_id: int,
        filters=None,
        sort=None,
        page=None,
    ) -> FeaturePage:
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        return self._query_features(
            run_id,
            filters or FilterSpec(),
            sort or SortSpec(),
            page or PageSpec(),
        )

    def get_run_dq_inputs(self, run_id: int) -> DQInputs:
        # Feature rows for this run
        rows = (
            self._session.query(
                StockFeatureDaily.symbol,
                StockFeatureDaily.composite_score,
                StockFeatureDaily.overall_rating,
            )
            .filter(StockFeatureDaily.run_id == run_id)
            .all()
        )

        # Universe symbols for this run
        universe_rows = (
            self._session.query(FeatureRunUniverseSymbol.symbol)
            .filter(FeatureRunUniverseSymbol.run_id == run_id)
            .all()
        )
        universe_symbols = tuple(r.symbol for r in universe_rows)

        result_symbols = tuple(r.symbol for r in rows)
        scores = tuple(r.composite_score for r in rows if r.composite_score is not None)
        ratings = tuple(r.overall_rating for r in rows if r.overall_rating is not None)
        null_count = sum(1 for r in rows if r.composite_score is None)

        return DQInputs(
            expected_row_count=len(universe_symbols),
            actual_row_count=len(rows),
            null_score_count=null_count,
            total_row_count=len(rows),
            scores=scores,
            ratings=ratings,
            universe_symbols=universe_symbols,
            result_symbols=result_symbols,
        )

    # -- Private helpers ---------------------------------------------------

    def _query_features(
        self,
        run_id: int,
        filters: FilterSpec,
        sort: SortSpec,
        page: PageSpec,
    ) -> FeaturePage:
        """Build and execute a filtered, sorted, paginated feature query."""
        q = (
            self._session.query(StockFeatureDaily)
            .filter(StockFeatureDaily.run_id == run_id)
        )
        q = apply_filters(q, filters)
        rows, total = apply_sort_and_paginate(q, sort, page)

        items = tuple(self._to_domain_row(r) for r in rows)
        return FeaturePage(
            items=items,
            total=total,
            page=page.page,
            per_page=page.per_page,
        )

    def query_run_as_scan_results(
        self,
        run_id: int,
        spec: QuerySpec,
        *,
        include_sparklines: bool = True,
        include_setup_payload: bool = True,
    ) -> ResultPage:
        """Query feature store and return results as ScanResultItemDomain.

        Bridge method for dual-source queries — allows GetScanResultsUseCase
        to read from the feature store while returning the same domain type
        as the legacy scan_results path.
        """
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        q = (
            self._session.query(StockFeatureDaily, StockUniverse.name)
            .outerjoin(
                StockUniverse,
                StockFeatureDaily.symbol == StockUniverse.symbol,
            )
            .filter(StockFeatureDaily.run_id == run_id)
        )
        q = apply_filters(q, spec.filters)
        rows, total = apply_sort_and_paginate(q, spec.sort, spec.page)

        items = tuple(
            _map_feature_to_scan_result(
                row,
                company_name,
                include_sparklines,
                include_setup_payload=include_setup_payload,
            )
            for row, company_name in rows
        )
        return ResultPage(
            items=items,
            total=total,
            page=spec.page.page,
            per_page=spec.page.per_page,
        )

    def get_filter_options_for_run(self, run_id: int) -> FilterOptions:
        """Return distinct filter dropdown values from the feature store.

        Bridge method for dual-source queries — matches the return type
        of ``ScanResultRepository.get_filter_options()``.
        """
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        # Ratings from indexed integer column (fast)
        rating_rows = (
            self._session.query(StockFeatureDaily.overall_rating)
            .filter(
                StockFeatureDaily.run_id == run_id,
                StockFeatureDaily.overall_rating.isnot(None),
            )
            .distinct()
            .all()
        )
        ratings = tuple(sorted(
            INT_TO_RATING[r.overall_rating]
            for r in rating_rows
            if r.overall_rating in INT_TO_RATING
        ))

        # Industries and sectors from JSON
        industry_col = func.json_extract(
            StockFeatureDaily.details_json, "$.ibd_industry_group"
        )
        industry_rows = (
            self._session.query(industry_col)
            .filter(
                StockFeatureDaily.run_id == run_id,
                industry_col.isnot(None),
                industry_col != "",
            )
            .distinct()
            .all()
        )
        industries = tuple(sorted(r[0] for r in industry_rows))

        sector_col = func.json_extract(
            StockFeatureDaily.details_json, "$.gics_sector"
        )
        sector_rows = (
            self._session.query(sector_col)
            .filter(
                StockFeatureDaily.run_id == run_id,
                sector_col.isnot(None),
                sector_col != "",
            )
            .distinct()
            .all()
        )
        sectors = tuple(sorted(r[0] for r in sector_rows))

        return FilterOptions(
            ibd_industries=industries,
            gics_sectors=sectors,
            ratings=ratings,
        )

    def get_by_symbol_for_run(
        self,
        run_id: int,
        symbol: str,
        *,
        include_sparklines: bool = True,
        include_setup_payload: bool = True,
    ) -> ScanResultItemDomain | None:
        """Look up a single symbol in a feature run.

        Bridge method for dual-source queries — matches the return type
        of ``ScanResultRepository.get_by_symbol()``.
        Returns None if symbol not found; raises EntityNotFoundError if
        the run itself doesn't exist.
        """
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        result = (
            self._session.query(StockFeatureDaily, StockUniverse.name)
            .outerjoin(
                StockUniverse,
                StockFeatureDaily.symbol == StockUniverse.symbol,
            )
            .filter(
                StockFeatureDaily.run_id == run_id,
                StockFeatureDaily.symbol == symbol,
            )
            .first()
        )
        if result is None:
            return None

        row, company_name = result
        return _map_feature_to_scan_result(
            row,
            company_name,
            include_sparklines,
            include_setup_payload=include_setup_payload,
        )

    def get_peers_by_industry_for_run(
        self,
        run_id: int,
        ibd_industry_group: str,
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return all stocks in the same IBD industry group for a feature run.

        Bridge method for dual-source queries — matches the return type
        of ``ScanResultRepository.get_peers_by_industry()``.
        """
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        industry_col = func.json_extract(
            StockFeatureDaily.details_json, "$.ibd_industry_group"
        )
        rows = (
            self._session.query(StockFeatureDaily, StockUniverse.name)
            .outerjoin(
                StockUniverse,
                StockFeatureDaily.symbol == StockUniverse.symbol,
            )
            .filter(
                StockFeatureDaily.run_id == run_id,
                industry_col == ibd_industry_group,
            )
            .order_by(StockFeatureDaily.composite_score.desc())
            .all()
        )
        return tuple(
            _map_feature_to_scan_result(row, company_name, include_sparklines=False)
            for row, company_name in rows
        )

    def get_peers_by_sector_for_run(
        self,
        run_id: int,
        gics_sector: str,
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return all stocks in the same GICS sector for a feature run.

        Bridge method for dual-source queries — matches the return type
        of ``ScanResultRepository.get_peers_by_sector()``.
        """
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        sector_col = func.json_extract(
            StockFeatureDaily.details_json, "$.gics_sector"
        )
        rows = (
            self._session.query(StockFeatureDaily, StockUniverse.name)
            .outerjoin(
                StockUniverse,
                StockFeatureDaily.symbol == StockUniverse.symbol,
            )
            .filter(
                StockFeatureDaily.run_id == run_id,
                sector_col == gics_sector,
            )
            .order_by(StockFeatureDaily.composite_score.desc())
            .all()
        )
        return tuple(
            _map_feature_to_scan_result(row, company_name, include_sparklines=False)
            for row, company_name in rows
        )

    def query_all_as_scan_results(
        self,
        run_id: int,
        filters: FilterSpec,
        sort: SortSpec,
        *,
        include_sparklines: bool = False,
    ) -> tuple[ScanResultItemDomain, ...]:
        """Query feature store and return ALL results (no pagination).

        Bridge method for dual-source export queries — matches the return
        type of ``ScanResultRepository.query_all()``.
        """
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        q = (
            self._session.query(StockFeatureDaily, StockUniverse.name)
            .outerjoin(
                StockUniverse,
                StockFeatureDaily.symbol == StockUniverse.symbol,
            )
            .filter(StockFeatureDaily.run_id == run_id)
        )
        q = apply_filters(q, filters)
        rows = apply_sort_all(q, sort)

        return tuple(
            _map_feature_to_scan_result(row, company_name, include_sparklines)
            for row, company_name in rows
        )

    def query_run_symbols(
        self,
        run_id: int,
        filters: FilterSpec,
        sort: SortSpec,
        page: PageSpec | None = None,
    ) -> tuple[tuple[str, ...], int]:
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        q = (
            self._session.query(StockFeatureDaily.symbol)
            .filter(StockFeatureDaily.run_id == run_id)
        )
        q = apply_filters(q, filters)

        if page is None:
            rows = apply_sort_all(q, sort)
            symbols = tuple(symbol for (symbol,) in rows)
            return symbols, len(symbols)

        rows, total = apply_sort_and_paginate(q, sort, page)
        symbols = tuple(symbol for (symbol,) in rows)
        return symbols, total

    def get_setup_payload_for_run(self, run_id: int, symbol: str) -> dict | None:
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        row = (
            self._session.query(StockFeatureDaily.details_json)
            .filter(
                StockFeatureDaily.run_id == run_id,
                StockFeatureDaily.symbol == symbol,
            )
            .first()
        )
        if not row:
            return None

        details = row[0] if isinstance(row[0], dict) else {}
        se_data = details.get("setup_engine") if isinstance(details, dict) else None
        se_data = se_data if isinstance(se_data, dict) else {}
        return {
            "se_explain": se_data.get("explain"),
            "se_candidates": se_data.get("candidates"),
        }

    def get_row_by_symbol(self, run_id: int, symbol: str) -> FeatureRow | None:
        row = (
            self._session.query(StockFeatureDaily)
            .filter(
                StockFeatureDaily.run_id == run_id,
                func.upper(StockFeatureDaily.symbol) == symbol.upper(),
            )
            .first()
        )
        if row is None:
            return None
        return self._to_domain_row(row)

    def get_scores_for_run(
        self, run_id: int
    ) -> dict[str, tuple[float | None, int | None]]:
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        rows = (
            self._session.query(
                StockFeatureDaily.symbol,
                StockFeatureDaily.composite_score,
                StockFeatureDaily.overall_rating,
            )
            .filter(StockFeatureDaily.run_id == run_id)
            .all()
        )
        return {r.symbol: (r.composite_score, r.overall_rating) for r in rows}

    @staticmethod
    def _to_domain_row(row: StockFeatureDaily) -> FeatureRow:
        """Map ORM model to domain value object."""
        return FeatureRow(
            run_id=row.run_id,
            symbol=row.symbol,
            as_of_date=row.as_of_date,
            composite_score=row.composite_score,
            overall_rating=row.overall_rating,
            passes_count=row.passes_count,
            details=row.details_json,
        )


# ---------------------------------------------------------------------------
# Bridge mapper: StockFeatureDaily → ScanResultItemDomain
# ---------------------------------------------------------------------------


def _map_feature_to_scan_result(
    row: StockFeatureDaily,
    company_name: str | None,
    include_sparklines: bool,
    *,
    include_setup_payload: bool = True,
) -> ScanResultItemDomain:
    """Map a feature store ORM row to a ScanResultItemDomain.

    Analogous to ``_map_row_to_domain`` in scan_result_repo.py, but
    extracts all fields from the details_json blob rather than from
    dedicated SQL columns.
    """
    d: dict[str, Any] = row.details_json or {}

    # Clamp score to 0-100 (matching legacy behavior)
    raw_score = row.composite_score or 0
    clamped_score = max(0.0, min(100.0, float(raw_score)))

    # Reverse-map integer rating back to string
    rating = INT_TO_RATING.get(row.overall_rating, d.get("rating", "Pass"))

    extended: dict[str, Any] = {
        "company_name": company_name,
        "minervini_score": d.get("minervini_score"),
        "canslim_score": d.get("canslim_score"),
        "ipo_score": d.get("ipo_score"),
        "custom_score": d.get("custom_score"),
        "volume_breakthrough_score": d.get("volume_breakthrough_score"),
        "rs_rating": d.get("rs_rating"),
        "rs_rating_1m": d.get("rs_rating_1m"),
        "rs_rating_3m": d.get("rs_rating_3m"),
        "rs_rating_12m": d.get("rs_rating_12m"),
        "stage": d.get("stage"),
        "stage_name": d.get("stage_name"),
        "volume": d.get("avg_dollar_volume"),
        "market_cap": d.get("market_cap"),
        "ma_alignment": d.get("ma_alignment"),
        "vcp_detected": d.get("vcp_detected"),
        "vcp_score": d.get("vcp_score"),
        "vcp_pivot": d.get("vcp_pivot"),
        "vcp_ready_for_breakout": d.get("vcp_ready_for_breakout"),
        "vcp_contraction_ratio": d.get("vcp_contraction_ratio"),
        "vcp_atr_score": d.get("vcp_atr_score"),
        "passes_template": d.get("passes_template", False),
        "adr_percent": d.get("adr_percent"),
        "eps_growth_qq": d.get("eps_growth_qq"),
        "sales_growth_qq": d.get("sales_growth_qq"),
        "eps_growth_yy": d.get("eps_growth_yy"),
        "sales_growth_yy": d.get("sales_growth_yy"),
        "peg_ratio": d.get("peg_ratio"),
        "eps_rating": d.get("eps_rating"),
        "ibd_industry_group": d.get("ibd_industry_group"),
        "ibd_group_rank": d.get("ibd_group_rank"),
        "gics_sector": d.get("gics_sector"),
        "gics_industry": d.get("gics_industry"),
        "rs_sparkline_data": d.get("rs_sparkline_data") if include_sparklines else None,
        "rs_trend": d.get("rs_trend"),
        "price_sparkline_data": d.get("price_sparkline_data") if include_sparklines else None,
        "price_change_1d": d.get("price_change_1d"),
        "price_trend": d.get("price_trend"),
        "ipo_date": d.get("ipo_date"),
        "beta": d.get("beta"),
        "beta_adj_rs": d.get("beta_adj_rs"),
        "beta_adj_rs_1m": d.get("beta_adj_rs_1m"),
        "beta_adj_rs_3m": d.get("beta_adj_rs_3m"),
        "beta_adj_rs_12m": d.get("beta_adj_rs_12m"),
        "perf_week": d.get("perf_week"),
        "perf_month": d.get("perf_month"),
        "perf_3m": d.get("perf_3m"),
        "perf_6m": d.get("perf_6m"),
        "gap_percent": d.get("gap_percent"),
        "volume_surge": d.get("volume_surge"),
        "ema_10_distance": d.get("ema_10_distance"),
        "ema_20_distance": d.get("ema_20_distance"),
        "ema_50_distance": d.get("ema_50_distance"),
        "week_52_high_distance": d.get("from_52w_high_pct"),
        "week_52_low_distance": d.get("above_52w_low_pct"),
    }

    # Setup Engine fields (extracted from details JSON blob)
    _se = d.get("setup_engine", {}) or {}
    extended["se_setup_score"] = _se.get("setup_score")
    extended["se_pattern_primary"] = _se.get("pattern_primary")
    extended["se_distance_to_pivot_pct"] = _se.get("distance_to_pivot_pct")
    extended["se_in_early_zone"] = _se.get("in_early_zone")
    extended["se_extended_from_pivot"] = _se.get("extended_from_pivot")
    extended["se_base_length_weeks"] = _se.get("base_length_weeks")
    extended["se_base_depth_pct"] = _se.get("base_depth_pct")
    extended["se_support_tests_count"] = _se.get("support_tests_count")
    extended["se_tight_closes_count"] = _se.get("tight_closes_count")
    extended["se_bb_width_pctile_252"] = _se.get("bb_width_pctile_252")
    extended["se_bb_squeeze"] = _se.get("bb_squeeze")
    extended["se_volume_vs_50d"] = _se.get("volume_vs_50d")
    extended["se_up_down_volume_ratio_10d"] = _se.get("up_down_volume_ratio_10d")
    extended["se_quiet_days_10d"] = _se.get("quiet_days_10d")
    extended["se_rs_line_new_high"] = _se.get("rs_line_new_high")
    extended["se_pivot_price"] = _se.get("pivot_price")
    extended["se_setup_ready"] = _se.get("setup_ready")
    extended["se_quality_score"] = _se.get("quality_score")
    extended["se_readiness_score"] = _se.get("readiness_score")
    extended["se_pattern_confidence"] = _se.get("pattern_confidence")
    extended["se_pivot_type"] = _se.get("pivot_type")
    extended["se_pivot_date"] = _se.get("pivot_date")
    extended["se_timeframe"] = _se.get("timeframe")
    extended["se_atr14_pct"] = _se.get("atr14_pct")
    if include_setup_payload:
        extended["se_explain"] = _se.get("explain")
        extended["se_candidates"] = _se.get("candidates")

    return ScanResultItemDomain(
        symbol=row.symbol,
        composite_score=clamped_score,
        rating=rating,
        current_price=d.get("current_price"),
        screener_outputs={},
        screeners_run=d.get("screeners_run", []),
        composite_method=d.get("composite_method", "weighted_average"),
        screeners_passed=d.get("screeners_passed", 0),
        screeners_total=d.get("screeners_total", 0),
        extended_fields=extended,
    )
