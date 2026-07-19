"""Calculate and persist one market's daily Group ranking publication."""

from __future__ import annotations

from datetime import date, datetime
import logging
from typing import Any

from sqlalchemy import and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.industry import IBDGroupRank
from app.services.canonical_group_ranking_service import CanonicalGroupRankingService
from app.services.group_rank_cache_policy import GroupRankCacheRequirement
from app.services.ibd_industry_service import IBDIndustryService
from app.services.legacy_group_rank_contracts import (
    GroupRankPrefetchData,
    IncompleteGroupRankingCacheError,
)
from app.services.legacy_group_rank_data import LegacyGroupRankingEngine

logger = logging.getLogger(__name__)


class MissingIBDIndustryMappingsError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("IBD industry mappings are not loaded")


class GroupRankingCalculationService:
    """Own the canonical/legacy dispatch and legacy daily persistence."""

    def __init__(
        self,
        *,
        ranking_engine: LegacyGroupRankingEngine,
        canonical_group_service: CanonicalGroupRankingService,
        market_rs_repository: MarketRsRunRepository,
    ) -> None:
        self.ranking_engine = ranking_engine
        self.canonical_group_service = canonical_group_service
        self.market_rs_repository = market_rs_repository

    def calculate_and_store(
        self,
        db: Session,
        calculation_date: date | None = None,
        *,
        market: str | None = None,
        cache_only: bool = False,
        cache_requirement: GroupRankCacheRequirement = GroupRankCacheRequirement.disabled(),
        formula_version: str | None = None,
    ) -> list[dict[str, Any]]:
        calculation_date = calculation_date or datetime.now().date()
        normalized_market = (market or "US").upper()
        requested_formula = formula_version or self.market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )
        if requested_formula == BALANCED_RS_FORMULA_VERSION:
            return self.canonical_group_service.calculate_and_store(
                db,
                market=normalized_market,
                as_of_date=calculation_date,
                formula_version=requested_formula,
            )
        if requested_formula != LEGACY_RS_FORMULA_VERSION:
            raise ValueError(f"Unsupported Group RS formula: {requested_formula}")

        started_at = datetime.now()
        groups = IBDIndustryService.get_all_groups(db, market=normalized_market)
        if not groups:
            raise MissingIBDIndustryMappingsError()
        prefetch = self.ranking_engine.prefetch_all_data(
            db,
            market=normalized_market,
            cache_only=cache_only,
        )
        self._require_cache_coverage(prefetch, cache_requirement)
        if prefetch.benchmark_prices is None or prefetch.benchmark_prices.empty:
            logger.error("Failed to get benchmark data for market %s", normalized_market)
            return []

        symbols_by_group = self.ranking_engine.symbols_by_group_for_run(
            db,
            groups,
            prefetch,
            market=normalized_market,
        )
        rs_by_symbol = self.ranking_engine.calculate_rs_by_symbol_for_dates(
            prefetch,
            [calculation_date],
        ).get(calculation_date, {})
        rankings = self.rank_legacy_metrics(
            groups=groups,
            symbols_by_group=symbols_by_group,
            prefetch=prefetch,
            rs_by_symbol=rs_by_symbol,
            calculation_date=calculation_date,
        )
        if not rankings:
            logger.error("No valid group metrics calculated")
            return []
        self.store_legacy_rankings(
            db,
            calculation_date,
            rankings,
            market=normalized_market,
        )
        logger.info(
            "Calculated rankings for %d groups in %.1fs",
            len(rankings),
            (datetime.now() - started_at).total_seconds(),
        )
        return rankings

    @staticmethod
    def _require_cache_coverage(
        prefetch: GroupRankPrefetchData,
        requirement: GroupRankCacheRequirement,
    ) -> None:
        if not requirement.enabled:
            return
        stats = prefetch.stats
        if not stats.get("spy_cached"):
            raise IncompleteGroupRankingCacheError(stats)
        target_symbols = stats.get("target_symbols", 0)
        coverage_ratio = (
            stats.get("symbols_with_prices", 0) / target_symbols
            if target_symbols > 0
            else 1.0
        )
        stats["cache_coverage_ratio"] = coverage_ratio
        stats["cache_coverage_min"] = requirement.min_coverage
        stats["cache_requirement_reason"] = requirement.reason
        if coverage_ratio < requirement.min_coverage:
            raise IncompleteGroupRankingCacheError(stats)
        cache_misses = stats.get("cache_miss_symbols", 0)
        if cache_misses:
            logger.warning(
                "Cache-only Group run has %d misses out of %d symbols "
                "(coverage %.1f%% >= %.1f%%)",
                cache_misses,
                target_symbols,
                coverage_ratio * 100,
                requirement.min_coverage * 100,
            )

    def rank_legacy_metrics(
        self,
        *,
        groups: list[str],
        symbols_by_group: dict[str, list[str]],
        prefetch: GroupRankPrefetchData,
        rs_by_symbol: dict[str, float],
        calculation_date: date,
    ) -> list[dict[str, Any]]:
        rankings: list[dict[str, Any]] = []
        for group_name in groups:
            try:
                metrics = self.ranking_engine.calculate_group_metrics_from_rs(
                    group_name,
                    symbols_by_group.get(group_name, []),
                    rs_by_symbol,
                    prefetch.market_caps,
                    calculation_date,
                )
            except Exception as exc:
                logger.error("Error calculating RS for group %s: %s", group_name, exc)
                continue
            if metrics:
                metrics.update(
                    {
                        "avg_rs_rating_1m": None,
                        "avg_rs_rating_3m": None,
                        "rs_formula_version": LEGACY_RS_FORMULA_VERSION,
                        "market_rs_run_id": None,
                    }
                )
                rankings.append(metrics)
        rankings.sort(key=lambda row: row["avg_rs_rating"], reverse=True)
        for rank, metrics in enumerate(rankings, start=1):
            metrics["rank"] = rank
        return rankings

    def store_legacy_rankings(
        self,
        db: Session,
        calculation_date: date,
        rankings: list[dict[str, Any]],
        *,
        market: str,
    ) -> None:
        try:
            values = [
                self._ranking_values(calculation_date, row, market=market)
                for row in rankings
            ]
            if not values:
                db.commit()
                return
            bind = db.get_bind()
            if bind is not None and bind.dialect.name == "postgresql":
                stmt = pg_insert(IBDGroupRank).values(values)
                db.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[
                            "industry_group",
                            "date",
                            "market",
                            "rs_formula_version",
                        ],
                        set_={
                            "rank": stmt.excluded.rank,
                            "avg_rs_rating": stmt.excluded.avg_rs_rating,
                            "median_rs_rating": stmt.excluded.median_rs_rating,
                            "weighted_avg_rs_rating": stmt.excluded.weighted_avg_rs_rating,
                            "rs_std_dev": stmt.excluded.rs_std_dev,
                            "num_stocks": stmt.excluded.num_stocks,
                            "num_stocks_rs_above_80": stmt.excluded.num_stocks_rs_above_80,
                            "top_symbol": stmt.excluded.top_symbol,
                            "top_rs_rating": stmt.excluded.top_rs_rating,
                            "avg_rs_rating_1m": stmt.excluded.avg_rs_rating_1m,
                            "avg_rs_rating_3m": stmt.excluded.avg_rs_rating_3m,
                            "market_rs_run_id": stmt.excluded.market_rs_run_id,
                        },
                    )
                )
            else:
                self._store_sqlalchemy_fallback(
                    db,
                    calculation_date,
                    values,
                    market=market,
                )
            db.commit()
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def _ranking_values(
        calculation_date: date,
        metrics: dict[str, Any],
        *,
        market: str,
    ) -> dict[str, Any]:
        return {
            "market": market,
            "industry_group": metrics["industry_group"],
            "date": calculation_date,
            "rank": metrics["rank"],
            "avg_rs_rating": metrics["avg_rs_rating"],
            "avg_rs_rating_1m": metrics.get("avg_rs_rating_1m"),
            "avg_rs_rating_3m": metrics.get("avg_rs_rating_3m"),
            "median_rs_rating": metrics.get("median_rs_rating"),
            "weighted_avg_rs_rating": metrics.get("weighted_avg_rs_rating"),
            "rs_std_dev": metrics.get("rs_std_dev"),
            "num_stocks": metrics["num_stocks"],
            "num_stocks_rs_above_80": metrics["num_stocks_rs_above_80"],
            "top_symbol": metrics["top_symbol"],
            "top_rs_rating": metrics["top_rs_rating"],
            "rs_formula_version": LEGACY_RS_FORMULA_VERSION,
            "market_rs_run_id": None,
        }

    @staticmethod
    def _store_sqlalchemy_fallback(
        db: Session,
        calculation_date: date,
        values: list[dict[str, Any]],
        *,
        market: str,
    ) -> None:
        group_names = [value["industry_group"] for value in values]
        existing = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.industry_group.in_(group_names),
                IBDGroupRank.date == calculation_date,
                IBDGroupRank.market == market,
                IBDGroupRank.rs_formula_version == LEGACY_RS_FORMULA_VERSION,
            )
        ).all()
        existing_by_group = {row.industry_group: row for row in existing}
        for value in values:
            row = existing_by_group.get(value["industry_group"])
            if row is None:
                db.add(IBDGroupRank(**value))
                continue
            for field, field_value in value.items():
                if field not in {"market", "industry_group", "date", "rs_formula_version"}:
                    setattr(row, field, field_value)


__all__ = [
    "GroupRankingCalculationService",
    "MissingIBDIndustryMappingsError",
]
