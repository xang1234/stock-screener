"""
Service for calculating and managing IBD Industry Group Rankings.

Calculates daily rankings based on average RS rating of constituent stocks.
"""
import logging
import statistics
from typing import Any, Optional, Dict, List
from datetime import datetime, date, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..models.industry import IBDGroupRank
from ..models.stock_universe import StockUniverse
from ..domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)
from ..infra.db.repositories.market_rs_repo import MarketRsRunRepository
from .canonical_group_ranking_service import CanonicalGroupRankingService
from .group_constituent_source import GroupConstituentSource
from .group_detail_payloads import constituent_stock_payloads_from_scan_items
from .group_ranking_history import build_group_detail_payload_from_parts
from .group_rank_cache_policy import GroupRankCacheRequirement
from .group_ranking_payloads import rank_record_payload
from .group_rank_snapshot_reader import GroupRankSnapshotReader
from .ibd_industry_service import IBDIndustryService
from .legacy_group_rank_backfill import LegacyGroupRankBackfillMixin
from .legacy_group_rank_contracts import (
    CACHE_MISS_TOLERANCE_RATIO as CACHE_MISS_TOLERANCE_RATIO,
    GroupRankPrefetchData as GroupRankPrefetchData,
    IncompleteGroupRankingCacheError,
)
from .legacy_group_rank_data import LegacyGroupRankDataMixin
from .price_cache_service import PriceCacheService
from .benchmark_cache_service import BenchmarkCacheService
from ..scanners.criteria.relative_strength import RelativeStrengthCalculator

logger = logging.getLogger(__name__)
GROUP_RANK_CHANGE_CALENDAR_DAYS = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
}


class MissingIBDIndustryMappingsError(RuntimeError):
    """Raised when tracked IBD industry mappings have not been loaded."""

    def __init__(self) -> None:
        super().__init__("IBD industry mappings are not loaded")


class IBDGroupRankService(
    LegacyGroupRankBackfillMixin,
    LegacyGroupRankDataMixin,
):
    """
    Service for calculating and managing IBD Industry Group Rankings.

    Methodology:
    1. For each IBD industry group, get all constituent stock symbols
    2. Calculate RS rating for each stock using RelativeStrengthCalculator
    3. Compute average RS for the group (excluding stocks with insufficient data)
    4. Rank all groups from highest to lowest avg_rs
    5. Store daily snapshot in IBDGroupRank table
    """

    def __init__(
        self,
        *,
        price_cache: PriceCacheService,
        benchmark_cache: BenchmarkCacheService,
        rs_calculator: RelativeStrengthCalculator | None = None,
        group_constituent_source: GroupConstituentSource | None = None,
        canonical_group_service: CanonicalGroupRankingService,
        market_rs_repository: MarketRsRunRepository,
    ):
        """Initialize the service."""
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.rs_calculator = rs_calculator or RelativeStrengthCalculator()
        self.group_constituent_source = (
            group_constituent_source or GroupConstituentSource()
        )
        self.canonical_group_service = canonical_group_service
        self.market_rs_repository = market_rs_repository

    def calculate_group_rankings(
        self,
        db: Session,
        calculation_date: date = None,
        *,
        market: str | None = None,
        cache_only: bool = False,
        cache_requirement: GroupRankCacheRequirement = GroupRankCacheRequirement.disabled(),
        formula_version: str | None = None,
    ) -> List[Dict]:
        """
        Calculate and store rankings for all IBD groups for a given date.

        Args:
            db: Database session
            calculation_date: Date to calculate rankings for (default: today)

        Returns:
            List of ranked groups with metrics
        """
        if calculation_date is None:
            calculation_date = datetime.now().date()

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
        logger.info(
            "Calculating industry group rankings for market=%s date=%s",
            normalized_market, calculation_date,
        )
        start_time = datetime.now()

        # Get all unique industry groups for this market
        all_groups = IBDIndustryService.get_all_groups(db, market=normalized_market)
        if not all_groups:
            logger.error("No industry groups found for market %s", normalized_market)
            raise MissingIBDIndustryMappingsError()

        logger.info(
            "Found %d industry groups for market %s", len(all_groups), normalized_market,
        )

        # Pre-fetch all data upfront (benchmark + all stock prices).
        prefetch = self._coerce_prefetch_data(
            self._prefetch_all_data(
                db,
                market=normalized_market,
                cache_only=cache_only,
            )
        )
        prefetch_stats = prefetch.stats

        if cache_requirement.enabled:
            if not prefetch_stats.get("spy_cached"):
                raise IncompleteGroupRankingCacheError(prefetch_stats)
            cache_miss_symbols = prefetch_stats.get("cache_miss_symbols", 0)
            target_symbols = prefetch_stats.get("target_symbols", 0)
            coverage_ratio = (
                prefetch_stats.get("symbols_with_prices", 0) / target_symbols
                if target_symbols > 0
                else 1.0
            )
            prefetch_stats["cache_coverage_ratio"] = coverage_ratio
            prefetch_stats["cache_coverage_min"] = cache_requirement.min_coverage
            prefetch_stats["cache_requirement_reason"] = cache_requirement.reason
            if coverage_ratio < cache_requirement.min_coverage:
                raise IncompleteGroupRankingCacheError(prefetch_stats)
            if cache_miss_symbols > 0:
                logger.warning(
                    "Cache-only group ranking run has %d cache misses out of %d symbols "
                    "(coverage %.1f%% >= %.1f%%)",
                    cache_miss_symbols,
                    target_symbols,
                    coverage_ratio * 100,
                    cache_requirement.min_coverage * 100,
                )

        if prefetch.benchmark_prices is None or prefetch.benchmark_prices.empty:
            logger.error(
                "Failed to get benchmark data for market %s", normalized_market,
            )
            return []

        rs_by_date = self._calculate_rs_by_symbol_for_dates(
            prefetch,
            [calculation_date],
        )
        rs_by_symbol = rs_by_date.get(calculation_date, {})

        # Calculate RS for each group using pre-fetched cached data
        group_metrics = []
        symbols_by_group = self._symbols_by_group_for_run(
            db,
            all_groups,
            prefetch,
            market=normalized_market,
        )

        for group_name in all_groups:
            try:
                metrics = self._calculate_group_metrics_from_rs(
                    group_name,
                    symbols_by_group.get(group_name, []),
                    rs_by_symbol,
                    prefetch.market_caps,
                    calculation_date,
                )
                if metrics:
                    metrics.update(
                        {
                            "avg_rs_rating_1m": None,
                            "avg_rs_rating_3m": None,
                            "rs_formula_version": LEGACY_RS_FORMULA_VERSION,
                            "market_rs_run_id": None,
                        }
                    )
                    group_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error calculating RS for group {group_name}: {e}")
                continue

        if not group_metrics:
            logger.error("No valid group metrics calculated")
            return []

        # Sort by avg_rs (descending) and assign ranks
        group_metrics.sort(key=lambda x: x['avg_rs_rating'], reverse=True)

        for rank, metrics in enumerate(group_metrics, start=1):
            metrics['rank'] = rank

        # Store in database
        self._store_rankings(
            db,
            calculation_date,
            group_metrics,
            market=normalized_market,
            formula_version=requested_formula,
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Calculated rankings for {len(group_metrics)} groups in {duration:.1f}s"
        )

        return group_metrics

    def _calculate_group_rs(
        self,
        db: Session,
        group_name: str,
        spy_prices: pd.Series,
        calculation_date: date
    ) -> Optional[Dict]:
        """
        Calculate average RS rating for a single industry group.

        Args:
            db: Database session
            group_name: IBD industry group name
            spy_prices: SPY price series (most recent first)
            calculation_date: Date for calculation

        Returns:
            Dict with group metrics or None if insufficient data
        """
        # Get all symbols in this group
        symbols = IBDIndustryService.get_group_symbols(db, group_name)

        if not symbols:
            logger.debug(f"No symbols found for group: {group_name}")
            return None

        # Fetch prices for all symbols in batch
        prices_dict = self.price_cache.get_many(symbols, period="2y")

        # Market caps for weighted RS
        market_caps = self._get_market_caps_for_symbols(db, symbols)

        # Calculate RS for each symbol
        rs_ratings = []
        top_symbol = None
        top_rs = -1
        rs_above_80_count = 0
        weighted_sum = 0.0
        weighted_total = 0.0

        for symbol in symbols:
            prices = prices_dict.get(symbol)
            if prices is None or prices.empty:
                continue

            # Filter to calculation_date for accurate historical rankings
            if calculation_date < datetime.now().date():
                prices_filtered = prices[prices.index.date <= calculation_date]
            else:
                prices_filtered = prices

            # Skip if insufficient data after filtering
            if prices_filtered.empty:
                continue

            # Prepare stock prices (most recent first)
            stock_prices = prices_filtered['Close'].sort_index(ascending=False)

            # Need at least 252 days for full RS calculation
            if len(stock_prices) < 252:
                continue

            try:
                rs_result = self.rs_calculator.calculate_rs_rating(
                    symbol, stock_prices, spy_prices
                )
                rs_rating = rs_result.get('rs_rating', 0)

                if rs_rating > 0:
                    rs_ratings.append(rs_rating)

                    if rs_rating > top_rs:
                        top_rs = rs_rating
                        top_symbol = symbol

                    if rs_rating >= 80:
                        rs_above_80_count += 1

                    market_cap = market_caps.get(symbol)
                    if market_cap:
                        weighted_sum += rs_rating * market_cap
                        weighted_total += market_cap

            except Exception as e:
                logger.debug(f"Error calculating RS for {symbol}: {e}")
                continue

        # Need at least 3 stocks with valid RS to rank the group
        if len(rs_ratings) < 3:
            logger.debug(
                f"Insufficient data for group {group_name}: "
                f"only {len(rs_ratings)} stocks with valid RS"
            )
            return None

        # Calculate average RS
        avg_rs = sum(rs_ratings) / len(rs_ratings)
        median_rs = statistics.median(rs_ratings)
        rs_std_dev = statistics.pstdev(rs_ratings) if len(rs_ratings) > 1 else None
        weighted_avg_rs = (weighted_sum / weighted_total) if weighted_total > 0 else None

        return {
            'industry_group': group_name,
            'date': calculation_date,
            'avg_rs_rating': round(avg_rs, 2),
            'median_rs_rating': round(median_rs, 2),
            'weighted_avg_rs_rating': round(weighted_avg_rs, 2) if weighted_avg_rs is not None else None,
            'rs_std_dev': round(rs_std_dev, 2) if rs_std_dev is not None else None,
            'num_stocks': len(rs_ratings),
            'num_stocks_rs_above_80': rs_above_80_count,
            'top_symbol': top_symbol,
            'top_rs_rating': round(top_rs, 2) if top_rs > 0 else None,
        }

    def _store_rankings(
        self,
        db: Session,
        calculation_date: date,
        group_metrics: List[Dict],
        *,
        market: str = "US",
        formula_version: str = LEGACY_RS_FORMULA_VERSION,
    ) -> None:
        """
        Store group rankings in database.

        Upserts records (updates if exists, inserts if new). Unique key is
        ``(industry_group, date, market, rs_formula_version)``.
        """
        try:
            values = [
                self._ranking_values(
                    calculation_date,
                    metrics,
                    market=market,
                    formula_version=formula_version,
                )
                for metrics in group_metrics
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
                self._store_rankings_sqlalchemy_fallback(
                    db,
                    calculation_date,
                    values,
                    market=market,
                    formula_version=formula_version,
                )

            db.commit()
            logger.info(
                "Stored %d group rankings for market=%s date=%s",
                len(group_metrics), market, calculation_date,
            )

        except Exception as e:
            logger.error(f"Error storing rankings: {e}", exc_info=True)
            db.rollback()
            raise

    @staticmethod
    def _ranking_values(
        calculation_date: date,
        metrics: Dict,
        *,
        market: str,
        formula_version: str,
    ) -> Dict[str, Any]:
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
            "rs_formula_version": formula_version,
            "market_rs_run_id": metrics.get("market_rs_run_id"),
        }

    def _store_rankings_sqlalchemy_fallback(
        self,
        db: Session,
        calculation_date: date,
        values: List[Dict[str, Any]],
        *,
        market: str,
        formula_version: str,
    ) -> None:
        """SQLite-compatible bulk upsert fallback for tests/local tools."""
        group_names = [value["industry_group"] for value in values]
        existing_records = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.industry_group.in_(group_names),
                IBDGroupRank.date == calculation_date,
                IBDGroupRank.market == market,
                IBDGroupRank.rs_formula_version == formula_version,
            )
        ).all()
        existing_by_group = {record.industry_group: record for record in existing_records}

        for value in values:
            existing = existing_by_group.get(value["industry_group"])
            if existing:
                existing.rank = value["rank"]
                existing.avg_rs_rating = value["avg_rs_rating"]
                existing.avg_rs_rating_1m = value.get("avg_rs_rating_1m")
                existing.avg_rs_rating_3m = value.get("avg_rs_rating_3m")
                existing.num_stocks = value["num_stocks"]
                existing.num_stocks_rs_above_80 = value["num_stocks_rs_above_80"]
                existing.top_symbol = value["top_symbol"]
                existing.top_rs_rating = value["top_rs_rating"]
                existing.median_rs_rating = value.get("median_rs_rating")
                existing.weighted_avg_rs_rating = value.get("weighted_avg_rs_rating")
                existing.rs_std_dev = value.get("rs_std_dev")
                existing.market_rs_run_id = value.get("market_rs_run_id")
            else:
                db.add(IBDGroupRank(**value))

    def get_current_rankings(
        self,
        db: Session,
        limit: int = 197,
        calculation_date: date | None = None,
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> List[Dict]:
        """
        Get most recent group rankings with rank changes for one market.

        Rankings are partitioned by market — each market maintains its own
        rank ordering (both US and HK can have a rank #1), so every query
        must filter by market to avoid mixing them.
        """
        normalized_market = (market or "US").upper()
        resolved_formula = formula_version or self.market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )
        latest_date = calculation_date or db.query(func.max(IBDGroupRank.date)).filter(
            IBDGroupRank.market == normalized_market,
            IBDGroupRank.rs_formula_version == resolved_formula,
        ).scalar()
        if latest_date is None:
            return []
        rankings = GroupRankSnapshotReader().load_exact(
            db,
            identity=GroupSnapshotIdentity(
                normalized_market,
                latest_date,
                resolved_formula,
            ),
        )
        if not rankings:
            return []
        rankings = rankings[:limit]

        # Calendar-day target offsets for rank changes. The lookup below picks
        # the closest stored ranking within a small window; these are not exact
        # trading-session offsets.
        period_days = dict(GROUP_RANK_CHANGE_CALENDAR_DAYS)

        # Batch fetch all historical ranks in ONE query instead of 197*4=788 queries
        group_names = [str(row["industry_group"]) for row in rankings]
        historical_ranks = self._get_historical_ranks_batch(
            db, group_names, latest_date, period_days,
            market=normalized_market,
            formula_version=resolved_formula,
        )

        # Build result with rank changes
        result = []
        for ranking in rankings:
            item = dict(ranking)
            for period_name in period_days.keys():
                historical_rank = historical_ranks.get(
                    (str(ranking["industry_group"]), period_name)
                )
                if historical_rank is not None:
                    item[f'rank_change_{period_name}'] = (
                        historical_rank - int(ranking["rank"])
                    )
                else:
                    item[f'rank_change_{period_name}'] = None

            result.append(item)
        return result

    @staticmethod
    def _rank_record_payload(
        ranking: IBDGroupRank,
        *,
        pct_rs_above_80: float | None,
        top_symbol_name: str | None = None,
    ) -> Dict:
        return rank_record_payload(
            ranking,
            pct_rs_above_80=pct_rs_above_80,
            top_symbol_name=top_symbol_name,
        )

    @staticmethod
    def _annotate_top_symbol_names(db: Session, rows: List[Dict]) -> None:
        name_map = IBDGroupRankService._get_symbol_name_map(
            db,
            [row.get("top_symbol") for row in rows],
        )
        for row in rows:
            row["top_symbol_name"] = name_map.get(row.get("top_symbol"))

    @staticmethod
    def _get_symbol_name_map(db: Session, symbols: List[str | None]) -> Dict[str, str | None]:
        normalized_symbols = {
            str(symbol).strip()
            for symbol in symbols
            if str(symbol or "").strip()
        }
        if not normalized_symbols:
            return {}
        return dict(
            db.query(StockUniverse.symbol, StockUniverse.name)
            .filter(StockUniverse.symbol.in_(normalized_symbols))
            .all()
        )

    def get_historical_ranks_batch(
        self,
        db: Session,
        group_names: List[str],
        current_date: date,
        period_days: Dict[str, int],
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> Dict[tuple, int]:
        """Public alias of :meth:`_get_historical_ranks_batch`."""
        return self._get_historical_ranks_batch(
            db,
            group_names,
            current_date,
            period_days,
            market=market,
            formula_version=formula_version,
        )

    def _get_historical_ranks_batch(
        self,
        db: Session,
        group_names: List[str],
        current_date: date,
        period_days: Dict[str, int],
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> Dict[tuple, int]:
        """Batch fetch historical ranks for all groups and periods in ONE query.

        Scoped by ``market`` so multi-market data doesn't cross-contaminate
        rank-change calculations.
        """
        if not group_names or not period_days:
            return {}

        normalized_market = (market or "US").upper()
        resolved_formula = formula_version or self.market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )

        # Calculate calendar-date range covering all target offsets plus the
        # closest-record match window.
        max_days = max(period_days.values())
        earliest_date = current_date - timedelta(days=max_days + 7)

        # Single query: fetch ALL historical records within the date range
        all_records = db.query(
            IBDGroupRank.industry_group,
            IBDGroupRank.date,
            IBDGroupRank.rank
        ).filter(
            and_(
                IBDGroupRank.industry_group.in_(group_names),
                IBDGroupRank.date >= earliest_date,
                IBDGroupRank.date < current_date,  # Exclude current date
                IBDGroupRank.market == normalized_market,
                IBDGroupRank.rs_formula_version == resolved_formula,
            )
        ).all()

        # Build lookup: group_name -> list of (date, rank)
        group_history = {}
        for record in all_records:
            if record.industry_group not in group_history:
                group_history[record.industry_group] = []
            group_history[record.industry_group].append(
                (record.date, record.rank)
            )

        # For each group and period, find the closest historical rank
        result = {}
        for group_name in group_names:
            history = group_history.get(group_name, [])
            if not history:
                continue

            for period_name, days in period_days.items():
                target_date = current_date - timedelta(days=days)

                # Filter to records within a 7-calendar-day window of target
                candidates = [
                    (d, r) for d, r in history
                    if abs((d - target_date).days) <= 7
                ]

                if not candidates:
                    continue

                # Pick closest date (prefer earlier if tied)
                def _distance_key(item):
                    d, _ = item
                    delta = d - target_date
                    return (abs(delta.days), delta.days > 0)

                closest_date, closest_rank = min(candidates, key=_distance_key)
                result[(group_name, period_name)] = closest_rank

        return result

    def get_group_history(
        self,
        db: Session,
        industry_group: str,
        days: int = 180,
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> Dict:
        """Get historical ranking data for a specific group, scoped by market."""
        normalized_market = (market or "US").upper()
        resolved_formula = formula_version or self.market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )
        cutoff_date = datetime.now().date() - timedelta(days=days)

        records = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.industry_group == industry_group,
                IBDGroupRank.date >= cutoff_date,
                IBDGroupRank.market == normalized_market,
                IBDGroupRank.rs_formula_version == resolved_formula,
            )
        ).order_by(IBDGroupRank.date.desc()).all()

        if not records:
            return {'industry_group': industry_group, 'history': []}

        # Current (most recent) data
        current = records[0]

        # Get rank changes using calendar-day target offsets with closest-record matching.
        period_days = dict(GROUP_RANK_CHANGE_CALENDAR_DAYS)
        rank_changes = {}

        historical_ranks = self._get_historical_ranks_batch(
            db, [industry_group], current.date, period_days,
            market=normalized_market,
            formula_version=resolved_formula,
        )
        for period_name in period_days:
            historical_rank = historical_ranks.get((industry_group, period_name))
            if historical_rank is not None:
                rank_changes[f'rank_change_{period_name}'] = historical_rank - current.rank
            else:
                rank_changes[f'rank_change_{period_name}'] = None

        # Build history data points
        history = [
            {
                'date': r.date.isoformat(),
                'rank': r.rank,
                'avg_rs_rating': r.avg_rs_rating,
                'avg_rs_rating_1m': r.avg_rs_rating_1m,
                'avg_rs_rating_3m': r.avg_rs_rating_3m,
                'num_stocks': r.num_stocks,
            }
            for r in records
        ]

        # Get constituent stocks with metrics
        stocks = self._get_constituent_stocks(
            db,
            industry_group,
            market=normalized_market,
            as_of_date=current.date,
        )

        pct_above_80 = self._calculate_pct_above_80(
            current.num_stocks_rs_above_80, current.num_stocks
        )

        top_symbol_name = self._get_symbol_name_map(db, [current.top_symbol]).get(
            current.top_symbol
        )
        ranking_payload = self._rank_record_payload(
            current,
            pct_rs_above_80=pct_above_80,
            top_symbol_name=top_symbol_name,
        )
        ranking_payload.update(rank_changes)

        return build_group_detail_payload_from_parts(
            industry_group,
            ranking=ranking_payload,
            history=history,
            stocks=stocks,
        )

    def _get_constituent_stocks(
        self,
        db: Session,
        industry_group: str,
        *,
        market: str = "US",
        as_of_date: date | None = None,
    ) -> List[Dict]:
        """
        Get constituent stocks for an industry group with their metrics.

        Fetches from the most recent completed scan for the requested market.

        Args:
            db: Database session
            industry_group: IBD industry group name

        Returns:
            List of stocks with RS, earnings, and sales metrics
        """
        items = self.group_constituent_source.get_constituent_items(
            db,
            industry_group,
            market=market,
            as_of_date=as_of_date,
        )
        return constituent_stock_payloads_from_scan_items(items)

    def get_rank_movers(
        self,
        db: Session,
        period: str = '1w',
        limit: int = 20,
        calculation_date: date | None = None,
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> Dict:
        """Get groups with biggest rank changes over a period, scoped by market."""
        current_rankings = self.get_current_rankings(
            db,
            limit=197,
            calculation_date=calculation_date,
            market=market,
            formula_version=formula_version,
        )

        if not current_rankings:
            return {'period': period, 'gainers': [], 'losers': []}

        # Filter to groups with rank change data for this period
        change_key = f'rank_change_{period}'
        groups_with_change = [
            r for r in current_rankings
            if r.get(change_key) is not None
        ]

        # Split by sign so a market with only positive (or only negative) movers
        # never reports the smallest gainers as "losers" or vice versa.
        gainers = [r for r in groups_with_change if r[change_key] > 0]
        losers = [r for r in groups_with_change if r[change_key] < 0]
        gainers.sort(key=lambda r: r[change_key], reverse=True)
        losers.sort(key=lambda r: r[change_key])

        return {
            'period': period,
            'gainers': gainers[:limit],
            'losers': losers[:limit],
        }
