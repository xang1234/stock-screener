"""
Service for calculating and managing IBD Industry Group Rankings.

Calculates daily rankings based on average RS rating of constituent stocks.
"""
import logging
from typing import Dict, List
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..models.industry import IBDGroupRank
from ..models.stock_universe import StockUniverse
from ..domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
    RsPublicationIdentity,
)
from ..infra.db.repositories.market_rs_repo import MarketRsRunRepository
from .canonical_group_ranking_service import CanonicalGroupRankingService
from .group_constituent_source import (
    GroupConstituentPublicationUnavailable,
    GroupConstituentSource,
)
from .group_detail_payloads import constituent_stock_payloads_from_scan_items
from .group_ranking_history import build_group_detail_payload_from_parts
from .group_rank_cache_policy import GroupRankCacheRequirement
from .group_ranking_calculation_service import (
    GroupRankingCalculationService,
    MissingIBDIndustryMappingsError as MissingIBDIndustryMappingsError,
)
from .group_ranking_payloads import rank_record_payload
from .group_rank_snapshot_reader import GroupRankSnapshotReader
from .legacy_group_rank_backfill import LegacyGroupBackfillService
from .legacy_group_rank_contracts import (
    CACHE_MISS_TOLERANCE_RATIO as CACHE_MISS_TOLERANCE_RATIO,
    GroupRankPrefetchData as GroupRankPrefetchData,
    IncompleteGroupRankingCacheError as IncompleteGroupRankingCacheError,
)
from .legacy_group_rank_data import ActiveSymbolsProvider, LegacyGroupRankingEngine
from .market_calendar_service import MarketCalendarService
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


class IBDGroupRankService:
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
        market_calendar: MarketCalendarService | None = None,
        active_symbols_provider: ActiveSymbolsProvider | None = None,
    ):
        """Initialize the service."""
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.rs_calculator = rs_calculator or RelativeStrengthCalculator()
        self.group_constituent_source = (
            group_constituent_source or GroupConstituentSource()
        )
        self.market_rs_repository = market_rs_repository
        self.legacy_ranking_engine = LegacyGroupRankingEngine(
            price_cache=price_cache,
            benchmark_cache=benchmark_cache,
            rs_calculator=self.rs_calculator,
            active_symbols_provider=active_symbols_provider,
        )
        self.group_calculation_service = GroupRankingCalculationService(
            ranking_engine=self.legacy_ranking_engine,
            canonical_group_service=canonical_group_service,
            market_rs_repository=market_rs_repository,
        )
        self.legacy_backfill_service = LegacyGroupBackfillService(
            ranking_engine=self.legacy_ranking_engine,
            market_calendar=market_calendar or MarketCalendarService(),
            market_rs_repository=market_rs_repository,
            calculation_service=self.group_calculation_service,
        )

    def backfill_rankings_optimized(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
    ) -> Dict:
        return self.legacy_backfill_service.backfill_rankings_optimized(
            db,
            start_date,
            end_date,
            market=market,
        )

    def backfill_rankings(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> Dict:
        return self.legacy_backfill_service.backfill_rankings(
            db,
            start_date,
            end_date,
            market=market,
            formula_version=formula_version,
        )

    def find_missing_dates(
        self,
        db: Session,
        lookback_days: int = 365,
        *,
        market: str = "US",
        end_date: date | None = None,
        formula_version: str | None = None,
    ) -> List[date]:
        return self.legacy_backfill_service.find_missing_dates(
            db,
            lookback_days,
            market=market,
            end_date=end_date,
            formula_version=formula_version,
        )

    def fill_gaps(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> Dict:
        return self.legacy_backfill_service.fill_gaps(
            db,
            missing_dates,
            market=market,
            formula_version=formula_version,
        )

    def fill_gaps_optimized(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
    ) -> Dict:
        return self.legacy_backfill_service.fill_gaps_optimized(
            db,
            missing_dates,
            market=market,
        )

    def backfill_rankings_chunked(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        chunk_size_days: int = 30,
        *,
        market: str = "US",
    ) -> Dict:
        return self.legacy_backfill_service.backfill_rankings_chunked(
            db,
            start_date,
            end_date,
            chunk_size_days,
            market=market,
        )

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
        return self.group_calculation_service.calculate_and_store(
            db,
            calculation_date,
            market=market,
            cache_only=cache_only,
            cache_requirement=cache_requirement,
            formula_version=formula_version,
        )

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
            ranking=current,
        )

        pct_above_80 = self.legacy_ranking_engine.calculate_pct_above_80(
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
        ranking: IBDGroupRank,
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
        formula_version = ranking.rs_formula_version
        snapshot = GroupSnapshotIdentity(
            ranking.market,
            ranking.date,
            formula_version,
        )
        if formula_version == BALANCED_RS_FORMULA_VERSION:
            run = self.market_rs_repository.get_completed_exact(
                db,
                market=snapshot.market,
                as_of_date=snapshot.as_of_date,
                formula_version=snapshot.formula_version,
            )
            if (
                run is None
                or run.id != ranking.market_rs_run_id
                or run.eligible_symbol_count <= 0
            ):
                raise GroupConstituentPublicationUnavailable(
                    "Canonical Group rank no longer resolves to its exact "
                    "Market RS publication."
                )
            publication = RsPublicationIdentity(
                snapshot=snapshot,
                market_rs_run_id=run.id,
                universe_size=run.eligible_symbol_count,
            )
        elif formula_version == LEGACY_RS_FORMULA_VERSION:
            publication = RsPublicationIdentity(
                snapshot=snapshot,
                market_rs_run_id=None,
                universe_size=None,
            )
        else:
            raise ValueError(f"Unsupported Group RS formula: {formula_version}")

        items = self.group_constituent_source.get_constituent_items(
            db,
            industry_group,
            publication=publication,
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
