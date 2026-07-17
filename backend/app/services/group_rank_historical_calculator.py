"""Historical and gap-fill orchestration for group rankings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import logging
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from ..config import settings
from .derived_data_execution_policy import (
    DerivedDataExecutionPolicy,
)
from .group_rank_input_loader import GroupRankInputLoader
from .group_rank_legacy_adapter import (
    LegacyGroupRankPrefetchAdapter,
)
from .group_rank_models import GroupRankPrefetchData
from .group_ranking_calculator import GroupRankingCalculator
from .group_ranking_repository import GroupRankingRepository
from .market_calendar_service import MarketCalendarService


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupRankHistoricalCalculator:
    input_loader: GroupRankInputLoader
    ranking_calculator: GroupRankingCalculator
    repository: GroupRankingRepository
    calendar_service: MarketCalendarService
    legacy_adapter: LegacyGroupRankPrefetchAdapter

    def backfill_rankings_optimized(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
    ) -> Dict:
        normalized_market = (market or "US").upper()
        start_time = datetime.now()
        deleted = self.repository.delete_range(
            db,
            start_date=start_date,
            end_date=end_date,
            market=normalized_market,
        )
        db.commit()

        prefetch = self._load_prefetch(
            db,
            market=normalized_market,
        )
        if (
            prefetch.benchmark_prices is None
            or prefetch.benchmark_prices.empty
        ):
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "deleted": deleted,
                "total_dates": 0,
                "processed": 0,
                "skipped": 0,
                "errors": 1,
                "error": "Failed to fetch SPY benchmark data",
            }

        group_names, prefetch = self._complete_prefetch(
            db,
            market=normalized_market,
            prefetch=prefetch,
        )
        dates_to_process = self._trading_dates_descending(
            start_date,
            end_date,
            market=normalized_market,
        )
        processed = 0
        errors = 0
        chunk_size = self._calculation_chunk_size()

        for chunk_start in range(
            0,
            len(dates_to_process),
            chunk_size,
        ):
            date_chunk = dates_to_process[
                chunk_start:chunk_start + chunk_size
            ]
            rankings_by_date = (
                self.ranking_calculator.calculate_for_dates(
                    prefetch=prefetch,
                    group_names=group_names,
                    calculation_dates=date_chunk,
                )
            )
            for calculation_date in date_chunk:
                rankings = rankings_by_date.get(
                    calculation_date,
                    (),
                )
                if not rankings:
                    errors += 1
                    continue
                try:
                    self.repository.store_rankings(
                        db,
                        calculation_date=calculation_date,
                        rankings=rankings,
                        market=normalized_market,
                    )
                    db.commit()
                    processed += 1
                except Exception:
                    db.rollback()
                    errors += 1
                    logger.exception(
                        "Error processing %s",
                        calculation_date,
                    )

        duration = (datetime.now() - start_time).total_seconds()
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "deleted": deleted,
            "total_dates": len(dates_to_process),
            "processed": processed,
            "skipped": 0,
            "errors": errors,
            "duration_seconds": round(duration, 2),
        }

    def backfill_rankings(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
    ) -> Dict:
        normalized_market = (market or "US").upper()
        dates_to_process = self._trading_dates_descending(
            start_date,
            end_date,
            market=normalized_market,
        )
        processed = 0
        skipped = 0
        errors = 0

        for calculation_date in dates_to_process:
            existing = self.repository.current_rank_rows(
                db,
                limit=1,
                market=normalized_market,
                calculation_date=calculation_date,
            )
            if existing:
                skipped += 1
                continue
            try:
                rankings = self._calculate_and_store_date(
                    db,
                    calculation_date=calculation_date,
                    market=normalized_market,
                )
                if rankings:
                    processed += 1
                else:
                    errors += 1
            except Exception:
                db.rollback()
                errors += 1
                logger.exception(
                    "Error processing %s",
                    calculation_date,
                )

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_dates": len(dates_to_process),
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }

    def find_missing_dates(
        self,
        db: Session,
        lookback_days: int = 365,
        *,
        market: str = "US",
        end_date: date | None = None,
    ) -> List[date]:
        normalized_market = (market or "US").upper()
        window_end = (
            end_date
            or self.calendar_service.market_now(
                normalized_market
            ).date()
        )
        start_date = window_end - timedelta(days=lookback_days)
        existing_date_set = self.repository.existing_dates(
            db,
            start_date=start_date,
            end_date=window_end,
            market=normalized_market,
        )

        missing_dates = []
        current_date = start_date
        while current_date < window_end:
            if (
                self.calendar_service.is_trading_day(
                    normalized_market,
                    current_date,
                )
                and current_date not in existing_date_set
            ):
                missing_dates.append(current_date)
            current_date += timedelta(days=1)
        return sorted(missing_dates)

    def fill_gaps(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
    ) -> Dict:
        normalized_market = (market or "US").upper()
        stats = {
            "total_dates": len(missing_dates),
            "processed": 0,
            "skipped": 0,
            "errors": 0,
        }
        for calculation_date in missing_dates:
            try:
                rankings = self._calculate_and_store_date(
                    db,
                    calculation_date=calculation_date,
                    market=normalized_market,
                )
                if rankings:
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
            except Exception:
                db.rollback()
                stats["errors"] += 1
                logger.exception(
                    "Error filling gap for %s",
                    calculation_date,
                )
        return stats

    def fill_gaps_optimized(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
        policy: DerivedDataExecutionPolicy = (
            DerivedDataExecutionPolicy.provider_allowed()
        ),
    ) -> Dict:
        if not missing_dates:
            return {
                "total_dates": 0,
                "processed": 0,
                "skipped": 0,
                "errors": 0,
            }

        normalized_market = (market or "US").upper()
        start_time = datetime.now()
        prefetch = self._load_prefetch(
            db,
            market=normalized_market,
            policy=policy,
        )
        if (
            prefetch.benchmark_prices is None
            or prefetch.benchmark_prices.empty
        ):
            return {
                "total_dates": len(missing_dates),
                "processed": 0,
                "skipped": 0,
                "errors": len(missing_dates),
                "error": "Failed to fetch SPY benchmark data",
                "prefetch_stats": prefetch.stats.to_dict(),
            }

        group_names, prefetch = self._complete_prefetch(
            db,
            market=normalized_market,
            prefetch=prefetch,
        )
        stats = {
            "total_dates": len(missing_dates),
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "prefetch_stats": prefetch.stats.to_dict(),
        }
        chunk_size = self._calculation_chunk_size()

        for chunk_start in range(
            0,
            len(missing_dates),
            chunk_size,
        ):
            date_chunk = missing_dates[
                chunk_start:chunk_start + chunk_size
            ]
            rankings_by_date = (
                self.ranking_calculator.calculate_for_dates(
                    prefetch=prefetch,
                    group_names=group_names,
                    calculation_dates=date_chunk,
                )
            )
            for calculation_date in date_chunk:
                rankings = rankings_by_date.get(
                    calculation_date,
                    (),
                )
                if not rankings:
                    stats["errors"] += 1
                    continue
                try:
                    self.repository.store_rankings(
                        db,
                        calculation_date=calculation_date,
                        rankings=rankings,
                        market=normalized_market,
                    )
                    db.commit()
                    stats["processed"] += 1
                except Exception:
                    db.rollback()
                    stats["errors"] += 1
                    logger.exception(
                        "Error filling gap for %s",
                        calculation_date,
                    )

        duration = (datetime.now() - start_time).total_seconds()
        stats["duration_seconds"] = round(duration, 2)
        return stats

    def backfill_rankings_chunked(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        chunk_size_days: int = 30,
        *,
        market: str = "US",
    ) -> Dict:
        normalized_market = (market or "US").upper()
        total_stats = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_dates": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
        }
        chunk_end = end_date
        while chunk_end >= start_date:
            chunk_start = max(
                start_date,
                chunk_end - timedelta(
                    days=chunk_size_days - 1
                ),
            )
            try:
                chunk_result = self.backfill_rankings(
                    db,
                    chunk_start,
                    chunk_end,
                    market=normalized_market,
                )
                for key in (
                    "total_dates",
                    "processed",
                    "skipped",
                    "errors",
                ):
                    total_stats[key] += chunk_result.get(key, 0)
            except Exception:
                logger.exception(
                    "Error processing chunk %s to %s",
                    chunk_start,
                    chunk_end,
                )
                total_stats["errors"] += chunk_size_days
            chunk_end = chunk_start - timedelta(days=1)
        return total_stats

    def _calculate_and_store_date(
        self,
        db: Session,
        *,
        calculation_date: date,
        market: str,
    ) -> tuple:
        prefetch = self._load_prefetch(
            db,
            market=market,
        )
        if (
            prefetch.benchmark_prices is None
            or prefetch.benchmark_prices.empty
        ):
            return ()
        group_names, prefetch = self._complete_prefetch(
            db,
            market=market,
            prefetch=prefetch,
        )
        rankings = self.ranking_calculator.calculate_for_date(
            prefetch=prefetch,
            group_names=group_names,
            calculation_date=calculation_date,
        )
        if rankings:
            self.repository.store_rankings(
                db,
                calculation_date=calculation_date,
                rankings=rankings,
                market=market,
            )
            db.commit()
        return rankings

    def _load_prefetch(
        self,
        db: Session,
        *,
        market: str,
        policy: DerivedDataExecutionPolicy = (
            DerivedDataExecutionPolicy.provider_allowed()
        ),
    ) -> GroupRankPrefetchData:
        return self.legacy_adapter.adapt(
            self.input_loader.load(
                db,
                market=market,
                policy=policy,
            )
        )

    def _complete_prefetch(
        self,
        db: Session,
        *,
        market: str,
        prefetch: GroupRankPrefetchData,
    ) -> tuple[tuple[str, ...], GroupRankPrefetchData]:
        group_names = (
            tuple(prefetch.symbols_by_group)
            or self.input_loader.taxonomy_source.groups(
                db,
                market,
            )
        )
        completed = self.input_loader.complete_legacy_symbols(
            db,
            market=market,
            group_names=group_names,
            prefetch=prefetch,
        )
        return tuple(group_names), completed

    def _trading_dates_descending(
        self,
        start_date: date,
        end_date: date,
        *,
        market: str,
    ) -> list[date]:
        dates = []
        current_date = end_date
        while current_date >= start_date:
            if self.calendar_service.is_trading_day(
                market,
                current_date,
            ):
                dates.append(current_date)
            current_date -= timedelta(days=1)
        return dates

    @staticmethod
    def _calculation_chunk_size() -> int:
        return max(
            1,
            int(
                settings.group_rank_gapfill_chunk_size
                or 30
            ),
        )
