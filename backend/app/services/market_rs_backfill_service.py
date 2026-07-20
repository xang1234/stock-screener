"""Discover and materialize the canonical balanced Market RS history."""

from __future__ import annotations

from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.domain.markets import get_market_catalog
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.stock import StockPrice
from app.services.benchmark_registry_service import benchmark_registry
from app.services.canonical_group_ranking_service import CanonicalGroupRankingService
from app.services.market_calendar_service import MarketCalendarService
from app.services.market_rs_inputs import MarketRsInputLoader, MarketRsInputUnavailable
from app.services.market_rs_rollout_contracts import (
    BackfillDateResult,
    BackfillReport,
    normalize_rollout_market,
)
from app.services.market_rs_snapshot_service import MarketRsSnapshotService


class MarketRsBackfillService:
    def __init__(
        self,
        *,
        calendar_service: MarketCalendarService,
        input_loader: MarketRsInputLoader,
        snapshot_service: MarketRsSnapshotService,
        repository: MarketRsRunRepository,
        group_service: CanonicalGroupRankingService,
    ) -> None:
        self.calendar_service = calendar_service
        self.input_loader = input_loader
        self.snapshot_service = snapshot_service
        self.repository = repository
        self.group_service = group_service

    @staticmethod
    def _reason_code(exc: Exception, *, stage: str) -> str:
        if isinstance(exc, MarketRsInputUnavailable):
            return exc.reason_code
        name = type(exc).__name__
        snake = "".join(
            ("_" + char.lower()) if char.isupper() else char
            for char in name
        ).lstrip("_")
        return f"{stage}_{snake}" if snake else f"{stage}_failed"

    def _earliest_available_price_date(
        self,
        db: Session,
        market: str,
    ) -> date | None:
        candidates = benchmark_registry.get_candidate_symbols(market)
        return (
            db.query(func.min(StockPrice.date))
            .filter(StockPrice.symbol.in_(tuple(candidates)))
            .scalar()
        )

    def earliest_backfillable_date(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
    ) -> date | None:
        normalized = normalize_rollout_market(market)
        available_start = self._earliest_available_price_date(db, normalized)
        if available_start is None or available_start > through_date:
            return None
        sessions = self.calendar_service.trading_days(
            normalized,
            available_start,
            through_date,
        )
        for session_date in sessions:
            try:
                inputs = self.input_loader.load(
                    db,
                    market=normalized,
                    as_of_date=session_date,
                )
            except MarketRsInputUnavailable:
                continue
            if len(inputs.excess_returns_by_symbol) >= 2:
                return session_date
        return None

    def candidate_dates(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        first_valid_date: date | None = None,
    ) -> tuple[date, ...]:
        normalized = normalize_rollout_market(market)
        boundary = first_valid_date or self.earliest_backfillable_date(
            db,
            market=normalized,
            through_date=through_date,
        )
        if boundary is None:
            return ()
        return tuple(
            session_date
            for session_date in self.calendar_service.trading_days(
                normalized,
                boundary,
                through_date,
            )
            if boundary <= session_date <= through_date
        )

    def backfill(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        start_date: date | None = None,
    ) -> BackfillReport:
        normalized = normalize_rollout_market(market)
        groups_applicable = (
            get_market_catalog().get(normalized).capabilities.group_rankings
        )
        first_valid = self.earliest_backfillable_date(
            db,
            market=normalized,
            through_date=through_date,
        )
        if first_valid is None:
            return BackfillReport(
                market=normalized,
                formula_version=BALANCED_RS_FORMULA_VERSION,
                requested_start_date=start_date,
                through_date=through_date,
                first_valid_date=None,
                candidate_count=0,
                completed_count=0,
                failed_count=1,
                latest_run_id=None,
                group_row_count=0,
                results=(),
                validation_errors=(
                    "No valid balanced Market RS history boundary was found.",
                ),
            )

        candidates = self.candidate_dates(
            db,
            market=normalized,
            through_date=through_date,
            first_valid_date=first_valid,
        )
        results: list[BackfillDateResult] = []
        for calculation_date in candidates:
            run = self.repository.get_completed_exact(
                db,
                market=normalized,
                as_of_date=calculation_date,
                formula_version=BALANCED_RS_FORMULA_VERSION,
            )
            if start_date is not None and calculation_date < start_date and run is None:
                results.append(
                    BackfillDateResult(
                        as_of_date=calculation_date,
                        status="failed",
                        market_rs_run_id=None,
                        group_market_rs_run_id=None,
                        eligible_symbol_count=0,
                        group_row_count=0,
                        reason_code="resume_limiter_skipped_incomplete",
                        diagnostics={
                            "start_date": start_date.isoformat(),
                            "error": (
                                "Required date is incomplete before the "
                                "calculation resume limit."
                            ),
                        },
                    )
                )
                continue
            stage = "stock_calculation"
            try:
                run = self.snapshot_service.calculate(
                    db,
                    market=normalized,
                    as_of_date=calculation_date,
                    formula_version=BALANCED_RS_FORMULA_VERSION,
                    rebuild_incompatible=True,
                )
                groups: list[dict[str, object]] = []
                group_run_id = None
                if groups_applicable:
                    stage = "group_calculation"
                    groups = self.group_service.calculate_and_store(
                        db,
                        market=normalized,
                        as_of_date=calculation_date,
                        formula_version=BALANCED_RS_FORMULA_VERSION,
                    )
                    if not groups:
                        raise RuntimeError("No eligible Group rows were produced")
                    group_run_ids = {
                        int(row["market_rs_run_id"])
                        for row in groups
                        if row.get("market_rs_run_id") is not None
                    }
                    group_run_id = (
                        next(iter(group_run_ids))
                        if len(group_run_ids) == 1
                        else None
                    )
                    if group_run_id != run.id:
                        raise RuntimeError(
                            "Group rows do not reference the exact Market RS run"
                        )
                results.append(
                    BackfillDateResult(
                        as_of_date=calculation_date,
                        status="completed",
                        market_rs_run_id=run.id,
                        group_market_rs_run_id=group_run_id,
                        eligible_symbol_count=int(run.eligible_symbol_count),
                        group_row_count=len(groups),
                    )
                )
            except Exception as exc:
                db.rollback()
                reason_code = self._reason_code(exc, stage=stage)
                if (
                    stage == "group_calculation"
                    and not isinstance(exc, MarketRsInputUnavailable)
                ):
                    reason_code = "group_calculation_failed"
                diagnostics: dict[str, object] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if isinstance(exc, MarketRsInputUnavailable):
                    diagnostics.update(exc.diagnostics)
                results.append(
                    BackfillDateResult(
                        as_of_date=calculation_date,
                        status="failed",
                        market_rs_run_id=getattr(run, "id", None),
                        group_market_rs_run_id=None,
                        eligible_symbol_count=int(
                            getattr(run, "eligible_symbol_count", 0) or 0
                        ),
                        group_row_count=0,
                        reason_code=reason_code,
                        diagnostics=diagnostics,
                    )
                )

        completed = tuple(item for item in results if item.status == "completed")
        failed = tuple(item for item in results if item.status == "failed")
        return BackfillReport(
            market=normalized,
            formula_version=BALANCED_RS_FORMULA_VERSION,
            requested_start_date=start_date or first_valid,
            through_date=through_date,
            first_valid_date=first_valid,
            candidate_count=len(candidates),
            completed_count=len(completed),
            failed_count=len(failed),
            latest_run_id=(
                completed[-1].market_rs_run_id if completed else None
            ),
            group_row_count=sum(item.group_row_count for item in completed),
            results=tuple(results),
        )


__all__ = ["MarketRsBackfillService"]
