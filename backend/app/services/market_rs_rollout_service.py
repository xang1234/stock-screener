"""Public facade for canonical Market RS backfill, validation, and activation."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.infra.db.repositories.feature_run_repo import SqlFeatureRunRepository
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.canonical_group_ranking_service import CanonicalGroupRankingService
from app.services.market_calendar_service import MarketCalendarService
from app.services.market_rs_activation_validator import (
    MarketRsActivationValidator,
)
from app.services.market_rs_activator import MarketRsActivator
from app.services.market_rs_backfill_service import MarketRsBackfillService
from app.services.market_rs_inputs import MarketRsInputLoader
from app.services.market_rs_rollout_contracts import (
    ActivationValidationReport,
    BackfillDateResult,
    BackfillReport,
    MarketRsActivationRejected,
)
from app.services.market_rs_snapshot_service import MarketRsSnapshotService


FeatureRunRepositoryFactory = Callable[[Session], SqlFeatureRunRepository]


class MarketRsRolloutService:
    """Coordinate explicit rollout collaborators behind the stable public API."""

    def __init__(
        self,
        *,
        calendar_service: MarketCalendarService,
        input_loader: MarketRsInputLoader,
        market_rs_snapshot_service: MarketRsSnapshotService,
        market_rs_repository: MarketRsRunRepository,
        canonical_group_service: CanonicalGroupRankingService,
        feature_run_repository_factory: FeatureRunRepositoryFactory | None = None,
    ) -> None:
        feature_factory = (
            feature_run_repository_factory or SqlFeatureRunRepository
        )
        self.backfill_service = MarketRsBackfillService(
            calendar_service=calendar_service,
            input_loader=input_loader,
            snapshot_service=market_rs_snapshot_service,
            repository=market_rs_repository,
            group_service=canonical_group_service,
        )
        self.validator = MarketRsActivationValidator(
            backfill_service=self.backfill_service,
            repository=market_rs_repository,
            feature_run_repository_factory=feature_factory,
        )
        self.activator = MarketRsActivator(
            repository=market_rs_repository,
            feature_run_repository_factory=feature_factory,
            validator=self.validator,
        )

    def earliest_backfillable_date(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
    ) -> date | None:
        return self.backfill_service.earliest_backfillable_date(
            db,
            market=market,
            through_date=through_date,
        )

    def candidate_dates(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        first_valid_date: date | None = None,
    ) -> tuple[date, ...]:
        return self.backfill_service.candidate_dates(
            db,
            market=market,
            through_date=through_date,
            first_valid_date=first_valid_date,
        )

    def backfill(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        start_date: date | None = None,
    ) -> BackfillReport:
        return self.backfill_service.backfill(
            db,
            market=market,
            through_date=through_date,
            start_date=start_date,
        )

    def validate_activation(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        feature_run_id: int,
        static_staging_dir: Path,
    ) -> ActivationValidationReport:
        return self.validator.validate(
            db,
            market=market,
            through_date=through_date,
            feature_run_id=feature_run_id,
            static_staging_dir=static_staging_dir,
        )

    def revalidate_static(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        feature_run_id: int,
        static_staging_dir: Path,
    ) -> tuple[str, ...]:
        return self.validator.revalidate_static(
            db,
            market=market,
            through_date=through_date,
            feature_run_id=feature_run_id,
            static_staging_dir=static_staging_dir,
        )

    def activate(
        self,
        db: Session,
        *,
        market: str,
        formula_version: str,
        feature_run_id: int,
        validation: ActivationValidationReport,
        static_staging_dir: Path,
    ) -> None:
        self.activator.activate(
            db,
            market=market,
            formula_version=formula_version,
            feature_run_id=feature_run_id,
            validation=validation,
            static_staging_dir=static_staging_dir,
        )


__all__ = [
    "ActivationValidationReport",
    "BackfillDateResult",
    "BackfillReport",
    "MarketRsActivationRejected",
    "MarketRsRolloutService",
]
