"""Validate database, feature-run, and static inputs before RS activation."""

from __future__ import annotations

from datetime import date
import math
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.domain.feature_store.models import RunStatus
from app.domain.markets import get_market_catalog
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
    RsPublicationIdentity,
    balanced_run_has_required_price_basis,
)
from app.infra.db.repositories.feature_run_repo import SqlFeatureRunRepository
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.industry import IBDGroupRank
from app.services.feature_run_rs_identity import (
    FeatureRunRsIdentityError,
    resolve_feature_run_rs_identity,
)
from app.services.ibd_industry_service import IBDIndustryService
from app.services.market_rs_backfill_service import MarketRsBackfillService
from app.services.market_rs_rollout_contracts import (
    ActivationValidationReport,
    normalize_rollout_market,
)
from app.services.market_rs_static_artifact_validator import (
    MarketRsStaticArtifactValidator,
)
from app.services.static_market_artifact_contract import STATIC_SITE_SCHEMA_VERSION


FeatureRunRepositoryFactory = Callable[[Session], SqlFeatureRunRepository]


class MarketRsActivationValidator:
    def __init__(
        self,
        *,
        backfill_service: MarketRsBackfillService,
        repository: MarketRsRunRepository,
        feature_run_repository_factory: FeatureRunRepositoryFactory,
        static_validator: MarketRsStaticArtifactValidator | None = None,
    ) -> None:
        self.backfill_service = backfill_service
        self.repository = repository
        self.feature_run_repository_factory = feature_run_repository_factory
        self.static_validator = static_validator or MarketRsStaticArtifactValidator()

    def _validate_run_and_groups(
        self,
        db: Session,
        *,
        market: str,
        calculation_date: date,
        errors: list[str],
    ) -> Any | None:
        run = self.repository.get_completed_exact(
            db,
            market=market,
            as_of_date=calculation_date,
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )
        if run is None:
            errors.append(
                f"Missing completed stock RS snapshot for {calculation_date}."
            )
            return None
        if not balanced_run_has_required_price_basis(run):
            errors.append(
                f"Market RS run for {calculation_date} has an "
                "incompatible price basis."
            )
        if len(run.rows) != int(run.eligible_symbol_count):
            errors.append(
                f"Stock row count mismatch for {calculation_date}: "
                f"{len(run.rows)} != {run.eligible_symbol_count}."
            )
        for row in run.rows:
            ratings = (
                row.overall_rs,
                row.rs_1m,
                row.rs_3m,
                row.rs_6m,
                row.rs_9m,
                row.rs_12m,
            )
            if any(
                not isinstance(value, int) or value < 1 or value > 99
                for value in ratings
            ):
                errors.append(
                    f"Out-of-range stock RS rating for {row.symbol} "
                    f"on {calculation_date}."
                )
            if not math.isfinite(float(row.weighted_composite)):
                errors.append(
                    f"Non-finite stock RS composite for {row.symbol} "
                    f"on {calculation_date}."
                )

        if not get_market_catalog().get(market).capabilities.group_rankings:
            return run

        group_rows = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == market,
                IBDGroupRank.date == calculation_date,
                IBDGroupRank.rs_formula_version == BALANCED_RS_FORMULA_VERSION,
            )
            .all()
        )
        eligible_symbols = {row.symbol for row in run.rows}
        expected_groups: set[str] = set()
        try:
            memberships = IBDIndustryService.get_group_memberships(
                db,
                market=market,
            )
            expected_groups = {
                group_name
                for group_name, symbols in memberships.items()
                if len(set(symbols) & eligible_symbols) >= 3
            }
        except Exception as exc:
            errors.append(
                f"Could not reconstruct expected Groups for "
                f"{calculation_date}: {exc}"
            )
        stored_groups = {row.industry_group for row in group_rows}
        missing_groups = sorted(expected_groups - stored_groups)
        if missing_groups:
            errors.append(
                f"Missing eligible Group rows for {calculation_date}: "
                f"{', '.join(missing_groups)}."
            )
        if any(
            row.rs_formula_version != BALANCED_RS_FORMULA_VERSION
            or row.market_rs_run_id != run.id
            for row in group_rows
        ):
            errors.append(f"Mixed Group formula/run IDs for {calculation_date}.")
        ordered = sorted(group_rows, key=lambda row: row.rank)
        if [row.rank for row in ordered] != list(range(1, len(ordered) + 1)):
            errors.append(f"Non-contiguous Group ranks for {calculation_date}.")
        deterministic = sorted(
            group_rows,
            key=lambda row: (-float(row.avg_rs_rating), row.industry_group),
        )
        if [row.industry_group for row in ordered] != [
            row.industry_group for row in deterministic
        ]:
            errors.append(
                f"Non-deterministic Group rank order for {calculation_date}."
            )
        return run

    @staticmethod
    def _validate_feature(
        feature: Any,
        *,
        market: str,
        through_date: date,
        latest_run: Any | None,
        errors: list[str],
    ) -> None:
        if feature.status != RunStatus.PUBLISHED or feature.as_of_date != through_date:
            errors.append(
                "Candidate Feature run is not published for the activation date."
            )
        if latest_run is None:
            return
        expected = RsPublicationIdentity(
            snapshot=GroupSnapshotIdentity(
                market,
                through_date,
                BALANCED_RS_FORMULA_VERSION,
            ),
            market_rs_run_id=latest_run.id,
            universe_size=latest_run.eligible_symbol_count,
        )
        try:
            resolved = resolve_feature_run_rs_identity(
                feature,
                ranking_date=through_date,
            )
        except FeatureRunRsIdentityError as exc:
            errors.append(
                "Candidate Feature run rs_formula_version/RS identity is "
                f"invalid: {exc}"
            )
            return
        if resolved.publication != expected:
            errors.append(
                "Candidate Feature run RS identity does not match the latest "
                "canonical Market RS publication."
            )

    def validate(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        feature_run_id: int,
        static_staging_dir: Path,
    ) -> ActivationValidationReport:
        normalized = normalize_rollout_market(market)
        errors: list[str] = []
        first_valid = self.backfill_service.earliest_backfillable_date(
            db,
            market=normalized,
            through_date=through_date,
        )
        candidates = (
            self.backfill_service.candidate_dates(
                db,
                market=normalized,
                through_date=through_date,
                first_valid_date=first_valid,
            )
            if first_valid is not None
            else ()
        )
        if not candidates:
            errors.append(
                "No required balanced Market RS candidate dates were found."
            )

        latest_run = None
        for calculation_date in candidates:
            run = self._validate_run_and_groups(
                db,
                market=normalized,
                calculation_date=calculation_date,
                errors=errors,
            )
            if calculation_date == through_date:
                latest_run = run
        if candidates and candidates[-1] != through_date:
            errors.append(
                "Candidate trading-date history does not reach the activation date."
            )

        feature_repo = self.feature_run_repository_factory(db)
        feature = None
        try:
            feature = feature_repo.get_run(feature_run_id)
        except Exception as exc:
            errors.append(f"Feature run {feature_run_id} is unavailable: {exc}")
        if feature is not None:
            self._validate_feature(
                feature,
                market=normalized,
                through_date=through_date,
                latest_run=latest_run,
                errors=errors,
            )

        bundle_hash = None
        rrg_status = None
        if latest_run is not None:
            try:
                static_result = self.static_validator.validate(
                    db,
                    market=normalized,
                    through_date=through_date,
                    latest_run=latest_run,
                    feature_run_id=feature_run_id,
                    static_staging_dir=Path(static_staging_dir),
                )
                bundle_hash = (
                    static_result.bundle_fingerprint.sha256
                    if static_result.bundle_fingerprint is not None
                    else None
                )
                rrg_status = static_result.rrg_status
                errors.extend(static_result.errors)
            except Exception as exc:
                errors.append(
                    f"Staged static artifact validation failed: {exc}"
                )
        elif not (Path(static_staging_dir) / "manifest.json").is_file():
            errors.append(f"Missing staged {STATIC_SITE_SCHEMA_VERSION} manifest.")

        return ActivationValidationReport(
            market=normalized,
            formula_version=BALANCED_RS_FORMULA_VERSION,
            through_date=through_date,
            first_valid_date=first_valid,
            candidate_count=len(candidates),
            latest_market_rs_run_id=getattr(latest_run, "id", None),
            latest_universe_hash=getattr(latest_run, "universe_hash", None),
            feature_run_id=getattr(feature, "id", feature_run_id),
            feature_universe_hash=getattr(feature, "universe_hash", None),
            static_bundle_sha256=bundle_hash,
            errors=tuple(dict.fromkeys(errors)),
            rrg_status=rrg_status,
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
        errors: list[str] = []
        latest_run = self._validate_run_and_groups(
            db,
            market=market,
            calculation_date=through_date,
            errors=errors,
        )
        if latest_run is not None:
            result = self.static_validator.validate(
                db,
                market=market,
                through_date=through_date,
                latest_run=latest_run,
                feature_run_id=feature_run_id,
                static_staging_dir=static_staging_dir,
            )
            errors.extend(result.errors)
        return tuple(dict.fromkeys(errors))


__all__ = ["MarketRsActivationValidator"]
