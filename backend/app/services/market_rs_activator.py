"""Atomically activate one previously validated Market RS publication."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.domain.feature_store.models import RunStatus
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
    RsPublicationIdentity,
)
from app.infra.db.repositories.feature_run_repo import SqlFeatureRunRepository
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.feature_run_rs_identity import (
    FeatureRunRsIdentityError,
    resolve_feature_run_rs_identity,
)
from app.services.market_rs_activation_validator import (
    MarketRsActivationValidator,
)
from app.services.market_rs_rollout_contracts import (
    ActivationValidationReport,
    MarketRsActivationRejected,
    normalize_rollout_market,
)
from app.services.market_rs_static_artifact_validator import (
    MarketRsStaticArtifactValidator,
)


FeatureRunRepositoryFactory = Callable[[Session], SqlFeatureRunRepository]


class MarketRsActivator:
    def __init__(
        self,
        *,
        repository: MarketRsRunRepository,
        feature_run_repository_factory: FeatureRunRepositoryFactory,
        validator: MarketRsActivationValidator,
    ) -> None:
        self.repository = repository
        self.feature_run_repository_factory = feature_run_repository_factory
        self.validator = validator

    @staticmethod
    def _validate_feature_candidate(
        feature,
        *,
        market: str,
        formula_version: str,
        through_date,
        market_rs_run,
    ) -> None:
        status = getattr(feature.status, "value", feature.status)
        if (
            status != RunStatus.PUBLISHED.value
            or feature.as_of_date != through_date
        ):
            raise MarketRsActivationRejected(
                (
                    "Candidate Feature run changed or no longer matches the "
                    "validated Market RS run.",
                )
            )
        expected = RsPublicationIdentity(
            snapshot=GroupSnapshotIdentity(
                market,
                through_date,
                formula_version,
            ),
            market_rs_run_id=market_rs_run.id,
            universe_size=market_rs_run.eligible_symbol_count,
        )
        try:
            resolved = resolve_feature_run_rs_identity(
                feature,
                ranking_date=through_date,
            )
        except FeatureRunRsIdentityError as exc:
            raise MarketRsActivationRejected(
                (
                    "Candidate Feature run changed or no longer matches the "
                    f"validated Market RS run: {exc}",
                )
            ) from exc
        if resolved.publication != expected:
            raise MarketRsActivationRejected(
                (
                    "Candidate Feature run changed or no longer matches the "
                    "validated Market RS run.",
                )
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
        normalized = normalize_rollout_market(market)
        if not validation.ok:
            raise MarketRsActivationRejected(validation.errors)
        if (
            validation.market != normalized
            or validation.formula_version != formula_version
            or validation.feature_run_id != feature_run_id
            or formula_version != BALANCED_RS_FORMULA_VERSION
            or validation.latest_market_rs_run_id is None
        ):
            raise MarketRsActivationRejected(
                ("Activation request does not match its validation report.",)
            )

        manifest_path = Path(static_staging_dir) / "manifest.json"
        if not manifest_path.is_file():
            raise MarketRsActivationRejected(
                ("Validated static manifest disappeared before activation.",)
            )
        current_hash = MarketRsStaticArtifactValidator.manifest_hash(manifest_path)
        if current_hash != validation.static_manifest_sha256:
            raise MarketRsActivationRejected(
                ("Validated static manifest changed after validation.",)
            )
        static_errors = self.validator.revalidate_static(
            db,
            market=normalized,
            through_date=validation.through_date,
            feature_run_id=feature_run_id,
            static_staging_dir=Path(static_staging_dir),
        )
        post_validation_hash = MarketRsStaticArtifactValidator.manifest_hash(
            manifest_path
        )
        if static_errors or post_validation_hash != current_hash:
            raise MarketRsActivationRejected(
                static_errors
                or (
                    "Validated static manifest changed during activation "
                    "revalidation.",
                )
            )

        try:
            current_run = self.repository.get_completed_exact(
                db,
                market=normalized,
                as_of_date=validation.through_date,
                formula_version=formula_version,
            )
            if (
                current_run is None
                or current_run.id != validation.latest_market_rs_run_id
                or current_run.universe_hash != validation.latest_universe_hash
            ):
                raise MarketRsActivationRejected(
                    ("Validated Market RS run changed before activation.",)
                )
            feature_repo = self.feature_run_repository_factory(db)
            feature = feature_repo.get_run(feature_run_id)
            self._validate_feature_candidate(
                feature,
                market=normalized,
                formula_version=formula_version,
                through_date=validation.through_date,
                market_rs_run=current_run,
            )
            if feature.universe_hash != validation.feature_universe_hash:
                raise MarketRsActivationRejected(
                    ("Validated Feature universe changed before activation.",)
                )
            self.repository.activate_formula(
                db,
                market=normalized,
                formula_version=formula_version,
            )
            feature_repo.repoint_published(
                feature_run_id,
                pointer_key=f"latest_published_market:{normalized}",
            )
            db.commit()
        except Exception:
            db.rollback()
            raise


__all__ = ["MarketRsActivator"]
