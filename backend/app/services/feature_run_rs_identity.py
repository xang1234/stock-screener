from dataclasses import dataclass
from datetime import date

from app.domain.feature_store.run_metadata import feature_run_market
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
    RsPublicationIdentity,
)


class FeatureRunRsIdentityError(RuntimeError):
    """Feature metadata cannot identify one coherent RS snapshot."""


@dataclass(frozen=True)
class FeatureRunRsIdentityResolution:
    publication: RsPublicationIdentity
    identity_source: str

    @property
    def identity(self) -> GroupSnapshotIdentity:
        return self.publication.snapshot

    @property
    def market_rs_run_id(self) -> int | None:
        return self.publication.market_rs_run_id

    @property
    def universe_size(self) -> int | None:
        return self.publication.universe_size


def resolve_feature_run_rs_identity(
    feature_run,
    *,
    ranking_date: date,
) -> FeatureRunRsIdentityResolution:
    if feature_run is None:
        raise FeatureRunRsIdentityError("Feature run does not exist")
    config = dict(
        getattr(feature_run, "config_json", None)
        or getattr(feature_run, "config", None)
        or {}
    )
    market = feature_run_market(feature_run) or config.get("market")
    if not str(market or "").strip():
        raise FeatureRunRsIdentityError("Feature run has no market identity")
    formula = config.get("rs_formula_version")
    canonical_values = (
        config.get("market_rs_run_id"),
        config.get("rs_as_of_date"),
        config.get("rs_universe_size"),
    )
    if formula is not None and str(formula).strip():
        normalized_formula = str(formula).strip()
        configured_as_of = config.get("rs_as_of_date")
        if (
            configured_as_of is not None
            and str(configured_as_of) != ranking_date.isoformat()
        ):
            raise FeatureRunRsIdentityError(
                f"Feature run RS date {configured_as_of} does not match "
                f"ranking date {ranking_date.isoformat()}"
            )
        if normalized_formula == BALANCED_RS_FORMULA_VERSION and any(
            value is None for value in canonical_values
        ):
            raise FeatureRunRsIdentityError(
                "Feature run has partial canonical RS metadata"
            )
        try:
            market_rs_run_id = (
                int(config["market_rs_run_id"])
                if config.get("market_rs_run_id") is not None
                else None
            )
            universe_size = (
                int(config["rs_universe_size"])
                if config.get("rs_universe_size") is not None
                else None
            )
            publication = RsPublicationIdentity(
                snapshot=GroupSnapshotIdentity(
                    market,
                    ranking_date,
                    normalized_formula,
                ),
                market_rs_run_id=market_rs_run_id,
                universe_size=universe_size,
            )
        except (TypeError, ValueError) as exc:
            raise FeatureRunRsIdentityError(
                f"Feature run has invalid canonical RS metadata: {exc}"
            ) from exc
        return FeatureRunRsIdentityResolution(
            publication=publication,
            identity_source="persisted",
        )
    if all(value is None for value in canonical_values):
        return FeatureRunRsIdentityResolution(
            publication=RsPublicationIdentity(
                snapshot=GroupSnapshotIdentity(
                    market,
                    ranking_date,
                    LEGACY_RS_FORMULA_VERSION,
                ),
                market_rs_run_id=None,
                universe_size=None,
            ),
            identity_source="inferred_legacy",
        )
    raise FeatureRunRsIdentityError(
        "Feature run has partial canonical RS metadata without rs_formula_version"
    )


def feature_run_matches_rs_source(
    feature_run,
    *,
    identity: GroupSnapshotIdentity,
    market_rs_run_id: int | None,
    universe_size: int | None = None,
) -> bool:
    try:
        resolved = resolve_feature_run_rs_identity(
            feature_run,
            ranking_date=identity.as_of_date,
        )
    except FeatureRunRsIdentityError:
        return False
    if resolved.identity != identity:
        return False
    if resolved.market_rs_run_id != market_rs_run_id:
        return False
    return universe_size is None or resolved.universe_size == universe_size
