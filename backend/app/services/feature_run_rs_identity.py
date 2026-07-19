from dataclasses import dataclass
from datetime import date

from app.domain.feature_store.run_metadata import feature_run_market
from app.domain.relative_strength import (
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)


class FeatureRunRsIdentityError(RuntimeError):
    """Feature metadata cannot identify one coherent RS snapshot."""


@dataclass(frozen=True)
class FeatureRunRsIdentityResolution:
    identity: GroupSnapshotIdentity
    identity_source: str


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
        return FeatureRunRsIdentityResolution(
            identity=GroupSnapshotIdentity(market, ranking_date, str(formula)),
            identity_source="persisted",
        )
    if all(value is None for value in canonical_values):
        return FeatureRunRsIdentityResolution(
            identity=GroupSnapshotIdentity(
                market,
                ranking_date,
                LEGACY_RS_FORMULA_VERSION,
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
    config = dict(
        getattr(feature_run, "config_json", None)
        or getattr(feature_run, "config", None)
        or {}
    )
    configured_as_of = config.get("rs_as_of_date")
    if (
        configured_as_of is not None
        and str(configured_as_of) != identity.as_of_date.isoformat()
    ):
        return False
    configured_run_id = config.get("market_rs_run_id")
    if market_rs_run_id is None:
        return configured_run_id is None
    try:
        return int(configured_run_id) == market_rs_run_id
    except (TypeError, ValueError):
        return False
