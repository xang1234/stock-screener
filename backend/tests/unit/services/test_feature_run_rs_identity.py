from datetime import date
from types import SimpleNamespace

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
    LEGACY_RS_FORMULA_VERSION,
)
from app.services.feature_run_rs_identity import (
    FeatureRunRsIdentityError,
    feature_run_matches_rs_source,
    resolve_feature_run_rs_identity,
)


def test_feature_identity_infers_only_fully_legacy_metadata():
    run = SimpleNamespace(
        as_of_date=date(2026, 4, 10),
        config_json={"market": "US", "universe": {"market": "US"}},
    )
    resolution = resolve_feature_run_rs_identity(
        run, ranking_date=date(2026, 4, 10)
    )
    assert resolution.identity.formula_version == LEGACY_RS_FORMULA_VERSION
    assert resolution.identity_source == "inferred_legacy"


def test_partial_canonical_feature_metadata_is_rejected():
    run = SimpleNamespace(
        as_of_date=date(2026, 4, 10),
        config_json={
            "market": "US",
            "universe": {"market": "US"},
            "market_rs_run_id": 7,
        },
    )
    with pytest.raises(FeatureRunRsIdentityError, match="partial canonical"):
        resolve_feature_run_rs_identity(run, ranking_date=date(2026, 4, 10))


def test_persisted_feature_identity_rejects_a_different_ranking_date():
    run = SimpleNamespace(
        as_of_date=date(2026, 4, 9),
        config_json={
            "market": "US",
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 42,
            "rs_as_of_date": "2026-04-09",
            "rs_universe_size": 100,
        },
    )

    with pytest.raises(FeatureRunRsIdentityError, match="RS date"):
        resolve_feature_run_rs_identity(run, ranking_date=date(2026, 4, 10))


def test_persisted_feature_identity_exposes_exact_run_and_universe():
    run = SimpleNamespace(
        as_of_date=date(2026, 4, 10),
        config_json={
            "market": "US",
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 42,
            "rs_as_of_date": "2026-04-10",
            "rs_universe_size": 100,
        },
    )

    resolution = resolve_feature_run_rs_identity(
        run,
        ranking_date=date(2026, 4, 10),
    )

    assert resolution.market_rs_run_id == 42
    assert resolution.universe_size == 100


def test_feature_source_match_includes_universe_size():
    run = SimpleNamespace(
        as_of_date=date(2026, 4, 10),
        config_json={
            "market": "US",
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 42,
            "rs_as_of_date": "2026-04-10",
            "rs_universe_size": 100,
        },
    )

    assert not feature_run_matches_rs_source(
        run,
        identity=GroupSnapshotIdentity(
            "US",
            date(2026, 4, 10),
            BALANCED_RS_FORMULA_VERSION,
        ),
        market_rs_run_id=42,
        universe_size=99,
    )
