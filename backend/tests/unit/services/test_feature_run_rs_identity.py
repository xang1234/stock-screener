from datetime import date
from types import SimpleNamespace

import pytest

from app.domain.relative_strength import LEGACY_RS_FORMULA_VERSION
from app.services.feature_run_rs_identity import (
    FeatureRunRsIdentityError,
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
