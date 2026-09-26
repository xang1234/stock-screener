from __future__ import annotations

from dataclasses import replace

import pytest

from app.services.company_exposure.activation import STAGES, evaluate
from app.services.company_exposure.config import ExposureRuntimeConfig
from tests.fixtures.company_exposure.research_harness import SHADOW


@pytest.fixture
def us_shadow_capability(tmp_path):
    return replace(SHADOW, document_store=str(tmp_path))


@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_us_shadow_allowed_while_full_admission_is_held(us_shadow_capability):
    assert evaluate(
        "shadow_verify_us", us_shadow_capability, check_storage=True
    ).allowed
    for stage in STAGES[1:]:
        decision = evaluate(stage, us_shadow_capability)
        assert (decision.allowed, decision.reasons) == (False, ("not_installed",))


@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_defaults_and_live_mode_do_not_activate():
    assert evaluate("shadow_verify_us", ExposureRuntimeConfig()).reasons == (
        "research_disabled",
        "text_route_not_enabled",
        "subscription_credentials_missing",
        "allocation_not_configured",
        "sec_user_agent_not_configured",
    )
    live = evaluate("shadow_verify_us", replace(SHADOW, research_mode="live"))
    assert "live_mode_not_installed" in live.reasons


def test_unwritable_store_is_a_typed_reason(tmp_path):
    missing = replace(SHADOW, document_store=str(tmp_path / "absent"))
    assert evaluate("shadow_verify_us", missing, check_storage=True).reasons == (
        "storage_not_writable",
    )


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        evaluate("everything", SHADOW)
