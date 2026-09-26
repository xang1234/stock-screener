from __future__ import annotations

from dataclasses import replace

import pytest

from app.services.company_exposure.activation import evaluate
from app.services.company_exposure.config import ExposureRuntimeConfig
from tests.fixtures.company_exposure.research_harness import SHADOW


@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_configured_us_shadow_is_allowed(tmp_path):
    decision = evaluate(
        replace(SHADOW, document_store=str(tmp_path)), check_storage=True
    )
    assert (decision.stage, decision.allowed, decision.reasons) == (
        "shadow_verify_us",
        True,
        (),
    )


@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_defaults_and_live_mode_do_not_activate():
    assert evaluate(ExposureRuntimeConfig()).reasons == (
        "research_disabled",
        "text_route_not_enabled",
        "subscription_credentials_missing",
        "allocation_not_configured",
        "sec_user_agent_not_configured",
    )
    live = evaluate(replace(SHADOW, research_mode="live"))
    assert "live_mode_not_installed" in live.reasons


def test_unwritable_store_is_a_typed_reason(tmp_path):
    missing = replace(SHADOW, document_store=str(tmp_path / "absent"))
    assert evaluate(missing, check_storage=True).reasons == ("storage_not_writable",)
