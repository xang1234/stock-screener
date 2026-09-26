from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.models.company_exposure import ExposureResearchRequest, ResearchProviderAttempt
from app.services.company_exposure.config import load_config
from app.services.company_exposure.research import (
    ExposureResearchCoordinator,
    ResearchRequestInput,
    ResearchUnavailable,
)
from app.tasks import company_exposure_tasks
from tests.fixtures.company_exposure.factory import make_security, make_theme


@pytest.fixture
def credentials_present(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "opencode_go_api_key", "configured-but-not-enabling")
    monkeypatch.setattr(
        settings, "exposure_sec_user_agent", "Research Ops ops@example.com"
    )
    monkeypatch.setattr(settings, "exposure_llm_text_route_enabled", True)
    monkeypatch.setattr(settings, "exposure_llm_daily_request_limit", 100)
    return settings


@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_credentials_do_not_enable_research(credentials_present, db_session):
    config = load_config(credentials_present)
    assert config.research_mode.value == "disabled"
    assert config.public_status()["subscription_key_present"] is True
    assert "configured-but-not-enabling" not in repr(config.public_status())

    outcome = company_exposure_tasks.process_exposure_work.run(
        max_steps=3,
        coordinator_factory=lambda *_: pytest.fail("no coordinator in disabled mode"),
    )
    assert (outcome["status"], outcome["reason"]) == ("skipped", "research_disabled")

    security = make_security(db_session, "EXMP")
    theme = make_theme(db_session, "disabled")
    with pytest.raises(ResearchUnavailable, match="research_disabled"):
        ExposureResearchCoordinator(db_session, config).request(
            ResearchRequestInput(economic_theme_id=theme.id, security_id=security.id),
            "test:admin",
            idempotency_key="disabled-1",
        )
    for model in (ExposureResearchRequest, ResearchProviderAttempt):
        assert db_session.execute(select(func.count()).select_from(model)).scalar() == 0
