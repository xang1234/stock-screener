from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
from sqlalchemy import func, select

from app.api.v1 import company_exposures
from app.database import get_db
from app.main import app
from app.models.company_exposure import (
    ExposureResearchRequest,
    IssuerSecurityLinkRevision,
    ResearchProviderAttempt,
)
from tests.fixtures.company_exposure.factory import make_security, make_theme
from tests.fixtures.company_exposure.research_harness import SHADOW

ADMIN_HEADERS = {"X-Admin-Key": "admin-secret"}
PATH = "/api/v1/company-exposures/research-requests"


@pytest.fixture
def api(db_session, monkeypatch):
    from app.api.v1 import config
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(config.settings, "admin_principal_id", "test:admin")
    dispatched = []
    monkeypatch.setattr(
        company_exposures,
        "_dispatch_research",
        lambda: dispatched.append(1) or "queued",
    )
    state = {"config": SHADOW, "dispatched": dispatched}
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[company_exposures.get_exposure_config] = lambda: state[
        "config"
    ]

    async def call(method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    state["call"] = call
    yield state
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(company_exposures.get_exposure_config, None)


@pytest.fixture
def subject(db_session):
    return {
        "security": make_security(db_session, "EXMP"),
        "theme": make_theme(db_session, "api-theme"),
    }


def _body(subject, **overrides):
    return {
        "kind": "verify",
        "security_id": subject["security"].id,
        "economic_theme_id": str(subject["theme"].id),
        "idempotency_key": "verify-exmp-1",
        **overrides,
    }


def _requests(db):
    return db.execute(
        select(func.count()).select_from(ExposureResearchRequest)
    ).scalar()


@pytest.mark.case("R13")
@pytest.mark.exposure_layer("api")
@pytest.mark.asyncio
async def test_body_actor_does_not_authorize_research(api, db_session, subject):
    response = await api["call"](
        "POST", PATH, json={**_body(subject), "actor": "admin"}
    )
    assert response.status_code in {401, 403}
    assert _requests(db_session) == 0
    assert (
        db_session.execute(
            select(func.count()).select_from(ResearchProviderAttempt)
        ).scalar()
        == 0
    )
    assert api["dispatched"] == []


@pytest.mark.case("R13")
@pytest.mark.exposure_layer("api")
@pytest.mark.asyncio
async def test_actor_field_is_rejected_even_for_admin(api, db_session, subject):
    response = await api["call"](
        "POST",
        PATH,
        headers=ADMIN_HEADERS,
        json={**_body(subject), "actor": "someone-else"},
    )
    assert response.status_code == 422
    assert _requests(db_session) == 0


@pytest.mark.asyncio
async def test_admin_request_is_queued_once_and_recorded_with_trusted_identity(
    api, db_session, subject
):
    first = await api["call"]("POST", PATH, headers=ADMIN_HEADERS, json=_body(subject))
    again = await api["call"]("POST", PATH, headers=ADMIN_HEADERS, json=_body(subject))
    assert (first.status_code, again.status_code) == (202, 200)
    assert first.json()["job_id"] == again.json()["job_id"]
    assert (first.json()["dispatch"], again.json()["dispatch"]) == (
        "queued",
        "not_needed",
    )
    assert api["dispatched"] == [1]
    request = db_session.execute(select(ExposureResearchRequest)).scalar_one()
    assert request.requester_principal == "test:admin"
    assert request.market == "US"


@pytest.mark.asyncio
async def test_unbound_admin_cannot_start_research(
    api, db_session, subject, monkeypatch
):
    from app.api.v1 import config

    monkeypatch.setattr(config.settings, "admin_principal_id", "")
    response = await api["call"](
        "POST", PATH, headers=ADMIN_HEADERS, json=_body(subject)
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "admin_principal_unbound"
    assert _requests(db_session) == 0


@pytest.mark.asyncio
async def test_disabled_research_and_uninstalled_actions_are_typed(
    api, db_session, subject
):
    api["config"] = replace(SHADOW, research_mode="disabled")
    disabled = await api["call"](
        "POST", PATH, headers=ADMIN_HEADERS, json=_body(subject)
    )
    assert (disabled.status_code, disabled.json()["detail"]["code"]) == (
        409,
        "research_disabled",
    )
    api["config"] = SHADOW
    discover = await api["call"](
        "POST", PATH, headers=ADMIN_HEADERS, json=_body(subject, kind="discover")
    )
    assert (discover.status_code, discover.json()["detail"]["code"]) == (
        501,
        "discovery_not_installed",
    )
    hk = make_security(db_session, "0700.HK", market="HK", exchange="HKEX")
    other_market = await api["call"](
        "POST", PATH, headers=ADMIN_HEADERS, json=_body(subject, security_id=hk.id)
    )
    assert other_market.json()["detail"]["code"] == "market_not_installed"
    assert _requests(db_session) == 0


@pytest.mark.asyncio
async def test_supplied_cik_becomes_a_reviewable_proposal(api, db_session, subject):
    response = await api["call"](
        "POST", PATH, headers=ADMIN_HEADERS, json=_body(subject, supplied_cik="1234567")
    )
    assert response.status_code == 202
    proposal = response.json()["issuer_link_proposal"]
    assert proposal["state"] in {"proposed", "review_required"}
    link = db_session.execute(select(IssuerSecurityLinkRevision)).scalar_one()
    assert link.state != "accepted"


@pytest.mark.asyncio
async def test_request_bounds_are_validated(api, subject):
    too_many = [f"https://www.sec.gov/{i}" for i in range(6)]
    for body in (
        _body(subject, supplied_links=too_many),
        _body(subject, idempotency_key="has spaces"),
        _body(subject, supplied_cik="12345678901"),
        _body(subject, security_id=0),
    ):
        response = await api["call"]("POST", PATH, headers=ADMIN_HEADERS, json=body)
        assert response.status_code == 422
