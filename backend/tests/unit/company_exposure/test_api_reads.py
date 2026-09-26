from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from app.database import get_db
from app.main import app
from tests.fixtures.company_exposure.research_harness import Harness, claims_for

ADMIN_HEADERS = {"X-Admin-Key": "admin-secret"}
BASE = "/api/v1/company-exposures"


@pytest.fixture
def call(db_session, monkeypatch):
    from app.api.v1 import config
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    app.dependency_overrides[get_db] = lambda: db_session

    async def request(method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    yield request
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def completed_shadow_job(db_session, tmp_path, clock):
    harness = Harness(db_session, tmp_path, clock)
    harness.serve_sec()
    harness.go.queue_builder(claims_for)
    ref = harness.request()
    results = harness.run_all()
    db_session.commit()
    return ref, results[-1].detail["assessment_revision_id"]


@pytest.mark.exposure_layer("api")
@pytest.mark.asyncio
async def test_shadow_preview_route_is_labeled_and_job_scoped(
    call, completed_shadow_job
):
    job, revision_id = completed_shadow_job
    response = await call(
        "GET", f"{BASE}/research-jobs/{job.id}/preview", headers=ADMIN_HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert body["view_kind"] == "shadow_preview"
    assert body["authoritative_membership"] is False
    assert body["assessment_revision_id"] == revision_id
    claim = body["claims"][0]
    assert (claim["claim_kind"], claim["support_basis"]) == (
        "product_application",
        "primary_explicit",
    )
    assert (
        claim["materiality"]["display"]
        == "Not separately disclosed in reviewed evidence"
    )
    evidence = claim["evidence"][0]
    assert evidence["evidence_role"] == "original_primary"
    assert "high-bandwidth memory" in evidence["quote"]
    assert any(c["reason"] == "http_status_404" for c in body["coverage"])


@pytest.mark.asyncio
async def test_job_status_is_operational_not_accepted(call, completed_shadow_job):
    job, revision_id = completed_shadow_job
    response = await call(
        "GET", f"{BASE}/research-jobs/{job.id}", headers=ADMIN_HEADERS
    )
    body = response.json()
    assert (body["view_kind"], body["accepted"]) == ("research_progress", False)
    assert body["state"] == "partial"
    assert body["assessment_revision_id"] == revision_id
    assert [s["stage"] for s in body["stages"]] == [
        "resolve_issuer",
        "acquire",
        "verify",
    ]


@pytest.mark.asyncio
async def test_preview_before_assessment_is_a_typed_conflict(
    call, db_session, tmp_path, clock
):
    harness = Harness(db_session, tmp_path, clock)
    ref = harness.request()
    db_session.commit()
    response = await call(
        "GET", f"{BASE}/research-jobs/{ref.id}/preview", headers=ADMIN_HEADERS
    )
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "assessment_not_ready",
        "state": "queued",
    }


@pytest.mark.asyncio
async def test_job_reads_require_admin_and_valid_ids(call):
    missing = await call(
        "GET", f"{BASE}/research-jobs/{uuid4()}", headers=ADMIN_HEADERS
    )
    assert missing.status_code == 404
    invalid = await call(
        "GET", f"{BASE}/research-jobs/not-a-uuid", headers=ADMIN_HEADERS
    )
    assert invalid.status_code == 422
    anonymous = await call("GET", f"{BASE}/research-jobs/{uuid4()}")
    assert anonymous.status_code == 401


def test_no_parallel_research_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert {
        f"{BASE}/research-requests",
        f"{BASE}/research-jobs/{{job_id}}",
        f"{BASE}/research-jobs/{{job_id}}/preview",
    } <= paths
    assert f"{BASE}/research" not in paths
    assert not any(path.startswith(f"{BASE}/research/") for path in paths)
    assert not any(path.startswith(f"{BASE}/securities") for path in paths)
    # Generation-bound product reads and decisions are not advertised yet.
    assert not any(path.startswith(f"{BASE}/admin") for path in paths)
    assert f"{BASE}/{{assessment_id}}" not in paths
