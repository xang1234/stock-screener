from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

import app.models  # noqa: F401
from app.infra.db.repositories.company_exposure_work_repo import (
    ROOT_PROVIDER_ATTEMPTS,
    CompanyExposureWorkRepository,
)
from app.models.company_exposure import (
    ResearchArtifact,
    ResearchProviderAttempt,
    ResearchRootBudget,
)
from app.models.economic_taxonomy_runtime_common import ImmutableRuntimePayload
from tests.fixtures.company_exposure.factory import make_security, make_theme

VERSIONS = Path(__file__).resolve().parents[3] / "alembic/versions"
TASK02_TABLES = (
    "company_exposure_runtime_policy_revisions",
    "company_exposure_research_requests",
    "company_exposure_research_events",
    "company_exposure_research_candidates",
    "company_exposure_work_items",
    "company_exposure_input_manifests",
    "company_exposure_resource_pools",
    "company_exposure_root_budgets",
    "company_exposure_reservations",
    "company_exposure_reservation_events",
    "company_exposure_provider_attempts",
    "company_exposure_provider_results",
    "company_exposure_artifacts",
    "company_exposure_coverage_items",
)


@pytest.fixture
def repo(db_session):
    return CompanyExposureWorkRepository(db_session)


@pytest.fixture
def root_request(db_session, repo):
    theme = make_theme(db_session)
    security = make_security(db_session, "ACME")
    request, created = repo.create_request(
        kind="discover",
        requester_principal="test:admin",
        idempotency_namespace="test:admin",
        idempotency_key="root-1",
        economic_theme_id=theme.id,
        security_id=security.id,
    )
    assert created
    return request


@pytest.mark.case("R05")
@pytest.mark.exposure_layer("schema")
def test_children_reference_one_root_budget(db_session, repo, root_request):
    children = [
        repo.create_request(
            kind="verify",
            requester_principal="system:company-exposure-research",
            idempotency_namespace="child",
            idempotency_key=f"child-{index}",
            economic_theme_id=root_request.economic_theme_id,
            security_id=root_request.security_id,
            parent=root_request,
        )[0]
        for index in range(10)
    ]
    db_session.flush()
    assert {child.root_request_id for child in children} == {root_request.id}
    budgets = db_session.query(ResearchRootBudget).all()
    assert {b.root_request_id for b in budgets} == {root_request.id}
    attempts = next(b for b in budgets if b.budget_key == ROOT_PROVIDER_ATTEMPTS)
    assert attempts.limit_amount == 24


def test_repeated_idempotency_key_reuses_request(repo, root_request):
    again, created = repo.create_request(
        kind="discover",
        requester_principal="test:admin",
        idempotency_namespace="test:admin",
        idempotency_key="root-1",
        economic_theme_id=root_request.economic_theme_id,
        security_id=root_request.security_id,
    )
    assert (again.id, created) == (root_request.id, False)


def test_attempt_numbers_are_unique_per_logical_operation(db_session):
    for number in (1, 1):
        db_session.add(
            ResearchProviderAttempt(
                logical_operation_key="op-1",
                attempt_number=number,
                operation="claim_review",
                route="opencode-go",
                model="kimi-k2.6",
                parameters={},
                input_hash="a" * 64,
                policy_hash="b" * 64,
            )
        )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_artifacts_are_append_only(db_session):
    artifact = ResearchArtifact(
        operation="claim_review",
        input_hash="a" * 64,
        policy_hash="b" * 64,
        model_identity="opencode-go/kimi-k2.6",
        payload={"claims": []},
        payload_hash="c" * 64,
    )
    db_session.add(artifact)
    db_session.flush()
    artifact.payload = {"claims": ["rewritten"]}
    with pytest.raises(ImmutableRuntimePayload):
        db_session.flush()
    db_session.rollback()
