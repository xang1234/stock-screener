from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from sqlalchemy import func, select

from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.company_exposure_work_repo import (
    CompanyExposureWorkRepository,
    WorkLeaseError,
)
from app.models.company_exposure import (
    AssessmentRevision,
    ExposureClaimRevision,
    ResearchProviderAttempt,
)
from app.services.company_exposure.acquisition import DocumentAcquisitionRegistry
from app.services.company_exposure.claims import ClaimVerifier
from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.issuer_identity import (
    IssuerIdentityAdapter,
    LinkProposal,
)
from app.services.company_exposure.markets.us import USDocumentAdapter
from app.services.company_exposure.network import PublicDocumentTransport
from app.services.company_exposure.providers import (
    SubscriptionArtifactRunner,
    SubscriptionProvider,
    default_client_factory,
)
from app.services.company_exposure.research import (
    ExposureResearchCoordinator,
    ResearchRequestInput,
    ResearchUnavailable,
    ThemeContext,
)
from app.services.company_exposure.resources import ResearchResources
from app.services.company_exposure.storage import OriginalStore
from app.tasks import company_exposure_tasks
from tests.fixtures.company_exposure.factory import (
    FakeGoTransport,
    FakeRateGate,
    SecMock,
    make_security,
    make_theme,
)

US_FIXTURES = (
    Path(__file__).resolve().parents[2] / "fixtures/company_exposure/documents/us"
)
TICKERS = json.loads((US_FIXTURES / "company_tickers_exchange.json").read_text())
SUBMISSIONS = json.loads((US_FIXTURES / "submissions_example.json").read_text())
REPORT_URL = "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000011/exmp-20251231a.htm"
AGENT = "Research Ops ops@example.com"
ADMIN = AdminPrincipal(
    subject="test:admin",
    auth_method="admin_api_key",
    roles=frozenset({"taxonomy:review"}),
)
SHADOW = ExposureRuntimeConfig(
    research_mode="shadow",
    text_route_enabled=True,
    subscription_key_present=True,
    daily_request_limit=10,
    daily_token_limit=200_000,
    sec_user_agent=AGENT,
)
QUOTE = "The ET-9000 supports testing of high-bandwidth memory (HBM) devices."


def _claims_for(request_json: dict) -> dict:
    ref = next(p["ref"] for p in request_json["passages"] if QUOTE in p["text"])
    return {
        "claims": [
            {
                "claim_kind": "product_application",
                "product_or_activity_key": "et-9000",
                "product_terms": ["ET-9000"],
                "reporting_scope": "issuer_consolidated",
                "commercial_status": "commercially_available",
                "statement": "The ET-9000 supports HBM testing.",
                "support": [{"ref": ref, "quote": QUOTE}],
            }
        ]
    }


class Harness:
    def __init__(self, db, tmp_path, clock, config=SHADOW):
        self.db = db
        self.clock = clock
        self.sec = SecMock()
        self.go = FakeGoTransport()
        self.rate = FakeRateGate()
        self.theme = make_theme(db, "ai-memory")
        self.security = make_security(db, "EXMP")
        self.store = OriginalStore(
            db,
            tmp_path / "store",
            max_bytes=10**9,
            min_free_bytes=0,
            clock=clock.now,
            disk_free=lambda _p: 10**12,
        )
        self.build(config)

    def build(self, config):
        self.config = config
        acquisition = DocumentAcquisitionRegistry(
            self.db,
            transport=PublicDocumentTransport(
                transport=httpx.MockTransport(self.sec.handler),
                resolver=lambda host: ["93.184.216.34"],
            ),
            store=self.store,
            rate_gate=self.rate,
            user_agent=config.sec_user_agent,
            clock=self.clock.now,
        )
        runner = SubscriptionArtifactRunner(
            self.db,
            ResearchResources(self.db, config, clock=self.clock.now),
            SubscriptionProvider(
                api_key="test-key",
                client_factory=default_client_factory(self.go.transport),
            ),
        )
        self.coordinator = ExposureResearchCoordinator(
            self.db,
            config,
            us_adapter=USDocumentAdapter(
                self.db, acquisition, user_agent=config.sec_user_agent
            ),
            verifier=ClaimVerifier(runner),
            store=self.store,
            theme_loader=lambda _s, theme_id: ThemeContext(
                theme_id, "AI Memory", ("HBM", "high-bandwidth memory"), "f" * 64
            ),
            clock=self.clock.now,
        )
        self.repo = CompanyExposureWorkRepository(self.db, clock=self.clock.now)

    def serve_sec(self):
        self.sec.serve_json(
            "https://www.sec.gov/files/company_tickers_exchange.json", TICKERS
        )
        self.sec.serve_json(
            "https://data.sec.gov/submissions/CIK0001234567.json", SUBMISSIONS
        )
        self.sec.serve_bytes(
            REPORT_URL, (US_FIXTURES / "annual_report_example.htm").read_bytes()
        )

    def request(self, key="verify-1", kind="verify"):
        return self.coordinator.request(
            ResearchRequestInput(
                economic_theme_id=self.theme.id, kind=kind, security_id=self.security.id
            ),
            "test:admin",
            idempotency_key=key,
        )

    def step(self):
        item = self.repo.claim_next(worker_id="test-worker")
        assert item is not None, "no runnable work"
        self.db.commit()
        return self.coordinator.run_step(item.id, item.lease_token)

    def run_all(self, limit=6):
        results = []
        for _ in range(limit):
            item = self.repo.claim_next(worker_id="test-worker")
            if item is None:
                break
            self.db.commit()
            results.append(self.coordinator.run_step(item.id, item.lease_token))
        return results


@pytest.fixture
def harness(db_session, tmp_path, clock):
    return Harness(db_session, tmp_path, clock)


@pytest.mark.case("R05")
@pytest.mark.exposure_layer("unit")
def test_repeat_request_reuses_job(harness):
    first = harness.request()
    repeated = harness.request()
    assert first.id == repeated.id
    assert (first.created, repeated.created) == (True, False)
    assert harness.run_all()[0].stage == "resolve_issuer"


def test_disabled_research_and_discovery_are_refused(harness, db_session):
    harness.build(replace(SHADOW, research_mode="disabled"))
    with pytest.raises(ResearchUnavailable, match="research_disabled"):
        harness.request()
    harness.build(SHADOW)
    with pytest.raises(ResearchUnavailable, match="discovery_not_installed"):
        harness.request(kind="discover")


def test_offline_us_verify_resolves_cik_then_assesses(harness, db_session):
    harness.serve_sec()
    harness.go.queue_builder(_claims_for)
    ref = harness.request()
    results = harness.run_all()
    assert [(r.stage, r.status) for r in results] == [
        ("resolve_issuer", "completed"),
        ("acquire", "completed"),
        ("verify", "completed"),
    ]
    assert results[0].detail["source"] == "official_registry"
    assert results[0].detail["acceptance_policy"] == "official_registry_single_listing"
    # The older 10-K is not served: coverage is partial, not an exposure change.
    assert results[-1].state == "partial"
    assert results[-1].detail["claims"] == 1 and results[-1].detail["shadow"] is True
    assert len(harness.go.requests) == 1
    assert set(harness.rate.provider_names) == {"sec_edgar"}

    revision = db_session.get(
        AssessmentRevision, UUID(results[-1].detail["assessment_revision_id"])
    )
    assert revision.request_id == ref.id
    claim = db_session.execute(select(ExposureClaimRevision)).scalar_one()
    assert (claim.support_basis, claim.conclusion) == ("primary_explicit", "supported")
    assert harness.coordinator.status(ref.id)["state"] == "partial"


def test_identical_refresh_reuses_artifact_without_new_spend(harness, db_session):
    harness.serve_sec()
    harness.go.queue_builder(_claims_for)
    harness.request()
    harness.run_all()
    attempts = db_session.execute(
        select(func.count()).select_from(ResearchProviderAttempt)
    ).scalar()
    harness.request(key="refresh-1", kind="refresh")
    results = harness.run_all()
    assert results[-1].detail["unchanged"] is True
    assert len(harness.go.requests) == 1
    assert (
        db_session.execute(
            select(func.count()).select_from(ResearchProviderAttempt)
        ).scalar()
        == attempts
    )


def test_ambiguous_cik_pauses_for_review_then_resumes_after_admin_link(
    harness, db_session
):
    duplicated = {
        "fields": TICKERS["fields"],
        "data": [*TICKERS["data"], [7654321, "Other Corp", "EXMP", "NYSE"]],
    }
    harness.serve_sec()
    harness.sec.serve_json(
        "https://www.sec.gov/files/company_tickers_exchange.json", duplicated
    )
    ref = harness.request()
    paused = harness.step()
    assert (paused.status, paused.state) == ("paused", "review_required")
    assert paused.detail["condition"] == "multiple_ciks"
    assert (
        harness.coordinator.status(ref.id)["stages"][0]["pause_reason"]
        == "review_required"
    )
    assert harness.repo.claim_next(worker_id="w") is None

    identity = IssuerIdentityAdapter(db_session)
    proposal = identity.propose_link(
        LinkProposal(
            security_id=harness.security.id,
            issuer_id=None,
            identifiers=(("US", "cik", "1234567"),),
            evidence={"reference": "10-K cover page"},
            requested_by="test:admin",
            reason="administrator-resolved CIK",
        )
    )
    identity.apply_link(proposal.link_revision_id, ADMIN, proposal.proposal_hash)
    harness.coordinator.resume(ref.id)
    db_session.commit()
    resumed = harness.step()
    assert (resumed.stage, resumed.status) == ("resolve_issuer", "completed")
    assert resumed.detail["source"] == "accepted_link"


def test_missing_allocation_pauses_without_dispatch(harness, db_session):
    harness.build(replace(SHADOW, daily_request_limit=None))
    harness.serve_sec()
    harness.request()
    results = harness.run_all()
    assert (results[-1].status, results[-1].state) == ("paused", "paused_allowance")
    assert results[-1].detail["condition"] == "allocation_not_configured"
    assert harness.go.requests == []


def test_unapproved_route_is_an_unavailable_capability(harness):
    harness.build(replace(SHADOW, text_route_enabled=False))
    harness.serve_sec()
    harness.request()
    results = harness.run_all()
    assert (results[-1].state, results[-1].detail["condition"]) == (
        "unavailable_capability",
        "route_not_approved",
    )


def test_missing_user_agent_pauses_before_network(harness):
    harness.build(replace(SHADOW, sec_user_agent=""))
    harness.request()
    result = harness.step()
    assert (result.state, result.detail["condition"]) == (
        "unavailable_capability",
        "sec_user_agent_not_configured",
    )
    assert harness.sec.requests == []


def test_stale_lease_cannot_run_a_step(harness):
    harness.request()
    item = harness.repo.claim_next(worker_id="w")
    harness.db.commit()
    with pytest.raises(WorkLeaseError):
        harness.coordinator.run_step(item.id, item.id)


def test_worker_task_runs_leased_stages(harness, monkeypatch):
    harness.serve_sec()
    harness.go.queue_builder(_claims_for)
    harness.request()
    harness.db.commit()
    monkeypatch.setattr(
        "app.services.company_exposure.config.load_config", lambda settings=None: SHADOW
    )
    outcome = company_exposure_tasks.process_exposure_work.run(
        max_steps=5, coordinator_factory=lambda _session, _config: harness.coordinator
    )
    assert outcome["status"] == "completed"
    assert [s["stage"] for s in outcome["steps"]] == [
        "resolve_issuer",
        "acquire",
        "verify",
    ]


def test_theme_context_reads_latest_sealed_definition(db_session):
    from datetime import datetime, timezone

    from app.models.economic_taxonomy import (
        EconomicThemeAlias,
        EconomicThemeRevision,
        TaxonomyVersion,
    )
    from app.services.company_exposure.research import load_theme_context

    theme = make_theme(db_session, "context")
    assert load_theme_context(db_session, theme.id) is None
    version = TaxonomyVersion(status="draft", created_by="test", reason="seed")
    db_session.add(version)
    db_session.flush()
    db_session.add(
        EconomicThemeRevision(
            taxonomy_version_id=version.id,
            theme_id=theme.id,
            display_name="AI Memory",
            definition="Memory for AI accelerators",
            mechanism="HBM demand",
            lifecycle="established",
            lifecycle_policy_version="v1",
            created_by="test",
        )
    )
    db_session.flush()
    db_session.add(
        EconomicThemeAlias(
            taxonomy_version_id=version.id,
            theme_id=theme.id,
            alias="HBM",
            normalized_alias="hbm",
            created_by="test",
        )
    )
    db_session.flush()
    assert load_theme_context(db_session, theme.id) is None  # drafts are not used
    version.status = "sealed"
    version.sealed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db_session.flush()
    context = load_theme_context(db_session, theme.id)
    assert (context.label, context.terms) == ("AI Memory", ("AI Memory", "HBM"))
    assert len(context.fingerprint) == 64
