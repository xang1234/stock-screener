"""Offline research harness: SEC mock, fake subscription route, real services."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.company_exposure_work_repo import (
    CompanyExposureWorkRepository,
)
from app.services.company_exposure.acquisition import DocumentAcquisitionRegistry
from app.services.company_exposure.claims import ClaimVerifier
from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.markets.us import USDocumentAdapter, USIssuerResolver
from app.services.company_exposure.network import PublicDocumentTransport
from app.services.company_exposure.providers import (
    SubscriptionArtifactRunner,
    SubscriptionProvider,
    default_client_factory,
)
from app.services.company_exposure.research import (
    MarketRoute,
    ResearchStageRunner,
    ThemeContext,
)
from app.services.company_exposure.research_requests import (
    ResearchRequestInput,
    ResearchRequests,
)
from app.services.company_exposure.resources import ResearchResources
from app.services.company_exposure.storage import OriginalStore
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


def claims_for(request_json: dict) -> dict:
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
        us_adapter = USDocumentAdapter(
            self.db, acquisition, user_agent=config.sec_user_agent
        )
        self.requests = ResearchRequests(self.db, config, clock=self.clock.now)
        self.runner = ResearchStageRunner(
            self.db,
            config,
            markets={
                "US": MarketRoute(
                    us_adapter, USIssuerResolver(self.db, us_adapter).resolve_cik
                )
            },
            verifier=ClaimVerifier(runner),
            store=self.store,
            theme_loader=lambda _s, theme_id: ThemeContext(
                theme_id, "AI Memory", ("HBM", "high-bandwidth memory"), "f" * 64
            ),
            clock=self.clock.now,
        )
        acquisition.before_io = self.runner.keep_lease
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
        return self.requests.request(
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
        return self.runner.run_step(item.id, item.lease_token)

    def run_all(self, limit=6):
        results = []
        for _ in range(limit):
            item = self.repo.claim_next(worker_id="test-worker")
            if item is None:
                break
            self.db.commit()
            results.append(self.runner.run_step(item.id, item.lease_token))
        return results
