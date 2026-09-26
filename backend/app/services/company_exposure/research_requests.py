"""Queue verify/refresh research requests (plan Task 17A, request side).

This is the only place that decides whether a request is acceptable: the
research mode, the request kind, the listing and its market, and the theme.
It does no network or provider work; the leased stages run in
``research.ResearchStageRunner``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    RESEARCH_STAGES,
    ResearchMode,
    content_hash,
    utc_now,
)
from app.infra.db.repositories.company_exposure_work_repo import (
    CompanyExposureWorkRepository,
)
from app.models.company_exposure import ExposureResearchRequest
from app.models.economic_taxonomy import EconomicTheme
from app.models.stock_universe import StockUniverse
from app.services.company_exposure.config import ExposureRuntimeConfig
from app.services.company_exposure.issuer_identity import (
    IssuerIdentityAdapter,
    LinkProposal,
    ProposalRef,
)

POLICY_BUNDLE = "exposure-verify-v1"
VERIFY_KINDS = frozenset({"verify", "refresh"})
INSTALLED_MARKETS = frozenset({"US"})


class ResearchUnavailable(RuntimeError):
    """A typed refusal: research_disabled, live_mode_not_installed,
    discovery_not_installed, security_not_found, market_not_installed or
    economic_theme_not_found."""

    def __init__(self, code: str, **detail):
        super().__init__(code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ResearchRequestInput:
    economic_theme_id: UUID
    kind: str = "verify"
    security_id: int | None = None
    symbol: str | None = None
    supplied_links: tuple[str, ...] = ()
    supplied_cik: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchRequestRef:
    id: UUID
    created: bool
    state: str | None
    issuer_link_proposal: ProposalRef | None = None


def enqueue_stage(
    repo: CompanyExposureWorkRepository, request: ExposureResearchRequest, stage: str
):
    return repo.enqueue(
        request=request,
        stage=stage,
        input_hash=content_hash({"request": request.id, "stage": stage}),
        policy_bundle_version=POLICY_BUNDLE,
    )


class ResearchRequests:
    def __init__(
        self,
        session: Session,
        config: ExposureRuntimeConfig,
        *,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.session = session
        self.config = config
        self.repo = CompanyExposureWorkRepository(session, clock=clock)
        self.identity = IssuerIdentityAdapter(session, clock=clock)

    def _listing(self, request: ResearchRequestInput) -> StockUniverse:
        if request.security_id is not None:
            security = self.session.get(StockUniverse, request.security_id)
        else:
            security = self.session.execute(
                select(StockUniverse).where(
                    StockUniverse.symbol == request.symbol.upper()
                )
            ).scalar_one_or_none()
        if security is None:
            raise ResearchUnavailable("security_not_found")
        if security.market not in INSTALLED_MARKETS:
            raise ResearchUnavailable("market_not_installed", market=security.market)
        return security

    def request(
        self, request: ResearchRequestInput, principal: str, idempotency_key: str
    ) -> ResearchRequestRef:
        if self.config.research_mode == ResearchMode.DISABLED:
            raise ResearchUnavailable("research_disabled")
        if self.config.research_mode == ResearchMode.LIVE:
            # This build publishes nothing; only shadow research is installed.
            raise ResearchUnavailable("live_mode_not_installed")
        if request.kind not in VERIFY_KINDS:
            raise ResearchUnavailable("discovery_not_installed")
        security = self._listing(request)
        if self.session.get(EconomicTheme, request.economic_theme_id) is None:
            raise ResearchUnavailable("economic_theme_not_found")
        row, created = self.repo.create_request(
            kind=request.kind,
            requester_principal=principal,
            idempotency_namespace=f"company-exposure:{principal}",
            idempotency_key=idempotency_key,
            economic_theme_id=request.economic_theme_id,
            security_id=security.id,
            market=security.market,
            supplied_links=list(request.supplied_links),
            limits=self.config.limits,
        )
        proposal = None
        if created:
            enqueue_stage(self.repo, row, RESEARCH_STAGES[0])
            if request.supplied_cik:
                proposal = self._propose_cik(
                    security.id, request.supplied_cik, principal, row.id
                )
        return ResearchRequestRef(
            row.id, created, self.repo.latest_state(row.id), proposal
        )

    def _propose_cik(
        self, security_id: int, cik: str, principal: str, job_id: UUID
    ) -> ProposalRef | None:
        """A supplied CIK is a reviewable proposal, never a trusted identifier."""

        if self.identity.resolve_security(security_id).resolved:
            return None
        return self.identity.propose_link(
            LinkProposal(
                security_id=security_id,
                issuer_id=None,
                identifiers=(("US", "cik", cik),),
                evidence={
                    "reference": "administrator-supplied CIK",
                    "job_id": str(job_id),
                },
                requested_by=principal,
                reason="administrator-supplied CIK with research request",
            )
        )

    def resume(self, request_id: UUID) -> None:
        self.repo.resume(request_id)


__all__ = (
    "INSTALLED_MARKETS",
    "ResearchRequestInput",
    "ResearchRequestRef",
    "ResearchRequests",
    "ResearchUnavailable",
    "enqueue_stage",
)
