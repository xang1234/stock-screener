"""Company-exposure research operations (spec §16 routes; plan Task 21A).

Only the shadow-slice routes are registered: a queued research request, the
operational job status and a job-scoped shadow preview. Generation-bound
product reads and reviewed decisions are separate, later routes. Every route
requires the trusted admin credential; no body or header field sets the
stored actor.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.v1.config import require_admin
from app.database import get_db
from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.models.economic_taxonomy import EconomicTheme
from app.models.stock_universe import StockUniverse
from app.schemas.company_exposure import (
    IssuerLinkProposalView,
    ResearchJobResponse,
    ResearchRequestBody,
    ResearchRequestResponse,
    ShadowPreviewResponse,
)
from app.services.company_exposure.config import ExposureRuntimeConfig, load_config
from app.services.company_exposure.issuer_identity import (
    IssuerIdentityAdapter,
    LinkProposal,
)
from app.services.company_exposure.reads import PreviewUnavailable, ResearchJobReader
from app.services.company_exposure.research import (
    STAGES,
    ExposureResearchCoordinator,
    ResearchRequestInput,
    ResearchUnavailable,
)

logger = logging.getLogger(__name__)
router = APIRouter()
INSTALLED_MARKETS = frozenset({"US"})
REVIEW_ROLE = "taxonomy:review"


def get_exposure_config() -> ExposureRuntimeConfig:
    return load_config()


def _typed(status_code: int, code: str, **extra) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, **extra})


def _require_bound_admin(principal: AdminPrincipal) -> AdminPrincipal:
    if REVIEW_ROLE not in (principal.roles or ()):
        raise _typed(status.HTTP_403_FORBIDDEN, "admin_principal_unbound")
    return principal


def _dispatch_research() -> str:
    """Queue the dedicated research worker; the job stays queued if not."""

    try:
        from app.tasks.company_exposure_tasks import process_exposure_work

        process_exposure_work.apply_async(kwargs={"max_steps": len(STAGES)})
        return "queued"
    except Exception:  # broker unavailable: the queued job is retried later
        logger.warning("exposure research dispatch deferred", exc_info=True)
        return "not_dispatched"


@router.post(
    "/research-requests",
    response_model=ResearchRequestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_research(
    body: ResearchRequestBody,
    response: Response,
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[ExposureRuntimeConfig, Depends(get_exposure_config)],
):
    principal = _require_bound_admin(principal)
    if body.kind == "discover":
        raise _typed(status.HTTP_501_NOT_IMPLEMENTED, "discovery_not_installed")
    if body.security_id is not None:
        security = db.get(StockUniverse, body.security_id)
    else:
        security = db.execute(
            select(StockUniverse).where(
                StockUniverse.symbol == body.symbol.upper(),
                StockUniverse.market.in_(INSTALLED_MARKETS),
            )
        ).scalar_one_or_none()
    if security is None:
        raise _typed(status.HTTP_404_NOT_FOUND, "security_not_found")
    if security.market not in INSTALLED_MARKETS:
        raise _typed(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "market_not_installed",
            market=security.market,
        )
    if db.get(EconomicTheme, body.economic_theme_id) is None:
        raise _typed(status.HTTP_404_NOT_FOUND, "economic_theme_not_found")

    coordinator = ExposureResearchCoordinator(db, config)
    try:
        ref = coordinator.request(
            ResearchRequestInput(
                economic_theme_id=body.economic_theme_id,
                kind=body.kind,
                security_id=security.id,
                market=security.market,
                supplied_links=tuple(str(link) for link in body.supplied_links),
            ),
            principal.subject,
            idempotency_key=body.idempotency_key,
        )
    except ResearchUnavailable as exc:
        db.rollback()
        raise _typed(status.HTTP_409_CONFLICT, exc.code) from None

    proposal = None
    if ref.created and body.supplied_cik:
        identity = IssuerIdentityAdapter(db)
        if not identity.resolve_security(security.id).resolved:
            # An administrator-supplied CIK is a reviewable proposal, never a
            # trusted identifier.
            proposed = identity.propose_link(
                LinkProposal(
                    security_id=security.id,
                    issuer_id=None,
                    identifiers=(("US", "cik", body.supplied_cik),),
                    evidence={
                        "reference": "administrator-supplied CIK",
                        "job_id": str(ref.id),
                    },
                    requested_by=principal.subject,
                    reason="administrator-supplied CIK with research request",
                )
            )
            proposal = IssuerLinkProposalView(
                state=proposed.state,
                link_revision_id=None
                if proposed.link_revision_id is None
                else str(proposed.link_revision_id),
                reason=proposed.reason,
            )
    db.commit()
    dispatch = _dispatch_research() if ref.created else "not_needed"
    if not ref.created:
        response.status_code = status.HTTP_200_OK
    return ResearchRequestResponse(
        job_id=str(ref.id),
        created=ref.created,
        state=ref.state,
        dispatch=dispatch,
        issuer_link_proposal=proposal,
    )


@router.get("/research-jobs/{job_id}", response_model=ResearchJobResponse)
def read_research_job(
    job_id: UUID,
    _principal: Annotated[AdminPrincipal, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    payload = ResearchJobReader(db).read(job_id)
    if payload is None:
        raise _typed(status.HTTP_404_NOT_FOUND, "job_not_found")
    return payload


@router.get("/research-jobs/{job_id}/preview", response_model=ShadowPreviewResponse)
def read_research_preview(
    job_id: UUID,
    _principal: Annotated[AdminPrincipal, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    try:
        return ResearchJobReader(db).preview(job_id)
    except PreviewUnavailable as exc:
        if exc.code == "job_not_found":
            raise _typed(status.HTTP_404_NOT_FOUND, exc.code) from None
        raise _typed(status.HTTP_409_CONFLICT, exc.code, state=exc.state) from None
