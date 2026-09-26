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
from sqlalchemy.orm import Session

from app.api.v1.config import require_admin
from app.database import get_db
from app.domain.company_exposure.contracts import RESEARCH_STAGES
from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.schemas.company_exposure import (
    IssuerLinkProposalView,
    ResearchJobResponse,
    ResearchRequestBody,
    ResearchRequestResponse,
    ShadowPreviewResponse,
)
from app.services.company_exposure.config import ExposureRuntimeConfig, load_config
from app.services.company_exposure.reads import PreviewUnavailable, ResearchJobReader
from app.services.company_exposure.research_requests import (
    ResearchRequestInput,
    ResearchRequests,
    ResearchUnavailable,
)

logger = logging.getLogger(__name__)
router = APIRouter()
REVIEW_ROLE = "taxonomy:review"
_REFUSAL_STATUS = {
    "research_disabled": status.HTTP_409_CONFLICT,
    "live_mode_not_installed": status.HTTP_409_CONFLICT,
    "discovery_not_installed": status.HTTP_501_NOT_IMPLEMENTED,
    "security_not_found": status.HTTP_404_NOT_FOUND,
    "economic_theme_not_found": status.HTTP_404_NOT_FOUND,
    "market_not_installed": status.HTTP_422_UNPROCESSABLE_ENTITY,
}


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

        process_exposure_work.apply_async(kwargs={"max_steps": len(RESEARCH_STAGES)})
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
    try:
        ref = ResearchRequests(db, config).request(
            ResearchRequestInput(
                economic_theme_id=body.economic_theme_id,
                kind=body.kind,
                security_id=body.security_id,
                symbol=body.symbol,
                supplied_links=tuple(str(link) for link in body.supplied_links),
                supplied_cik=body.supplied_cik,
            ),
            principal.subject,
            idempotency_key=body.idempotency_key,
        )
    except ResearchUnavailable as exc:
        db.rollback()
        raise _typed(_REFUSAL_STATUS[exc.code], exc.code, **exc.detail) from None
    db.commit()
    if not ref.created:
        response.status_code = status.HTTP_200_OK
    proposal = ref.issuer_link_proposal
    return ResearchRequestResponse(
        job_id=str(ref.id),
        created=ref.created,
        state=ref.state,
        dispatch=_dispatch_research() if ref.created else "not_needed",
        issuer_link_proposal=None
        if proposal is None
        else IssuerLinkProposalView(
            state=proposal.state,
            link_revision_id=None
            if proposal.link_revision_id is None
            else str(proposal.link_revision_id),
            reason=proposal.reason,
        ),
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
