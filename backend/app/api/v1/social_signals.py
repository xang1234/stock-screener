"""Authenticated public and administrator Social Signal routes."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.social_signals import (
    MarketCode, QueueView, RankMode, SocialAssociationDecisionRequest,
    SocialCompanyIdentityUpdate, SocialEvidenceResponse, SocialQueueResponse,
    SocialRuntimeUpdate, SocialSectionResponse, SocialSourceCreateRequest,
    SocialSourceRenameRequest, SocialSourceTransitionRequest,
    SocialSourceVersionRequest, SocialSummaryResponse, WindowCode,
)
from app.services.social_signal_query_service import SocialSignalQueries
from app.api.v1.config import require_admin


router = APIRouter()
ADMIN_ACTOR = "server-admin"
_SAFE_REASON_CODES = {
    "analysis_incomplete", "bounded_provider_read", "daily_budget_exhausted",
    "invalid_provider_json", "invalid_provider_schema", "provider_error",
    "provider_lease_unavailable", "provider_network_error", "provider_timeout",
    "provider_unavailable",
    "rate_limited", "reauthentication_required", "social_runtime_changed",
    "source_participation_failed",
}


def _admin_error(exc):
    from app.services.social_source_admin_service import SocialSourceVersionError

    code = str(exc)
    if isinstance(exc, SocialSourceVersionError) or "version_conflict" in code:
        return HTTPException(status_code=409, detail={"code": code})
    if code in {"source_not_found", "candidate_not_found", "work_not_found"}:
        return HTTPException(status_code=404, detail={"code": code})
    return HTTPException(status_code=422, detail={"code": code})


def _source_payload(value, audits=()):
    payload = asdict(value)
    payload["audit"] = [asdict(item) for item in audits]
    return payload


@router.get("/summary", response_model=SocialSummaryResponse)
def summary(market: MarketCode = Query(...), db: Session = Depends(get_db)):
    return SocialSignalQueries(db).summary(market=market)


@router.get("/queue", response_model=SocialQueueResponse)
def queue(
    market: MarketCode = Query(...),
    window: WindowCode = Query("7d"),
    view: QueueView = Query("actionable"),
    rank_mode: RankMode = Query("blended"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    source: str | None = Query(None, max_length=100),
    theme: str | None = Query(None, max_length=100),
    instrument: str | None = Query(None, max_length=100),
    state: str | None = Query(None, max_length=100),
    ticker: str | None = Query(None, max_length=100),
    db: Session = Depends(get_db),
):
    return SocialSignalQueries(db).queue(
        market=market, window=window, view=view, rank_mode=rank_mode,
        page=page, page_size=page_size, filters={
            "source": source, "theme": theme, "instrument": instrument,
            "state": state, "ticker": ticker,
        },
    )


@router.get("/context", response_model=SocialSectionResponse)
def context(
    market: MarketCode = Query(...), window: WindowCode = Query("7d"),
    page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return SocialSignalQueries(db).section(
        market=market, window=window, section="context",
        page=page, page_size=page_size,
    )


@router.get("/unresolved", response_model=SocialSectionResponse)
def unresolved(
    scope: str = Query("market", pattern="^(market|unknown)$"),
    market: MarketCode = Query(...), window: WindowCode = Query("7d"),
    page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    # Unknown identities are deliberately global; market remains the caller's UI context.
    del scope
    return SocialSignalQueries(db).section(
        market=market, window=window, section="unresolved",
        page=page, page_size=page_size,
    )


@router.get("/candidates/{candidate_key}/evidence", response_model=SocialEvidenceResponse)
def evidence(
    candidate_key: str, window: WindowCode = Query("7d"),
    db: Session = Depends(get_db),
):
    return SocialSignalQueries(db).evidence(candidate_key=candidate_key, window=window)


@router.get("/theme-pulse")
def theme_pulse(market: MarketCode = Query(...), db: Session = Depends(get_db)):
    from app.infra.db.models.social_signals import SocialSignalRun, SocialSignalRunPointer

    queries = SocialSignalQueries(db)
    if not queries._supported():
        return {"supported": False, "available": False,
                "reason_code": "social_signals_disabled", "market": market, "items": []}
    pointer = db.get(SocialSignalRunPointer, "latest_published")
    if pointer is None:
        return {"supported": True, "available": False,
                "reason_code": "no_published_run", "market": market, "items": []}
    run = db.get(SocialSignalRun, pointer.run_id)
    prepared = run.application_progress_json.get("prepared", {})
    frozen = prepared.get("theme_evidence", [])
    context = prepared.get("context") or {}
    projection = prepared.get("projection") or {}
    proposals_by_theme = {}
    theme_names = {}
    for claim, resolution in zip(
        projection.get("proposals", []), projection.get("resolutions", [])
    ):
        if resolution.get("market") != market:
            continue
        theme_key = claim.get("theme_key")
        if not isinstance(theme_key, str) or not theme_key:
            continue
        proposals_by_theme.setdefault(theme_key, []).append(resolution)
        theme_names.setdefault(theme_key, claim.get("raw_theme"))

    def number(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def social_strength(theme_key):
        values = []
        symbols = {resolution.get("symbol") for resolution in proposals_by_theme.get(theme_key, [])}
        for candidate in context.get("candidates", []):
            confirmation = dict((candidate.get("confirmation") or {}).get("components", ()))
            selected_theme = (confirmation.get("theme") or {}).get("selected_key")
            candidate_market, _, symbol = candidate.get("candidate_key", "").partition(":")
            if (candidate.get("window_days") != 7 or candidate_market != market
                    or (selected_theme != theme_key and symbol not in symbols)):
                continue
            score = number((candidate.get("social_result") or {}).get("social_score"))
            if score is not None:
                values.append(score)
        return sum(values) / len(values) if values else None

    items = []
    observed_keys = set()
    for evidence in frozen:
        if evidence.get("market") != market:
            continue
        observed_keys.add(evidence["theme_key"])
        components = dict(evidence.get("components", ()))
        members = evidence.get("membership", ())
        market_values = [number(value) for value in components.values()]
        market_values = [value for value in market_values if value is not None]
        measured_counts = dict(evidence.get("measured_company_counts", ()))
        items.append({
            "theme_key": evidence["theme_key"],
            "name": theme_names.get(evidence["theme_key"])
                    or evidence["theme_key"].replace("_", " ").title(),
            "status": "confirmed" if market_values
                      else "insufficient_market_data",
            "social_strength": social_strength(evidence["theme_key"]),
            "market_strength": sum(market_values) / len(market_values) if market_values else None,
            "accepted_company_count": evidence.get("accepted_company_count", 0),
            "measured_company_count": max(measured_counts.values(), default=0),
            "benchmark_symbol": evidence.get("benchmark_symbol"),
            "components": components,
            "measured_company_counts": measured_counts,
            "reasons": dict(evidence.get("reasons", ())),
            "accepted_symbols": sorted({item["canonical_symbol"] for item in members}),
        })
    for theme_key in sorted(set(proposals_by_theme) - observed_keys):
        items.append({
            "theme_key": theme_key,
            "name": theme_names.get(theme_key) or theme_key.replace("_", " ").title(),
            "status": "discovering", "social_strength": social_strength(theme_key),
            "market_strength": None, "accepted_company_count": 0,
            "measured_company_count": 0, "benchmark_symbol": None,
            "components": {}, "measured_company_counts": {},
            "reasons": {"state": "candidate_theme"}, "accepted_symbols": [],
        })
    items.sort(key=lambda item: item["theme_key"])
    return {"supported": True, "available": True, "reason_code": None,
            "market": market, "run_id": pointer.run_id, "items": items}


@router.get("/admin/runtime", dependencies=[Depends(require_admin)])
def admin_runtime(db: Session = Depends(get_db)):
    from app.services.social_source_admin_service import SocialSourceAdminService
    return asdict(SocialSourceAdminService(db).read_runtime())


@router.patch("/admin/runtime", dependencies=[Depends(require_admin)])
def update_admin_runtime(body: SocialRuntimeUpdate, db: Session = Depends(get_db)):
    from app.services.social_source_admin_service import SocialSourceAdminService
    try:
        return asdict(SocialSourceAdminService(db).apply_runtime(
            body.mode, body.provider, body.expected_version, ADMIN_ACTOR
        ))
    except ValueError as exc:
        raise _admin_error(exc) from exc


@router.get("/admin/sources", dependencies=[Depends(require_admin)])
def admin_sources(
    include_archived: bool = Query(False), db: Session = Depends(get_db)
):
    from app.services.social_source_admin_service import SocialSourceAdminService
    service = SocialSourceAdminService(db)
    rows = service.list_sources(include_archived)
    return [_source_payload(row, service.audit_events(row.source_id)) for row in rows]


@router.post("/admin/sources", status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_admin)])
def create_admin_source(body: SocialSourceCreateRequest, db: Session = Depends(get_db)):
    from app.services.social_source_admin_service import SocialSourceAdminService
    try:
        return _source_payload(SocialSourceAdminService(db).create_source(
            body.name, body.list_ref, ADMIN_ACTOR
        ))
    except ValueError as exc:
        raise _admin_error(exc) from exc


@router.patch("/admin/sources/{source_id}", dependencies=[Depends(require_admin)])
def rename_admin_source(
    source_id: int, body: SocialSourceRenameRequest, db: Session = Depends(get_db)
):
    from app.services.social_source_admin_service import SocialSourceAdminService
    try:
        return _source_payload(SocialSourceAdminService(db).rename_source(
            source_id, body.name, body.expected_version, ADMIN_ACTOR
        ))
    except ValueError as exc:
        raise _admin_error(exc) from exc


@router.post("/admin/sources/{source_id}/test", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_admin)])
def test_admin_source(
    source_id: int, body: SocialSourceVersionRequest, db: Session = Depends(get_db)
):
    from app.interfaces.tasks.social_signal_tasks import validate_social_source
    from app.services.social_source_admin_service import SocialSourceAdminService
    try:
        request = SocialSourceAdminService(db).request_test(
            source_id, body.expected_version, ADMIN_ACTOR
        )
    except ValueError as exc:
        raise _admin_error(exc) from exc
    task = validate_social_source.apply_async(
        args=[source_id, ADMIN_ACTOR], queue="social_ingestion"
    )
    return {"task_id": task.id, "status": "queued", "request_id": request.request_id}


@router.post("/admin/sources/{source_id}/transition", dependencies=[Depends(require_admin)])
def transition_admin_source(
    source_id: int, body: SocialSourceTransitionRequest, db: Session = Depends(get_db)
):
    from app.services.social_source_admin_service import SocialSourceAdminService
    try:
        return _source_payload(SocialSourceAdminService(db).transition_source(
            source_id, body.target, body.expected_version, ADMIN_ACTOR
        ))
    except ValueError as exc:
        raise _admin_error(exc) from exc


@router.get("/admin/health", dependencies=[Depends(require_admin)])
def admin_health(db: Session = Depends(get_db)):
    from app.services.social_signal_operations_service import SocialSignalOperationsService
    return SocialSignalOperationsService().snapshot(db)


@router.get("/admin/runs", dependencies=[Depends(require_admin)])
def admin_runs(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    from app.infra.db.models.social_signals import SocialSignalRun
    rows = db.scalars(select(SocialSignalRun).order_by(
        SocialSignalRun.created_at.desc(), SocialSignalRun.id.desc()
    ).limit(limit)).all()
    return [{
        "run_id": row.id, "mode": row.mode, "provider": row.provider,
        "status": row.status, "registry_version": row.registry_version,
        "created_at": row.created_at, "completed_at": row.completed_at,
        "published_at": row.published_at,
        "source_count": len(row.application_progress_json.get("sources", {})),
        "reason_codes": sorted({reason for value in row.source_outcomes_json.values()
                                for reason in value.get("coverage_reason_codes", [])
                                if reason in _SAFE_REASON_CODES}),
    } for row in rows]


@router.get("/admin/validation/{run_id}", dependencies=[Depends(require_admin)])
def admin_validation(run_id: str, db: Session = Depends(get_db)):
    from app.infra.db.models.social_signals import SocialSignalRun, SocialSignalSnapshot
    run = db.get(SocialSignalRun, run_id)
    if run is None or run.mode != "validation":
        raise HTTPException(status_code=404, detail={"code": "validation_run_not_found"})
    rows = db.scalars(select(SocialSignalSnapshot).where(
        SocialSignalSnapshot.run_id == run_id
    ).order_by(SocialSignalSnapshot.window_days, SocialSignalSnapshot.candidate_key)).all()
    projection = run.application_progress_json.get("prepared", {}).get(
        "projection", {}
    )
    proposals = projection.get("proposals", ())
    resolutions = projection.get("resolutions", ())
    associations = [
        {**proposal, "resolution": resolution}
        for proposal, resolution in zip(proposals, resolutions)
    ]
    return {"run_id": run.id, "status": run.status, "provider": run.provider,
            "created_at": run.created_at, "completed_at": run.completed_at,
            "associations": associations,
            "candidates": [{"candidate_key": row.candidate_key, "symbol": row.canonical_symbol,
                            "market": row.market, "window_days": row.window_days,
                            "state": row.state, "social_score": row.social_score,
                            "confirmation_score": row.confirmation_score,
                            "queue_score": row.queue_score} for row in rows]}


@router.get("/admin/analysis", dependencies=[Depends(require_admin)])
def admin_analysis(
    state: str | None = Query(None), limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    query = select(SocialExtractionWork)
    if state:
        query = query.where(SocialExtractionWork.state == state)
    rows = db.scalars(query.order_by(
        SocialExtractionWork.created_at, SocialExtractionWork.id
    ).limit(limit)).all()
    return [{"work_id": row.id, "state": row.state,
             "selected_model": row.selected_model, "actual_model": row.actual_model,
             "error_code": row.error_code if row.error_code in _SAFE_REASON_CODES else (
                 "provider_error" if row.error_code else None
             ), "requested_by_admin": row.requested_by_admin,
             "created_at": row.created_at, "updated_at": row.updated_at} for row in rows]


@router.post("/admin/analysis/{work_id}/retry", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_admin)])
def retry_admin_analysis(work_id: int, db: Session = Depends(get_db)):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
    from app.infra.db.models.social_signals import SocialSignalRun
    from app.interfaces.tasks.social_signal_tasks import resume_social_analysis
    row = db.get(SocialExtractionWork, work_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "work_not_found"})
    if row.state not in {"waiting_budget", "failed_retryable", "failed_terminal", "outside_window"}:
        raise HTTPException(status_code=422, detail={"code": "work_not_retryable"})
    run_id = db.scalar(
        select(SocialRunWork.run_id)
        .join(SocialSignalRun, SocialSignalRun.id == SocialRunWork.run_id)
        .where(SocialRunWork.work_id == work_id)
        .order_by(SocialSignalRun.created_at.desc(), SocialSignalRun.id.desc())
        .limit(1)
    )
    if run_id is None:
        raise HTTPException(
            status_code=422, detail={"code": "work_generation_not_found"}
        )
    prior = (row.state, row.error_code, row.requested_by_admin)
    row.requested_by_admin = True
    row.state = "pending"
    row.error_code = None
    db.commit()
    try:
        task = resume_social_analysis.apply_async(
            args=[run_id], queue="social_ingestion"
        )
    except Exception:
        db.rollback()
        row = db.get(SocialExtractionWork, work_id)
        row.state, row.error_code, row.requested_by_admin = prior
        db.commit()
        raise
    return {
        "task_id": task.id,
        "status": "queued",
        "work_id": work_id,
        "run_id": run_id,
    }


@router.get("/admin/associations", dependencies=[Depends(require_admin)])
def admin_associations(
    state: str | None = Query(None), market: MarketCode | None = Query(None),
    db: Session = Depends(get_db),
):
    from app.infra.db.models.social_analysis import SocialThemeAssociation
    from app.models.theme import ThemeCluster
    query = select(SocialThemeAssociation, ThemeCluster).join(
        ThemeCluster, ThemeCluster.id == SocialThemeAssociation.theme_cluster_id
    )
    if state:
        query = query.where(SocialThemeAssociation.state == state)
    if market:
        query = query.where(SocialThemeAssociation.market == market)
    rows = db.execute(query.order_by(SocialThemeAssociation.id)).all()
    return [{"association_id": row.id, "theme_id": theme.id,
             "theme_name": theme.display_name, "market": row.market,
             "canonical_symbol": row.canonical_symbol, "state": row.state,
             "origin": row.origin, "decision_owner": row.decision_owner,
             "version": row.version, "evidence_work_ids": row.evidence_work_ids}
            for row, theme in rows]


@router.post("/admin/associations/{association_id}/decision",
             dependencies=[Depends(require_admin)])
def decide_admin_association(
    association_id: int, body: SocialAssociationDecisionRequest,
    db: Session = Depends(get_db),
):
    from app.services.social_theme_projection_service import SocialThemeProjectionService
    try:
        with db.begin():
            SocialThemeProjectionService(db, admin_authorized=True).decide(
                association_id, body.target, body.reason, ADMIN_ACTOR,
                body.expected_version,
            )
        return {"association_id": association_id, "status": body.target}
    except ValueError as exc:
        raise _admin_error(exc) from exc


@router.get("/admin/company-identities", dependencies=[Depends(require_admin)])
def admin_company_identities(db: Session = Depends(get_db)):
    from app.services.social_company_identity_service import SocialCompanyIdentityService
    return asdict(SocialCompanyIdentityService(db).read())


@router.patch("/admin/company-identities", dependencies=[Depends(require_admin)])
def update_admin_company_identities(
    body: SocialCompanyIdentityUpdate, db: Session = Depends(get_db)
):
    from app.services.social_company_identity_service import SocialCompanyIdentityService
    try:
        return asdict(SocialCompanyIdentityService(
            db, admin_authorized=True
        ).replace(body.entries, expected_version=body.expected_version, actor=ADMIN_ACTOR))
    except ValueError as exc:
        raise _admin_error(exc) from exc


@router.post("/admin/refresh", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_admin)])
def admin_refresh(response: Response, db: Session = Depends(get_db)):
    from app.services.task_registry_service import TaskCooldownError, TaskRegistryService
    try:
        return TaskRegistryService().trigger_task("social-signal-refresh", db)
    except TaskCooldownError as exc:
        response.headers["Retry-After"] = str(exc.retry_after)
        raise HTTPException(status_code=429, detail={"code": "manual_refresh_cooldown"},
                            headers={"Retry-After": str(exc.retry_after)}) from exc


__all__ = ["router"]
