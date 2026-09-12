"""Reviewed reversible theme grouping and attributed development timelines."""

import os
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import String, cast, exists, func, or_
from sqlalchemy.orm import Session

from app.api.v1.config import require_admin
from app.database import get_db
from app.models.theme import ContentItem, ThemeCluster, ThemeMention
from app.models.theme_intelligence import (
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
    ThemeDevelopmentWork,
)
from app.services.theme_equivalence_service import (
    EquivalenceConflict,
    ThemeEquivalenceService,
)
from app.services.theme_evidence_eligibility_service import legacy_eligibility_exists

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


class GroupRequest(BaseModel):
    source_id: int
    target_id: int
    reason: str = Field(min_length=1, max_length=2000)
    operation_key: str = Field(min_length=1, max_length=120)
    expected_version: str = Field(min_length=64, max_length=64)


class UndoRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class BackfillRequest(BaseModel):
    item_ids: list[int] = Field(min_length=1, max_length=100)
    apply: bool = False


@router.get("/equivalence/preview")
def preview_equivalence(source_id: int, target_id: int, db: DbSession):
    try:
        return ThemeEquivalenceService(db).preview(source_id, target_id)
    except EquivalenceConflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/equivalence", dependencies=[Depends(require_admin)])
def apply_equivalence(
    request: GroupRequest,
    db: DbSession,
    x_admin_actor: str = Header(default="admin", alias="X-Admin-Actor"),
):
    service = ThemeEquivalenceService(db)
    try:
        result = service.apply(
            request.source_id,
            request.target_id,
            actor=x_admin_actor,
            reason=request.reason,
            key=request.operation_key,
            expected_version=request.expected_version,
        )
        db.commit()
    except EquivalenceConflict as exc:
        db.rollback()
        raise HTTPException(409, str(exc)) from exc
    return {
        **result,
        "version": service.version(result["pipeline"]),
        "refresh_status": "pending",
    }


@router.get("/equivalence/history")
def equivalence_history(
    db: DbSession,
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$"),
):
    service = ThemeEquivalenceService(db)
    return {
        "version": service.version(pipeline),
        "operations": [
            {
                "id": row.id,
                "source_id": row.source_id,
                "target_id": row.target_id,
                "member_ids": row.member_ids,
                "aliases": row.aliases,
                "active": row.active,
                "actor": row.actor,
                "reason": row.reason,
                "refresh_pending": row.refresh_pending,
                "created_at": row.created_at,
                "undone_at": row.undone_at,
                "undone_by": row.undone_by,
                "undo_reason": row.undo_reason,
            }
            for row in reversed(service.operations(pipeline))
        ],
    }


@router.post(
    "/equivalence/{operation_id}/undo", dependencies=[Depends(require_admin)]
)
def undo_equivalence(
    operation_id: int,
    request: UndoRequest,
    db: DbSession,
    x_admin_actor: str = Header(default="admin", alias="X-Admin-Actor"),
):
    service = ThemeEquivalenceService(db)
    try:
        result = service.undo(operation_id, actor=x_admin_actor, reason=request.reason)
        db.commit()
    except EquivalenceConflict as exc:
        db.rollback()
        raise HTTPException(409, str(exc)) from exc
    return {
        **result,
        "version": service.version(result["pipeline"]),
        "refresh_status": "pending",
    }


@router.get("/equivalence/search")
def search_equivalent_themes(
    db: DbSession,
    q: str = "",
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$"),
):
    service = ThemeEquivalenceService(db)
    snapshot = service.snapshot(pipeline)
    mapping = snapshot.mapping
    query = db.query(ThemeCluster).filter_by(
        pipeline=pipeline, is_active=True, is_l1=False
    )
    normalized_q = q.strip()
    if normalized_q:
        query = query.filter(
            or_(
                ThemeCluster.name.icontains(normalized_q, autoescape=True),
                ThemeCluster.display_name.icontains(normalized_q, autoescape=True),
                cast(ThemeCluster.aliases, String).icontains(
                    normalized_q, autoescape=True
                ),
            )
        )
    matching = query.order_by(ThemeCluster.name).limit(100).all()
    root_ids = {mapping.get(row.id, row.id) for row in matching}
    roots = {
        row.id: row
        for row in db.query(ThemeCluster).filter(ThemeCluster.id.in_(root_ids)).all()
    }
    results = {}
    for row in matching:
        root = mapping.get(row.id, row.id)
        target = roots[root]
        results[root] = {
            "id": root,
            "name": target.display_name or target.name,
            "member_ids": snapshot.members(root),
        }
        if len(results) >= 100:
            break
    return {"themes": list(results.values()), "version": snapshot.version}


@router.post("/developments/backfill")
def backfill_developments(request: BackfillRequest, db: DbSession):
    from app.services.theme_development_worker import discover

    items = db.query(ContentItem.id).filter(ContentItem.id.in_(request.item_ids)).all()
    ids = [row.id for row in items]
    if not request.apply:
        return {"item_ids": ids, "max_items": 100, "model_calls_enabled": False}
    if os.environ.get("THEME_DEVELOPMENT_TRACKING_ENABLED", "false").lower() not in {
        "1",
        "true",
        "yes",
    }:
        raise HTTPException(
            409, "Enable development tracking before queuing model work"
        )
    queued = discover(db, limit=100, item_ids=ids)
    db.commit()
    return {"queued": queued, "item_ids": ids}


@router.get("/{theme_id}/developments")
def theme_developments(
    theme_id: int, db: DbSession, limit: int = Query(50, ge=1, le=200)
):
    theme = db.get(ThemeCluster, theme_id)
    if theme is None:
        raise HTTPException(404, "Theme not found")
    service = ThemeEquivalenceService(db)
    snapshot = service.snapshot(theme.pipeline)
    members = snapshot.members(theme_id)
    membership = exists().where(
        ThemeDevelopmentTheme.observation_id == ThemeDevelopmentObservation.id,
        ThemeDevelopmentTheme.theme_id.in_(members),
    )
    query = db.query(ThemeDevelopmentObservation).filter(
        ThemeDevelopmentObservation.pipeline == theme.pipeline,
        membership,
        legacy_eligibility_exists(
            ThemeDevelopmentObservation.content_item_id,
            theme.pipeline,
            active_only=True,
        ),
    )
    current = query.filter(ThemeDevelopmentObservation.superseded.is_(False))
    event_count = current.with_entities(
        func.count(func.distinct(ThemeDevelopmentObservation.event_id))
    ).scalar()
    update_count = current.filter(
        ThemeDevelopmentObservation.classification == "material_update"
    ).count()
    matching = (
        query.add_entity(ContentItem)
        .join(
            ContentItem,
            ContentItem.id == ThemeDevelopmentObservation.content_item_id,
        )
        .order_by(
            ThemeDevelopmentObservation.available_at.desc(),
            ThemeDevelopmentObservation.id.desc(),
        )
        .limit(limit)
        .all()
    )
    work_items = db.query(ThemeMention.content_item_id).filter(
        ThemeMention.theme_cluster_id.in_(members),
        ThemeMention.pipeline == theme.pipeline,
        ThemeMention.social_work_id.is_(None),
        legacy_eligibility_exists(
            ThemeMention.content_item_id, theme.pipeline, active_only=True
        ),
    )
    work_counts = dict(
        db.query(ThemeDevelopmentWork.status, func.count(ThemeDevelopmentWork.id))
        .filter(
            ThemeDevelopmentWork.pipeline == theme.pipeline,
            ThemeDevelopmentWork.content_item_id.in_(work_items),
        )
        .group_by(ThemeDevelopmentWork.status)
        .all()
    )
    return {
        "theme_id": snapshot.representative(theme_id),
        "grouping_version": snapshot.version,
        "tracking_enabled": os.environ.get(
            "THEME_DEVELOPMENT_TRACKING_ENABLED", "false"
        ).lower()
        in {"1", "true", "yes"},
        "work_counts": work_counts,
        "event_count": event_count,
        "material_update_count": update_count,
        "observations": [
            {
                "id": row.id,
                "event_id": row.event_id,
                "content_item_id": item.id,
                "url": item.url,
                "source_name": item.source_name,
                "facts": row.facts,
                "citations": row.citations,
                "classification": row.classification,
                "superseded": row.superseded,
                "published_at": row.published_at,
                "available_at": row.available_at,
            }
            for row, item in matching
        ],
    }
