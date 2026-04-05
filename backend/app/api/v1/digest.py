"""Daily digest API endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from ...database import get_db
from ...schemas.digest import DailyDigestResponse
from ...services.digest_service import DigestService

router = APIRouter()


def _get_digest_service() -> DigestService:
    return DigestService()


@router.get("/daily", response_model=DailyDigestResponse)
async def get_daily_digest(
    as_of_date: date | None = Query(None),
    db: Session = Depends(get_db),
    service: DigestService = Depends(_get_digest_service),
):
    """Return the deterministic daily digest payload."""

    return service.get_daily_digest(db, as_of_date=as_of_date)


@router.get("/daily/markdown", response_class=PlainTextResponse)
async def get_daily_digest_markdown(
    as_of_date: date | None = Query(None),
    db: Session = Depends(get_db),
    service: DigestService = Depends(_get_digest_service),
):
    """Render the daily digest as markdown using the same normalized payload."""

    payload = service.get_daily_digest(db, as_of_date=as_of_date)
    return PlainTextResponse(service.render_markdown(payload))
