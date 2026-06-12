"""Scheduled static-site data export for the server deployment.

Rebuilds the per-market JSON bundles (the same format the GitHub Pages
workflow publishes) into a shared volume that nginx serves at
``/static-data/``. Read-mostly pages can then load pre-aggregated payloads
at static-file speed while the live API remains the source of truth for
interactive features.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..celery_app import celery_app
from ..config import settings

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.static_export_tasks.export_static_site_data",
    queue="celery",
    soft_time_limit=3600,
)
def export_static_site_data(
    self,
    output_dir: str | None = None,
    markets: list[str] | None = None,
):
    """Export the static-site bundles to the shared volume.

    Builds into ``<target>.tmp`` and swaps directories at the end so nginx
    never serves a half-written export.
    """
    from ..database import SessionLocal
    from ..services.static_site_export_service import (
        NoPublishedStaticMarketArtifact,
        StaticSiteExportService,
    )

    target = Path(output_dir or settings.static_export_output_dir)
    build_dir = target.with_name(target.name + ".tmp")
    stale_dir = target.with_name(target.name + ".old")

    # Published feature runs are keyed by uppercase market codes; lowercase
    # input would silently match nothing and abort the export.
    normalized_markets = (
        tuple(code for code in (str(m).strip().upper() for m in markets) if code)
        if markets
        else None
    )

    service = StaticSiteExportService(SessionLocal)
    try:
        result = service.export(
            build_dir,
            clean=True,
            markets=normalized_markets,
        )
    except NoPublishedStaticMarketArtifact as exc:
        logger.warning("Static export skipped: %s", exc)
        return {"status": "skipped", "reason": str(exc)}

    if stale_dir.exists():
        shutil.rmtree(stale_dir)
    if target.exists():
        target.rename(stale_dir)
    try:
        build_dir.rename(target)
    except Exception:
        # Promote failed: restore the previous export so nginx keeps serving
        # the last good bundle instead of 404ing until the next run.
        if stale_dir.exists() and not target.exists():
            stale_dir.rename(target)
        raise
    if stale_dir.exists():
        shutil.rmtree(stale_dir)

    logger.info(
        "Static export completed: %s markets as of %s -> %s",
        len(result.manifest.get("markets", {})),
        result.as_of_date,
        target,
    )
    return {
        "status": "completed",
        "output_dir": str(target),
        "as_of_date": result.as_of_date,
        "markets": sorted((result.manifest.get("markets") or {}).keys()),
        "warnings": list(result.warnings),
    }
