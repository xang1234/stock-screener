"""Aggregator router for themes feature surfaces.

This module also re-exports legacy callables used by existing tests and
internal imports while the router decomposition is rolling out.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ...database import is_corruption_error, safe_rollback
from ...models.theme import ContentItem, ContentItemPipelineState, ContentSource, ThemeMention
from ...services.theme_content_recovery_service import (
    attempt_reindex_theme_content_storage as _attempt_reindex_theme_content_storage,
    drop_theme_content_tables as _drop_theme_content_tables,
    force_forget_theme_content_tables as _force_forget_theme_content_tables,
    recreate_theme_content_tables as _recreate_theme_content_tables,
    reset_corrupt_theme_content_storage as _reset_corrupt_theme_content_storage,
    rewind_theme_content_source_cursors as _rewind_theme_content_source_cursors,
)
from ...theme_platform.content_browser_queries import (
    _build_content_items_browser_base_query as _platform_build_content_items_browser_base_query,
    _fetch_content_items_with_themes as _platform_fetch_content_items_with_themes,
)
from .themes_content_pipeline import router as content_pipeline_router
from .themes_content_pipeline import get_pipeline_observability
from .themes_content_sources import router as content_sources_router
from .themes_content_sources import XUISessionBridgeService
from .themes_common import (
    detect_source_type_from_url,
    resolve_source_ids_for_pipeline as _resolve_source_ids_for_pipeline,
    safe_theme_cluster_response as _safe_theme_cluster_response,
)
from ...wiring.bootstrap import get_session_factory
from .themes_queries import router as queries_router
from .themes_queries import (
    get_lifecycle_transitions,
    get_matching_telemetry,
    get_theme_rankings,
)
from .themes_review_merge import router as review_merge_router
from .themes_review_merge import (
    get_candidate_theme_queue,
    get_relationship_graph,
    review_candidate_themes,
)
from .themes_taxonomy import router as taxonomy_router

router = APIRouter()
logger = logging.getLogger(__name__)


_build_content_items_browser_base_query = _platform_build_content_items_browser_base_query
_fetch_content_items_with_themes = _platform_fetch_content_items_with_themes


def _corruption_targets_theme_content_storage(
    *,
    source_type=None,
    date_from=None,
    date_to=None,
    pipeline=None,
    **_ignored,
):
    """Compatibility wrapper that preserves monkeypatchable classifier probes."""
    with get_session_factory()() as probe_db:
        try:
            probe_db.query(ContentSource.id).filter(
                ContentSource.is_active == True
            ).limit(1).all()

            pipeline_source_ids = None
            if pipeline:
                pipeline_source_ids = _resolve_source_ids_for_pipeline(probe_db, pipeline)
        except Exception as exc:
            if is_corruption_error(exc):
                logger.warning(
                    "Theme content storage probe hit database corruption while reading content_sources; "
                    "skipping destructive reset: %s",
                    exc,
                )
                return False
            raise

        try:
            base_query = _build_content_items_browser_base_query(
                probe_db,
                source_type=source_type,
                date_from=date_from,
                date_to=date_to,
                pipeline=pipeline,
                pipeline_source_ids=pipeline_source_ids,
            )
            if base_query is None:
                return False

            base_query.count()
            ordered_query = base_query.order_by(ContentItem.published_at.desc())
            content_ids = [
                row.id
                for row in ordered_query.with_entities(ContentItem.id).limit(1).all()
            ]

            mentions_query = probe_db.query(
                ThemeMention.content_item_id,
                ThemeMention.theme_cluster_id,
                ThemeMention.sentiment,
            )
            if content_ids:
                mentions_query = mentions_query.filter(
                    ThemeMention.content_item_id.in_(content_ids)
                )
            if pipeline:
                mentions_query = mentions_query.filter(ThemeMention.pipeline == pipeline)
            mentions_query.limit(1).all()

            pipeline_probe = probe_db.query(
                ContentItemPipelineState.content_item_id,
                ContentItemPipelineState.status,
            )
            if content_ids:
                pipeline_probe = pipeline_probe.filter(
                    ContentItemPipelineState.content_item_id.in_(content_ids)
                )
            pipeline_probe.filter(
                ContentItemPipelineState.pipeline == (pipeline or "technical")
            ).limit(1).all()
        except Exception as exc:
            if is_corruption_error(exc):
                logger.warning(
                    "Theme content storage probe detected database corruption on query path: %s",
                    exc,
                )
                return True
            raise
    return False


def _fetch_content_items_with_themes_with_recovery(db, **kwargs):
    """Compatibility wrapper for legacy tests/patches during router decomposition."""
    try:
        return _fetch_content_items_with_themes(db, **kwargs)
    except Exception as exc:
        if not is_corruption_error(exc):
            raise
        safe_rollback(db)
        _attempt_reindex_theme_content_storage(exc)
        try:
            with get_session_factory()() as retry_db:
                return _fetch_content_items_with_themes(retry_db, **kwargs)
        except Exception as retry_exc:
            if not is_corruption_error(retry_exc):
                raise
            logger.warning(
                "Theme content browser query still hits corruption after REINDEX; "
                "checking whether rebuildable theme-content storage can be reset: %s",
                retry_exc,
            )
            if not _corruption_targets_theme_content_storage(**kwargs):
                logger.error(
                    "Theme content browser detected database corruption outside rebuildable "
                    "theme-content storage. Run backend/scripts/check_db_integrity.py --repair "
                    "or restore the latest valid backup. Initial error: %s. Retry error: %s",
                    exc,
                    retry_exc,
                )
                raise
            _reset_corrupt_theme_content_storage(retry_exc)
            with get_session_factory()() as reset_retry_db:
                return _fetch_content_items_with_themes(reset_retry_db, **kwargs)

# Keep static/source/pipeline paths before parameterized theme routes.
router.include_router(content_sources_router)
router.include_router(content_pipeline_router)
router.include_router(review_merge_router)
router.include_router(taxonomy_router)
router.include_router(queries_router)
