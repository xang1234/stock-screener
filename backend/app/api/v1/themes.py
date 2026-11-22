"""Aggregator router for themes feature surfaces.

This module also re-exports legacy callables used by existing tests and
internal imports while the router decomposition is rolling out.
"""

from __future__ import annotations

from fastapi import APIRouter

from .themes_content_pipeline import router as content_pipeline_router
from .themes_content_pipeline import get_pipeline_observability
from .themes_content_sources import router as content_sources_router
from .themes_common import (
    detect_source_type_from_url,
    safe_theme_cluster_response as _safe_theme_cluster_response,
)
from .themes_intelligence import router as intelligence_router
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


# Keep static/source/pipeline paths before parameterized theme routes.
router.include_router(content_sources_router)
router.include_router(content_pipeline_router)
router.include_router(review_merge_router)
router.include_router(taxonomy_router)
router.include_router(intelligence_router)
router.include_router(queries_router)
