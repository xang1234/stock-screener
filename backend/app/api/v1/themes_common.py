"""Shared helpers for themes API routers."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

from pydantic import ValidationError
from sqlalchemy.orm import Session

from ...models.theme import ContentSource, ThemeCluster
from ...schemas.theme import ThemeClusterResponse
from ...services.theme_identity_normalization import (
    UNKNOWN_THEME_KEY,
    canonical_theme_key,
    display_theme_name,
)
from ...services.theme_pipeline_state_service import normalize_pipelines

logger = logging.getLogger(__name__)

_VALID_THEME_PIPELINES = {"technical", "fundamental"}


def detect_source_type_from_url(url: str, provided_type: str | None) -> str:
    """Auto-detect source type from URL when type is missing or likely wrong."""
    if not url:
        return provided_type or "news"

    raw_url = url.strip()
    url_lower = raw_url.lower()

    if url_lower.startswith("@"):
        return "twitter"
    if url_lower.startswith("r/"):
        return "reddit"

    # Parse hostnames to avoid broad substring matches like "examplex.com".
    parsed = urlparse(raw_url if "://" in raw_url else f"https://{raw_url}")
    hostname = (parsed.hostname or "").lower().strip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]

    if hostname in {"twitter.com", "x.com"} or hostname.endswith(".twitter.com") or hostname.endswith(".x.com"):
        return "twitter"
    if hostname == "reddit.com" or hostname.endswith(".reddit.com"):
        return "reddit"
    if hostname == "substack.com" or hostname.endswith(".substack.com"):
        return "substack"

    return provided_type or "news"


def normalize_aliases(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(alias).strip() for alias in value if str(alias).strip()]
    alias = str(value).strip()
    return [alias] if alias else None


def safe_theme_cluster_response(cluster: ThemeCluster) -> ThemeClusterResponse:
    """Return schema-valid cluster responses, normalizing legacy invalid identity rows."""
    try:
        return ThemeClusterResponse.model_validate(cluster)
    except ValidationError:
        raw_pipeline = str(getattr(cluster, "pipeline", "") or "").strip().lower()
        pipeline = raw_pipeline if raw_pipeline in _VALID_THEME_PIPELINES else "technical"

        raw_name = str(getattr(cluster, "name", "") or "").strip()
        raw_display = str(getattr(cluster, "display_name", "") or "").strip()
        seed_label = raw_display or raw_name or str(getattr(cluster, "canonical_key", "") or "").strip()
        display_name = display_theme_name(seed_label)
        raw_canonical_key = str(getattr(cluster, "canonical_key", "") or "").strip()
        canonical_key = canonical_theme_key(raw_canonical_key or display_name)
        if canonical_key == UNKNOWN_THEME_KEY:
            canonical_key = canonical_theme_key(display_name)
        name = raw_name or display_name

        logger.warning(
            "Theme cluster id=%s has invalid identity fields; serving normalized response values",
            getattr(cluster, "id", None),
        )

        return ThemeClusterResponse.model_validate(
            {
                "id": cluster.id,
                "name": name,
                "canonical_key": canonical_key,
                "display_name": display_name,
                "aliases": normalize_aliases(getattr(cluster, "aliases", None)),
                "description": getattr(cluster, "description", None),
                "pipeline": pipeline,
                "category": getattr(cluster, "category", None),
                "is_emerging": bool(getattr(cluster, "is_emerging", False)),
                "is_validated": bool(getattr(cluster, "is_validated", False)),
                "lifecycle_state": str(getattr(cluster, "lifecycle_state", "") or "candidate"),
                "lifecycle_state_updated_at": getattr(cluster, "lifecycle_state_updated_at", None),
                "candidate_since_at": getattr(cluster, "candidate_since_at", None),
                "activated_at": getattr(cluster, "activated_at", None),
                "dormant_at": getattr(cluster, "dormant_at", None),
                "reactivated_at": getattr(cluster, "reactivated_at", None),
                "retired_at": getattr(cluster, "retired_at", None),
                "discovery_source": getattr(cluster, "discovery_source", None),
                "first_seen_at": getattr(cluster, "first_seen_at", None),
                "last_seen_at": getattr(cluster, "last_seen_at", None),
            }
        )


def parse_csv_values(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def resolve_source_ids_for_pipeline(db: Session, pipeline: str) -> list[int]:
    """Resolve active source IDs assigned to a pipeline."""
    source_ids: list[int] = []
    sources = db.query(ContentSource.id, ContentSource.pipelines).filter(
        ContentSource.is_active == True
    ).all()
    for source_id, source_pipelines in sources:
        if pipeline in normalize_pipelines(source_pipelines):
            source_ids.append(source_id)
    return source_ids
