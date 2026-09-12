"""Durable invalidation after reviewed membership changes."""

import logging

from app.models.theme_intelligence import ThemeEquivalenceOperation
from app.services.theme_equivalence_service import ThemeEquivalenceService

logger = logging.getLogger(__name__)


def refresh_groups(db, pipeline):
    from .theme_group_coordination import publication_scope

    with publication_scope(db):
        return _refresh_groups(db, pipeline)


def _refresh_groups(db, pipeline):
    from app.services.theme_discovery_service import ThemeDiscoveryService
    from app.services.theme_taxonomy_service import ThemeTaxonomyService
    from app.services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    version = ThemeEquivalenceService(db).version(pipeline)
    try:
        result = ThemeDiscoveryService(db, pipeline=pipeline).update_all_theme_metrics()
        if result.get("errors"):
            return "pending"
        ThemeTaxonomyService(db, pipeline=pipeline).compute_all_l1_metrics()
        db.commit()
        published = safe_publish_themes_bootstrap_variants(pipeline=pipeline)
        if not published or any(value is None for value in published.values()):
            return "pending"
        grouping = ThemeEquivalenceService(db)
        grouping._lock()
        if grouping.version(pipeline) != version:
            db.query(ThemeEquivalenceOperation).filter_by(pipeline=pipeline).update(
                {"refresh_pending": True}, synchronize_session=False
            )
            db.commit()
            return "pending"
        db.query(ThemeEquivalenceOperation).filter_by(
            pipeline=pipeline, refresh_pending=True
        ).update({"refresh_pending": False}, synchronize_session=False)
        db.commit()
        return "complete"
    except Exception:
        db.rollback()
        logger.exception("Theme group refresh deferred", extra={"pipeline": pipeline})
        return "pending"
