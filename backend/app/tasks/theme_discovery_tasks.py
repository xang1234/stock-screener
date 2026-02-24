"""
Celery tasks for Theme Discovery pipeline.

Provides background tasks for:
- Content ingestion from sources (RSS, Twitter, News)
- LLM extraction of themes and tickers
- Theme metrics calculation
- Correlation validation
- Alert generation

Recommended scheduling:
- Content ingestion: Every 2-4 hours during market hours
- Theme extraction: Every 4 hours or after ingestion
- Metrics calculation: Daily after market close
- Correlation validation: Weekly
"""
import os

# Disable MPS/Metal before any PyTorch imports to avoid fork() issues on macOS
# Must be set before sentence-transformers or torch is imported anywhere
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
from datetime import datetime
import time
from uuid import uuid4

from ..celery_app import celery_app
from ..database import SessionLocal
from ..services.redis_pool import get_redis_client

logger = logging.getLogger(__name__)

_THEME_STALE_EMBED_LOCK_KEY = "theme:stale_embedding_recompute:lock"
_THEME_STALE_EMBED_LOCK_TTL_SECONDS = 3600
_LOCK_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


def _acquire_theme_stale_embedding_lock(task_id: str) -> tuple[object | None, str | None]:
    client = get_redis_client()
    if client is None:
        return None, None
    token = f"{task_id}:{uuid4().hex}"
    acquired = client.set(
        _THEME_STALE_EMBED_LOCK_KEY,
        token,
        nx=True,
        ex=_THEME_STALE_EMBED_LOCK_TTL_SECONDS,
    )
    if acquired:
        return client, token
    return client, None


def _release_theme_stale_embedding_lock(client, token: str) -> None:
    if client is None or not token:
        return
    try:
        client.eval(_LOCK_RELEASE_LUA, 1, _THEME_STALE_EMBED_LOCK_KEY, token)
    except Exception as exc:
        logger.warning("Failed to release stale embedding recompute lock: %s", exc)


@celery_app.task(name='app.tasks.theme_discovery_tasks.ingest_content')
def ingest_content():
    """
    Fetch new content from all active sources.

    This pulls content from RSS feeds, Twitter, news APIs, and Reddit.
    Should be run every 2-4 hours during market hours.

    Returns:
        Dict with ingestion statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: Content Ingestion for Theme Discovery")
    logger.info("=" * 60)

    from ..services.content_ingestion_service import ContentIngestionService

    db = SessionLocal()
    start_time = time.time()

    try:
        service = ContentIngestionService(db)
        result = service.fetch_all_active_sources()

        duration = time.time() - start_time

        logger.info(f"Content ingestion complete in {duration:.2f}s")
        logger.info(f"  Sources fetched: {result['total_sources']}")
        logger.info(f"  New items: {result['total_new_items']}")
        logger.info(f"  Errors: {len(result['errors'])}")

        if result['errors']:
            for error in result['errors']:
                logger.warning(f"  - {error['name']}: {error['error']}")

        logger.info("=" * 60)

        return {
            'total_sources': result['total_sources'],
            'new_items': result['total_new_items'],
            'errors': len(result['errors']),
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in content ingestion task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.extract_themes')
def extract_themes(limit: int = 50, pipeline: str = None):
    """
    Extract themes from unprocessed content using LLM.

    Processes content items that haven't been analyzed yet,
    extracting themes, tickers, and sentiment.

    Args:
        limit: Maximum number of content items to process
        pipeline: Pipeline to run extraction for (technical/fundamental).
                  If None, runs for both pipelines sequentially.

    Returns:
        Dict with extraction statistics
    """
    pipelines = [pipeline] if pipeline else ["technical", "fundamental"]

    logger.info("=" * 60)
    logger.info("TASK: Theme Extraction via LLM")
    logger.info(f"Processing up to {limit} items for pipelines: {pipelines}")
    logger.info("=" * 60)

    from ..services.theme_extraction_service import ThemeExtractionService

    db = SessionLocal()
    start_time = time.time()

    combined_result = {
        'processed': 0,
        'total_mentions': 0,
        'errors': 0,
        'new_themes': [],
        'pipelines': pipelines,
    }

    try:
        for p in pipelines:
            logger.info(f"Processing pipeline: {p}")
            service = ThemeExtractionService(db, pipeline=p)
            result = service.process_batch(limit=limit)

            # Aggregate results
            combined_result['processed'] += result.get('processed', 0)
            combined_result['total_mentions'] += result.get('total_mentions', 0)
            combined_result['errors'] += result.get('errors', 0)
            combined_result['new_themes'].extend(result.get('new_themes', []))

        result = combined_result

        duration = time.time() - start_time

        logger.info(f"Theme extraction complete in {duration:.2f}s")
        logger.info(f"  Items processed: {result['processed']}")
        logger.info(f"  Theme mentions extracted: {result['total_mentions']}")
        logger.info(f"  Errors: {result['errors']}")

        if result['new_themes']:
            logger.info(f"  New themes discovered: {', '.join(result['new_themes'])}")

        logger.info("=" * 60)

        return {
            'processed': result['processed'],
            'total_mentions': result['total_mentions'],
            'errors': result['errors'],
            'new_themes': result['new_themes'],
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in theme extraction task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.reprocess_failed_themes')
def reprocess_failed_themes(limit: int = 500, pipeline: str = None):
    """
    Reprocess content items that previously failed LLM extraction.

    Finds items with extraction_error set, resets them, and retries
    extraction. Should be run before new extraction to use fresh rate limits.

    Args:
        limit: Maximum number of failed items to reprocess per pipeline
        pipeline: Pipeline to reprocess for (technical/fundamental).
                  If None, runs for both pipelines sequentially.

    Returns:
        Dict with reprocessing statistics
    """
    pipelines = [pipeline] if pipeline else ["technical", "fundamental"]

    logger.info("=" * 60)
    logger.info("TASK: Reprocess Failed Theme Extractions")
    logger.info(f"Processing up to {limit} failed items for pipelines: {pipelines}")
    logger.info("=" * 60)

    from ..services.theme_extraction_service import ThemeExtractionService

    db = SessionLocal()
    start_time = time.time()

    combined_result = {
        'reprocessed_count': 0,
        'processed': 0,
        'total_mentions': 0,
        'errors': 0,
        'pipelines': pipelines,
    }

    try:
        for p in pipelines:
            logger.info(f"Reprocessing failed items for pipeline: {p}")
            service = ThemeExtractionService(db, pipeline=p)
            result = service.reprocess_failed_items(limit=limit)

            combined_result['reprocessed_count'] += result.get('reprocessed_count', 0)
            combined_result['processed'] += result.get('processed', 0)
            combined_result['total_mentions'] += result.get('total_mentions', 0)
            combined_result['errors'] += result.get('errors', 0)

        duration = time.time() - start_time

        logger.info(f"Reprocessing complete in {duration:.2f}s")
        logger.info(f"  Items reset for retry: {combined_result['reprocessed_count']}")
        logger.info(f"  Successfully processed: {combined_result['processed']}")
        logger.info(f"  Theme mentions recovered: {combined_result['total_mentions']}")
        logger.info(f"  Errors: {combined_result['errors']}")
        logger.info("=" * 60)

        return {
            'reprocessed_count': combined_result['reprocessed_count'],
            'processed': combined_result['processed'],
            'total_mentions': combined_result['total_mentions'],
            'errors': combined_result['errors'],
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in reprocess failed themes task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.calculate_theme_metrics')
def calculate_theme_metrics(pipeline: str = None):
    """
    Calculate and update metrics for all active themes.

    This updates:
    - Mention velocity (social signals)
    - Theme basket performance (price metrics)
    - Internal correlation
    - Screener integration (Minervini, RS)
    - Composite momentum score
    - Rankings

    Args:
        pipeline: Pipeline to calculate metrics for (technical/fundamental).
                  If None, runs for both pipelines sequentially.

    Should be run daily after market close.

    Returns:
        Dict with calculation results
    """
    pipelines = [pipeline] if pipeline else ["technical", "fundamental"]

    logger.info("=" * 60)
    logger.info("TASK: Calculate Theme Metrics")
    logger.info(f"Pipelines: {pipelines}")
    logger.info("=" * 60)

    from ..services.theme_discovery_service import ThemeDiscoveryService

    db = SessionLocal()
    start_time = time.time()

    combined_result = {
        'themes_updated': 0,
        'errors': 0,
        'rankings': [],
        'pipelines': pipelines,
    }

    try:
        for p in pipelines:
            logger.info(f"Calculating metrics for pipeline: {p}")
            service = ThemeDiscoveryService(db, pipeline=p)
            result = service.update_all_theme_metrics()

            # Aggregate results
            combined_result['themes_updated'] += result.get('themes_updated', 0)
            combined_result['errors'] += result.get('errors', 0)
            # Keep track of rankings per pipeline
            for r in result.get('rankings', []):
                r['pipeline'] = p
            combined_result['rankings'].extend(result.get('rankings', []))

        result = combined_result

        duration = time.time() - start_time

        logger.info(f"Theme metrics calculation complete in {duration:.2f}s")
        logger.info(f"  Themes updated: {result['themes_updated']}")
        logger.info(f"  Errors: {result['errors']}")

        if result.get('rankings'):
            logger.info("Top 5 themes:")
            for r in result['rankings'][:5]:
                logger.info(
                    f"  #{r['rank']}: {r['theme']} "
                    f"(score: {r['score']:.1f}, status: {r['status']})"
                )

        logger.info("=" * 60)

        return {
            'themes_updated': result['themes_updated'],
            'errors': result['errors'],
            'top_themes': result.get('rankings', [])[:10],
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in theme metrics task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    name='app.tasks.theme_discovery_tasks.recompute_stale_theme_embeddings',
)
def recompute_stale_theme_embeddings(
    self,
    pipeline: str = None,
    batch_size: int = 100,
    max_batches: int = 20,
):
    """
    Incrementally recompute stale theme embeddings in bounded batches.
    """
    logger.info("=" * 60)
    logger.info("TASK: Recompute Stale Theme Embeddings")
    logger.info("Pipeline: %s | batch_size=%s | max_batches=%s", pipeline or "all", batch_size, max_batches)
    logger.info("=" * 60)

    from ..services.theme_merging_service import ThemeMergingService

    task_id = getattr(getattr(self, "request", None), "id", None) or "unknown"
    lock_client, lock_token = _acquire_theme_stale_embedding_lock(task_id)
    if lock_client is None:
        logger.warning("Redis unavailable; cannot enforce strict stale-embedding task concurrency")
        return {
            "status": "skipped",
            "reason": "lock_unavailable",
            "message": "Redis lock unavailable; strict single-flight concurrency not satisfied",
            "timestamp": datetime.now().isoformat(),
        }
    if lock_token is None:
        holder = None
        try:
            holder = lock_client.get(_THEME_STALE_EMBED_LOCK_KEY)
        except Exception:
            holder = None
        holder_display = holder.decode() if isinstance(holder, (bytes, bytearray)) else str(holder or "")
        logger.info("Skipping stale embedding recompute because lock is already held (%s)", holder_display)
        return {
            "status": "skipped",
            "reason": "lock_held",
            "holder": holder_display,
            "timestamp": datetime.now().isoformat(),
        }

    db = SessionLocal()
    start_time = time.time()
    last_progress: dict[str, object] = {}

    try:
        service = ThemeMergingService(db)

        def _on_batch(progress: dict):
            nonlocal last_progress
            last_progress = dict(progress)
            total = int(progress.get("stale_total_before") or 0)
            remaining = int(progress.get("stale_remaining") or 0)
            done = max(0, total - remaining)
            percent = float(done / total * 100.0) if total > 0 else 100.0
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": done,
                    "total": total,
                    "percent": round(percent, 2),
                    "message": (
                        f"Processed {progress.get('processed', 0)} stale candidates "
                        f"across {progress.get('batches_processed', 0)} batches"
                    ),
                    "batch": progress.get("batch"),
                    "failed": progress.get("failed", 0),
                    "remaining": remaining,
                },
            )

        result = service.recompute_stale_embeddings(
            pipeline=pipeline,
            batch_size=batch_size,
            max_batches=max_batches,
            on_batch=_on_batch,
        )

        duration = time.time() - start_time
        logger.info("Stale embedding recompute complete in %.2fs", duration)
        logger.info(
            "  stale_before=%s processed=%s refreshed=%s failed=%s remaining=%s",
            result.get("stale_total_before", 0),
            result.get("processed", 0),
            result.get("refreshed", 0),
            result.get("failed", 0),
            result.get("stale_remaining_after", 0),
        )
        logger.info("=" * 60)
        return {
            "status": "completed",
            "summary": result,
            "progress": last_progress,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        _release_theme_stale_embedding_lock(lock_client, lock_token or "")


@celery_app.task(name='app.tasks.theme_discovery_tasks.promote_candidate_themes')
def promote_candidate_themes(pipeline: str = None, limit: int = 1000):
    """
    Promote candidate themes to active based on evidence thresholds.
    """
    pipelines = [pipeline] if pipeline else ["technical", "fundamental"]
    logger.info("=" * 60)
    logger.info("TASK: Promote Candidate Themes")
    logger.info(f"Pipelines: {pipelines}")
    logger.info("=" * 60)

    from ..services.theme_discovery_service import ThemeDiscoveryService

    db = SessionLocal()
    start_time = time.time()
    summary = {"pipelines": {}, "promoted_total": 0, "scanned_total": 0, "errors": 0}

    try:
        for p in pipelines:
            service = ThemeDiscoveryService(db, pipeline=p)
            result = service.promote_candidate_themes(limit=limit)
            summary["pipelines"][p] = result
            summary["promoted_total"] += result.get("promoted", 0)
            summary["scanned_total"] += result.get("scanned", 0)
            summary["errors"] += result.get("errors", 0)

        duration = time.time() - start_time
        logger.info("Candidate promotion complete in %.2fs", duration)
        logger.info("  Themes scanned: %s", summary["scanned_total"])
        logger.info("  Themes promoted: %s", summary["promoted_total"])
        logger.info("  Errors: %s", summary["errors"])
        logger.info("=" * 60)
        return {
            "status": "completed",
            "summary": summary,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        db.rollback()
        logger.error("Error in promote_candidate_themes task: %s", e, exc_info=True)
        return {"error": str(e), "timestamp": datetime.now().isoformat()}
    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.apply_lifecycle_policies')
def apply_lifecycle_policies(pipeline: str = None, limit: int = 1000):
    """
    Apply dormancy/reactivation lifecycle policies with explainable counters.
    """
    pipelines = [pipeline] if pipeline else ["technical", "fundamental"]
    logger.info("=" * 60)
    logger.info("TASK: Apply Lifecycle Dormancy/Reactivation Policies")
    logger.info(f"Pipelines: {pipelines}")
    logger.info("=" * 60)

    from ..services.theme_discovery_service import ThemeDiscoveryService

    db = SessionLocal()
    start_time = time.time()
    summary = {
        "pipelines": {},
        "to_dormant_total": 0,
        "to_reactivated_total": 0,
        "scanned_total": 0,
        "errors": 0,
    }

    try:
        for p in pipelines:
            service = ThemeDiscoveryService(db, pipeline=p)
            result = service.apply_dormancy_and_reactivation_policies(limit=limit)
            summary["pipelines"][p] = result
            summary["to_dormant_total"] += result.get("to_dormant", 0)
            summary["to_reactivated_total"] += result.get("to_reactivated", 0)
            summary["scanned_total"] += result.get("scanned", 0)
            summary["errors"] += result.get("errors", 0)

        duration = time.time() - start_time
        logger.info("Lifecycle policy pass complete in %.2fs", duration)
        logger.info("  Themes scanned: %s", summary["scanned_total"])
        logger.info("  Dormant transitions: %s", summary["to_dormant_total"])
        logger.info("  Reactivated transitions: %s", summary["to_reactivated_total"])
        logger.info("  Errors: %s", summary["errors"])
        logger.info("=" * 60)
        return {
            "status": "completed",
            "summary": summary,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        db.rollback()
        logger.error("Error in apply_lifecycle_policies task: %s", e, exc_info=True)
        return {"error": str(e), "timestamp": datetime.now().isoformat()}
    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.infer_theme_relationships')
def infer_theme_relationships(pipeline: str = None, max_merge_suggestions: int = 300):
    """
    Infer theme relationship edges from merge analysis and rule-based overlap checks.
    """
    pipelines = [pipeline] if pipeline else ["technical", "fundamental"]
    logger.info("=" * 60)
    logger.info("TASK: Infer Theme Relationships")
    logger.info(f"Pipelines: {pipelines}")
    logger.info("=" * 60)

    from ..services.theme_discovery_service import ThemeDiscoveryService

    db = SessionLocal()
    start_time = time.time()
    summary = {"pipelines": {}, "edges_written": 0, "errors": 0}

    try:
        for p in pipelines:
            service = ThemeDiscoveryService(db, pipeline=p)
            result = service.infer_theme_relationships(max_merge_suggestions=max_merge_suggestions)
            summary["pipelines"][p] = result
            summary["edges_written"] += (
                result.get("merge_edges_written", 0) + result.get("rule_edges_written", 0)
            )
            summary["errors"] += result.get("errors", 0)

        duration = time.time() - start_time
        logger.info("Theme relationship inference complete in %.2fs", duration)
        logger.info("  Edges written/updated: %s", summary["edges_written"])
        logger.info("  Errors: %s", summary["errors"])
        logger.info("=" * 60)
        return {
            "status": "completed",
            "summary": summary,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        db.rollback()
        logger.error("Error in infer_theme_relationships task: %s", e, exc_info=True)
        return {"error": str(e), "timestamp": datetime.now().isoformat()}
    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.validate_themes')
def validate_themes(min_correlation: float = 0.5):
    """
    Validate all themes by checking internal correlations.

    A valid theme has high average correlation between constituents.
    Should be run weekly.

    Args:
        min_correlation: Minimum correlation threshold for validation

    Returns:
        Dict with validation results
    """
    logger.info("=" * 60)
    logger.info("TASK: Validate Theme Correlations")
    logger.info(f"Minimum correlation threshold: {min_correlation}")
    logger.info("=" * 60)

    from ..services.theme_correlation_service import ThemeCorrelationService

    db = SessionLocal()
    start_time = time.time()

    try:
        service = ThemeCorrelationService(db)
        result = service.run_full_validation(min_correlation)

        duration = time.time() - start_time

        logger.info(f"Theme validation complete in {duration:.2f}s")
        logger.info(f"  Themes validated: {result['themes_validated']}")
        logger.info(f"  Valid: {result['themes_valid']}")
        logger.info(f"  Invalid: {result['themes_invalid']}")

        # Log invalid themes
        invalid_themes = [d for d in result['details'] if not d['is_valid']]
        if invalid_themes:
            logger.warning("Invalid themes (low correlation):")
            for t in invalid_themes:
                logger.warning(f"  - {t['theme']}: avg_corr={t['avg_correlation']:.3f}")

        logger.info("=" * 60)

        return {
            'themes_validated': result['themes_validated'],
            'themes_valid': result['themes_valid'],
            'themes_invalid': result['themes_invalid'],
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in theme validation task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.discover_correlation_clusters')
def discover_correlation_clusters(
    correlation_threshold: float = 0.6,
    min_cluster_size: int = 3
):
    """
    Discover hidden themes via correlation clustering.

    Finds groups of stocks that move together regardless of
    industry classification. These may represent emerging themes
    not yet identified in social/news sources.

    Args:
        correlation_threshold: Minimum correlation for clustering
        min_cluster_size: Minimum stocks per cluster

    Returns:
        Dict with discovered clusters
    """
    logger.info("=" * 60)
    logger.info("TASK: Discover Correlation Clusters")
    logger.info(f"Correlation threshold: {correlation_threshold}")
    logger.info(f"Minimum cluster size: {min_cluster_size}")
    logger.info("=" * 60)

    from ..services.theme_correlation_service import ThemeCorrelationService

    db = SessionLocal()
    start_time = time.time()

    try:
        service = ThemeCorrelationService(db)
        clusters = service.discover_correlation_clusters(
            correlation_threshold=correlation_threshold,
            min_cluster_size=min_cluster_size
        )

        duration = time.time() - start_time

        logger.info(f"Correlation discovery complete in {duration:.2f}s")
        logger.info(f"  Clusters found: {len(clusters)}")

        # Log cross-industry clusters (potential hidden themes)
        cross_industry = [c for c in clusters if c['is_cross_industry']]
        if cross_industry:
            logger.info(f"  Cross-industry clusters: {len(cross_industry)}")
            for c in cross_industry[:5]:
                logger.info(
                    f"    - {c['num_stocks']} stocks, "
                    f"corr={c['avg_correlation']:.3f}, "
                    f"industries={len(c['industries'])}"
                )

        logger.info("=" * 60)

        return {
            'clusters_found': len(clusters),
            'cross_industry_clusters': len(cross_industry),
            'clusters': clusters[:20],  # Top 20
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in correlation discovery task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.check_alerts')
def check_alerts():
    """
    Check for and generate theme alerts.

    Creates alerts for:
    - New themes discovered in last 24h
    - Velocity spikes (>3x normal)
    - Theme breakouts (RS > 70)

    Should be run after metrics calculation.

    Returns:
        Dict with alert summary
    """
    logger.info("=" * 60)
    logger.info("TASK: Check Theme Alerts")
    logger.info("=" * 60)

    from ..services.theme_discovery_service import ThemeDiscoveryService

    db = SessionLocal()
    start_time = time.time()

    try:
        service = ThemeDiscoveryService(db)
        alerts = service.check_for_alerts()

        duration = time.time() - start_time

        logger.info(f"Alert check complete in {duration:.2f}s")
        logger.info(f"  New alerts: {len(alerts)}")

        for alert in alerts:
            logger.info(f"  - [{alert.severity.upper()}] {alert.title}")

        logger.info("=" * 60)

        return {
            'new_alerts': len(alerts),
            'alerts': [
                {'type': a.alert_type, 'title': a.title, 'severity': a.severity}
                for a in alerts
            ],
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in alert check task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.theme_discovery_tasks.run_full_pipeline')
def run_full_pipeline(self, run_id: str = None, pipeline: str = None):
    """
    Run the complete theme discovery pipeline with progress tracking.

    Executes in order:
    1. Content ingestion
    2. Reprocess previously failed items (retry before consuming rate limits)
    3. Theme extraction (pipeline-specific)
    4. Metrics calculation (pipeline-specific)
    5. Alert checking

    This is a convenience task for running everything at once.
    Ideal for daily runs after market close.

    Args:
        self: Celery task instance (bind=True enables this)
        run_id: Optional pipeline run ID for database tracking
        pipeline: Pipeline to run (technical/fundamental).
                  If None, runs for both pipelines sequentially.

    Returns:
        Dict with pipeline results
    """
    from ..models.theme import ThemePipelineRun

    pipelines_desc = pipeline if pipeline else "both (technical + fundamental)"

    logger.info("=" * 60)
    logger.info("TASK: Full Theme Discovery Pipeline")
    if run_id:
        logger.info(f"Run ID: {run_id}")
    logger.info(f"Pipeline(s): {pipelines_desc}")
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()
    results = {}
    pipeline_run = None

    try:
        # Get or update pipeline run record
        if run_id:
            pipeline_run = db.query(ThemePipelineRun).filter(
                ThemePipelineRun.run_id == run_id
            ).first()
            if pipeline_run:
                pipeline_run.status = 'running'
                db.commit()

        # Step 1/5: Content Ingestion (0%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'ingestion',
                'step_number': 1,
                'total_steps': 5,
                'percent': 0,
                'message': 'Fetching content from sources...'
            }
        )

        logger.info("\n[Step 1/5] Content Ingestion...")
        results['ingestion'] = ingest_content()

        if pipeline_run:
            pipeline_run.current_step = 'ingestion'
            pipeline_run.total_sources = results['ingestion'].get('total_sources', 0)
            pipeline_run.items_ingested = results['ingestion'].get('new_items', 0)
            db.commit()

        # Step 2/5: Reprocess Failed Items (20%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'reprocessing',
                'step_number': 2,
                'total_steps': 5,
                'percent': 20,
                'message': 'Retrying previously failed extractions...',
                'ingestion_result': results['ingestion']
            }
        )

        logger.info("\n[Step 2/5] Reprocess Failed Items...")
        results['reprocessing'] = reprocess_failed_themes(limit=500, pipeline=pipeline)

        if pipeline_run:
            pipeline_run.current_step = 'reprocessing'
            pipeline_run.items_reprocessed = results['reprocessing'].get('reprocessed_count', 0)
            db.commit()

        # Step 3/5: Theme Extraction (40%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'extraction',
                'step_number': 3,
                'total_steps': 5,
                'percent': 40,
                'message': 'Extracting themes from content...',
                'ingestion_result': results['ingestion'],
                'reprocessing_result': results['reprocessing'],
            }
        )

        logger.info("\n[Step 3/5] Theme Extraction...")
        results['extraction'] = extract_themes(limit=500, pipeline=pipeline)

        if pipeline_run:
            pipeline_run.current_step = 'extraction'
            pipeline_run.items_processed = results['extraction'].get('processed', 0)
            pipeline_run.themes_extracted = len(results['extraction'].get('new_themes', []))
            db.commit()

        # Step 4/5: Metrics Calculation (60%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'metrics',
                'step_number': 4,
                'total_steps': 5,
                'percent': 60,
                'message': 'Calculating theme metrics...',
                'ingestion_result': results['ingestion'],
                'reprocessing_result': results['reprocessing'],
                'extraction_result': results['extraction'],
            }
        )

        logger.info("\n[Step 4/5] Metrics Calculation...")
        results['metrics'] = calculate_theme_metrics(pipeline=pipeline)

        if pipeline_run:
            pipeline_run.current_step = 'metrics'
            pipeline_run.themes_updated = results['metrics'].get('themes_updated', 0)
            db.commit()

        # Step 5/5: Alert Check (80%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'alerts',
                'step_number': 5,
                'total_steps': 5,
                'percent': 80,
                'message': 'Checking for alerts...',
                'ingestion_result': results['ingestion'],
                'reprocessing_result': results['reprocessing'],
                'extraction_result': results['extraction'],
                'metrics_result': results['metrics'],
            }
        )

        logger.info("\n[Step 5/5] Alert Check...")
        results['alerts'] = check_alerts()

        total_duration = time.time() - start_time

        # Mark as completed
        if pipeline_run:
            pipeline_run.current_step = 'completed'
            pipeline_run.status = 'completed'
            pipeline_run.alerts_generated = results['alerts'].get('new_alerts', 0)
            pipeline_run.completed_at = datetime.now()
            db.commit()

        logger.info("=" * 60)
        logger.info("Full Pipeline Complete!")
        logger.info(f"Total duration: {total_duration:.2f}s")
        logger.info("=" * 60)

        return {
            'status': 'complete',
            'run_id': run_id,
            'steps': results,
            'total_duration_seconds': round(total_duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        if pipeline_run:
            try:
                pipeline_run.status = 'failed'
                pipeline_run.error_message = str(e)
                pipeline_run.completed_at = datetime.now()
                db.commit()
            except Exception:
                pass
        raise

    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.poll_due_sources')
def poll_due_sources():
    """
    Poll content sources that are due for refresh based on fetch_interval_minutes.

    Sources are fetched if:
    - last_fetched_at is NULL, OR
    - last_fetched_at + fetch_interval_minutes < now()

    Sources are processed in priority order (highest first).

    Returns:
        Dict with polling results
    """
    from datetime import timedelta
    from sqlalchemy import or_

    from ..models.theme import ContentSource
    from ..services.content_ingestion_service import ContentIngestionService

    logger.info("=" * 60)
    logger.info("TASK: Poll Due Content Sources")
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        now = datetime.utcnow()

        # Find sources due for refresh:
        # - last_fetched_at is NULL (never fetched), OR
        # - last_fetched_at + fetch_interval_minutes < now
        #
        # SQLAlchemy doesn't support arithmetic with columns directly in filter,
        # so we fetch all active sources and filter in Python
        all_active_sources = db.query(ContentSource).filter(
            ContentSource.is_active == True
        ).order_by(ContentSource.priority.desc()).all()

        due_sources = []
        for source in all_active_sources:
            if source.last_fetched_at is None:
                # Never fetched - always due
                due_sources.append(source)
            else:
                # Check if interval has elapsed
                interval = timedelta(minutes=source.fetch_interval_minutes or 60)
                if source.last_fetched_at + interval < now:
                    due_sources.append(source)

        if not due_sources:
            duration = time.time() - start_time
            logger.info(f"No sources due for polling (checked {len(all_active_sources)} sources)")
            logger.info("=" * 60)
            return {
                "status": "no_sources_due",
                "sources_checked": len(all_active_sources),
                "sources_polled": 0,
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.now().isoformat()
            }

        service = ContentIngestionService(db)
        results = []

        for source in due_sources:
            try:
                count = service.fetch_source(source)
                results.append({
                    "source": source.name,
                    "type": source.source_type,
                    "items": count,
                    "status": "success"
                })
                logger.info(f"  Polled {source.name}: {count} new items")
            except Exception as e:
                # Rollback the failed transaction before continuing
                db.rollback()
                results.append({
                    "source": source.name,
                    "type": source.source_type,
                    "error": str(e),
                    "status": "error"
                })
                logger.error(f"  Error polling {source.name}: {e}")

        duration = time.time() - start_time
        total_items = sum(r.get("items", 0) for r in results if r.get("status") == "success")
        errors = sum(1 for r in results if r.get("status") == "error")

        logger.info(f"Content source polling complete in {duration:.2f}s")
        logger.info(f"  Sources polled: {len(due_sources)}")
        logger.info(f"  Total new items: {total_items}")
        logger.info(f"  Errors: {errors}")
        logger.info("=" * 60)

        return {
            "status": "completed",
            "sources_checked": len(all_active_sources),
            "sources_polled": len(due_sources),
            "total_new_items": total_items,
            "errors": errors,
            "results": results,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in poll_due_sources task: {e}", exc_info=True)
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.theme_discovery_tasks.consolidate_themes')
def consolidate_themes(dry_run: bool = False):
    """
    Run theme consolidation to identify and merge duplicate themes.

    Uses embedding similarity + LLM verification to:
    1. Generate/update embeddings for all active themes
    2. Find similar theme pairs via cosine similarity
    3. Verify potential merges with LLM
    4. Auto-merge high confidence pairs (similarity >= 0.95, LLM confidence >= 0.90)
    5. Queue moderate confidence pairs for human review

    Args:
        dry_run: If True, only report what would happen without executing merges

    Returns:
        Dict with consolidation results

    Recommended scheduling: Weekly (Sunday 4 AM)
    """
    logger.info("=" * 60)
    logger.info(f"TASK: Theme Consolidation (dry_run={dry_run})")
    logger.info("=" * 60)

    from ..services.theme_merging_service import ThemeMergingService

    db = SessionLocal()
    start_time = time.time()

    try:
        service = ThemeMergingService(db)
        result = service.run_consolidation(dry_run=dry_run)

        duration = time.time() - start_time

        logger.info(f"Theme consolidation complete in {duration:.2f}s")
        logger.info(f"  Embeddings updated: {result['embeddings_updated']}")
        logger.info(f"  Pairs found: {result['pairs_found']}")
        logger.info(f"  LLM verified: {result['llm_verified']}")
        logger.info(f"  Auto-merged: {result['auto_merged']}")
        logger.info(f"  Queued for review: {result['queued_for_review']}")

        if result['errors']:
            logger.warning(f"  Errors: {len(result['errors'])}")
            for error in result['errors'][:5]:  # Log first 5 errors
                logger.warning(f"    - {error}")

        logger.info("=" * 60)

        return {
            'dry_run': dry_run,
            'embeddings_updated': result['embeddings_updated'],
            'pairs_found': result['pairs_found'],
            'llm_verified': result['llm_verified'],
            'auto_merged': result['auto_merged'],
            'queued_for_review': result['queued_for_review'],
            'errors': len(result['errors']),
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in theme consolidation task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()
