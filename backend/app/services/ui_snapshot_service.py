"""Published UI bootstrap snapshot service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import logging
from threading import Lock
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database import is_corruption_error
from app.domain.scanning.filter_spec import PageSpec, QuerySpec, SortOrder, SortSpec
from app.infra.db.uow import SqlUnitOfWork
from app.models.industry import IBDGroupRank
from app.models.market_breadth import MarketBreadth
from app.models.scan_result import Scan
from app.models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ContentSource,
    ThemeAlert,
    ThemeCluster,
    ThemeMergeSuggestion,
    ThemeMetrics,
    ThemePipelineRun,
)
from app.models.ui_view_snapshot import UIViewSnapshot, UIViewSnapshotPointer
from app.schemas.groups import GroupRankResponse, GroupRankingsResponse, MoversResponse
from app.schemas.scanning import (
    FilterOptionsResponse,
    ScanListItem,
    ScanListResponse,
    ScanResultItem,
    ScanResultsResponse,
    ScanStatusResponse,
)
from app.schemas.theme import (
    AlertsResponse,
    CandidateThemeQueueItemResponse,
    CandidateThemeQueueResponse,
    CandidateThemeQueueSummaryBandResponse,
    EmergingThemeResponse,
    EmergingThemesResponse,
    L1CategoriesResponse,
    L1CategoryItem,
    L1ThemeRankingItem,
    L1ThemeRankingsResponse,
    ThemeAlertResponse,
    ThemePipelineObservabilityResponse,
    ThemeRankingItem,
    ThemeRankingsResponse,
)
from app.services.ibd_group_rank_service import IBDGroupRankService
from app.services.price_cache_service import PriceCacheService
from app.services.stock_universe_service import stock_universe_service
from app.services.theme_discovery_service import ThemeDiscoveryService
from app.services.theme_pipeline_state_service import compute_pipeline_observability
from app.services.theme_taxonomy_service import ThemeTaxonomyService
from app.use_cases.scanning.get_filter_options import GetFilterOptionsQuery, GetFilterOptionsUseCase
from app.use_cases.scanning.get_scan_results import GetScanResultsQuery, GetScanResultsUseCase

logger = logging.getLogger(__name__)


SCAN_VIEW_KEY = "scan_bootstrap"
BREADTH_VIEW_KEY = "breadth_bootstrap"
GROUPS_VIEW_KEY = "groups_bootstrap"
THEMES_VIEW_KEY = "themes_bootstrap"

DEFAULT_SCAN_PER_PAGE = 50
DEFAULT_BREADTH_RANGE = "1M"
DEFAULT_GROUP_PERIOD = "1w"
DEFAULT_THEME_SOURCE_TYPES = ["substack", "twitter", "news", "reddit"]
_SNAPSHOT_SCHEMA_LOCK = Lock()
_SNAPSHOT_SCHEMA_READY = False
_SNAPSHOT_CACHE_TABLES = ("ui_view_snapshot_pointers", "ui_view_snapshots")


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_revision: str
    source_revision: str
    published_at: datetime
    is_stale: bool
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_revision": self.snapshot_revision,
            "source_revision": self.source_revision,
            "published_at": self.published_at,
            "is_stale": self.is_stale,
            "payload": self.payload,
        }


class UISnapshotService:
    """Build, publish, and retrieve UI bootstrap snapshots."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        self._engine = session_factory.kw.get("bind")

    def ui_snapshot_flags(self) -> dict[str, bool]:
        """Expose frontend-visible bootstrap availability flags."""
        return {
            "enabled": True,
            "scan": True,
            "breadth": True,
            "groups": True,
            "themes": settings.feature_themes,
        }

    def get_scan_bootstrap(self, scan_id: str | None = None) -> SnapshotResult | None:
        self._ensure_schema()
        variant_key = self._scan_variant_key(scan_id)
        return self._run_with_storage_recovery(
            lambda db: self._get_snapshot(
                db=db,
                view_key=SCAN_VIEW_KEY,
                variant_key=variant_key,
                source_revision=self._resolve_scan_source_revision(db, scan_id),
            )
        )

    def publish_scan_bootstrap(self, scan_id: str | None = None) -> SnapshotResult:
        self._ensure_schema()
        variant_key = self._scan_variant_key(scan_id)
        return self._run_with_storage_recovery(
            lambda db: self._publish(
                db=db,
                view_key=SCAN_VIEW_KEY,
                variant_key=variant_key,
                source_revision=self._resolve_scan_source_revision(db, scan_id),
                payload=self._build_scan_payload(scan_id),
            )
        )

    def get_breadth_bootstrap(self) -> SnapshotResult | None:
        self._ensure_schema()
        return self._run_with_storage_recovery(
            lambda db: self._get_snapshot(
                db=db,
                view_key=BREADTH_VIEW_KEY,
                variant_key="default",
                source_revision=self._resolve_breadth_source_revision(db),
            )
        )

    def publish_breadth_bootstrap(self) -> SnapshotResult:
        self._ensure_schema()
        return self._run_with_storage_recovery(
            lambda db: self._publish(
                db=db,
                view_key=BREADTH_VIEW_KEY,
                variant_key="default",
                source_revision=self._resolve_breadth_source_revision(db),
                payload=self._build_breadth_payload(),
            )
        )

    def get_groups_bootstrap(self) -> SnapshotResult | None:
        self._ensure_schema()
        return self._run_with_storage_recovery(
            lambda db: self._get_snapshot(
                db=db,
                view_key=GROUPS_VIEW_KEY,
                variant_key="default",
                source_revision=self._resolve_groups_source_revision(db),
            )
        )

    def publish_groups_bootstrap(self) -> SnapshotResult:
        self._ensure_schema()
        return self._run_with_storage_recovery(
            lambda db: self._publish(
                db=db,
                view_key=GROUPS_VIEW_KEY,
                variant_key="default",
                source_revision=self._resolve_groups_source_revision(db),
                payload=self._build_groups_payload(),
            )
        )

    def get_themes_bootstrap(self, pipeline: str = "technical", theme_view: str = "grouped") -> SnapshotResult | None:
        self._ensure_schema()
        variant_key = self._themes_variant_key(pipeline, theme_view)
        return self._run_with_storage_recovery(
            lambda db: self._get_snapshot(
                db=db,
                view_key=THEMES_VIEW_KEY,
                variant_key=variant_key,
                source_revision=self._resolve_themes_source_revision(db, pipeline),
            )
        )

    def publish_themes_bootstrap(self, pipeline: str = "technical", theme_view: str = "grouped") -> SnapshotResult:
        self._ensure_schema()
        variant_key = self._themes_variant_key(pipeline, theme_view)
        return self._run_with_storage_recovery(
            lambda db: self._publish(
                db=db,
                view_key=THEMES_VIEW_KEY,
                variant_key=variant_key,
                source_revision=self._resolve_themes_source_revision(db, pipeline),
                payload=self._build_themes_payload(pipeline=pipeline, theme_view=theme_view),
            )
        )

    def publish_all(self) -> dict[str, dict[str, Any]]:
        """Rebuild all bootstrap variants."""
        published = {
            "scan_latest": self.publish_scan_bootstrap().to_dict(),
            "breadth": self.publish_breadth_bootstrap().to_dict(),
            "groups": self.publish_groups_bootstrap().to_dict(),
        }
        if settings.feature_themes:
            published.update({
                "themes_technical_grouped": self.publish_themes_bootstrap("technical", "grouped").to_dict(),
                "themes_technical_flat": self.publish_themes_bootstrap("technical", "flat").to_dict(),
                "themes_fundamental_grouped": self.publish_themes_bootstrap("fundamental", "grouped").to_dict(),
                "themes_fundamental_flat": self.publish_themes_bootstrap("fundamental", "flat").to_dict(),
            })
        return published

    def _ensure_schema(self) -> None:
        global _SNAPSHOT_SCHEMA_READY
        if _SNAPSHOT_SCHEMA_READY or self._engine is None:
            return
        with _SNAPSHOT_SCHEMA_LOCK:
            if _SNAPSHOT_SCHEMA_READY:
                return
            from app.db_migrations.ui_view_snapshot_migration import migrate_ui_view_snapshot_tables

            migrate_ui_view_snapshot_tables(self._engine)
            _SNAPSHOT_SCHEMA_READY = True

    def _run_with_storage_recovery(self, fn):
        """Retry UI snapshot reads/writes once after resetting corrupt cache tables."""
        try:
            with self._session_factory() as db:
                return fn(db)
        except Exception as exc:
            if not is_corruption_error(exc):
                raise
            self._reset_corrupt_snapshot_storage(exc)
            with self._session_factory() as db:
                return fn(db)

    def _reset_corrupt_snapshot_storage(self, exc: Exception) -> None:
        """Drop and recreate rebuildable UI snapshot cache tables after corruption."""
        logger.warning(
            "Resetting UI snapshot cache tables after SQLite corruption signature: %s",
            exc,
        )
        global _SNAPSHOT_SCHEMA_READY
        with _SNAPSHOT_SCHEMA_LOCK:
            if self._engine is None:
                return
            with self._engine.begin() as conn:
                try:
                    _drop_snapshot_tables(conn)
                except Exception as reset_exc:
                    if not is_corruption_error(reset_exc):
                        raise
                    logger.warning(
                        "Normal UI snapshot table drop failed due to SQLite corruption; "
                        "forcing schema cleanup: %s",
                        reset_exc,
                    )
                    _force_forget_snapshot_tables(conn)
            _SNAPSHOT_SCHEMA_READY = False
            from app.db_migrations.ui_view_snapshot_migration import migrate_ui_view_snapshot_tables

            migrate_ui_view_snapshot_tables(self._engine)
            _SNAPSHOT_SCHEMA_READY = True

    def _get_snapshot(
        self,
        *,
        db: Session,
        view_key: str,
        variant_key: str,
        source_revision: str,
    ) -> SnapshotResult | None:
        current = self._get_current(db, view_key=view_key, variant_key=variant_key)
        if current is None:
            return None
        if current.source_revision == source_revision:
            return current
        return SnapshotResult(
            snapshot_revision=current.snapshot_revision,
            source_revision=current.source_revision,
            published_at=current.published_at,
            is_stale=True,
            payload=current.payload,
        )

    def _get_current(self, db: Session, *, view_key: str, variant_key: str) -> SnapshotResult | None:
        pointer = (
            db.query(UIViewSnapshotPointer)
            .filter(
                UIViewSnapshotPointer.view_key == view_key,
                UIViewSnapshotPointer.variant_key == variant_key,
            )
            .first()
        )
        if pointer is None or pointer.snapshot is None:
            return None
        snapshot = pointer.snapshot
        return SnapshotResult(
            snapshot_revision=str(snapshot.id),
            source_revision=snapshot.source_revision,
            published_at=snapshot.published_at,
            is_stale=False,
            payload=snapshot.payload_json or {},
        )

    def _publish(
        self,
        *,
        db: Session,
        view_key: str,
        variant_key: str,
        source_revision: str,
        payload: dict[str, Any],
    ) -> SnapshotResult:
        payload = _json_safe(payload)
        current = self._get_current(db, view_key=view_key, variant_key=variant_key)
        row = (
            db.query(UIViewSnapshot)
            .filter(
                UIViewSnapshot.view_key == view_key,
                UIViewSnapshot.variant_key == variant_key,
                UIViewSnapshot.source_revision == source_revision,
            )
            .first()
        )
        if row is None:
            row = UIViewSnapshot(
                view_key=view_key,
                variant_key=variant_key,
                source_revision=source_revision,
                payload_json=payload,
            )
            db.add(row)
            db.flush()
        else:
            row.payload_json = payload
            db.flush()

        pointer = (
            db.query(UIViewSnapshotPointer)
            .filter(
                UIViewSnapshotPointer.view_key == view_key,
                UIViewSnapshotPointer.variant_key == variant_key,
            )
            .first()
        )
        if pointer is None:
            pointer = UIViewSnapshotPointer(
                view_key=view_key,
                variant_key=variant_key,
                snapshot_id=row.id,
            )
            db.add(pointer)
        else:
            pointer.snapshot_id = row.id
        db.flush()

        keep_ids = {row.id}
        if current is not None:
            keep_ids.add(int(current.snapshot_revision))
        self._prune_old_revisions(db, view_key=view_key, variant_key=variant_key, keep_ids=keep_ids)
        db.commit()
        db.refresh(row)
        return SnapshotResult(
            snapshot_revision=str(row.id),
            source_revision=row.source_revision,
            published_at=row.published_at,
            is_stale=False,
            payload=row.payload_json or {},
        )

    def _prune_old_revisions(self, db: Session, *, view_key: str, variant_key: str, keep_ids: set[int]) -> None:
        keep = tuple(sorted(i for i in keep_ids if i > 0))
        query = db.query(UIViewSnapshot).filter(
            UIViewSnapshot.view_key == view_key,
            UIViewSnapshot.variant_key == variant_key,
        )
        if keep:
            query = query.filter(~UIViewSnapshot.id.in_(keep))
        query.delete(synchronize_session=False)
        db.flush()

    def _resolve_scan_source_revision(self, db: Session, scan_id: str | None) -> str:
        if scan_id:
            return scan_id
        latest = (
            db.query(Scan.scan_id)
            .filter(Scan.status.in_(("completed", "cancelled")))
            .order_by(Scan.completed_at.desc(), Scan.started_at.desc())
            .first()
        )
        return latest[0] if latest else "none"

    def _resolve_breadth_source_revision(self, db: Session) -> str:
        latest = db.query(func.max(MarketBreadth.date)).scalar()
        return latest.isoformat() if latest else "none"

    def _resolve_groups_source_revision(self, db: Session) -> str:
        latest = db.query(func.max(IBDGroupRank.date)).scalar()
        return latest.isoformat() if latest else "none"

    def _resolve_themes_source_revision(self, db: Session, pipeline: str) -> str:
        latest_metrics = db.query(func.max(ThemeMetrics.date)).filter(ThemeMetrics.pipeline == pipeline).scalar()
        latest_cluster_update = db.query(func.max(ThemeCluster.updated_at)).filter(ThemeCluster.pipeline == pipeline).scalar()
        latest_pipeline_run = db.query(func.max(ThemePipelineRun.completed_at)).filter(
            (ThemePipelineRun.pipeline == pipeline) | (ThemePipelineRun.pipeline.is_(None))
        ).scalar()
        latest_alert = db.query(func.max(ThemeAlert.triggered_at)).scalar()
        latest_merge = db.query(func.max(ThemeMergeSuggestion.created_at)).scalar()
        latest_merge_review = db.query(func.max(ThemeMergeSuggestion.reviewed_at)).scalar()
        candidate_count = db.query(func.count(ThemeCluster.id)).filter(
            ThemeCluster.pipeline == pipeline,
            ThemeCluster.lifecycle_state == "candidate",
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
        ).scalar() or 0
        failed_count = self._query_failed_items_count(db, pipeline)
        parts = [
            latest_metrics.isoformat() if latest_metrics else "none",
            latest_cluster_update.isoformat() if latest_cluster_update else "none",
            latest_pipeline_run.isoformat() if latest_pipeline_run else "none",
            latest_alert.isoformat() if latest_alert else "none",
            latest_merge.isoformat() if latest_merge else "none",
            latest_merge_review.isoformat() if latest_merge_review else "none",
            str(candidate_count),
            str(failed_count),
        ]
        return "|".join(parts)

    def _build_scan_payload(self, scan_id: str | None) -> dict[str, Any]:
        filter_use_case = GetFilterOptionsUseCase()
        results_use_case = GetScanResultsUseCase()
        with SqlUnitOfWork(self._session_factory) as uow:
            scans = uow.scans.list_recent(limit=20)
            scan_items = [
                ScanListItem(
                    scan_id=scan.scan_id,
                    status=scan.status,
                    universe=scan.universe or "unknown",
                    universe_type=scan.universe_type,
                    universe_exchange=scan.universe_exchange,
                    universe_index=scan.universe_index,
                    universe_symbols_count=len(scan.universe_symbols) if scan.universe_symbols else None,
                    total_stocks=scan.total_stocks or 0,
                    passed_stocks=scan.passed_stocks or 0,
                    started_at=scan.started_at,
                    completed_at=scan.completed_at,
                    source="feature_store" if scan.feature_run_id else "scan_results",
                )
                for scan in scans
            ]
            recent_scans = ScanListResponse(scans=scan_items).model_dump(mode="json")

            selected_scan = uow.scans.get_by_scan_id(scan_id) if scan_id else next(
                (scan for scan in scans if scan.status in {"completed", "cancelled"}),
                None,
            )
            payload: dict[str, Any] = {
                "universe_stats": stock_universe_service.get_stats(uow.session),
                "recent_scans": recent_scans,
                "selected_scan": None,
                "selected_scan_status": None,
                "filter_options": FilterOptionsResponse(
                    ibd_industries=[],
                    gics_sectors=[],
                    ratings=[],
                ).model_dump(mode="json"),
                "results_page": ScanResultsResponse(
                    scan_id=scan_id or "",
                    total=0,
                    page=1,
                    per_page=DEFAULT_SCAN_PER_PAGE,
                    pages=0,
                    results=[],
                ).model_dump(mode="json"),
            }

            if selected_scan is None:
                return payload

            payload["selected_scan"] = next(
                (item for item in recent_scans["scans"] if item["scan_id"] == selected_scan.scan_id),
                ScanListItem(
                    scan_id=selected_scan.scan_id,
                    status=selected_scan.status,
                    universe=selected_scan.universe or "unknown",
                    universe_type=selected_scan.universe_type,
                    universe_exchange=selected_scan.universe_exchange,
                    universe_index=selected_scan.universe_index,
                    universe_symbols_count=len(selected_scan.universe_symbols) if selected_scan.universe_symbols else None,
                    total_stocks=selected_scan.total_stocks or 0,
                    passed_stocks=selected_scan.passed_stocks or 0,
                    started_at=selected_scan.started_at,
                    completed_at=selected_scan.completed_at,
                    source="feature_store" if selected_scan.feature_run_id else "scan_results",
                ).model_dump(mode="json"),
            )
            completed_stocks = selected_scan.total_stocks or 0
            progress = 100.0 if selected_scan.status == "completed" else 0.0
            if selected_scan.status == "cancelled":
                completed_stocks = uow.scan_results.count_by_scan_id(selected_scan.scan_id)
                progress = (
                    round((completed_stocks / selected_scan.total_stocks) * 100, 2)
                    if selected_scan.total_stocks
                    else 0.0
                )
            elif selected_scan.status in {"queued", "running"}:
                completed_stocks = uow.scan_results.count_by_scan_id(selected_scan.scan_id)
                progress = (
                    round((completed_stocks / selected_scan.total_stocks) * 100, 2)
                    if selected_scan.total_stocks
                    else 0.0
                )
            payload["selected_scan_status"] = ScanStatusResponse(
                scan_id=selected_scan.scan_id,
                status=selected_scan.status,
                progress=progress,
                total_stocks=selected_scan.total_stocks or 0,
                completed_stocks=completed_stocks,
                passed_stocks=selected_scan.passed_stocks or 0,
                started_at=selected_scan.started_at,
                eta_seconds=None,
            ).model_dump(mode="json")

            if selected_scan.status not in {"completed", "cancelled"}:
                return payload

            filter_result = filter_use_case.execute(uow, GetFilterOptionsQuery(scan_id=selected_scan.scan_id))
            payload["filter_options"] = FilterOptionsResponse(
                ibd_industries=list(filter_result.options.ibd_industries),
                gics_sectors=list(filter_result.options.gics_sectors),
                ratings=list(filter_result.options.ratings),
            ).model_dump(mode="json")

            result = results_use_case.execute(
                uow,
                GetScanResultsQuery(
                    scan_id=selected_scan.scan_id,
                    query_spec=QuerySpec(
                        sort=SortSpec(field="composite_score", order=SortOrder.DESC),
                        page=PageSpec(page=1, per_page=DEFAULT_SCAN_PER_PAGE),
                    ),
                    include_sparklines=True,
                    include_setup_payload=False,
                ),
            )
            payload["results_page"] = ScanResultsResponse(
                scan_id=selected_scan.scan_id,
                total=result.page.total,
                page=result.page.page,
                per_page=result.page.per_page,
                pages=result.page.total_pages,
                results=[
                    ScanResultItem.model_validate(self._scan_item_to_payload(item))
                    for item in result.page.items
                ],
            ).model_dump(mode="json")
            return payload

    def _build_breadth_payload(self) -> dict[str, Any]:
        with self._session_factory() as db:
            current = db.query(MarketBreadth).order_by(MarketBreadth.date.desc()).first()
            total_records = db.query(func.count(MarketBreadth.id)).scalar() or 0
            min_date = db.query(func.min(MarketBreadth.date)).scalar()
            max_date = db.query(func.max(MarketBreadth.date)).scalar()
            end_date = datetime.utcnow().date()
            history_start = end_date - timedelta(days=90)
            chart_start = end_date - timedelta(days=31)
            history = (
                db.query(MarketBreadth)
                .filter(MarketBreadth.date >= history_start, MarketBreadth.date <= end_date)
                .order_by(MarketBreadth.date.desc())
                .all()
            )
            chart = (
                db.query(MarketBreadth)
                .filter(MarketBreadth.date >= chart_start, MarketBreadth.date <= end_date)
                .order_by(MarketBreadth.date.desc())
                .all()
            )
            return {
                "current": market_breadth_to_dict(current),
                "summary": {
                    "latest_date": max_date,
                    "total_records": total_records,
                    "date_range_start": min_date,
                    "date_range_end": max_date,
                },
                "history_90d": [market_breadth_to_dict(row) for row in history],
                "chart_range": DEFAULT_BREADTH_RANGE,
                "chart_data": [market_breadth_to_dict(row) for row in chart],
                "spy_overlay": self._get_cached_price_history("SPY", "1mo"),
            }

    def _build_groups_payload(self) -> dict[str, Any]:
        with self._session_factory() as db:
            service = IBDGroupRankService.get_instance()
            rankings = service.get_current_rankings(db, limit=197)
            movers = service.get_rank_movers(db, period=DEFAULT_GROUP_PERIOD, limit=10)
            ranking_date = rankings[0]["date"] if rankings else None
            return {
                "rankings": GroupRankingsResponse(
                    date=ranking_date,
                    total_groups=len(rankings),
                    rankings=[GroupRankResponse(**row) for row in rankings],
                ).model_dump(mode="json"),
                "movers_period": DEFAULT_GROUP_PERIOD,
                "movers": MoversResponse(
                    period=movers["period"],
                    gainers=[GroupRankResponse(**row) for row in movers.get("gainers", [])],
                    losers=[GroupRankResponse(**row) for row in movers.get("losers", [])],
                ).model_dump(mode="json"),
                "task_controls_enabled": settings.feature_tasks,
            }

    def _build_themes_payload(self, *, pipeline: str, theme_view: str) -> dict[str, Any]:
        with self._session_factory() as db:
            discovery = ThemeDiscoveryService(db, pipeline=pipeline)
            taxonomy = ThemeTaxonomyService(db, pipeline=pipeline)

            self._ensure_l2_theme_metrics(db, pipeline)
            emerging = discovery.discover_emerging_themes(min_velocity=1.5, min_mentions=3)
            alerts_rows = (
                db.query(ThemeAlert)
                .filter(ThemeAlert.is_dismissed == False)
                .order_by(ThemeAlert.triggered_at.desc())
                .limit(50)
                .all()
            )
            unread_count = (
                db.query(func.count(ThemeAlert.id))
                .filter(ThemeAlert.is_read == False, ThemeAlert.is_dismissed == False)
                .scalar()
                or 0
            )
            pending_merges = (
                db.query(func.count(ThemeMergeSuggestion.id))
                .filter(ThemeMergeSuggestion.status == "pending")
                .scalar()
                or 0
            )
            queue_rows, queue_total = discovery.get_candidate_theme_queue(limit=1, offset=0)
            confidence_bands = discovery.get_candidate_theme_confidence_bands()
            failed_count = self._query_failed_items_count(db, pipeline)
            observability = compute_pipeline_observability(db=db, pipeline=pipeline, max_age_days=30)

            payload: dict[str, Any] = {
                "pipeline": pipeline,
                "theme_view": theme_view,
                "defaults": {
                    "selected_tab": "all",
                    "selected_source_types": list(DEFAULT_THEME_SOURCE_TYPES),
                    "page": 0,
                    "order_by": "rank",
                    "order": "asc",
                },
                "emerging": EmergingThemesResponse(
                    count=len(emerging),
                    themes=[EmergingThemeResponse(**row) for row in emerging],
                ).model_dump(mode="json"),
                "alerts": AlertsResponse(
                    total=len(alerts_rows),
                    unread=unread_count,
                    alerts=[ThemeAlertResponse.model_validate(row) for row in alerts_rows],
                ).model_dump(mode="json"),
                "pending_merge_count": int(pending_merges),
                "candidate_queue_summary": CandidateThemeQueueResponse(
                    total=queue_total,
                    items=[CandidateThemeQueueItemResponse(**row) for row in queue_rows[:1]],
                    confidence_bands=[CandidateThemeQueueSummaryBandResponse(**row) for row in confidence_bands],
                ).model_dump(mode="json"),
                "failed_items_count": {
                    "failed_count": int(failed_count),
                    "max_age_days": 30,
                },
                "observability": ThemePipelineObservabilityResponse(**observability).model_dump(mode="json"),
            }

            if theme_view == "grouped":
                categories = taxonomy.get_categories()
                rankings, total = taxonomy.get_l1_themes(
                    category_filter=None,
                    limit=50,
                    offset=0,
                    sort_by="momentum_score",
                    sort_order="desc",
                )
                payload["l1_categories"] = L1CategoriesResponse(
                    categories=[L1CategoryItem(**row) for row in categories]
                ).model_dump(mode="json")
                payload["l1_rankings"] = L1ThemeRankingsResponse(
                    pipeline=pipeline,
                    total=total,
                    rankings=[L1ThemeRankingItem(**row) for row in rankings],
                ).model_dump(mode="json")
            else:
                rankings, total = discovery.get_theme_rankings(
                    limit=50,
                    status_filter=None,
                    source_types_filter=list(DEFAULT_THEME_SOURCE_TYPES),
                    offset=0,
                )
                payload["rankings"] = ThemeRankingsResponse(
                    date=datetime.utcnow().date().isoformat(),
                    total_themes=total,
                    pipeline=pipeline,
                    rankings=[ThemeRankingItem(**row) for row in rankings],
                ).model_dump(mode="json")
            return payload

    def _query_failed_items_count(self, db: Session, pipeline: str) -> int:
        cutoff = datetime.utcnow() - timedelta(days=30)
        query = db.query(func.count(ContentItemPipelineState.id)).join(
            ContentItem,
            ContentItem.id == ContentItemPipelineState.content_item_id,
        ).filter(
            ContentItemPipelineState.status == "failed_retryable",
            ContentItem.published_at >= cutoff,
            ContentItemPipelineState.pipeline == pipeline,
        )
        source_ids = self._resolve_source_ids_for_pipeline(db, pipeline)
        if source_ids:
            query = query.filter(ContentItem.source_id.in_(source_ids))
        else:
            return 0
        return int(query.scalar() or 0)

    def _resolve_source_ids_for_pipeline(self, db: Session, pipeline: str) -> list[int]:
        source_ids: list[int] = []
        rows = db.query(ContentSource.id, ContentSource.pipelines).filter(ContentSource.is_active == True).all()
        for source_id, source_pipelines in rows:
            parsed = source_pipelines or ["technical", "fundamental"]
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except Exception:
                    parsed = ["technical", "fundamental"]
            if pipeline in parsed:
                source_ids.append(source_id)
        return source_ids

    def _ensure_l2_theme_metrics(self, db: Session, pipeline: str) -> None:
        themes_without_metrics = db.query(ThemeCluster).filter(
            ThemeCluster.pipeline == pipeline,
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ~ThemeCluster.id.in_(db.query(ThemeMetrics.theme_cluster_id).distinct()),
        ).count()
        if themes_without_metrics > 0:
            logger.info("UI snapshot builder backfilling %s theme metrics for pipeline=%s", themes_without_metrics, pipeline)
            ThemeDiscoveryService(db, pipeline=pipeline).update_all_theme_metrics()

    def _scan_variant_key(self, scan_id: str | None) -> str:
        return f"scan:{scan_id}" if scan_id else "latest"

    def _themes_variant_key(self, pipeline: str, theme_view: str) -> str:
        return f"{pipeline}:{theme_view}"

    def _get_cached_price_history(self, symbol: str, period: str) -> list[dict[str, Any]]:
        import pandas as pd

        cache_service = PriceCacheService.get_instance()
        data = cache_service.get_cached_only(symbol.upper(), period="2y")
        if data is None or len(data) == 0:
            return []
        period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        cutoff_date = pd.Timestamp(datetime.utcnow() - timedelta(days=period_days.get(period, 30)))
        if data.index.tz is not None:
            cutoff_date = cutoff_date.tz_localize(data.index.tz)
        data = data[data.index >= cutoff_date]
        df = data.reset_index()
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        return [
            {
                "date": row["Date"],
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
            for _, row in df.iterrows()
        ]

    def _scan_item_to_payload(self, item) -> dict[str, Any]:
        extended_fields = dict(item.extended_fields or {})
        extended_fields.pop("se_explain", None)
        extended_fields.pop("se_candidates", None)
        payload = {
            "symbol": item.symbol,
            "company_name": extended_fields.get("company_name"),
            "composite_score": item.composite_score,
            "rating": item.rating,
            "current_price": item.current_price,
            "screeners_run": item.screeners_run,
        }
        payload.update(extended_fields)
        return payload


def market_breadth_to_dict(row: MarketBreadth | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "date": row.date,
        "stocks_up_4pct": row.stocks_up_4pct,
        "stocks_down_4pct": row.stocks_down_4pct,
        "ratio_5day": row.ratio_5day,
        "ratio_10day": row.ratio_10day,
        "stocks_up_25pct_quarter": row.stocks_up_25pct_quarter,
        "stocks_down_25pct_quarter": row.stocks_down_25pct_quarter,
        "stocks_up_25pct_month": row.stocks_up_25pct_month,
        "stocks_down_25pct_month": row.stocks_down_25pct_month,
        "stocks_up_50pct_month": row.stocks_up_50pct_month,
        "stocks_down_50pct_month": row.stocks_down_50pct_month,
        "stocks_up_13pct_34days": row.stocks_up_13pct_34days,
        "stocks_down_13pct_34days": row.stocks_down_13pct_34days,
        "total_stocks_scanned": row.total_stocks_scanned,
    }


def _json_safe(value: Any) -> Any:
    """Convert snapshot payloads into plain JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _drop_snapshot_tables(conn) -> None:
    """Drop rebuildable UI snapshot cache tables using normal DDL."""
    from ..infra.db.portability import is_sqlite

    if is_sqlite(conn):
        conn.execute(text("DROP TABLE IF EXISTS ui_view_snapshot_pointers"))
        conn.execute(text("DROP TABLE IF EXISTS ui_view_snapshots"))
        return

    conn.execute(text("DROP TABLE IF EXISTS ui_view_snapshot_pointers CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS ui_view_snapshots CASCADE"))


def _force_forget_snapshot_tables(conn) -> None:
    """Remove UI snapshot cache tables from sqlite_master when normal DDL can't read them."""
    from ..infra.db.portability import is_sqlite

    if not is_sqlite(conn):
        _drop_snapshot_tables(conn)
        return

    conn.execute(text("PRAGMA foreign_keys=OFF"))
    conn.execute(text("PRAGMA writable_schema=ON"))
    try:
        conn.execute(
            text(
                """
                DELETE FROM sqlite_master
                WHERE name IN ('ui_view_snapshots', 'ui_view_snapshot_pointers')
                   OR tbl_name IN ('ui_view_snapshots', 'ui_view_snapshot_pointers')
                """
            )
        )
        current_version = conn.execute(text("PRAGMA schema_version")).scalar() or 0
        conn.execute(text(f"PRAGMA schema_version = {int(current_version) + 1}"))
    finally:
        conn.execute(text("PRAGMA writable_schema=OFF"))
        conn.execute(text("PRAGMA foreign_keys=ON"))


def _safe_publish(action: str, *, view_key: str, variant_key: str, source_revision: str | None = None, fn) -> dict[str, Any] | None:
    try:
        result = fn()
        return result.to_dict()
    except Exception:
        logger.exception(
            "UI snapshot publish failed",
            extra={
                "action": action,
                "view_key": view_key,
                "variant_key": variant_key,
                "source_revision": source_revision,
            },
        )
        return None


def safe_publish_scan_bootstrap(scan_id: str | None = None) -> dict[str, Any] | None:
    from app.wiring.bootstrap import get_ui_snapshot_service

    variant_key = f"scan:{scan_id}" if scan_id else "latest"
    return _safe_publish(
        "publish_scan_bootstrap",
        view_key=SCAN_VIEW_KEY,
        variant_key=variant_key,
        source_revision=scan_id,
        fn=lambda: get_ui_snapshot_service().publish_scan_bootstrap(scan_id),
    )


def safe_publish_breadth_bootstrap() -> dict[str, Any] | None:
    from app.wiring.bootstrap import get_ui_snapshot_service

    return _safe_publish(
        "publish_breadth_bootstrap",
        view_key=BREADTH_VIEW_KEY,
        variant_key="default",
        fn=get_ui_snapshot_service().publish_breadth_bootstrap,
    )


def safe_publish_groups_bootstrap() -> dict[str, Any] | None:
    from app.wiring.bootstrap import get_ui_snapshot_service

    return _safe_publish(
        "publish_groups_bootstrap",
        view_key=GROUPS_VIEW_KEY,
        variant_key="default",
        fn=get_ui_snapshot_service().publish_groups_bootstrap,
    )


def safe_publish_themes_bootstrap_variants(pipeline: str | None = None) -> dict[str, dict[str, Any] | None]:
    from app.wiring.bootstrap import get_ui_snapshot_service

    service = get_ui_snapshot_service()
    pipelines = [pipeline] if pipeline else ["technical", "fundamental"]
    published: dict[str, dict[str, Any] | None] = {}
    for pipeline_name in pipelines:
        for theme_view in ("grouped", "flat"):
            variant_key = f"{pipeline_name}:{theme_view}"
            published[variant_key] = _safe_publish(
                "publish_themes_bootstrap",
                view_key=THEMES_VIEW_KEY,
                variant_key=variant_key,
                fn=lambda pipeline_name=pipeline_name, theme_view=theme_view: service.publish_themes_bootstrap(
                    pipeline_name,
                    theme_view,
                ),
            )
    return published


def safe_publish_all_bootstraps() -> dict[str, dict[str, Any]] | None:
    from app.wiring.bootstrap import get_ui_snapshot_service

    try:
        return get_ui_snapshot_service().publish_all()
    except Exception:
        logger.exception("UI snapshot rebuild failed", extra={"action": "publish_all_bootstraps"})
        return None
