"""Theme content browser query use-cases."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from ..api.v1.themes_common import resolve_source_ids_for_pipeline
from ..database import SessionLocal, is_corruption_error, safe_rollback
from ..models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ContentSource,
    ThemeCluster,
    ThemeMention,
)
from ..schemas.theme import ContentItemWithThemesResponse, ThemeReference
from ..services.theme_content_recovery_service import (
    attempt_reindex_theme_content_storage,
    reset_corrupt_theme_content_storage,
)

logger = logging.getLogger(__name__)

_CONTENT_ITEMS_CSV_HEADER = [
    "ID",
    "Title",
    "Content",
    "URL",
    "Themes",
    "Sentiment",
    "Tickers",
    "Published Date",
    "Source Type",
    "Source Name",
    "Author",
    "Processing Status",
]


def _content_item_csv_row(item: ContentItemWithThemesResponse) -> list[object]:
    return [
        item.id,
        item.title or "",
        item.content or "",
        item.url or "",
        "; ".join(theme.name for theme in item.themes),
        item.primary_sentiment or "",
        "; ".join(item.tickers),
        item.published_at.strftime("%Y-%m-%d %H:%M") if item.published_at else "",
        item.source_type or "",
        item.source_name or "",
        item.author or "",
        item.processing_status or "",
    ]


def _build_content_items_browser_base_query(
    db: Session,
    *,
    source_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    pipeline: Optional[str] = None,
    pipeline_source_ids: Optional[list[int]] = None,
):
    """Build shared base query for content browser list/export reads."""
    from datetime import datetime as dt, timedelta

    base_query = db.query(ContentItem).join(
        ContentSource, ContentItem.source_id == ContentSource.id
    ).filter(
        ContentSource.is_active == True
    )

    if pipeline:
        if pipeline_source_ids is None:
            pipeline_source_ids = resolve_source_ids_for_pipeline(db, pipeline)
        if not pipeline_source_ids:
            return None
        base_query = base_query.filter(ContentItem.source_id.in_(pipeline_source_ids))
    else:
        base_query = base_query.filter(ContentItem.is_processed == True)

    if source_type:
        base_query = base_query.filter(ContentItem.source_type == source_type)

    if date_from:
        try:
            from_date = dt.strptime(date_from, "%Y-%m-%d")
            base_query = base_query.filter(ContentItem.published_at >= from_date)
        except ValueError:
            pass

    if date_to:
        try:
            to_date = dt.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
            base_query = base_query.filter(ContentItem.published_at < to_date)
        except ValueError:
            pass

    return base_query


def _fetch_content_items_with_themes(
    db: Session,
    search: Optional[str] = None,
    source_type: Optional[str] = None,
    sentiment: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "published_at",
    sort_order: str = "desc",
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    pipeline: Optional[str] = None,
) -> tuple[list[ContentItemWithThemesResponse], int]:
    """Fetch browser content items with associated theme/sentiment/ticker aggregations."""
    from sqlalchemy import asc, desc, func, or_

    base_query = _build_content_items_browser_base_query(
        db,
        source_type=source_type,
        date_from=date_from,
        date_to=date_to,
        pipeline=pipeline,
    )
    if base_query is None:
        return [], 0

    if sentiment:
        sentiment_query = db.query(ThemeMention.content_item_id).filter(
            ThemeMention.content_item_id.in_(base_query.with_entities(ContentItem.id))
        )
        if pipeline:
            sentiment_query = sentiment_query.filter(ThemeMention.pipeline == pipeline)
        sentiment_content_ids = {
            row.content_item_id
            for row in sentiment_query.filter(
                ThemeMention.sentiment.isnot(None),
                func.lower(ThemeMention.sentiment) == sentiment.lower(),
            ).distinct().all()
        }
        if not sentiment_content_ids:
            return [], 0
        base_query = base_query.filter(ContentItem.id.in_(sentiment_content_ids))

    if search:
        search_term = f"%{search}%"
        text_match_query = base_query.with_entities(ContentItem.id).filter(
            or_(
                ContentItem.title.ilike(search_term),
                ContentItem.source_name.ilike(search_term),
            )
        )
        text_matched_ids = {row.id for row in text_match_query.all()}

        ticker_match_query = db.query(
            ThemeMention.content_item_id,
            ThemeMention.tickers,
        ).filter(
            ThemeMention.content_item_id.in_(base_query.with_entities(ContentItem.id))
        )
        if pipeline:
            ticker_match_query = ticker_match_query.filter(ThemeMention.pipeline == pipeline)

        search_upper = search.upper()
        ticker_matched_ids: set[int] = set()
        for mention in ticker_match_query.all():
            if mention.tickers:
                for ticker in mention.tickers:
                    if search_upper in str(ticker).upper():
                        ticker_matched_ids.add(mention.content_item_id)
                        break

        matched_ids = text_matched_ids | ticker_matched_ids
        if not matched_ids:
            return [], 0
        base_query = base_query.filter(ContentItem.id.in_(matched_ids))

    total = base_query.count()

    sort_column = getattr(ContentItem, sort_by, ContentItem.published_at)
    if sort_order.lower() == "asc":
        base_query = base_query.order_by(asc(sort_column))
    else:
        base_query = base_query.order_by(desc(sort_column))

    if limit is not None and offset is not None:
        content_items = base_query.offset(offset).limit(limit).all()
    else:
        content_items = base_query.all()

    if not content_items:
        return [], total

    content_ids = [item.id for item in content_items]

    mentions = db.query(
        ThemeMention.content_item_id,
        ThemeMention.theme_cluster_id,
        ThemeMention.sentiment,
        ThemeMention.tickers,
        ThemeCluster.id.label("cluster_id"),
        ThemeCluster.display_name.label("cluster_name"),
    ).outerjoin(
        ThemeCluster, ThemeMention.theme_cluster_id == ThemeCluster.id
    ).filter(
        ThemeMention.content_item_id.in_(content_ids)
    )
    if pipeline:
        mentions = mentions.filter(ThemeMention.pipeline == pipeline)
    mentions = mentions.all()

    pipeline_status_by_content_id: dict[int, str] = {}
    if pipeline:
        status_rows = db.query(
            ContentItemPipelineState.content_item_id,
            ContentItemPipelineState.status,
        ).filter(
            ContentItemPipelineState.content_item_id.in_(content_ids),
            ContentItemPipelineState.pipeline == pipeline,
        ).all()
        pipeline_status_by_content_id = {
            row.content_item_id: row.status for row in status_rows
        }

    mentions_by_content = {}
    for mention in mentions:
        content_id = mention.content_item_id
        if content_id not in mentions_by_content:
            mentions_by_content[content_id] = {
                "themes": [],
                "sentiments": [],
                "sentiment_counts": {},
                "tickers": set(),
            }

        if mention.cluster_id and mention.cluster_name:
            theme_ref = {"id": mention.cluster_id, "name": mention.cluster_name}
            if theme_ref not in mentions_by_content[content_id]["themes"]:
                mentions_by_content[content_id]["themes"].append(theme_ref)

        if mention.sentiment:
            sentiment_counts = mentions_by_content[content_id]["sentiment_counts"]
            sentiment_counts[mention.sentiment] = sentiment_counts.get(mention.sentiment, 0) + 1
            if mention.sentiment not in mentions_by_content[content_id]["sentiments"]:
                mentions_by_content[content_id]["sentiments"].append(mention.sentiment)

        if mention.tickers:
            mentions_by_content[content_id]["tickers"].update(mention.tickers)

    items = []
    for content in content_items:
        mention_data = mentions_by_content.get(
            content.id,
            {"themes": [], "sentiments": [], "sentiment_counts": {}, "tickers": set()},
        )

        sentiments_list = mention_data["sentiments"]
        primary_sentiment = None
        sentiment_counts = mention_data["sentiment_counts"]
        if sentiment_counts:
            primary_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        items.append(
            ContentItemWithThemesResponse(
                id=content.id,
                title=content.title,
                content=content.content,
                url=content.url,
                source_type=content.source_type,
                source_name=content.source_name,
                author=content.author,
                published_at=content.published_at,
                themes=[ThemeReference(**theme_ref) for theme_ref in mention_data["themes"]],
                sentiments=sentiments_list,
                primary_sentiment=primary_sentiment,
                tickers=sorted(list(mention_data["tickers"])),
                processing_status=(
                    pipeline_status_by_content_id.get(content.id, "pending")
                    if pipeline
                    else None
                ),
            )
        )

    return items, total


def _corruption_targets_theme_content_storage(
    *,
    source_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    pipeline: Optional[str] = None,
    **_ignored,
) -> bool:
    """Return True when filtered browser probes hit rebuildable theme-content storage corruption."""
    with SessionLocal() as probe_db:
        try:
            probe_db.query(ContentSource.id).filter(
                ContentSource.is_active == True
            ).limit(1).all()

            pipeline_source_ids = None
            if pipeline:
                pipeline_source_ids = resolve_source_ids_for_pipeline(probe_db, pipeline)
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


def fetch_content_items_with_themes_with_recovery(
    db: Session,
    **kwargs,
) -> tuple[list[ContentItemWithThemesResponse], int]:
    """Retry theme content browser queries after reindexing or rebuilding recoverable storage."""
    try:
        return _fetch_content_items_with_themes(db, **kwargs)
    except Exception as exc:
        if not is_corruption_error(exc):
            raise
        safe_rollback(db)
        attempt_reindex_theme_content_storage(exc)
        try:
            with SessionLocal() as retry_db:
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
            reset_corrupt_theme_content_storage(retry_exc)
            with SessionLocal() as reset_retry_db:
                return _fetch_content_items_with_themes(reset_retry_db, **kwargs)


def render_content_items_csv(items: list[ContentItemWithThemesResponse]) -> tuple[bytes, str]:
    """Render content browser items as UTF-8 BOM CSV bytes and filename."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CONTENT_ITEMS_CSV_HEADER)

    for item in items:
        writer.writerow(_content_item_csv_row(item))

    csv_bytes = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
    filename = f"theme_articles_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    return csv_bytes, filename


def render_content_items_csv_chunk(
    items: list[ContentItemWithThemesResponse],
    *,
    include_header: bool,
    include_bom: bool,
) -> bytes:
    """Render a CSV chunk for streaming exports without buffering the full dataset."""
    if not items and not include_header:
        return b""

    buf = io.StringIO()
    writer = csv.writer(buf)
    if include_header:
        writer.writerow(_CONTENT_ITEMS_CSV_HEADER)
    for item in items:
        writer.writerow(_content_item_csv_row(item))

    chunk = buf.getvalue().encode("utf-8")
    if include_bom:
        return b"\xef\xbb\xbf" + chunk
    return chunk
