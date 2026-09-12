"""SQL aggregation over original evidence using one grouping snapshot."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import JSON, case, func, literal, true

from app.models.theme import ThemeCluster, ThemeConstituent, ThemeMention
from app.services.theme_evidence_eligibility_service import legacy_eligibility_exists
from app.services.theme_group_snapshot import ThemeGroupSnapshot


@dataclass(frozen=True)
class GroupConstituent:
    symbol: str
    source: str | None
    confidence: float
    mention_count: int
    correlation_to_theme: float | None
    first_mentioned_at: datetime | None
    last_mentioned_at: datetime | None


def _post_counts(db, members, pipeline):
    if db.get_bind().dialect.name == "postgresql":
        tickers = (
            func.json_array_elements_text(
                case(
                    (
                        func.json_typeof(ThemeMention.tickers) == "array",
                        ThemeMention.tickers,
                    ),
                    else_=literal([], type_=JSON),
                )
            )
            .table_valued("symbol")
            .render_derived(name="expanded_tickers")
        )
        symbol = tickers.c.symbol
    else:
        tickers = func.json_each(ThemeMention.tickers).table_valued("value")
        symbol = tickers.c.value
    return (
        db.query(
            symbol.label("symbol"),
            func.count(func.distinct(ThemeMention.content_item_id)).label("count"),
        )
        .select_from(ThemeMention)
        .join(tickers, true())
        .filter(
            ThemeMention.theme_cluster_id.in_(members),
            ThemeMention.pipeline == pipeline,
            ThemeMention.social_work_id.is_(None),
            legacy_eligibility_exists(
                ThemeMention.content_item_id, pipeline, active_only=True
            ),
        )
        .group_by(symbol)
        .subquery()
    )


def grouped_constituents(db, theme_id, *, snapshot=None, limit=None):
    snapshot = snapshot if snapshot is not None else ThemeGroupSnapshot.read(db)
    members = snapshot.members(theme_id)
    c = ThemeConstituent
    if len(members) == 1:
        query = db.query(
            c.symbol,
            c.source,
            func.coalesce(c.confidence, 0).label("confidence"),
            c.mention_count,
            c.correlation_to_theme,
            c.first_mentioned_at,
            c.last_mentioned_at,
        ).filter(c.theme_cluster_id == members[0], c.is_active.is_(True))
        ordering = c.mention_count
    else:
        pipeline = db.get(ThemeCluster, theme_id).pipeline
        counts = _post_counts(db, members, pipeline)
        ordering = func.coalesce(func.max(counts.c.count), 0)
        query = (
            db.query(
                c.symbol,
                case(
                    (func.count(func.distinct(c.source)) > 1, "mixed"),
                    else_=func.min(c.source),
                ).label("source"),
                func.coalesce(func.max(c.confidence), 0).label("confidence"),
                ordering.label("mention_count"),
                func.avg(c.correlation_to_theme).label("correlation_to_theme"),
                func.min(c.first_mentioned_at).label("first_mentioned_at"),
                func.max(c.last_mentioned_at).label("last_mentioned_at"),
            )
            .outerjoin(counts, counts.c.symbol == c.symbol)
            .filter(c.theme_cluster_id.in_(members), c.is_active.is_(True))
            .group_by(c.symbol)
        )
    query = query.order_by(ordering.desc(), c.symbol)
    if limit is not None:
        query = query.limit(limit)
    return [GroupConstituent(**row._mapping) for row in query.all()]
