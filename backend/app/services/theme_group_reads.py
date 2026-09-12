"""Current group views built from original source assignments."""

from types import SimpleNamespace

from app.models.theme import ThemeCluster, ThemeConstituent, ThemeMention
from app.services.theme_equivalence_service import ThemeEquivalenceService
from app.services.theme_evidence_eligibility_service import legacy_eligibility_exists


def grouped_constituents(db, theme_id):
    members = ThemeEquivalenceService(db).members(theme_id)
    rows = (
        db.query(ThemeConstituent)
        .filter(
            ThemeConstituent.theme_cluster_id.in_(members),
            ThemeConstituent.is_active.is_(True),
        )
        .order_by(ThemeConstituent.mention_count.desc())
        .all()
    )
    if len(members) == 1:
        return rows
    pipeline = db.get(ThemeCluster, theme_id).pipeline
    mentions = (
        db.query(ThemeMention)
        .filter(
            ThemeMention.theme_cluster_id.in_(members),
            ThemeMention.social_work_id.is_(None),
            legacy_eligibility_exists(
                ThemeMention.content_item_id, pipeline, active_only=True
            ),
        )
        .all()
    )
    parents = {}
    for mention in mentions:
        for symbol in set(mention.tickers or []):
            parents.setdefault(symbol, set()).add(mention.content_item_id)
    result = {}
    for row in rows:
        if row.symbol not in result:
            result[row.symbol] = SimpleNamespace(
                **{
                    column.name: getattr(row, column.name)
                    for column in ThemeConstituent.__table__.columns
                }
            )
        value = result[row.symbol]
        value.mention_count = len(parents.get(row.symbol, set()))
        for field, choose in [("first_mentioned_at", min), ("last_mentioned_at", max)]:
            times = [
                v for v in (getattr(value, field), getattr(row, field)) if v is not None
            ]
            setattr(value, field, choose(times) if times else None)
        value.confidence = max(value.confidence or 0, row.confidence or 0)
    return sorted(result.values(), key=lambda row: (-row.mention_count, row.symbol))
