"""Replace an item's legacy contributions within the extraction transaction."""

from collections import Counter

from app.models.theme import ThemeConstituent, ThemeMention


def remove_previous_legacy_mentions(db, item_id, pipeline):
    previous = (
        db.query(ThemeMention)
        .filter(
            ThemeMention.content_item_id == item_id,
            ThemeMention.pipeline == pipeline,
            ThemeMention.social_work_id.is_(None),
        )
        .all()
    )
    removed = Counter(
        (m.theme_cluster_id, symbol)
        for m in previous
        for symbol in set(m.tickers or [])
    )
    for (cluster_id, symbol), count in removed.items():
        constituent = (
            db.query(ThemeConstituent)
            .filter(
                ThemeConstituent.theme_cluster_id == cluster_id,
                ThemeConstituent.symbol == symbol,
            )
            .with_for_update()
            .first()
        )
        if constituent:
            constituent.mention_count = max(0, (constituent.mention_count or 0) - count)
            if not constituent.mention_count and constituent.source == "llm_extraction":
                constituent.is_active = False
    for mention in previous:
        db.delete(mention)
    db.flush()
