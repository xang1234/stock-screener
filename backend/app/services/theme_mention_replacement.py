"""Replace an item's legacy contributions within the extraction transaction."""

from collections import Counter

from app.models.theme import ThemeAlias, ThemeConstituent, ThemeMention
from app.services.theme_identity_normalization import canonical_theme_key


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
    for mention in previous:
        alias = db.query(ThemeAlias).filter(
            ThemeAlias.pipeline == pipeline,
            ThemeAlias.alias_key == canonical_theme_key(mention.raw_theme),
        ).with_for_update().first()
        if alias is None:
            continue
        count = max(0, alias.evidence_count or 0)
        contribution = (
            0.0 if mention.match_fallback_reason == "alias_match_below_auto_attach_threshold"
            else float(mention.confidence or 0.5)
        )
        remaining_count = max(0, count - 1)
        alias.confidence = (
            max(0.0, min(1.0, (alias.confidence * count - contribution) / remaining_count))
            if remaining_count else 0.0
        )
        alias.evidence_count = remaining_count
        if not remaining_count and alias.source == "llm_extraction":
            alias.is_active = False

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
    refresh_constituent_confidence(db, removed)
    return set(removed)


def refresh_constituent_confidence(db, pairs):
    """Use stable source order, independent of replacement mention IDs."""
    for cluster_id, symbol in pairs:
        constituent = db.query(ThemeConstituent).filter(
            ThemeConstituent.theme_cluster_id == cluster_id,
            ThemeConstituent.symbol == symbol,
        ).with_for_update().first()
        if constituent is None:
            continue
        surviving = db.query(ThemeMention).filter(
            ThemeMention.theme_cluster_id == cluster_id,
        ).order_by(ThemeMention.content_item_id, ThemeMention.id).all()
        values = [float(m.confidence or 0.0) for m in surviving if symbol in (m.tickers or [])]
        if values:
            confidence = values[0]
            for value in values[1:]:
                confidence = confidence * 0.8 + value * 0.2
            constituent.confidence = confidence
        elif constituent.source == "llm_extraction":
            constituent.confidence = 0.0
