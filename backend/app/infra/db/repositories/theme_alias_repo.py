"""SQLAlchemy repository for theme alias lookup and quality tracking."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.theme import ThemeAlias
from app.services.theme_identity_normalization import UNKNOWN_THEME_KEY, canonical_theme_key


class SqlThemeAliasRepository:
    """Persist and query theme_aliases for exact matching."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_exact(self, *, pipeline: str, alias_key: str) -> ThemeAlias | None:
        return (
            self._session.query(ThemeAlias)
            .filter(
                ThemeAlias.pipeline == pipeline,
                ThemeAlias.alias_key == alias_key,
                ThemeAlias.is_active.is_(True),
            )
            .first()
        )

    def list_for_cluster(self, *, theme_cluster_id: int, limit: int = 100) -> list[ThemeAlias]:
        return (
            self._session.query(ThemeAlias)
            .filter(
                ThemeAlias.theme_cluster_id == theme_cluster_id,
                ThemeAlias.is_active.is_(True),
            )
            .order_by(ThemeAlias.evidence_count.desc(), ThemeAlias.confidence.desc(), ThemeAlias.last_seen_at.desc())
            .limit(limit)
            .all()
        )

    def record_observation(
        self,
        *,
        theme_cluster_id: int,
        pipeline: str,
        alias_text: str,
        source: str = "llm_extraction",
        confidence: float = 0.5,
        seen_at: datetime | None = None,
    ) -> ThemeAlias:
        alias_key = canonical_theme_key(alias_text)
        if alias_key == UNKNOWN_THEME_KEY:
            raise ValueError("Cannot record alias with unknown_theme key")

        when = seen_at or datetime.utcnow()
        existing = (
            self._session.query(ThemeAlias)
            .filter(
                ThemeAlias.pipeline == pipeline,
                ThemeAlias.alias_key == alias_key,
            )
            .first()
        )

        if existing is None:
            row = ThemeAlias(
                theme_cluster_id=theme_cluster_id,
                pipeline=pipeline,
                alias_text=alias_text,
                alias_key=alias_key,
                source=source,
                confidence=max(0.0, min(1.0, confidence)),
                evidence_count=1,
                first_seen_at=when,
                last_seen_at=when,
                is_active=True,
            )
            self._session.add(row)
            self._session.flush()
            return row

        existing.is_active = True
        existing.alias_text = alias_text or existing.alias_text
        existing.source = source or existing.source
        existing.last_seen_at = when
        if existing.first_seen_at is None:
            existing.first_seen_at = when
        # Smoothed confidence update weighted by historical observations.
        old_count = max(1, existing.evidence_count or 1)
        bounded = max(0.0, min(1.0, confidence))
        existing.confidence = ((existing.confidence * old_count) + bounded) / (old_count + 1)
        existing.evidence_count = old_count + 1
        self._session.flush()
        return existing

    def deactivate(self, *, pipeline: str, alias_key: str) -> bool:
        row = (
            self._session.query(ThemeAlias)
            .filter(
                ThemeAlias.pipeline == pipeline,
                ThemeAlias.alias_key == alias_key,
                ThemeAlias.is_active.is_(True),
            )
            .first()
        )
        if row is None:
            return False
        row.is_active = False
        self._session.flush()
        return True
