"""Schema tests for content language / translation columns (T7.1).

Also covers endpoint-mapping regressions for asia.8.4 — the
GET /themes/{id}/mentions and the content-items list endpoint must now
surface `source_language` / `translated_*` / `translation_metadata`
fields on the wire, not just accept them on the schema.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
# Register ORM models used by metadata.create_all.
import app.models.theme  # noqa: F401
import app.models.scan_result  # noqa: F401
import app.infra.db.models.feature_store  # noqa: F401

from app.models.theme import (
    ContentItem,
    ContentSource,
    ThemeCluster,
    ThemeMention,
)
from app.schemas.theme import (
    ContentItemWithThemesResponse,
    ThemeMentionDetailResponse,
    TranslationMetadata,
)


class TestTranslationMetadataShape:
    """Pin the typed replay-snapshot model so T7.2 / T7.3 writers agree."""

    def test_all_fields_optional(self):
        # Empty snapshot must round-trip (detection-only path in T7.2).
        meta = TranslationMetadata()
        dumped = meta.model_dump()
        assert dumped == {
            "provider": None, "model": None, "source_language": None,
            "target_language": None, "confidence": None, "translated_at": None,
        }

    def test_extra_fields_allowed(self):
        # Forward-compat: future T7.x may add keys without a schema bump.
        meta = TranslationMetadata(
            provider="deepl",
            source_language="ja",
            target_language="en",
            extra_field="some future value",
        )
        dumped = meta.model_dump()
        assert dumped["extra_field"] == "some future value"


class TestContentItemTranslationFields:
    """New columns exist on the ORM model and accept the expected types."""

    def test_content_item_accepts_translation_fields(self):
        item = ContentItem(
            source_type="news",
            external_id="abc-1",
            title="日経平均、続伸",
            content="東京株式市場…",
            source_language="ja",
            translated_title="Nikkei average continues to rise",
            translated_content="Tokyo stock market…",
            translation_metadata={
                "provider": "deepl",
                "model": "deepl-free-v1",
                "source_language": "ja",
                "target_language": "en",
                "confidence": 0.98,
                "translated_at": "2026-04-13T00:00:00Z",
            },
        )
        assert item.source_language == "ja"
        assert item.translated_title.startswith("Nikkei")
        assert item.translation_metadata["provider"] == "deepl"
        assert item.translation_metadata["confidence"] == 0.98

    def test_content_item_translation_fields_are_optional(self):
        # Pre-E7 legacy rows (and US rows where source == extraction target)
        # must instantiate without any translation metadata — all four
        # columns are nullable in the migration.
        item = ContentItem(
            source_type="substack",
            external_id="legacy-row",
            title="Why NVDA matters",
            content="English content body",
        )
        assert item.source_language is None
        assert item.translated_title is None
        assert item.translated_content is None
        assert item.translation_metadata is None


class TestThemeMentionTranslationFields:
    def test_mention_accepts_translated_excerpt(self):
        mention = ThemeMention(
            content_item_id=1,
            source_type="news",
            raw_theme="半導体",
            excerpt="半導体大手は増産計画を発表した。",
            translated_raw_theme="semiconductors",
            translated_excerpt="Major semiconductor firms announced expansion plans.",
            translation_metadata={
                "provider": "deepl",
                "source_language": "ja",
                "target_language": "en",
            },
            mentioned_at=datetime.now(tz=timezone.utc),
        )
        assert mention.raw_theme == "半導体"
        assert mention.translated_raw_theme == "semiconductors"
        assert mention.translation_metadata["source_language"] == "ja"

    def test_mention_translation_fields_are_optional(self):
        mention = ThemeMention(
            content_item_id=1,
            source_type="news",
            raw_theme="AI infrastructure",
            excerpt="NVIDIA remains dominant.",
            mentioned_at=datetime.now(tz=timezone.utc),
        )
        assert mention.translated_raw_theme is None
        assert mention.translated_excerpt is None
        assert mention.translation_metadata is None


class TestResponseSchemasExposeTranslation:
    """Pydantic response schemas expose the new columns as optional fields."""

    def test_content_item_response_serializes_translation(self):
        resp = ContentItemWithThemesResponse(
            id=1,
            source_type="news",
            source_language="ja",
            translated_title="Nikkei continues to rise",
            translated_content="Tokyo market…",
            translation_metadata={"provider": "deepl", "target_language": "en"},
        )
        dumped = resp.model_dump()
        assert dumped["source_language"] == "ja"
        assert dumped["translated_title"] == "Nikkei continues to rise"
        assert dumped["translation_metadata"]["provider"] == "deepl"

    def test_content_item_response_defaults_when_untagged(self):
        # Legacy callers that don't set translation fields still serialize
        # successfully (all four default to None).
        resp = ContentItemWithThemesResponse(id=1, source_type="news")
        dumped = resp.model_dump()
        assert dumped["source_language"] is None
        assert dumped["translated_title"] is None
        assert dumped["translated_content"] is None
        assert dumped["translation_metadata"] is None


# ---------------------------------------------------------------------------
# Endpoint-mapping regression tests (asia.8.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


class TestGetThemeMentionsPopulatesTranslationFields:
    """``themes_queries.get_theme_mentions`` must map the language + translation
    columns from ``ThemeMention`` / ``ContentItem`` onto the response schema,
    otherwise the frontend language-indicator UX has nothing to render.
    """

    def test_mention_response_carries_source_language_and_translated_excerpt(
        self, _session
    ):
        from app.api.v1.themes_queries import get_theme_mentions

        source = ContentSource(name="Nikkei", source_type="news")
        _session.add(source)
        _session.flush()

        content = ContentItem(
            source_id=source.id,
            source_type="news",
            external_id="jp-1",
            title="日経平均、続伸",
            content="東京株式市場…",
            source_language="ja",
            translated_title="Nikkei continues to rise",
            translated_content="Tokyo market…",
            translation_metadata={"provider": "deepl", "confidence": 0.94},
        )
        _session.add(content)
        _session.flush()

        cluster = ThemeCluster(
            name="AI infrastructure",
            display_name="AI infrastructure",
            canonical_key="ai_infrastructure",
        )
        _session.add(cluster)
        _session.flush()

        mention = ThemeMention(
            theme_cluster_id=cluster.id,
            content_item_id=content.id,
            source_type="news",
            raw_theme="半導体",
            excerpt="半導体大手は増産計画を発表した。",
            translated_raw_theme="semiconductors",
            translated_excerpt="Major semiconductor firms announced expansion plans.",
            translation_metadata={"provider": "deepl", "confidence": 0.92},
            mentioned_at=datetime.now(tz=timezone.utc),
        )
        _session.add(mention)
        _session.commit()

        resp = get_theme_mentions(theme_id=cluster.id, limit=50, db=_session)

        assert len(resp.mentions) == 1
        m = resp.mentions[0]
        assert m.source_language == "ja"
        assert m.translated_excerpt.startswith("Major semiconductor")
        assert m.translated_raw_theme == "semiconductors"
        assert m.translation_metadata.provider == "deepl"
        assert m.translation_metadata.confidence == 0.92

    def test_mention_falls_back_to_content_translation_metadata(self, _session):
        # Pre-7.3 mentions rely on the ContentItem-level translation snapshot
        # when the mention itself has no per-row metadata.
        from app.api.v1.themes_queries import get_theme_mentions

        source = ContentSource(name="Nikkei", source_type="news")
        _session.add(source)
        _session.flush()

        content = ContentItem(
            source_id=source.id,
            source_type="news",
            external_id="jp-2",
            title="…",
            content="…",
            source_language="ja",
            translation_metadata={"provider": "deepl", "confidence": 0.88},
        )
        _session.add(content)
        _session.flush()

        cluster = ThemeCluster(name="Rates", display_name="Rates", canonical_key="rates")
        _session.add(cluster)
        _session.flush()

        mention = ThemeMention(
            theme_cluster_id=cluster.id,
            content_item_id=content.id,
            source_type="news",
            raw_theme="金利",
            excerpt="日銀が据え置きを発表",
            mentioned_at=datetime.now(tz=timezone.utc),
        )
        _session.add(mention)
        _session.commit()

        resp = get_theme_mentions(theme_id=cluster.id, limit=50, db=_session)

        m = resp.mentions[0]
        # ThemeMention row itself has no language — falls back to ContentItem.
        assert m.source_language == "ja"
        assert m.translation_metadata.provider == "deepl"

    def test_english_source_leaves_translation_fields_null(self, _session):
        from app.api.v1.themes_queries import get_theme_mentions

        source = ContentSource(name="Substack", source_type="substack")
        _session.add(source)
        _session.flush()

        content = ContentItem(
            source_id=source.id,
            source_type="substack",
            external_id="en-1",
            title="Why NVDA matters",
            content="English content body",
        )
        _session.add(content)
        _session.flush()

        cluster = ThemeCluster(name="GPU supply", display_name="GPU supply", canonical_key="gpu_supply")
        _session.add(cluster)
        _session.flush()

        mention = ThemeMention(
            theme_cluster_id=cluster.id,
            content_item_id=content.id,
            source_type="substack",
            raw_theme="AI infra",
            excerpt="NVIDIA dominates accelerator shipments.",
            mentioned_at=datetime.now(tz=timezone.utc),
        )
        _session.add(mention)
        _session.commit()

        m = get_theme_mentions(theme_id=cluster.id, limit=50, db=_session).mentions[0]

        assert m.source_language is None
        assert m.translated_excerpt is None
        assert m.translation_metadata is None

    def test_mention_response_serializes_translation(self):
        resp = ThemeMentionDetailResponse(
            mention_id=42,
            content_title="日経平均、続伸",
            content_url=None,
            author=None,
            published_at=None,
            excerpt="半導体大手は…",
            sentiment="bullish",
            confidence=0.9,
            tickers=["6758.T"],
            source_type="news",
            source_name="Nikkei",
            source_language="ja",
            translated_excerpt="Major semiconductor firms…",
            translated_raw_theme="semiconductors",
            translation_metadata={"provider": "deepl"},
        )
        dumped = resp.model_dump()
        assert dumped["source_language"] == "ja"
        assert dumped["translated_raw_theme"] == "semiconductors"
        assert dumped["translated_excerpt"].startswith("Major")


class TestMigrationContract:
    """Pin the migration revision ID so downstream beads import the right head."""

    def test_migration_head_is_reachable(self):
        # Load the migration via spec (versions files use date-prefixed names
        # that aren't valid Python identifiers, so direct import isn't
        # possible). A syntax or revision typo would fail here before a
        # downstream migration has a chance to take a bad dependency.
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2]
            / "alembic"
            / "versions"
            / "20260413_0010_add_content_language_and_translation_schema.py"
        )
        spec = importlib.util.spec_from_file_location("t7_1_migration", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.revision == "20260413_0010"
        assert module.down_revision == "20260412_0009"
