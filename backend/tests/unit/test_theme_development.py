"""Theme identity stays stable while source-specific developments survive storage."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.api.v1.themes_queries import get_theme_mentions
from app.models.theme import ContentItem, ThemeCluster, ThemeMention
from app.services.theme_extraction_service import (
    ThemeExtractionParseError,
    ThemeExtractionService,
)


@pytest.fixture
def extractor(universe_session, monkeypatch):
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.db = universe_session
    service.pipeline = "fundamental"
    service.claim_review_enabled = False
    service.provider = "litellm"
    service.pipeline_config = None
    service.max_age_days = 30
    service.theme_policy_overrides = {}
    service._valid_tickers = set()
    monkeypatch.setattr(service, "_rate_limit", lambda: None)
    return service


def mention(**overrides):
    return {
        "theme": "HBM",
        "development": "Samsung reportedly may supply HBM4; no agreement is confirmed.",
        "tickers": [],
        "sentiment": "neutral",
        "confidence": 0.8,
        "excerpt": "Samsung may supply HBM4. Talks are ongoing.",
        **overrides,
    }


def item():
    return ContentItem(
        source_type="news",
        source_name="Example",
        title="Supplier talks",
        content="Samsung may supply HBM4. Talks are ongoing.",
        published_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )


def test_developments_survive_extraction_storage_and_api_without_splitting_theme(
    extractor,
    universe_session,
    monkeypatch,
):
    parent = ThemeCluster(
        name="Memory",
        canonical_key="memory",
        display_name="Memory",
        pipeline="fundamental",
        is_l1=True,
    )
    universe_session.add(parent)
    universe_session.flush()
    cluster = ThemeCluster(
        name="HBM",
        canonical_key="hbm",
        display_name="HBM",
        pipeline="fundamental",
        parent_cluster_id=parent.id,
    )
    universe_session.add(cluster)
    universe_session.flush()
    for development in [
        mention()["development"],
        "Suppliers reported rising HBM demand.",
    ]:
        source = item()
        universe_session.add(source)
        universe_session.flush()
        monkeypatch.setattr(
            extractor,
            "_try_generate_litellm",
            lambda prompt, development=development: json.dumps(
                [mention(development=development)]
            ),
        )
        assert extractor._extract_and_store_mentions(source) == 1
    universe_session.commit()
    universe_session.expire_all()

    stored = universe_session.query(ThemeMention).order_by(ThemeMention.id).all()
    assert [m.theme_cluster_id for m in stored] == [cluster.id, cluster.id]
    assert [m.development for m in stored] == [
        "Samsung reportedly may supply HBM4; no agreement is confirmed.",
        "Suppliers reported rising HBM demand.",
    ]
    response = get_theme_mentions(cluster.id, limit=50, db=universe_session)
    assert {m.development for m in response.mentions} == {m.development for m in stored}
    assert response.theme_name == "HBM"
    assert all(m.excerpt == mention()["excerpt"] for m in response.mentions)


def test_grounding_context_survives_extraction_storage_and_mentions_api(
    extractor,
    universe_session,
    monkeypatch,
):
    cluster = ThemeCluster(
        name="HBM", canonical_key="hbm", display_name="HBM", pipeline="fundamental"
    )
    universe_session.add(cluster)
    universe_session.flush()
    source = item()
    universe_session.add(source)
    universe_session.flush()
    monkeypatch.setattr(
        extractor, "_try_generate_litellm", lambda _prompt: json.dumps([mention()])
    )

    assert extractor._extract_and_store_mentions(source) == 1
    universe_session.commit()
    stored = universe_session.query(ThemeMention).one()

    assert stored.grounding_context["policy_version"] == "grounding-v1"
    response = get_theme_mentions(cluster.id, limit=50, db=universe_session)
    assert response.mentions[0].grounding_context == stored.grounding_context


@pytest.mark.parametrize("value", [None, "", "  "])
def test_empty_development_remains_absent_without_copying_excerpt(
    extractor, monkeypatch, value
):
    monkeypatch.setattr(
        extractor,
        "_try_generate_litellm",
        lambda prompt: json.dumps([mention(development=value)]),
    )
    result = extractor.extract_from_content(item(), verify_claims=False)[0]
    assert result["development"] is None
    assert result["excerpt"] == mention()["excerpt"]


def test_legacy_response_does_not_invent_a_development(extractor, monkeypatch):
    legacy = mention()
    del legacy["development"]
    monkeypatch.setattr(
        extractor, "_try_generate_litellm", lambda prompt: json.dumps([legacy])
    )
    assert (
        extractor.extract_from_content(item(), verify_claims=False)[0]["development"]
        is None
    )


@pytest.mark.parametrize("value", [42, ["demand"], {"event": "demand"}, "x" * 1001])
def test_invalid_development_fails_instead_of_dropping_or_truncating_it(
    extractor, monkeypatch, value
):
    monkeypatch.setattr(
        extractor,
        "_try_generate_litellm",
        lambda prompt: json.dumps([mention(development=value)]),
    )
    with pytest.raises(ThemeExtractionParseError, match="development"):
        extractor.extract_from_content(item(), verify_claims=False)


def test_development_migration_preserves_legacy_mentions():
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic/versions/20260910_0038_add_theme_mention_development.py"
    )
    spec = importlib.util.spec_from_file_location("development_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE theme_mentions (id INTEGER PRIMARY KEY, raw_theme TEXT, excerpt TEXT)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO theme_mentions VALUES (1, 'HBM Memory Demand', 'Demand rose')"
            )
        )
        module.op = Operations(MigrationContext.configure(connection))
        module.upgrade()
        row = connection.execute(
            sa.text("SELECT raw_theme, excerpt, development FROM theme_mentions")
        ).one()
        assert tuple(row) == ("HBM Memory Demand", "Demand rose", None)
        connection.execute(
            sa.text("UPDATE theme_mentions SET development = 'Demand rose'")
        )
        module.downgrade()
        assert connection.execute(
            sa.text("SELECT raw_theme, excerpt FROM theme_mentions")
        ).one() == ("HBM Memory Demand", "Demand rose")
        assert "development" not in {
            c["name"] for c in sa.inspect(connection).get_columns("theme_mentions")
        }


def test_grounding_context_migration_preserves_legacy_mentions():
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic/versions/20260911_0039_add_theme_mention_grounding_context.py"
    )
    spec = importlib.util.spec_from_file_location("grounding_context_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE theme_mentions (id INTEGER PRIMARY KEY, raw_theme TEXT)"
            )
        )
        connection.execute(
            sa.text("INSERT INTO theme_mentions VALUES (1, 'HBM Memory Demand')")
        )
        module.op = Operations(MigrationContext.configure(connection))
        module.upgrade()
        row = connection.execute(
            sa.text("SELECT raw_theme, grounding_context FROM theme_mentions")
        ).one()
        assert tuple(row) == ("HBM Memory Demand", None)
        connection.execute(
            sa.text("UPDATE theme_mentions SET grounding_context = '{}'")
        )
        module.downgrade()
        assert connection.execute(
            sa.text("SELECT raw_theme FROM theme_mentions")
        ).one() == ("HBM Memory Demand",)
        assert "grounding_context" not in {
            c["name"] for c in sa.inspect(connection).get_columns("theme_mentions")
        }


def test_review_support_survives_mentions_storage_and_api(
    extractor, universe_session, monkeypatch
):
    extractor.claim_review_enabled = True
    cluster = ThemeCluster(
        name="HBM", canonical_key="hbm", display_name="HBM", pipeline="fundamental"
    )
    universe_session.add(cluster)
    universe_session.flush()
    source = item()
    universe_session.add(source)
    universe_session.flush()
    evidence = [{"source_id": "primary", "quote": source.content}]
    review = [
        {
            "index": 0,
            "theme": {
                "status": "inferred",
                "reason": "HBM4 identifies the HBM exposure.",
                "evidence": evidence,
            },
            "development": {
                "status": "supported",
                "reason": "The claim preserves the uncertainty.",
                "evidence": evidence,
            },
        }
    ]
    monkeypatch.setattr(
        extractor,
        "_try_generate_litellm",
        lambda prompt, **kwargs: json.dumps(review if kwargs else [mention()]),
    )
    assert extractor._extract_and_store_mentions(source) == 1
    universe_session.commit()
    response = get_theme_mentions(cluster.id, limit=50, db=universe_session)
    assert response.mentions[0].claim_support == {
        "theme": "inferred",
        "development": "inferred",
    }
    assert response.mentions[0].development.startswith("Inference:")
