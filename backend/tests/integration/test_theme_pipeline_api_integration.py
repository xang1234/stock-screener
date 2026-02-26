"""Integration tests for pipeline-scoped theme extraction/reprocess and API reads."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, get_db
from app.db_migrations.theme_pipeline_state_migration import migrate_theme_pipeline_state
from app.main import app
from app.models.theme import ContentItem, ContentItemPipelineState, ContentSource, ThemeMention
from app.services.theme_extraction_service import ThemeExtractionService


def _seed_source_and_item(
    session: Session,
    *,
    external_id: str,
    pipelines: list[str] | str,
) -> tuple[ContentSource, ContentItem]:
    source = ContentSource(
        name=f"Source-{external_id}",
        source_type="news",
        url=f"https://example.com/{external_id}",
        is_active=True,
        pipelines=pipelines,
    )
    session.add(source)
    session.flush()
    item = ContentItem(
        source_id=source.id,
        source_type=source.source_type,
        source_name=source.name,
        external_id=external_id,
        title=f"Title {external_id}",
        content=f"Body {external_id}",
        published_at=datetime.utcnow() - timedelta(days=1),
    )
    session.add(item)
    session.flush()
    return source, item


@pytest.mark.asyncio
async def test_extract_endpoint_contract_supports_both_pipelines(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    def _override_get_db():
        try:
            yield session
        finally:
            pass

    def _fake_process_batch(self, limit: int = 50, item_ids=None):
        _ = item_ids
        return {
            "processed": min(limit, 1),
            "total_mentions": 1,
            "errors": 0,
            "new_themes": [f"{self.pipeline}-theme"],
        }

    monkeypatch.setattr(ThemeExtractionService, "process_batch", _fake_process_batch)
    app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for pipeline in ("technical", "fundamental"):
                response = await client.post(f"/api/v1/themes/extract?pipeline={pipeline}&limit=10")
                assert response.status_code == 200
                payload = response.json()
                assert payload["processed"] == 1
                assert payload["total_mentions"] == 1
                assert payload["errors"] == 0
                assert payload["new_themes"] == [f"{pipeline}-theme"]
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()


@pytest.mark.asyncio
async def test_reprocess_is_pipeline_scoped_and_theme_api_reads_reflect_both_pipelines(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    _, technical_item = _seed_source_and_item(session, external_id="tech-item", pipelines=["technical", "fundamental"])
    _, fundamental_item = _seed_source_and_item(
        session,
        external_id="fund-item",
        pipelines=["technical", "fundamental"],
    )
    session.add_all(
        [
            ContentItemPipelineState(
                content_item_id=technical_item.id,
                pipeline="technical",
                status="failed_retryable",
                attempt_count=1,
            ),
            ContentItemPipelineState(
                content_item_id=fundamental_item.id,
                pipeline="fundamental",
                status="failed_retryable",
                attempt_count=1,
            ),
        ]
    )
    session.commit()

    monkeypatch.setattr(ThemeExtractionService, "_init_client", lambda self: None)
    monkeypatch.setattr(ThemeExtractionService, "_load_configured_model", lambda self: None)

    def _fake_extract(self, content_item: ContentItem):
        return [
            {
                "theme": f"{self.pipeline.title()} Momentum",
                "tickers": [],
                "sentiment": "neutral",
                "confidence": 0.9,
                "excerpt": content_item.title or "",
            }
        ]

    monkeypatch.setattr(ThemeExtractionService, "extract_from_content", _fake_extract)

    technical_service = ThemeExtractionService(session, pipeline="technical")
    technical_result = technical_service.reprocess_failed_items(limit=10)
    assert technical_result["reprocessed_count"] == 1
    assert technical_result["processed"] == 1

    fundamental_service = ThemeExtractionService(session, pipeline="fundamental")
    fundamental_result = fundamental_service.reprocess_failed_items(limit=10)
    assert fundamental_result["reprocessed_count"] == 1
    assert fundamental_result["processed"] == 1

    mentions = session.query(ThemeMention).all()
    assert {m.pipeline for m in mentions} == {"technical", "fundamental"}
    assert session.query(ContentItemPipelineState).filter(
        ContentItemPipelineState.status == "failed_retryable"
    ).count() == 0

    def _override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            tech_failed = await client.get("/api/v1/themes/pipeline/failed-count?pipeline=technical")
            fund_failed = await client.get("/api/v1/themes/pipeline/failed-count?pipeline=fundamental")
            assert tech_failed.status_code == 200
            assert fund_failed.status_code == 200
            assert tech_failed.json()["failed_count"] == 0
            assert fund_failed.json()["failed_count"] == 0

            tech_health = await client.get("/api/v1/themes/pipeline/state-health?pipeline=technical&window_days=30")
            fund_health = await client.get("/api/v1/themes/pipeline/state-health?pipeline=fundamental&window_days=30")
            assert tech_health.status_code == 200
            assert fund_health.status_code == 200

            tech_counts = tech_health.json()["pipelines"][0]["counts"]
            fund_counts = fund_health.json()["pipelines"][0]["counts"]
            assert tech_counts["processed"] >= 1
            assert fund_counts["processed"] >= 1
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()


@pytest.mark.asyncio
async def test_migration_upgraded_pipeline_state_db_is_served_by_theme_api(tmp_path: Path):
    db_path = tmp_path / "theme-migration-integration.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        source, item = _seed_source_and_item(
            session,
            external_id="legacy-migrated-item",
            pipelines='["technical","fundamental"]',
        )
        session.commit()
        source_id = source.id
        item_id = item.id

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS content_item_pipeline_state"))
        conn.execute(
            text(
                """
                CREATE TABLE content_item_pipeline_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_item_id INTEGER NOT NULL,
                    pipeline TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at DATETIME,
                    processed_at DATETIME,
                    created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO content_item_pipeline_state
                    (content_item_id, pipeline, status, attempt_count, created_at, updated_at)
                VALUES
                    (:content_item_id, 'technical', 'failed_retryable', 2, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            ),
            {"content_item_id": item_id},
        )

    stats = migrate_theme_pipeline_state(engine)
    assert stats["table_created"] is False

    session = SessionLocal()

    def _override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            failed_count = await client.get("/api/v1/themes/pipeline/failed-count?pipeline=technical")
            assert failed_count.status_code == 200
            assert failed_count.json()["failed_count"] == 1

            health = await client.get("/api/v1/themes/pipeline/state-health?pipeline=technical&window_days=30")
            assert health.status_code == 200
            counts = health.json()["pipelines"][0]["counts"]
            assert counts["failed_retryable"] == 1

        row = session.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item_id,
            ContentItemPipelineState.pipeline == "technical",
        ).one()
        assert row.error_code is None
        assert row.error_message is None
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()

    # Keep lints honest about source fixture usage in migration scenario.
    assert source_id > 0
