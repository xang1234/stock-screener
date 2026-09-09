from datetime import datetime, timezone

import pytest

from app.models.theme import ContentSource
from app.services.content_ingestion_service import ContentIngestionService


@pytest.mark.parametrize("lifecycle", ["pending", "enabled", "disabled", "archived"])
@pytest.mark.parametrize("alias", [False, True])
def test_direct_legacy_fetch_cannot_read_social_owned_list(db_session, lifecycle, alias, monkeypatch):
    from app.infra.db.models.social_signals import SocialSourceConfiguration
    source = ContentSource(name="Social", source_type="twitter", url="https://x.com/i/lists/3001", is_active=lifecycle == "enabled")
    db_session.add(source)
    db_session.flush()
    db_session.add(SocialSourceConfiguration(content_source_id=source.id, x_list_id="3001", lifecycle_state=lifecycle, provenance="admin", archived_at=datetime.now(timezone.utc) if lifecycle == "archived" else None))
    if alias:
        source = ContentSource(name="Alias", source_type="twitter", url="https://twitter.com/i/lists/3001", is_active=True)
        db_session.add(source)
    db_session.commit()
    service = ContentIngestionService(db_session)

    class ForbiddenProvider:
        def fetch(self, *args):
            pytest.fail("legacy provider called for socially owned list")

    service.fetchers["twitter"] = ForbiddenProvider()
    assert service.fetch_source(source) == 0
    assert service.fetch_source_by_id(source.id) == 0
    assert service.fetch_all_active_sources()["total_sources"] == 0
    from app.tasks.theme_discovery_tasks import poll_due_sources
    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.tasks.theme_discovery_tasks._theme_automation_gate_result", lambda db: None)
    monkeypatch.setattr("app.services.content_ingestion_service.TwitterFetcher", ForbiddenProvider)
    assert poll_due_sources()["sources_polled"] == 0
