from __future__ import annotations

import httpx
import pytest

from app.database import get_db
from app.main import app


async def _request(db, method, path, **kwargs):
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def social_sources(db_session):
    from app.services.social_source_admin_service import SocialSourceAdminService

    service = SocialSourceAdminService(db_session)
    rows = service.ensure_seed_sources()
    return service, rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"name": "changed"},
        {"url": "https://example.com"},
        {"source_type": "news"},
        {"pipelines": ["fundamental"]},
        {"is_active": False},
        {"is_active": True},
    ],
)
async def test_legacy_update_cannot_mutate_social_owned_source(
    db_session, social_sources, monkeypatch, payload
):
    from app.services import server_auth
    from app.models.theme import ContentSource

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _, rows = social_sources
    source_id = int(rows[0].source_id)
    before = db_session.get(ContentSource, source_id)
    snapshot = (before.name, before.url, before.source_type, before.is_active,
                tuple(before.pipelines))

    response = await _request(
        db_session, "PUT", f"/api/v1/themes/sources/{source_id}", json=payload
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "social_source_managed_elsewhere"
    db_session.expire_all()
    after = db_session.get(ContentSource, source_id)
    assert (after.name, after.url, after.source_type, after.is_active,
            tuple(after.pipelines)) == snapshot


@pytest.mark.asyncio
async def test_legacy_delete_and_equivalent_create_are_rejected(
    db_session, social_sources, monkeypatch
):
    from app.services import server_auth
    from app.models.theme import ContentSource

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _, rows = social_sources
    source_id = int(rows[0].source_id)
    list_id = rows[0].list_id

    deleted = await _request(
        db_session, "DELETE", f"/api/v1/themes/sources/{source_id}"
    )
    duplicate = await _request(
        db_session, "POST", "/api/v1/themes/sources", json={
            "name": "alternate", "source_type": "twitter",
            "url": f"https://twitter.com/i/lists/{list_id}/",
        }
    )

    assert deleted.status_code == duplicate.status_code == 409
    assert db_session.get(ContentSource, source_id).is_active is True


@pytest.mark.asyncio
async def test_ordinary_theme_source_operations_remain_available(
    db_session, social_sources, monkeypatch
):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    created = await _request(
        db_session, "POST", "/api/v1/themes/sources", json={
            "name": "Ordinary", "source_type": "news",
            "url": "https://example.com/feed",
        }
    )
    assert created.status_code == 200
    source_id = created.json()["id"]
    updated = await _request(
        db_session, "PUT", f"/api/v1/themes/sources/{source_id}",
        json={"name": "Ordinary renamed", "is_active": False},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Ordinary renamed"
    deleted = await _request(
        db_session, "DELETE", f"/api/v1/themes/sources/{source_id}"
    )
    assert deleted.status_code == 200


@pytest.mark.asyncio
async def test_live_theme_detail_unions_legacy_and_accepted_social_membership(
    db_session, social_sources, monkeypatch
):
    from datetime import datetime, timezone
    from app.infra.db.models.social_analysis import SocialThemeAssociation
    from app.models.stock_universe import StockUniverse
    from app.models.theme import ThemeCluster, ThemeConstituent
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    service, _ = social_sources
    runtime = service.read_runtime()
    service.apply_runtime("live", "official", runtime.version, "admin")
    theme = ThemeCluster(
        name="AI Infrastructure", display_name="AI Infrastructure",
        canonical_key="ai_infrastructure", pipeline="technical", aliases=[],
        lifecycle_state="active", is_active=True,
    )
    db_session.add_all([
        theme,
        StockUniverse(symbol="AAA", market="US", exchange="NASDAQ", is_active=True),
        StockUniverse(symbol="BBB", market="US", exchange="NASDAQ", is_active=True),
    ])
    db_session.flush()
    db_session.add(ThemeConstituent(
        theme_cluster_id=theme.id, symbol="AAA", source="manual",
        confidence=0.8, mention_count=2, is_active=True,
    ))
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    db_session.add(SocialThemeAssociation(
        theme_cluster_id=theme.id, company_key="US:BBB", market="US",
        canonical_symbol="BBB", state="accepted", origin="social",
        decision_owner="system", evidence_work_ids=[], policy_version="policy-v1",
        version=1, first_seen_at=now, accepted_at=now, updated_at=now,
    ))
    db_session.commit()

    response = await _request(db_session, "GET", f"/api/v1/themes/{theme.id}")

    assert response.status_code == 200
    rows = {row["symbol"]: row for row in response.json()["constituents"]}
    assert set(rows) == {"AAA", "BBB"}
    assert rows["AAA"]["origins"] == ["legacy"]
    assert rows["BBB"]["origins"] == ["social"]
