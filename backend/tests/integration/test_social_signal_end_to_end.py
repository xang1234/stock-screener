"""Fixture-only Social Signal acceptance flow; no X or model network access."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from app.database import SessionLocal, get_db
from app.domain.social_signals.records import (
    SocialPostRecord,
    SocialReadRequest,
    SocialSnapshotRecord,
    SocialSourceBatch,
    SocialSourceOutcome,
    SourceTestOutcome,
)
from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceConfiguration
from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
from app.infra.providers.official_x_social_provider import OfficialXSocialProvider
from app.infra.providers.xui_cli_social_provider import XuiCliSocialProvider
from app.main import app
from app.models.stock_universe import StockUniverse
from app.services.social_source_admin_service import SocialSourceAdminService
from tests.integration.test_social_theme_projection import social_fixture


NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parents[1] / "fixtures" / "social"


async def request(db, method, path, **kwargs):
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as client:
            return await client.request(method, path, **kwargs)
    finally:
        app.dependency_overrides.pop(get_db, None)


def read_request(source_id, list_id):
    return SocialReadRequest(
        f"fixture-{source_id}", str(source_id), list_id, "initial", NOW, 5,
        NOW - timedelta(days=14),
    )


def completed_empty_batch(source):
    outcome = SocialSourceOutcome(
        "success", "complete", "warming_up", ("bounded_provider_read",), (),
        None, None, 0, None, None,
    )
    return SocialSourceBatch(
        read_request(source.content_source_id, source.x_list_id), (), outcome
    )


def snapshot(
    run_id,
    symbol,
    *,
    social,
    confirmation,
    queue,
    state="actionable",
    market="US",
    candidate_key=None,
):
    key = candidate_key or f"{market}:{symbol}"
    return SocialSnapshotRecord(
        run_id, key, symbol, market, state,
        Decimal(str(social)) if social is not None else None,
        Decimal(str(confirmation)) if confirmation is not None else None,
        Decimal(str(queue)) if queue is not None else None,
        (("state_reasons", "fixture_verified"),), (), NOW,
        symbol, key, 7, 3, 2, 3, "market" if market else "global",
    )


@pytest.mark.asyncio
async def test_fixture_providers_source_lifecycle_atomic_publish_and_rank_modes(
    db_session, monkeypatch
):
    """The public seams form one usable flow without private code or live reads."""
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    admin = SocialSourceAdminService(db_session)
    seeds = admin.ensure_seed_sources()
    runtime = admin.read_runtime()
    admin.apply_runtime("live", "official", runtime.version, "fixture-admin")

    provider_reads = []
    created = admin.create_source("Japan Growth", "3000000000000000000", "fixture-admin")
    assert provider_reads == []
    queued = admin.request_test(created.source_id, created.version, "fixture-admin")
    claimed = admin.claim_test(created.source_id, "fixture-worker")
    assert claimed.request_id == queued.request_id
    tested = admin.record_test_result(
        created.source_id, "official", SourceTestOutcome("official", "passed", 5, NOW),
        "fixture-worker", request_id=claimed.request_id,
        expected_version=claimed.version,
    )
    enabled = admin.transition_source(
        created.source_id, "enabled", tested.version, "fixture-admin"
    )
    assert enabled.name == "Japan Growth" and enabled.lifecycle == "enabled"

    official_payload = json.loads((FIXTURES / "official_list_read.json").read_text())
    xui_payload = json.loads((FIXTURES / "xui_list_read.json").read_text())
    official_posts, _ = OfficialXSocialProvider._normalize(
        official_payload, read_request(seeds[0].source_id, seeds[0].list_id)
    )
    xui_posts = XuiCliSocialProvider._normalize(
        xui_payload, read_request(seeds[1].source_id, seeds[0].list_id), 5
    )
    assert all(isinstance(post, SocialPostRecord) for post in official_posts + xui_posts)
    assert official_posts[0].provider_post_id == xui_posts[0].provider_post_id == "syn-101"
    assert {post.provider for post in official_posts + xui_posts} == {"official", "xui"}

    for symbol, market in (
        ("NVDA", "US"), ("0700", "HK"), ("688981", "CN"),
        ("6758", "JP"), ("2330", "TW"),
    ):
        db_session.add(StockUniverse(symbol=symbol, market=market, is_active=True))
    db_session.commit()
    assert {row.market for row in db_session.query(StockUniverse)} == {
        "US", "HK", "CN", "JP", "TW"
    }

    writer = SocialSignalWriter(SessionLocal, clock=lambda: NOW)
    run_id = "fixture-e2e"
    writer.create_run(run_id, NOW)
    with SessionLocal() as db:
        pinned = db.query(SocialSourceConfiguration).filter_by(
            lifecycle_state="enabled"
        ).order_by(SocialSourceConfiguration.content_source_id).all()
    assert len(pinned) == 3
    for source in pinned:
        writer.persist_observations(completed_empty_batch(source), run_id=run_id)

    rows = (
        snapshot(run_id, "NVDA", social=96, confirmation=30, queue=60),
        snapshot(run_id, "AMD", social=75, confirmation=95, queue=85),
        snapshot(
            run_id,
            "$MYSTERY",
            social=None,
            confirmation=None,
            queue=None,
            state="unresolved",
            market=None,
            candidate_key="unresolved:fixture-mystery",
        ),
    )
    writer.prepare_run(run_id, rows, NOW)
    with SessionLocal() as db:
        version = db.get(SocialSignalRun, run_id).registry_version
    assert writer.publish(run_id, version).published is True

    db_session.expire_all()
    blended = await request(
        db_session, "GET",
        "/api/v1/social-signals/queue?market=US&window=7d&view=all&rank_mode=blended",
    )
    pure = await request(
        db_session, "GET",
        "/api/v1/social-signals/queue?market=US&window=7d&view=all&rank_mode=pure_social",
    )
    unresolved = await request(
        db_session, "GET",
        "/api/v1/social-signals/unresolved?scope=unknown&market=US&window=7d&page=1&page_size=50",
    )
    assert [row["canonical_symbol"] for row in blended.json()["items"]] == [
        "AMD", "NVDA", "$MYSTERY",
    ]
    assert [row["canonical_symbol"] for row in pure.json()["items"]] == [
        "NVDA", "AMD", "$MYSTERY",
    ]
    assert pure.json()["items"][0]["state"] == "actionable"
    assert pure.json()["items"][0]["explanation"]["state_reasons"] == ["fixture_verified"]
    assert pure.json()["items"][0]["observed_list_count"] == 2
    assert [row["canonical_symbol"] for row in unresolved.json()["items"]] == ["$MYSTERY"]
    assert unresolved.json()["items"][0]["market"] is None
    encoded = json.dumps({
        "blended": blended.json(), "pure": pure.json(),
        "unresolved": unresolved.json(),
    }).lower()
    assert all(term not in encoded for term in (
        "cookie", "storage_state", "bearer", "deploy_key", "config_path"
    ))

    db_session.rollback()
    audit = [event.action for event in admin.audit_events(created.source_id)]
    assert audit == ["created", "test_requested", "test_completed", "enabled"]


def test_warming_history_validation_isolation_and_empty_catalog_discovery(social_fixture):
    """A fresh Theme catalog can discover safely; validation never mutates it."""
    from app.models.theme import ThemeCluster

    fixture = social_fixture
    work = fixture.save(("AAA", "BBB", "CCC"), age=3)
    validation = fixture.prepare([work], "validation")
    assert len(validation.proposals) == 3
    assert fixture.db.query(ThemeCluster).count() == 0

    first = fixture.prepare([work], "live")
    fixture.apply(first)
    assert fixture.db.query(ThemeCluster).one().lifecycle_state == "candidate"
    fixture.apply(fixture.prepare([
        fixture.save(("AAA", "BBB", "CCC"), age=2)
    ], "live"))
    fixture.apply(fixture.prepare([fixture.save(("AAA",), age=1)], "live"))
    assert fixture.db.query(ThemeCluster).one().lifecycle_state == "active"


def test_two_transaction_admin_contention_and_minimum_enabled_sources(tmp_path):
    """Contention is safe and a saved budget backlog becomes eligible at midnight."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base
    from app.services.social_source_admin_service import SocialSourceStateError

    engine = create_engine(f"sqlite:///{tmp_path / 'admin-contention.sqlite'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as first:
        service = SocialSourceAdminService(first)
        seeds = service.ensure_seed_sources()
        runtime = service.read_runtime()
        service.apply_runtime("live", "official", runtime.version, "admin")
    with factory() as first, factory() as second:
        first_service = SocialSourceAdminService(first)
        second_service = SocialSourceAdminService(second)
        observed = second_service.list_sources()[0]
        renamed = first_service.rename_source(
            observed.source_id, "Readable Seed", observed.version, "first"
        )
        with pytest.raises(SocialSourceStateError, match="version_conflict"):
            second_service.rename_source(
                observed.source_id, "Stale Rename", observed.version, "second"
            )
        with pytest.raises(SocialSourceStateError, match="minimum_two_enabled"):
            first_service.transition_source(
                renamed.source_id, "disabled", renamed.version, "first"
            )
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    before_midnight = datetime(2026, 9, 7, 15, 59, tzinfo=timezone.utc)
    budget = SocialLLMBudgetService(factory)
    assert budget.reserve("fixture-full-day", (1,), Decimal("2"), before_midnight)
    assert budget.status(before_midnight).remaining_usd == 0
    assert budget.status(before_midnight + timedelta(minutes=1)).remaining_usd == 2
    engine.dispose()
