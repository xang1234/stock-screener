from __future__ import annotations

import httpx
import pytest

from app.database import get_db
from app.main import app


@pytest.fixture
def social_runtime(db_session):
    from app.services.social_source_admin_service import SocialSourceAdminService

    admin = SocialSourceAdminService(db_session)
    admin.ensure_seed_sources()
    return admin


async def _request(db_session, method, path, **kwargs):
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _publish_rows(
    db, records, *, observations=None, context=None,
    theme_evidence=None, projection=None, generated_at=None, published_at=None,
):
    from app.infra.db.models.social_signals import (
        SocialSignalRun, SocialSignalRunPointer, SocialSignalSnapshot,
    )
    from app.infra.db.repositories.social_signal_writer import serialized

    run = SocialSignalRun(
        id="published-1", registry_id=1, registry_version=2,
        mode="live", provider="official", status="running",
        source_outcomes_json={
            "1": {"read_status": "success", "history_status": "limited"},
            "2": {"read_status": "success", "history_status": "limited"},
        },
        application_progress_json={
            "sources": {
                "1": {"list_id": "111", "name": "One", "version": 1},
                "2": {"list_id": "222", "name": "Two", "version": 1},
            },
            "observations": observations or {"1": {}, "2": {}},
            "prepared": {
                "context": context or {"formula_version": "social-signal-v1"},
                "theme_evidence": theme_evidence or [],
                "projection": projection or {},
            },
        },
        feature_run_ids_json={}, exposure_dates_json={}, coverage_json={},
        created_at=generated_at or records[0].latest_mention,
        completed_at=published_at or records[0].latest_mention,
        published_at=published_at or records[0].latest_mention,
    )
    db.add(run)
    db.flush()
    for record in records:
        db.add(SocialSignalSnapshot(
            run_id=run.id, window_days=record.window_days,
            candidate_key=record.candidate_key,
            canonical_symbol=record.canonical_symbol or None,
            market=record.market, state=record.candidate_state,
            social_score=record.social_score,
            confirmation_score=record.confirmation_score,
            queue_score=record.queue_score,
            explanation_json={"record": serialized(record), "evidence_run_id": run.id},
            coverage_json={"evidence_run_id": run.id, "reasons": list(record.coverage)},
            resolution_policy_version="social-resolution-v1",
            formula_version=record.formula_version,
            latest_mention=record.latest_mention,
            mention_count=record.mention_count,
            observed_list_count=record.observed_list_count,
            enabled_list_count=record.enabled_list_count,
            normalization_scope=record.normalization_scope,
        ))
    db.commit()
    run = db.get(SocialSignalRun, "published-1")
    assert run.status == "running"
    run.status = "published"
    db.add(SocialSignalRunPointer(key="latest_published", run_id=run.id))
    db.commit()


@pytest.mark.asyncio
async def test_social_read_routes_require_server_session(db_session, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "server-secret")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "signing-secret")

    response = await _request(
        db_session,
        "GET",
        "/api/v1/social-signals/queue?market=US&window=7d",
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_off_runtime_returns_typed_unsupported_summary(
    db_session, social_runtime, monkeypatch
):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    response = await _request(
        db_session, "GET", "/api/v1/social-signals/summary?market=US"
    )

    assert response.status_code == 200
    assert {key: response.json()[key] for key in (
        "supported", "available", "reason_code", "market"
    )} == {
        "supported": False,
        "available": False,
        "reason_code": "social_signals_disabled",
        "market": "US",
    }


@pytest.mark.asyncio
async def test_live_without_publication_is_typed_unavailable(
    db_session, social_runtime, monkeypatch
):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    runtime = social_runtime.read_runtime()
    social_runtime.apply_runtime("live", "official", runtime.version, "admin")

    response = await _request(
        db_session,
        "GET",
        "/api/v1/social-signals/queue?market=US&window=7d&view=actionable&rank_mode=blended&page=1&page_size=50",
    )

    assert response.status_code == 200
    payload = response.json()
    assert {key: payload[key] for key in (
        "supported", "available", "reason_code", "market", "window", "view",
        "rank_mode", "page", "page_size", "total", "items",
    )} == {
        "supported": True,
        "available": False,
        "reason_code": "no_published_run",
        "market": "US",
        "window": "7d",
        "view": "actionable",
        "rank_mode": "blended",
        "page": 1,
        "page_size": 50,
        "total": 0,
        "items": [],
    }


@pytest.mark.asyncio
async def test_unpublished_queue_exposes_sanitized_latest_source_outcomes(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone

    from app.infra.db.models.social_signals import SocialSignalRun
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    runtime = social_runtime.read_runtime()
    current = social_runtime.apply_runtime(
        "live", "xui", runtime.version, "admin"
    )
    started = datetime(2026, 9, 8, 5, 49, 14, tzinfo=timezone.utc)
    db_session.add(SocialSignalRun(
        id="failed-collection-1", registry_id=1,
        registry_version=current.version, mode="live", provider="xui",
        status="running",
        source_outcomes_json={
            "1": {
                "read_status": "failed", "processing_status": "failed",
                "history_status": "limited", "received_count": 0,
                "coverage_reason_codes": ["provider_unavailable"],
            },
            "2": {
                "read_status": "success", "processing_status": "pending",
                "history_status": "warming_up", "received_count": 50,
                "coverage_reason_codes": ["bounded_provider_read"],
            },
        },
        application_progress_json={
            "sources": {
                "1": {"name": "Minervini", "list_id": "private-list-1"},
                "2": {"name": "AI Investing", "list_id": "private-list-2"},
            },
            "observations": {"2": {"observed_at": started.isoformat()}},
        },
        feature_run_ids_json={}, exposure_dates_json={}, coverage_json={},
        created_at=started,
    ))
    db_session.commit()

    response = await _request(
        db_session, "GET",
        "/api/v1/social-signals/queue?market=US&window=7d",
    )

    assert response.status_code == 200
    attempt = response.json().get("latest_attempt")
    assert attempt == {
        "run_id": "failed-collection-1",
        "status": "collection_failed",
        "started_at": "2026-09-08T05:49:14Z",
        "completed_at": None,
        "sources": [
            {
                "name": "Minervini", "read_status": "failed",
                "received_count": 0, "history_status": "limited",
                "reason_codes": ["provider_unavailable"],
            },
            {
                "name": "AI Investing", "read_status": "success",
                "received_count": 50, "history_status": "warming_up",
                "reason_codes": ["bounded_provider_read"],
            },
        ],
    }
    encoded = response.text.lower()
    assert "private-list-1" not in encoded
    assert "private-list-2" not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "market=XX&window=7d",
        "market=US&window=3d",
        "market=US&window=7d&view=invalid",
        "market=US&window=7d&rank_mode=invalid",
        "market=US&window=7d&page_size=101",
    ],
)
async def test_queue_rejects_unsupported_controls(db_session, monkeypatch, query):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    response = await _request(
        db_session, "GET", f"/api/v1/social-signals/queue?{query}"
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_published_queue_keeps_rank_modes_and_unranked_sections_separate(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone
    from decimal import Decimal
    from app.domain.social_signals.records import SocialSnapshotRecord
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    runtime = social_runtime.read_runtime()
    social_runtime.apply_runtime("live", "official", runtime.version, "admin")
    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)

    def row(key, state, social, confirmation, queue, symbol, market="US"):
        return SocialSnapshotRecord(
            "published-1", key, symbol or None, market, state,
            Decimal(str(social)) if social is not None else None,
            Decimal(str(confirmation)) if confirmation is not None else None,
            Decimal(str(queue)) if queue is not None else None,
            (("state_reasons", state),), (), now, symbol, key, 7, 2, 2, 2,
        )

    _publish_rows(db_session, (
        row("US:AAA", "actionable", 90, 15, 60, "AAA"),
        row("US:BBB", "actionable", 70, 95, 80, "BBB"),
        row("US:SPY", "context", None, None, None, "SPY"),
        row("unresolved:z", "unresolved", None, None, None, "$ZZZ", None),
    ))

    blended = await _request(
        db_session, "GET",
        "/api/v1/social-signals/queue?market=US&window=7d&view=all&rank_mode=blended",
    )
    pure = await _request(
        db_session, "GET",
        "/api/v1/social-signals/queue?market=US&window=7d&view=all&rank_mode=pure_social",
    )
    actionable = await _request(
        db_session, "GET",
        "/api/v1/social-signals/queue?market=US&window=7d&view=actionable&rank_mode=blended",
    )

    assert [item["canonical_symbol"] for item in blended.json()["items"]] == [
        "BBB", "AAA",
    ]
    assert [item["canonical_symbol"] for item in pure.json()["items"]] == [
        "AAA", "BBB",
    ]
    assert [item["canonical_symbol"] for item in actionable.json()["items"]] == [
        "BBB", "AAA",
    ]
    assert pure.json()["items"][0]["confirmation_score"] == 15.0

    context = await _request(
        db_session, "GET",
        "/api/v1/social-signals/context?market=US&window=7d&page=1&page_size=50",
    )
    unresolved = await _request(
        db_session, "GET",
        "/api/v1/social-signals/unresolved?scope=unknown&market=US&window=7d&page=1&page_size=50",
    )
    assert [item["canonical_symbol"] for item in context.json()["items"]] == ["SPY"]
    assert [item["canonical_symbol"] for item in unresolved.json()["items"]] == ["$ZZZ"]


def test_published_queue_freshness_uses_oldest_successful_collection(
    db_session, social_runtime, monkeypatch,
):
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal

    from app.domain.social_signals.records import SocialSnapshotRecord
    from app.services.social_signal_query_service import SocialSignalQueries

    runtime = social_runtime.read_runtime()
    social_runtime.apply_runtime("live", "official", runtime.version, "admin")
    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    monkeypatch.setattr("app.services.social_signal_query_service.settings.social_stale_after_hours", 7)
    record = SocialSnapshotRecord(
        "published-1", "US:AAA", "AAA", "US", "actionable",
        Decimal("82"), Decimal("79"), Decimal("81"), (), (), now,
        "AAA", "US:AAA", 7, 4, 2, 2,
    )
    _publish_rows(db_session, (record,), observations={
        "1": {"observed_at": (now - timedelta(hours=8)).isoformat()},
        "2": {"observed_at": (now - timedelta(hours=1)).isoformat()},
    }, generated_at=now, published_at=now)

    payload = SocialSignalQueries(
        db_session, clock=lambda: now,
    ).queue(
        market="US", window="7d", view="actionable",
        rank_mode="blended", page=1, page_size=50,
    )

    assert payload["stale"] is True


@pytest.mark.asyncio
async def test_published_queue_exposes_frozen_candidate_context(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone
    from decimal import Decimal
    from app.domain.social_signals.records import SocialSnapshotRecord
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    runtime = social_runtime.read_runtime()
    social_runtime.apply_runtime("live", "official", runtime.version, "admin")
    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    record = SocialSnapshotRecord(
        "published-1", "US:AAA", "AAA", "US", "actionable",
        Decimal("82"), Decimal("79"), Decimal("81"),
        (("state_reasons", "legacy_reason"),), (), now, "AAA", "US:AAA",
        7, 4, 2, 2,
    )
    context = {
        "formula_version": "social-signal-v1",
        "candidates": [{
            "candidate_key": "US:AAA",
            "window_days": 7,
            "state_input": {
                "security_kind": "stock", "setup_score": "77",
                "setup_ready": True, "market_exposure": "65",
            },
            "state_decision": {"reasons": ["setup_ready", "theme_confirmed"]},
            "social_result": {
                "components": [["authors", {"value": "80"}]],
                "acceleration": "2.5", "post_memberships": ["111"],
            },
            "confirmation": {"components": [
                ["setup", {"value": "77"}],
                ["rs", {"value": "91"}],
                ["group", {"value": "88"}],
                ["theme", {
                    "value": "83", "selected_key": "ai_infrastructure",
                }],
            ]},
        }],
        "market_batches": [{"inputs": [{
            "candidate_key": "US:AAA", "rs_rating_1m": "89",
            "rs_rating_3m": "93", "group_rank": 4,
        }]}],
    }
    _publish_rows(db_session, (record,), context=context)

    response = await _request(
        db_session, "GET",
        "/api/v1/social-signals/queue?market=US&window=7d",
    )

    assert response.status_code == 200
    explanation = response.json()["items"][0]["explanation"]
    assert explanation["state_reasons"] == ["setup_ready", "theme_confirmed"]
    assert explanation["security_kind"] == "stock"
    assert explanation["setup_score"] == "77"
    assert explanation["readiness"] == "ready"
    assert explanation["market_exposure"] == "65"
    assert explanation["rs_rating_1m"] == "89"
    assert explanation["rs_rating_3m"] == "93"
    assert explanation["group_rank"] == 4
    assert explanation["theme"] == "ai_infrastructure"
    assert explanation["acceleration"] == "2.5"
    assert explanation["post_memberships"] == ["111"]
    assert explanation["source_names"] == ["One"]


@pytest.mark.asyncio
async def test_queue_filters_before_server_pagination(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone
    from decimal import Decimal

    from app.domain.social_signals.records import SocialSnapshotRecord
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    runtime = social_runtime.read_runtime()
    social_runtime.apply_runtime("live", "official", runtime.version, "admin")
    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)

    def record(symbol, score):
        return SocialSnapshotRecord(
            "published-1", f"US:{symbol}", symbol, "US", "actionable",
            Decimal(str(score)), Decimal("50"), Decimal(str(score)), (), (),
            now, symbol, f"US:{symbol}", 7, 1, 1, 2,
        )

    context = {
        "formula_version": "social-signal-v1",
        "candidates": [
            {
                "candidate_key": "US:AAA", "window_days": 7,
                "state_input": {"security_kind": "stock"},
                "social_result": {"post_memberships": [["post-a", ["2"]]]},
                "confirmation": {"components": [["theme", {"selected_key": "cooling"}]]},
            },
            {
                "candidate_key": "US:BBB", "window_days": 7,
                "state_input": {"security_kind": "etf"},
                "social_result": {"post_memberships": [["post-b", ["1"]]]},
                "confirmation": {"components": [["theme", {"selected_key": "banks"}]]},
            },
        ],
    }
    _publish_rows(db_session, (record("BBB", 90), record("AAA", 80)), context=context)

    response = await _request(
        db_session,
        "GET",
        "/api/v1/social-signals/queue?market=US&window=7d&view=all"
        "&page=1&page_size=1&source=Two&theme=cool&instrument=stock&ticker=AA",
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [item["canonical_symbol"] for item in response.json()["items"]] == ["AAA"]


@pytest.mark.asyncio
async def test_theme_pulse_keeps_social_and_market_strength_distinct(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone
    from decimal import Decimal
    from app.domain.social_signals.records import SocialSnapshotRecord
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    runtime = social_runtime.read_runtime()
    social_runtime.apply_runtime("live", "official", runtime.version, "admin")
    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    record = SocialSnapshotRecord(
        "published-1", "HK:0700", "0700", "HK", "watch",
        Decimal("84"), Decimal("71"), Decimal("78.8"), (), (), now,
        "0700", "HK:0700", 7, 2, 2, 2,
    )
    context = {"formula_version": "social-signal-v1", "candidates": [{
        "candidate_key": "HK:0700", "window_days": 7,
        "social_result": {"social_score": "84", "components": []},
        "confirmation": {"components": []}, "state_input": {},
        "state_decision": {"reasons": []},
    }]}
    evidence = [{
        "theme_key": "ai_datacentres", "market": "HK",
        "benchmark_symbol": "HSI", "accepted_company_count": 5,
        "components": [["basket_rs_vs_benchmark", "70"], ["avg_rs_rating", "72"]],
        "measured_company_counts": [["basket_rs_vs_benchmark", 4], ["avg_rs_rating", 4]],
        "reasons": [], "membership": [{"canonical_symbol": "0700"}],
    }]
    projection = {"proposals": [
        {"theme_key": "ai_datacentres"}, {"theme_key": "robotics"},
    ], "resolutions": [
        {"market": "HK", "symbol": "0700", "company_count_eligible": True,
         "company_id": "tencent"},
        {"market": "HK", "symbol": "9988", "company_count_eligible": True,
         "company_id": "alibaba"},
    ]}
    _publish_rows(
        db_session, (record,), context=context,
        theme_evidence=evidence, projection=projection,
    )

    response = await _request(
        db_session, "GET", "/api/v1/social-signals/theme-pulse?market=HK"
    )

    assert response.status_code == 200
    by_key = {item["theme_key"]: item for item in response.json()["items"]}
    assert by_key["ai_datacentres"]["social_strength"] == 84.0
    assert by_key["ai_datacentres"]["market_strength"] == 71.0
    assert by_key["ai_datacentres"]["measured_company_count"] == 4
    assert by_key["ai_datacentres"]["benchmark_symbol"] == "HSI"
    assert by_key["robotics"]["status"] == "discovering"
    assert by_key["robotics"]["social_strength"] is None
    assert by_key["robotics"]["market_strength"] is None


@pytest.mark.asyncio
async def test_evidence_is_run_scoped_canonical_and_limited_to_three_plain_excerpts(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    from app.domain.social_signals.records import SocialSnapshotRecord
    from app.infra.db.models.social_signals import SocialPostTicker, SocialSignalRun
    from app.models.stock_universe import StockUniverse
    from app.models.theme import ContentItem
    from app.services import server_auth
    from app.services.social_company_identity_service import SocialCompanyIdentityService

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    runtime = social_runtime.read_runtime()
    runtime = social_runtime.apply_runtime("live", "official", runtime.version, "admin")
    SocialCompanyIdentityService(db_session, admin_authorized=True).replace([
        {"symbol": "AAA", "company_id": "issuer-aaa",
         "verification_reference": "issuer register 2026-09-01",
         "verified_at": "2026-09-01T00:00:00+00:00"},
        {"symbol": "BBB", "company_id": "issuer-aaa",
         "verification_reference": "issuer register 2026-09-01",
         "verified_at": "2026-09-01T00:00:00+00:00"},
    ], expected_version=runtime.version, actor="test")
    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    security = StockUniverse(symbol="AAA", market="US", exchange="NASDAQ", is_active=True)
    related_security = StockUniverse(
        symbol="BBB", market="HK", exchange="HKEX", is_active=True,
    )
    db_session.add_all([security, related_security])
    db_session.flush()
    inputs = []
    for number in range(4):
        item = ContentItem(
            source_id=1, source_type="twitter", external_id=f"p-{number}",
            content="x" * 400, url=f"https://example.invalid/{number}",
            author=f"author{number}", published_at=now - timedelta(hours=number),
            fetched_at=now,
        )
        db_session.add(item)
        db_session.flush()
        db_session.add(SocialPostTicker(
            content_item_id=item.id, candidate_key="US:AAA", raw_token="$AAA",
            stock_universe_id=security.id, canonical_symbol="AAA", market="US",
            mic="XNAS", local_code="AAA", resolution_state="resolved",
            resolution_policy_version="social-resolution-v1", explanation_json={},
        ))
        inputs.append({
            "content_item_id": item.id,
            "post": {
                "provider": "official", "provider_post_id": f"p-{number}",
                "source_id": "1", "text": "x" * 400,
                "url": f"https://example.invalid/{number}",
                "author_handle": f"author{number}",
                "created_at": (now - timedelta(hours=number)).isoformat(),
                "observed_at": now.isoformat(), "likes": number,
                "reposts": 0, "replies": 0, "quotes": None,
                "bookmarks": None, "views": None,
                "canonical_url": None, "is_repost": False,
                "quoted_text": None, "has_new_thesis": None,
                "canonical_claim_key": None,
            },
        })
    retained_input = inputs.pop(0)
    db_session.add(SocialSignalRun(
        id="older-evidence", registry_id=1, registry_version=2,
        mode="live", provider="official", status="completed",
        source_outcomes_json={
            "1": {"read_status": "success", "history_status": "limited"},
            "2": {"read_status": "success", "history_status": "limited"},
        },
        application_progress_json={
            "sources": {
                "1": {"list_id": "111", "name": "One", "version": 1},
                "2": {"list_id": "222", "name": "Two", "version": 1},
            },
            "observations": {
                "1": {"inputs": [retained_input]}, "2": {"inputs": []},
            },
        }, feature_run_ids_json={}, exposure_dates_json={}, coverage_json={},
        created_at=now - timedelta(days=1), completed_at=now - timedelta(days=1),
    ))
    db_session.commit()
    record = SocialSnapshotRecord(
        "published-1", "US:AAA", "AAA", "US", "watch",
        Decimal("80"), Decimal("50"), Decimal("68"),
        (("state_reasons", "setup_not_ready"),), (), now, "AAA", "US:AAA",
        7, 4, 1, 2,
    )
    observations = {
        "1": {"inputs": inputs, "observed_at": now.isoformat()},
        "2": {"inputs": [], "observed_at": now.isoformat()},
    }
    _publish_rows(db_session, (record,), observations=observations)

    response = await _request(
        db_session, "GET",
        "/api/v1/social-signals/candidates/US:AAA/evidence?window=7d",
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["posts"]) == 3
    assert payload["posts"][0]["post_id"] == "p-0"
    assert all(len(post["excerpt"]) <= 280 for post in payload["posts"])
    assert payload["posts"][0]["url"] == "https://x.com/author0/status/p-0"
    assert payload["posts"][0]["source_names"] == ["One"]
    assert payload["related_listings"] == [
        {"market": "HK", "canonical_symbol": "BBB"},
    ]


@pytest.mark.asyncio
async def test_admin_routes_require_admin_key_even_when_social_is_off(
    db_session, social_runtime, monkeypatch
):
    from app.api.v1 import config
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")

    denied = await _request(
        db_session, "GET", "/api/v1/social-signals/admin/runtime"
    )
    allowed = await _request(
        db_session, "GET", "/api/v1/social-signals/admin/runtime",
        headers={"X-Admin-Key": "admin-secret"},
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["mode"] == "off"
    assert allowed.json()["provider"] == "disabled"
    assert allowed.json()["version"] >= 1


@pytest.mark.asyncio
async def test_admin_runtime_and_source_lifecycle_are_audited_and_redacted(
    db_session, social_runtime, monkeypatch
):
    from app.api.v1 import config
    from app.interfaces.tasks import social_signal_tasks
    from app.services import server_auth

    class Dispatched:
        id = "validation-task"

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(
        social_signal_tasks.validate_social_source, "apply_async",
        lambda **_kwargs: Dispatched(),
    )
    headers = {"X-Admin-Key": "admin-secret"}
    runtime = social_runtime.read_runtime()
    changed = await _request(
        db_session, "PATCH", "/api/v1/social-signals/admin/runtime",
        headers=headers, json={"mode": "validation", "provider": "official",
                               "expected_version": runtime.version},
    )
    assert changed.status_code == 200

    created = await _request(
        db_session, "POST", "/api/v1/social-signals/admin/sources",
        headers=headers, json={"name": "  Asia Semis  ",
                               "list_ref": "https://x.com/i/lists/123456"},
    )
    assert created.status_code == 201
    source = created.json()
    assert (source["name"], source["list_id"], source["lifecycle"]) == (
        "Asia Semis", "123456", "pending"
    )

    stale = await _request(
        db_session, "PATCH", f"/api/v1/social-signals/admin/sources/{source['source_id']}",
        headers=headers, json={"name": "Nope", "expected_version": 999},
    )
    assert stale.status_code == 409

    renamed = await _request(
        db_session, "PATCH", f"/api/v1/social-signals/admin/sources/{source['source_id']}",
        headers=headers, json={"name": "Asia Chips", "expected_version": source["version"]},
    )
    assert renamed.status_code == 200
    renamed_source = renamed.json()
    assert renamed_source["list_id"] == "123456"

    queued = await _request(
        db_session, "POST",
        f"/api/v1/social-signals/admin/sources/{source['source_id']}/test",
        headers=headers, json={"expected_version": renamed_source["version"]},
    )
    assert queued.status_code == 202
    assert queued.json()["task_id"] == "validation-task"

    listing = await _request(
        db_session, "GET", "/api/v1/social-signals/admin/sources",
        headers=headers,
    )
    encoded = listing.text.lower()
    assert listing.status_code == 200
    assert all(secret not in encoded for secret in (
        "cookie", "storage_state", "config_path", "stderr", "xui-reader"
    ))
    current = next(row for row in listing.json() if row["source_id"] == source["source_id"])
    assert current["test_progress"] == "queued"
    assert [event["action"] for event in current["audit"]] == [
        "created", "renamed", "test_requested",
    ]


@pytest.mark.asyncio
async def test_admin_company_identity_configuration_is_optimistic(
    db_session, social_runtime, monkeypatch
):
    from app.api.v1 import config
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    headers = {"X-Admin-Key": "admin-secret"}
    before = await _request(
        db_session, "GET", "/api/v1/social-signals/admin/company-identities",
        headers=headers,
    )
    db_session.rollback()
    updated = await _request(
        db_session, "PATCH", "/api/v1/social-signals/admin/company-identities",
        headers=headers, json={"expected_version": before.json()["registry_version"], "entries": [{
            "symbol": "BRK.B", "company_id": "berkshire-hathaway",
            "verification_reference": "issuer filing 2026-01-01",
            "verified_at": "2026-01-01T00:00:00+00:00",
        }]},
    )
    conflict = await _request(
        db_session, "PATCH", "/api/v1/social-signals/admin/company-identities",
        headers=headers, json={"expected_version": before.json()["registry_version"], "entries": []},
    )

    assert updated.status_code == 200, updated.text
    assert updated.json()["entries"][0]["company_id"] == "berkshire-hathaway"
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_admin_analysis_retry_replays_the_work_linked_generation(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.api.v1 import config
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
    from app.infra.db.models.social_signals import SocialSignalRun
    from app.interfaces.tasks.social_signal_tasks import resume_social_analysis
    from app.models.theme import ContentItem
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    runtime = social_runtime.read_runtime()
    runtime = social_runtime.apply_runtime(
        "validation", "official", runtime.version, "admin"
    )
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    item = ContentItem(
        source_type="twitter", external_id="retry-item", content="$AAA",
        url="https://x.com/a/status/retry", published_at=now,
    )
    db_session.add(item)
    db_session.flush()
    work = SocialExtractionWork(
        content_item_id=item.id,
        input_hash="retry-hash",
        prompt_version="v1",
        schema_version="v1",
        selected_model="model",
        input_snapshot_json={},
        state="failed_terminal",
        error_code="original_error",
        requested_by_admin=False,
        created_at=now,
        updated_at=now,
    )
    db_session.add(work)
    db_session.flush()
    run = SocialSignalRun(
        id="retry-generation", registry_id=1, registry_version=runtime.version,
        mode="validation", provider="official", status="running",
        source_outcomes_json={}, application_progress_json={
            "sources": {}, "observations": {},
        }, feature_run_ids_json={}, exposure_dates_json={}, coverage_json={},
        created_at=now,
    )
    db_session.add(run)
    db_session.flush()
    db_session.add(SocialRunWork(
        run_id=run.id, work_id=work.id, input_hash=work.input_hash,
        included_at=now,
    ))
    db_session.commit()
    calls = []
    def dispatch(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("synthetic dispatch failure")
        return SimpleNamespace(id="retry-task")
    monkeypatch.setattr(resume_social_analysis, "apply_async", dispatch)

    with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
        await _request(
            db_session,
            "POST",
            f"/api/v1/social-signals/admin/analysis/{work.id}/retry",
            headers={"X-Admin-Key": "admin-secret"},
        )
    db_session.refresh(work)
    assert (work.state, work.error_code, work.requested_by_admin) == (
        "failed_terminal", "original_error", False,
    )

    response = await _request(
        db_session,
        "POST",
        f"/api/v1/social-signals/admin/analysis/{work.id}/retry",
        headers={"X-Admin-Key": "admin-secret"},
    )

    assert response.status_code == 202
    assert calls == [
        {"args": ["retry-generation"], "queue": "social_ingestion"},
        {"args": ["retry-generation"], "queue": "social_ingestion"},
    ]


@pytest.mark.asyncio
async def test_validation_preview_includes_staged_association_proposals(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone

    from app.api.v1 import config
    from app.infra.db.models.social_signals import SocialSignalRun
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    runtime = social_runtime.read_runtime()
    runtime = social_runtime.apply_runtime(
        "validation", "official", runtime.version, "admin"
    )
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    db_session.add(SocialSignalRun(
        id="validation-associations", registry_id=1,
        registry_version=runtime.version, mode="validation", provider="official",
        status="staged", source_outcomes_json={}, application_progress_json={
            "sources": {}, "observations": {}, "prepared": {"projection": {
                "proposals": [{
                    "post_id": "post-1", "theme_key": "cooling",
                    "raw_theme": "Cooling", "company_token": "$AAA",
                    "relationship": "supplies", "support": "supported",
                }],
                "resolutions": [{
                    "raw_token": "$AAA", "status": "resolved", "symbol": "AAA",
                    "market": "US", "reason_codes": [],
                }],
            }},
        }, feature_run_ids_json={}, exposure_dates_json={}, coverage_json={},
        created_at=now, completed_at=now,
    ))
    db_session.commit()

    response = await _request(
        db_session,
        "GET",
        "/api/v1/social-signals/admin/validation/validation-associations",
        headers={"X-Admin-Key": "admin-secret"},
    )

    assert response.status_code == 200
    assert response.json()["associations"] == [{
        "post_id": "post-1", "theme_key": "cooling", "raw_theme": "Cooling",
        "company_token": "$AAA", "relationship": "supplies",
        "support": "supported", "resolution": {
            "raw_token": "$AAA", "status": "resolved", "symbol": "AAA",
            "market": "US", "reason_codes": [],
        },
    }]


@pytest.mark.asyncio
async def test_association_decision_requires_live_mode_reason_and_current_version(
    db_session, social_runtime, monkeypatch
):
    from datetime import datetime, timezone
    from app.api.v1 import config
    from app.infra.db.models.social_analysis import SocialThemeAssociation, SocialThemeDecision
    from app.models.theme import ThemeCluster
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    headers = {"X-Admin-Key": "admin-secret"}
    theme = ThemeCluster(
        name="Robotics", display_name="Robotics", canonical_key="robotics",
        pipeline="technical", aliases=[], lifecycle_state="candidate", is_active=True,
    )
    db_session.add(theme)
    db_session.flush()
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    association = SocialThemeAssociation(
        theme_cluster_id=theme.id, market="JP", canonical_symbol="6954",
        state="proposed", origin="social", decision_owner="system",
        evidence_work_ids=[], policy_version="policy-v1", version=1,
        first_seen_at=now, updated_at=now,
    )
    db_session.add(association)
    db_session.flush()
    association_id = association.id
    db_session.commit()

    off = await _request(
        db_session, "POST",
        f"/api/v1/social-signals/admin/associations/{association_id}/decision",
        headers=headers, json={"target": "accepted", "reason": "reviewed filing",
                               "expected_version": 1},
    )
    assert off.status_code == 422
    runtime = social_runtime.read_runtime()
    social_runtime.apply_runtime("live", "official", runtime.version, "admin")
    accepted = await _request(
        db_session, "POST",
        f"/api/v1/social-signals/admin/associations/{association_id}/decision",
        headers=headers, json={"target": "accepted", "reason": " reviewed filing ",
                               "expected_version": 1},
    )
    stale = await _request(
        db_session, "POST",
        f"/api/v1/social-signals/admin/associations/{association_id}/decision",
        headers=headers, json={"target": "rejected", "reason": "changed",
                               "expected_version": 1},
    )

    assert accepted.status_code == 200
    assert stale.status_code == 409
    decision = db_session.query(SocialThemeDecision).one()
    assert (decision.before_state, decision.after_state, decision.reason) == (
        "proposed", "accepted", "reviewed filing"
    )
