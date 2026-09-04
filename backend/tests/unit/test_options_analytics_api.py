from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from app.database import get_db
from app.main import app
from app.schemas.options_analytics import (
    OptionsCommandCenterResponse,
    OptionsMetricResponse,
)


def _item(symbol: str, *, kind="current", state="available", core_valid=None):
    return SimpleNamespace(
        security_symbol=symbol,
        candidate_kind=kind,
        candidate_rank=1 if symbol == "AAPL" else None,
        leader_rank=1 if symbol == "AAPL" else 2,
        spot_price=100.0,
        expiration=date(2026, 9, 18),
        observation_state=state,
        core_valid=(state == "available" if core_valid is None else core_valid),
        observation_at=datetime(2026, 9, 4, 2, tzinfo=timezone.utc),
        max_pain=100.0 if state == "available" else None,
        net_gex=2500.0 if state == "available" else None,
        gamma_flip=None,
        call_wall=105.0,
        put_wall=95.0,
        atm_iv=0.25 if state == "available" else None,
        skew_25_delta=None,
        realized_volatility=0.20,
        vrp=0.05,
        activity_intensity=0.8,
        activity_rank=1,
        call_open_interest=1000,
        put_open_interest=900,
        call_volume=300,
        put_volume=250,
        volume_oi_ratio=550 / 1900,
        near_spot_volume_concentration=0.7,
        short_history_observation_count=3,
        iv_history_observation_count=3,
        lifetime_observation_count=3,
        retry_count=0,
        evidence_json={
            "quality": {
                "provider_spot_price": 101.0,
                "normalized_call_count": 5,
                "normalized_put_count": 5,
            },
            "gamma_flip": {
                "available": False,
                "label": "Estimated Gamma Flip",
                "reason_codes": ["gamma_crossing_unavailable"],
                "evidence": {},
            }
        },
        assumptions_json={"risk_free_rate": 0.04},
        warnings_json=[],
        reasons_json=["building_history"] if state == "available" else ["provider_unavailable"],
        strike_points=[],
    )


def _run():
    return SimpleNamespace(
        id=17,
        market="US",
        source_feature_run_id=33,
        calculation_version="options-analytics-v1",
        schema_version="options-analytics-v1",
        provider="yahoo",
        as_of_date=date(2026, 9, 4),
        created_at=datetime(2026, 9, 4, 1, tzinfo=timezone.utc),
        published_at=datetime(2026, 9, 4, 3, tzinfo=timezone.utc),
        expected_count=3,
        current_count=2,
        continuity_count=1,
        completed_count=3,
        core_valid_current_count=1,
        failed_count=1,
        retried_count=0,
        coverage=0.5,
        warnings_json=[],
        diagnostics_json={},
        assumptions_json={"risk_free_source": "Yahoo ^IRX close"},
        items=[
            _item("AAPL"),
            _item("MSFT", state="unavailable"),
            _item("OLD", kind="continuity"),
        ],
    )


def test_command_center_contract_keeps_all_current_rows_and_excludes_continuity() -> None:
    payload = OptionsCommandCenterResponse.from_run(_run())

    assert payload.run_id == 17
    assert [row.symbol for row in payload.items] == ["AAPL", "MSFT"]
    assert payload.items[0].source_badges == ["candidate", "leader"]
    assert payload.items[0].metrics.net_gex.label == "Estimated Net GEX"
    assert payload.items[0].metrics.gamma_flip.reason_codes == [
        "gamma_crossing_unavailable"
    ]
    assert payload.items[0].quality_evidence["provider_spot_price"] == 101.0
    assert payload.items[1].state == "unavailable"


def test_public_contract_rejects_extra_and_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        OptionsMetricResponse(available=True, value=float("nan"), surprise=True)


def test_non_core_observation_is_distinct_from_provider_unavailable() -> None:
    run = _run()
    run.items = [_item("AAPL", core_valid=False)]

    payload = OptionsCommandCenterResponse.from_run(run)

    assert payload.items[0].state == "insufficient_quality"


@pytest.mark.asyncio
async def test_live_reads_return_typed_404_and_refresh_is_immediate(monkeypatch) -> None:
    from app.api.v1 import options_analytics as api
    from app.services import server_auth

    class Queries:
        def get_published_command_center(self, market):
            assert market == "US"

    monkeypatch.setattr(api, "get_options_analytics_queries", lambda _db: Queries())
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(
        api,
        "dispatch_options_refresh",
        lambda **values: {"task_id": "task-7", "run_id": None, **values},
    )
    app.dependency_overrides[get_db] = lambda: object()
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.get("/api/v1/options-analytics/command-center")
            accepted = await client.post(
                "/api/v1/options-analytics/refresh",
                json={"source_run_id": 33, "force": False},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "options_analytics_unavailable"
    assert accepted.status_code == 202
    assert accepted.json() == {
        "status": "accepted",
        "task_id": "task-7",
        "run_id": None,
        "source_run_id": 33,
    }
