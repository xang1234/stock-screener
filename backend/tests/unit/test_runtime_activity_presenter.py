"""Unit tests for runtime activity presentation."""

from __future__ import annotations

from types import SimpleNamespace


def _bootstrap_status(
    *,
    bootstrap_state: str = "not_started",
    bootstrap_required: bool = True,
    primary_market: str = "US",
    enabled_markets: list[str] | None = None,
):
    return SimpleNamespace(
        bootstrap_state=bootstrap_state,
        bootstrap_required=bootstrap_required,
        primary_market=primary_market,
        enabled_markets=enabled_markets or [primary_market],
    )


def test_bootstrap_summary_ignores_daily_refresh_activity_when_not_started():
    from app.services.runtime_activity_contract import RuntimeActivityRecord
    from app.services.runtime_activity_presenter import build_runtime_activity_status

    daily_failure = RuntimeActivityRecord.create(
        market="US",
        lifecycle="daily_refresh",
        stage_key="breadth",
        status="failed",
        message="Daily breadth calculation failed: no usable stocks",
    ).to_payload()

    status = build_runtime_activity_status(
        bootstrap_status=_bootstrap_status(),
        bootstrap_run={},
        market_payloads=[daily_failure],
    )

    assert status["bootstrap"]["state"] == "not_started"
    assert status["bootstrap"]["current_stage"] is None
    assert status["bootstrap"]["message"] == "Bootstrap queued."
    assert status["summary"]["status"] == "warning"
    assert status["markets"] == [daily_failure]
