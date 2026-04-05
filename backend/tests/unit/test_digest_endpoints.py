from __future__ import annotations

from datetime import date

import httpx
import pytest
import pytest_asyncio

from app.api.v1.digest import _get_digest_service
from app.database import get_db
from app.main import app
from app.schemas.digest import (
    DailyDigestResponse,
    DigestBreadthMetrics,
    DigestFreshness,
    DigestLeaderItem,
    DigestMarketSection,
    DigestRiskNote,
    DigestThemeSection,
    DigestValidationSection,
    DigestValidationSourceSnapshot,
    DigestWatchlistHighlight,
)
from app.schemas.validation import ValidationHorizonSummary, ValidationSourceKind
from app.services import server_auth

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _disable_server_auth():
    server_auth.settings.server_auth_enabled = False
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class _FakeDigestService:
    def __init__(self):
        self.requested_dates: list[date | None] = []

    def get_daily_digest(self, db, *, as_of_date=None):  # noqa: ANN001
        self.requested_dates.append(as_of_date)
        return DailyDigestResponse(
            as_of_date="2026-04-04",
            freshness=DigestFreshness(
                latest_feature_as_of_date="2026-04-04",
                latest_breadth_date="2026-04-04",
                latest_theme_metrics_date="2026-04-04",
                latest_theme_alert_at="2026-04-04T15:30:00+00:00",
                validation_lookback_days=90,
            ),
            market=DigestMarketSection(
                stance="offense",
                summary="Current stance is offense.",
                breadth_metrics=DigestBreadthMetrics(up_4pct=100, down_4pct=20, ratio_5day=1.5, ratio_10day=1.3, total_stocks_scanned=4200),
            ),
            leaders=[
                DigestLeaderItem(
                    symbol="NVDA",
                    name="NVIDIA",
                    composite_score=96.0,
                    rating="Strong Buy",
                    industry_group="Semiconductors",
                    reason_summary="Strengths led by stage, rs rating.",
                )
            ],
            themes=DigestThemeSection(leaders=[], laggards=[], recent_alerts=[]),
            validation=DigestValidationSection(
                lookback_days=90,
                scan_pick=DigestValidationSourceSnapshot(
                    source_kind=ValidationSourceKind.SCAN_PICK,
                    horizons=[ValidationHorizonSummary(horizon_sessions=1, sample_size=10)],
                    degraded_reasons=[],
                ),
                theme_alert=DigestValidationSourceSnapshot(
                    source_kind=ValidationSourceKind.THEME_ALERT,
                    horizons=[ValidationHorizonSummary(horizon_sessions=1, sample_size=4)],
                    degraded_reasons=["missing_price_cache"],
                ),
            ),
            watchlists=[
                DigestWatchlistHighlight(
                    watchlist_id=1,
                    watchlist_name="Core Leaders",
                    matched_symbols=["NVDA"],
                    alert_symbols=["NVDA"],
                    notes="1 leader overlap and 1 alert overlap out of 2 tracked symbols.",
                )
            ],
            risks=[DigestRiskNote(kind="validation", message="Recent theme-alert follow-through is weak.", severity="warning")],
            degraded_reasons=["missing_recent_theme_alerts"],
        )

    def render_markdown(self, payload):
        return f"# Daily Digest ({payload.as_of_date})\n\n- NVDA\n"


@pytest.mark.asyncio
async def test_daily_digest_endpoint_returns_payload_and_passes_date_query(client):
    fake_service = _FakeDigestService()
    app.dependency_overrides[_get_digest_service] = lambda: fake_service
    app.dependency_overrides[get_db] = lambda: iter([None])

    response = await client.get("/api/v1/digest/daily", params={"as_of_date": "2026-04-03"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["as_of_date"] == "2026-04-04"
    assert payload["leaders"][0]["symbol"] == "NVDA"
    assert fake_service.requested_dates == [date(2026, 4, 3)]


@pytest.mark.asyncio
async def test_daily_digest_markdown_endpoint_returns_text(client):
    fake_service = _FakeDigestService()
    app.dependency_overrides[_get_digest_service] = lambda: fake_service
    app.dependency_overrides[get_db] = lambda: iter([None])

    response = await client.get("/api/v1/digest/daily/markdown")

    assert response.status_code == 200
    assert response.text.startswith("# Daily Digest (2026-04-04)")
    assert response.headers["content-type"].startswith("text/plain")
