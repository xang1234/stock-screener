from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster, ThemeMetrics
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.schemas.validation import (
    ValidationFreshness,
    ValidationHorizonSummary,
    ValidationOverviewResponse,
    ValidationSourceKind,
)
from app.services.digest_service import DigestService


class _FakeValidationService:
    def get_overview(self, db, *, source_kind, lookback_days, as_of_date=None):  # noqa: ANN001
        if source_kind == ValidationSourceKind.SCAN_PICK:
            horizons = [
                ValidationHorizonSummary(
                    horizon_sessions=1,
                    sample_size=12,
                    positive_rate=0.67,
                    avg_return_pct=1.8,
                    median_return_pct=1.4,
                    avg_mfe_pct=None,
                    avg_mae_pct=None,
                    skipped_missing_history=0,
                ),
                ValidationHorizonSummary(
                    horizon_sessions=5,
                    sample_size=11,
                    positive_rate=0.64,
                    avg_return_pct=3.6,
                    median_return_pct=2.9,
                    avg_mfe_pct=5.4,
                    avg_mae_pct=-2.0,
                    skipped_missing_history=1,
                ),
            ]
            degraded_reasons: list[str] = []
        else:
            horizons = [
                ValidationHorizonSummary(
                    horizon_sessions=1,
                    sample_size=6,
                    positive_rate=0.33,
                    avg_return_pct=-0.4,
                    median_return_pct=-0.2,
                    avg_mfe_pct=None,
                    avg_mae_pct=None,
                    skipped_missing_history=0,
                ),
                ValidationHorizonSummary(
                    horizon_sessions=5,
                    sample_size=5,
                    positive_rate=0.4,
                    avg_return_pct=-1.1,
                    median_return_pct=-0.7,
                    avg_mfe_pct=1.8,
                    avg_mae_pct=-3.2,
                    skipped_missing_history=0,
                ),
            ]
            degraded_reasons = ["theme_alert_missing_price_cache"]

        return ValidationOverviewResponse(
            source_kind=source_kind,
            lookback_days=lookback_days,
            horizons=horizons,
            recent_events=[],
            failure_clusters=[],
            freshness=ValidationFreshness(
                latest_feature_as_of_date="2026-04-04",
                latest_theme_alert_at="2026-04-04T15:30:00+00:00",
                price_cache_period="2y",
            ),
            degraded_reasons=degraded_reasons,
        )


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            StockUniverse.__table__,
            MarketBreadth.__table__,
            ThemeCluster.__table__,
            ThemeMetrics.__table__,
            ThemeAlert.__table__,
            UserWatchlist.__table__,
            WatchlistItem.__table__,
        ],
    )
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(
            engine,
            tables=[
                WatchlistItem.__table__,
                UserWatchlist.__table__,
                ThemeAlert.__table__,
                ThemeMetrics.__table__,
                ThemeCluster.__table__,
                MarketBreadth.__table__,
                StockUniverse.__table__,
                StockFeatureDaily.__table__,
                FeatureRun.__table__,
            ],
        )


def _seed_digest_data(session):
    run = FeatureRun(
        as_of_date=date(2026, 4, 4),
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(2026, 4, 4, 21, 0, tzinfo=UTC),
    )
    session.add(run)
    session.add(
        MarketBreadth(
            date=date(2026, 4, 4),
            stocks_up_4pct=120,
            stocks_down_4pct=35,
            ratio_5day=1.7,
            ratio_10day=1.4,
            stocks_up_25pct_quarter=0,
            stocks_down_25pct_quarter=0,
            stocks_up_25pct_month=0,
            stocks_down_25pct_month=0,
            stocks_up_50pct_month=0,
            stocks_down_50pct_month=0,
            stocks_up_13pct_34days=0,
            stocks_down_13pct_34days=0,
            total_stocks_scanned=4200,
        )
    )
    session.add_all(
        [
            StockUniverse(symbol="NVDA", name="NVIDIA", sector="Technology", industry="Semiconductors", is_active=True, status="active"),
            StockUniverse(symbol="AVGO", name="Broadcom", sector="Technology", industry="Semiconductors", is_active=True, status="active"),
            StockUniverse(symbol="MSFT", name="Microsoft", sector="Technology", industry="Software", is_active=True, status="active"),
        ]
    )
    session.commit()
    session.refresh(run)

    base_details = {
        "details": {
            "screeners": {
                "minervini": {
                    "score": 90,
                    "passes": True,
                    "rating": "Strong Buy",
                    "breakdown": {
                        "rs_rating": {"points": 18, "max_points": 20, "passes": True},
                        "stage": {"points": 20, "max_points": 20, "passes": True},
                    },
                }
            }
        },
        "screeners_run": ["minervini"],
        "screeners_passed": 1,
        "screeners_total": 1,
        "composite_method": "weighted_average",
    }

    session.add_all(
        [
            StockFeatureDaily(
                run_id=run.id,
                symbol="NVDA",
                as_of_date=run.as_of_date,
                composite_score=96.0,
                overall_rating=5,
                passes_count=1,
                details_json={**base_details, "company_name": "NVIDIA", "ibd_industry_group": "Semiconductors"},
            ),
            StockFeatureDaily(
                run_id=run.id,
                symbol="AVGO",
                as_of_date=run.as_of_date,
                composite_score=92.0,
                overall_rating=4,
                passes_count=1,
                details_json={**base_details, "company_name": "Broadcom", "ibd_industry_group": "Semiconductors"},
            ),
            StockFeatureDaily(
                run_id=run.id,
                symbol="MSFT",
                as_of_date=run.as_of_date,
                composite_score=88.0,
                overall_rating=4,
                passes_count=1,
                details_json={**base_details, "company_name": "Microsoft", "ibd_industry_group": "Software"},
            ),
        ]
    )

    ai_theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infrastructure",
        display_name="AI Infrastructure",
        pipeline="technical",
        category="technology",
        lifecycle_state="active",
    )
    software_theme = ThemeCluster(
        name="Enterprise Software",
        canonical_key="enterprise-software",
        display_name="Enterprise Software",
        pipeline="technical",
        category="technology",
        lifecycle_state="active",
    )
    lagging_theme = ThemeCluster(
        name="Solar",
        canonical_key="solar",
        display_name="Solar",
        pipeline="technical",
        category="energy",
        lifecycle_state="active",
    )
    session.add_all([ai_theme, software_theme, lagging_theme])
    session.commit()
    session.refresh(ai_theme)
    session.refresh(software_theme)
    session.refresh(lagging_theme)

    session.add_all(
        [
            ThemeMetrics(
                theme_cluster_id=ai_theme.id,
                date=date(2026, 4, 4),
                pipeline="technical",
                mention_velocity=1.8,
                basket_return_1m=12.5,
                momentum_score=84.0,
                status="trending",
            ),
            ThemeMetrics(
                theme_cluster_id=software_theme.id,
                date=date(2026, 4, 4),
                pipeline="technical",
                mention_velocity=1.2,
                basket_return_1m=8.4,
                momentum_score=76.0,
                status="trending",
            ),
            ThemeMetrics(
                theme_cluster_id=lagging_theme.id,
                date=date(2026, 4, 4),
                pipeline="technical",
                mention_velocity=0.6,
                basket_return_1m=-11.2,
                momentum_score=18.0,
                status="fading",
            ),
        ]
    )
    session.add(
        ThemeAlert(
            theme_cluster_id=ai_theme.id,
            alert_type="breakout",
            title="AI infrastructure breakout",
            severity="warning",
            related_tickers=["NVDA", "AVGO"],
            triggered_at=datetime(2026, 4, 4, 15, 30, tzinfo=UTC),
        )
    )

    watchlist = UserWatchlist(name="Core Leaders", description=None, color=None, position=0)
    session.add(watchlist)
    session.commit()
    session.refresh(watchlist)
    session.add_all(
        [
            WatchlistItem(watchlist_id=watchlist.id, symbol="NVDA", position=0),
            WatchlistItem(watchlist_id=watchlist.id, symbol="AMD", position=1),
        ]
    )
    session.commit()


def test_digest_service_builds_daily_digest_and_markdown(session):
    _seed_digest_data(session)
    service = DigestService(validation_service=_FakeValidationService())

    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4))

    assert payload.as_of_date == "2026-04-04"
    assert payload.market.stance == "offense"
    assert [leader.symbol for leader in payload.leaders][:2] == ["NVDA", "AVGO"]
    assert payload.leaders[0].reason_summary.startswith("Strengths led by")
    assert payload.themes.leaders[0].display_name == "AI Infrastructure"
    assert payload.themes.laggards[0].display_name == "Solar"
    assert payload.themes.recent_alerts[0].related_tickers == ["NVDA", "AVGO"]
    assert payload.validation.scan_pick.horizons[1].avg_return_pct == 3.6
    assert payload.watchlists[0].matched_symbols == ["NVDA"]
    assert payload.watchlists[0].alert_symbols == ["NVDA"]
    assert "theme_alert_missing_price_cache" in payload.degraded_reasons
    assert any(risk.kind == "theme_alert_validation" for risk in payload.risks)

    markdown = service.render_markdown(payload)
    assert "# Daily Digest (2026-04-04)" in markdown
    assert "## Leaders" in markdown
    assert "**NVDA**" in markdown
    assert "## Validation Snapshot" in markdown


def test_digest_service_degrades_cleanly_when_sections_are_missing(session):
    service = DigestService(validation_service=_FakeValidationService())

    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4))

    assert payload.market.stance == "unavailable"
    assert payload.leaders == []
    assert payload.themes.leaders == []
    assert payload.watchlists == []
    assert "missing_breadth_snapshot" in payload.degraded_reasons
    assert "missing_published_feature_run" in payload.degraded_reasons
