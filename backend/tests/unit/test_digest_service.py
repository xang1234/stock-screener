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
from app.services.strategy_profile_service import StrategyProfileService


class _FakeValidationService:
    def __init__(self) -> None:
        self.requests: list[tuple[ValidationSourceKind, date | None]] = []

    def get_overview(self, db, *, source_kind, lookback_days, as_of_date=None):  # noqa: ANN001
        self.requests.append((source_kind, as_of_date))
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
    robotics_theme = ThemeCluster(
        name="Robotics",
        canonical_key="robotics",
        display_name="Robotics",
        pipeline="technical",
        category="industrials",
        lifecycle_state="active",
    )
    cybersecurity_theme = ThemeCluster(
        name="Cybersecurity",
        canonical_key="cybersecurity",
        display_name="Cybersecurity",
        pipeline="technical",
        category="technology",
        lifecycle_state="active",
    )
    cloud_theme = ThemeCluster(
        name="Cloud Platforms",
        canonical_key="cloud-platforms",
        display_name="Cloud Platforms",
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
    session.add_all([ai_theme, software_theme, robotics_theme, cybersecurity_theme, cloud_theme, lagging_theme])
    session.commit()
    session.refresh(ai_theme)
    session.refresh(software_theme)
    session.refresh(robotics_theme)
    session.refresh(cybersecurity_theme)
    session.refresh(cloud_theme)
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
                theme_cluster_id=robotics_theme.id,
                date=date(2026, 4, 4),
                pipeline="technical",
                mention_velocity=1.1,
                basket_return_1m=6.7,
                momentum_score=66.0,
                status="trending",
            ),
            ThemeMetrics(
                theme_cluster_id=cybersecurity_theme.id,
                date=date(2026, 4, 4),
                pipeline="technical",
                mention_velocity=1.0,
                basket_return_1m=5.2,
                momentum_score=58.0,
                status="trending",
            ),
            ThemeMetrics(
                theme_cluster_id=cloud_theme.id,
                date=date(2026, 4, 4),
                pipeline="technical",
                mention_velocity=0.8,
                basket_return_1m=2.9,
                momentum_score=45.0,
                status="stable",
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
            triggered_at=datetime(2026, 4, 5, 1, 30, tzinfo=UTC),
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
    fake_validation_service = _FakeValidationService()
    service = DigestService(validation_service=fake_validation_service)

    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4))

    assert fake_validation_service.requests == [
        (ValidationSourceKind.SCAN_PICK, date(2026, 4, 4)),
        (ValidationSourceKind.THEME_ALERT, date(2026, 4, 4)),
    ]
    assert payload.as_of_date == "2026-04-04"
    assert payload.market.stance == "offense"
    assert [leader.symbol for leader in payload.leaders][:2] == ["NVDA", "AVGO"]
    assert payload.leaders[0].reason_summary.startswith("Strengths led by")
    assert payload.themes.leaders[0].display_name == "AI Infrastructure"
    assert payload.themes.laggards[0].display_name == "Solar"
    assert payload.themes.recent_alerts[0].related_tickers == ["NVDA", "AVGO"]
    assert {theme.theme_id for theme in payload.themes.leaders}.isdisjoint(
        {theme.theme_id for theme in payload.themes.laggards}
    )
    assert payload.validation.scan_pick.horizons[1].avg_return_pct == 3.6
    assert payload.watchlists[0].matched_symbols == ["NVDA"]
    assert payload.watchlists[0].alert_symbols == ["NVDA"]
    assert payload.freshness.latest_theme_alert_at == "2026-04-05T01:30:00+00:00"
    assert "theme_alert_missing_price_cache" in payload.degraded_reasons
    assert any(risk.kind == "theme_alert_validation" for risk in payload.risks)

    markdown = service.render_markdown(payload)
    assert "# Daily Digest (2026-04-04)" in markdown
    assert "## Leaders" in markdown
    assert "**NVDA**" in markdown
    assert "## Validation Snapshot" in markdown


def test_digest_service_applies_profile_specific_leader_selection(session):
    _seed_digest_data(session)
    run = session.query(FeatureRun).filter(FeatureRun.as_of_date == date(2026, 4, 4)).first()
    session.add(
        StockFeatureDaily(
            run_id=run.id,
            symbol="NOW",
            as_of_date=run.as_of_date,
            composite_score=79.0,
            overall_rating=4,
            passes_count=1,
            details_json={
                "company_name": "ServiceNow",
                "ibd_industry_group": "Software",
                "screeners_run": ["minervini"],
                "screeners_passed": 1,
                "screeners_total": 1,
                "composite_method": "weighted_average",
                "details": {
                    "screeners": {
                        "minervini": {
                            "score": 79,
                            "passes": True,
                            "rating": "Buy",
                            "breakdown": {
                                "rs_rating": {"points": 18, "max_points": 20, "passes": True},
                            },
                        }
                    }
                },
            },
        )
    )
    session.add(StockUniverse(symbol="NOW", name="ServiceNow", sector="Technology", industry="Software", is_active=True, status="active"))
    session.commit()

    fake_validation_service = _FakeValidationService()
    service = DigestService(validation_service=fake_validation_service)

    default_payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4), profile="default")
    risk_off_payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4), profile="risk_off")

    assert [leader.symbol for leader in default_payload.leaders] == ["NVDA", "AVGO", "MSFT", "NOW"]
    assert [leader.symbol for leader in risk_off_payload.leaders] == ["NVDA", "AVGO", "MSFT"]


def test_digest_service_falls_back_to_ranked_rows_when_threshold_excludes_scored_rows(session):
    _seed_digest_data(session)
    run = session.query(FeatureRun).filter(FeatureRun.as_of_date == date(2026, 4, 4)).first()
    session.add(
        StockFeatureDaily(
            run_id=run.id,
            symbol="NULL",
            as_of_date=run.as_of_date,
            composite_score=None,
            overall_rating=3,
            passes_count=1,
            details_json={
                "company_name": "Null Score Inc",
                "ibd_industry_group": "Software",
                "screeners_run": ["minervini"],
                "screeners_passed": 1,
                "screeners_total": 1,
                "composite_method": "weighted_average",
                "details": {"screeners": {"minervini": {"score": None, "passes": True, "rating": "Hold"}}},
            },
        )
    )
    session.commit()

    profile_service = StrategyProfileService()
    profile_service._registry["risk_off"].digest.leader_min_composite_score = 99.0
    service = DigestService(
        validation_service=_FakeValidationService(),
        profile_service=profile_service,
    )

    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4), profile="risk_off")

    assert [leader.symbol for leader in payload.leaders] == ["NVDA", "AVGO", "MSFT"]


def test_digest_service_falls_back_when_profile_theme_sort_is_invalid(session, caplog):
    _seed_digest_data(session)
    profile_service = StrategyProfileService()
    profile_service._registry["default"].digest.theme_sort = "not_a_real_metric"
    service = DigestService(
        validation_service=_FakeValidationService(),
        profile_service=profile_service,
    )

    with caplog.at_level("WARNING"):
        payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4), profile="default")

    assert payload.themes.leaders[0].display_name == "AI Infrastructure"
    assert "Unknown digest theme_sort 'not_a_real_metric'" in caplog.text


def test_digest_service_excludes_null_theme_metrics_from_leaders_and_laggards(session):
    _seed_digest_data(session)
    theme = session.query(ThemeCluster).filter(ThemeCluster.display_name == "Solar").first()
    metrics = session.query(ThemeMetrics).filter(ThemeMetrics.theme_cluster_id == theme.id).first()
    metrics.momentum_score = None
    session.commit()

    service = DigestService(validation_service=_FakeValidationService())
    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4))

    leader_names = [item.display_name for item in payload.themes.leaders]
    laggard_names = [item.display_name for item in payload.themes.laggards]
    assert "Solar" not in leader_names
    assert "Solar" not in laggard_names


def test_digest_service_degrades_cleanly_when_sections_are_missing(session):
    fake_validation_service = _FakeValidationService()
    service = DigestService(validation_service=fake_validation_service)

    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4))

    assert fake_validation_service.requests == [
        (ValidationSourceKind.SCAN_PICK, date(2026, 4, 4)),
        (ValidationSourceKind.THEME_ALERT, date(2026, 4, 4)),
    ]
    assert payload.market.stance == "unavailable"
    assert payload.leaders == []
    assert payload.themes.leaders == []
    assert payload.watchlists == []
    assert "missing_breadth_snapshot" in payload.degraded_reasons
    assert "missing_published_feature_run" in payload.degraded_reasons


def test_digest_service_reports_theme_alert_freshness_outside_recent_display_window(session):
    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(theme)
    session.add(
        ThemeMetrics(
            theme_cluster_id=theme.id,
            date=date(2026, 4, 4),
            pipeline="technical",
            mention_velocity=1.0,
            basket_return_1m=5.0,
            momentum_score=50.0,
            status="stable",
        )
    )
    session.add(
        ThemeAlert(
            theme_cluster_id=theme.id,
            alert_type="breakout",
            title="Older alert",
            severity="info",
            related_tickers=["NVDA"],
            triggered_at=datetime(2026, 3, 20, 14, 0, tzinfo=UTC),
        )
    )
    session.commit()

    fake_validation_service = _FakeValidationService()
    service = DigestService(validation_service=fake_validation_service)

    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4))

    assert fake_validation_service.requests == [
        (ValidationSourceKind.SCAN_PICK, date(2026, 4, 4)),
        (ValidationSourceKind.THEME_ALERT, date(2026, 4, 4)),
    ]
    assert payload.themes.recent_alerts == []
    assert payload.freshness.latest_theme_alert_at == "2026-03-20T14:00:00+00:00"
    assert "missing_recent_theme_alerts" in payload.degraded_reasons


def test_digest_service_recent_alert_window_covers_exactly_seven_eastern_days(session):
    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(theme)
    session.add(
        ThemeMetrics(
            theme_cluster_id=theme.id,
            date=date(2026, 4, 4),
            pipeline="technical",
            mention_velocity=1.0,
            basket_return_1m=5.0,
            momentum_score=50.0,
            status="stable",
        )
    )
    session.add_all(
        [
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="breakout",
                title="Inside window",
                severity="info",
                related_tickers=["NVDA"],
                triggered_at=datetime(2026, 3, 29, 15, 0, tzinfo=UTC),
            ),
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="breakout",
                title="Too old",
                severity="info",
                related_tickers=["AVGO"],
                triggered_at=datetime(2026, 3, 28, 15, 0, tzinfo=UTC),
            ),
        ]
    )
    session.commit()

    fake_validation_service = _FakeValidationService()
    service = DigestService(validation_service=fake_validation_service)

    payload = service.get_daily_digest(session, as_of_date=date(2026, 4, 4))

    assert fake_validation_service.requests == [
        (ValidationSourceKind.SCAN_PICK, date(2026, 4, 4)),
        (ValidationSourceKind.THEME_ALERT, date(2026, 4, 4)),
    ]
    assert [alert.title for alert in payload.themes.recent_alerts] == ["Inside window"]
