"""Shared fixtures for Hermes Market Copilot tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
from app.models.industry import IBDGroupRank
from app.models.market_breadth import MarketBreadth
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse
from app.models.task_execution import TaskExecutionHistory
from app.models.theme import ThemeAlert, ThemeCluster, ThemeConstituent, ThemeMetrics
from app.models.user_watchlist import UserWatchlist, WatchlistItem

_TABLES = (
    FeatureRun,
    FeatureRunPointer,
    StockFeatureDaily,
    StockUniverse,
    StockFundamental,
    UserWatchlist,
    WatchlistItem,
    ThemeCluster,
    ThemeConstituent,
    ThemeMetrics,
    ThemeAlert,
    MarketBreadth,
    TaskExecutionHistory,
    IBDGroupRank,
)


def create_mcp_test_session_factory(database_url: str = "sqlite://"):
    """Create a SQLAlchemy session factory with only the tables MCP tests need."""

    engine_kwargs: dict = {}
    if database_url in {"sqlite://", "sqlite:///:memory:"}:
        engine_kwargs = {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
    elif database_url.startswith("sqlite:///"):
        engine_kwargs = {"connect_args": {"check_same_thread": False}}

    engine = create_engine(database_url, **engine_kwargs)
    for model in _TABLES:
        model.__table__.create(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False), engine


def seed_market_copilot_data(session_factory: sessionmaker) -> None:
    """Populate a compact, deterministic dataset for MCP tests."""

    session = session_factory()
    try:
        for symbol, name, sector, industry, market_cap in (
            ("NVDA", "NVIDIA Corporation", "Information Technology", "Semiconductors", 3_000_000_000_000),
            ("AVGO", "Broadcom Inc.", "Information Technology", "Semiconductors", 900_000_000_000),
            ("PANW", "Palo Alto Networks", "Information Technology", "Cybersecurity", 120_000_000_000),
            ("MSFT", "Microsoft Corporation", "Information Technology", "Software", 3_200_000_000_000),
            ("AAPL", "Apple Inc.", "Information Technology", "Consumer Electronics", 2_700_000_000_000),
            ("SNOW", "Snowflake Inc.", "Information Technology", "Cloud Software", 60_000_000_000),
        ):
            session.add(
                StockUniverse(
                    symbol=symbol,
                    name=name,
                    sector=sector,
                    industry=industry,
                    market_cap=market_cap,
                    exchange="NASDAQ",
                    is_active=True,
                    status="active",
                    source="fixture",
                )
            )

        run_one = FeatureRun(
            id=1,
            as_of_date=date(2026, 3, 28),
            run_type="daily_snapshot",
            status="published",
            created_at=datetime(2026, 3, 28, 20, 0, tzinfo=UTC),
            completed_at=datetime(2026, 3, 28, 21, 0, tzinfo=UTC),
            published_at=datetime(2026, 3, 28, 21, 30, tzinfo=UTC),
        )
        run_two = FeatureRun(
            id=2,
            as_of_date=date(2026, 3, 29),
            run_type="daily_snapshot",
            status="published",
            created_at=datetime(2026, 3, 29, 20, 0, tzinfo=UTC),
            completed_at=datetime(2026, 3, 29, 21, 0, tzinfo=UTC),
            published_at=datetime(2026, 3, 29, 21, 30, tzinfo=UTC),
        )
        session.add_all([run_one, run_two, FeatureRunPointer(key="latest_published", run_id=2)])

        for run_id, as_of_date, symbol, score, rating, details in (
            (1, date(2026, 3, 28), "NVDA", 88.0, 5, _details("NVDA", 88.0, "Strong Buy", 2, 92, "Semiconductors", 140.0, 25_000_000, 3_000_000_000_000, 34.0, 28.0)),
            (1, date(2026, 3, 28), "AAPL", 76.0, 4, _details("AAPL", 76.0, "Buy", 2, 85, "Consumer Electronics", 210.0, 18_000_000, 2_700_000_000_000, 18.0, 14.0)),
            (1, date(2026, 3, 28), "AVGO", 81.0, 4, _details("AVGO", 81.0, "Buy", 2, 89, "Semiconductors", 1_120.0, 12_000_000, 900_000_000_000, 22.0, 17.0)),
            (1, date(2026, 3, 28), "PANW", 79.0, 4, _details("PANW", 79.0, "Buy", 2, 87, "Cybersecurity", 325.0, 8_000_000, 120_000_000_000, 20.0, 19.0)),
            (1, date(2026, 3, 28), "SNOW", 60.0, 3, _details("SNOW", 60.0, "Watch", 1, 70, "Cloud Software", 182.0, 6_000_000, 60_000_000_000, 9.0, 11.0)),
            (2, date(2026, 3, 29), "NVDA", 92.0, 5, _details("NVDA", 92.0, "Strong Buy", 2, 95, "Semiconductors", 145.0, 26_000_000, 3_000_000_000_000, 36.0, 30.0)),
            (2, date(2026, 3, 29), "MSFT", 82.0, 4, _details("MSFT", 82.0, "Buy", 2, 90, "Software", 430.0, 14_000_000, 3_200_000_000_000, 19.0, 16.0)),
            (2, date(2026, 3, 29), "AVGO", 83.0, 4, _details("AVGO", 83.0, "Buy", 2, 90, "Semiconductors", 1_135.0, 13_000_000, 900_000_000_000, 24.0, 18.0)),
            (2, date(2026, 3, 29), "PANW", 84.0, 5, _details("PANW", 84.0, "Strong Buy", 2, 91, "Cybersecurity", 332.0, 8_500_000, 120_000_000_000, 23.0, 20.0)),
            (2, date(2026, 3, 29), "SNOW", 55.0, 3, _details("SNOW", 55.0, "Watch", 1, 67, "Cloud Software", 176.0, 5_500_000, 60_000_000_000, 7.0, 9.0)),
        ):
            session.add(
                StockFeatureDaily(
                    run_id=run_id,
                    symbol=symbol,
                    as_of_date=as_of_date,
                    composite_score=score,
                    overall_rating=rating,
                    passes_count=2 if rating >= 4 else 1,
                    details_json=details,
                )
            )

        leaders = UserWatchlist(id=1, name="Leaders", description="Leadership names", color="#00838f", position=0)
        breakouts = UserWatchlist(id=2, name="Breakouts", description="Setup focus list", color="#2e7d32", position=1)
        session.add_all([leaders, breakouts])
        session.add_all(
            [
                WatchlistItem(watchlist_id=1, symbol="NVDA", display_name="NVIDIA Corporation", position=0),
                WatchlistItem(watchlist_id=1, symbol="AAPL", display_name="Apple Inc.", position=1),
                WatchlistItem(watchlist_id=2, symbol="PANW", display_name="Palo Alto Networks", position=0),
                WatchlistItem(watchlist_id=2, symbol="AVGO", display_name="Broadcom Inc.", position=1),
            ]
        )

        ai_theme = ThemeCluster(
            id=1,
            name="artificial intelligence",
            canonical_key="artificial-intelligence",
            display_name="AI Infrastructure",
            pipeline="technical",
            category="technology",
            is_emerging=False,
            is_validated=True,
            is_active=True,
            lifecycle_state="active",
            first_seen_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
            last_seen_at=datetime(2026, 3, 29, 19, 0, tzinfo=UTC),
        )
        cyber_theme = ThemeCluster(
            id=2,
            name="cybersecurity",
            canonical_key="cybersecurity",
            display_name="Cybersecurity",
            pipeline="technical",
            category="technology",
            is_emerging=False,
            is_validated=True,
            is_active=True,
            lifecycle_state="active",
            first_seen_at=datetime(2026, 1, 20, 10, 0, tzinfo=UTC),
            last_seen_at=datetime(2026, 3, 29, 18, 0, tzinfo=UTC),
        )
        session.add_all([ai_theme, cyber_theme])
        session.add_all(
            [
                ThemeConstituent(theme_cluster_id=1, symbol="NVDA", source="llm_extraction", confidence=0.98, mention_count=14, is_active=True),
                ThemeConstituent(theme_cluster_id=1, symbol="AVGO", source="llm_extraction", confidence=0.92, mention_count=11, is_active=True),
                ThemeConstituent(theme_cluster_id=2, symbol="PANW", source="llm_extraction", confidence=0.94, mention_count=9, is_active=True),
            ]
        )
        session.add_all(
            [
                ThemeMetrics(
                    theme_cluster_id=1,
                    date=date(2026, 3, 29),
                    pipeline="technical",
                    mentions_7d=18,
                    mention_velocity=1.6,
                    sentiment_score=0.8,
                    basket_rs_vs_spy=89.0,
                    num_constituents=2,
                    avg_rs_rating=92.5,
                    momentum_score=94.0,
                    rank=1,
                    status="trending",
                ),
                ThemeMetrics(
                    theme_cluster_id=2,
                    date=date(2026, 3, 29),
                    pipeline="technical",
                    mentions_7d=10,
                    mention_velocity=1.2,
                    sentiment_score=0.6,
                    basket_rs_vs_spy=81.0,
                    num_constituents=1,
                    avg_rs_rating=91.0,
                    momentum_score=84.0,
                    rank=2,
                    status="trending",
                ),
            ]
        )
        session.add(
            ThemeAlert(
                id=1,
                theme_cluster_id=1,
                alert_type="velocity_spike",
                title="AI Infrastructure velocity spike",
                description="Mentions and RS accelerated together.",
                severity="warning",
                related_tickers=["NVDA", "AVGO"],
                metrics={"momentum_score": 94.0},
                is_read=False,
                is_dismissed=False,
                triggered_at=datetime(2026, 3, 29, 22, 15, tzinfo=UTC),
            )
        )

        session.add_all(
            [
                MarketBreadth(
                    date=date(2026, 3, 29),
                    stocks_up_4pct=128,
                    stocks_down_4pct=34,
                    ratio_5day=1.82,
                    ratio_10day=1.54,
                    total_stocks_scanned=4081,
                ),
                MarketBreadth(
                    date=date(2026, 3, 28),
                    stocks_up_4pct=112,
                    stocks_down_4pct=41,
                    ratio_5day=1.70,
                    ratio_10day=1.48,
                    total_stocks_scanned=4075,
                ),
                MarketBreadth(
                    date=date(2026, 3, 27),
                    stocks_up_4pct=98,
                    stocks_down_4pct=52,
                    ratio_5day=1.55,
                    ratio_10day=1.42,
                    total_stocks_scanned=4070,
                ),
            ]
        )

        # IBD Group Rankings (2 groups, 2 dates for mover calculation)
        session.add_all(
            [
                IBDGroupRank(
                    industry_group="Semiconductors",
                    date=date(2026, 3, 29),
                    rank=1,
                    avg_rs_rating=91.0,
                    num_stocks=25,
                    top_symbol="NVDA",
                    top_rs_rating=95.0,
                ),
                IBDGroupRank(
                    industry_group="Cybersecurity",
                    date=date(2026, 3, 29),
                    rank=2,
                    avg_rs_rating=87.0,
                    num_stocks=15,
                    top_symbol="PANW",
                    top_rs_rating=91.0,
                ),
                IBDGroupRank(
                    industry_group="Semiconductors",
                    date=date(2026, 3, 22),
                    rank=3,
                    avg_rs_rating=85.0,
                    num_stocks=25,
                    top_symbol="NVDA",
                    top_rs_rating=92.0,
                ),
                IBDGroupRank(
                    industry_group="Cybersecurity",
                    date=date(2026, 3, 22),
                    rank=5,
                    avg_rs_rating=80.0,
                    num_stocks=15,
                    top_symbol="PANW",
                    top_rs_rating=87.0,
                ),
            ]
        )
        session.add_all(
            [
                TaskExecutionHistory(
                    task_name="daily-smart-refresh",
                    task_function="app.tasks.cache_tasks.smart_refresh_cache",
                    task_id="task-ok",
                    status="completed",
                    started_at=datetime(2026, 3, 29, 21, 30, tzinfo=UTC),
                    completed_at=datetime(2026, 3, 29, 21, 45, tzinfo=UTC),
                    duration_seconds=900.0,
                    triggered_by="schedule",
                ),
                TaskExecutionHistory(
                    task_name="daily-group-ranking-calculation",
                    task_function="app.tasks.group_rank_tasks.calculate_daily_group_rankings",
                    task_id="task-fail",
                    status="failed",
                    started_at=datetime(2026, 3, 29, 21, 40, tzinfo=UTC),
                    completed_at=datetime(2026, 3, 29, 21, 42, tzinfo=UTC),
                    duration_seconds=120.0,
                    triggered_by="schedule",
                    error_message="Fixture failure for MCP tests",
                ),
            ]
        )

        session.commit()
    finally:
        session.close()


def _details(
    symbol: str,
    composite_score: float,
    rating: str,
    stage: int,
    rs_rating: int,
    industry_group: str,
    current_price: float,
    avg_dollar_volume: int,
    market_cap: int,
    eps_growth_qq: float,
    sales_growth_qq: float,
) -> dict:
    return {
        "symbol": symbol,
        "rating": rating,
        "composite_score": composite_score,
        "current_price": current_price,
        "avg_dollar_volume": avg_dollar_volume,
        "market_cap": market_cap,
        "stage": stage,
        "stage_name": f"Stage {stage}",
        "rs_rating": rs_rating,
        "eps_growth_qq": eps_growth_qq,
        "sales_growth_qq": sales_growth_qq,
        "ibd_industry_group": industry_group,
        "gics_sector": "Information Technology",
        "screeners_run": ["minervini", "canslim"],
        "screeners_passed": 2 if rating in {"Strong Buy", "Buy"} else 1,
        "screeners_total": 2,
        "composite_method": "weighted_average",
        "details": {
            "screeners": {
                "minervini": {
                    "score": composite_score,
                    "passes": True,
                    "rating": rating,
                    "breakdown": {
                        "rs_rating": {"points": 20, "max_points": 20, "passes": True},
                        "stage": {"points": 20 if stage == 2 else 5, "max_points": 20, "passes": stage == 2},
                        "ma_alignment": {"points": 15, "max_points": 15, "passes": True},
                    },
                    "details": {},
                },
                "canslim": {
                    "score": max(composite_score - 6, 0),
                    "passes": rating in {"Strong Buy", "Buy"},
                    "rating": "Buy" if rating == "Strong Buy" else rating,
                    "breakdown": {
                        "current_earnings": 20,
                        "annual_earnings": 15,
                        "leader": 20,
                    },
                    "details": {},
                },
            }
        },
        "setup_engine": {
            "setup_score": composite_score - 5,
            "quality_score": composite_score - 10,
            "readiness_score": composite_score - 3,
            "pattern_primary": "VCP",
            "pivot_price": round(current_price * 1.04, 2),
            "distance_to_pivot_pct": 4.0,
            "setup_ready": rating in {"Strong Buy", "Buy"},
            "explain": {"thesis": f"{symbol} is acting constructively in the fixture run."},
            "candidates": [{"pattern": "VCP", "score": composite_score - 5}],
        },
    }
