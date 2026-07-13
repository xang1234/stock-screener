"""Integration tests for feature store → ScanResultItemDomain bridge.

Uses real SQLAlchemy session with in-memory SQLite to verify:
- query_run_as_scan_results() returns correct ScanResultItemDomain objects
- JSON extraction paths work for all filter types
- LEFT JOIN with StockUniverse correctly resolves company_name
- INT_TO_RATING mapping produces correct rating strings
- Sort, pagination, and sparkline suppression work correctly
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.domain.common.query import (
    CategoricalFilter,
    FilterExpression,
    FilterGroup,
    FilterSpec,
    MatchOperator,
    PageSpec,
    QuerySpec,
    RangeFilter,
    SortOrder,
    SortSpec,
    TextSearchFilter,
)
from app.domain.scanning.models import ResultPage, ScanResultItemDomain
from app.infra.db.models.feature_store import (
    FeatureRun,
    StockFeatureDaily,
)
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.models.stock_universe import StockUniverse

AS_OF = date(2026, 2, 17)


@pytest.fixture
def session():
    """Create an in-memory SQLite database with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


@pytest.fixture
def seeded_session(session):
    """Seed the database with a feature run, feature rows, and universe data."""
    # Create a feature run
    run = FeatureRun(
        id=1,
        as_of_date=AS_OF,
        run_type="daily_snapshot",
        status="published",
    )
    session.add(run)

    # Add universe entries for company name lookup
    session.add(StockUniverse(symbol="AAPL", name="Apple Inc", exchange="NASDAQ"))
    session.add(StockUniverse(symbol="MSFT", name="Microsoft Corp", exchange="NASDAQ"))
    session.add(StockUniverse(symbol="GOOGL", name="Alphabet Inc", exchange="NASDAQ"))

    # Add feature rows with full orchestrator-style details
    rows = [
        StockFeatureDaily(
            run_id=1,
            symbol="AAPL",
            as_of_date=AS_OF,
            composite_score=92.5,
            overall_rating=5,  # Strong Buy
            passes_count=2,
            details_json={
                "composite_score": 92.5,
                "rating": "Strong Buy",
                "current_price": 185.50,
                "minervini_score": 88.0,
                "canslim_score": 90.0,
                "rs_rating": 95.0,
                "stage": 2,
                "gics_sector": "Technology",
                "ibd_industry_group": "Comp-Peripherals",
                "screeners_run": ["minervini", "canslim"],
                "composite_method": "weighted_average",
                "screeners_passed": 2,
                "screeners_total": 2,
                "rs_sparkline_data": [80, 85, 90, 95],
                "price_sparkline_data": [170, 175, 180, 185],
                "adr_percent": 2.5,
                "passes_template": True,
                "from_52w_high_pct": -5.0,
                "above_52w_low_pct": 45.0,
            },
        ),
        StockFeatureDaily(
            run_id=1,
            symbol="MSFT",
            as_of_date=AS_OF,
            composite_score=78.0,
            overall_rating=4,  # Buy
            passes_count=1,
            details_json={
                "composite_score": 78.0,
                "rating": "Buy",
                "current_price": 420.00,
                "minervini_score": 75.0,
                "canslim_score": 65.0,
                "rs_rating": 80.0,
                "stage": 2,
                "gics_sector": "Technology",
                "ibd_industry_group": "Comp-Software",
                "screeners_run": ["minervini"],
                "composite_method": "weighted_average",
                "screeners_passed": 1,
                "screeners_total": 2,
                "adr_percent": 1.8,
                "passes_template": True,
            },
        ),
        StockFeatureDaily(
            run_id=1,
            symbol="GOOGL",
            as_of_date=AS_OF,
            composite_score=55.0,
            overall_rating=3,  # Watch
            passes_count=0,
            details_json={
                "composite_score": 55.0,
                "rating": "Watch",
                "current_price": 150.00,
                "minervini_score": 50.0,
                "rs_rating": 60.0,
                "stage": 3,
                "gics_sector": "Communication Services",
                "ibd_industry_group": "Internet-Content",
                "screeners_run": ["minervini"],
                "composite_method": "weighted_average",
                "screeners_passed": 0,
                "screeners_total": 1,
                "adr_percent": 1.5,
                "passes_template": False,
            },
        ),
    ]
    session.add_all(rows)
    session.commit()
    return session


class TestQueryRunAsScanResults:
    """Integration tests for the feature store → scan result bridge."""

    def test_returns_result_page_with_scan_result_items(self, seeded_session):
        repo = SqlFeatureStoreRepository(seeded_session)

        page = repo.query_run_as_scan_results(1, QuerySpec())

        assert isinstance(page, ResultPage)
        assert page.total == 3
        assert len(page.items) == 3
        for item in page.items:
            assert isinstance(item, ScanResultItemDomain)

    @pytest.mark.parametrize(
        ("group_join", "expected"),
        [
            (MatchOperator.ANY, {"AAPL", "MSFT"}),
            (MatchOperator.ALL, {"AAPL"}),
        ],
    )
    def test_grouped_expression_compiles_across_json_and_columns(
        self, seeded_session, group_join, expected
    ):
        repo = SqlFeatureStoreRepository(seeded_session)
        expression = FilterExpression(
            required=FilterGroup(
                id="required",
                name="Always require",
                conditions=(RangeFilter("stage", min_value=2, max_value=2),),
            ),
            group_join=group_join,
            groups=(
                FilterGroup(
                    id="leadership",
                    name="Leadership",
                    conditions=(RangeFilter("rs_rating", min_value=90),),
                ),
                FilterGroup(
                    id="score",
                    name="Score",
                    conditions=(RangeFilter("composite_score", min_value=75),),
                ),
            ),
        )

        page = repo.query_run_as_scan_results(
            1,
            QuerySpec(expression=expression),
        )

        assert {str(item.symbol) for item in page.items} == expected

    def test_company_name_resolved_via_join(self, seeded_session):
        repo = SqlFeatureStoreRepository(seeded_session)

        page = repo.query_run_as_scan_results(1, QuerySpec())

        names = {
            item.symbol: item.extended_fields["company_name"]
            for item in page.items
        }
        assert names["AAPL"] == "Apple Inc"
        assert names["MSFT"] == "Microsoft Corp"
        assert names["GOOGL"] == "Alphabet Inc"

    def test_listing_discovery_searches_company_and_preserves_listing_only_rows(
        self, seeded_session
    ):
        seeded_session.add(
            StockFeatureDaily(
                run_id=1,
                symbol="NEWCO",
                as_of_date=AS_OF,
                composite_score=60,
                overall_rating=3,
                passes_count=0,
                details_json={
                    "rating": "Watch",
                    "current_price": 20,
                    "scan_mode": "listing_only",
                },
            )
        )
        seeded_session.commit()
        expression = FilterExpression(
            required=FilterGroup(
                id="required",
                name="Always require",
                conditions=(
                    TextSearchFilter("listing_search", "newco"),
                    RangeFilter("discovery_volume", min_value=1_000_000),
                ),
            )
        )

        page = SqlFeatureStoreRepository(seeded_session).query_run_as_scan_results(
            1,
            QuerySpec(expression=expression),
        )

        assert [str(item.symbol) for item in page.items] == ["NEWCO"]
        assert page.items[0].extended_fields["passes_template"] is None

    def test_int_to_rating_mapping(self, seeded_session):
        repo = SqlFeatureStoreRepository(seeded_session)

        page = repo.query_run_as_scan_results(1, QuerySpec())

        ratings = {item.symbol: item.rating for item in page.items}
        assert ratings["AAPL"] == "Strong Buy"
        assert ratings["MSFT"] == "Buy"
        assert ratings["GOOGL"] == "Watch"

    def test_score_clamping(self, seeded_session):
        repo = SqlFeatureStoreRepository(seeded_session)

        page = repo.query_run_as_scan_results(1, QuerySpec())

        for item in page.items:
            assert 0.0 <= item.composite_score <= 100.0

    def test_extended_fields_populated(self, seeded_session):
        repo = SqlFeatureStoreRepository(seeded_session)

        page = repo.query_run_as_scan_results(1, QuerySpec())

        aapl = next(i for i in page.items if i.symbol == "AAPL")
        ef = aapl.extended_fields
        assert ef["minervini_score"] == 88.0
        assert ef["canslim_score"] == 90.0
        assert ef["rs_rating"] == 95.0
        assert ef["stage"] == 2
        assert ef["gics_sector"] == "Technology"
        assert ef["adr_percent"] == 2.5
        assert ef["passes_template"] is True
        assert ef["week_52_high_distance"] == -5.0
        assert ef["week_52_low_distance"] == 45.0

    def test_range_filter_on_json_field(self, seeded_session):
        """Range filter on rs_rating (JSON field) works correctly."""
        repo = SqlFeatureStoreRepository(seeded_session)

        spec = QuerySpec.from_filter_spec(
            FilterSpec(
                range_filters=[RangeFilter(field="rs_rating", min_value=75.0)]
            )
        )
        page = repo.query_run_as_scan_results(1, spec)

        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "MSFT"}  # GOOGL has rs_rating=60

    def test_categorical_filter_on_json_field(self, seeded_session):
        """Categorical filter on gics_sector (JSON field) works correctly."""
        repo = SqlFeatureStoreRepository(seeded_session)

        spec = QuerySpec.from_filter_spec(
            FilterSpec(
                categorical_filters=[
                    CategoricalFilter(
                        field="gics_sector",
                        values=("Technology",),
                    )
                ]
            )
        )
        page = repo.query_run_as_scan_results(1, spec)

        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "MSFT"}

    def test_text_search_on_json_field(self, seeded_session):
        """Text search on ibd_industry_group (JSON field) works correctly."""
        repo = SqlFeatureStoreRepository(seeded_session)

        spec = QuerySpec.from_filter_spec(
            FilterSpec(
                text_searches=[
                    TextSearchFilter(field="ibd_industry_group", pattern="Comp")
                ]
            )
        )
        page = repo.query_run_as_scan_results(1, spec)

        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "MSFT"}  # Comp-Peripherals, Comp-Software

    def test_sort_by_json_field(self, seeded_session):
        """Sorting by a JSON field (rs_rating) produces correct order."""
        repo = SqlFeatureStoreRepository(seeded_session)

        spec = QuerySpec(
            sort=SortSpec(field="rs_rating", order=SortOrder.ASC),
        )
        page = repo.query_run_as_scan_results(1, spec)

        symbols = [item.symbol for item in page.items]
        assert symbols == ["GOOGL", "MSFT", "AAPL"]

    def test_pagination(self, seeded_session):
        """Pagination returns correct page/total counts."""
        repo = SqlFeatureStoreRepository(seeded_session)

        spec = QuerySpec(page=PageSpec(page=1, per_page=2))
        page = repo.query_run_as_scan_results(1, spec)

        assert page.total == 3
        assert len(page.items) == 2
        assert page.page == 1
        assert page.per_page == 2
        assert page.total_pages == 2

    def test_sparkline_suppression(self, seeded_session):
        """include_sparklines=False removes sparkline arrays."""
        repo = SqlFeatureStoreRepository(seeded_session)

        page = repo.query_run_as_scan_results(
            1, QuerySpec(), include_sparklines=False
        )

        aapl = next(i for i in page.items if i.symbol == "AAPL")
        assert aapl.extended_fields["rs_sparkline_data"] is None
        assert aapl.extended_fields["price_sparkline_data"] is None

    def test_sparklines_included_by_default(self, seeded_session):
        """include_sparklines=True (default) includes sparkline arrays."""
        repo = SqlFeatureStoreRepository(seeded_session)

        page = repo.query_run_as_scan_results(1, QuerySpec())

        aapl = next(i for i in page.items if i.symbol == "AAPL")
        assert aapl.extended_fields["rs_sparkline_data"] == [80, 85, 90, 95]
        assert aapl.extended_fields["price_sparkline_data"] == [170, 175, 180, 185]

    def test_missing_run_raises_not_found(self, seeded_session):
        """Querying a non-existent run raises EntityNotFoundError."""
        from app.domain.common.errors import EntityNotFoundError

        repo = SqlFeatureStoreRepository(seeded_session)

        with pytest.raises(EntityNotFoundError, match="FeatureRun"):
            repo.query_run_as_scan_results(9999, QuerySpec())

    def test_symbol_without_universe_entry(self, seeded_session):
        """Symbol not in stock_universe gets company_name=None."""
        # Add a feature row for a symbol not in stock_universe
        seeded_session.add(
            StockFeatureDaily(
                run_id=1,
                symbol="NEWCO",
                as_of_date=AS_OF,
                composite_score=60.0,
                overall_rating=3,
                passes_count=0,
                details_json={
                    "composite_score": 60.0,
                    "rating": "Watch",
                    "current_price": 25.0,
                    "screeners_run": ["minervini"],
                    "screeners_passed": 0,
                    "screeners_total": 1,
                },
            )
        )
        seeded_session.commit()

        repo = SqlFeatureStoreRepository(seeded_session)
        spec = QuerySpec.from_filter_spec(
            FilterSpec(
                text_searches=[TextSearchFilter(field="symbol", pattern="NEWCO")]
            )
        )
        page = repo.query_run_as_scan_results(1, spec)

        assert len(page.items) == 1
        assert page.items[0].symbol == "NEWCO"
        assert page.items[0].extended_fields["company_name"] is None
