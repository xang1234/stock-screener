"""Golden parity tests: legacy scan_results vs feature store.

Both pipelines consume the same orchestrator output dict but persist/read
through different paths.  These tests seed an in-memory SQLite database via
the *real* ORM write helpers, then read back through the *real* repository
query methods and assert the resulting ``ScanResultItemDomain`` objects are
field-for-field identical.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.domain.common.query import (
    CategoricalFilter,
    FilterSpec,
    PageSpec,
    RangeFilter,
    SortOrder,
    SortSpec,
    TextSearchFilter,
)
from app.domain.scanning.filter_expression_model import (
    FilterExpression,
    FilterGroup,
    MatchOperator,
    QuerySpec,
)
from app.domain.scanning.models import ResultPage, ScanResultItemDomain
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.infra.db.repositories.scan_result_repo import SqlScanResultRepository
from app.models.stock_universe import StockUniverse

from .conftest import FEATURE_RUN_ID, LEGACY_SCAN_ID
from .golden_fixtures import GOLDEN_TICKERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_legacy(session: Session, spec: QuerySpec | None = None) -> ResultPage:
    repo = SqlScanResultRepository(session)
    return repo.query(LEGACY_SCAN_ID, spec or QuerySpec())


def _query_feature(session: Session, spec: QuerySpec | None = None) -> ResultPage:
    repo = SqlFeatureStoreRepository(session)
    return repo.query_run_as_scan_results(FEATURE_RUN_ID, spec or QuerySpec())


def _find(page: ResultPage, symbol: str) -> ScanResultItemDomain:
    for item in page.items:
        if item.symbol == symbol:
            return item
    raise ValueError(f"{symbol} not found in page ({[i.symbol for i in page.items]})")


def _compare(
    mismatches: list[str],
    field: str,
    legacy_val: Any,
    feature_val: Any,
    tolerance: float | None = None,
) -> None:
    """Compare two values, appending a mismatch string if they differ."""
    if legacy_val is None and feature_val is None:
        return
    if tolerance is not None and isinstance(legacy_val, (int, float)) and isinstance(feature_val, (int, float)):
        if abs(float(legacy_val) - float(feature_val)) > tolerance:
            mismatches.append(
                f"{field}: legacy={legacy_val!r} vs fs={feature_val!r} (delta={abs(float(legacy_val) - float(feature_val)):.6f})"
            )
        return
    if legacy_val != feature_val:
        mismatches.append(f"{field}: legacy={legacy_val!r} vs fs={feature_val!r}")


def assert_scan_result_parity(
    legacy: ScanResultItemDomain,
    feature_store: ScanResultItemDomain,
    tolerance: float = 0.01,
) -> None:
    """Assert two ScanResultItemDomain objects are equivalent.

    Collects ALL mismatches before failing so a single run reveals every
    drifted field — no fix-one-rerun-fix-another cycle.
    """
    mismatches: list[str] = []

    # Core scalar fields
    _compare(mismatches, "composite_score", legacy.composite_score, feature_store.composite_score, tolerance)
    _compare(mismatches, "rating", legacy.rating, feature_store.rating)
    _compare(mismatches, "current_price", legacy.current_price, feature_store.current_price, tolerance)
    _compare(mismatches, "screeners_run", legacy.screeners_run, feature_store.screeners_run)
    _compare(mismatches, "composite_method", legacy.composite_method, feature_store.composite_method)
    _compare(mismatches, "screeners_passed", legacy.screeners_passed, feature_store.screeners_passed)
    _compare(mismatches, "screeners_total", legacy.screeners_total, feature_store.screeners_total)
    _compare(mismatches, "matched_groups", legacy.matched_groups, feature_store.matched_groups)

    # Extended fields — compare every key present in either side
    all_keys = set(legacy.extended_fields) | set(feature_store.extended_fields)
    for key in sorted(all_keys):
        lv = legacy.extended_fields.get(key)
        fv = feature_store.extended_fields.get(key)
        tol = tolerance if isinstance(lv, (int, float)) or isinstance(fv, (int, float)) else None
        _compare(mismatches, f"extended_fields[{key}]", lv, fv, tol)

    if mismatches:
        report = f"\n{legacy.symbol} parity failures ({len(mismatches)}):\n"
        for m in mismatches:
            report += f"  {m}\n"
        pytest.fail(report)


# ═══════════════════════════════════════════════════════════════════════════
# Test Class 1: Schema alignment
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaAlignment:
    """Detect extended_fields key drift between the two mappers."""

    def test_extended_field_keys_match(self, seeded_session: Session):
        """Both paths must produce exactly the same set of extended_fields keys."""
        legacy_page = _query_legacy(seeded_session)
        feature_page = _query_feature(seeded_session)

        # Pick the first ticker from each (both have all 20)
        legacy_item = legacy_page.items[0]
        feature_item = _find(feature_page, legacy_item.symbol)

        legacy_keys = set(legacy_item.extended_fields.keys())
        feature_keys = set(feature_item.extended_fields.keys())

        only_legacy = legacy_keys - feature_keys
        only_feature = feature_keys - legacy_keys

        if only_legacy or only_feature:
            msg = "extended_fields key mismatch:\n"
            if only_legacy:
                msg += f"  Only in legacy: {sorted(only_legacy)}\n"
            if only_feature:
                msg += f"  Only in feature store: {sorted(only_feature)}\n"
            pytest.fail(msg)


# ═══════════════════════════════════════════════════════════════════════════
# Test Class 2: Field-by-field parity (parametrized per ticker)
# ═══════════════════════════════════════════════════════════════════════════


class TestFieldByFieldParity:
    """Core parity validation — one test case per golden ticker."""

    @pytest.mark.parametrize("ticker", GOLDEN_TICKERS)
    def test_parity(self, seeded_session: Session, ticker: str):
        legacy_page = _query_legacy(
            seeded_session,
            QuerySpec(page=PageSpec(page=1, per_page=100)),
        )
        feature_page = _query_feature(
            seeded_session,
            QuerySpec(page=PageSpec(page=1, per_page=100)),
        )

        legacy_item = _find(legacy_page, ticker)
        feature_item = _find(feature_page, ticker)

        assert_scan_result_parity(legacy_item, feature_item)


# ═══════════════════════════════════════════════════════════════════════════
# Test Class 3: Filter / sort parity
# ═══════════════════════════════════════════════════════════════════════════


_FILTER_SORT_SPECS: list[tuple[str, QuerySpec]] = [
    (
        "range_rs_rating_min_80",
        QuerySpec.from_filter_spec(
            FilterSpec(
                range_filters=[RangeFilter(field="rs_rating", min_value=80.0)]
            ),
            page=PageSpec(page=1, per_page=100),
        ),
    ),
    (
        "range_stage_2",
        QuerySpec.from_filter_spec(
            FilterSpec(
                range_filters=[RangeFilter(field="stage", min_value=2, max_value=2)]
            ),
            page=PageSpec(page=1, per_page=100),
        ),
    ),
    (
        "categorical_gics_technology",
        QuerySpec.from_filter_spec(
            FilterSpec(
                categorical_filters=[
                    CategoricalFilter(field="gics_sector", values=("Technology",))
                ]
            ),
            page=PageSpec(page=1, per_page=100),
        ),
    ),
    (
        "listing_search_company_name",
        QuerySpec(
            expression=FilterExpression(
                required=FilterGroup(
                    id="required",
                    name="Always require",
                    conditions=(TextSearchFilter("listing_search", "Apple"),),
                )
            ),
            page=PageSpec(page=1, per_page=100),
        ),
    ),
    (
        "grouped_any_expression",
        QuerySpec(
            expression=FilterExpression(
                group_join=MatchOperator.ANY,
                groups=(
                    FilterGroup(
                        id="leadership",
                        name="Leadership",
                        conditions=(RangeFilter("rs_rating", min_value=90),),
                    ),
                    FilterGroup(
                        id="stage-two",
                        name="Stage two",
                        conditions=(RangeFilter("stage", min_value=2, max_value=2),),
                    ),
                ),
            ),
            page=PageSpec(page=1, per_page=100),
        ),
    ),
    (
        "sort_composite_desc",
        QuerySpec(
            sort=SortSpec(field="composite_score", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=100),
        ),
    ),
    (
        "sort_rs_rating_desc",
        QuerySpec(
            sort=SortSpec(field="rs_rating", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=100),
        ),
    ),
]


class TestFilterSortParity:
    """Same filters/sorts produce the same symbol ordering from both sources."""

    @pytest.mark.parametrize(
        "name,spec",
        _FILTER_SORT_SPECS,
        ids=[s[0] for s in _FILTER_SORT_SPECS],
    )
    def test_ordering_parity(self, seeded_session: Session, name: str, spec: QuerySpec):
        legacy_page = _query_legacy(seeded_session, spec)
        feature_page = _query_feature(seeded_session, spec)

        legacy_symbols = [item.symbol for item in legacy_page.items]
        feature_symbols = [item.symbol for item in feature_page.items]

        assert legacy_symbols == feature_symbols, (
            f"[{name}] Symbol order mismatch:\n"
            f"  legacy:  {legacy_symbols}\n"
            f"  feature: {feature_symbols}"
        )

    def test_listing_search_treats_like_metacharacters_as_literal(
        self, seeded_session: Session
    ):
        seeded_session.query(StockUniverse).filter_by(symbol="AAPL").one().name = (
            "Fund_100% Labs"
        )
        seeded_session.query(StockUniverse).filter_by(symbol="MSFT").one().name = (
            "FundA100Z Labs"
        )
        seeded_session.flush()
        spec = QuerySpec(
            expression=FilterExpression(
                required=FilterGroup(
                    id="required",
                    name="Always require",
                    conditions=(TextSearchFilter("listing_search", "_100%"),),
                )
            ),
            page=PageSpec(page=1, per_page=100),
        )

        legacy_symbols = [item.symbol for item in _query_legacy(seeded_session, spec).items]
        feature_symbols = [item.symbol for item in _query_feature(seeded_session, spec).items]

        assert legacy_symbols == ["AAPL"]
        assert feature_symbols == legacy_symbols


# ═══════════════════════════════════════════════════════════════════════════
# Test Class 4: Pagination parity
# ═══════════════════════════════════════════════════════════════════════════


class TestPaginationParity:
    """Page boundary consistency — same pagination returns same slices."""

    def test_page_1_matches(self, seeded_session: Session):
        spec = QuerySpec(page=PageSpec(page=1, per_page=5))

        legacy_page = _query_legacy(seeded_session, spec)
        feature_page = _query_feature(seeded_session, spec)

        assert legacy_page.total == feature_page.total, (
            f"total mismatch: legacy={legacy_page.total} vs fs={feature_page.total}"
        )
        assert legacy_page.total_pages == feature_page.total_pages, (
            f"total_pages mismatch: legacy={legacy_page.total_pages} vs fs={feature_page.total_pages}"
        )
        assert len(legacy_page.items) == len(feature_page.items), (
            f"item count mismatch: legacy={len(legacy_page.items)} vs fs={len(feature_page.items)}"
        )
