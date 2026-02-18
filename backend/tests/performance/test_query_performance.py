"""Query performance guardrails for feature store and legacy scan result paths.

Verifies:
  1. Query count budgets (N+1 detection)
  2. Query count invariance across page sizes and depths
  3. Wall-clock timing budgets (CI-safe generous limits)

All tests are marked with ``@pytest.mark.performance`` so they can be
selected or excluded via ``pytest -m performance``.
"""

from __future__ import annotations

import time

import pytest

from app.domain.common.query import (
    FilterSpec,
    PageSpec,
    QuerySpec,
    SortOrder,
    SortSpec,
)
from tests.helpers.query_counter import count_queries

from .conftest import PERF_FEATURE_RUN_ID, PERF_SCAN_ID

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

# Feature store: session.get(FeatureRun) + COUNT(*) + SELECT...LIMIT
FS_QUERY_BUDGET = 3

# Legacy: COUNT(*) + SELECT...LIMIT (no run existence check)
LEGACY_QUERY_BUDGET = 2

# Wall-clock budgets (ms) — generous for CI
TIMING_BUDGET_SIMPLE_MS = 100
TIMING_BUDGET_COMBINED_MS = 250


# ═══════════════════════════════════════════════════════════════════════════
# 1. Feature Store Query Count
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestFeatureStoreQueryCount:
    """Assert the feature store path stays within its 3-query budget."""

    def test_paginated_default(self, fs_repo, perf_engine):
        """Page 1, per_page 50 — the most common query."""
        spec = QuerySpec(page=PageSpec(page=1, per_page=50))
        with count_queries(perf_engine) as counter:
            result = fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert result.total == 500
        assert len(result.items) == 50
        assert counter["count"] <= FS_QUERY_BUDGET, (
            f"Expected <={FS_QUERY_BUDGET} queries, got {counter['count']}:\n"
            + "\n".join(counter["statements"])
        )

    def test_filtered_by_score_range(self, fs_repo, perf_engine):
        """Range filter on composite_score (indexed SQL column)."""
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=60.0)
        spec = QuerySpec(filters=filters, page=PageSpec(page=1, per_page=50))
        with count_queries(perf_engine) as counter:
            fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert counter["count"] <= FS_QUERY_BUDGET

    def test_sorted_by_composite_desc(self, fs_repo, perf_engine):
        """Sort by composite_score DESC (SQL column)."""
        spec = QuerySpec(
            sort=SortSpec(field="composite_score", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=50),
        )
        with count_queries(perf_engine) as counter:
            result = fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert result.items[0].composite_score >= result.items[-1].composite_score
        assert counter["count"] <= FS_QUERY_BUDGET

    def test_filter_and_sort_combined(self, fs_repo, perf_engine):
        """Score range filter + sort by rs_rating (JSON field)."""
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=40.0, max_value=80.0)
        spec = QuerySpec(
            filters=filters,
            sort=SortSpec(field="rs_rating", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=50),
        )
        with count_queries(perf_engine) as counter:
            fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert counter["count"] <= FS_QUERY_BUDGET

    def test_categorical_filter_sector(self, fs_repo, perf_engine):
        """Categorical filter on gics_sector (JSON field)."""
        filters = FilterSpec()
        filters.add_categorical("gics_sector", ("Technology",))
        spec = QuerySpec(filters=filters, page=PageSpec(page=1, per_page=50))
        with count_queries(perf_engine) as counter:
            result = fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert result.total > 0
        assert counter["count"] <= FS_QUERY_BUDGET

    def test_text_search_symbol(self, fs_repo, perf_engine):
        """Text search on symbol — LIKE '%PERF1%'."""
        filters = FilterSpec()
        filters.add_text_search("symbol", "PERF1")
        spec = QuerySpec(filters=filters, page=PageSpec(page=1, per_page=50))
        with count_queries(perf_engine) as counter:
            result = fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert result.total > 0
        assert counter["count"] <= FS_QUERY_BUDGET

    def test_json_field_filter(self, fs_repo, perf_engine):
        """Range filter on rs_rating via json_extract()."""
        filters = FilterSpec()
        filters.add_range("rs_rating", min_value=70.0)
        spec = QuerySpec(filters=filters, page=PageSpec(page=1, per_page=50))
        with count_queries(perf_engine) as counter:
            result = fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert result.total > 0
        assert counter["count"] <= FS_QUERY_BUDGET

    def test_json_field_sort(self, fs_repo, perf_engine):
        """Sort by minervini_score via json_extract()."""
        spec = QuerySpec(
            sort=SortSpec(field="minervini_score", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=50),
        )
        with count_queries(perf_engine) as counter:
            fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        assert counter["count"] <= FS_QUERY_BUDGET


# ═══════════════════════════════════════════════════════════════════════════
# 2. Legacy Query Count
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestLegacyQueryCount:
    """Assert the legacy path stays within its 2-query budget."""

    def test_paginated_default(self, legacy_repo, perf_engine):
        spec = QuerySpec(page=PageSpec(page=1, per_page=50))
        with count_queries(perf_engine) as counter:
            result = legacy_repo.query(PERF_SCAN_ID, spec)
        assert result.total == 500
        assert len(result.items) == 50
        assert counter["count"] <= LEGACY_QUERY_BUDGET, (
            f"Expected <={LEGACY_QUERY_BUDGET} queries, got {counter['count']}:\n"
            + "\n".join(counter["statements"])
        )

    def test_filtered_by_score_range(self, legacy_repo, perf_engine):
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=60.0)
        spec = QuerySpec(filters=filters, page=PageSpec(page=1, per_page=50))
        with count_queries(perf_engine) as counter:
            legacy_repo.query(PERF_SCAN_ID, spec)
        assert counter["count"] <= LEGACY_QUERY_BUDGET

    def test_sorted_by_composite_desc(self, legacy_repo, perf_engine):
        spec = QuerySpec(
            sort=SortSpec(field="composite_score", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=50),
        )
        with count_queries(perf_engine) as counter:
            result = legacy_repo.query(PERF_SCAN_ID, spec)
        assert result.items[0].composite_score >= result.items[-1].composite_score
        assert counter["count"] <= LEGACY_QUERY_BUDGET

    def test_filter_and_sort_combined(self, legacy_repo, perf_engine):
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=40.0, max_value=80.0)
        spec = QuerySpec(
            filters=filters,
            sort=SortSpec(field="rs_rating", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=50),
        )
        with count_queries(perf_engine) as counter:
            legacy_repo.query(PERF_SCAN_ID, spec)
        assert counter["count"] <= LEGACY_QUERY_BUDGET


# ═══════════════════════════════════════════════════════════════════════════
# 3. Query Count Invariance (the definitive N+1 test)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestQueryCountInvariance:
    """Prove query count is constant regardless of page size or depth."""

    def test_feature_store_invariant_to_page_size(self, fs_repo, perf_engine):
        """per_page=5 and per_page=50 must produce the same query count."""
        spec_small = QuerySpec(page=PageSpec(page=1, per_page=5))
        spec_large = QuerySpec(page=PageSpec(page=1, per_page=50))

        with count_queries(perf_engine) as c_small:
            fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec_small)
        with count_queries(perf_engine) as c_large:
            fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec_large)

        assert c_small["count"] == c_large["count"], (
            f"Query count varied with page size: "
            f"per_page=5 → {c_small['count']}, per_page=50 → {c_large['count']}"
        )

    def test_feature_store_invariant_to_page_depth(self, fs_repo, perf_engine):
        """page=1 and page=5 must produce the same query count."""
        spec_p1 = QuerySpec(page=PageSpec(page=1, per_page=50))
        spec_p5 = QuerySpec(page=PageSpec(page=5, per_page=50))

        with count_queries(perf_engine) as c_p1:
            fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec_p1)
        with count_queries(perf_engine) as c_p5:
            fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec_p5)

        assert c_p1["count"] == c_p5["count"], (
            f"Query count varied with page depth: "
            f"page=1 → {c_p1['count']}, page=5 → {c_p5['count']}"
        )

    def test_legacy_invariant_to_page_size(self, legacy_repo, perf_engine):
        """per_page=5 and per_page=50 must produce the same query count."""
        spec_small = QuerySpec(page=PageSpec(page=1, per_page=5))
        spec_large = QuerySpec(page=PageSpec(page=1, per_page=50))

        with count_queries(perf_engine) as c_small:
            legacy_repo.query(PERF_SCAN_ID, spec_small)
        with count_queries(perf_engine) as c_large:
            legacy_repo.query(PERF_SCAN_ID, spec_large)

        assert c_small["count"] == c_large["count"], (
            f"Query count varied with page size: "
            f"per_page=5 → {c_small['count']}, per_page=50 → {c_large['count']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature Store Query Timing
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestFeatureStoreQueryTiming:
    """Wall-clock timing with generous CI-safe budgets."""

    def test_paginated_query_under_budget(self, fs_repo):
        spec = QuerySpec(page=PageSpec(page=1, per_page=50))
        # Warmup: prime SA query compilation + SQLite page cache
        fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)

        t0 = time.perf_counter()
        fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < TIMING_BUDGET_SIMPLE_MS, (
            f"Paginated query took {elapsed_ms:.1f}ms (budget: {TIMING_BUDGET_SIMPLE_MS}ms)"
        )

    def test_filtered_query_under_budget(self, fs_repo):
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=60.0)
        spec = QuerySpec(filters=filters, page=PageSpec(page=1, per_page=50))
        fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)

        t0 = time.perf_counter()
        fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < TIMING_BUDGET_SIMPLE_MS, (
            f"Filtered query took {elapsed_ms:.1f}ms (budget: {TIMING_BUDGET_SIMPLE_MS}ms)"
        )

    def test_combined_query_under_budget(self, fs_repo):
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=40.0, max_value=80.0)
        filters.add_categorical("gics_sector", ("Technology",))
        spec = QuerySpec(
            filters=filters,
            sort=SortSpec(field="rs_rating", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=50),
        )
        fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)

        t0 = time.perf_counter()
        fs_repo.query_run_as_scan_results(PERF_FEATURE_RUN_ID, spec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < TIMING_BUDGET_COMBINED_MS, (
            f"Combined query took {elapsed_ms:.1f}ms (budget: {TIMING_BUDGET_COMBINED_MS}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Legacy Query Timing
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestLegacyQueryTiming:
    """Wall-clock timing for the legacy scan_results path."""

    def test_paginated_query_under_budget(self, legacy_repo):
        spec = QuerySpec(page=PageSpec(page=1, per_page=50))
        legacy_repo.query(PERF_SCAN_ID, spec)

        t0 = time.perf_counter()
        legacy_repo.query(PERF_SCAN_ID, spec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < TIMING_BUDGET_SIMPLE_MS, (
            f"Paginated query took {elapsed_ms:.1f}ms (budget: {TIMING_BUDGET_SIMPLE_MS}ms)"
        )

    def test_filtered_query_under_budget(self, legacy_repo):
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=60.0)
        spec = QuerySpec(filters=filters, page=PageSpec(page=1, per_page=50))
        legacy_repo.query(PERF_SCAN_ID, spec)

        t0 = time.perf_counter()
        legacy_repo.query(PERF_SCAN_ID, spec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < TIMING_BUDGET_SIMPLE_MS, (
            f"Filtered query took {elapsed_ms:.1f}ms (budget: {TIMING_BUDGET_SIMPLE_MS}ms)"
        )

    def test_combined_query_under_budget(self, legacy_repo):
        filters = FilterSpec()
        filters.add_range("composite_score", min_value=40.0, max_value=80.0)
        spec = QuerySpec(
            filters=filters,
            sort=SortSpec(field="rs_rating", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=50),
        )
        legacy_repo.query(PERF_SCAN_ID, spec)

        t0 = time.perf_counter()
        legacy_repo.query(PERF_SCAN_ID, spec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < TIMING_BUDGET_COMBINED_MS, (
            f"Combined query took {elapsed_ms:.1f}ms (budget: {TIMING_BUDGET_COMBINED_MS}ms)"
        )
