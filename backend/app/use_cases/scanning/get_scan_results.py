"""GetScanResultsUseCase — paginated, filtered, sorted query of scan results.

This use case owns the business rules for retrieving scan results:
  1. Verify the scan exists (raise EntityNotFoundError if not)
  2. If bound to a feature run → query the feature store
  3. Otherwise → fall back to scan_results table
  4. Return a ResultPage (source-agnostic)

The use case depends ONLY on domain ports — never on SQLAlchemy,
FastAPI, or any other infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from app.domain.common.uow import UnitOfWork
from app.domain.scanning.filter_expression_evaluator import (
    annotate_matched_groups,
    require_passing_ratings,
)
from app.domain.scanning.filter_expression_model import QuerySpec
from app.domain.scanning.filter_expression_serialization import expression_fingerprint
from app.domain.scanning.models import ResultPage

from ._resolve import resolve_scan

logger = logging.getLogger(__name__)


# ── Query (input) ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class GetScanResultsQuery:
    """Immutable value object describing what the caller wants to read."""

    scan_id: str
    query_spec: QuerySpec = field(default_factory=QuerySpec)
    include_sparklines: bool = True
    include_setup_payload: bool = False
    passes_only: bool = False


# ── Result (output) ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class GetScanResultsResult:
    """What the use case returns to the caller."""

    page: ResultPage
    unfiltered_total: int
    query_fingerprint: str


# ── Use Case ────────────────────────────────────────────────────────────


class GetScanResultsUseCase:
    """Retrieve a filtered, sorted, paginated page of scan results."""

    def execute(
        self, uow: UnitOfWork, query: GetScanResultsQuery
    ) -> GetScanResultsResult:
        with uow:
            scan, run_id = resolve_scan(uow, query.scan_id)

            expression = require_passing_ratings(
                query.query_spec.expression,
                enabled=query.passes_only,
            )
            query_spec = replace(query.query_spec, expression=expression)

            if run_id:
                logger.info(
                    "Scan %s: querying feature_store (run_id=%d)",
                    query.scan_id,
                    run_id,
                )
                result_page = uow.feature_store.query_run_as_scan_results(
                    run_id,
                    query_spec,
                    include_sparklines=query.include_sparklines,
                    include_setup_payload=query.include_setup_payload,
                )
                unfiltered_total = uow.feature_store.count_by_run_id(run_id)
            else:
                logger.info(
                    "Scan %s: reading from scan_results (no feature run)",
                    query.scan_id,
                )
                result_page = uow.scan_results.query(
                    query.scan_id,
                    query_spec,
                    include_sparklines=query.include_sparklines,
                    include_setup_payload=query.include_setup_payload,
                )
                unfiltered_total = uow.scan_results.count_by_scan_id(query.scan_id)

        result_page = replace(
            result_page,
            items=annotate_matched_groups(result_page.items, expression),
        )
        return GetScanResultsResult(
            page=result_page,
            unfiltered_total=unfiltered_total,
            query_fingerprint=expression_fingerprint(expression),
        )
