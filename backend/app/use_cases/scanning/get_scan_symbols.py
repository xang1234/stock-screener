"""GetScanSymbolsUseCase — lightweight symbol list query for scan navigation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.domain.common.uow import UnitOfWork
from app.domain.scanning.filter_spec import FilterSpec, PageSpec, SortSpec

from ._resolve import resolve_scan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GetScanSymbolsQuery:
    """Immutable value object describing a symbol-list lookup."""

    scan_id: str
    filters: FilterSpec = field(default_factory=FilterSpec)
    sort: SortSpec = field(default_factory=SortSpec)
    page: PageSpec | None = None
    passes_only: bool = False


@dataclass(frozen=True)
class GetScanSymbolsResult:
    """What the use case returns to the caller."""

    symbols: tuple[str, ...]
    total: int
    page: int | None = None
    per_page: int | None = None


class GetScanSymbolsUseCase:
    """Retrieve filtered/sorted symbols without hydrating full row payloads."""

    def execute(
        self,
        uow: UnitOfWork,
        query: GetScanSymbolsQuery,
    ) -> GetScanSymbolsResult:
        with uow:
            _scan, run_id = resolve_scan(uow, query.scan_id)

            filters = query.filters
            if query.passes_only:
                augmented = FilterSpec(
                    range_filters=list(filters.range_filters),
                    categorical_filters=list(filters.categorical_filters),
                    boolean_filters=list(filters.boolean_filters),
                    text_searches=list(filters.text_searches),
                )
                augmented.add_categorical("rating", ("Strong Buy", "Buy"))
                filters = augmented

            if run_id:
                logger.info(
                    "Scan %s: querying symbol list from feature_store (run_id=%d)",
                    query.scan_id,
                    run_id,
                )
                symbols, total = uow.feature_store.query_run_symbols(
                    run_id,
                    filters,
                    query.sort,
                    page=query.page,
                )
            else:
                logger.info(
                    "Scan %s: querying symbol list from scan_results (no feature run)",
                    query.scan_id,
                )
                symbols, total = uow.scan_results.query_symbols(
                    query.scan_id,
                    filters,
                    query.sort,
                    page=query.page,
                )

        return GetScanSymbolsResult(
            symbols=symbols,
            total=total,
            page=query.page.page if query.page else None,
            per_page=query.page.per_page if query.page else None,
        )
