"""Scan result, symbol-navigation, export, and filter-option endpoints."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.query import PageSpec
from app.domain.scanning.filter_expression_model import QuerySpec, filter_spec_to_expression
from app.domain.scanning.models import ExportFormat
from app.schemas.filter_expression import ScanQueryRequest
from app.schemas.scanning import (
    FilterOptionsResponse,
    ScanResultItem,
    ScanResultsResponse,
    ScanSymbolsResponse,
)
from app.use_cases.scanning.export_scan_results import ExportScanResultsQuery
from app.use_cases.scanning.get_filter_options import GetFilterOptionsQuery
from app.use_cases.scanning.get_scan_results import GetScanResultsQuery
from app.use_cases.scanning.get_scan_symbols import GetScanSymbolsQuery
from app.wiring.bootstrap import (
    get_export_scan_results_use_case,
    get_get_filter_options_use_case,
    get_get_scan_results_use_case,
    get_get_scan_symbols_use_case,
    get_uow,
)

from .scan_filter_params import parse_page_spec, parse_scan_filters, parse_scan_sort


logger = logging.getLogger(__name__)
router = APIRouter()
T = TypeVar("T")


def _execute_query(
    operation: Callable[[], T],
    *,
    log_context: str,
    public_error: str,
) -> T:
    """Map shared use-case failures consistently at the HTTP boundary."""

    try:
        return operation()
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("%s: %s", log_context, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=public_error) from exc


def _results_response(
    scan_id: str,
    result: Any,
    *,
    include_setup_payload: bool,
) -> ScanResultsResponse:
    return ScanResultsResponse(
        scan_id=scan_id,
        total=result.page.total,
        unfiltered_total=result.unfiltered_total,
        page=result.page.page,
        per_page=result.page.per_page,
        pages=result.page.total_pages,
        query_fingerprint=result.query_fingerprint,
        results=[
            ScanResultItem.from_domain(
                item,
                include_setup_payload=include_setup_payload,
            )
            for item in result.page.items
        ],
    )


def _symbols_response(scan_id: str, result: Any) -> ScanSymbolsResponse:
    return ScanSymbolsResponse(
        scan_id=scan_id,
        total=result.total,
        symbols=list(result.symbols),
        page=result.page,
        per_page=result.per_page,
        next_cursor=None,
        query_fingerprint=result.query_fingerprint,
    )


def _export_response(result: Any) -> StreamingResponse:
    return StreamingResponse(
        iter([result.content]),
        media_type=result.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{result.filename}"',
            "Content-Length": str(len(result.content)),
        },
    )


def _optional_page_spec(page: int | None, per_page: int | None) -> PageSpec | None:
    if page is None and per_page is None:
        return None
    return PageSpec(page=page or 1, per_page=per_page or 100)


@router.get("/{scan_id}/results", response_model=ScanResultsResponse)
async def get_scan_results(
    scan_id: str,
    passes_only: bool = Query(False, description="Show only stocks passing template"),
    include_sparklines: bool = Query(True, description="Include sparkline data"),
    detail_level: Literal["table", "full"] = Query(
        "table",
        description="Response detail level. 'table' excludes heavy setup-engine payload fields.",
    ),
    filters=Depends(parse_scan_filters),
    sort=Depends(parse_scan_sort),
    page=Depends(parse_page_spec),
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_scan_results_use_case),
):
    include_setup_payload = detail_level == "full"
    result = _execute_query(
        lambda: use_case.execute(
            uow,
            GetScanResultsQuery(
                scan_id=scan_id,
                query_spec=QuerySpec.from_filter_spec(filters, sort=sort, page=page),
                include_sparklines=include_sparklines,
                include_setup_payload=include_setup_payload,
                passes_only=passes_only,
            ),
        ),
        log_context="Error getting scan results",
        public_error="Error getting scan results",
    )
    return _results_response(
        scan_id,
        result,
        include_setup_payload=include_setup_payload,
    )


@router.post("/{scan_id}/results/query", response_model=ScanResultsResponse)
async def query_scan_results(
    scan_id: str,
    request: ScanQueryRequest,
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_scan_results_use_case),
):
    include_setup_payload = request.options.detail_level == "full"
    result = _execute_query(
        lambda: use_case.execute(
            uow,
            GetScanResultsQuery(
                scan_id=scan_id,
                query_spec=QuerySpec(
                    expression=request.to_expression(),
                    sort=request.sort.to_domain(),
                    page=request.page.to_domain() if request.page else PageSpec(),
                ),
                include_sparklines=request.options.include_sparklines,
                include_setup_payload=include_setup_payload,
                passes_only=request.passes_only,
            ),
        ),
        log_context="Error querying grouped scan results",
        public_error="Error querying scan results",
    )
    return _results_response(
        scan_id,
        result,
        include_setup_payload=include_setup_payload,
    )


@router.get("/{scan_id}/symbols", response_model=ScanSymbolsResponse)
async def get_scan_symbols(
    scan_id: str,
    passes_only: bool = Query(False, description="Show only stocks passing template"),
    page: int | None = Query(None, ge=1, description="Optional page number"),
    per_page: int | None = Query(None, ge=1, le=100, description="Optional results per page"),
    filters=Depends(parse_scan_filters),
    sort=Depends(parse_scan_sort),
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_scan_symbols_use_case),
):
    result = _execute_query(
        lambda: use_case.execute(
            uow,
            GetScanSymbolsQuery(
                scan_id=scan_id,
                expression=filter_spec_to_expression(filters),
                sort=sort,
                page=_optional_page_spec(page, per_page),
                passes_only=passes_only,
            ),
        ),
        log_context="Error getting scan symbols",
        public_error="Error getting scan symbols",
    )
    return _symbols_response(scan_id, result)


@router.post("/{scan_id}/symbols/query", response_model=ScanSymbolsResponse)
async def query_scan_symbols(
    scan_id: str,
    request: ScanQueryRequest,
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_scan_symbols_use_case),
):
    result = _execute_query(
        lambda: use_case.execute(
            uow,
            GetScanSymbolsQuery(
                scan_id=scan_id,
                expression=request.to_expression(),
                sort=request.sort.to_domain(),
                page=request.page.to_domain() if request.page else None,
                passes_only=request.passes_only,
            ),
        ),
        log_context="Error querying grouped scan symbols",
        public_error="Error querying scan symbols",
    )
    return _symbols_response(scan_id, result)


@router.get("/{scan_id}/export")
async def export_scan_results(
    scan_id: str,
    export_format: str = Query(default="csv", pattern="^(csv)$", alias="format"),
    passes_only: bool = Query(False, description="Show only stocks passing template"),
    filters=Depends(parse_scan_filters),
    sort=Depends(parse_scan_sort),
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_export_scan_results_use_case),
):
    result = _execute_query(
        lambda: use_case.execute(
            uow,
            ExportScanResultsQuery(
                scan_id=scan_id,
                expression=filter_spec_to_expression(filters),
                sort=sort,
                export_format=ExportFormat(export_format),
                passes_only=passes_only,
            ),
        ),
        log_context="Error exporting scan results",
        public_error="Error exporting scan results",
    )
    return _export_response(result)


@router.post("/{scan_id}/export/query")
async def export_grouped_scan_results(
    scan_id: str,
    request: ScanQueryRequest,
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_export_scan_results_use_case),
):
    result = _execute_query(
        lambda: use_case.execute(
            uow,
            ExportScanResultsQuery(
                scan_id=scan_id,
                expression=request.to_expression(),
                sort=request.sort.to_domain(),
                export_format=ExportFormat.CSV,
                passes_only=request.passes_only,
            ),
        ),
        log_context="Error exporting grouped scan results",
        public_error="Error exporting scan results",
    )
    return _export_response(result)


@router.get("/{scan_id}/filter-options", response_model=FilterOptionsResponse)
async def get_filter_options(
    scan_id: str,
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_filter_options_use_case),
):
    result = _execute_query(
        lambda: use_case.execute(uow, GetFilterOptionsQuery(scan_id=scan_id)),
        log_context="Error getting filter options",
        public_error="Error getting filter options",
    )
    return FilterOptionsResponse(
        ibd_industries=list(result.options.ibd_industries),
        gics_sectors=list(result.options.gics_sectors),
        ratings=list(result.options.ratings),
    )
