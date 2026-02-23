"""GetSingleResultUseCase — retrieve a single scan result by symbol.

Business rules:
  1. Verify the scan exists (raise EntityNotFoundError if not)
  2. Normalise the symbol to uppercase (case-insensitive lookup)
  3. If bound to a feature run → query the feature store
  4. Otherwise → fall back to scan_results table
  5. Raise EntityNotFoundError if the symbol is not in the scan

The use case depends ONLY on domain ports — never on SQLAlchemy,
FastAPI, or any other infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.models import ScanResultItemDomain

from ._resolve import resolve_scan

logger = logging.getLogger(__name__)


# ── Query (input) ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class GetSingleResultQuery:
    """Immutable value object describing the single-result lookup."""

    scan_id: str
    symbol: str
    include_setup_payload: bool = False

    def __post_init__(self) -> None:
        # Business rule: symbols are case-insensitive.
        object.__setattr__(self, "symbol", self.symbol.upper())


# ── Result (output) ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class GetSingleResultResult:
    """What the use case returns to the caller."""

    item: ScanResultItemDomain


# ── Use Case ────────────────────────────────────────────────────────────


class GetSingleResultUseCase:
    """Retrieve a single scan result by scan_id and symbol."""

    def execute(
        self, uow: UnitOfWork, query: GetSingleResultQuery
    ) -> GetSingleResultResult:
        with uow:
            scan, run_id = resolve_scan(uow, query.scan_id)

            if run_id:
                logger.info(
                    "Scan %s: querying feature_store for %s (run_id=%d)",
                    query.scan_id,
                    query.symbol,
                    run_id,
                )
                item = uow.feature_store.get_by_symbol_for_run(
                    run_id,
                    query.symbol,
                    include_setup_payload=query.include_setup_payload,
                )
            else:
                logger.info(
                    "Scan %s: reading %s from scan_results (no feature run)",
                    query.scan_id,
                    query.symbol,
                )
                item = uow.scan_results.get_by_symbol(
                    query.scan_id,
                    query.symbol,
                    include_setup_payload=query.include_setup_payload,
                )

            if item is None:
                raise EntityNotFoundError("ScanResult", query.symbol)

        return GetSingleResultResult(item=item)
