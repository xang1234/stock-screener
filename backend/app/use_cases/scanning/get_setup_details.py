"""GetSetupDetailsUseCase — retrieve setup-engine explain payload by symbol."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.uow import UnitOfWork

from ._resolve import resolve_scan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GetSetupDetailsQuery:
    """Immutable value object describing the setup payload lookup."""

    scan_id: str
    symbol: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())


@dataclass(frozen=True)
class GetSetupDetailsResult:
    """Setup explain payload for a single symbol."""

    symbol: str
    se_explain: dict[str, Any] | None
    se_candidates: list[Any] | None


class GetSetupDetailsUseCase:
    """Retrieve setup-engine explain/candidates payload for one symbol."""

    def execute(
        self,
        uow: UnitOfWork,
        query: GetSetupDetailsQuery,
    ) -> GetSetupDetailsResult:
        with uow:
            _scan, run_id = resolve_scan(uow, query.scan_id)

            if run_id:
                logger.info(
                    "Scan %s: querying setup payload from feature_store for %s (run_id=%d)",
                    query.scan_id,
                    query.symbol,
                    run_id,
                )
                payload = uow.feature_store.get_setup_payload_for_run(
                    run_id,
                    query.symbol,
                )
            else:
                logger.info(
                    "Scan %s: reading setup payload for %s from scan_results (no feature run)",
                    query.scan_id,
                    query.symbol,
                )
                payload = uow.scan_results.get_setup_payload(
                    query.scan_id,
                    query.symbol,
                )

            if payload is None:
                raise EntityNotFoundError("ScanResult", query.symbol)

        return GetSetupDetailsResult(
            symbol=query.symbol,
            se_explain=payload.get("se_explain"),
            se_candidates=payload.get("se_candidates"),
        )
