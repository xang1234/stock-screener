"""GetPeersUseCase — retrieve peer stocks from the same industry or sector.

Business rules:
  1. Verify the scan exists (raise EntityNotFoundError if not)
  2. Look up the target symbol (raise EntityNotFoundError if not found)
  3. Read the group value from extended_fields based on peer_type
  4. If no group value, return empty result
  5. Delegate to the appropriate repo method (industry or sector)

The use case depends ONLY on domain ports — never on SQLAlchemy,
FastAPI, or any other infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.models import PeerType, ScanResultItemDomain


# ── Mapping tables ─────────────────────────────────────────────────────

# PeerType → key in ScanResultItemDomain.extended_fields
_GROUP_FIELD: dict[PeerType, str] = {
    PeerType.INDUSTRY: "ibd_industry_group",
    PeerType.SECTOR: "gics_sector",
}


# ── Query (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class GetPeersQuery:
    """Immutable value object describing the peers lookup."""

    scan_id: str
    symbol: str
    peer_type: PeerType = PeerType.INDUSTRY

    def __post_init__(self) -> None:
        # Business rule: symbols are case-insensitive.
        object.__setattr__(self, "symbol", self.symbol.upper())


# ── Result (output) ────────────────────────────────────────────────────


@dataclass(frozen=True)
class GetPeersResult:
    """What the use case returns to the caller."""

    peers: tuple[ScanResultItemDomain, ...]
    group_name: str | None
    peer_type: PeerType


# ── Use Case ───────────────────────────────────────────────────────────


class GetPeersUseCase:
    """Retrieve peer stocks from the same industry group or sector."""

    def execute(
        self, uow: UnitOfWork, query: GetPeersQuery
    ) -> GetPeersResult:
        with uow:
            # 1. Verify scan exists
            scan = uow.scans.get_by_scan_id(query.scan_id)
            if scan is None:
                raise EntityNotFoundError("Scan", query.scan_id)

            # 2. Look up target symbol
            target = uow.scan_results.get_by_symbol(
                scan_id=query.scan_id,
                symbol=query.symbol,
            )
            if target is None:
                raise EntityNotFoundError("ScanResult", query.symbol)

            # 3. Read group value from extended_fields
            field_key = _GROUP_FIELD[query.peer_type]
            group_value = target.extended_fields.get(field_key)

            # 4. No group → empty result
            if not group_value or not str(group_value).strip():
                return GetPeersResult(
                    peers=(), group_name=None, peer_type=query.peer_type
                )

            # 5. Delegate to appropriate repo method
            if query.peer_type == PeerType.INDUSTRY:
                peers = uow.scan_results.get_peers_by_industry(
                    query.scan_id, group_value
                )
            else:
                peers = uow.scan_results.get_peers_by_sector(
                    query.scan_id, group_value
                )

        return GetPeersResult(
            peers=peers, group_name=group_value, peer_type=query.peer_type
        )
