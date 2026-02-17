"""Unit tests for GetPeersUseCase.

Pure in-memory tests — no infrastructure.
"""

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.scanning.models import PeerType, ScanResultItemDomain
from app.use_cases.scanning.get_peers import (
    GetPeersQuery,
    GetPeersResult,
    GetPeersUseCase,
)

from tests.unit.scanning_fakes import (
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
    setup_scan,
)


# ── Specialised fake ────────────────────────────────────────────────────


class PeersResultRepo(FakeScanResultRepository):
    """Fake that supports get_by_symbol, get_peers_by_industry, get_peers_by_sector."""

    def __init__(
        self,
        target: ScanResultItemDomain | None = None,
        industry_peers: tuple[ScanResultItemDomain, ...] = (),
        sector_peers: tuple[ScanResultItemDomain, ...] = (),
    ):
        self._target = target
        self._industry_peers = industry_peers
        self._sector_peers = sector_peers
        self.last_get_by_symbol_args: dict | None = None
        self.last_get_peers_by_industry_args: dict | None = None
        self.last_get_peers_by_sector_args: dict | None = None

    def get_by_symbol(self, scan_id: str, symbol: str) -> ScanResultItemDomain | None:
        self.last_get_by_symbol_args = {"scan_id": scan_id, "symbol": symbol}
        return self._target

    def get_peers_by_industry(
        self, scan_id: str, ibd_industry_group: str
    ) -> tuple[ScanResultItemDomain, ...]:
        self.last_get_peers_by_industry_args = {
            "scan_id": scan_id,
            "ibd_industry_group": ibd_industry_group,
        }
        return self._industry_peers

    def get_peers_by_sector(
        self, scan_id: str, gics_sector: str
    ) -> tuple[ScanResultItemDomain, ...]:
        self.last_get_peers_by_sector_args = {
            "scan_id": scan_id,
            "gics_sector": gics_sector,
        }
        return self._sector_peers


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_query(**overrides) -> GetPeersQuery:
    defaults = dict(scan_id="scan-123", symbol="AAPL")
    defaults.update(overrides)
    return GetPeersQuery(**defaults)


# ── Tests: Happy Path ──────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for industry peer lookup."""

    def test_returns_peers_for_industry(self):
        target = make_domain_item("AAPL", ibd_industry_group="Semiconductor")
        peer1 = make_domain_item("NVDA", score=95.0, ibd_industry_group="Semiconductor")
        peer2 = make_domain_item("AMD", score=80.0, ibd_industry_group="Semiconductor")
        repo = PeersResultRepo(
            target=target,
            industry_peers=(peer1, target, peer2),
        )
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, GetPeersResult)
        assert len(result.peers) == 3
        assert result.group_name == "Semiconductor"
        assert result.peer_type == PeerType.INDUSTRY

    def test_passes_correct_args_to_repository(self):
        target = make_domain_item("TSLA", ibd_industry_group="Auto Manufacturers")
        repo = PeersResultRepo(target=target, industry_peers=(target,))
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow, "scan-xyz")
        uc = GetPeersUseCase()

        uc.execute(uow, _make_query(scan_id="scan-xyz", symbol="TSLA"))

        assert repo.last_get_by_symbol_args == {
            "scan_id": "scan-xyz",
            "symbol": "TSLA",
        }
        assert repo.last_get_peers_by_industry_args == {
            "scan_id": "scan-xyz",
            "ibd_industry_group": "Auto Manufacturers",
        }

    def test_target_included_in_peers(self):
        target = make_domain_item("AAPL", ibd_industry_group="Semiconductor")
        repo = PeersResultRepo(target=target, industry_peers=(target,))
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert any(p.symbol == "AAPL" for p in result.peers)


# ── Tests: Sector Peers ────────────────────────────────────────────────


class TestSectorPeers:
    """Peer lookup by GICS sector."""

    def test_returns_peers_for_sector(self):
        target = make_domain_item("AAPL", gics_sector="Information Technology")
        peer1 = make_domain_item("MSFT", score=90.0, gics_sector="Information Technology")
        repo = PeersResultRepo(
            target=target,
            sector_peers=(peer1, target),
        )
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query(peer_type=PeerType.SECTOR))

        assert isinstance(result, GetPeersResult)
        assert len(result.peers) == 2
        assert result.group_name == "Information Technology"
        assert result.peer_type == PeerType.SECTOR

    def test_passes_correct_args_to_get_peers_by_sector(self):
        target = make_domain_item("AAPL", gics_sector="Information Technology")
        repo = PeersResultRepo(target=target, sector_peers=(target,))
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        uc.execute(uow, _make_query(peer_type=PeerType.SECTOR))

        assert repo.last_get_peers_by_sector_args == {
            "scan_id": "scan-123",
            "gics_sector": "Information Technology",
        }
        # Should NOT call industry method
        assert repo.last_get_peers_by_industry_args is None


# ── Tests: No Group ────────────────────────────────────────────────────


class TestNoGroup:
    """Empty peers when target has no group value."""

    def test_none_group_returns_empty(self):
        target = make_domain_item("AAPL")  # no ibd_industry_group
        repo = PeersResultRepo(target=target)
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert result.peers == ()
        assert result.group_name is None
        assert result.peer_type == PeerType.INDUSTRY

    def test_empty_string_group_returns_empty(self):
        target = make_domain_item("AAPL", ibd_industry_group="")
        repo = PeersResultRepo(target=target)
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert result.peers == ()
        assert result.group_name is None

    def test_whitespace_only_group_returns_empty(self):
        target = make_domain_item("AAPL", ibd_industry_group="   ")
        repo = PeersResultRepo(target=target)
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert result.peers == ()
        assert result.group_name is None


# ── Tests: Error cases ─────────────────────────────────────────────────


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = GetPeersUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))

    def test_not_found_error_has_entity_and_identifier(self):
        uow = FakeUnitOfWork()
        uc = GetPeersUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(scan_id="missing"))

        assert exc_info.value.entity == "Scan"
        assert exc_info.value.identifier == "missing"


class TestSymbolNotFound:
    """Use case raises EntityNotFoundError when symbol is not in the scan."""

    def test_missing_symbol_raises_not_found(self):
        repo = PeersResultRepo()  # target=None
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        with pytest.raises(EntityNotFoundError, match="ScanResult.*AAPL"):
            uc.execute(uow, _make_query(symbol="AAPL"))

    def test_not_found_error_has_scan_result_entity(self):
        repo = PeersResultRepo()
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(symbol="NOPE"))

        assert exc_info.value.entity == "ScanResult"
        assert exc_info.value.identifier == "NOPE"


# ── Tests: Symbol Normalisation ────────────────────────────────────────


class TestSymbolNormalisation:
    """Lowercase input is normalised to uppercase before querying."""

    def test_lowercase_symbol_normalised(self):
        target = make_domain_item("AAPL", ibd_industry_group="Semiconductor")
        repo = PeersResultRepo(target=target, industry_peers=(target,))
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = GetPeersUseCase()

        uc.execute(uow, _make_query(symbol="aapl"))

        assert repo.last_get_by_symbol_args["symbol"] == "AAPL"


# ── Tests: Default PeerType ───────────────────────────────────────────


class TestDefaultPeerType:
    """Default peer_type is INDUSTRY."""

    def test_default_is_industry(self):
        query = GetPeersQuery(scan_id="scan-123", symbol="AAPL")
        assert query.peer_type == PeerType.INDUSTRY
