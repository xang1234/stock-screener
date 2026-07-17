"""Unit tests for GetPeersUseCase.

Pure in-memory tests — no infrastructure.
"""

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.models import PeerType
from app.use_cases.scanning.get_peers import (
    GetPeersQuery,
    GetPeersResult,
    GetPeersUseCase,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeUnitOfWork,
)


# ── Constants ────────────────────────────────────────────────────────────

AS_OF = date(2026, 2, 17)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_feature_row(symbol, score=85.0, **details_extra):
    details = {
        "composite_score": score,
        "rating": "Buy",
        "current_price": 150.0,
        "screeners_run": ["minervini"],
        "composite_method": "weighted_average",
        "screeners_passed": 1,
        "screeners_total": 1,
    }
    details.update(details_extra)
    return FeatureRow(
        run_id=1, symbol=symbol, as_of_date=AS_OF,
        composite_score=score, overall_rating=4,
        passes_count=1, details=details,
    )


def _make_query(**overrides) -> GetPeersQuery:
    defaults = dict(scan_id="scan-123", symbol="AAPL")
    defaults.update(overrides)
    return GetPeersQuery(**defaults)


# ── Tests: Happy Path ──────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for industry peer lookup."""

    def test_returns_peers_for_industry(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", ibd_industry_group="Semiconductor"),
            _make_feature_row("NVDA", score=95.0, ibd_industry_group="Semiconductor"),
            _make_feature_row("AMD", score=80.0, ibd_industry_group="Semiconductor"),
        ])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, GetPeersResult)
        assert len(result.peers) == 3
        assert result.group_name == "Semiconductor"
        assert result.peer_type == PeerType.INDUSTRY

    def test_passes_correct_args(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-xyz", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("TSLA", ibd_industry_group="Auto Manufacturers"),
        ])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-xyz", symbol="TSLA"))

        assert result.group_name == "Auto Manufacturers"
        assert result.peer_type == PeerType.INDUSTRY

    def test_target_included_in_peers(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", ibd_industry_group="Semiconductor"),
        ])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert any(p.symbol == "AAPL" for p in result.peers)


# ── Tests: Sector Peers ────────────────────────────────────────────────


class TestSectorPeers:
    """Peer lookup by GICS sector."""

    def test_returns_peers_for_sector(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", gics_sector="Information Technology"),
            _make_feature_row("MSFT", score=90.0, gics_sector="Information Technology"),
        ])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query(peer_type=PeerType.SECTOR))

        assert isinstance(result, GetPeersResult)
        assert len(result.peers) == 2
        assert result.group_name == "Information Technology"
        assert result.peer_type == PeerType.SECTOR

    def test_sector_peers_does_not_call_industry(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", gics_sector="Information Technology"),
        ])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query(peer_type=PeerType.SECTOR))

        # Just verify it returned sector peers, not industry
        assert result.peer_type == PeerType.SECTOR
        assert result.group_name == "Information Technology"


# ── Tests: No Group ────────────────────────────────────────────────────


class TestNoGroup:
    """Empty peers when target has no group value."""

    def test_none_group_returns_empty(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        # No ibd_industry_group in details
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("AAPL")])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert result.peers == ()
        assert result.group_name is None
        assert result.peer_type == PeerType.INDUSTRY

    def test_empty_string_group_returns_empty(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", ibd_industry_group=""),
        ])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query())

        assert result.peers == ()
        assert result.group_name is None

    def test_whitespace_only_group_returns_empty(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", ibd_industry_group="   "),
        ])
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
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        # Run exists but has no AAPL
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("MSFT")])
        uc = GetPeersUseCase()

        with pytest.raises(EntityNotFoundError, match="ScanResult.*AAPL"):
            uc.execute(uow, _make_query(symbol="AAPL"))

    def test_not_found_error_has_scan_result_entity(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("MSFT")])
        uc = GetPeersUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(symbol="NOPE"))

        assert exc_info.value.entity == "ScanResult"
        assert exc_info.value.identifier == "NOPE"


# ── Tests: Symbol Normalisation ────────────────────────────────────────


class TestSymbolNormalisation:
    """Lowercase input is normalised to uppercase before querying."""

    def test_lowercase_symbol_normalised(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", ibd_industry_group="Semiconductor"),
        ])
        uc = GetPeersUseCase()

        result = uc.execute(uow, _make_query(symbol="aapl"))

        assert result.group_name == "Semiconductor"


# ── Tests: Default PeerType ───────────────────────────────────────────


class TestDefaultPeerType:
    """Default peer_type is INDUSTRY."""

    def test_default_is_industry(self):
        query = GetPeersQuery(scan_id="scan-123", symbol="AAPL")
        assert query.peer_type == PeerType.INDUSTRY
