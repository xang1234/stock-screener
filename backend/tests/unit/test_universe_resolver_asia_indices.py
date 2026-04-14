"""Resolver dispatch for HSI / NIKKEI225 / TAIEX.

Verifies the INDEX branch of ``resolve_symbols`` routes per-index rather
than hardcoding SP500 lookup for every index request. Service-layer
membership lookup is covered in
``test_stock_universe_service_index_membership.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.schemas.universe import IndexName, UniverseDefinition, UniverseType
from app.services.universe_resolver import resolve_symbols


class TestIndexResolverDispatch:
    def setup_method(self):
        self.mock_db = MagicMock()

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_hsi_dispatches_with_index_name_hsi(self, mock_service):
        mock_service.return_value.get_active_symbols.return_value = ["0700.HK"]
        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.HSI)

        resolve_symbols(self.mock_db, u)

        mock_service.return_value.get_active_symbols.assert_called_once_with(
            self.mock_db, market=None, exchange=None, index_name="HSI", limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_nikkei225_dispatches_with_index_name_nikkei225(self, mock_service):
        mock_service.return_value.get_active_symbols.return_value = ["6758.T"]
        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.NIKKEI225)

        resolve_symbols(self.mock_db, u)

        mock_service.return_value.get_active_symbols.assert_called_once_with(
            self.mock_db, market=None, exchange=None, index_name="NIKKEI225", limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_taiex_dispatches_with_index_name_taiex(self, mock_service):
        mock_service.return_value.get_active_symbols.return_value = ["2330.TW"]
        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.TAIEX)

        resolve_symbols(self.mock_db, u)

        mock_service.return_value.get_active_symbols.assert_called_once_with(
            self.mock_db, market=None, exchange=None, index_name="TAIEX", limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_sp500_still_dispatches_via_index_name_not_legacy_flag(self, mock_service):
        # Backward-compat check: SP500 no longer passes sp500_only=True; the
        # service method accepts both but the resolver now emits the typed
        # index_name parameter for consistency.
        mock_service.return_value.get_active_symbols.return_value = ["AAPL"]
        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.SP500)

        resolve_symbols(self.mock_db, u)

        mock_service.return_value.get_active_symbols.assert_called_once_with(
            self.mock_db, market=None, exchange=None, index_name="SP500", limit=None
        )
