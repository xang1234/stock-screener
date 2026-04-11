"""Unit tests for UniverseResolver service."""
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.universe import (
    Exchange,
    IndexName,
    Market,
    UniverseDefinition,
    UniverseType,
)
from app.services.universe_resolver import (
    normalize_universe_definition,
    resolve_count,
    resolve_symbols,
)


class TestNormalizeUniverseDefinition:
    def test_passes_through_typed_definition(self):
        universe = UniverseDefinition(type=UniverseType.ALL)

        result = normalize_universe_definition(universe)

        assert result is universe

    def test_maps_active_string_to_all(self):
        result = normalize_universe_definition("active")

        assert result.type == UniverseType.ALL

    def test_accepts_typed_dict_payload(self):
        result = normalize_universe_definition({"type": "all"})

        assert result.type == UniverseType.ALL

    def test_accepts_legacy_name_dict_payload(self):
        result = normalize_universe_definition({"name": "active"})

        assert result.type == UniverseType.ALL

    def test_invalid_dict_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported universe definition dict"):
            normalize_universe_definition({"foo": "bar"})


@pytest.fixture
def mock_db():
    return MagicMock()


class TestResolveSymbols:
    """Test resolve_symbols for each UniverseType."""

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_all_universe(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["AAPL", "MSFT", "GOOGL"]

        u = UniverseDefinition(type=UniverseType.ALL)
        result = resolve_symbols(mock_db, u)

        assert result == ["AAPL", "MSFT", "GOOGL"]
        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange=None, sp500_only=False, limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_market_passes_market_param(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["0700.HK", "9988.HK"]

        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.HK)
        result = resolve_symbols(mock_db, u)

        assert result == ["0700.HK", "9988.HK"]
        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market="HK", exchange=None, sp500_only=False, limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_exchange_passes_exchange_param(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["IBM", "GE"]

        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NYSE)
        result = resolve_symbols(mock_db, u)

        assert result == ["IBM", "GE"]
        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange="NYSE", sp500_only=False, limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_exchange_nasdaq(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["AAPL", "MSFT"]

        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NASDAQ)
        resolve_symbols(mock_db, u)

        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange="NASDAQ", sp500_only=False, limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_index_sp500(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["AAPL", "MSFT"]

        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.SP500)
        result = resolve_symbols(mock_db, u)

        assert result == ["AAPL", "MSFT"]
        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange=None, sp500_only=True, limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_custom_filters_to_active_symbols_by_default(self, mock_service, mock_db):
        mock_service.return_value.filter_active_symbols.return_value = ["AAPL"]

        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "TSLA"])
        result = resolve_symbols(mock_db, u)

        assert result == ["AAPL"]
        mock_service.return_value.filter_active_symbols.assert_called_once_with(
            mock_db, ["AAPL", "TSLA"]
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_test_filters_to_active_symbols_by_default(self, mock_service, mock_db):
        mock_service.return_value.filter_active_symbols.return_value = ["SPY"]

        u = UniverseDefinition(type=UniverseType.TEST, symbols=["SPY", "QQQ"])
        result = resolve_symbols(mock_db, u)

        assert result == ["SPY"]
        mock_service.return_value.filter_active_symbols.assert_called_once_with(
            mock_db, ["SPY", "QQQ"]
        )

    def test_custom_can_include_inactive_symbols(self, mock_db):
        u = UniverseDefinition(
            type=UniverseType.CUSTOM,
            symbols=["AAPL", "TSLA"],
            allow_inactive_symbols=True,
        )
        result = resolve_symbols(mock_db, u)
        assert result == ["AAPL", "TSLA"]

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_limit_param_passed_through(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["AAPL"]

        u = UniverseDefinition(type=UniverseType.ALL)
        resolve_symbols(mock_db, u, limit=10)

        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange=None, sp500_only=False, limit=10
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_custom_limit(self, mock_service, mock_db):
        mock_service.return_value.filter_active_symbols.return_value = ["A", "B", "D"]

        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["A", "B", "C", "D"])
        result = resolve_symbols(mock_db, u, limit=2)

        assert result == ["A", "B"]

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_active_string_normalizes_to_all(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["AAPL", "MSFT"]

        result = resolve_symbols(mock_db, "active")

        assert result == ["AAPL", "MSFT"]
        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange=None, sp500_only=False, limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_typed_dict_normalizes_before_resolution(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["AAPL"]

        result = resolve_symbols(mock_db, {"type": "all"})

        assert result == ["AAPL"]
        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange=None, sp500_only=False, limit=None
        )

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_legacy_name_dict_normalizes_before_resolution(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["AAPL"]

        result = resolve_symbols(mock_db, {"name": "active"})

        assert result == ["AAPL"]
        mock_service.return_value.get_active_symbols.assert_called_once_with(
            mock_db, market=None, exchange=None, sp500_only=False, limit=None
        )


class TestResolveCount:
    """Test resolve_count returns correct counts."""

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_custom_count(self, mock_service, mock_db):
        mock_service.return_value.filter_active_symbols.return_value = ["AAPL"]
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "MSFT"])
        assert resolve_count(mock_db, u) == 1

    def test_test_count_with_inactive_override(self, mock_db):
        u = UniverseDefinition(
            type=UniverseType.TEST,
            symbols=["TSLA"],
            allow_inactive_symbols=True,
        )
        assert resolve_count(mock_db, u) == 1

    @patch("app.services.universe_resolver.get_stock_universe_service")
    def test_all_count(self, mock_service, mock_db):
        mock_service.return_value.get_active_symbols.return_value = ["A", "B", "C"]

        u = UniverseDefinition(type=UniverseType.ALL)
        assert resolve_count(mock_db, u) == 3
