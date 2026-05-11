"""Unit tests for UniverseDefinition schema."""
import pytest
from pydantic import ValidationError

from app.schemas.universe import (
    Exchange,
    IndexName,
    Market,
    UniverseDefinition,
    UniverseType,
)


class TestUniverseDefinitionConstruction:
    """Test valid construction for each UniverseType."""

    def test_all_universe(self):
        u = UniverseDefinition(type=UniverseType.ALL)
        assert u.type == UniverseType.ALL
        assert u.exchange is None
        assert u.index is None
        assert u.symbols is None

    def test_exchange_nyse(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NYSE)
        assert u.type == UniverseType.EXCHANGE
        assert u.exchange == Exchange.NYSE

    def test_market_hk(self):
        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.HK)
        assert u.type == UniverseType.MARKET
        assert u.market == Market.HK

    def test_market_cn(self):
        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.CN)
        assert u.type == UniverseType.MARKET
        assert u.market == Market.CN

    def test_exchange_nasdaq(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NASDAQ)
        assert u.exchange == Exchange.NASDAQ

    def test_exchange_amex(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.AMEX)
        assert u.exchange == Exchange.AMEX

    @pytest.mark.parametrize("exchange", [Exchange.SSE, Exchange.SZSE, Exchange.BJSE])
    def test_china_exchanges(self, exchange):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=exchange)
        assert u.exchange == exchange

    def test_market_ca(self):
        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.CA)
        assert u.type == UniverseType.MARKET
        assert u.market == Market.CA

    @pytest.mark.parametrize("exchange", [Exchange.TSX, Exchange.TSXV])
    def test_canada_exchanges(self, exchange):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=exchange)
        assert u.exchange == exchange

    def test_index_sp500(self):
        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.SP500)
        assert u.type == UniverseType.INDEX
        assert u.index == IndexName.SP500

    def test_custom_universe(self):
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "MSFT"])
        assert u.type == UniverseType.CUSTOM
        assert u.symbols == ["AAPL", "MSFT"]

    def test_test_universe(self):
        u = UniverseDefinition(type=UniverseType.TEST, symbols=["TSLA"])
        assert u.type == UniverseType.TEST
        assert u.symbols == ["TSLA"]

    def test_custom_universe_can_allow_inactive_symbols(self):
        u = UniverseDefinition(
            type=UniverseType.CUSTOM,
            symbols=["AAPL", "MSFT"],
            allow_inactive_symbols=True,
        )
        assert u.allow_inactive_symbols is True


class TestUniverseDefinitionValidation:
    """Test validation errors for invalid field combinations."""

    def test_all_with_exchange_raises(self):
        with pytest.raises(ValidationError, match="ALL universe must not specify"):
            UniverseDefinition(type=UniverseType.ALL, exchange=Exchange.NYSE)

    def test_all_with_symbols_raises(self):
        with pytest.raises(ValidationError, match="ALL universe must not specify"):
            UniverseDefinition(type=UniverseType.ALL, symbols=["AAPL"])

    def test_exchange_without_exchange_raises(self):
        with pytest.raises(ValidationError, match="requires 'exchange' field"):
            UniverseDefinition(type=UniverseType.EXCHANGE)

    def test_market_without_market_raises(self):
        with pytest.raises(ValidationError, match="requires 'market' field"):
            UniverseDefinition(type=UniverseType.MARKET)

    def test_market_with_exchange_raises(self):
        with pytest.raises(ValidationError, match="must not specify exchange"):
            UniverseDefinition(
                type=UniverseType.MARKET,
                market=Market.HK,
                exchange=Exchange.NYSE,
            )

    def test_exchange_with_index_raises(self):
        with pytest.raises(ValidationError, match="must not specify"):
            UniverseDefinition(
                type=UniverseType.EXCHANGE, exchange=Exchange.NYSE, index=IndexName.SP500
            )

    def test_index_without_index_raises(self):
        with pytest.raises(ValidationError, match="requires 'index' field"):
            UniverseDefinition(type=UniverseType.INDEX)

    def test_index_with_exchange_raises(self):
        with pytest.raises(ValidationError, match="must not specify"):
            UniverseDefinition(
                type=UniverseType.INDEX, index=IndexName.SP500, exchange=Exchange.NYSE
            )

    def test_custom_without_symbols_raises(self):
        with pytest.raises(ValidationError, match="requires a non-empty"):
            UniverseDefinition(type=UniverseType.CUSTOM)

    def test_custom_with_empty_symbols_raises(self):
        with pytest.raises(ValidationError, match="requires a non-empty"):
            UniverseDefinition(type=UniverseType.CUSTOM, symbols=[])

    def test_test_without_symbols_raises(self):
        with pytest.raises(ValidationError, match="requires a non-empty"):
            UniverseDefinition(type=UniverseType.TEST)

    def test_max_symbol_count_exceeded(self):
        symbols = [f"SYM{i}" for i in range(501)]
        with pytest.raises(ValidationError, match="Maximum is 500"):
            UniverseDefinition(type=UniverseType.CUSTOM, symbols=symbols)

    def test_max_symbol_count_at_limit(self):
        symbols = [f"SYM{i}" for i in range(500)]
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=symbols)
        assert len(u.symbols) == 500


class TestSymbolNormalization:
    """Test symbol normalization (trim, uppercase, deduplicate, empty removal)."""

    def test_uppercase(self):
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["aapl", "msft"])
        assert u.symbols == ["AAPL", "MSFT"]

    def test_trim_whitespace(self):
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["  AAPL  ", " MSFT"])
        assert u.symbols == ["AAPL", "MSFT"]

    def test_deduplicate(self):
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "aapl", "MSFT"])
        assert u.symbols == ["AAPL", "MSFT"]

    def test_remove_empty_strings(self):
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "", "  ", "MSFT"])
        assert u.symbols == ["AAPL", "MSFT"]

    def test_whitespace_only_symbols_removed(self):
        """All-whitespace symbols should be removed, not kept."""
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "   ", "MSFT"])
        assert u.symbols == ["AAPL", "MSFT"]


class TestKey:
    """Test key() canonical string output."""

    def test_all_key(self):
        assert UniverseDefinition(type=UniverseType.ALL).key() == "all"

    def test_exchange_key(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NYSE)
        assert u.key() == "exchange:NYSE"

    def test_exchange_nasdaq_key(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NASDAQ)
        assert u.key() == "exchange:NASDAQ"

    def test_index_key(self):
        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.SP500)
        assert u.key() == "index:SP500"

    def test_market_key(self):
        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.JP)
        assert u.key() == "market:JP"

    def test_china_market_key(self):
        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.CN)
        assert u.key() == "market:CN"

    def test_custom_key_is_deterministic(self):
        u1 = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "MSFT"])
        u2 = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["MSFT", "AAPL"])
        assert u1.key() == u2.key()  # Same symbols, different order → same key

    def test_custom_key_starts_with_custom(self):
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL"])
        assert u.key().startswith("custom:")

    def test_custom_key_changes_when_inactive_symbols_allowed(self):
        active_only = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL"])
        include_inactive = UniverseDefinition(
            type=UniverseType.CUSTOM,
            symbols=["AAPL"],
            allow_inactive_symbols=True,
        )
        assert active_only.key() != include_inactive.key()

    def test_test_key_starts_with_test(self):
        u = UniverseDefinition(type=UniverseType.TEST, symbols=["AAPL"])
        assert u.key().startswith("test:")

    def test_different_custom_sets_different_keys(self):
        u1 = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL"])
        u2 = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["MSFT"])
        assert u1.key() != u2.key()


class TestLabel:
    """Test label() human-readable output."""

    def test_all_label(self):
        assert UniverseDefinition(type=UniverseType.ALL).label() == "All Stocks"

    def test_exchange_label(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NYSE)
        assert u.label() == "NYSE"

    def test_market_label(self):
        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.US)
        assert u.label() == "US Market"

    def test_index_label(self):
        u = UniverseDefinition(type=UniverseType.INDEX, index=IndexName.SP500)
        assert u.label() == "S&P 500"

    def test_custom_label(self):
        u = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "MSFT"])
        assert u.label() == "Custom (2 symbols)"

    def test_custom_label_with_inactive_override(self):
        u = UniverseDefinition(
            type=UniverseType.CUSTOM,
            symbols=["AAPL", "MSFT"],
            allow_inactive_symbols=True,
        )
        assert u.label() == "Custom (2 symbols, incl. inactive)"

    def test_test_label(self):
        u = UniverseDefinition(type=UniverseType.TEST, symbols=["AAPL"])
        assert u.label() == "Test (1 symbol)"


class TestFromLegacy:
    """Test from_legacy() backward-compatible parsing."""

    def test_all(self):
        u = UniverseDefinition.from_legacy("all")
        assert u.type == UniverseType.ALL

    def test_nyse(self):
        u = UniverseDefinition.from_legacy("nyse")
        assert u.type == UniverseType.EXCHANGE
        assert u.exchange == Exchange.NYSE

    def test_market_prefixed(self):
        u = UniverseDefinition.from_legacy("market:jp")
        assert u.type == UniverseType.MARKET
        assert u.market == Market.JP

    def test_market_prefixed_china(self):
        u = UniverseDefinition.from_legacy("market:cn")
        assert u.type == UniverseType.MARKET
        assert u.market == Market.CN

    def test_market_prefixed_germany(self):
        u = UniverseDefinition.from_legacy("market:de")
        assert u.type == UniverseType.MARKET
        assert u.market == Market.DE

    def test_market_short_code_without_prefix_is_not_legacy(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            UniverseDefinition.from_legacy("hk")

    def test_nasdaq(self):
        u = UniverseDefinition.from_legacy("nasdaq")
        assert u.type == UniverseType.EXCHANGE
        assert u.exchange == Exchange.NASDAQ

    def test_exchange_can_carry_market_to_disambiguate_bse(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, market=Market.CN, exchange=Exchange.BJSE)
        assert u.key() == "exchange:CN:BJSE"

    def test_amex(self):
        u = UniverseDefinition.from_legacy("amex")
        assert u.type == UniverseType.EXCHANGE
        assert u.exchange == Exchange.AMEX

    def test_sp500(self):
        u = UniverseDefinition.from_legacy("sp500")
        assert u.type == UniverseType.INDEX
        assert u.index == IndexName.SP500

    def test_custom_with_symbols(self):
        u = UniverseDefinition.from_legacy("custom", ["AAPL", "MSFT"])
        assert u.type == UniverseType.CUSTOM
        assert u.symbols == ["AAPL", "MSFT"]

    def test_test_with_symbols(self):
        u = UniverseDefinition.from_legacy("test", ["TSLA"])
        assert u.type == UniverseType.TEST
        assert u.symbols == ["TSLA"]

    def test_case_insensitive(self):
        u = UniverseDefinition.from_legacy("NYSE")
        assert u.type == UniverseType.EXCHANGE
        assert u.exchange == Exchange.NYSE

    def test_whitespace_trimmed(self):
        u = UniverseDefinition.from_legacy("  all  ")
        assert u.type == UniverseType.ALL

    def test_unknown_without_symbols_raises(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            UniverseDefinition.from_legacy("invalid_value")

    def test_unknown_with_symbols_becomes_custom(self):
        u = UniverseDefinition.from_legacy("weird_value", ["AAPL"])
        assert u.type == UniverseType.CUSTOM
        assert u.symbols == ["AAPL"]


class TestJsonSerialization:
    """Test JSON round-trip compatibility (important for FastAPI)."""

    def test_all_serializes(self):
        u = UniverseDefinition(type=UniverseType.ALL)
        data = u.model_dump()
        assert data["type"] == "all"

    def test_exchange_serializes(self):
        u = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NYSE)
        data = u.model_dump()
        assert data["type"] == "exchange"
        assert data["exchange"] == "NYSE"

    def test_market_serializes(self):
        u = UniverseDefinition(type=UniverseType.MARKET, market=Market.TW)
        data = u.model_dump()
        assert data["type"] == "market"
        assert data["market"] == "TW"

    def test_round_trip(self):
        original = UniverseDefinition(type=UniverseType.CUSTOM, symbols=["AAPL", "MSFT"])
        data = original.model_dump()
        restored = UniverseDefinition(**data)
        assert restored.type == original.type
        assert restored.symbols == original.symbols
        assert restored.key() == original.key()
