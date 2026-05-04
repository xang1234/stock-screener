"""Market domain module exports."""

from .catalog import (
    MARKET_CATALOG,
    MarketCapabilities,
    MarketCatalog,
    MarketCatalogEntry,
    MarketCatalogError,
    get_market_catalog,
)
from .market import Market, SUPPORTED_MARKET_CODES, UnsupportedMarketError
from .registry import MarketProfile, MarketRegistry, market_registry

__all__ = [
    "MARKET_CATALOG",
    "Market",
    "MarketCapabilities",
    "MarketCatalog",
    "MarketCatalogEntry",
    "MarketCatalogError",
    "MarketProfile",
    "MarketRegistry",
    "SUPPORTED_MARKET_CODES",
    "UnsupportedMarketError",
    "get_market_catalog",
    "market_registry",
]
