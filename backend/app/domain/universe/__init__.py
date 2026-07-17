"""Universe domain helpers."""

from enum import Enum

from .definitions import (
    NormalizedMarketScope,
    UniverseStorageProjection,
    normalize_market_scope,
    parse_market_key_components,
    validate_legacy_exchange_scope,
)
from .listing_tiers import (
    ListingTierDefinition,
    ListingTierRegistry,
    listing_tier_registry,
)
from .indexes import IndexDefinition, IndexRegistry, index_registry
from .ingestion import (
    ACTIVE_UNIVERSE_STATUS,
    CanonicalUniverseIngestionResult,
    CanonicalUniverseRow,
    DuplicateActiveUniverseRowError,
    RejectedUniverseRow,
    UniverseLifecycleMetadata,
    UniverseSourceProvenance,
)


class UniverseType(str, Enum):
    """Type of stock universe to scan."""

    ALL = "all"
    MARKET = "market"
    EXCHANGE = "exchange"
    INDEX = "index"
    CUSTOM = "custom"
    TEST = "test"

__all__ = [
    "ACTIVE_UNIVERSE_STATUS",
    "CanonicalUniverseIngestionResult",
    "CanonicalUniverseRow",
    "DuplicateActiveUniverseRowError",
    "IndexDefinition",
    "IndexRegistry",
    "ListingTierDefinition",
    "ListingTierRegistry",
    "NormalizedMarketScope",
    "RejectedUniverseRow",
    "UniverseStorageProjection",
    "UniverseType",
    "UniverseLifecycleMetadata",
    "UniverseSourceProvenance",
    "index_registry",
    "listing_tier_registry",
    "normalize_market_scope",
    "parse_market_key_components",
    "validate_legacy_exchange_scope",
]
