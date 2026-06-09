"""Schemas for runtime capabilities and local bootstrap controls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_serializer

from app.domain.markets.catalog import get_market_catalog
from app.schemas.universe import UniverseDefinition


def _supported_market_codes() -> list[str]:
    return get_market_catalog().supported_market_codes()


class ScanDefaultsResponse(BaseModel):
    """Backend-owned default scan profile exposed to the frontend."""

    universe: str
    screeners: list[str] = Field(default_factory=list)
    composite_method: str
    criteria: dict[str, Any] = Field(default_factory=dict)
    benchmark_symbol: str | None = None


class AppAuthStatusResponse(BaseModel):
    """Authentication state exposed to the frontend shell."""

    required: bool = False
    configured: bool = True
    authenticated: bool = True
    mode: str = "session_cookie"
    message: str | None = None


class MarketCapabilitiesResponse(BaseModel):
    """Coarse capabilities for a supported Market."""

    benchmark: bool = False
    breadth: bool = False
    fundamentals: bool = False
    group_rankings: bool = False
    feature_snapshot: bool = False
    official_universe: bool = False
    finviz_screening: bool = False


class MicFactsResponse(BaseModel):
    """Canonical per-MIC facts in the runtime Market Catalog."""

    mic: str
    calendar_id: str
    timezone: str
    default_currency: str
    provider_calendar_id: str | None = None


class MarketCatalogEntryResponse(BaseModel):
    """Stable Market facts exposed to the frontend runtime."""

    code: str
    label: str
    primary_mic: str
    mics: list[str] = Field(min_length=1)
    supported_currencies: list[str] = Field(min_length=1)
    default_currency: str
    mic_facts: list[MicFactsResponse] = Field(min_length=1)
    currency: str
    timezone: str
    calendar_id: str
    provider_calendar_id: str | None = None
    exchanges: list[str]
    indexes: list[str]
    capabilities: MarketCapabilitiesResponse


class MarketCatalogResponse(BaseModel):
    """Frontend-ready Market Catalog payload."""

    version: str
    markets: list[MarketCatalogEntryResponse]


class RuntimeUniverseSelectionResponse(BaseModel):
    """Display metadata plus a canonical UniverseDefinition payload."""

    value: str
    label: str
    universe_def: UniverseDefinition

    @field_serializer("universe_def", when_used="json")
    def serialize_universe_def(self, value: UniverseDefinition) -> dict[str, Any]:
        return value.model_dump(
            mode="json",
            exclude_none=True,
            exclude_defaults=True,
        )


class RuntimeMicUniverseOptionResponse(RuntimeUniverseSelectionResponse):
    """Canonical MIC-scoped Market Universe option."""

    mic: str
    aliases: list[str] = Field(default_factory=list)


class RuntimeMicAliasOptionResponse(RuntimeUniverseSelectionResponse):
    """Market-scoped compatibility alias for a canonical MIC option."""

    alias: str
    mic: str


class RuntimeIndexUniverseOptionResponse(RuntimeUniverseSelectionResponse):
    """Canonical Index Universe option."""

    key: str
    aliases: list[str] = Field(default_factory=list)


class RuntimeListingTierUniverseOptionResponse(RuntimeUniverseSelectionResponse):
    """Market or MIC-scoped listing tier filter option."""

    key: str
    mic: str | None = None
    aliases: list[str] = Field(default_factory=list)


class RuntimeUniverseMarketOptionsResponse(BaseModel):
    """Catalog-backed Universe options for one supported Market."""

    code: str
    label: str
    enabled: bool
    capabilities: MarketCapabilitiesResponse
    market: RuntimeUniverseSelectionResponse
    mics: list[RuntimeMicUniverseOptionResponse] = Field(default_factory=list)
    mic_aliases: list[RuntimeMicAliasOptionResponse] = Field(default_factory=list)
    indexes: list[RuntimeIndexUniverseOptionResponse] = Field(default_factory=list)
    listing_tiers: list[RuntimeListingTierUniverseOptionResponse] = Field(
        default_factory=list
    )


class RuntimeUniverseOptionsResponse(BaseModel):
    """Stable Universe choices plus runtime preference overlays."""

    version: str
    supported_markets: list[str]
    enabled_markets: list[str]
    markets: list[RuntimeUniverseMarketOptionsResponse]


class AppCapabilitiesResponse(BaseModel):
    """Feature/capability flags exposed to the frontend."""

    features: dict[str, bool]
    ui_snapshots: dict[str, bool] = Field(default_factory=dict)
    scan_defaults: ScanDefaultsResponse
    bootstrap_required: bool = False
    primary_market: str = "US"
    enabled_markets: list[str] = Field(default_factory=lambda: ["US"])
    bootstrap_state: str = "not_started"
    supported_markets: list[str] = Field(default_factory=_supported_market_codes)
    market_catalog: MarketCatalogResponse
    universe_options: RuntimeUniverseOptionsResponse
    api_base_path: str = "/api"
    auth: AppAuthStatusResponse = Field(default_factory=AppAuthStatusResponse)


class RuntimeBootstrapStatusResponse(BaseModel):
    """Current persisted local-runtime bootstrap state."""

    bootstrap_required: bool
    empty_system: bool
    primary_market: str
    enabled_markets: list[str]
    bootstrap_state: str
    supported_markets: list[str] = Field(default_factory=_supported_market_codes)


class RuntimeBootstrapRequest(BaseModel):
    """Bootstrap request payload for first-run local setup."""

    primary_market: str = "US"
    enabled_markets: list[str] = Field(default_factory=lambda: ["US"])


class RuntimeBootstrapStartResponse(RuntimeBootstrapStatusResponse):
    """Bootstrap response including the queued orchestration task."""

    task_id: str | None = None


class RuntimeActivityBootstrapResponse(BaseModel):
    """Bootstrap progress summary for the frontend shell."""

    state: str
    app_ready: bool
    primary_market: str
    enabled_markets: list[str] = Field(default_factory=list)
    current_stage: str | None = None
    progress_mode: str = "indeterminate"
    percent: float | None = None
    message: str | None = None
    background_warning: str | None = None


class RuntimeActivitySummaryResponse(BaseModel):
    """Compact runtime activity summary for the header."""

    active_market_count: int = 0
    active_markets: list[str] = Field(default_factory=list)
    status: str = "idle"


class RuntimeActivityMarketResponse(BaseModel):
    """Per-market runtime activity row."""

    market: str
    lifecycle: str
    stage_key: str | None = None
    stage_label: str | None = None
    status: str
    progress_mode: str = "indeterminate"
    percent: float | None = None
    current: int | None = None
    total: int | None = None
    message: str | None = None
    task_name: str | None = None
    task_id: str | None = None
    updated_at: str | None = None


class RuntimeActivityResponse(BaseModel):
    """Unified runtime activity payload for bootstrap and operations UI."""

    bootstrap: RuntimeActivityBootstrapResponse
    summary: RuntimeActivitySummaryResponse
    markets: list[RuntimeActivityMarketResponse] = Field(default_factory=list)


class RuntimeMarketsUpdateRequest(BaseModel):
    """Patch request for persisted local market preferences."""

    primary_market: str
    enabled_markets: list[str]
