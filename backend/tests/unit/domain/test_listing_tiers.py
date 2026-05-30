from __future__ import annotations

from app.domain.universe.listing_tiers import (
    ListingTierDefinition,
    ListingTierRegistry,
    listing_tier_registry,
)


def test_listing_tier_registry_normalizes_source_aliases_by_market() -> None:
    assert listing_tier_registry.normalize("HK", "Main Board") == "main_board"
    assert listing_tier_registry.normalize("HK", "GEM") == "gem"
    assert listing_tier_registry.normalize("SG", "Catalist") == "catalist"


def test_listing_tier_registry_normalizes_au_asx_main_aliases() -> None:
    assert listing_tier_registry.normalize("AU", "Main", mic="XASX") == "main"
    assert listing_tier_registry.normalize("AU", "ASX Main", mic="XASX") == "main"


def test_listing_tier_registry_returns_none_for_unknown_or_blank_tiers() -> None:
    assert listing_tier_registry.normalize("HK", "") is None
    assert listing_tier_registry.normalize("HK", "Not A Tier") is None
    assert listing_tier_registry.normalize("US", "Main Board") is None


def test_listing_tier_registry_requires_mic_for_ambiguous_market_alias() -> None:
    registry = ListingTierRegistry(
        (
            ListingTierDefinition(
                key="primary",
                label="Primary",
                market="XX",
                mic="XAAA",
                aliases=("MAIN",),
            ),
            ListingTierDefinition(
                key="secondary",
                label="Secondary",
                market="XX",
                mic="XBBB",
                aliases=("MAIN",),
            ),
        )
    )

    assert registry.normalize("XX", "MAIN") is None
    assert registry.normalize("XX", "MAIN", mic="XAAA") == "primary"
    assert registry.normalize("XX", "MAIN", mic="XBBB") == "secondary"
