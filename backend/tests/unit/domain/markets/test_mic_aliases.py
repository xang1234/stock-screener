from __future__ import annotations

from app.domain.markets import mic_alias_registry


def test_mic_alias_registry_resolves_ambiguous_alias_with_market_context() -> None:
    india = mic_alias_registry.resolve("IN", "BSE")
    china = mic_alias_registry.resolve("CN", "BSE")

    assert india is not None
    assert india.market == "IN"
    assert india.mic == "XBOM"
    assert china is not None
    assert china.market == "CN"
    assert china.mic == "XBSE"


def test_mic_alias_registry_does_not_resolve_ambiguous_alias_globally() -> None:
    assert mic_alias_registry.resolve_global("BSE") is None
    assert mic_alias_registry.market_for_alias("BSE") is None
    assert mic_alias_registry.is_ambiguous("BSE") is True


def test_mic_alias_registry_resolves_globally_unambiguous_aliases() -> None:
    resolved = mic_alias_registry.resolve_global("SEHK")

    assert resolved is not None
    assert resolved.market == "HK"
    assert resolved.mic == "XHKG"
    assert mic_alias_registry.market_for_alias("SEHK") == "HK"


def test_mic_alias_registry_accepts_canonical_mic_as_alias() -> None:
    resolved = mic_alias_registry.resolve("IN", "XBOM")

    assert resolved is not None
    assert resolved.market == "IN"
    assert resolved.mic == "XBOM"


def test_au_mic_aliases_resolve_to_xasx() -> None:
    asx = mic_alias_registry.resolve("AU", "ASX")
    xasx = mic_alias_registry.resolve("AU", "XASX")

    assert asx is not None
    assert asx.mic == "XASX"
    assert xasx is not None
    assert xasx.mic == "XASX"


def test_mic_alias_registry_lists_canonical_and_legacy_aliases_for_mic() -> None:
    assert mic_alias_registry.aliases_for_mic("US", "XNYS") == ("XNYS", "NYSE")
