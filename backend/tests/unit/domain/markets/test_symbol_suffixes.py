from __future__ import annotations

from app.domain.markets import market_symbol_suffix_registry


def test_symbol_suffix_registry_resolves_alias_specific_suffixes_for_shared_mics() -> None:
    assert market_symbol_suffix_registry.suffix_for("TW", "TWSE") == ".TW"
    assert market_symbol_suffix_registry.suffix_for("TW", "TPEX") == ".TWO"
    assert market_symbol_suffix_registry.suffix_for("KR", "KOSPI") == ".KS"
    assert market_symbol_suffix_registry.suffix_for("KR", "KOSDAQ") == ".KQ"
    assert market_symbol_suffix_registry.suffix_for("MY", "KLSE") == ".KL"


def test_symbol_suffix_registry_resolves_ambiguous_aliases_with_market_context() -> None:
    assert market_symbol_suffix_registry.suffix_for("IN", "BSE") == ".BO"
    assert market_symbol_suffix_registry.suffix_for("CN", "BSE") == ".BJ"


def test_symbol_suffix_registry_uses_market_default_without_exchange_alias() -> None:
    assert market_symbol_suffix_registry.suffix_for("IN", None) == ".NS"
    assert market_symbol_suffix_registry.suffix_for("CN", None) == ".SS"
    assert market_symbol_suffix_registry.suffix_for("AU", None) == ".AX"
    assert market_symbol_suffix_registry.suffix_for("MY", None) == ".KL"


def test_symbol_suffix_registry_infers_market_from_symbol_suffix() -> None:
    assert market_symbol_suffix_registry.market_for_symbol("3008.TWO") == "TW"
    assert market_symbol_suffix_registry.market_for_symbol("SAP.DE") == "DE"
    assert market_symbol_suffix_registry.market_for_symbol("BHP.AX") == "AU"
    assert market_symbol_suffix_registry.market_for_symbol("1155.KL") == "MY"
    assert market_symbol_suffix_registry.market_for_symbol("AAPL") is None


def test_symbol_suffix_registry_infers_mic_from_symbol_suffix() -> None:
    assert market_symbol_suffix_registry.mic_for_symbol("920118.BJ") == "XBSE"
    assert market_symbol_suffix_registry.mic_for_symbol("000001.SZ") == "XSHE"
    assert market_symbol_suffix_registry.mic_for_symbol("BHP.AX") == "XASX"
    assert market_symbol_suffix_registry.mic_for_symbol("1155.KL") == "XKLS"
    assert market_symbol_suffix_registry.mic_for_symbol("AAPL") is None


def test_au_symbol_suffix_registry_maps_asx_to_ax() -> None:
    assert market_symbol_suffix_registry.suffix_for("AU", "ASX") == ".AX"
    assert market_symbol_suffix_registry.suffix_for("AU", "XASX") == ".AX"
