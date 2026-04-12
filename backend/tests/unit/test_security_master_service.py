from app.services.security_master_service import SecurityMasterResolver


def test_resolve_identity_prefers_explicit_market_and_exchange():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(
        symbol=" 0700.hk ",
        market="hk",
        exchange=" sehk ",
    )

    assert identity.normalized_symbol == "0700.HK"
    assert identity.market == "HK"
    assert identity.exchange == "SEHK"
    assert identity.currency == "HKD"
    assert identity.timezone == "Asia/Hong_Kong"
    assert identity.local_code == "0700"


def test_resolve_identity_infers_market_from_exchange_alias():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="9984", exchange="XTKS")

    assert identity.market == "JP"
    assert identity.currency == "JPY"
    assert identity.timezone == "Asia/Tokyo"
    assert identity.canonical_symbol == "9984.T"


def test_resolve_identity_infers_market_from_suffix_when_exchange_missing():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="2330.tw")

    assert identity.market == "TW"
    assert identity.currency == "TWD"
    assert identity.timezone == "Asia/Taipei"
    assert identity.local_code == "2330"


def test_resolve_identity_uses_tpex_suffix_for_unsuffixed_tw_symbol():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="3008", exchange="TPEX")

    assert identity.market == "TW"
    assert identity.exchange == "TPEX"
    assert identity.canonical_symbol == "3008.TWO"


def test_resolve_identity_rewrites_mismatched_tw_suffix_for_tpex_symbol():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="3008.TW", exchange="TPEX")

    assert identity.market == "TW"
    assert identity.exchange == "TPEX"
    assert identity.local_code == "3008"
    assert identity.canonical_symbol == "3008.TWO"


def test_resolve_identity_preserves_explicit_two_suffix_without_exchange_override():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="3008.TWO")

    assert identity.market == "TW"
    assert identity.exchange is None
    assert identity.canonical_symbol == "3008.TWO"


def test_normalize_symbol_strips_dollar_prefix():
    resolver = SecurityMasterResolver()

    assert resolver.normalize_symbol("$nvda") == "NVDA"
