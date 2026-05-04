from app.services.security_master_service import SecurityMasterResolver
from app.domain.markets.registry import market_registry


def test_security_master_normalize_market_accepts_all_registry_markets():
    resolver = SecurityMasterResolver()

    for market in market_registry.supported_markets():
        assert resolver.normalize_market(market.code) == market.code


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


def test_resolve_identity_defaults_india_market_to_nse_suffix():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="reliance", market="in")

    assert identity.market == "IN"
    assert identity.currency == "INR"
    assert identity.timezone == "Asia/Kolkata"
    assert identity.canonical_symbol == "RELIANCE.NS"
    assert identity.local_code == "RELIANCE"


def test_resolve_identity_uses_bse_suffix_for_explicit_bse_exchange():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="500325", exchange="xbom")

    assert identity.market == "IN"
    assert identity.exchange == "XBOM"
    assert identity.currency == "INR"
    assert identity.timezone == "Asia/Kolkata"
    assert identity.canonical_symbol == "500325.BO"
    assert identity.local_code == "500325"


def test_resolve_identity_uses_kospi_suffix_for_korea_default():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="005930", market="kr")

    assert identity.market == "KR"
    assert identity.currency == "KRW"
    assert identity.timezone == "Asia/Seoul"
    assert identity.canonical_symbol == "005930.KS"
    assert identity.local_code == "005930"


def test_resolve_identity_uses_kosdaq_suffix_for_explicit_exchange():
    resolver = SecurityMasterResolver()

    identity = resolver.resolve_identity(symbol="091990.KS", exchange="kosdaq")

    assert identity.market == "KR"
    assert identity.exchange == "KOSDAQ"
    assert identity.currency == "KRW"
    assert identity.timezone == "Asia/Seoul"
    assert identity.canonical_symbol == "091990.KQ"
    assert identity.local_code == "091990"


def test_resolve_identity_uses_china_suffixes_by_exchange():
    resolver = SecurityMasterResolver()

    sse = resolver.resolve_identity(symbol="600519", exchange="xshg")
    szse = resolver.resolve_identity(symbol="000001.SS", exchange="szse")
    bjse = resolver.resolve_identity(symbol="920118", market="cn", exchange="bjse")
    india_bse = resolver.resolve_identity(symbol="500325", exchange="bse")

    assert sse.market == "CN"
    assert sse.currency == "CNY"
    assert sse.timezone == "Asia/Shanghai"
    assert sse.canonical_symbol == "600519.SS"
    assert szse.market == "CN"
    assert szse.canonical_symbol == "000001.SZ"
    assert bjse.market == "CN"
    assert bjse.exchange == "BJSE"
    assert bjse.canonical_symbol == "920118.BJ"
    assert india_bse.market == "IN"
    assert india_bse.canonical_symbol == "500325.BO"
