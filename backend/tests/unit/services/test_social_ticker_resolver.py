import pytest

from app.models.stock_universe import StockUniverse


@pytest.fixture
def resolver(db_session):
    from app.services.social_ticker_resolver import SocialTickerResolver
    for symbol, market in [("HSBC", "US"), ("0005.HK", "HK"), ("2330.TW", "TW"),
                           ("TSM", "US"), ("6758.T", "JP"), ("600519.SS", "CN"),
                           ("SPY", "US"), ("QQQ", "US"), ("SMH", "US"), ("REMX", "US")]:
        db_session.add(StockUniverse(symbol=symbol, market=market, is_active=True))
    db_session.add(StockUniverse(symbol="DEAD", market="US", is_active=False))
    db_session.flush()
    return SocialTickerResolver(db_session, verified_company_ids={
        "HSBC": "company:hsbc", "0005.HK": "company:hsbc",
        "TSM": "company:tsmc", "2330.TW": "company:tsmc"})


@pytest.mark.parametrize("token,market,want", [
    ("$HSBC", None, "HSBC"), ("HSBC", None, "HSBC"),
    ("$0005.HK", None, "0005.HK"), ("$600519.SS", None, "600519.SS"),
    ("$5.HK", None, "0005.HK"), ("$5", "HK", "0005.HK"),
    ("$6758.T", None, "6758.T"), ("$2330.TW", None, "2330.TW"),
    ("$2330", "TW", "2330.TW"), ("台積電", None, "2330.TW"),
    ("Sony", None, "6758.T"), ("ソニー", None, "6758.T"),
    ("$TSM", None, "TSM"),
])
def test_explicit_listing_precedes_company_alias(resolver, token, market, want):
    result = resolver.resolve(token, market)
    assert (result.status, result.symbol) == ("resolved", want)


@pytest.mark.parametrize("token,market", [("unknown business", None), ("DEAD", None),
    ("$HSBC", "HK"), ("6758", None), ("$HSBC", "ZZ"), ("$2330.TW", "US")])
def test_unknown_inactive_or_conflicting_listing_remains_unresolved(resolver, token, market):
    result = resolver.resolve(token, market)
    assert result.status == "unresolved"
    assert result.market is None and result.security_id is None


def test_related_instruments_do_not_replace_two_explicit_listings(resolver):
    us, hk = resolver.resolve("$HSBC", None), resolver.resolve("$0005.HK", None)
    assert us.symbol != hk.symbol
    assert us.company_id == hk.company_id == "company:hsbc"
    assert us.related_symbols == ("0005.HK",)
    assert hk.related_symbols == ("HSBC",)
    assert us.company_count_eligible and hk.company_count_eligible


def test_no_company_linkage_is_fabricated(resolver):
    result = resolver.resolve("6758.T", None)
    assert result.company_id is None and not result.company_count_eligible
    assert result.ranking_eligible and result.related_symbols == ()
    assert "company_identity_unknown" in result.reason_codes


@pytest.mark.parametrize("symbol,kind,eligible", [("SPY", "broad_etf", False),
    ("QQQ", "broad_etf", False), ("SMH", "thematic_etf", True), ("REMX", "thematic_etf", True)])
def test_etf_context_and_thematic_eligibility(resolver, symbol, kind, eligible):
    result = resolver.resolve(symbol, None)
    assert result.security_kind == kind
    assert result.ranking_eligible is eligible
    assert not result.company_count_eligible
