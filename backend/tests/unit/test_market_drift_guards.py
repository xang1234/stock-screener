"""Cross-layer drift guards for duplicated Market facts.

These tests intentionally document today's compatibility drift instead of
changing runtime behavior. Later harmonization tasks should remove the
exception dictionaries as each consumer moves behind Market Catalog.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.api.v1 import breadth, groups, scans
from app.domain.markets import SUPPORTED_MARKET_CODES, market_registry
from app.domain.markets.catalog import get_market_catalog
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from app.services import provider_routing_policy
from app.services.field_capability_registry import field_capability_registry
from app.services.official_market_universe_source_service import (
    OfficialMarketUniverseSourceService,
)
from app.services.official_universe_dispatch import (
    OFFICIAL_SOURCE_MARKETS,
    OFFICIAL_UNIVERSE_INGEST_METHODS,
)
from app.services.security_master_service import SecurityMasterResolver
from app.services.universe_row_facts import (
    active_universe_currency_drift,
    active_universe_timezone_drift,
)
from app.tasks import market_queues
from app.domain.universe.indexes import index_registry


REPO_ROOT = Path(__file__).resolve().parents[2].parent

DOCUMENTED_CATALOG_MIC_CODES: dict[str, set[str]] = {
    "US": set(),
    "HK": {"XHKG"},
    "IN": {"XNSE", "XBOM"},
    "JP": {"XTKS"},
    "KR": {"XKRX"},
    "TW": {"XTAI"},
    "CN": {"XSHG", "XSHE", "XBSE"},
    "CA": {"XTSE", "XTNX"},
    "DE": {"XETR", "XFRA"},
    "SG": {"XSES"},
    "AU": {"XASX"},
    "MY": {"XKLS"},
}

DOCUMENTED_CATALOG_ALIAS_CODES: dict[str, set[str]] = {
    "US": {"NYSE", "NASDAQ", "AMEX"},
    "HK": {"HKEX", "SEHK"},
    "IN": {"NSE", "BSE"},
    "JP": {"TSE", "JPX"},
    "KR": {"KOSPI", "KOSDAQ", "KRX"},
    "TW": {"TWSE", "TPEX"},
    "CN": {"SSE", "SZSE", "BJSE"},
    "CA": {"TSX", "TSXV"},
    "DE": {"XETRA", "FRA", "FWB"},
    "SG": {"SGX", "SES"},
    "AU": {"ASX"},
    "MY": {"KLSE", "MYX", "BURSA"},
}

DOCUMENTED_CATALOG_CANONICAL_MICS: dict[str, tuple[str, ...]] = {
    "US": ("XNYS", "XNAS", "XASE"),
    "HK": ("XHKG",),
    "IN": ("XNSE", "XBOM"),
    "JP": ("XTKS",),
    "KR": ("XKRX",),
    "TW": ("XTAI",),
    "CN": ("XSHG", "XSHE", "XBSE"),
    "CA": ("XTSE", "XTNX"),
    "DE": ("XETR", "XFRA"),
    "SG": ("XSES",),
    "AU": ("XASX",),
    "MY": ("XKLS",),
}

DOCUMENTED_REGISTRY_ONLY_EXCHANGE_ALIASES: dict[str, set[str]] = {}

DOCUMENTED_CATALOG_ONLY_EXCHANGE_ALIASES: dict[str, set[str]] = {}


def _catalog_codes() -> list[str]:
    return get_market_catalog().supported_market_codes()


def _catalog_market_codes_by_capability(capability: str) -> tuple[str, ...]:
    return get_market_catalog().market_codes_with_capability(capability)


def _catalog_market_codes_by_rrg_scope(scope: str) -> tuple[str, ...]:
    return get_market_catalog().market_codes_with_rrg_scope(scope)


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _fallback_catalog_codes_from_frontend() -> list[str]:
    # Phase 0 only guards fallback code drift. A generated/structured frontend
    # catalog fixture belongs with the later frontend contract standardization
    # work, when fallback data is removed or generated from backend facts.
    runtime_context = REPO_ROOT / "frontend" / "src" / "contexts" / "RuntimeContext.jsx"
    source = runtime_context.read_text()
    fallback_catalog = source.split(
        "export const DEFAULT_MARKET_CATALOG = {", maxsplit=1
    )[1].split("const DEFAULT_SUPPORTED_MARKETS", maxsplit=1)[0]
    return re.findall(r"code:\s*'([A-Z]{2})'", fallback_catalog)


def _frontend_scan_geographic_market_codes() -> list[str]:
    scan_constants = (
        REPO_ROOT / "frontend" / "src" / "features" / "scan" / "constants.js"
    )
    source = scan_constants.read_text()
    match = re.search(r"UNIVERSE_GEOGRAPHIC_MARKETS = \[([^\]]+)\]", source)
    assert match is not None, "frontend scan market list is missing"
    return re.findall(r"'([A-Z]{2})'", match.group(1))


def _frontend_scan_market_options() -> dict[str, str]:
    scan_constants = (
        REPO_ROOT / "frontend" / "src" / "features" / "scan" / "constants.js"
    )
    source = scan_constants.read_text()
    match = re.search(r"UNIVERSE_MARKETS = \[(?P<body>.*?)\n\];", source, re.S)
    assert match is not None, "frontend scan market options are missing"
    return {
        value: label
        for value, label in re.findall(
            r"\{\s*value: '([^']+)', label: '([^']+)'\s*\}",
            match.group("body"),
        )
        if value != "TEST"
    }


def _frontend_scan_scope_options() -> dict[str, list[tuple[str, str]]]:
    scan_constants = (
        REPO_ROOT / "frontend" / "src" / "features" / "scan" / "constants.js"
    )
    source = scan_constants.read_text()
    body = source.split("export const UNIVERSE_SCOPES_BY_MARKET = {", maxsplit=1)[
        1
    ].split("\n};", maxsplit=1)[0]
    scopes: dict[str, list[tuple[str, str]]] = {}
    for code in (*_catalog_codes(), "TEST"):
        match = re.search(rf"^\s*{code}: \[(?P<body>.*?)^\s*\],", body, re.S | re.M)
        if match is None:
            empty_match = re.search(rf"^\s*{code}: \[\],", body, re.M)
            assert empty_match is not None, f"{code} scan scopes are missing"
            scopes[code] = []
            continue
        scopes[code] = re.findall(
            r"\{\s*value: '([^']+)', label: '([^']+)'\s*\}",
            match.group("body"),
        )
    return scopes


def _frontend_breadth_object_market_keys(object_name: str) -> list[str]:
    breadth_page = REPO_ROOT / "frontend" / "src" / "pages" / "BreadthPage.jsx"
    source = breadth_page.read_text()
    match = re.search(rf"const {object_name} = \{{(?P<body>.*?)\n\}};", source, re.S)
    assert match is not None, f"{object_name} is missing from BreadthPage"
    return re.findall(r"^\s*([A-Z]{2}):", match.group("body"), re.M)


def test_supported_market_code_surfaces_match_catalog_codes() -> None:
    expected = set(_catalog_codes())

    assert SUPPORTED_MARKET_CODES == expected
    assert set(market_registry.supported_market_codes()) == expected
    assert set(market_queues.SUPPORTED_MARKETS) == expected
    assert set(provider_routing_policy.KNOWN_MARKETS) == expected
    assert set(provider_routing_policy.supported_markets()) == expected
    assert set(field_capability_registry.MARKET_ORDER) == expected


def test_runtime_order_surfaces_match_catalog_order() -> None:
    catalog_order = tuple(_catalog_codes())

    assert market_registry.supported_market_codes() == catalog_order
    assert market_queues.SUPPORTED_MARKETS == catalog_order


def test_catalog_exchange_codes_are_documented_as_mics_or_compatibility_aliases() -> None:
    catalog = get_market_catalog()

    for code in catalog.supported_market_codes():
        documented_codes = (
            DOCUMENTED_CATALOG_MIC_CODES.get(code, set())
            | DOCUMENTED_CATALOG_ALIAS_CODES.get(code, set())
        )
        assert set(catalog.get(code).exchanges) == documented_codes


def test_catalog_canonical_mics_are_documented_and_have_facts() -> None:
    catalog = get_market_catalog()

    for code in catalog.supported_market_codes():
        entry = catalog.get(code)
        assert entry.mics == DOCUMENTED_CATALOG_CANONICAL_MICS[code]
        assert entry.primary_mic == entry.mics[0]
        assert {facts.mic for facts in entry.mic_facts} == set(entry.mics)
        assert entry.default_currency in entry.supported_currencies


def test_active_universe_row_currencies_are_catalog_supported() -> None:
    db = _make_session()
    db.add_all(
        [
            StockUniverse(
                symbol="0700.HK",
                market="HK",
                exchange="XHKG",
                currency="HKD",
                timezone="Asia/Hong_Kong",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
            StockUniverse(
                symbol="SAP.DE",
                market="DE",
                exchange="XETR",
                currency="EUR",
                timezone="Europe/Berlin",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
        ]
    )
    db.commit()

    assert active_universe_currency_drift(db) == []
    db.close()


def test_active_universe_currency_drift_reports_unsupported_currency() -> None:
    db = _make_session()
    db.add(
        StockUniverse(
            symbol="BAD.HK",
            market="HK",
            exchange="XHKG",
            currency="EUR",
            timezone="Asia/Hong_Kong",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()

    assert active_universe_currency_drift(db) == [
        {
            "symbol": "BAD.HK",
            "market": "HK",
            "currency": "EUR",
            "supported_currencies": ("HKD",),
        }
    ]
    db.close()


def test_active_universe_timezone_drift_reports_mic_mismatch() -> None:
    db = _make_session()
    db.add_all(
        [
            StockUniverse(
                symbol="0700.HK",
                market="HK",
                exchange="XHKG",
                currency="HKD",
                timezone="Asia/Hong_Kong",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
            StockUniverse(
                symbol="BAD.SI",
                market="SG",
                exchange="XSES",
                currency="SGD",
                timezone="Asia/Hong_Kong",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
        ]
    )
    db.commit()

    assert active_universe_timezone_drift(db) == [
        {
            "symbol": "BAD.SI",
            "market": "SG",
            "mic": "XSES",
            "timezone": "Asia/Hong_Kong",
            "expected_timezone": "Asia/Singapore",
        }
    ]
    db.close()


def test_catalog_and_registry_exchange_alias_drift_is_documented() -> None:
    catalog = get_market_catalog()

    for code in catalog.supported_market_codes():
        catalog_exchanges = set(catalog.get(code).exchanges)
        registry_exchanges = set(market_registry.profile(code).exchanges)

        assert catalog_exchanges - registry_exchanges == (
            DOCUMENTED_CATALOG_ONLY_EXCHANGE_ALIASES.get(code, set())
        )
        assert registry_exchanges - catalog_exchanges == (
            DOCUMENTED_REGISTRY_ONLY_EXCHANGE_ALIASES.get(code, set())
        )


def test_bse_alias_ambiguity_is_documented_with_market_context() -> None:
    resolver = SecurityMasterResolver()

    india_bse = resolver.resolve_identity(symbol="500325", market="IN", exchange="BSE")
    china_bse = resolver.resolve_identity(symbol="920118", market="CN", exchange="BSE")

    assert india_bse.market == "IN"
    assert india_bse.canonical_symbol == "500325.BO"
    assert china_bse.market == "CN"
    assert china_bse.canonical_symbol == "920118.BJ"


def test_frontend_fallback_catalog_codes_match_backend_catalog_codes() -> None:
    assert _fallback_catalog_codes_from_frontend() == _catalog_codes()


def test_frontend_fallback_market_lists_match_backend_catalog_order() -> None:
    catalog_codes = _catalog_codes()

    assert _frontend_scan_geographic_market_codes() == catalog_codes
    assert list(_frontend_scan_market_options()) == catalog_codes
    assert (
        _frontend_breadth_object_market_keys("MARKET_LIVE_BENCHMARK_SYMBOLS")
        == catalog_codes
    )


def test_frontend_scan_market_labels_match_backend_catalog_labels() -> None:
    catalog = get_market_catalog()
    frontend_labels = _frontend_scan_market_options()

    assert frontend_labels == {
        code: catalog.get(code).label for code in catalog.supported_market_codes()
    }


def test_frontend_scan_scope_values_match_backend_catalog_facts() -> None:
    catalog = get_market_catalog()
    scopes_by_market = _frontend_scan_scope_options()

    assert list(scopes_by_market) == [*_catalog_codes(), "TEST"]
    assert scopes_by_market["TEST"] == []
    for code in catalog.supported_market_codes():
        entry = catalog.get(code)
        scopes = scopes_by_market[code]
        assert scopes[0] == ("market", f"All {entry.label}")

        exchange_values = [
            value.removeprefix("exchange:")
            for value, _label in scopes
            if value.startswith("exchange:")
        ]
        assert set(exchange_values) <= set(entry.exchanges)

        index_values = [
            (value.removeprefix("index:"), label)
            for value, label in scopes
            if value.startswith("index:")
        ]
        assert index_values == [
            (definition.key, definition.label)
            for definition in index_registry.definitions(code)
        ]


def test_official_universe_dispatch_matches_backend_catalog_capability() -> None:
    expected = set(_catalog_market_codes_by_capability("official_universe"))

    assert OFFICIAL_SOURCE_MARKETS == expected
    assert set(OFFICIAL_UNIVERSE_INGEST_METHODS) == expected
    assert set(OfficialMarketUniverseSourceService()._snapshot_fetchers()) == expected


def test_endpoint_capability_allowlists_match_catalog_capabilities() -> None:
    assert breadth.SUPPORTED_BREADTH_MARKETS == _catalog_market_codes_by_capability(
        "breadth"
    )
    assert groups.SUPPORTED_GROUP_MARKETS == _catalog_market_codes_by_capability(
        "group_rankings"
    )
    assert groups.SUPPORTED_RRG_MARKETS == _catalog_market_codes_by_rrg_scope(
        "groups"
    )
    assert scans.SUPPORTED_SCAN_REFRESH_MARKETS == _catalog_market_codes_by_capability(
        "feature_snapshot"
    )


def _assert_allowlist_assigned_from_capability_query(
    module,
    constant_name: str,
    capability_name: str,
) -> None:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == constant_name
            for target in node.targets
        ):
            continue
        call = node.value
        assert isinstance(call, ast.Call), (
            f"{module.__name__}.{constant_name} must be assigned from a "
            "Market Catalog capability query, not a local literal/comprehension"
        )
        function_name = (
            call.func.id
            if isinstance(call.func, ast.Name)
            else call.func.attr
            if isinstance(call.func, ast.Attribute)
            else None
        )
        assert function_name == "market_codes_with_capability"
        assert len(call.args) == 1
        capability_arg = call.args[0]
        assert isinstance(capability_arg, ast.Constant)
        assert capability_arg.value == capability_name
        return

    raise AssertionError(f"{module.__name__}.{constant_name} is missing")


def _assert_allowlist_assigned_from_rrg_scope_query(
    module,
    constant_name: str,
    scope: str,
) -> None:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == constant_name
            for target in node.targets
        ):
            continue
        call = node.value
        assert isinstance(call, ast.Call), (
            f"{module.__name__}.{constant_name} must be assigned from a "
            "Market Catalog RRG scope query, not a local literal/comprehension"
        )
        function_name = (
            call.func.id
            if isinstance(call.func, ast.Name)
            else call.func.attr
            if isinstance(call.func, ast.Attribute)
            else None
        )
        assert function_name == "market_codes_with_rrg_scope"
        assert len(call.args) == 1
        scope_arg = call.args[0]
        assert isinstance(scope_arg, ast.Constant)
        assert scope_arg.value == scope
        return

    raise AssertionError(f"{module.__name__}.{constant_name} is missing")


def test_endpoint_capability_allowlists_are_catalog_queries_not_local_lists() -> None:
    _assert_allowlist_assigned_from_capability_query(
        breadth,
        "SUPPORTED_BREADTH_MARKETS",
        "breadth",
    )
    _assert_allowlist_assigned_from_capability_query(
        groups,
        "SUPPORTED_GROUP_MARKETS",
        "group_rankings",
    )
    _assert_allowlist_assigned_from_rrg_scope_query(
        groups,
        "SUPPORTED_RRG_MARKETS",
        "groups",
    )
    _assert_allowlist_assigned_from_capability_query(
        scans,
        "SUPPORTED_SCAN_REFRESH_MARKETS",
        "feature_snapshot",
    )
