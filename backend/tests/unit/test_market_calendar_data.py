from __future__ import annotations

from datetime import date, time
from pathlib import Path
from urllib.parse import urlparse

import pytest

from app.domain.markets.calendar_coverage import CalendarCoverageRegistry
from app.domain.markets.catalog import get_market_catalog
from app.domain.markets.reviewed_calendar_input import ReviewedCalendarInput
from app.services.market_calendar_service import MarketCalendarService


CALENDAR_ROOT = Path(__file__).resolve().parents[2] / "data" / "market_calendars"
REVIEWED_INPUT = (
    CALENDAR_ROOT / "inputs" / "reviewed_official_calendars.json"
)

OFFICIAL_SOURCE_DOMAINS = {
    "US": "nyse.com",
    "HK": "hkex.com.hk",
    "IN": "nseindia.com",
    "JP": "jpx.co.jp",
    "KR": "krx.co.kr",
    "TW": "twse.com.tw",
    "CN": "szse.cn",
    "CA": "tsx.com",
    "DE": "deutsche-boerse.com",
    "SG": "sgx.com",
    "AU": "asx.com.au",
    "MY": "bursamalaysia.com",
}

KNOWN_2026_CLOSURES = {
    "KR": (date(2026, 5, 25), date(2026, 6, 3), date(2026, 7, 17)),
    "TW": (date(2026, 2, 12), date(2026, 2, 20)),
    "SG": (
        date(2026, 1, 1),
        date(2026, 2, 17),
        date(2026, 2, 18),
        date(2026, 4, 3),
        date(2026, 5, 1),
        date(2026, 6, 1),
    ),
    "MY": (date(2026, 3, 23), date(2026, 6, 1), date(2026, 6, 17)),
    "CN": (
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 2, 16),
        date(2026, 2, 17),
        date(2026, 2, 18),
        date(2026, 2, 19),
        date(2026, 2, 20),
        date(2026, 2, 23),
        date(2026, 4, 6),
        date(2026, 5, 1),
        date(2026, 5, 4),
        date(2026, 5, 5),
        date(2026, 6, 19),
    ),
    "JP": (date(2026, 3, 20), date(2026, 9, 22), date(2026, 9, 23)),
}

VERIFIED_EXCEPTIONAL_CLOSES = {
    "AU": {
        date(2026, 12, 24): time(14, 10),
        date(2026, 12, 31): time(14, 10),
    },
    "CA": {date(2026, 12, 24): time(13, 0)},
    "HK": {
        date(2026, 2, 16): time(12, 0),
        date(2026, 12, 24): time(12, 0),
        date(2026, 12, 31): time(12, 0),
    },
    "SG": {
        date(2026, 2, 16): time(12, 0),
        date(2026, 12, 24): time(12, 0),
        date(2026, 12, 31): time(12, 0),
    },
    "US": {
        date(2026, 11, 27): time(13, 0),
        date(2026, 12, 24): time(13, 0),
        date(2027, 11, 26): time(13, 0),
        date(2028, 7, 3): time(13, 0),
        date(2028, 11, 24): time(13, 0),
    },
}


@pytest.fixture(scope="module")
def registry() -> CalendarCoverageRegistry:
    return CalendarCoverageRegistry.load(CALENDAR_ROOT)


def test_repository_calendar_data_covers_every_supported_market(
    registry: CalendarCoverageRegistry,
) -> None:
    catalog = get_market_catalog()

    for market in catalog.supported_market_codes():
        coverage = registry.coverage_for(market)
        assert coverage.annual[2026].status == "official"
        assert coverage.verified_through >= date(2026, 12, 31)
        assert 2030 in coverage.annual


def test_repository_official_calendars_use_first_party_sources(
    registry: CalendarCoverageRegistry,
) -> None:
    for market, expected_domain in OFFICIAL_SOURCE_DOMAINS.items():
        annual = registry.coverage_for(market).annual[2026]
        hostname = urlparse(annual.source.url).hostname or ""
        assert hostname == expected_domain or hostname.endswith(
            f".{expected_domain}"
        )


@pytest.mark.parametrize(
    ("market", "closed_days"),
    KNOWN_2026_CLOSURES.items(),
)
def test_repository_official_sessions_exclude_known_2026_closures(
    registry: CalendarCoverageRegistry,
    market: str,
    closed_days: tuple[date, ...],
) -> None:
    sessions = set(
        registry.official_sessions(
            market,
            date(2026, 1, 1),
            date(2026, 12, 31),
        )
    )

    assert sessions
    assert sessions.isdisjoint(closed_days)


def test_repository_uses_available_official_forward_coverage(
    registry: CalendarCoverageRegistry,
) -> None:
    assert registry.coverage_for("US").verified_through == date(2028, 12, 31)
    assert registry.coverage_for("JP").verified_through == date(2027, 12, 31)
    assert registry.coverage_for("DE").verified_through == date(2030, 12, 31)


@pytest.mark.parametrize(
    ("market", "expected"),
    VERIFIED_EXCEPTIONAL_CLOSES.items(),
)
def test_repository_preserves_verified_exceptional_closes(
    registry: CalendarCoverageRegistry,
    market: str,
    expected: dict[date, time],
) -> None:
    actual = {
        exception_date: close_time
        for annual in registry.coverage_for(market).annual.values()
        if annual.status == "official"
        for exception_date, close_time in annual.close_exceptions.items()
    }

    assert actual == expected


def test_every_official_year_has_explicit_reviewed_closure_inputs(
    registry: CalendarCoverageRegistry,
) -> None:
    official_years = {
        (coverage.market, year)
        for coverage in registry.coverages()
        for year, annual in coverage.annual.items()
        if annual.status == "official"
    }

    reviewed = ReviewedCalendarInput.load(
        REVIEWED_INPUT,
        market_catalog=get_market_catalog(),
        first_year=2026,
    )
    reviewed_years = {
        (market, year)
        for market, facts in reviewed.markets.items()
        for year in facts.closures
    }

    assert reviewed_years == official_years


def test_singapore_provisional_calendar_excludes_fixed_holidays(
    registry: CalendarCoverageRegistry,
) -> None:
    annual = registry.coverage_for("SG").annual[2027]

    assert annual.status == "provisional"
    assert date(2027, 1, 1) not in annual.sessions
    assert date(2027, 3, 26) not in annual.sessions  # Good Friday
    assert date(2027, 8, 9) not in annual.sessions


@pytest.mark.parametrize(
    ("start", "end", "sessions"),
    [
        (date(2026, 1, 1), date(2026, 1, 5), [date(2026, 1, 2), date(2026, 1, 5)]),
        (date(2026, 7, 9), date(2026, 7, 13), [date(2026, 7, 9), date(2026, 7, 13)]),
        (date(2026, 10, 22), date(2026, 10, 27), [date(2026, 10, 22), date(2026, 10, 27)]),
    ],
)
def test_tw_2026_matches_sessions_twse_reported(start, end, sessions) -> None:
    assert MarketCalendarService().trading_days("TW", start, end) == sessions
