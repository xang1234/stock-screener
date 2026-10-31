"""Contract tests for the read-only calendar and earnings MCP tools.

Both tools expose facts that already exist in the backend: the canonical
``MarketCalendarService`` and the stored ``stock_fundamentals`` observation with
the canonical freshness rules from ``app.services.yahoo_earnings_calendar``.
The calendar is therefore a configured double per case - this file holds no
weekend, holiday or timezone rules - and the earnings rows are seeded, not
derived.

Deterministic by construction: the service takes a ``now_provider`` and the
calendar answers with the configured market moment, so no test reads the clock
or the network.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import event

from app.interfaces.mcp import market_copilot as market_copilot_module
from app.interfaces.mcp.market_copilot import MarketCopilotService, ToolSourceUnavailable
from app.services.market_calendar_service import CalendarCoverageExpired
from app.utils.market_hours import to_eastern_date
from app.models.stock import StockFundamental
from app.services import yahoo_earnings_calendar
from tests.helpers.mcp_fixture import (
    create_mcp_test_session_factory,
    seed_market_copilot_data,
)

NEW_YORK = ZoneInfo("America/New_York")
HONG_KONG = ZoneInfo("Asia/Hong_Kong")
FIXED_NOW = datetime(2026, 10, 15, 15, 0, tzinfo=UTC)  # 11:00 in New York

CALENDAR_GATEWAY_KEYS = {
    "market",
    "market_timezone",
    "market_now",
    "market_date",
    "is_trading_day",
    "last_completed_trading_day",
    "expected_trading_date",
    "calendar_source",
}

EARNINGS_GATEWAY_KEYS = {
    "symbol",
    "next_earnings_date",
    "event_calendar_as_of_date",
    "source",
    "freshness_status",
    "date_confirmation",
    "available",
    "reasons",
    "generated_at",
}


class FixedCalendar:
    """The canonical service's interface, with the facts configured per case.

    A test double, not a second calendar: the trading-day, holiday and
    timezone judgments stay in ``MarketCalendarService``.
    """

    def __init__(
        self,
        *,
        market_now: datetime,
        is_trading_day: bool,
        last_completed: date,
        supported: tuple[str, ...] = ("US",),
        is_open: bool | None = None,
    ) -> None:
        self._market_now = market_now
        self._is_trading_day = is_trading_day
        self._last_completed = last_completed
        self._supported = supported
        # Default: an open session exactly when the day trades, which is the
        # mid-session case; the pre-open case sets this to False.
        self._is_open = is_trading_day if is_open is None else is_open

    def normalize_market(self, market: str | None) -> str:
        normalized = (market or "US").strip().upper()
        if normalized not in self._supported:
            raise ValueError(f"Unsupported market: {market}")
        return normalized

    def market_timezone(self, market: str) -> ZoneInfo:
        return NEW_YORK

    def market_now(self, market: str, now: datetime | None = None) -> datetime:
        return self._market_now

    def is_trading_day(self, market: str, day: date | None = None) -> bool:
        return self._is_trading_day

    def last_completed_trading_day(self, market: str, now: datetime | None = None) -> date:
        return self._last_completed

    def is_market_open(self, market: str, now: datetime | None = None) -> bool:
        return self._is_open


class MarketAwareCalendar(FixedCalendar):
    """One market moment per market code, for the non-US reference date."""

    def __init__(self, moments: dict[str, datetime]) -> None:
        super().__init__(
            market_now=next(iter(moments.values())),
            is_trading_day=False,
            last_completed=date(2026, 10, 9),
            supported=tuple(moments),
        )
        self._moments = moments
        self.lookups: list[tuple[str, datetime | None]] = []

    def market_now(self, market: str, now: datetime | None = None) -> datetime:
        self.lookups.append((market, now))
        return self._moments[market]


class RaisingCalendar(FixedCalendar):
    """A calendar whose provider fails at one lookup - coverage, catalog drift."""

    def __init__(self, fail_on: str = "market_now", **kwargs) -> None:
        super().__init__(**kwargs)
        self._fail_on = fail_on

    def _fail(self, lookup: str) -> None:
        if lookup == self._fail_on:
            raise RuntimeError(f"calendar lookup failed: {lookup}")

    def market_now(self, market: str, now: datetime | None = None) -> datetime:
        self._fail("market_now")
        return super().market_now(market, now)

    def is_trading_day(self, market: str, day: date | None = None) -> bool:
        self._fail("is_trading_day")
        return super().is_trading_day(market, day)

    def last_completed_trading_day(self, market: str, now: datetime | None = None) -> date:
        self._fail("last_completed_trading_day")
        return super().last_completed_trading_day(market, now)

    def market_timezone(self, market: str) -> ZoneInfo:
        self._fail("market_timezone")
        return super().market_timezone(market)

    def is_market_open(self, market: str, now: datetime | None = None) -> bool:
        self._fail("is_market_open")
        return super().is_market_open(market, now)


def _factory():
    factory, _engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    return factory


def _service(session_factory, calendar=None, now: datetime = FIXED_NOW) -> MarketCopilotService:
    return MarketCopilotService(
        session_factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
        calendar_service=calendar
        or FixedCalendar(
            market_now=FIXED_NOW.astimezone(NEW_YORK),
            is_trading_day=True,
            last_completed=date(2026, 10, 14),
        ),
        now_provider=lambda: now,
    )


def _payload(service: MarketCopilotService, tool_name: str, arguments: dict) -> dict:
    result = service.call_tool(tool_name, arguments)
    assert result.get("isError") is not True, result["content"][0]["text"]
    return result["structuredContent"]


def _error(service: MarketCopilotService, tool_name: str, arguments: dict) -> dict:
    result = service.call_tool(tool_name, arguments)
    assert result.get("isError") is True, result["content"][0]["text"]
    return result["structuredContent"]["error"]


def _seed(session_factory, rows: list[dict]) -> None:
    session = session_factory()
    try:
        for row in rows:
            session.add(StockFundamental(**row))
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def session_factory():
    return _factory()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_both_tools_are_registered_and_listed(session_factory):
    listed = {tool["name"]: tool for tool in _service(session_factory).list_tools()}

    for name in ("market_calendar_status", "earnings_lookup"):
        assert name in listed
        assert listed[name]["description"]
        assert listed[name]["inputSchema"]["type"] == "object"


def test_calendar_tool_defaults_to_the_us_market(session_factory):
    payload = _payload(_service(session_factory), "market_calendar_status", {})

    assert payload["market"] == "US"


# ---------------------------------------------------------------------------
# market_calendar_status
# ---------------------------------------------------------------------------

CALENDAR_CASES = [
    # moment UTC, trading day?, last completed, market date, expected, open?, phase
    pytest.param(
        datetime(2026, 10, 15, 15, 0, tzinfo=UTC),  # 11:00 ET, session running
        True,
        date(2026, 10, 14),
        "2026-10-15",
        "2026-10-15",
        True,
        "is open",
        id="trading-day-mid-session",
    ),
    pytest.param(
        datetime(2026, 10, 15, 12, 0, tzinfo=UTC),  # 08:00 ET, before the open
        True,
        date(2026, 10, 14),
        "2026-10-15",
        "2026-10-14",
        False,
        "has not opened yet",
        id="trading-day-pre-open",
    ),
    pytest.param(
        datetime(2026, 10, 15, 22, 0, tzinfo=UTC),  # 18:00 ET, after the close
        True,
        date(2026, 10, 15),
        "2026-10-15",
        "2026-10-15",
        False,
        "is complete",
        id="trading-day-after-close",
    ),
    pytest.param(
        datetime(2026, 10, 17, 12, 0, tzinfo=UTC),  # Saturday
        False,
        date(2026, 10, 16),
        "2026-10-17",
        "2026-10-16",
        False,
        "has no session",
        id="weekend",
    ),
    pytest.param(
        datetime(2026, 11, 26, 15, 0, tzinfo=UTC),  # Thanksgiving
        False,
        date(2026, 11, 25),
        "2026-11-26",
        "2026-11-25",
        False,
        "has no session",
        id="exchange-holiday",
    ),
    pytest.param(
        datetime(2026, 10, 15, 1, 30, tzinfo=UTC),  # 21:30 ET on the 14th
        True,
        date(2026, 10, 14),
        "2026-10-14",
        "2026-10-14",
        False,
        "is complete",
        id="market-timezone-boundary",
    ),
]


@pytest.mark.parametrize(
    ("moment_utc", "trading", "last_completed", "market_date", "expected", "open_session", "phase"),
    CALENDAR_CASES,
)
def test_the_calendar_reports_the_expected_trading_date(
    session_factory,
    moment_utc,
    trading,
    last_completed,
    market_date,
    expected,
    open_session,
    phase,
):
    """``expected_trading_date`` names the session whose bars can exist.

    Mid-session that is today, because the session is producing bars. Before
    the open it is the last completed session - otherwise an up-to-date cache
    would look stale until the session trades. After the close the last
    completed session is today again.
    """
    calendar = FixedCalendar(
        market_now=moment_utc.astimezone(NEW_YORK),
        is_trading_day=trading,
        last_completed=last_completed,
        is_open=open_session,
    )
    service = _service(session_factory, calendar, now=moment_utc)

    payload = _payload(service, "market_calendar_status", {"market": "us"})

    assert payload["market"] == "US"
    assert payload["market_timezone"] == "America/New_York"
    assert payload["market_now"] == moment_utc.astimezone(NEW_YORK).isoformat()
    assert payload["market_date"] == market_date
    assert payload["is_trading_day"] is trading
    assert payload["last_completed_trading_day"] == last_completed.isoformat()
    assert payload["expected_trading_date"] == expected
    assert payload["calendar_source"] == "xang_market_calendar_service"
    # A consumer must be able to tell the phase from the summary.
    assert phase in payload["summary"]


def test_an_unsupported_market_is_rejected_as_bad_arguments(session_factory):
    error = _error(_service(session_factory), "market_calendar_status", {"market": "XX"})

    assert error["code"] == "invalid_arguments"
    assert "expected_trading_date" not in error["details"]


@pytest.mark.parametrize(
    "fail_on",
    ["market_now", "is_trading_day", "last_completed_trading_day", "market_timezone"],
)
def test_an_unavailable_calendar_fails_closed(session_factory, fail_on):
    """Every lookup inside the block maps to the same code.

    A regression that narrowed the ``try`` block, or that let a later lookup
    raise something else, must fail this test.
    """
    service = _service(
        session_factory,
        RaisingCalendar(
            fail_on=fail_on,
            market_now=FIXED_NOW.astimezone(NEW_YORK),
            is_trading_day=True,
            last_completed=date(2026, 10, 14),
        ),
    )

    result = service.call_tool("market_calendar_status", {"market": "US"})

    assert result.get("isError") is True
    payload = result["structuredContent"]
    assert payload["error"]["code"] == "calendar_unavailable"
    # No half-filled success payload: there is no date to read.
    assert "expected_trading_date" not in payload
    assert "calendar lookup failed" not in payload["error"]["message"]


def test_the_calendar_payload_keeps_the_gateway_keys(session_factory):
    """Consumers of the temporary gateway read these keys by name."""
    payload = _payload(_service(session_factory), "market_calendar_status", {"market": "US"})

    assert CALENDAR_GATEWAY_KEYS <= set(payload)
    fact_keys = {fact["key"] for fact in payload["facts"]}
    assert CALENDAR_GATEWAY_KEYS - {"calendar_source"} <= fact_keys


@pytest.mark.parametrize(
    ("moment", "market_date", "trading", "last_completed"),
    [
        pytest.param(
            datetime(2026, 10, 15, 15, 0, tzinfo=UTC),
            "2026-10-15",
            True,
            "2026-10-14",
            id="session-day",
        ),
        pytest.param(
            datetime(2026, 10, 17, 12, 0, tzinfo=UTC),
            "2026-10-17",
            False,
            "2026-10-16",
            id="saturday",
        ),
        pytest.param(
            datetime(2026, 11, 26, 15, 0, tzinfo=UTC),
            "2026-11-26",
            False,
            "2026-11-25",
            id="thanksgiving",
        ),
    ],
)
def test_the_canonical_service_confirms_the_expected_date_semantics(
    session_factory, moment, market_date, trading, last_completed
):
    """The contract holds against the real calendar, not only the double."""
    service = MarketCopilotService(
        session_factory,
        SimpleNamespace(mcp_watchlist_writes_enabled=False, mcp_server_name="test"),
        now_provider=lambda: moment,
    )

    payload = _payload(service, "market_calendar_status", {"market": "US"})

    assert payload["market_date"] == market_date
    assert payload["is_trading_day"] is trading
    assert payload["last_completed_trading_day"] == last_completed
    assert payload["expected_trading_date"] == (market_date if trading else last_completed)


def test_a_non_us_market_uses_its_own_timezone_and_sessions(session_factory):
    """Japan is closed on 2026-09-23 while the US trades.

    The double answers with New York for every market, so only the real service
    can show that the handler passes the market to every lookup instead of
    hardcoding one calendar. The expected trading date is not asserted for JP -
    that needs the exchange calendar checked separately.
    """
    moment = datetime(2026, 9, 23, 4, 0, tzinfo=UTC)  # 13:00 JST, 00:00 EDT
    service = MarketCopilotService(
        session_factory,
        SimpleNamespace(mcp_watchlist_writes_enabled=False, mcp_server_name="test"),
        now_provider=lambda: moment,
    )

    japan = _payload(service, "market_calendar_status", {"market": "JP"})
    states = _payload(service, "market_calendar_status", {"market": "US"})

    assert japan["market_timezone"] == "Asia/Tokyo"
    assert japan["market_date"] == "2026-09-23"
    assert japan["is_trading_day"] is False
    assert japan["last_completed_trading_day"] == "2026-09-18"
    # Closed day: the expected date is the last completed session, not the day.
    assert japan["expected_trading_date"] == "2026-09-18"
    # The same moment, the other calendar: a hardcoded market would match here.
    assert states["is_trading_day"] is True
    assert states["last_completed_trading_day"] == "2026-09-22"


# ---------------------------------------------------------------------------
# earnings_lookup
# ---------------------------------------------------------------------------

EARNINGS_ROWS = [
    # current observation, date that looks confirmed - must stay unknown
    {
        "symbol": "AAPL",
        "next_earnings_date": date(2026, 11, 5),
        "event_calendar_as_of_date": date(2026, 10, 10),
    },
    # current observation of "nothing upcoming"
    {
        "symbol": "MSFT",
        "next_earnings_date": None,
        "event_calendar_as_of_date": date(2026, 10, 12),
    },
    # 25 days old, beyond the 14-day window
    {
        "symbol": "NVDA",
        "next_earnings_date": date(2026, 11, 20),
        "event_calendar_as_of_date": date(2026, 9, 20),
    },
    # observation dated in the future, beyond the 3-day tolerance
    {
        "symbol": "PANW",
        "next_earnings_date": date(2026, 11, 1),
        "event_calendar_as_of_date": date(2026, 10, 20),
    },
    # current observation, stored date already in the past
    {
        "symbol": "SNOW",
        "next_earnings_date": date(2026, 10, 1),
        "event_calendar_as_of_date": date(2026, 10, 14),
    },
    # no observation at all: provider failure and empty response look alike
    {
        "symbol": "AVGO",
        "next_earnings_date": date(2026, 12, 1),
        "event_calendar_as_of_date": None,
    },
    # exactly at the future tolerance (reference 2026-10-15 + 3): still fresh
    {
        "symbol": "TJX",
        "next_earnings_date": date(2026, 11, 10),
        "event_calendar_as_of_date": date(2026, 10, 18),
    },
]

EARNINGS_CASES = [
    pytest.param(
        "aapl", "ok", True, ["date_confirmation_unrecorded"], "2026-11-05", id="current-with-date"
    ),
    pytest.param(
        "MSFT",
        "ok",
        True,
        ["no_upcoming_earnings_date", "date_confirmation_unrecorded"],
        None,
        id="current-no-date",
    ),
    pytest.param("ZZZZ", "unknown", False, ["symbol_not_found"], None, id="unknown-symbol"),
    pytest.param("NVDA", "stale", False, ["observation_stale"], None, id="stale-observation"),
    pytest.param(
        "PANW", "invalid", False, ["observation_date_in_future"], None, id="future-observation"
    ),
    pytest.param("SNOW", "stale", False, ["earnings_date_in_past"], None, id="past-stored-date"),
    pytest.param("AVGO", "unknown", False, ["observation_date_missing"], None, id="missing-observation"),
    pytest.param(
        "TJX", "ok", True, ["date_confirmation_unrecorded"], "2026-11-10", id="at-future-tolerance"
    ),
]


@pytest.fixture()
def earnings_factory():
    factory = _factory()
    _seed(factory, EARNINGS_ROWS)
    return factory


@pytest.mark.parametrize(("symbol", "freshness", "available", "reasons", "next_date"), EARNINGS_CASES)
def test_the_stored_observation_is_reported_with_its_freshness(
    earnings_factory, symbol, freshness, available, reasons, next_date
):
    """Every state keeps a date unconfirmed and names what is missing."""
    payload = _payload(_service(earnings_factory), "earnings_lookup", {"symbol": symbol})

    assert payload["symbol"] == symbol.upper()
    assert payload["freshness_status"] == freshness
    assert payload["available"] is available
    # Exact list: an extra or dropped reason must fail the test.
    assert payload["reasons"] == reasons
    # A stored or current date is never a confirmation.
    assert payload["date_confirmation"] == "unknown"
    assert payload["next_earnings_date"] == next_date


def test_the_earnings_reference_date_is_the_eastern_date(earnings_factory):
    """22:00 in New York is already the next day in UTC.

    The stored date is still today's, so it must not be called past.
    """
    _seed(
        earnings_factory,
        [
            {
                "symbol": "ORCL",
                "next_earnings_date": date(2026, 10, 15),
                "event_calendar_as_of_date": date(2026, 10, 14),
            }
        ],
    )
    after_midnight_utc = datetime(2026, 10, 16, 2, 0, tzinfo=UTC)  # 22:00 ET, Oct 15
    service = _service(earnings_factory, now=after_midnight_utc)

    payload = _payload(service, "earnings_lookup", {"symbol": "ORCL"})

    assert payload["freshness_status"] == "ok"
    assert payload["next_earnings_date"] == "2026-10-15"
    assert payload["observation_age_days"] == 1


def test_a_padded_symbol_is_normalized_before_the_length_check(earnings_factory):
    payload = _payload(_service(earnings_factory), "earnings_lookup", {"symbol": "   aapl   "})

    assert payload["symbol"] == "AAPL"
    assert payload["freshness_status"] == "ok"


@pytest.mark.parametrize(
    "arguments",
    [pytest.param({"symbol": "AAPL; DROP TABLE"}, id="malformed"), pytest.param({}, id="missing")],
)
def test_bad_arguments_are_refused_before_any_lookup(session_factory, arguments):
    error = _error(_service(session_factory), "earnings_lookup", arguments)

    assert error["code"] == "invalid_arguments"


@pytest.mark.parametrize("symbol", ["0700.HK", "7203.T", "2330.TW", "BRK.B"])
def test_stored_provider_symbols_are_accepted(earnings_factory, symbol):
    """HK/JP/TW codes start with a digit or carry a suffix - all valid."""
    result = _service(earnings_factory).call_tool("earnings_lookup", {"symbol": symbol})

    assert result.get("isError") is not True, result["content"][0]["text"]
    assert result["structuredContent"]["symbol"] == symbol


def test_a_suffixed_symbol_matches_its_stored_row(earnings_factory):
    """Acceptance is not enough: the row must actually be found and read."""
    _seed(earnings_factory, [
        {
            "symbol": "0700.HK",
            "next_earnings_date": date(2026, 11, 12),
            "event_calendar_as_of_date": date(2026, 10, 10),
        }
    ])
    service = _service(earnings_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "0700.HK"})

    assert payload["next_earnings_date"] == "2026-11-12"
    assert payload["freshness_status"] == "ok"
    assert payload["available"] is True


def test_the_earnings_reference_date_follows_the_symbol_market(session_factory):
    """08:00 HKT on the 11th is not judged against the Eastern 10th."""
    calendar = MarketAwareCalendar(
        {
            "US": datetime(2026, 10, 10, 20, 0, tzinfo=NEW_YORK),
            "HK": datetime(2026, 10, 11, 8, 0, tzinfo=HONG_KONG),
            "JP": datetime(2026, 10, 11, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
            "TW": datetime(2026, 10, 11, 9, 0, tzinfo=ZoneInfo("Asia/Taipei")),
        }
    )
    service = _service(session_factory, calendar)

    # No suffix: the Eastern calendar date of the lookup moment.
    assert service._earnings_reference_date("AAPL", FIXED_NOW) == (date(2026, 10, 15), "eastern")
    assert service._earnings_reference_date("0700.HK", FIXED_NOW) == (date(2026, 10, 11), "market")
    assert service._earnings_reference_date("7203.T", FIXED_NOW) == (date(2026, 10, 11), "market")
    assert service._earnings_reference_date("2330.TW", FIXED_NOW) == (date(2026, 10, 11), "market")
    # Every non-US lookup is made with the caller's moment, not the clock.
    assert calendar.lookups == [("HK", FIXED_NOW), ("JP", FIXED_NOW), ("TW", FIXED_NOW)]


def test_the_reference_basis_names_an_eastern_fallback(earnings_factory):
    """A suffixed symbol whose calendar fails says so in the payload."""
    class BrokenMarketCalendar(FixedCalendar):
        def market_now(self, market: str, now: datetime | None = None) -> datetime:
            raise RuntimeError("coverage expired")

    service = _service(earnings_factory, BrokenMarketCalendar(
        market_now=FIXED_NOW.astimezone(NEW_YORK),
        is_trading_day=True,
        last_completed=date(2026, 10, 14),
        supported=("HK",),
    ))

    payload = _payload(service, "earnings_lookup", {"symbol": "0700.HK"})

    assert payload["freshness_reference_basis"] == "eastern_fallback"
    assert payload["freshness_reference_date"] == to_eastern_date(FIXED_NOW).isoformat()


def test_a_broken_runtime_calendar_is_not_masked_by_a_private_instance(
    session_factory, monkeypatch
):
    """Only 'services not initialised' may fall back; expired coverage may not."""
    built: list[int] = []

    def exploding_resolver():
        raise CalendarCoverageExpired(
            market="US",
            requested_date=date(2026, 10, 15),
            verified_through=date(2026, 10, 1),
            source_url="fixture",
            operations_doc="fixture",
        )

    monkeypatch.setattr(market_copilot_module, "get_market_calendar_service", exploding_resolver)
    monkeypatch.setattr(
        market_copilot_module, "MarketCalendarService", lambda *a, **k: built.append(1)
    )
    service = MarketCopilotService(
        session_factory,
        SimpleNamespace(mcp_watchlist_writes_enabled=False, mcp_server_name="test"),
        now_provider=lambda: FIXED_NOW,
    )

    with pytest.raises(ToolSourceUnavailable) as caught:
        service._calendar()

    # The canonical failure is reported as its own cause, and no private
    # instance was built to hide it.
    assert isinstance(caught.value.__cause__, CalendarCoverageExpired)
    assert built == []


def test_missing_runtime_services_still_build_the_canonical_calendar(
    session_factory, monkeypatch
):
    """The stdio process has no FastAPI startup, so it builds the same class."""
    built: list[object] = []
    calendar = FixedCalendar(
        market_now=FIXED_NOW.astimezone(NEW_YORK),
        is_trading_day=True,
        last_completed=date(2026, 10, 14),
    )

    def no_runtime_services():
        raise RuntimeError(
            "RuntimeServices are not initialized for this context. "
            "Use the FastAPI lifespan or runtime_context.use_runtime_services()."
        )

    monkeypatch.setattr(
        market_copilot_module, "get_market_calendar_service", no_runtime_services
    )
    monkeypatch.setattr(
        market_copilot_module,
        "MarketCalendarService",
        lambda *a, **k: built.append(calendar) or calendar,
    )
    service = MarketCopilotService(
        session_factory,
        SimpleNamespace(mcp_watchlist_writes_enabled=False, mcp_server_name="test"),
        now_provider=lambda: FIXED_NOW,
    )

    payload = _payload(service, "market_calendar_status", {"market": "US"})

    assert built == [calendar]
    assert payload["market"] == "US"


def test_a_database_failure_fails_closed(session_factory):
    def broken_factory():
        raise RuntimeError("connection refused for postgresql://user:hunter2@db/app")

    service = _service(broken_factory)

    result = service.call_tool("earnings_lookup", {"symbol": "AAPL"})

    assert result.get("isError") is True
    payload = result["structuredContent"]
    assert payload["error"]["code"] == "database_unavailable"
    # No connection string, no driver text in the answer.
    assert "postgresql" not in payload["error"]["message"]
    assert "hunter2" not in result["content"][0]["text"]


def test_the_published_thresholds_come_from_the_canonical_module(earnings_factory):
    payload = _payload(_service(earnings_factory), "earnings_lookup", {"symbol": "AAPL"})

    assert payload["event_calendar_max_age_days"] == yahoo_earnings_calendar.EVENT_CALENDAR_MAX_AGE_DAYS
    assert (
        payload["event_calendar_producer_refresh_after_days"]
        == yahoo_earnings_calendar.EVENT_CALENDAR_PRODUCER_REFRESH_AFTER_DAYS
    )
    assert (
        payload["event_calendar_future_tolerance_days"]
        == yahoo_earnings_calendar.EVENT_CALENDAR_FUTURE_TOLERANCE_DAYS
    )


def test_the_earnings_payload_keeps_the_gateway_keys(earnings_factory):
    payload = _payload(_service(earnings_factory), "earnings_lookup", {"symbol": "AAPL"})

    assert EARNINGS_GATEWAY_KEYS <= set(payload)
    assert payload["source"] == "xang_yahoo_calendar"
    fact_keys = {fact["key"] for fact in payload["facts"]}
    assert {
        "symbol",
        "freshness_status",
        "date_confirmation",
        "available",
        "event_calendar_as_of_date",
        "next_earnings_date",
        "observation_age_days",
    } <= fact_keys
    assert all(fact["source"] == "xang_yahoo_calendar" for fact in payload["facts"])
    # Values, not just keys: _facts must copy what the payload says.
    for fact in payload["facts"]:
        assert fact["value"] == payload[fact["key"]]


def test_the_lookup_only_emits_select_statements():
    """Read-only means read-only: no INSERT, UPDATE or DELETE reaches the driver."""
    factory, engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    _seed(factory, EARNINGS_ROWS)
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        statements.append(statement)

    service = _service(factory)
    _payload(service, "earnings_lookup", {"symbol": "AAPL"})
    _payload(service, "earnings_lookup", {"symbol": "ZZZZ"})

    assert statements, "the lookup must read the stored row"
    assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
