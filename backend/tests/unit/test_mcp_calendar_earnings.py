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

from app.interfaces.mcp.market_copilot import MarketCopilotService
from app.models.stock import StockFundamental
from app.services import yahoo_earnings_calendar
from tests.helpers.mcp_fixture import (
    create_mcp_test_session_factory,
    seed_market_copilot_data,
)

NEW_YORK = ZoneInfo("America/New_York")
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
    ) -> None:
        self._market_now = market_now
        self._is_trading_day = is_trading_day
        self._last_completed = last_completed
        self._supported = supported

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


class RaisingCalendar(FixedCalendar):
    """A calendar whose provider fails - coverage expired, provider drift."""

    def market_now(self, market: str, now: datetime | None = None) -> datetime:
        raise RuntimeError("calendar coverage expired for this day")


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
    # moment in UTC, trading day?, last completed session, market date, expected
    pytest.param(
        datetime(2026, 10, 15, 15, 0, tzinfo=UTC),  # 11:00 ET, session running
        True,
        date(2026, 10, 14),
        "2026-10-15",
        "2026-10-15",
        True,
        id="trading-day-mid-session",
    ),
    pytest.param(
        datetime(2026, 10, 15, 12, 0, tzinfo=UTC),  # 08:00 ET, before the open
        True,
        date(2026, 10, 14),
        "2026-10-15",
        "2026-10-15",
        True,
        id="trading-day-pre-open",
    ),
    pytest.param(
        datetime(2026, 10, 17, 12, 0, tzinfo=UTC),  # Saturday
        False,
        date(2026, 10, 16),
        "2026-10-17",
        "2026-10-16",
        False,
        id="weekend",
    ),
    pytest.param(
        datetime(2026, 11, 26, 15, 0, tzinfo=UTC),  # Thanksgiving
        False,
        date(2026, 11, 25),
        "2026-11-26",
        "2026-11-25",
        False,
        id="exchange-holiday",
    ),
    pytest.param(
        datetime(2026, 10, 15, 1, 30, tzinfo=UTC),  # 21:30 ET on the 14th
        True,
        date(2026, 10, 13),
        "2026-10-14",
        "2026-10-14",
        True,
        id="market-timezone-boundary",
    ),
]


@pytest.mark.parametrize(
    ("moment_utc", "trading", "last_completed", "market_date", "expected", "session_running"),
    CALENDAR_CASES,
)
def test_the_calendar_reports_the_expected_trading_date(
    session_factory, moment_utc, trading, last_completed, market_date, expected, session_running
):
    """``expected_trading_date`` is today's session, not yesterday's close.

    On a trading day - mid-session and pre-open alike, matching the gateway
    contract - the expected date is the market's own calendar day. Only on a
    day without a session does it fall back to the last completed one.
    """
    calendar = FixedCalendar(
        market_now=moment_utc.astimezone(NEW_YORK),
        is_trading_day=trading,
        last_completed=last_completed,
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
    # A consumer must be able to tell "today's session counts" from "closed".
    assert ("not the last completed one" in payload["summary"]) is session_running


def test_an_unsupported_market_is_rejected_as_bad_arguments(session_factory):
    error = _error(_service(session_factory), "market_calendar_status", {"market": "XX"})

    assert error["code"] == "invalid_arguments"
    assert "expected_trading_date" not in error["details"]


def test_an_unavailable_calendar_fails_closed(session_factory):
    service = _service(
        session_factory,
        RaisingCalendar(
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
    assert "calendar coverage" not in payload["error"]["message"]


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
]

EARNINGS_CASES = [
    pytest.param("aapl", "ok", True, "date_confirmation_unrecorded", "2026-11-05", id="current-with-date"),
    pytest.param("MSFT", "ok", True, "no_upcoming_earnings_date", None, id="current-no-date"),
    pytest.param("ZZZZ", "unknown", False, "symbol_not_found", None, id="unknown-symbol"),
    pytest.param("NVDA", "stale", False, "observation_stale", None, id="stale-observation"),
    pytest.param("PANW", "invalid", False, "observation_date_in_future", None, id="future-observation"),
    pytest.param("SNOW", "invalid", False, "earnings_date_in_past", None, id="past-stored-date"),
    pytest.param("AVGO", "unknown", False, "observation_date_missing", None, id="missing-observation"),
]


@pytest.fixture()
def earnings_factory():
    factory = _factory()
    _seed(factory, EARNINGS_ROWS)
    return factory


@pytest.mark.parametrize(("symbol", "freshness", "available", "reason", "next_date"), EARNINGS_CASES)
def test_the_stored_observation_is_reported_with_its_freshness(
    earnings_factory, symbol, freshness, available, reason, next_date
):
    """Every state keeps a date unconfirmed and names what is missing."""
    payload = _payload(_service(earnings_factory), "earnings_lookup", {"symbol": symbol})

    assert payload["symbol"] == symbol.upper()
    assert payload["freshness_status"] == freshness
    assert payload["available"] is available
    assert reason in payload["reasons"]
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
