"""Contract tests for the read-only calendar and earnings MCP tools.

Both tools expose facts that already exist in the backend: the canonical
``MarketCalendarService`` and the stored ``stock_fundamentals`` observation with
the canonical freshness rules from ``app.services.yahoo_earnings_calendar``.
The tests therefore configure a calendar double per case - they do not
re-implement weekend, holiday or timezone rules - and seed stored rows instead
of deriving dates.

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
REFERENCE = FIXED_NOW.date()


class FixedCalendar:
    """The canonical service's interface, with the facts configured per case.

    This is a test double, not a second calendar: the trading-day, holiday and
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


@pytest.fixture()
def session_factory():
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


def _seed_fundamentals(session_factory, rows: list[dict]) -> None:
    session = session_factory()
    try:
        for row in rows:
            session.add(StockFundamental(**row))
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_both_tools_are_registered_and_listed(session_factory):
    service = _service(session_factory)

    listed = {tool["name"]: tool for tool in service.list_tools()}

    for name in ("market_calendar_status", "earnings_lookup"):
        assert name in listed
        assert listed[name]["description"]
        assert listed[name]["inputSchema"]["type"] == "object"


def test_calendar_tool_defaults_to_the_us_market(session_factory):
    service = _service(session_factory)

    payload = _payload(service, "market_calendar_status", {})

    assert payload["market"] == "US"


# ---------------------------------------------------------------------------
# market_calendar_status
# ---------------------------------------------------------------------------


def test_a_running_session_expects_today_not_the_last_completed_day(session_factory):
    """Mid-session the expected date is the session running now.

    ``last_completed_trading_day`` only moves after the close, so returning it
    during the session would make a consumer think today's run is stale.
    """
    service = _service(session_factory)

    payload = _payload(service, "market_calendar_status", {"market": "us"})

    assert payload["market_date"] == "2026-10-15"
    assert payload["is_trading_day"] is True
    assert payload["last_completed_trading_day"] == "2026-10-14"
    assert payload["expected_trading_date"] == "2026-10-15"
    assert payload["market"] == "US"
    assert payload["calendar_source"] == "xang_market_calendar_service"
    assert payload["market_timezone"] == "America/New_York"
    assert payload["market_now"].startswith("2026-10-15T11:00:00-04:00")
    assert "still running" in payload["summary"]


def test_a_weekend_answers_with_the_last_completed_session(session_factory):
    saturday = datetime(2026, 10, 17, 12, 0, tzinfo=NEW_YORK)
    service = _service(
        session_factory,
        FixedCalendar(
            market_now=saturday,
            is_trading_day=False,
            last_completed=date(2026, 10, 16),
        ),
        now=saturday.astimezone(UTC),
    )

    payload = _payload(service, "market_calendar_status", {"market": "US"})

    assert payload["market_date"] == "2026-10-17"
    assert payload["is_trading_day"] is False
    assert payload["last_completed_trading_day"] == "2026-10-16"
    assert payload["expected_trading_date"] == "2026-10-16"


def test_an_exchange_holiday_answers_with_the_last_completed_session(session_factory):
    thanksgiving = datetime(2026, 11, 26, 10, 0, tzinfo=NEW_YORK)
    service = _service(
        session_factory,
        FixedCalendar(
            market_now=thanksgiving,
            is_trading_day=False,
            last_completed=date(2026, 11, 25),
        ),
        now=thanksgiving.astimezone(UTC),
    )

    payload = _payload(service, "market_calendar_status", {"market": "US"})

    assert payload["is_trading_day"] is False
    assert payload["expected_trading_date"] == "2026-11-25"
    assert "no session" in payload["summary"]


def test_the_market_date_follows_the_market_timezone_not_utc(session_factory):
    """01:30 UTC is still the previous evening in New York."""
    just_after_midnight_utc = datetime(2026, 10, 15, 1, 30, tzinfo=UTC)
    market_moment = just_after_midnight_utc.astimezone(NEW_YORK)
    service = _service(
        session_factory,
        FixedCalendar(
            market_now=market_moment,
            is_trading_day=True,
            last_completed=date(2026, 10, 13),
        ),
        now=just_after_midnight_utc,
    )

    payload = _payload(service, "market_calendar_status", {"market": "US"})

    assert payload["market_date"] == "2026-10-14"
    assert payload["market_now"].startswith("2026-10-14T21:30:00-04:00")
    assert payload["expected_trading_date"] == "2026-10-14"


def test_an_unsupported_market_is_rejected_as_bad_arguments(session_factory):
    service = _service(session_factory)

    error = _error(service, "market_calendar_status", {"market": "XX"})

    assert error["code"] == "invalid_arguments"
    assert "expected_trading_date" not in error["details"]


def test_an_unavailable_calendar_fails_closed(session_factory):
    raising = FixedCalendar(
        market_now=FIXED_NOW.astimezone(NEW_YORK),
        is_trading_day=True,
        last_completed=date(2026, 10, 14),
    )
    service = _service(session_factory, RaisingCalendar(
        market_now=raising._market_now,
        is_trading_day=True,
        last_completed=date(2026, 10, 14),
    ))

    result = service.call_tool("market_calendar_status", {"market": "US"})

    assert result.get("isError") is True
    payload = result["structuredContent"]
    assert payload["error"]["code"] == "calendar_unavailable"
    # No half-filled success payload: there is no date to read.
    assert "expected_trading_date" not in payload
    assert "calendar coverage" not in payload["error"]["message"]


def test_the_calendar_payload_keeps_the_gateway_keys(session_factory):
    """Consumers of the temporary gateway read these keys by name."""
    gateway_keys = {
        "market",
        "market_timezone",
        "market_now",
        "market_date",
        "is_trading_day",
        "last_completed_trading_day",
        "expected_trading_date",
        "calendar_source",
    }
    service = _service(session_factory)

    payload = _payload(service, "market_calendar_status", {"market": "US"})

    assert gateway_keys <= set(payload)
    fact_keys = {fact["key"] for fact in payload["facts"]}
    assert {
        "market_timezone",
        "market_now",
        "market_date",
        "is_trading_day",
        "last_completed_trading_day",
        "expected_trading_date",
    } <= fact_keys


def test_the_canonical_service_confirms_the_expected_date_semantics(session_factory):
    """The contract holds against the real calendar, not only the double.

    Thursday 2026-10-15 is a session day: mid-session the expected date is that
    day, while ``last_completed_trading_day`` is still Wednesday. Saturday
    2026-10-17 and Thanksgiving 2026-11-26 are not sessions, so the expected
    date is the last completed one.
    """
    cases = (
        (datetime(2026, 10, 15, 15, 0, tzinfo=UTC), "2026-10-15", True, "2026-10-14"),
        (datetime(2026, 10, 17, 12, 0, tzinfo=UTC), "2026-10-17", False, "2026-10-16"),
        (datetime(2026, 11, 26, 15, 0, tzinfo=UTC), "2026-11-26", False, "2026-11-25"),
    )

    for moment, market_date, trading, last_completed in cases:
        service = MarketCopilotService(
            session_factory,
            SimpleNamespace(mcp_watchlist_writes_enabled=False, mcp_server_name="test"),
            now_provider=lambda m=moment: m,
        )

        payload = _payload(service, "market_calendar_status", {"market": "US"})
        expected = market_date if trading else last_completed
        context = f"{market_date} (trading={trading})"
        assert payload["market_date"] == market_date, context
        assert payload["is_trading_day"] is trading, context
        assert payload["last_completed_trading_day"] == last_completed, context
        assert payload["expected_trading_date"] == expected, context


# ---------------------------------------------------------------------------
# earnings_lookup
# ---------------------------------------------------------------------------


def _seed_earnings_rows(session_factory) -> None:
    _seed_fundamentals(
        session_factory,
        [
            # current observation, confirmed-looking date - must stay unknown
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
        ],
    )


def test_a_current_observation_reports_the_date_but_never_a_confirmation(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "aapl"})

    assert payload["symbol"] == "AAPL"
    assert payload["available"] is True
    assert payload["freshness_status"] == "ok"
    assert payload["next_earnings_date"] == "2026-11-05"
    assert payload["event_calendar_as_of_date"] == "2026-10-10"
    assert payload["date_confirmation"] == "unknown"
    assert "date_confirmation_unrecorded" in payload["reasons"]
    assert payload["observation_age_days"] == 5
    assert payload["source"] == "xang_yahoo_calendar"


def test_a_current_observation_without_a_date_is_still_evidence(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "MSFT"})

    assert payload["available"] is True
    assert payload["freshness_status"] == "ok"
    assert payload["next_earnings_date"] is None
    assert "no_upcoming_earnings_date" in payload["reasons"]
    assert payload["date_confirmation"] == "unknown"


def test_an_unknown_symbol_is_an_answer_not_an_error(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "ZZZZ"})

    assert payload["available"] is False
    assert payload["freshness_status"] == "unknown"
    assert payload["reasons"] == ["symbol_not_found"]


def test_an_observation_older_than_the_window_is_stale(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "NVDA"})

    assert payload["freshness_status"] == "stale"
    assert payload["available"] is False
    assert "observation_stale" in payload["reasons"]
    assert payload["observation_age_days"] == 25


def test_an_observation_beyond_the_future_tolerance_is_invalid(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "PANW"})

    assert payload["freshness_status"] == "invalid"
    assert "observation_date_in_future" in payload["reasons"]


def test_a_past_stored_date_on_a_current_observation_is_invalid(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "SNOW"})

    assert payload["freshness_status"] == "invalid"
    assert "earnings_date_in_past" in payload["reasons"]
    assert payload["next_earnings_date"] is None


def test_a_missing_observation_date_stays_unknown(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "AVGO"})

    assert payload["freshness_status"] == "unknown"
    assert payload["available"] is False
    assert "observation_date_missing" in payload["reasons"]


def test_an_invalid_symbol_is_refused_before_the_lookup(session_factory):
    service = _service(session_factory)

    error = _error(service, "earnings_lookup", {"symbol": "AAPL; DROP TABLE"})

    assert error["code"] == "invalid_arguments"


def test_a_missing_symbol_argument_is_refused(session_factory):
    service = _service(session_factory)

    error = _error(service, "earnings_lookup", {})

    assert error["code"] == "invalid_arguments"


def test_a_database_failure_fails_closed(session_factory):
    def broken_factory():
        raise RuntimeError("connection refused for postgresql://user:secret@db/app")

    _seed_earnings_rows(session_factory)
    service = _service(broken_factory)

    result = service.call_tool("earnings_lookup", {"symbol": "AAPL"})

    assert result.get("isError") is True
    payload = result["structuredContent"]
    assert payload["error"]["code"] == "database_unavailable"
    # No connection string, no driver text in the answer.
    assert "postgresql" not in payload["error"]["message"]
    assert "secret" not in result["content"][0]["text"]


def test_the_published_thresholds_come_from_the_canonical_module(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "AAPL"})

    assert payload["event_calendar_max_age_days"] == yahoo_earnings_calendar.EVENT_CALENDAR_MAX_AGE_DAYS
    assert (
        payload["event_calendar_producer_refresh_after_days"]
        == yahoo_earnings_calendar.EVENT_CALENDAR_PRODUCER_REFRESH_AFTER_DAYS
    )
    assert (
        payload["event_calendar_future_tolerance_days"]
        == yahoo_earnings_calendar.EVENT_CALENDAR_FUTURE_TOLERANCE_DAYS
    )


def test_the_earnings_payload_keeps_the_gateway_keys(session_factory):
    _seed_earnings_rows(session_factory)
    service = _service(session_factory)

    payload = _payload(service, "earnings_lookup", {"symbol": "AAPL"})

    gateway_keys = {
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
    assert gateway_keys <= set(payload)
    assert payload["date_confirmation"] == "unknown"


def test_the_lookup_only_emits_select_statements(session_factory):
    """Read-only means read-only: no INSERT, UPDATE or DELETE reaches the driver."""
    factory, engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    _seed_earnings_rows(factory)
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        statements.append(statement)

    service = _service(factory)
    _payload(service, "earnings_lookup", {"symbol": "AAPL"})
    _payload(service, "earnings_lookup", {"symbol": "ZZZZ"})

    assert statements, "the lookup must read the stored row"
    assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
