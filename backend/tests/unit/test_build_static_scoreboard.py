from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.models.stock import StockPrice
from app.schemas.validation import ValidationSourceKind
from app.scripts import build_static_scoreboard as builder
from app.scripts.build_static_scoreboard import (
    StockPriceTable,
    build_scoreboard,
    entry_anchor,
    write_scoreboard,
)
from app.scripts.scoreboard_ledger import (
    LedgerDay,
    Origin,
    PresetMatch,
    Slot,
    asset_name,
    canonical_line,
    parse_timestamp,
    write_ledger,
)
from app.services.market_calendar_service import MarketCalendarService
from app.services.static_advertised_paths import (
    StaticAdvertisedPathError,
    validate_advertised_paths,
)
from app.services.static_artifact_combiner import StaticArtifactCombiner
from app.services.static_data_generations import relocate_into_generation
from app.services.static_market_artifact_contract import STATIC_SITE_SCHEMA_VERSION
from app.services.static_social_isolation import assert_live_only_static_isolation
from app.services.validation_service import PriceOutcomeCalculator, RawValidationEvent

DEF_A = "aaaaaaaaaaaaaaaa"
DEF_B = "bbbbbbbbbbbbbbbb"
NOW = datetime(2026, 11, 5, 3, 0, tzinfo=timezone.utc)
PRESETS = [
    {"id": "minervini", "short_name": "Minervini", "name": "Minervini Trend Template"},
    {"id": "canslim", "name": "CANSLIM"},
]


@pytest.fixture(scope="module")
def calendar() -> MarketCalendarService:
    return MarketCalendarService()


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    StockPrice.__table__.create(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def add_bars(
    db,
    calendar,
    symbol: str,
    first: date,
    last: date,
    *,
    market: str = "US",
    open_: float = 100.0,
    close: float = 100.0,
    opens: dict[date, float] | None = None,
    closes: dict[date, float] | None = None,
) -> None:
    for day in calendar.trading_days(market, first, last):
        day_open = (opens or {}).get(day, open_)
        day_close = (closes or {}).get(day, close)
        db.add(
            StockPrice(
                symbol=symbol,
                date=day,
                open=day_open,
                high=max(day_open, day_close) + 1,
                low=min(day_open, day_close) - 1,
                close=day_close,
                adj_close=day_close,
                volume=1000,
            )
        )
    db.flush()


def ledger_day(
    as_of: date,
    published_at: str,
    presets: dict[str, tuple[str, list[str]]],
    origin: Origin = Origin.LIVE,
) -> LedgerDay:
    return LedgerDay(
        as_of_date=as_of,
        published_at=parse_timestamp(published_at),
        origin=origin,
        presets={preset: PresetMatch(definition, tuple(sorted(symbols))) for preset, (definition, symbols) in presets.items()},
    )


def as_ledger(*days: LedgerDay) -> dict[date, LedgerDay]:
    return {day.as_of_date: day for day in days}


def after_baseline(*days: LedgerDay) -> dict[date, LedgerDay]:
    """``days`` after an empty first day, so each of their matches is a new entrant."""
    presets = {preset: (match.definition, []) for day in days for preset, match in day.presets.items()}
    return as_ledger(ledger_day(date(2026, 10, 2), "2026-10-03T02:00:00Z", presets, days[0].origin), *days)


def score(db, calendar, ledger, as_of: date, *, market: str = "US", presets=PRESETS) -> dict:
    return build_scoreboard(
        ledger,
        market=market,
        as_of_date=as_of,
        preset_screens=presets,
        prices=StockPriceTable(db, start=min(ledger), through=as_of),
        calendar=calendar,
        generated_at=NOW,
    )


def horizon(cohort: dict, sessions: int) -> dict:
    return next(block for block in cohort["horizons"] if block["horizon_sessions"] == sessions)


def only_cohort(payload: dict, preset: str = "minervini") -> dict:
    matches = [cohort for cohort in payload["cohorts"] if cohort["preset"] == preset]
    assert len(matches) == 1
    return matches[0]


EMPTY_STATS = {
    "positive_rate": None,
    "avg_return_pct": None,
    "median_return_pct": None,
    "avg_excess_return_pct": None,
    "median_excess_return_pct": None,
    "excess_sample_size": 0,
}


@pytest.mark.parametrize(
    ("market", "published_at", "as_of", "expected_entry"),
    [
        pytest.param("US", "2026-11-04T23:00:00Z", date(2026, 11, 4), date(2026, 11, 5), id="us-after-close"),
        pytest.param("US", "2026-11-05T12:00:00Z", date(2026, 11, 4), date(2026, 11, 5), id="us-pre-open"),
        pytest.param("US", "2026-11-05T14:30:00Z", date(2026, 11, 4), date(2026, 11, 6), id="us-exactly-at-open"),
        pytest.param("US", "2026-11-05T14:29:59Z", date(2026, 11, 4), date(2026, 11, 5), id="us-second-before-open"),
        pytest.param("US", "2026-11-05T15:00:00Z", date(2026, 11, 4), date(2026, 11, 6), id="us-mid-session"),
        pytest.param("AU", "2026-11-05T03:00:00Z", date(2026, 11, 4), date(2026, 11, 6), id="au-late-publish"),
        pytest.param("AU", "2026-11-04T20:00:00Z", date(2026, 11, 4), date(2026, 11, 5), id="au-before-local-open"),
        pytest.param("US", "2026-11-26T18:00:00Z", date(2026, 11, 25), date(2026, 11, 27), id="us-holiday"),
        pytest.param("US", "2026-11-07T15:00:00Z", date(2026, 11, 6), date(2026, 11, 9), id="us-saturday"),
        pytest.param("US", "2026-11-09T12:00:00Z", date(2026, 11, 6), date(2026, 11, 9), id="us-monday-pre-open"),
        pytest.param("US", "2026-11-05T12:00:00Z", date(2026, 11, 5), date(2026, 11, 6), id="never-before-the-as-of-close"),
    ],
)
def test_the_entry_is_the_first_open_after_publication(db, calendar, market, published_at, as_of, expected_entry) -> None:
    anchor = entry_anchor(parse_timestamp(published_at), as_of, market, calendar)
    add_bars(db, calendar, "X", date(2026, 11, 2), date(2026, 11, 30), market=market)
    sessions = calendar.trading_days(market, date(2026, 11, 2), date(2026, 11, 30))

    evaluations, _ = PriceOutcomeCalculator(
        StockPriceTable(db, start=date(2026, 11, 2), through=date(2026, 11, 30))
    ).evaluate_many(
        [RawValidationEvent("X", ValidationSourceKind.SCAN_PICK, "", anchor, {})],
        sessions=sessions,
    )

    assert evaluations[0].entry_at == expected_entry


def test_the_payload_is_exactly_the_published_schema(db, calendar) -> None:
    add_bars(
        db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 12),
        open_=100.0, close=105.0, closes={date(2026, 10, 12): 110.0},
    )
    add_bars(
        db, calendar, "SPY", date(2026, 10, 5), date(2026, 10, 12),
        open_=400.0, close=401.0, closes={date(2026, 10, 12): 404.0},
    )
    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}))

    pending = {
        "sample_size": 0,
        "signal_dates": 0,
        "pending_windows": 1,
        "oldest_open_sessions": 5,
        "thin": True,
        **EMPTY_STATS,
        "skipped_missing_history": 0,
        "suppressed_overlapping": 0,
    }
    assert score(db, calendar, ledger, date(2026, 10, 12)) == {
        "schema_version": "static-scoreboard-v1",
        "market": "US",
        "as_of_date": "2026-10-12",
        "generated_at": "2026-11-05T03:00:00Z",
        "benchmark": "SPY",
        "recording_since": "2026-10-02",
        "ledger_through": "2026-10-05",
        "missing_sessions": 0,
        "cohorts": [
            {
                "preset": "minervini",
                "name": "Minervini",
                "retired": False,
                "definition": DEF_A,
                "origin": "live",
                "first_date": "2026-10-02",
                "last_date": "2026-10-05",
                "events": 1,
                "signal_dates": 1,
                "horizons": [
                    {
                        "horizon_sessions": 5,
                        "sample_size": 1,
                        "signal_dates": 1,
                        "pending_windows": 0,
                        "oldest_open_sessions": None,
                        "thin": True,
                        "positive_rate": 1.0,
                        "avg_return_pct": 10.0,
                        "median_return_pct": 10.0,
                        "avg_excess_return_pct": 9.0,
                        "median_excess_return_pct": 9.0,
                        "excess_sample_size": 1,
                        "skipped_missing_history": 0,
                        "suppressed_overlapping": 0,
                    },
                    {"horizon_sessions": 20, **pending},
                    {"horizon_sessions": 60, **pending},
                ],
            }
        ],
    }


def test_a_symbol_in_two_presets_counts_once_in_each(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 12), closes={date(2026, 10, 12): 110.0})
    ledger = after_baseline(
        ledger_day(
            date(2026, 10, 5),
            "2026-10-06T02:00:00Z",
            {"minervini": (DEF_A, ["AAPL"]), "canslim": (DEF_B, ["AAPL"])},
        )
    )

    payload = score(db, calendar, ledger, date(2026, 10, 12))

    for preset in ("minervini", "canslim"):
        block = horizon(only_cohort(payload, preset), 5)
        assert (block["sample_size"], block["median_return_pct"], block["suppressed_overlapping"]) == (1, 10.0, 0)


def test_a_first_ledger_day_is_a_baseline_with_no_events(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 23))
    ledger = as_ledger(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}))

    cohort = only_cohort(score(db, calendar, ledger, date(2026, 10, 23)))

    assert (cohort["events"], cohort["signal_dates"], cohort["first_date"]) == (0, 0, "2026-10-05")
    assert horizon(cohort, 5)["pending_windows"] == 0
    assert horizon(cohort, 5)["sample_size"] == 0


def test_only_a_symbol_new_to_the_preset_is_a_pick(db, calendar) -> None:
    add_bars(db, calendar, "OLD", date(2026, 10, 5), date(2026, 10, 23), closes={date(2026, 10, 13): 150.0})
    add_bars(db, calendar, "NEW", date(2026, 10, 5), date(2026, 10, 23), closes={date(2026, 10, 13): 120.0})
    ledger = as_ledger(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["OLD"])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["NEW", "OLD"])}),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["NEW", "OLD"])}),
    )

    cohort = only_cohort(score(db, calendar, ledger, date(2026, 10, 23)))

    assert (cohort["events"], cohort["signal_dates"]) == (1, 1)
    block = horizon(cohort, 5)
    assert (block["sample_size"], block["median_return_pct"], block["suppressed_overlapping"]) == (1, 20.0, 0)


def test_a_preset_added_later_starts_with_its_own_baseline(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 23))
    ledger = as_ledger(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, [])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["AAPL"]), "canslim": (DEF_A, ["AAPL"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 10, 23))

    assert only_cohort(payload, "minervini")["events"] == 1
    assert only_cohort(payload, "canslim")["events"] == 0
    assert only_cohort(payload, "canslim")["first_date"] == "2026-10-06"


def test_a_preset_missing_from_a_day_compares_with_its_last_appearance(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 23))
    add_bars(db, calendar, "NEW", date(2026, 10, 5), date(2026, 10, 23))
    ledger = as_ledger(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"]), "canslim": (DEF_A, [])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"canslim": (DEF_A, [])}),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["AAPL", "NEW"]), "canslim": (DEF_A, [])}),
    )

    cohort = only_cohort(score(db, calendar, ledger, date(2026, 10, 23)))

    assert (cohort["events"], cohort["first_date"], cohort["last_date"]) == (1, "2026-10-05", "2026-10-07")


def test_a_symbol_flipping_in_and_out_cannot_stack_windows(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, [])}),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}),
        ledger_day(date(2026, 10, 8), "2026-10-09T02:00:00Z", {"minervini": (DEF_A, [])}),
        ledger_day(date(2026, 10, 12), "2026-10-13T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}),
    )

    cohort = only_cohort(score(db, calendar, ledger, date(2026, 10, 23)))

    block = horizon(cohort, 5)
    assert (cohort["events"], cohort["signal_dates"]) == (3, 3)
    assert (block["sample_size"], block["suppressed_overlapping"], block["signal_dates"]) == (2, 1, 2)


def test_a_definition_change_is_no_new_baseline_and_does_not_reset_overlap(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    add_bars(db, calendar, "STAYS", date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL", "STAYS"])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_B, ["STAYS"])}),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_B, ["AAPL", "STAYS"])}),
    )

    cohorts = {cohort["definition"]: cohort for cohort in score(db, calendar, ledger, date(2026, 10, 23))["cohorts"]}

    assert cohorts[DEF_A]["events"] == 2
    assert horizon(cohorts[DEF_A], 5)["sample_size"] == 2
    assert cohorts[DEF_B]["events"] == 1
    assert horizon(cohorts[DEF_B], 5)["sample_size"] == 0
    assert horizon(cohorts[DEF_B], 5)["suppressed_overlapping"] == 1
    assert horizon(cohorts[DEF_B], 5)["skipped_missing_history"] == 0


def test_an_era_change_is_no_new_baseline_and_does_not_reset_overlap(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}, Origin.ARTIFACT),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, [])}, Origin.ARTIFACT),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}, Origin.LIVE),
    )

    cohorts = {cohort["origin"]: cohort for cohort in score(db, calendar, ledger, date(2026, 10, 23))["cohorts"]}

    assert horizon(cohorts["artifact"], 5)["sample_size"] == 1
    assert (cohorts["live"]["events"], horizon(cohorts["live"], 5)["suppressed_overlapping"]) == (1, 1)


def test_a_later_publication_enters_a_session_later(db, calendar) -> None:
    opens = {date(2026, 10, 6): 100.0, date(2026, 10, 7): 200.0}
    closes = {date(2026, 10, 13): 220.0}
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 13), open_=200.0, opens=opens, closes=closes)
    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-06T15:00:00Z", {"minervini": (DEF_A, ["AAPL"])}))

    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 13))), 5)

    assert (block["sample_size"], block["median_return_pct"]) == (1, 10.0)


def test_an_au_late_publication_enters_the_next_sydney_session(db, calendar) -> None:
    opens = {date(2026, 10, 6): 50.0, date(2026, 10, 7): 100.0}
    closes = {date(2026, 10, 13): 105.0}
    add_bars(
        db, calendar, "BHP.AX", date(2026, 10, 5), date(2026, 10, 13),
        market="AU", open_=100.0, opens=opens, closes=closes,
    )
    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-06T03:00:00Z", {"minervini": (DEF_A, ["BHP.AX"])}))

    payload = score(db, calendar, ledger, date(2026, 10, 13), market="AU")

    assert payload["benchmark"] is None
    assert horizon(only_cohort(payload), 5)["median_return_pct"] == 5.0


def test_bars_after_the_as_of_date_are_never_read(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 16), closes={date(2026, 10, 12): 999.0})
    table = StockPriceTable(db, start=date(2026, 10, 5), through=date(2026, 10, 9))

    frame = table.get_many_cached_only(["AAPL"])["AAPL"]

    assert frame.index[-1].date() == date(2026, 10, 9)
    assert len(frame) == 5

    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}))
    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 9))), 5)
    assert block == {
        "horizon_sessions": 5,
        "sample_size": 0,
        "signal_dates": 0,
        "pending_windows": 1,
        "oldest_open_sessions": 4,
        "thin": True,
        **EMPTY_STATS,
        "skipped_missing_history": 0,
        "suppressed_overlapping": 0,
    }


def test_a_rewound_export_keeps_later_ledger_days_pending(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 16), close=500.0)
    ledger = after_baseline(ledger_day(date(2026, 10, 12), "2026-10-13T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}))

    payload = score(db, calendar, ledger, date(2026, 10, 9))

    block = horizon(only_cohort(payload), 5)
    assert payload["ledger_through"] == "2026-10-12"
    assert (block["sample_size"], block["pending_windows"], block["oldest_open_sessions"]) == (0, 1, 0)


def test_a_recent_ipo_with_few_bars_still_scores(db, calendar) -> None:
    add_bars(db, calendar, "IPO", date(2026, 10, 6), date(2026, 10, 12), closes={date(2026, 10, 12): 120.0})
    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["IPO"])}))

    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 12))), 5)

    assert (block["sample_size"], block["median_return_pct"], block["skipped_missing_history"]) == (1, 20.0, 0)


def test_a_symbol_with_no_price_counts_as_skipped_once_its_window_closes(db, calendar) -> None:
    add_bars(db, calendar, "SPY", date(2026, 10, 5), date(2026, 10, 12))
    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["GONE"])}))

    cohort = only_cohort(score(db, calendar, ledger, date(2026, 10, 12)))

    assert horizon(cohort, 5)["sample_size"] == 0
    assert horizon(cohort, 5)["skipped_missing_history"] == 1
    assert horizon(cohort, 5)["pending_windows"] == 0
    assert horizon(cohort, 20)["skipped_missing_history"] == 0
    assert horizon(cohort, 20)["pending_windows"] == 1


def test_a_symbol_one_session_behind_stays_pending_short_of_its_last_bar(db, calendar) -> None:
    add_bars(db, calendar, "LAGS", date(2026, 10, 5), date(2026, 10, 9))
    add_bars(db, calendar, "SPY", date(2026, 10, 5), date(2026, 10, 12))
    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["LAGS"])}))

    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 12))), 5)

    assert (block["sample_size"], block["pending_windows"], block["oldest_open_sessions"]) == (0, 1, 4)
    assert block["skipped_missing_history"] == 0


def test_cohorts_segment_by_definition_and_origin_live_first_in_manifest_order(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 9))
    ledger = as_ledger(
        ledger_day(
            date(2026, 10, 5),
            "2026-10-06T02:00:00Z",
            {"canslim": (DEF_A, ["AAPL"]), "minervini": (DEF_A, []), "old_screen": (DEF_A, [])},
            Origin.ARTIFACT_PRE_STAGE_FIX,
        ),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, [])}, Origin.ARTIFACT),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}),
        ledger_day(date(2026, 10, 8), "2026-10-09T02:00:00Z", {"minervini": (DEF_B, ["AAPL"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 10, 9))

    assert [
        (c["preset"], c["name"], c["retired"], c["definition"], c["origin"], c["first_date"], c["last_date"], c["events"])
        for c in payload["cohorts"]
    ] == [
        ("minervini", "Minervini", False, DEF_A, "live", "2026-10-07", "2026-10-07", 1),
        ("minervini", "Minervini", False, DEF_B, "live", "2026-10-08", "2026-10-08", 0),
        ("minervini", "Minervini", False, DEF_A, "artifact", "2026-10-06", "2026-10-06", 0),
        ("minervini", "Minervini", False, DEF_A, "artifact_pre_stage_fix", "2026-10-05", "2026-10-05", 0),
        ("canslim", "CANSLIM", False, DEF_A, "artifact_pre_stage_fix", "2026-10-05", "2026-10-05", 0),
        ("old_screen", "old_screen", True, DEF_A, "artifact_pre_stage_fix", "2026-10-05", "2026-10-05", 0),
    ]


def test_a_zero_match_preset_is_listed_with_null_stats(db, calendar) -> None:
    add_bars(db, calendar, "SPY", date(2026, 10, 5), date(2026, 10, 12))
    ledger = as_ledger(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, [])}))

    cohort = only_cohort(score(db, calendar, ledger, date(2026, 10, 12)))

    assert (cohort["events"], cohort["signal_dates"]) == (0, 0)
    assert horizon(cohort, 5) == {
        "horizon_sessions": 5,
        "sample_size": 0,
        "signal_dates": 0,
        "pending_windows": 0,
        "oldest_open_sessions": None,
        "thin": True,
        **EMPTY_STATS,
        "skipped_missing_history": 0,
        "suppressed_overlapping": 0,
    }


def test_missing_sessions_counts_recorded_range_gaps_and_ignores_weekends(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 12))
    presets = {"minervini": (DEF_A, ["AAPL"])}
    ledger = as_ledger(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", presets),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", presets),
        ledger_day(date(2026, 10, 9), "2026-10-10T02:00:00Z", presets),
        ledger_day(date(2026, 10, 12), "2026-10-13T02:00:00Z", presets),
    )

    payload = score(db, calendar, ledger, date(2026, 10, 12))

    assert payload["missing_sessions"] == 2
    assert (payload["recording_since"], payload["ledger_through"]) == ("2026-10-05", "2026-10-12")


def test_thin_needs_both_twenty_closed_events_and_five_dates(db, calendar) -> None:
    dates = [date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7), date(2026, 10, 8), date(2026, 10, 9)]
    symbols = [f"S{index:02d}" for index in range(25)]
    for symbol in symbols:
        add_bars(db, calendar, symbol, date(2026, 10, 5), date(2026, 10, 16), close=110.0)
    published = ["2026-10-06T02:00:00Z", "2026-10-07T02:00:00Z", "2026-10-08T02:00:00Z", "2026-10-09T02:00:00Z", "2026-10-10T02:00:00Z"]

    spread = after_baseline(
        *(
            ledger_day(day, stamp, {"minervini": (DEF_A, symbols[index * 5 : index * 5 + 5])})
            for index, (day, stamp) in enumerate(zip(dates, published))
        )
    )
    block = horizon(only_cohort(score(db, calendar, spread, date(2026, 10, 16))), 5)
    assert (block["sample_size"], block["signal_dates"], block["thin"]) == (25, 5, False)

    one_date = after_baseline(ledger_day(dates[0], published[0], {"minervini": (DEF_A, symbols)}))
    block = horizon(only_cohort(score(db, calendar, one_date, date(2026, 10, 16))), 5)
    assert (block["sample_size"], block["signal_dates"], block["thin"]) == (25, 1, True)

    too_few = after_baseline(
        *(
            ledger_day(day, stamp, {"minervini": (DEF_A, symbols[index * 3 : index * 3 + 3])})
            for index, (day, stamp) in enumerate(zip(dates, published))
        )
    )
    block = horizon(only_cohort(score(db, calendar, too_few, date(2026, 10, 16))), 5)
    assert (block["sample_size"], block["signal_dates"], block["thin"]) == (15, 5, True)


def write_artifact(root: Path, as_of: str = "2026-10-12") -> Path:
    market_dir = root / "static-market-US"
    (market_dir / "scan" / "chunks").mkdir(parents=True)
    (market_dir / "scan" / "chunks" / "chunk-0001.json").write_text(
        json.dumps({"rs_formula_version": BALANCED_RS_FORMULA_VERSION, "rows": []})
    )
    (market_dir / "scan" / "manifest.json").write_text(
        json.dumps(
            {
                "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
                "as_of_date": as_of,
                "preset_screens": PRESETS,
                "chunks": [{"path": "markets/us/scan/chunks/chunk-0001.json"}],
            }
        )
    )
    entry = {
        "market": "US",
        "display_name": "United States",
        "as_of_date": as_of,
        "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
        "features": {"scan": True, "breadth": False, "groups": False, "charts": False},
        "pages": {"scan": {"path": "markets/us/scan/manifest.json"}},
        "assets": {},
    }
    (market_dir / "manifest.market.json").write_text(
        json.dumps(
            {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": "2026-10-13T02:00:00Z",
                "market": "US",
                "entry": entry,
                "warnings": [],
            }
        )
    )
    return market_dir


def write_ledger_assets(directory: Path, *, canonical: list[LedgerDay] | None = None, previous: list[LedgerDay] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for slot, days in ((Slot.CANONICAL, canonical), (Slot.PREVIOUS, previous)):
        if days is not None:
            write_ledger(directory / asset_name("US", slot), [canonical_line(day) for day in days])
    return directory


def run_write(market_dir: Path, ledger_dir: Path, db, calendar) -> bool:
    return write_scoreboard(market="US", market_dir=market_dir, ledger_dir=ledger_dir, db=db, calendar=calendar, now=NOW)


DAY_1002 = ledger_day(date(2026, 10, 2), "2026-10-03T02:00:00Z", {"minervini": (DEF_A, [])})
DAY_1005 = ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])})
DAY_1006 = ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])})


def test_success_writes_the_asset_and_advertises_it(tmp_path, db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 12), closes={date(2026, 10, 12): 110.0})
    market_dir = write_artifact(tmp_path / "artifacts")
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[DAY_1005])

    assert run_write(market_dir, ledger_dir, db, calendar) is True

    manifest = json.loads((market_dir / "manifest.market.json").read_text())
    assert manifest["entry"]["assets"]["scoreboard"] == {"path": "markets/us/scoreboard.json"}
    assert json.loads((market_dir / "scoreboard.json").read_text())["as_of_date"] == "2026-10-12"
    assert sorted(path.name for path in market_dir.iterdir()) == ["manifest.market.json", "scan", "scoreboard.json"]


def test_a_corrupt_canonical_falls_back_to_previous(tmp_path, db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 12))
    market_dir = write_artifact(tmp_path / "artifacts")
    ledger_dir = write_ledger_assets(tmp_path / "ledger", previous=[DAY_1005])
    (ledger_dir / asset_name("US")).write_bytes(b"truncated")

    assert run_write(market_dir, ledger_dir, db, calendar) is True

    assert json.loads((market_dir / "scoreboard.json").read_text())["ledger_through"] == "2026-10-05"


@pytest.mark.parametrize(
    "assets",
    [
        pytest.param({}, id="both-absent"),
        pytest.param({"canonical_bytes": b"truncated"}, id="canonical-corrupt-no-previous"),
        pytest.param({"canonical": []}, id="empty-ledger"),
    ],
)
def test_no_usable_ledger_leaves_the_manifest_unchanged(tmp_path, db, calendar, assets, capsys) -> None:
    market_dir = write_artifact(tmp_path / "artifacts")
    before = (market_dir / "manifest.market.json").read_bytes()
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=assets.get("canonical"))
    if "canonical_bytes" in assets:
        (ledger_dir / asset_name("US")).write_bytes(assets["canonical_bytes"])

    assert run_write(market_dir, ledger_dir, db, calendar) is False

    assert (market_dir / "manifest.market.json").read_bytes() == before
    assert not (market_dir / "scoreboard.json").exists()
    assert "::warning::No usable US ledger" in capsys.readouterr().out


def test_a_scoring_failure_leaves_the_manifest_unchanged(tmp_path, calendar) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    no_price_table = sessionmaker(bind=engine)()
    market_dir = write_artifact(tmp_path / "artifacts")
    before = (market_dir / "manifest.market.json").read_bytes()
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[DAY_1005])

    with pytest.raises(OperationalError):
        run_write(market_dir, ledger_dir, no_price_table, calendar)

    assert (market_dir / "manifest.market.json").read_bytes() == before
    assert not (market_dir / "scoreboard.json").exists()


def test_the_asset_survives_validation_combine_and_generation_relocation(tmp_path, db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 12))
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[DAY_1005, DAY_1006])
    assert run_write(market_dir, ledger_dir, db, calendar) is True
    written = (market_dir / "scoreboard.json").read_text()

    entry = json.loads((market_dir / "manifest.market.json").read_text())["entry"]
    validate_advertised_paths(market="US", entry=entry, market_dir=market_dir)

    combined = tmp_path / "combined"
    result = StaticArtifactCombiner(
        schema_version=STATIC_SITE_SCHEMA_VERSION,
        supported_markets=("US",),
        default_market="US",
    ).combine(artifacts_dir=artifacts, fallback_artifacts_dir=None, output_dir=combined, clean=True)

    assert result.warnings == ()
    assert result.manifest["markets"]["US"]["assets"]["scoreboard"] == {"path": "markets/us/scoreboard.json"}
    assert (combined / "markets" / "us" / "scoreboard.json").read_text() == written

    generation = relocate_into_generation(combined)

    root_manifest = json.loads((combined / "manifest.json").read_text())
    advertised = root_manifest["markets"]["US"]["assets"]["scoreboard"]["path"]
    assert (combined / root_manifest["data_root"] / advertised).read_text() == written
    assert root_manifest["data_root"] == f"g/{generation}/"
    assert_live_only_static_isolation(combined)


GOOD_SHAPE = {"schema_version": "static-scoreboard-v1", "market": "US", "cohorts": [{"horizons": []}]}


def combine_one_market(artifacts: Path, output: Path, *, as_fallback: bool = False):
    empty = output.parent / "no-current-artifacts"
    return StaticArtifactCombiner(
        schema_version=STATIC_SITE_SCHEMA_VERSION,
        supported_markets=("US",),
        default_market="US",
    ).combine(
        artifacts_dir=empty if as_fallback else artifacts,
        fallback_artifacts_dir=artifacts if as_fallback else None,
        output_dir=output,
        clean=True,
    )


def advertise_scoreboard(market_dir: Path, descriptor: object) -> None:
    manifest_path = market_dir / "manifest.market.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entry"]["assets"]["scoreboard"] = descriptor
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        pytest.param("{truncated", "does not parse", id="truncated"),
        pytest.param(None, "advertised file is absent", id="absent"),
        pytest.param("[]", "is not a static-scoreboard-v1 file for US", id="not-an-object"),
        pytest.param("{}", "is not a static-scoreboard-v1 file for US", id="empty-object"),
        pytest.param(json.dumps({**GOOD_SHAPE, "schema_version": "static-scoreboard-v0"}), "is not a static-scoreboard-v1 file for US", id="wrong-schema"),
        pytest.param(json.dumps({**GOOD_SHAPE, "market": "HK"}), "is not a static-scoreboard-v1 file for US", id="another-market"),
        pytest.param(json.dumps({**GOOD_SHAPE, "cohorts": {}}), "is not a static-scoreboard-v1 file for US", id="cohorts-not-a-list"),
        pytest.param(json.dumps({**GOOD_SHAPE, "cohorts": [1]}), "is not a static-scoreboard-v1 file for US", id="cohort-not-an-object"),
        pytest.param(json.dumps({**GOOD_SHAPE, "cohorts": [{}]}), "is not a static-scoreboard-v1 file for US", id="cohort-without-horizons"),
    ],
)
def test_a_broken_scoreboard_loses_its_entry_and_the_market_stays(tmp_path, content, reason) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    if content is not None:
        (market_dir / "scoreboard.json").write_text(content)

    result = combine_one_market(artifacts, tmp_path / "combined")

    assert result.manifest["markets"]["US"]["assets"] == {}
    assert result.manifest["markets"]["US"]["pages"] == {"scan": {"path": "markets/us/scan/manifest.json"}}
    assert result.manifest.get("unavailable_markets", []) == []
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("US scoreboard asset ignored: ")
    assert reason in result.warnings[0]


@pytest.mark.parametrize(
    "descriptor",
    [
        pytest.param(None, id="null"),
        pytest.param("markets/us/scoreboard.json", id="string"),
        pytest.param({}, id="no-path"),
        pytest.param({"path": "../../outside.json"}, id="escapes-the-artifact"),
        pytest.param({"path": "markets/hk/scoreboard.json"}, id="another-markets-path"),
    ],
)
def test_a_broken_scoreboard_descriptor_loses_its_entry_and_the_market_stays(tmp_path, descriptor) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, descriptor)

    result = combine_one_market(artifacts, tmp_path / "combined")

    assert result.manifest["markets"]["US"]["assets"] == {}
    assert [warning.split(":")[0] for warning in result.warnings] == ["US scoreboard asset ignored"]


def test_a_fallback_artifact_with_a_broken_scoreboard_still_serves_the_market(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").write_text("{truncated")
    metadata = json.loads((market_dir / "manifest.market.json").read_text())

    assert StaticArtifactCombiner.rejection_reason(
        market="US", source_label="fallback", metadata=metadata, market_dir=market_dir
    ) is None

    result = combine_one_market(artifacts, tmp_path / "combined", as_fallback=True)

    assert result.manifest["markets"]["US"]["assets"] == {}
    assert result.manifest["markets"]["US"]["as_of_date"] == "2026-10-12"


def test_a_good_scoreboard_keeps_its_entry_and_every_other_broken_asset_still_rejects(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").write_text(json.dumps(GOOD_SHAPE))

    result = combine_one_market(artifacts, tmp_path / "combined")

    assert result.manifest["markets"]["US"]["assets"] == {"scoreboard": {"path": "markets/us/scoreboard.json"}}
    assert result.warnings == ()

    entry = json.loads((market_dir / "manifest.market.json").read_text())["entry"]
    (market_dir / "breadth.json").write_text("{truncated")
    entry["assets"]["breadth"] = {"path": "markets/us/breadth.json"}
    with pytest.raises(StaticAdvertisedPathError, match="does not parse"):
        validate_advertised_paths(market="US", entry=entry, market_dir=market_dir)


def test_command_line_scores_from_the_configured_database(tmp_path, db, calendar, monkeypatch) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 12), closes={date(2026, 10, 12): 110.0})
    db.commit()
    monkeypatch.setattr(builder, "SessionLocal", sessionmaker(bind=db.get_bind()))
    market_dir = write_artifact(tmp_path / "artifacts")
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[DAY_1002, DAY_1005])

    assert builder.main(["--market", "US", "--market-dir", str(market_dir), "--ledger-dir", str(ledger_dir)]) == 0

    payload = json.loads((market_dir / "scoreboard.json").read_text())
    assert horizon(only_cohort(payload), 5)["median_return_pct"] == 10.0
