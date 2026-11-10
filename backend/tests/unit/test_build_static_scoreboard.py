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
    presets = {preset: (match.definition, []) for day in reversed(days) for preset, match in day.presets.items()}
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
    "beat_benchmark_rate": None,
    "median_return_pct": None,
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
        pytest.param("AU", "2026-11-02T23:30:00Z", date(2026, 11, 2), date(2026, 11, 4), id="au-after-the-open-while-utc-is-a-day-behind"),
        pytest.param("AU", "2026-11-02T22:59:59Z", date(2026, 11, 2), date(2026, 11, 3), id="au-second-before-the-open"),
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
        "effective_dates": 0.0,
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
        "scored_since": "2026-10-02",
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
                        "effective_dates": 1.0,
                        "pending_windows": 0,
                        "oldest_open_sessions": None,
                        "thin": True,
                        "positive_rate": 1.0,
                        "beat_benchmark_rate": 1.0,
                        "median_return_pct": 10.0,
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


def test_days_before_the_window_are_not_scored_and_the_last_of_them_is_the_baseline(db, calendar) -> None:
    add_bars(db, calendar, "NEW", date(2025, 10, 30), date(2025, 11, 17), closes={date(2025, 11, 10): 130.0})
    add_bars(db, calendar, "STAYS", date(2025, 10, 30), date(2025, 11, 17), closes={date(2025, 11, 10): 500.0})
    add_bars(db, calendar, "OLDEST", date(2025, 10, 30), date(2025, 11, 17), closes={date(2025, 11, 10): 500.0})
    ledger = as_ledger(
        ledger_day(date(2025, 10, 30), "2025-10-31T02:00:00Z", {"minervini": (DEF_B, ["OLDEST"]), "old_screen": (DEF_B, ["OLDEST"])}),
        ledger_day(date(2025, 10, 31), "2025-11-01T02:00:00Z", {"minervini": (DEF_A, ["STAYS"])}),
        ledger_day(date(2025, 11, 3), "2025-11-04T02:00:00Z", {"minervini": (DEF_A, ["NEW", "STAYS"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 11, 3))

    assert payload["recording_since"] == "2025-10-30"
    assert payload["scored_since"] == "2025-11-03"
    assert payload["ledger_through"] == "2025-11-03"
    assert [(c["preset"], c["definition"], c["first_date"], c["last_date"], c["events"]) for c in payload["cohorts"]] == [
        ("minervini", DEF_A, "2025-11-03", "2025-11-03", 1)
    ]
    block = horizon(only_cohort(payload), 5)
    assert (block["sample_size"], block["median_return_pct"]) == (1, 30.0)


def test_the_day_one_past_the_window_is_the_baseline_not_a_scored_day(db, calendar) -> None:
    add_bars(db, calendar, "NEW", date(2025, 11, 3), date(2025, 11, 17))
    ledger = as_ledger(
        ledger_day(date(2025, 11, 3), "2025-11-04T02:00:00Z", {"minervini": (DEF_A, ["NEW"])}),
        ledger_day(date(2025, 11, 4), "2025-11-05T02:00:00Z", {"minervini": (DEF_A, ["NEW"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 11, 4))

    assert payload["scored_since"] == "2025-11-04"
    assert only_cohort(payload)["events"] == 0
    assert only_cohort(payload)["first_date"] == "2025-11-04"


def test_a_preset_missing_from_the_day_before_the_window_compares_with_its_latest_earlier_day(db, calendar) -> None:
    add_bars(db, calendar, "NEW", date(2025, 11, 3), date(2025, 11, 14), closes={date(2025, 11, 10): 130.0})
    add_bars(db, calendar, "OLD", date(2025, 11, 3), date(2025, 11, 14))
    ledger = as_ledger(
        ledger_day(date(2025, 10, 29), "2025-10-30T02:00:00Z", {"minervini": (DEF_A, ["STALE"])}),
        ledger_day(date(2025, 10, 30), "2025-10-31T02:00:00Z", {"minervini": (DEF_A, ["OLD"])}),
        ledger_day(date(2025, 10, 31), "2025-11-01T02:00:00Z", {"canslim": (DEF_A, [])}),
        ledger_day(date(2025, 11, 3), "2025-11-04T02:00:00Z", {"minervini": (DEF_A, ["NEW", "OLD"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 11, 3))

    assert payload["scored_since"] == "2025-11-03"
    cohort = only_cohort(payload)
    assert (cohort["preset"], cohort["events"], cohort["signal_dates"]) == ("minervini", 1, 1)
    block = horizon(cohort, 5)
    assert (block["sample_size"], block["median_return_pct"]) == (1, 30.0)


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


def test_a_definition_change_is_a_baseline_and_does_not_reset_overlap(db, calendar) -> None:
    for symbol in ("AAPL", "STAYS", "FLOOD", "NEW", "OTHER"):
        add_bars(db, calendar, symbol, date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL", "STAYS"]), "canslim": (DEF_A, [])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_B, ["FLOOD", "STAYS"]), "canslim": (DEF_A, ["OTHER"])}),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_B, ["AAPL", "FLOOD", "NEW", "STAYS"]), "canslim": (DEF_A, ["OTHER"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 10, 23))
    cohorts = {cohort["definition"]: cohort for cohort in payload["cohorts"] if cohort["preset"] == "minervini"}

    assert cohorts[DEF_A]["events"] == 2
    assert horizon(cohorts[DEF_A], 5)["sample_size"] == 2
    assert cohorts[DEF_B]["events"] == 2
    assert cohorts[DEF_B]["signal_dates"] == 1
    assert (cohorts[DEF_B]["first_date"], cohorts[DEF_B]["last_date"]) == ("2026-10-06", "2026-10-07")
    assert horizon(cohorts[DEF_B], 5)["sample_size"] == 1
    assert horizon(cohorts[DEF_B], 5)["suppressed_overlapping"] == 1
    assert horizon(cohorts[DEF_B], 5)["skipped_missing_history"] == 0
    assert only_cohort(payload, "canslim")["events"] == 1


@pytest.mark.parametrize("after_fix", [Origin.ARTIFACT, Origin.LIVE])
def test_crossing_the_stage_fix_is_a_baseline_and_does_not_reset_overlap(db, calendar, after_fix) -> None:
    for symbol in ("AAPL", "FLOOD", "NEW"):
        add_bars(db, calendar, symbol, date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}, Origin.ARTIFACT_PRE_STAGE_FIX),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["FLOOD"])}, after_fix),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["AAPL", "FLOOD", "NEW"])}, after_fix),
    )

    cohorts = {cohort["origin"]: cohort for cohort in score(db, calendar, ledger, date(2026, 10, 23))["cohorts"]}

    assert cohorts["artifact_pre_stage_fix"]["events"] == 1
    assert cohorts[after_fix.value]["events"] == 2
    assert cohorts[after_fix.value]["first_date"] == "2026-10-06"
    assert horizon(cohorts[after_fix.value], 5)["sample_size"] == 1
    assert horizon(cohorts[after_fix.value], 5)["suppressed_overlapping"] == 1


def test_crossing_the_stage_fix_is_judged_by_the_presets_own_last_day(db, calendar) -> None:
    add_bars(db, calendar, "FLOOD", date(2026, 10, 5), date(2026, 10, 23))
    ledger = as_ledger(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, []), "canslim": (DEF_A, [])}, Origin.ARTIFACT_PRE_STAGE_FIX),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"canslim": (DEF_A, [])}, Origin.ARTIFACT),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["FLOOD"]), "canslim": (DEF_A, ["FLOOD"])}, Origin.ARTIFACT),
    )

    payload = score(db, calendar, ledger, date(2026, 10, 23))

    assert [(c["preset"], c["origin"], c["events"]) for c in payload["cohorts"]] == [
        ("minervini", "artifact", 0),
        ("minervini", "artifact_pre_stage_fix", 0),
        ("canslim", "artifact", 1),
        ("canslim", "artifact_pre_stage_fix", 0),
    ]


def test_the_day_before_the_window_with_another_definition_makes_the_first_scored_day_a_baseline(db, calendar) -> None:
    add_bars(db, calendar, "NEW", date(2025, 10, 31), date(2025, 11, 17))
    ledger = as_ledger(
        ledger_day(date(2025, 10, 31), "2025-11-01T02:00:00Z", {"minervini": (DEF_B, [])}),
        ledger_day(date(2025, 11, 3), "2025-11-04T02:00:00Z", {"minervini": (DEF_A, ["NEW"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 11, 3))

    assert only_cohort(payload)["events"] == 0


def test_seed_to_live_keeps_its_entrants_and_does_not_reset_overlap(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    add_bars(db, calendar, "NEW", date(2026, 10, 5), date(2026, 10, 23), close=110.0)
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}, Origin.ARTIFACT),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, [])}, Origin.ARTIFACT),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["AAPL", "NEW"])}, Origin.LIVE),
    )

    cohorts = {cohort["origin"]: cohort for cohort in score(db, calendar, ledger, date(2026, 10, 23))["cohorts"]}

    assert horizon(cohorts["artifact"], 5)["sample_size"] == 1
    assert cohorts["live"]["events"] == 2
    assert horizon(cohorts["live"], 5)["sample_size"] == 1
    assert horizon(cohorts["live"], 5)["suppressed_overlapping"] == 1


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
        "effective_dates": 0.0,
        "pending_windows": 1,
        "oldest_open_sessions": 4,
        "thin": True,
        **EMPTY_STATS,
        "skipped_missing_history": 0,
        "suppressed_overlapping": 0,
    }


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


def test_beat_benchmark_counts_only_excess_above_zero(db, calendar) -> None:
    exit_day = date(2026, 10, 12)
    add_bars(db, calendar, "SPY", date(2026, 10, 5), exit_day, open_=400.0, close=401.0, closes={exit_day: 404.0})
    add_bars(db, calendar, "BEATS", date(2026, 10, 5), exit_day, closes={exit_day: 110.0})
    add_bars(db, calendar, "TIES", date(2026, 10, 5), exit_day, closes={exit_day: 101.0})
    add_bars(db, calendar, "UP_BUT_BEHIND", date(2026, 10, 5), exit_day, closes={exit_day: 100.5})
    add_bars(db, calendar, "FALLS", date(2026, 10, 5), exit_day, closes={exit_day: 95.0})
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["BEATS", "TIES", "UP_BUT_BEHIND", "FALLS"])})
    )

    block = horizon(only_cohort(score(db, calendar, ledger, exit_day)), 5)

    assert block["sample_size"] == 4
    assert block["positive_rate"] == 0.75
    assert block["beat_benchmark_rate"] == 0.25
    assert block["excess_sample_size"] == 4


def test_beat_benchmark_leaves_out_events_the_benchmark_does_not_cover(db, calendar) -> None:
    add_bars(db, calendar, "SPY", date(2026, 10, 5), date(2026, 10, 12), open_=400.0, close=404.0)
    add_bars(db, calendar, "BEATS", date(2026, 10, 5), date(2026, 10, 13), close=110.0)
    add_bars(db, calendar, "FALLS", date(2026, 10, 5), date(2026, 10, 13), close=50.0)
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["BEATS"])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["BEATS", "FALLS"])}),
    )

    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 13))), 5)

    assert (block["sample_size"], block["excess_sample_size"]) == (2, 1)
    assert block["beat_benchmark_rate"] == 1.0


def test_beat_benchmark_is_null_when_no_closed_event_has_a_benchmark(db, calendar) -> None:
    add_bars(db, calendar, "BHP.AX", date(2026, 10, 5), date(2026, 10, 16), market="AU", close=110.0)
    ledger = after_baseline(ledger_day(date(2026, 10, 5), "2026-10-05T22:00:00Z", {"minervini": (DEF_A, ["BHP.AX"])}))

    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 16), market="AU")), 5)

    assert block["sample_size"] == 1
    assert block["positive_rate"] == 1.0
    assert block["beat_benchmark_rate"] is None


def test_a_zero_match_preset_is_listed_with_null_stats(db, calendar) -> None:
    add_bars(db, calendar, "SPY", date(2026, 10, 5), date(2026, 10, 12))
    ledger = as_ledger(ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, [])}))

    cohort = only_cohort(score(db, calendar, ledger, date(2026, 10, 12)))

    assert (cohort["events"], cohort["signal_dates"]) == (0, 0)
    assert horizon(cohort, 5) == {
        "horizon_sessions": 5,
        "sample_size": 0,
        "signal_dates": 0,
        "effective_dates": 0.0,
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


SCORING_DATES = [
    (date(2026, 10, 5), "2026-10-06T02:00:00Z"),
    (date(2026, 10, 6), "2026-10-07T02:00:00Z"),
    (date(2026, 10, 7), "2026-10-08T02:00:00Z"),
    (date(2026, 10, 8), "2026-10-09T02:00:00Z"),
    (date(2026, 10, 9), "2026-10-10T02:00:00Z"),
    (date(2026, 10, 12), "2026-10-13T02:00:00Z"),
]


@pytest.mark.parametrize(
    ("picks_per_date", "sample_size", "signal_dates", "effective_dates", "thin"),
    [
        pytest.param([4, 4, 4, 4, 4], 20, 5, 5.0, False, id="exactly-20-and-exactly-5"),
        pytest.param([4, 3, 3, 3, 3, 3], 19, 6, 5.9, True, id="19-events-enough-dates"),
        pytest.param([6, 6, 5, 2, 1], 20, 5, 3.9, True, id="five-dates-but-uneven"),
        pytest.param([5, 5, 5, 5, 4], 24, 5, 5.0, False, id="rounds-up-to-exactly-5"),
        pytest.param([6, 5, 5, 5, 4], 25, 5, 4.9, True, id="rounds-down-below-5"),
        pytest.param([25], 25, 1, 1.0, True, id="one-busy-day"),
    ],
)
def test_thin_needs_twenty_closed_events_and_five_effective_dates(
    db, calendar, picks_per_date, sample_size, signal_dates, effective_dates, thin
) -> None:
    symbols = iter(f"S{index:02d}" for index in range(sum(picks_per_date)))
    days = []
    seen: list[str] = []
    for count, (day, stamp) in zip(picks_per_date, SCORING_DATES):
        seen = [*seen, *(next(symbols) for _ in range(count))]
        days.append(ledger_day(day, stamp, {"minervini": (DEF_A, seen)}))
    for symbol in seen:
        add_bars(db, calendar, symbol, date(2026, 10, 5), date(2026, 10, 23), close=110.0)

    block = horizon(only_cohort(score(db, calendar, after_baseline(*days), date(2026, 10, 23))), 5)

    assert (block["sample_size"], block["signal_dates"], block["effective_dates"], block["thin"]) == (
        sample_size,
        signal_dates,
        effective_dates,
        thin,
    )


def test_effective_dates_count_entry_sessions_so_a_late_publish_shares_the_next_days_entry(db, calendar) -> None:
    add_bars(db, calendar, "LATE", date(2026, 10, 5), date(2026, 10, 23))
    add_bars(db, calendar, "NEXT", date(2026, 10, 5), date(2026, 10, 23))
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T15:00:00Z", {"minervini": (DEF_A, ["LATE"])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["LATE", "NEXT"])}),
    )

    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 23))), 5)

    assert (block["sample_size"], block["signal_dates"], block["effective_dates"]) == (2, 2, 1.0)


def test_effective_dates_count_only_closed_events(db, calendar) -> None:
    add_bars(db, calendar, "OLDER", date(2026, 10, 5), date(2026, 10, 13))
    add_bars(db, calendar, "OPEN", date(2026, 10, 5), date(2026, 10, 13))
    ledger = after_baseline(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["OLDER"])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["OLDER"])}),
        ledger_day(date(2026, 10, 12), "2026-10-13T02:00:00Z", {"minervini": (DEF_A, ["OLDER", "OPEN"])}),
    )

    block = horizon(only_cohort(score(db, calendar, ledger, date(2026, 10, 13))), 5)

    assert (block["sample_size"], block["pending_windows"], block["effective_dates"]) == (1, 1, 1.0)


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


def test_a_day_lost_from_canonical_cannot_change_the_published_result(tmp_path, db, calendar, capsys) -> None:
    first = ledger_day(date(2026, 10, 5), "2026-10-06T01:00:00Z", {"minervini": (DEF_A, [])})
    lost = ledger_day(date(2026, 10, 6), "2026-10-07T01:00:00Z", {"minervini": (DEF_A, ["X"])})
    later = ledger_day(date(2026, 10, 7), "2026-10-08T01:00:00Z", {"minervini": (DEF_A, ["X"])})
    add_bars(db, calendar, "X", date(2026, 10, 5), date(2026, 10, 30), closes={date(2026, 10, 13): 200.0})
    market_dir = write_artifact(tmp_path / "artifacts", as_of="2026-10-30")
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[first, later], previous=[first, lost])

    assert run_write(market_dir, ledger_dir, db, calendar) is True

    board = json.loads((market_dir / "scoreboard.json").read_text())
    assert board["ledger_through"] == "2026-10-06"
    assert horizon(board["cohorts"][0], 5)["median_return_pct"] == 100.0
    assert "::warning::US canonical lost 1 line(s) that .previous holds" in capsys.readouterr().out


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


def test_a_rewound_export_ignores_ledger_days_after_its_as_of_date(db, calendar) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 10, 30))
    ledger = as_ledger(
        ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, [])}),
        ledger_day(date(2026, 10, 6), "2026-10-07T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}),
        ledger_day(date(2026, 10, 7), "2026-10-08T02:00:00Z", {"minervini": (DEF_A, ["AAPL", "MSFT"])}),
    )

    payload = score(db, calendar, ledger, date(2026, 10, 6))

    assert payload["as_of_date"] == "2026-10-06"
    assert payload["ledger_through"] == "2026-10-06"
    cohort = only_cohort(payload)
    assert (cohort["last_date"], cohort["events"]) == ("2026-10-06", 1)


def test_a_ledger_entirely_after_a_rewound_export_advertises_nothing(tmp_path, db, calendar, capsys) -> None:
    market_dir = write_artifact(tmp_path / "artifacts", as_of="2026-11-03")
    before = (market_dir / "manifest.market.json").read_bytes()
    later = ledger_day(date(2026, 11, 4), "2026-11-05T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])})
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[later])

    assert run_write(market_dir, ledger_dir, db, calendar) is False

    assert (market_dir / "manifest.market.json").read_bytes() == before
    assert "::warning::No usable US ledger" in capsys.readouterr().out


def test_a_ledger_that_ended_before_the_window_advertises_nothing(tmp_path, db, calendar, capsys) -> None:
    market_dir = write_artifact(tmp_path / "artifacts", as_of="2026-11-03")
    before = (market_dir / "manifest.market.json").read_bytes()
    stale = ledger_day(date(2025, 11, 2), "2025-11-03T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])})
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[stale])

    assert run_write(market_dir, ledger_dir, db, calendar) is False

    assert (market_dir / "manifest.market.json").read_bytes() == before
    assert not (market_dir / "scoreboard.json").exists()
    assert "::warning::No usable US ledger" in capsys.readouterr().out


def test_prices_are_read_only_from_the_window(tmp_path, db, calendar, monkeypatch, capsys) -> None:
    add_bars(db, calendar, "AAPL", date(2026, 10, 5), date(2026, 11, 3))
    market_dir = write_artifact(tmp_path / "artifacts", as_of="2026-11-03")
    ancient = ledger_day(date(2024, 1, 2), "2024-01-03T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])})
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[ancient, DAY_1005])
    seen = {}
    real_table = builder.StockPriceTable

    def recording_table(session, *, start, through):
        seen.update(start=start, through=through)
        return real_table(session, start=start, through=through)

    monkeypatch.setattr(builder, "StockPriceTable", recording_table)

    assert run_write(market_dir, ledger_dir, db, calendar) is True

    assert seen == {"start": date(2025, 11, 3), "through": date(2026, 11, 3)}
    payload = json.loads((market_dir / "scoreboard.json").read_text())
    assert (payload["recording_since"], payload["scored_since"]) == ("2024-01-02", "2026-10-05")
    assert "US: scored 1 of 2 ledger days into 1 cohorts" in capsys.readouterr().out


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


def test_a_nan_in_the_payload_fails_and_advertises_nothing(tmp_path, db, calendar, monkeypatch) -> None:
    market_dir = write_artifact(tmp_path / "artifacts")
    before = (market_dir / "manifest.market.json").read_bytes()
    ledger_dir = write_ledger_assets(tmp_path / "ledger", canonical=[DAY_1005])
    monkeypatch.setattr(builder, "build_scoreboard", lambda *_args, **_kwargs: {"cohorts": [float("nan")]})

    with pytest.raises(ValueError, match="Out of range float values"):
        run_write(market_dir, ledger_dir, db, calendar)

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


def closed_window_board(db, calendar) -> dict:
    """What the builder writes for a pick whose 5-session window closed."""
    for symbol in ("AAPL", "SPY"):
        add_bars(db, calendar, symbol, date(2026, 10, 5), date(2026, 10, 30), closes={date(2026, 10, 12): 110.0})
    return json.loads(json.dumps(score(db, calendar, as_ledger(DAY_1002, DAY_1005), date(2026, 10, 30)), allow_nan=False))


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
    # The publisher scans every JSON file it ships, so the dropped file cannot stay.
    assert not (tmp_path / "combined" / "markets" / "us" / "scoreboard.json").exists()
    relocate_into_generation(tmp_path / "combined")
    assert_live_only_static_isolation(tmp_path / "combined")


MISSING = object()
FIRST_COHORT = ("cohorts", 0)
FIRST_HORIZON = (*FIRST_COHORT, "horizons", 0)
HORIZON_FIELDS = (
    "horizon_sessions", "sample_size", "signal_dates", "effective_dates", "pending_windows", "oldest_open_sessions",
    "thin", "positive_rate", "beat_benchmark_rate", "median_return_pct", "median_excess_return_pct",
    "excess_sample_size", "skipped_missing_history", "suppressed_overlapping",
)
COHORT_FIELDS = ("preset", "name", "retired", "definition", "origin", "first_date", "last_date", "events", "horizons")
BOARD_FIELDS = (
    "schema_version", "market", "as_of_date", "benchmark", "recording_since", "scored_since", "ledger_through",
    "missing_sessions", "cohorts",
)
COUNTS = ("sample_size", "signal_dates", "pending_windows", "excess_sample_size", "skipped_missing_history", "suppressed_overlapping")
STATISTICS = ("positive_rate", "beat_benchmark_rate", "median_return_pct", "median_excess_return_pct", "effective_dates")


def change(path: tuple, value: object, name: str) -> object:
    return pytest.param(path, value, id=name)


MALFORMED = [
    change(("schema_version",), "static-scoreboard-v0", "wrong-schema"),
    change(("market",), "HK", "another-market"),
    change(("market",), ["US"], "market-not-text"),
    change(("as_of_date",), 20261012, "as-of-date-not-text"),
    change(("benchmark",), 5, "benchmark-not-text"),
    change(("recording_since",), 5, "recording-since-not-text"),
    change(("missing_sessions",), None, "missing-sessions-null"),
    change(("missing_sessions",), True, "missing-sessions-boolean"),
    change(("missing_sessions",), 1.5, "missing-sessions-fractional"),
    change(("missing_sessions",), -1, "missing-sessions-negative"),
    change(("cohorts",), {}, "cohorts-not-a-list"),
    change(FIRST_COHORT, 1, "cohort-not-an-object"),
    change(FIRST_COHORT, {}, "cohort-empty"),
    change((*FIRST_COHORT, "preset"), None, "preset-null"),
    change((*FIRST_COHORT, "retired"), "no", "retired-not-a-boolean"),
    change((*FIRST_COHORT, "origin"), "imagined", "origin-unknown"),
    change((*FIRST_COHORT, "origin"), [], "origin-unhashable"),
    change((*FIRST_COHORT, "events"), True, "events-boolean"),
    change((*FIRST_COHORT, "horizons"), {}, "horizons-not-a-list"),
    change((*FIRST_COHORT, "horizons"), [], "horizons-empty"),
    change((*FIRST_COHORT, "horizons", 2), MISSING, "horizon-60-absent"),
    change((*FIRST_COHORT, "horizons", 2, "sample_size"), "x", "last-horizon-count-not-a-number"),
    change((*FIRST_COHORT, "preset"), 5, "preset-not-text"),
    change((*FIRST_COHORT, "name"), {}, "name-not-text"),
    change((*FIRST_COHORT, "definition"), 5, "definition-not-text"),
    change((*FIRST_COHORT, "first_date"), None, "first-date-null"),
    change((*FIRST_COHORT, "last_date"), 5, "last-date-not-text"),
    change((*FIRST_COHORT, "events"), "x", "events-not-a-number"),
    change(("scored_since",), 5, "scored-since-not-text"),
    change(("ledger_through",), 5, "ledger-through-not-text"),
    change(FIRST_HORIZON, 1, "horizon-not-an-object"),
    change((*FIRST_HORIZON, "horizon_sessions"), 20, "horizon-repeated"),
    change((*FIRST_HORIZON, "horizon_sessions"), 10, "horizon-unknown"),
    change((*FIRST_HORIZON, "horizon_sessions"), "5", "horizon-not-a-number"),
    change((*FIRST_HORIZON, "horizon_sessions"), 5.0, "horizon-a-float"),
    change((*FIRST_HORIZON, "thin"), 1, "thin-not-a-boolean"),
    change((*FIRST_HORIZON, "oldest_open_sessions"), True, "oldest-open-boolean"),
    change((*FIRST_HORIZON, "oldest_open_sessions"), 1.5, "oldest-open-fractional"),
    change((*FIRST_HORIZON, "effective_dates"), None, "effective-dates-null"),
    change((*FIRST_HORIZON, "median_return_pct"), 10**400, "median-beyond-a-double"),
    *[change((name,), MISSING, f"without-{name}") for name in BOARD_FIELDS],
    *[change((*FIRST_COHORT, name), MISSING, f"cohort-without-{name}") for name in COHORT_FIELDS],
    *[change((*FIRST_HORIZON, name), MISSING, f"horizon-without-{name}") for name in HORIZON_FIELDS],
    *[change((*FIRST_HORIZON, name), None, f"{name}-null") for name in COUNTS],
    *[change((*FIRST_HORIZON, name), True, f"{name}-boolean") for name in COUNTS],
    *[change((*FIRST_HORIZON, name), 1.5, f"{name}-fractional") for name in COUNTS],
    *[change((*FIRST_HORIZON, name), "12", f"{name}-text") for name in STATISTICS],
    *[change((*FIRST_HORIZON, name), True, f"{name}-boolean") for name in STATISTICS],
    *[change((*FIRST_HORIZON, name), float("nan"), f"{name}-nan") for name in STATISTICS],
    *[change((*FIRST_HORIZON, name), float("inf"), f"{name}-infinite") for name in STATISTICS],
]


def _combined_assets_with_second_cohort(tmp_path, board: dict, second_cohort: object) -> dict:
    board["cohorts"].append(second_cohort)
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").write_text(json.dumps(board))
    return combine_one_market(artifacts, tmp_path / "combined").manifest["markets"]["US"]["assets"]


def test_a_second_valid_cohort_keeps_the_entry(tmp_path, db, calendar) -> None:
    board = closed_window_board(db, calendar)

    assets = _combined_assets_with_second_cohort(tmp_path, board, json.loads(json.dumps(board["cohorts"][0])))

    assert assets == {"scoreboard": {"path": "markets/us/scoreboard.json"}}


def test_a_malformed_second_cohort_loses_the_entry(tmp_path, db, calendar) -> None:
    board = closed_window_board(db, calendar)
    second = {**board["cohorts"][0], "name": {}}

    assert _combined_assets_with_second_cohort(tmp_path, board, second) == {}


def test_a_deeply_nested_scoreboard_loses_its_entry_and_the_market_stays(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").write_text("[" * 200000)

    result = combine_one_market(artifacts, tmp_path / "combined")

    assert result.manifest["markets"]["US"]["assets"] == {}
    assert result.manifest.get("unavailable_markets", []) == []


@pytest.mark.parametrize(("path", "value"), MALFORMED)
def test_a_malformed_real_scoreboard_loses_its_entry_and_the_market_stays(tmp_path, db, calendar, path, value) -> None:
    board = closed_window_board(db, calendar)
    target = board
    for key in path[:-1]:
        target = target[key]
    if value is MISSING:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").write_text(json.dumps(board))

    result = combine_one_market(artifacts, tmp_path / "combined")

    assert result.manifest["markets"]["US"]["assets"] == {}
    assert result.manifest["markets"]["US"]["pages"] == {"scan": {"path": "markets/us/scan/manifest.json"}}
    assert result.manifest.get("unavailable_markets", []) == []
    assert result.warnings == (
        "US scoreboard asset ignored: advertised file 'markets/us/scoreboard.json' "
        "is not a static-scoreboard-v1 file for US",
    )
    assert not (tmp_path / "combined" / "markets" / "us" / "scoreboard.json").exists()


def combined_scoreboard(tmp_path: Path, board: dict) -> dict:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").write_text(json.dumps(board, allow_nan=False))
    result = combine_one_market(artifacts, tmp_path / "combined")
    assert result.warnings == ()
    return result.manifest["markets"]["US"]["assets"]


def test_real_output_with_closed_and_open_windows_keeps_its_entry(tmp_path, db, calendar) -> None:
    board = closed_window_board(db, calendar)
    closed, open_ = horizon(board["cohorts"][0], 5), horizon(board["cohorts"][0], 20)

    assert (closed["sample_size"], closed["median_return_pct"], closed["median_excess_return_pct"]) == (1, 10.0, 0.0)
    assert (closed["oldest_open_sessions"], open_["sample_size"], open_["median_return_pct"]) == (None, 0, None)
    assert combined_scoreboard(tmp_path, board) == {"scoreboard": {"path": "markets/us/scoreboard.json"}}


def test_real_output_of_a_first_day_with_no_picks_keeps_its_entry(tmp_path, db, calendar) -> None:
    board = json.loads(json.dumps(score(db, calendar, as_ledger(DAY_1005), date(2026, 10, 12))))

    assert (board["benchmark"], board["cohorts"][0]["events"], horizon(board["cohorts"][0], 5)["positive_rate"]) == (None, 0, None)
    assert combined_scoreboard(tmp_path, board) == {"scoreboard": {"path": "markets/us/scoreboard.json"}}


@pytest.mark.parametrize("origin", list(Origin))
def test_real_output_of_every_origin_keeps_its_entry(tmp_path, db, calendar, origin) -> None:
    day = ledger_day(date(2026, 10, 5), "2026-10-06T02:00:00Z", {"minervini": (DEF_A, ["AAPL"])}, origin)
    board = json.loads(json.dumps(score(db, calendar, as_ledger(day), date(2026, 10, 12))))

    assert board["cohorts"][0]["origin"] == origin.value
    assert combined_scoreboard(tmp_path, board) == {"scoreboard": {"path": "markets/us/scoreboard.json"}}


def test_horizons_in_another_order_keep_their_entry(tmp_path, db, calendar) -> None:
    board = closed_window_board(db, calendar)
    board["cohorts"][0]["horizons"].reverse()

    assert [block["horizon_sessions"] for block in board["cohorts"][0]["horizons"]] == [60, 20, 5]
    assert combined_scoreboard(tmp_path, board) == {"scoreboard": {"path": "markets/us/scoreboard.json"}}


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


@pytest.mark.parametrize("advertised", [True, False], ids=["advertised", "unadvertised"])
def test_a_directory_at_the_scoreboard_path_loses_only_the_asset(tmp_path, advertised) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    if advertised:
        advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").mkdir()
    (market_dir / "scoreboard.json" / "nested.json").write_text("{}")

    result = combine_one_market(artifacts, tmp_path / "combined")

    assert result.manifest["markets"]["US"]["assets"] == {}
    assert result.manifest["markets"]["US"]["pages"] == {"scan": {"path": "markets/us/scan/manifest.json"}}
    assert not (tmp_path / "combined" / "markets" / "us" / "scoreboard.json").exists()


def test_an_unadvertised_scoreboard_file_is_not_shipped(tmp_path, db, calendar) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    (market_dir / "scoreboard.json").write_text(json.dumps(closed_window_board(db, calendar)))
    output = tmp_path / "combined"
    (output / "markets" / "us").mkdir(parents=True)
    (output / "markets" / "us" / "scoreboard.json").write_text("{left over from an earlier publish")

    StaticArtifactCombiner(
        schema_version=STATIC_SITE_SCHEMA_VERSION,
        supported_markets=("US",),
        default_market="US",
    ).combine(artifacts_dir=artifacts, fallback_artifacts_dir=None, output_dir=output, clean=False)

    assert not (output / "markets" / "us" / "scoreboard.json").exists()
    assert (output / "markets" / "us" / "scan" / "manifest.json").is_file()


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


def test_a_good_scoreboard_keeps_its_entry_and_every_other_broken_asset_still_rejects(tmp_path, db, calendar) -> None:
    artifacts = tmp_path / "artifacts"
    market_dir = write_artifact(artifacts)
    advertise_scoreboard(market_dir, {"path": "markets/us/scoreboard.json"})
    (market_dir / "scoreboard.json").write_text(json.dumps(closed_window_board(db, calendar)))

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
