"""SQL-level integration tests for scan_result_query builder.

Uses an in-memory SQLite database to verify that json_extract(),
CAST(), and ORDER BY work correctly for setup_engine fields.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Fixtures: lightweight in-memory schema (no ORM models needed)
# ---------------------------------------------------------------------------

_SETUP_ENGINE_A = {
    "setup_engine": {
        "setup_score": 85.5,
        "quality_score": 72.0,
        "readiness_score": 90.0,
        "pattern_confidence": 0.88,
        "pivot_price": 142.50,
        "distance_to_pivot_pct": 2.3,
        "atr14_pct": 3.1,
        "atr14_pct_trend": -0.2,
        "bb_width_pct": 5.5,
        "bb_width_pctile_252": 22.0,
        "volume_vs_50d": 1.5,
        "rs": 1.12,
        "rs_vs_spy_65d": 8.5,
        "rs_vs_spy_trend_20d": 0.03,
        "setup_ready": 1,
        "rs_line_new_high": 0,
        "pattern_primary": "VCP",
        "pivot_type": "breakout",
    },
}

_SETUP_ENGINE_B = {
    "setup_engine": {
        "setup_score": 9.2,
        "quality_score": 40.0,
        "readiness_score": 15.0,
        "pattern_confidence": 0.55,
        "pivot_price": 55.00,
        "distance_to_pivot_pct": 8.1,
        "atr14_pct": 6.0,
        "atr14_pct_trend": 0.5,
        "bb_width_pct": 12.0,
        "bb_width_pctile_252": 80.0,
        "volume_vs_50d": 0.7,
        "rs": 0.95,
        "rs_vs_spy_65d": -3.2,
        "rs_vs_spy_trend_20d": -0.01,
        "setup_ready": 0,
        "rs_line_new_high": 1,
        "pattern_primary": "Cup-with-Handle",
        "pivot_type": "pullback",
    },
}

_NO_SETUP_ENGINE = {"vcp_score": 50.0}  # has details but no setup_engine


@pytest.fixture()
def db_session():
    """Create an in-memory SQLite database with a minimal scan_results table."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    meta = MetaData()

    Table(
        "scans", meta,
        Column("scan_id", String(36), primary_key=True),
    )

    Table(
        "scan_results", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("scan_id", String(36)),
        Column("symbol", String(10), nullable=False),
        Column("composite_score", Float),
        Column("details", Text),  # JSON stored as text in SQLite
    )

    meta.create_all(engine)

    with Session(engine) as session:
        # Insert parent scan
        session.execute(
            text("INSERT INTO scans (scan_id) VALUES (:sid)"),
            {"sid": "scan-1"},
        )
        # Insert test rows
        rows = [
            ("scan-1", "AAPL", 80.0, json.dumps(_SETUP_ENGINE_A)),
            ("scan-1", "MSFT", 60.0, json.dumps(_SETUP_ENGINE_B)),
            ("scan-1", "GOOG", 90.0, json.dumps(_NO_SETUP_ENGINE)),
            ("scan-1", "AMZN", 70.0, None),  # NULL details
        ]
        for scan_id, sym, score, details in rows:
            session.execute(
                text(
                    "INSERT INTO scan_results (scan_id, symbol, composite_score, details) "
                    "VALUES (:sid, :sym, :score, :details)"
                ),
                {"sid": scan_id, "sym": sym, "score": score, "details": details},
            )
        session.commit()
        yield session


# ---------------------------------------------------------------------------
# Sort tests
# ---------------------------------------------------------------------------


class TestNumericSortDescending:
    """Sort by se_setup_score DESC — numeric values correct, nulls last."""

    def test_sort_desc(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM scan_results "
                "ORDER BY score DESC NULLS LAST"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        # AAPL (85.5), MSFT (9.2), then GOOG/AMZN (null) last
        assert symbols[0] == "AAPL"
        assert symbols[1] == "MSFT"
        assert set(symbols[2:]) == {"GOOG", "AMZN"}


class TestNumericSortAscending:
    """Sort by se_setup_score ASC — smallest first, nulls last."""

    def test_sort_asc(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM scan_results "
                "ORDER BY score ASC NULLS LAST"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        assert symbols[0] == "MSFT"  # 9.2
        assert symbols[1] == "AAPL"  # 85.5
        assert set(symbols[2:]) == {"GOOG", "AMZN"}


class TestStringSortLexicographic:
    """Sort by se_pattern_primary — lexicographic, no cast."""

    def test_string_sort(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  json_extract(details, '$.setup_engine.pattern_primary') as pattern "
                "FROM scan_results "
                "ORDER BY pattern ASC NULLS LAST"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        # Cup-with-Handle < VCP lexicographically
        assert symbols[0] == "MSFT"
        assert symbols[1] == "AAPL"
        assert set(symbols[2:]) == {"GOOG", "AMZN"}


class TestNumericCastPreventsLexicographicSort:
    """Verify CAST ensures numeric sort even when JSON values are numbers.

    SQLite's json_extract preserves JSON types (numbers stay numeric),
    so native JSON numbers sort correctly. The CAST is a safety net for
    edge cases (string-encoded numbers). We verify CAST produces correct
    results and that the sort order is numerically correct.
    """

    def test_with_cast_is_correct(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM scan_results "
                "WHERE score IS NOT NULL "
                "ORDER BY score DESC"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        assert symbols[0] == "AAPL", "With CAST, numeric sort puts 85.5 before 9.2"
        assert symbols[1] == "MSFT"

    def test_string_encoded_numbers_need_cast(self, db_session):
        """When numbers are stored as JSON strings, CAST is essential."""
        db_session.execute(
            text(
                "INSERT INTO scan_results (scan_id, symbol, composite_score, details) "
                "VALUES (:sid, :sym, :score, :details)"
            ),
            {
                "sid": "scan-1", "sym": "TSLA", "score": 50.0,
                "details": '{"setup_engine": {"setup_score": "9.2"}}',
            },
        )
        db_session.execute(
            text(
                "INSERT INTO scan_results (scan_id, symbol, composite_score, details) "
                "VALUES (:sid, :sym, :score, :details)"
            ),
            {
                "sid": "scan-1", "sym": "META", "score": 50.0,
                "details": '{"setup_engine": {"setup_score": "85.5"}}',
            },
        )

        # Without CAST: string "9.2" > "85.5" lexicographically
        rows_nocast = db_session.execute(
            text(
                "SELECT symbol, json_extract(details, '$.setup_engine.setup_score') as score "
                "FROM scan_results "
                "WHERE symbol IN ('TSLA', 'META') "
                "ORDER BY score DESC"
            )
        ).fetchall()
        assert rows_nocast[0][0] == "TSLA", "String '9.2' > '85.5' lexicographically"

        # With CAST: 85.5 > 9.2 numerically
        rows_cast = db_session.execute(
            text(
                "SELECT symbol, "
                "  CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM scan_results "
                "WHERE symbol IN ('TSLA', 'META') "
                "ORDER BY score DESC"
            )
        ).fetchall()
        assert rows_cast[0][0] == "META", "CAST fixes: 85.5 > 9.2 numerically"


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


class TestRangeFilter:
    """Range filter on se_setup_score."""

    def test_min_filter(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM scan_results "
                "WHERE json_extract(details, '$.setup_engine.setup_score') IS NOT NULL "
                "AND CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) >= 50"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"AAPL"}

    def test_max_filter(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM scan_results "
                "WHERE json_extract(details, '$.setup_engine.setup_score') IS NOT NULL "
                "AND CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) <= 50"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"MSFT"}

    def test_range_filter(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM scan_results "
                "WHERE json_extract(details, '$.setup_engine.setup_score') IS NOT NULL "
                "AND CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) >= 5 "
                "AND CAST(json_extract(details, '$.setup_engine.setup_score') AS REAL) <= 50"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"MSFT"}


class TestBooleanFilter:
    """Boolean filter on se_setup_ready."""

    def test_true_filter(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM scan_results "
                "WHERE json_extract(details, '$.setup_engine.setup_ready') IS NOT NULL "
                "AND json_extract(details, '$.setup_engine.setup_ready') = 1"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"AAPL"}

    def test_false_filter(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM scan_results "
                "WHERE json_extract(details, '$.setup_engine.setup_ready') IS NOT NULL "
                "AND json_extract(details, '$.setup_engine.setup_ready') = 0"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"MSFT"}


class TestCategoricalFilter:
    """Categorical filter on se_pattern_primary."""

    def test_include(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM scan_results "
                "WHERE json_extract(details, '$.setup_engine.pattern_primary') IN ('VCP')"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"AAPL"}

    def test_exclude(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM scan_results "
                "WHERE json_extract(details, '$.setup_engine.pattern_primary') NOT IN ('VCP') "
                "AND json_extract(details, '$.setup_engine.pattern_primary') IS NOT NULL"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"MSFT"}


class TestMissingSetupEngineKey:
    """When setup_engine key is absent, json_extract returns NULL."""

    def test_no_setup_engine_returns_null(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  json_extract(details, '$.setup_engine.setup_score') as score "
                "FROM scan_results "
                "WHERE symbol = 'GOOG'"
            )
        ).fetchall()

        assert rows[0][1] is None

    def test_null_details_returns_null(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  json_extract(details, '$.setup_engine.setup_score') as score "
                "FROM scan_results "
                "WHERE symbol = 'AMZN'"
            )
        ).fetchall()

        assert rows[0][1] is None
