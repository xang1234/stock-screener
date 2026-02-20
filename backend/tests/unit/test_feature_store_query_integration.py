"""SQL-level integration tests for feature_store_query builder.

Uses an in-memory SQLite database to verify that json_extract(),
CAST(), and ORDER BY work correctly for setup_engine fields
in the stock_feature_daily table.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Fixtures: lightweight in-memory schema
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
    "composite_score": 80.0,
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
    "composite_score": 60.0,
}

_NO_SETUP_ENGINE = {"vcp_score": 50.0, "composite_score": 90.0}


@pytest.fixture()
def db_session():
    """Create an in-memory SQLite database with a minimal feature store schema."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    meta = MetaData()

    Table(
        "feature_runs", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("as_of_date", Date, nullable=False),
        Column("run_type", Text, nullable=False),
        Column("status", Text, nullable=False),
    )

    Table(
        "stock_feature_daily", meta,
        Column("run_id", Integer, primary_key=True),
        Column("symbol", Text, nullable=False, primary_key=True),
        Column("as_of_date", Date, nullable=False),
        Column("composite_score", Float, nullable=True),
        Column("overall_rating", Integer, nullable=True),
        Column("passes_count", Integer, nullable=True),
        Column("details_json", Text, nullable=True),
    )

    meta.create_all(engine)

    with Session(engine) as session:
        # Insert parent run
        session.execute(
            text(
                "INSERT INTO feature_runs (id, as_of_date, run_type, status) "
                "VALUES (1, '2026-01-15', 'daily_snapshot', 'published')"
            )
        )
        # Insert test rows
        rows = [
            (1, "AAPL", "2026-01-15", 80.0, json.dumps(_SETUP_ENGINE_A)),
            (1, "MSFT", "2026-01-15", 60.0, json.dumps(_SETUP_ENGINE_B)),
            (1, "GOOG", "2026-01-15", 90.0, json.dumps(_NO_SETUP_ENGINE)),
            (1, "AMZN", "2026-01-15", 70.0, None),
        ]
        for run_id, sym, date, score, details in rows:
            session.execute(
                text(
                    "INSERT INTO stock_feature_daily "
                    "(run_id, symbol, as_of_date, composite_score, details_json) "
                    "VALUES (:rid, :sym, :dt, :score, :details)"
                ),
                {"rid": run_id, "sym": sym, "dt": date, "score": score, "details": details},
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
                "  CAST(json_extract(details_json, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM stock_feature_daily "
                "ORDER BY score DESC NULLS LAST"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        assert symbols[0] == "AAPL"
        assert symbols[1] == "MSFT"
        assert set(symbols[2:]) == {"GOOG", "AMZN"}


class TestNumericSortAscending:
    """Sort by se_setup_score ASC — smallest first, nulls last."""

    def test_sort_asc(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  CAST(json_extract(details_json, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM stock_feature_daily "
                "ORDER BY score ASC NULLS LAST"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        assert symbols[0] == "MSFT"
        assert symbols[1] == "AAPL"
        assert set(symbols[2:]) == {"GOOG", "AMZN"}


class TestStringSortLexicographic:
    """Sort by se_pattern_primary — lexicographic, no cast."""

    def test_string_sort(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  json_extract(details_json, '$.setup_engine.pattern_primary') as pattern "
                "FROM stock_feature_daily "
                "ORDER BY pattern ASC NULLS LAST"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        assert symbols[0] == "MSFT"  # Cup-with-Handle
        assert symbols[1] == "AAPL"  # VCP
        assert set(symbols[2:]) == {"GOOG", "AMZN"}


class TestNumericCastPreventsLexicographicSort:
    """Verify CAST ensures numeric sort even when JSON values are numbers.

    SQLite's json_extract preserves JSON types (numbers stay numeric),
    so native JSON numbers sort correctly. The CAST is a safety net for
    edge cases (string-encoded numbers).
    """

    def test_with_cast_is_correct(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  CAST(json_extract(details_json, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM stock_feature_daily "
                "WHERE score IS NOT NULL "
                "ORDER BY score DESC"
            )
        ).fetchall()

        symbols = [r[0] for r in rows]
        assert symbols[0] == "AAPL"
        assert symbols[1] == "MSFT"

    def test_string_encoded_numbers_need_cast(self, db_session):
        """When numbers are stored as JSON strings, CAST is essential."""
        db_session.execute(
            text(
                "INSERT INTO stock_feature_daily "
                "(run_id, symbol, as_of_date, composite_score, details_json) "
                "VALUES (:rid, :sym, :dt, :score, :details)"
            ),
            {
                "rid": 1, "sym": "TSLA", "dt": "2026-01-15", "score": 50.0,
                "details": '{"setup_engine": {"setup_score": "9.2"}}',
            },
        )
        db_session.execute(
            text(
                "INSERT INTO stock_feature_daily "
                "(run_id, symbol, as_of_date, composite_score, details_json) "
                "VALUES (:rid, :sym, :dt, :score, :details)"
            ),
            {
                "rid": 1, "sym": "META", "dt": "2026-01-15", "score": 50.0,
                "details": '{"setup_engine": {"setup_score": "85.5"}}',
            },
        )

        # Without CAST: string "9.2" > "85.5" lexicographically
        rows_nocast = db_session.execute(
            text(
                "SELECT symbol, json_extract(details_json, '$.setup_engine.setup_score') as score "
                "FROM stock_feature_daily "
                "WHERE symbol IN ('TSLA', 'META') "
                "ORDER BY score DESC"
            )
        ).fetchall()
        assert rows_nocast[0][0] == "TSLA", "String '9.2' > '85.5' lexicographically"

        # With CAST: 85.5 > 9.2 numerically
        rows_cast = db_session.execute(
            text(
                "SELECT symbol, "
                "  CAST(json_extract(details_json, '$.setup_engine.setup_score') AS REAL) as score "
                "FROM stock_feature_daily "
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
                "SELECT symbol FROM stock_feature_daily "
                "WHERE json_extract(details_json, '$.setup_engine.setup_score') IS NOT NULL "
                "AND CAST(json_extract(details_json, '$.setup_engine.setup_score') AS REAL) >= 50"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"AAPL"}

    def test_max_filter(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM stock_feature_daily "
                "WHERE json_extract(details_json, '$.setup_engine.setup_score') IS NOT NULL "
                "AND CAST(json_extract(details_json, '$.setup_engine.setup_score') AS REAL) <= 50"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"MSFT"}


class TestBooleanFilter:
    """Boolean filter on se_setup_ready."""

    def test_true_filter(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM stock_feature_daily "
                "WHERE json_extract(details_json, '$.setup_engine.setup_ready') IS NOT NULL "
                "AND json_extract(details_json, '$.setup_engine.setup_ready') = 1"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"AAPL"}


class TestCategoricalFilter:
    """Categorical filter on se_pattern_primary."""

    def test_include(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM stock_feature_daily "
                "WHERE json_extract(details_json, '$.setup_engine.pattern_primary') IN ('VCP')"
            )
        ).fetchall()

        symbols = {r[0] for r in rows}
        assert symbols == {"AAPL"}

    def test_exclude(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol FROM stock_feature_daily "
                "WHERE json_extract(details_json, '$.setup_engine.pattern_primary') NOT IN ('VCP') "
                "AND json_extract(details_json, '$.setup_engine.pattern_primary') IS NOT NULL"
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
                "  json_extract(details_json, '$.setup_engine.setup_score') as score "
                "FROM stock_feature_daily "
                "WHERE symbol = 'GOOG'"
            )
        ).fetchall()

        assert rows[0][1] is None

    def test_null_details_returns_null(self, db_session):
        rows = db_session.execute(
            text(
                "SELECT symbol, "
                "  json_extract(details_json, '$.setup_engine.setup_score') as score "
                "FROM stock_feature_daily "
                "WHERE symbol = 'AMZN'"
            )
        ).fetchall()

        assert rows[0][1] is None
