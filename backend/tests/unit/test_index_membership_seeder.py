"""Tests for the index-membership seeder.

Covers idempotency (re-running with same CSV = all unchanged), upsert
semantics (bumped as_of_date = all updated), dry-run (no writes), and
input validation (missing header, empty rows).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
import app.models.stock_universe  # noqa: F401
from app.models.stock_universe import StockUniverseIndexMembership
from app.services.index_membership_seeder import SeedCounts, seed_from_csv


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def _write_csv(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    """Build a symbol,name CSV. Header is always emitted."""
    path = tmp_path / name
    buf = io.StringIO()
    buf.write("symbol,name\n")
    for symbol, company in rows:
        buf.write(f"{symbol},{company}\n")
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


class TestSeedFromCsv:
    def test_initial_seed_inserts_all_rows(self, session, tmp_path):
        csv_path = _write_csv(
            tmp_path,
            "hsi_constituents_2025-05.csv",
            [("0700.HK", "Tencent Holdings"), ("0005.HK", "HSBC Holdings")],
        )
        counts = seed_from_csv(
            session, csv_path, index_name="HSI", as_of_date="2025-05-01"
        )
        assert counts == SeedCounts(added=2, updated=0, unchanged=0, skipped=0)

        rows = session.query(StockUniverseIndexMembership).all()
        assert {r.symbol for r in rows} == {"0700.HK", "0005.HK"}
        assert all(r.index_name == "HSI" for r in rows)
        assert all(r.as_of_date == "2025-05-01" for r in rows)
        assert all(r.source == "seed_v1" for r in rows)

    def test_rerun_with_same_csv_is_all_unchanged(self, session, tmp_path):
        csv_path = _write_csv(
            tmp_path, "hsi.csv", [("0700.HK", "Tencent")]
        )
        seed_from_csv(session, csv_path, index_name="HSI", as_of_date="2025-05-01")
        counts = seed_from_csv(
            session, csv_path, index_name="HSI", as_of_date="2025-05-01"
        )
        assert counts == SeedCounts(added=0, updated=0, unchanged=1, skipped=0)

    def test_bumping_as_of_date_updates_existing_rows(self, session, tmp_path):
        csv_path = _write_csv(tmp_path, "hsi.csv", [("0700.HK", "Tencent")])
        seed_from_csv(session, csv_path, index_name="HSI", as_of_date="2025-05-01")

        counts = seed_from_csv(
            session, csv_path, index_name="HSI", as_of_date="2025-08-01"
        )
        assert counts == SeedCounts(added=0, updated=1, unchanged=0, skipped=0)

        row = session.query(StockUniverseIndexMembership).one()
        assert row.as_of_date == "2025-08-01"

    def test_dry_run_reports_counts_without_writing(self, session, tmp_path):
        csv_path = _write_csv(
            tmp_path, "hsi.csv", [("0700.HK", "Tencent"), ("0005.HK", "HSBC")]
        )
        counts = seed_from_csv(
            session,
            csv_path,
            index_name="HSI",
            as_of_date="2025-05-01",
            dry_run=True,
        )
        assert counts == SeedCounts(added=2, updated=0, unchanged=0, skipped=0)
        # DB is untouched.
        assert session.query(StockUniverseIndexMembership).count() == 0

    def test_symbols_are_normalized_to_uppercase_and_stripped(
        self, session, tmp_path
    ):
        # Seed operators sometimes paste from provider websites with
        # inconsistent whitespace / case. The seeder normalises so the
        # membership table stores the canonical form that get_active_symbols
        # filter matches.
        csv_path = _write_csv(
            tmp_path, "hsi.csv", [("  0700.hk  ", "Tencent")]
        )
        seed_from_csv(session, csv_path, index_name="HSI", as_of_date="2025-05-01")
        row = session.query(StockUniverseIndexMembership).one()
        assert row.symbol == "0700.HK"

    def test_index_name_is_uppercased(self, session, tmp_path):
        csv_path = _write_csv(tmp_path, "x.csv", [("0700.HK", "Tencent")])
        seed_from_csv(session, csv_path, index_name="hsi", as_of_date="2025-05-01")
        row = session.query(StockUniverseIndexMembership).one()
        assert row.index_name == "HSI"

    def test_blank_rows_are_skipped_not_errored(self, session, tmp_path):
        # The csv reader will yield a dict with symbol="" for a line with
        # only commas; the seeder drops these silently rather than inserting
        # empty strings (which would violate the NOT NULL+empty-check pairing
        # callers rely on).
        path = tmp_path / "hsi.csv"
        path.write_text(
            "symbol,name\n0700.HK,Tencent\n,\n0005.HK,HSBC\n", encoding="utf-8"
        )
        counts = seed_from_csv(
            session, path, index_name="HSI", as_of_date="2025-05-01"
        )
        assert counts.added == 2
        # The blank row isn't surfaced in skipped either — DictReader
        # just doesn't yield a symbol for it. This is fine; skipped is
        # reserved for explicit seeder rejections if we add any.

    def test_missing_symbol_header_raises(self, session, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("ticker,company\n0700.HK,Tencent\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required 'symbol' header"):
            seed_from_csv(
                session, path, index_name="HSI", as_of_date="2025-05-01"
            )

    def test_different_source_triggers_update_even_when_asof_unchanged(
        self, session, tmp_path
    ):
        csv_path = _write_csv(tmp_path, "hsi.csv", [("0700.HK", "Tencent")])
        seed_from_csv(
            session,
            csv_path,
            index_name="HSI",
            as_of_date="2025-05-01",
            source="seed_v1",
        )
        counts = seed_from_csv(
            session,
            csv_path,
            index_name="HSI",
            as_of_date="2025-05-01",
            source="manual_correction",
        )
        assert counts.updated == 1
        row = session.query(StockUniverseIndexMembership).one()
        assert row.source == "manual_correction"
