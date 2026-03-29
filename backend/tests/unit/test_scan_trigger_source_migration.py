from __future__ import annotations

from sqlalchemy import create_engine, text

from app.db_migrations.scan_trigger_source_migration import migrate_scan_trigger_source
from app.infra.db.portability import column_names


def test_scan_trigger_source_migration_adds_column_and_backfills_manual():
    engine = create_engine("sqlite:///:memory:")

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE scans (
                    id INTEGER PRIMARY KEY,
                    scan_id VARCHAR(36) NOT NULL UNIQUE
                )
                """
            )
        )
        conn.execute(
            text("INSERT INTO scans (id, scan_id) VALUES (1, 'scan-001')")
        )

    migrate_scan_trigger_source(engine)

    with engine.connect() as conn:
        assert "trigger_source" in column_names(conn, "scans")
        rows = conn.execute(
            text("SELECT trigger_source FROM scans WHERE scan_id = 'scan-001'")
        ).fetchall()

    assert rows == [("manual",)]
