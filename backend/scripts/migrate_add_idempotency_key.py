#!/usr/bin/env python3
"""Migration: Add idempotency_key column to scans table.

Safe to run multiple times — checks if column already exists.

Usage:
    cd backend
    source venv/bin/activate
    python scripts/migrate_add_idempotency_key.py
"""

import sqlite3
import sys
from pathlib import Path

# Resolve DB path (same logic as app config)
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "stockscanner.db"


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(scans)")
    columns = {row[1] for row in cursor.fetchall()}

    if "idempotency_key" in columns:
        print("Column 'idempotency_key' already exists — nothing to do.")
        conn.close()
        return

    print("Adding 'idempotency_key' column to scans table...")
    cursor.execute(
        "ALTER TABLE scans ADD COLUMN idempotency_key VARCHAR(64)"
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_scans_idempotency_key "
        "ON scans(idempotency_key)"
    )
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
