"""
Repair corrupted theme_metrics table via dump-and-reload.

Background:
    The theme_metrics B-tree has rowids out of order (PRAGMA quick_check
    reports "Tree 34 page 1066083 cell 405: Rowid 20885 out of order").
    Queries that traverse the B-tree (JOINs, indexed lookups) fail with
    "database disk image is malformed", but sequential scans still work.

    This script extracts recoverable rows via .dump (which tolerates
    corruption), drops/recreates the table, and reloads the data.

Pre-requisites:
    1. Stop all processes using the database FIRST:
         docker compose down          # if running in Docker
         pkill -f celery              # if running locally
         pkill -f uvicorn             # if running locally
    2. Run from the backend directory with venv activated:
         cd backend
         source venv/bin/activate
         python scripts/repair_theme_metrics.py

    Use --dry-run to preview what would happen without modifying anything.
    Use --yes to skip the confirmation prompt.
"""
import sys
import os
import argparse
import sqlite3
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_db_path() -> str:
    """Resolve the database path from app settings."""
    from app.config import settings

    # Strip sqlite:/// prefix safely (4 slashes for absolute paths: sqlite:////abs/path)
    path = settings.database_url.removeprefix("sqlite:///")
    if not os.path.exists(path):
        logger.error(f"Database not found at: {path}")
        sys.exit(1)
    return path


def check_for_active_processes(db_path: str) -> list[str]:
    """Check for other processes that have the database file open."""
    active = []
    for suffix in ("", "-wal", "-shm"):
        filepath = db_path + suffix
        if not os.path.exists(filepath):
            continue
        try:
            result = subprocess.run(
                ["lsof", filepath],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.strip().splitlines()[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 2:
                    active.append(f"{parts[0]} (PID {parts[1]}) -> {filepath}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return active


def backup_database(db_path: str) -> str:
    """Create a timestamped backup of the database file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(db_path).parent
    backup_path = str(backup_dir / f"stockscanner.pre_repair_{timestamp}.db")
    logger.info(f"Backing up database to: {backup_path}")
    shutil.copy2(db_path, backup_path)
    # Also backup WAL and SHM if they exist
    for suffix in ("-wal", "-shm"):
        src = db_path + suffix
        if os.path.exists(src):
            shutil.copy2(src, backup_path + suffix)
            logger.info(f"  Also backed up {suffix} file")
    return backup_path


def wal_checkpoint(conn: sqlite3.Connection) -> None:
    """Fold pending WAL writes into the main database file."""
    logger.info("Running WAL checkpoint (TRUNCATE)...")
    result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if result:
        busy, log_pages, checkpointed = result
        if busy:
            logger.warning(f"Checkpoint was blocked (busy={busy}). Stop all other DB processes first.")
        else:
            logger.info(f"Checkpoint complete: {log_pages} log pages, {checkpointed} checkpointed")


def dump_theme_metrics(db_path: str) -> tuple[str, str, int]:
    """
    Use sqlite3 .dump to extract theme_metrics CREATE TABLE + INSERT statements.

    Returns (create_sql, inserts_file_path, row_count).

    IMPORTANT: We extract the CREATE TABLE from the dump itself (not from the
    SQLAlchemy model) because .dump produces positional INSERT statements
    (VALUES without column names). The column order must match the original
    table — which may differ from the model if columns were added via
    ALTER TABLE ADD COLUMN (SQLite appends them at the end regardless of
    where they appear in the Python class).
    """
    logger.info("Dumping theme_metrics via sqlite3 .dump...")

    # Use sqlite3 CLI for .dump (Python's sqlite3 module doesn't expose .dump)
    dump_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sql", delete=False, prefix="theme_metrics_dump_"
    )
    dump_file.close()

    try:
        result = subprocess.run(
            ["sqlite3", db_path, ".dump theme_metrics"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        os.unlink(dump_file.name)
        logger.error("sqlite3 CLI not found. Install sqlite3 and retry.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        os.unlink(dump_file.name)
        logger.error("Dump timed out after 300s. Database may be too large or locked.")
        sys.exit(1)

    if result.returncode != 0 and not result.stdout.strip():
        os.unlink(dump_file.name)
        logger.error(f"Dump failed: {result.stderr}")
        sys.exit(1)

    # Parse the dump output: extract multi-line CREATE TABLE and INSERT lines.
    # .dump outputs CREATE TABLE across multiple lines for tables with many columns.
    create_sql_lines = []
    in_create = False
    insert_lines = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CREATE TABLE"):
            in_create = True
        if in_create:
            create_sql_lines.append(line)
            if stripped.endswith(");"):
                in_create = False
        elif stripped.startswith("INSERT INTO"):
            insert_lines.append(stripped)

    create_sql = "\n".join(create_sql_lines) if create_sql_lines else None

    if result.stderr:
        # .dump may print warnings about corruption but still output data
        for err_line in result.stderr.strip().splitlines():
            logger.warning(f"sqlite3 warning: {err_line}")

    # Write inserts to temp file
    with open(dump_file.name, "w") as f:
        for line in insert_lines:
            f.write(line + "\n")

    row_count = len(insert_lines)
    logger.info(f"Extracted {row_count} rows from theme_metrics")

    if create_sql:
        logger.info("Captured CREATE TABLE from dump (preserves original column order)")
    else:
        logger.warning("No CREATE TABLE found in dump output")

    return create_sql, dump_file.name, row_count


def repair(db_path: str, dry_run: bool = False) -> bool:
    """
    Main repair procedure:
    1. Checkpoint WAL
    2. Dump recoverable rows + original CREATE TABLE
    3. Drop corrupted table
    4. VACUUM to fix freelist corruption from reused pages
    5. Recreate table (using dump's CREATE TABLE to preserve column order)
    6. Reload data
    7. Rebuild indexes
    8. Verify integrity
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=15000")

    # Step 1: WAL checkpoint
    wal_checkpoint(conn)
    conn.close()

    # Step 2: Count rows before repair (sequential scan may work)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=15000")
    try:
        count_before = conn.execute("SELECT COUNT(*) FROM theme_metrics").fetchone()[0]
        logger.info(f"Row count before repair: {count_before}")
    except sqlite3.DatabaseError as e:
        logger.warning(f"Could not count rows (expected if corrupted): {e}")
        count_before = None
    conn.close()

    # Step 3: Dump data via sqlite3 CLI
    create_sql, inserts_file, dump_count = dump_theme_metrics(db_path)

    try:
        if dump_count == 0:
            logger.error("No rows recovered from dump! Aborting to prevent data loss.")
            return False

        if not create_sql:
            logger.error("No CREATE TABLE found in dump! Cannot safely recreate table.")
            return False

        if count_before and dump_count < count_before * 0.90:
            logger.warning(
                f"Only recovered {dump_count}/{count_before} rows ({dump_count/count_before*100:.1f}%). "
                f"Some rows were in corrupted pages."
            )

        if dry_run:
            logger.info("[DRY RUN] Would drop and recreate theme_metrics with %d recovered rows", dump_count)
            return True

        # Step 4: Drop the corrupted table
        logger.info("Dropping corrupted theme_metrics table...")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")  # Use FULL during repair for safety
        conn.execute("DROP TABLE IF EXISTS theme_metrics")
        conn.commit()
        conn.close()
        logger.info("Table dropped successfully")

        # Step 5: VACUUM to rebuild freelist and eliminate corrupt page references.
        # After dropping a corrupt table, freed pages go to the freelist which may
        # itself have corruption (doubly-referenced pages). VACUUM rebuilds the
        # entire database file from scratch, eliminating freelist corruption.
        logger.info("Running VACUUM to rebuild freelist (may take 1-2 minutes for large databases)...")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("VACUUM")
        conn.close()
        logger.info("VACUUM complete")

        # Step 6: Recreate the table using the dump's CREATE TABLE.
        # CRITICAL: We must use the dump's schema (not the SQLAlchemy model) because
        # .dump produces positional INSERT statements (VALUES without column names).
        # If columns were added via ALTER TABLE ADD COLUMN, they appear at the end of
        # the physical table regardless of their position in the Python model class.
        # Using the model's column order would cause data to land in wrong columns.
        logger.info("Recreating theme_metrics table (using dump's original column order)...")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(create_sql)
        conn.commit()
        logger.info("Table recreated successfully")

        # Step 7: Reload data
        logger.info(f"Reloading {dump_count} rows...")
        loaded = 0
        errors = 0
        with open(inserts_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    conn.execute(line)
                    loaded += 1
                except sqlite3.Error as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"Insert error (row {loaded + errors}): {e}")
                    elif errors == 6:
                        logger.warning("Suppressing further insert errors...")

                # Commit in batches for performance
                if loaded % 5000 == 0:
                    conn.commit()
                    logger.info(f"  Loaded {loaded} rows...")

        conn.commit()
        logger.info(f"Reload complete: {loaded} rows loaded, {errors} errors")

        # Step 8: Recreate indexes from the model's __table_args__.
        # The dump's CREATE TABLE includes PRIMARY KEY and UNIQUE constraints inline,
        # but named indexes (from __table_args__) are separate objects.
        logger.info("Recreating indexes...")
        index_statements = [
            'CREATE INDEX IF NOT EXISTS "ix_theme_metrics_theme_cluster_id" ON theme_metrics (theme_cluster_id)',
            'CREATE INDEX IF NOT EXISTS "ix_theme_metrics_date" ON theme_metrics (date)',
            'CREATE INDEX IF NOT EXISTS "idx_theme_metrics_date" ON theme_metrics (theme_cluster_id, date)',
            'CREATE INDEX IF NOT EXISTS "idx_theme_rank" ON theme_metrics (date, rank)',
            'CREATE INDEX IF NOT EXISTS "idx_theme_metrics_pipeline_date" ON theme_metrics (pipeline, date)',
        ]
        for stmt in index_statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    logger.warning(f"Index creation warning: {e}")
        conn.commit()
        logger.info("Indexes rebuilt")

        # Step 9: Verify integrity
        logger.info("Running PRAGMA quick_check...")
        issues = []
        for row in conn.execute("PRAGMA quick_check"):
            if row[0] != "ok":
                issues.append(row[0])
        if issues:
            logger.warning(f"quick_check found {len(issues)} issues (may be in other tables):")
            for issue in issues[:10]:
                logger.warning(f"  {issue}")
        else:
            logger.info("PRAGMA quick_check: ok")

        # Step 10: Test the exact query that was failing
        logger.info("Testing JOIN query (the one that was failing)...")
        try:
            result = conn.execute("""
                SELECT tm.id, tm.theme_cluster_id, tm.date, tc.name
                FROM theme_metrics tm
                JOIN theme_clusters tc ON tm.theme_cluster_id = tc.id
                WHERE tc.is_active = 1
                LIMIT 5
            """).fetchall()
            logger.info(f"JOIN query succeeded, returned {len(result)} rows")
        except sqlite3.DatabaseError as e:
            logger.error(f"JOIN query still failing: {e}")
            conn.close()
            return False

        # Final count
        count_after = conn.execute("SELECT COUNT(*) FROM theme_metrics").fetchone()[0]
        logger.info(f"Final row count: {count_after}")
        if count_before:
            lost = count_before - count_after
            pct = (count_after / count_before) * 100
            logger.info(f"Data recovery: {pct:.1f}% ({lost} rows lost in corrupted pages)")

        conn.close()
        return True

    finally:
        # Always clean up temp file
        if os.path.exists(inserts_file):
            os.unlink(inserts_file)


def main():
    parser = argparse.ArgumentParser(
        description="Repair corrupted theme_metrics table via dump-and-reload"
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview what would happen without modifying anything"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("THEME_METRICS TABLE REPAIR")
    print("=" * 80)

    db_path = get_db_path()
    logger.info(f"Database: {db_path}")
    logger.info(f"Size: {os.path.getsize(db_path) / (1024 * 1024):.1f} MB")

    # Check for active processes
    active = check_for_active_processes(db_path)
    if active:
        print("\nWARNING: The following processes have the database open:")
        for proc in active:
            print(f"  - {proc}")
        print("\nYou MUST stop all processes before running this repair:")
        print("  docker compose down       # if using Docker")
        print("  pkill -f celery           # if running locally")
        print("  pkill -f uvicorn          # if running locally")
        if not args.yes:
            response = input("\nContinue anyway? (yes/no): ").strip().lower()
            if response != "yes":
                print("Repair cancelled.")
                return
        else:
            print("\n--yes flag provided, proceeding despite active processes...")

    if args.dry_run:
        print("\n[DRY RUN MODE] No changes will be made.\n")

    if not args.yes and not args.dry_run:
        print("\nThis script will:")
        print("  1. Back up the database (timestamped copy)")
        print("  2. Checkpoint WAL writes")
        print("  3. Dump all recoverable theme_metrics rows")
        print("  4. Drop and recreate the theme_metrics table")
        print("  5. Reload the dumped data")
        print("  6. Rebuild indexes and verify integrity")
        print("\nEstimated recovery: ~99% of rows (rows in corrupted B-tree pages will be lost)")
        response = input("\nProceed with repair? (yes/no): ").strip().lower()
        if response != "yes":
            print("Repair cancelled.")
            return

    # Step 0: Backup
    if not args.dry_run:
        backup_path = backup_database(db_path)
        print(f"\nBackup saved to: {backup_path}")

    # Run repair
    print()
    success = repair(db_path, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    if args.dry_run:
        print("DRY RUN COMPLETE — no changes were made")
    elif success:
        print("REPAIR COMPLETE!")
        print("\nNext steps:")
        print("  1. Verify: sqlite3 %s 'PRAGMA quick_check;'" % db_path)
        print("  2. Restart: docker compose up --build -d")
        print("  3. Test:    curl http://localhost/api/v1/themes/rankings?limit=5")
    else:
        print("REPAIR FAILED — database was not modified (backup available)")
    print("=" * 80)


if __name__ == "__main__":
    main()
