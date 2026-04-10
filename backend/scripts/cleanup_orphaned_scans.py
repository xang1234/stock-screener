"""
One-time cleanup script to remove orphaned scan data.

This script cleans up:
1. All cancelled scans and their results (never cleaned up before)
2. All stale running/queued scans (will never complete)
3. Runs WAL checkpoint to reclaim WAL space (safe, no exclusive lock)
4. Optionally runs VACUUM (--vacuum) after checking for active processes

Usage:
    cd backend
    python scripts/cleanup_orphaned_scans.py          # Interactive mode (WAL checkpoint)
    python scripts/cleanup_orphaned_scans.py --yes     # Skip confirmation
    python scripts/cleanup_orphaned_scans.py --vacuum  # Full VACUUM (checks for active processes)
"""
import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging
from datetime import datetime, timedelta
from app.models.scan_result import Scan, ScanResult
from app.wiring.bootstrap import get_session_factory, initialize_process_runtime_services
from sqlalchemy import func, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_scan_stats(db):
    """Get current scan statistics by status."""
    stats = db.query(
        Scan.status,
        func.count(Scan.scan_id).label('scan_count')
    ).group_by(Scan.status).all()

    result_counts = {}
    for status, scan_count in stats:
        # Count results for scans with this status
        result_count = db.query(func.count(ScanResult.id)).join(
            Scan, ScanResult.scan_id == Scan.scan_id
        ).filter(Scan.status == status).scalar() or 0
        result_counts[status] = {'scans': scan_count, 'results': result_count}

    return result_counts


def check_for_active_processes(db_path: str) -> list:
    """Check for other processes that have the database file open."""
    active = []
    for suffix in ("", "-wal", "-shm"):
        try:
            result = subprocess.run(
                ["lsof", db_path + suffix],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines()[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 2:
                    active.append(f"{parts[0]} (PID {parts[1]}) -> {db_path + suffix}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return active


def main(skip_confirmation: bool = False, do_vacuum: bool = False):
    print("=" * 80)
    print("ORPHANED SCAN DATA CLEANUP")
    print("=" * 80)

    initialize_process_runtime_services()
    db = get_session_factory()()

    try:
        # Show current state
        print("\nCURRENT STATE (Before Cleanup):")
        print("-" * 60)
        stats_before = get_scan_stats(db)
        total_scans_before = 0
        total_results_before = 0
        for status, counts in stats_before.items():
            print(f"  {status:<15}: {counts['scans']:>5} scans, {counts['results']:>8} results")
            total_scans_before += counts['scans']
            total_results_before += counts['results']
        print("-" * 60)
        print(f"  {'TOTAL':<15}: {total_scans_before:>5} scans, {total_results_before:>8} results")

        # Confirm
        print("\n" + "=" * 80)
        print("This script will DELETE:")
        print("  1. All CANCELLED scans and their results")
        print("  2. All STALE RUNNING/QUEUED scans (older than 1 hour) and their results")
        if do_vacuum:
            print("  3. Run VACUUM to reclaim disk space (requires no active DB connections)")
        else:
            print("  3. Run WAL checkpoint to reclaim WAL file space (safe, no lock needed)")
        print("=" * 80)

        if not skip_confirmation:
            response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Cleanup cancelled.")
                return
        else:
            print("\n--yes flag provided, proceeding with cleanup...")

        # 1. Delete cancelled scans
        print("\n[1/4] Deleting cancelled scans...")
        cancelled_scan_ids = [s.scan_id for s in db.query(Scan).filter(Scan.status == "cancelled").all()]
        if cancelled_scan_ids:
            cancelled_results = db.query(ScanResult).filter(
                ScanResult.scan_id.in_(cancelled_scan_ids)
            ).delete(synchronize_session=False)
            cancelled_scans = db.query(Scan).filter(Scan.status == "cancelled").delete(synchronize_session=False)
            print(f"  Deleted {cancelled_scans} cancelled scans, {cancelled_results} results")
        else:
            print("  No cancelled scans found")

        # 2. Delete stale running/queued scans (older than 1 hour)
        print("\n[2/4] Deleting stale running/queued scans...")
        stale_cutoff = datetime.utcnow() - timedelta(hours=1)
        stale_scan_ids = [
            s.scan_id for s in db.query(Scan).filter(
                Scan.status.in_(["running", "queued"]),
                Scan.started_at < stale_cutoff
            ).all()
        ]
        if stale_scan_ids:
            stale_results = db.query(ScanResult).filter(
                ScanResult.scan_id.in_(stale_scan_ids)
            ).delete(synchronize_session=False)
            stale_scans = db.query(Scan).filter(
                Scan.status.in_(["running", "queued"]),
                Scan.started_at < stale_cutoff
            ).delete(synchronize_session=False)
            print(f"  Deleted {stale_scans} stale scans, {stale_results} results")
        else:
            print("  No stale running/queued scans found")

        # Commit deletions
        print("\n[3/4] Committing changes...")
        db.commit()
        print("  Changes committed successfully")

        # Show state after deletion
        print("\nSTATE AFTER DELETION:")
        print("-" * 60)
        stats_after = get_scan_stats(db)
        total_scans_after = 0
        total_results_after = 0
        for status, counts in stats_after.items():
            print(f"  {status:<15}: {counts['scans']:>5} scans, {counts['results']:>8} results")
            total_scans_after += counts['scans']
            total_results_after += counts['results']
        print("-" * 60)
        print(f"  {'TOTAL':<15}: {total_scans_after:>5} scans, {total_results_after:>8} results")
        print(f"\n  Deleted: {total_scans_before - total_scans_after} scans, {total_results_before - total_results_after} results")

        # Close session before space reclamation
        db.close()

        # Get database path
        from app.config import settings
        import sqlite3

        db_path = settings.database_url.replace("sqlite:///", "")
        size_before = os.path.getsize(db_path) / (1024 * 1024)  # MB

        # Show WAL file size
        wal_path = db_path + "-wal"
        wal_size = 0
        if os.path.exists(wal_path):
            wal_size = os.path.getsize(wal_path) / (1024 * 1024)
            print(f"\n  WAL file size: {wal_size:.1f} MB")

        if do_vacuum:
            # Full VACUUM mode: check for active processes first
            print("\n[4/4] Running VACUUM to reclaim disk space...")
            active = check_for_active_processes(db_path)
            if active:
                print("  WARNING: Other processes have the database open:")
                for proc in active:
                    print(f"    - {proc}")
                if not skip_confirmation:
                    response = input("\n  VACUUM with active processes risks corruption. Proceed? (yes/no): ").strip().lower()
                    if response != 'yes':
                        print("  VACUUM skipped. Running WAL checkpoint instead...")
                        conn = sqlite3.connect(db_path)
                        conn.execute("PRAGMA busy_timeout=15000")
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        conn.close()
                        print("  WAL checkpoint complete")
                        return
                else:
                    print("  --yes flag provided, proceeding despite active processes...")

            print(f"  Database size before VACUUM: {size_before:.1f} MB")
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("VACUUM")
            conn.close()

            size_after = os.path.getsize(db_path) / (1024 * 1024)
            print(f"  Database size after VACUUM: {size_after:.1f} MB")
            print(f"  Space reclaimed: {size_before - size_after:.1f} MB")
        else:
            # Default: WAL checkpoint (safe, no exclusive lock needed)
            print("\n[4/4] Running WAL checkpoint to reclaim WAL space...")
            print(f"  Database size: {size_before:.1f} MB")

            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout=15000")
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            conn.close()

            busy, log_pages, checkpointed = result if result else (None, None, None)
            if busy:
                print(f"  WARNING: Checkpoint blocked by active reader (busy={busy})")
            else:
                print(f"  Checkpoint complete: {log_pages} log pages, {checkpointed} checkpointed")

            if os.path.exists(wal_path):
                wal_size_after = os.path.getsize(wal_path) / (1024 * 1024)
                print(f"  WAL file size after checkpoint: {wal_size_after:.1f} MB (was {wal_size:.1f} MB)")

            print("\n  Tip: Use --vacuum for full space reclamation (requires no active DB connections)")

        print("\n" + "=" * 80)
        print("CLEANUP COMPLETE!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up orphaned scan data")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--vacuum", action="store_true",
                        help="Run full VACUUM instead of WAL checkpoint (requires no active DB connections)")
    args = parser.parse_args()
    main(skip_confirmation=args.yes, do_vacuum=args.vacuum)
