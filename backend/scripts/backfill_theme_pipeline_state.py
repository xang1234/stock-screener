#!/usr/bin/env python3
"""One-shot, resumable backfill for content_item_pipeline_state.

Usage:
    cd backend
    source venv/bin/activate
    python scripts/backfill_theme_pipeline_state.py --yes
    python scripts/backfill_theme_pipeline_state.py --no-resume --yes
    python scripts/backfill_theme_pipeline_state.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal
from app.services.theme_pipeline_state_backfill_service import (  # noqa: E402
    ThemePipelineStateBackfillService,
)


def _default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "theme_pipeline_state_backfill_checkpoint.json"


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill content_item_pipeline_state with checkpointing.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Rows per chunk (default: 500)")
    parser.add_argument("--max-chunks", type=int, default=0, help="Stop after N chunks (0 = no limit)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from cursor 0 regardless of any existing checkpoint",
    )
    parser.add_argument("--reset-checkpoint", action="store_true", help="Ignore and overwrite checkpoint")
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=str(_default_checkpoint_path()),
        help="Checkpoint file path",
    )
    parser.add_argument("--dry-run", action="store_true", help="No writes, no checkpoint persistence")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip interactive confirmation")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint_file).expanduser()
    resume = not args.no_resume

    checkpoint = {}
    if resume and not args.reset_checkpoint:
        checkpoint = _load_checkpoint(checkpoint_path)

    start_cursor = int(checkpoint.get("cursor_last_content_item_id", 0))
    totals = {
        "rows_read": int(checkpoint.get("totals", {}).get("rows_read", 0)) if (resume and not args.reset_checkpoint) else 0,
        "rows_written": int(checkpoint.get("totals", {}).get("rows_written", 0)) if (resume and not args.reset_checkpoint) else 0,
        "conflicts": int(checkpoint.get("totals", {}).get("conflicts", 0)) if (resume and not args.reset_checkpoint) else 0,
        "chunks_completed": int(checkpoint.get("totals", {}).get("chunks_completed", 0)) if (resume and not args.reset_checkpoint) else 0,
    }

    if not args.yes and not args.dry_run:
        answer = input(
            f"Backfill pipeline-state rows from content_item_id > {start_cursor}. Continue? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    db = SessionLocal()
    service = ThemePipelineStateBackfillService(db)

    cursor = start_cursor
    chunk_index = 0
    try:
        while True:
            chunk = service.process_chunk(
                after_content_item_id=cursor,
                limit=args.chunk_size,
                dry_run=args.dry_run,
            )

            if chunk.rows_read == 0:
                print("No more rows to process.")
                break

            chunk_index += 1
            cursor = chunk.next_cursor
            totals["rows_read"] += chunk.rows_read
            totals["rows_written"] += chunk.rows_written
            totals["conflicts"] += chunk.conflicts
            totals["chunks_completed"] += 1

            print(
                f"Chunk {chunk_index}: cursor={cursor} "
                f"read={chunk.rows_read} written={chunk.rows_written} conflicts={chunk.conflicts}"
            )

            if not args.dry_run:
                _save_checkpoint(
                    checkpoint_path,
                    {
                        "cursor_last_content_item_id": cursor,
                        "last_chunk": {
                            "rows_read": chunk.rows_read,
                            "rows_written": chunk.rows_written,
                            "conflicts": chunk.conflicts,
                            "writes_by_pipeline_status": chunk.writes_by_pipeline_status,
                            "processed_at": datetime.utcnow().isoformat(),
                        },
                        "totals": totals,
                    },
                )

            if args.max_chunks and chunk_index >= args.max_chunks:
                print(f"Stopping early at --max-chunks={args.max_chunks}")
                break

        print("\nBackfill summary")
        print(f"  cursor_last_content_item_id: {cursor}")
        print(f"  rows_read: {totals['rows_read']}")
        print(f"  rows_written: {totals['rows_written']}")
        print(f"  conflicts: {totals['conflicts']}")
        print(f"  chunks_completed: {totals['chunks_completed']}")

        if not args.dry_run:
            report = service.summary_counts()
            print("  by_pipeline_status:")
            for pipeline, counts in sorted(report["by_pipeline_status"].items()):
                print(f"    {pipeline}: {counts}")
    except Exception as exc:
        db.rollback()
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
