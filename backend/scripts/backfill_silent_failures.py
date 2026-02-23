#!/usr/bin/env python3
"""One-time script to identify and reset content items that silently failed LLM extraction.

Before the fix in extract_from_content(), LLM errors (rate limits, timeouts) were caught
and returned as [] — making failed items appear "successfully processed with 0 themes."
These items have is_processed=True, extraction_error=NULL, but NO theme_mentions.

This script finds those items and resets them to is_processed=False so the next
pipeline run will re-extract them.

Usage:
    cd backend
    source venv/bin/activate
    python scripts/backfill_silent_failures.py           # Execute reset
    python scripts/backfill_silent_failures.py --dry-run  # Preview only
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal
from app.services.theme_extraction_service import ThemeExtractionService


def main():
    parser = argparse.ArgumentParser(description="Identify and reset silently failed extractions")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying data")
    parser.add_argument("--max-age-days", type=int, default=30, help="Max age of items to check (default: 30)")
    args = parser.parse_args()

    db = SessionLocal()

    try:
        pipelines = ["technical", "fundamental"]
        total_reset = 0

        for pipeline in pipelines:
            print(f"\n{'='*50}")
            print(f"Pipeline: {pipeline}")
            print(f"{'='*50}")

            service = ThemeExtractionService(db, pipeline=pipeline)

            if args.dry_run:
                # For dry run, query without resetting
                from datetime import datetime, timedelta
                from sqlalchemy import and_
                from app.models.theme import ContentItem, ContentItemPipelineState, ThemeMention

                cutoff = datetime.utcnow() - timedelta(days=args.max_age_days)
                pipeline_source_ids = service._get_pipeline_source_ids()

                mentioned_ids = db.query(ThemeMention.content_item_id).filter(
                    ThemeMention.pipeline == pipeline
                ).distinct().subquery()

                query = db.query(ContentItem).join(
                    ContentItemPipelineState,
                    and_(
                        ContentItemPipelineState.content_item_id == ContentItem.id,
                        ContentItemPipelineState.pipeline == pipeline,
                    ),
                ).filter(
                    ContentItemPipelineState.status == "processed",
                    ContentItem.published_at >= cutoff,
                    ~ContentItem.id.in_(db.query(mentioned_ids.c.content_item_id)),
                )
                if pipeline_source_ids:
                    query = query.filter(ContentItem.source_id.in_(pipeline_source_ids))

                items = query.all()
                print(f"  Would reset {len(items)} silently failed items")
                for item in items[:10]:
                    print(f"    - ID {item.id}: {item.title[:60] if item.title else 'No title'}...")
                if len(items) > 10:
                    print(f"    ... and {len(items) - 10} more")
                total_reset += len(items)
            else:
                result = service.identify_silent_failures(max_age_days=args.max_age_days)
                print(f"  Reset {result['reset_count']} silently failed items")
                total_reset += result['reset_count']

        print(f"\n{'='*50}")
        if args.dry_run:
            print(f"DRY RUN: Would reset {total_reset} total items across all pipelines")
            print("Run without --dry-run to execute.")
        else:
            print(f"Reset {total_reset} total items. They will be re-extracted on the next pipeline run.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
