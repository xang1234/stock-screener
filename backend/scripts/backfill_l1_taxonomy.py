#!/usr/bin/env python3
"""
One-time backfill script to build L1 taxonomy from existing themes.

Usage:
    cd backend && source venv/bin/activate
    python scripts/backfill_l1_taxonomy.py --pipeline technical --dry-run
    python scripts/backfill_l1_taxonomy.py --pipeline technical --output report.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal, engine
from app.db_migrations.theme_taxonomy_migration import migrate_theme_taxonomy
from app.services.theme_taxonomy_service import ThemeTaxonomyService


def main():
    parser = argparse.ArgumentParser(description="Backfill L1 taxonomy from existing themes")
    parser.add_argument("--pipeline", default="technical", choices=["technical", "fundamental"])
    parser.add_argument("--dry-run", action="store_true", help="Preview assignments without applying")
    parser.add_argument("--output", type=str, help="Write report to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print(f"L1 Taxonomy Backfill {'(DRY RUN)' if args.dry_run else '(LIVE)'}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Ensure taxonomy columns exist (migration normally runs on app startup,
    # but this script bypasses FastAPI lifespan)
    print("\n[Pre-flight] Ensuring taxonomy schema...")
    migration_result = migrate_theme_taxonomy(engine)
    if migration_result["columns_added"]:
        print(f"  Added columns: {migration_result['columns_added']}")
    else:
        print("  Schema already up to date.")

    db = SessionLocal()
    try:
        service = ThemeTaxonomyService(db, pipeline=args.pipeline)

        # Phase 1-3: Full assignment pipeline
        print("\n[Phase 1] Rule-based prefix matching...")
        print("[Phase 2] HDBSCAN embedding clustering...")
        print("[Phase 3] LLM naming (with non-LLM fallback)...")

        report = service.run_full_taxonomy_assignment(dry_run=args.dry_run)

        # Print summary
        print("\n" + "=" * 60)
        print("ASSIGNMENT REPORT")
        print("=" * 60)

        p1 = report.get("phase1_rule_based", {})
        p2 = report.get("phase2_clustering", {})
        p3 = report.get("phase3_llm_naming", {})

        print(f"\nTotal unassigned themes: {report['total_unassigned']}")
        print(f"Phase 1 (rules): {p1.get('assigned', 0)} assigned")
        print(f"Phase 2 (clustering): {p2.get('num_clusters', 0)} clusters, {p2.get('noise_themes', 0)} noise")
        print(f"Phase 3 (LLM naming): {len(p3.get('l1_themes', []))} L1 themes created")
        print(f"\nTotal assigned: {report['l2_themes_assigned']}")
        print(f"Still unassigned: {report['still_unassigned']}")

        if not args.dry_run:
            print(f"L1 themes in database: {report.get('l1_themes_created', 0)}")

        # Print phase details
        if p1.get("assignments"):
            print(f"\n--- Phase 1: Rule-based assignments ({len(p1['assignments'])}) ---")
            for a in p1["assignments"][:20]:
                print(f"  {a['l2_name']:40s} → {a['l1_name']}")
            if len(p1["assignments"]) > 20:
                print(f"  ... and {len(p1['assignments']) - 20} more")

        if p2.get("clusters"):
            print(f"\n--- Phase 2: HDBSCAN clusters ({len(p2['clusters'])}) ---")
            for i, cluster in enumerate(p2["clusters"][:15]):
                members_str = ", ".join(cluster["members"][:3])
                if len(cluster["members"]) > 3:
                    members_str += f" (+{len(cluster['members']) - 3} more)"
                print(f"  Cluster {i + 1} ({cluster['size']} members): {members_str}")

        if p3.get("l1_themes"):
            print(f"\n--- Phase 3: L1 themes ({len(p3['l1_themes'])}) ---")
            for l1 in p3["l1_themes"]:
                members_str = ", ".join(l1["members"][:3])
                if len(l1["members"]) > 3:
                    members_str += f" (+{len(l1['members']) - 3} more)"
                print(f"  {l1['l1_name']:30s} [{l1['category']}] → {members_str}")

        if not args.dry_run:
            # Post-assignment steps
            print("\n[Phase 4] Computing L1 centroid embeddings...")
            centroid_result = service.compute_l1_centroid_embeddings()
            db.commit()
            print(f"  Updated: {centroid_result['updated']}, Skipped: {centroid_result['skipped']}")

            print("\n[Phase 5] Computing initial L1 metrics...")
            metrics_result = service.compute_all_l1_metrics()
            db.commit()
            print(f"  L1 count: {metrics_result['l1_count']}, Metrics updated: {metrics_result['metrics_updated']}")

        # Write report to file if requested
        if args.output:
            # Serialize report (strip non-serializable objects)
            serializable_report = _make_serializable(report)
            serializable_report["timestamp"] = datetime.now().isoformat()
            serializable_report["pipeline"] = args.pipeline

            with open(args.output, "w") as f:
                json.dump(serializable_report, f, indent=2)
            print(f"\nReport written to: {args.output}")

        print("\n" + "=" * 60)
        if args.dry_run:
            print("DRY RUN complete. No changes made. Run without --dry-run to apply.")
        else:
            print("BACKFILL COMPLETE.")
        print("=" * 60)

    except Exception as e:
        db.rollback()
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
    finally:
        db.close()


def _make_serializable(obj):
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return str(obj)
    return obj


if __name__ == "__main__":
    main()
