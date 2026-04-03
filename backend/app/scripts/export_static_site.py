"""CLI for building the read-only static-site data bundle."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.static_site_export_service import StaticSiteExportService
from app.services.provider_snapshot_service import provider_snapshot_service
from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock
from app.utils.market_hours import get_last_market_close


def _default_output_dir() -> Path:
    return repo_root() / "frontend" / "public" / "static-data"


def _resolve_latest_completed_us_trading_date() -> date:
    """Return the latest completed NYSE session date for static exports."""
    return get_last_market_close().date()


def _run_daily_refresh(
    *,
    skip_universe_refresh: bool = False,
    skip_fundamentals_refresh: bool = False,
    hydrate_published_snapshot: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import calculate_daily_breadth_with_gapfill
    from app.tasks.cache_tasks import (
        allow_smart_refresh_time_window_bypass,
        smart_refresh_cache,
    )
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.group_rank_tasks import (
        allow_same_day_group_rank_warmup_bypass,
        calculate_daily_group_rankings,
    )
    from app.tasks.universe_tasks import refresh_stock_universe

    warnings: list[str] = []
    as_of_date = _resolve_latest_completed_us_trading_date()
    with disable_serialized_data_fetch_lock():
        results: dict[str, Any] = {}
        if not skip_universe_refresh:
            results["universe_refresh"] = refresh_stock_universe.run()

        with allow_smart_refresh_time_window_bypass():
            results["cache_refresh"] = smart_refresh_cache.run(mode="full")

        if not skip_fundamentals_refresh:
            results["fundamentals_refresh"] = refresh_all_fundamentals.run()

        if hydrate_published_snapshot:
            with SessionLocal() as db:
                results["fundamentals_hydrate"] = provider_snapshot_service.hydrate_published_snapshot(
                    db,
                    allow_yahoo_hydration=False,
                )

        results["breadth_refresh"] = calculate_daily_breadth_with_gapfill.run()
        with allow_same_day_group_rank_warmup_bypass():
            results["groups_refresh"] = calculate_daily_group_rankings.run()
        results["feature_snapshot"] = build_daily_snapshot.run(
            as_of_date_str=as_of_date.isoformat(),
            static_daily_mode=True,
        )

    return results, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory to receive the generated static JSON bundle.",
    )
    parser.add_argument(
        "--refresh-daily",
        action="store_true",
        help="Run the synchronous daily refresh/build steps before exporting.",
    )
    parser.add_argument(
        "--skip-universe-refresh",
        action="store_true",
        help="Do not refresh the live stock universe before exporting.",
    )
    parser.add_argument(
        "--skip-fundamentals-refresh",
        action="store_true",
        help="Do not run the live weekly fundamentals refresh before exporting.",
    )
    parser.add_argument(
        "--hydrate-published-snapshot",
        action="store_true",
        help="Hydrate stock_fundamentals from the currently imported published snapshot.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the output directory before exporting.",
    )
    args = parser.parse_args()

    prepare_runtime()

    refresh_warnings: list[str] = []
    if args.refresh_daily:
        refresh_results, refresh_warnings = _run_daily_refresh(
            skip_universe_refresh=args.skip_universe_refresh,
            skip_fundamentals_refresh=args.skip_fundamentals_refresh,
            hydrate_published_snapshot=args.hydrate_published_snapshot,
        )
        print("Daily refresh complete:")
        for name, result in refresh_results.items():
            print(f"  - {name}: {result}")
        for warning in refresh_warnings:
            print(f"  - warning: {warning}")

    service = StaticSiteExportService(SessionLocal)
    result = service.export(Path(args.output_dir), clean=not args.no_clean)

    print("Static site export complete:")
    print(f"  - output_dir: {result.output_dir}")
    print(f"  - generated_at: {result.generated_at}")
    print(f"  - as_of_date: {result.as_of_date}")
    for warning in (*refresh_warnings, *result.warnings):
        print(f"  - warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
