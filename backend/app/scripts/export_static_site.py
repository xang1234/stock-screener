"""CLI for building the read-only static-site data bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.config import settings
from app.database import SessionLocal
from app.infra.db.portability import is_sqlite
from app.main import initialize_runtime
from app.services.static_site_export_service import StaticSiteExportService
from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_dir() -> Path:
    return _repo_root() / "frontend" / "public" / "static-data"


def _ensure_database_path_ready() -> None:
    if not is_sqlite(settings.database_url):
        return

    database_path = Path(settings.database_url.removeprefix("sqlite:///"))
    database_path.parent.mkdir(parents=True, exist_ok=True)


def _run_daily_refresh(*, refresh_themes_best_effort: bool) -> tuple[dict[str, Any], list[str]]:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import calculate_daily_breadth_with_gapfill
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.group_rank_tasks import calculate_daily_group_rankings
    from app.tasks.universe_tasks import refresh_stock_universe

    warnings: list[str] = []
    with disable_serialized_data_fetch_lock():
        results: dict[str, Any] = {
            "universe_refresh": refresh_stock_universe.run(),
            "cache_refresh": smart_refresh_cache.run(mode="full"),
            "fundamentals_refresh": refresh_all_fundamentals.run(),
            "breadth_refresh": calculate_daily_breadth_with_gapfill.run(),
            "groups_refresh": calculate_daily_group_rankings.run(),
            "feature_snapshot": build_daily_snapshot.run(),
        }

        if refresh_themes_best_effort:
            from app.tasks.theme_discovery_tasks import run_full_pipeline

            for pipeline in ("technical", "fundamental"):
                try:
                    results[f"themes_{pipeline}"] = run_full_pipeline.run(pipeline=pipeline)
                except Exception as exc:  # noqa: BLE001 - explicit best effort behavior
                    warnings.append(f"Theme refresh failed for {pipeline}: {exc}")

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
        "--refresh-themes-best-effort",
        action="store_true",
        help="Attempt the theme pipelines during refresh and continue if they fail.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the output directory before exporting.",
    )
    args = parser.parse_args()

    _ensure_database_path_ready()
    initialize_runtime()

    refresh_warnings: list[str] = []
    if args.refresh_daily:
        refresh_results, refresh_warnings = _run_daily_refresh(
            refresh_themes_best_effort=args.refresh_themes_best_effort,
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
