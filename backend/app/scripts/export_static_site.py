"""CLI for building the read-only static-site data bundle."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from app.config import settings
from app.database import SessionLocal
from app.infra.db.models.feature_store import FeatureRunPointer
from app.models.industry import IBDGroupRank
from app.models.market_breadth import MarketBreadth
from app.domain.markets import market_registry
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.ibd_industry_service import IBDIndustryService
from app.services.static_daily_price_refresh_service import (
    StaticDailyPriceRefreshService,
    static_daily_price_refresh_batch_size as _static_daily_price_refresh_batch_size,
)
from app.services.static_site_export_service import (
    NoPublishedStaticMarketArtifact,
    StaticSiteExportService,
)
from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock
from app.tasks.workload_coordination import disable_serialized_market_workload
from app.wiring.bootstrap import (
    get_group_rank_service,
    get_market_calendar_service,
    get_price_cache,
    get_provider_snapshot_service,
)


STATIC_GROUP_HISTORY_LOOKBACK_DAYS = 100
STATIC_BREADTH_HISTORY_MIN_TRADING_DAYS = 20
STATIC_BREADTH_HISTORY_LOOKBACK_DAYS = 90
STATIC_BUILD_MODE_PRICE_DELTA = "price_delta"
STATIC_BUILD_MODE_FULL = "full"
STATIC_EXPORT_MARKETS = market_registry.supported_market_codes()
STATIC_DEFAULT_MARKET = "US"
STATIC_EXPORT_SKIPPED_EXIT_CODE = 78
STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE = 79


def _default_output_dir() -> Path:
    return repo_root() / "frontend" / "public" / "static-data"


def _tracked_ibd_csv_path() -> Path:
    return IBDIndustryService.resolve_tracked_csv_path(settings.ibd_industry_csv_path)


def _resolve_latest_completed_trading_date(market: str) -> date:
    """Return the latest completed trading session date for ``market``.

    Each market has its own calendar (NYSE, HKEX, NSE, …). Using the NYSE
    date for non-US markets either falsely skips them on days NYSE was
    closed but the target exchange traded, or builds their snapshot for a
    stale date when NYSE traded but the target exchange did not.
    """
    return get_market_calendar_service().last_completed_trading_day(market)


def _market_pointer_key(market: str) -> str:
    return f"latest_published_market:{market.upper()}"


def _upsert_feature_run_pointer(*, pointer_key: str, run_id: int) -> None:
    with SessionLocal() as db:
        if not hasattr(db, "query"):
            return
        pointer = (
            db.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == pointer_key)
            .first()
        )
        if pointer is None:
            db.add(FeatureRunPointer(key=pointer_key, run_id=run_id))
        else:
            pointer.run_id = run_id
        db.commit()


def _snapshot_publishable(snapshot: dict[str, Any]) -> bool:
    status = snapshot.get("status")
    if status == "published":
        return True
    if status == "skipped" and snapshot.get("reason") == "already_published":
        return True
    if status is None and (
        snapshot.get("run_id") is not None
        or snapshot.get("existing_run_id") is not None
    ):
        return True
    return False


def _selected_market_non_publishable_snapshot(
    refresh_results: dict[str, Any],
    market: str | None,
) -> dict[str, Any] | None:
    if market is None:
        return None
    feature_snapshots = refresh_results.get("feature_snapshots", {})
    if not isinstance(feature_snapshots, dict):
        return None
    snapshot = feature_snapshots.get(market.upper())
    if not isinstance(snapshot, dict):
        return None
    return None if _snapshot_publishable(snapshot) else snapshot


def _snapshot_skipped_not_trading_day(snapshot: dict[str, Any] | None) -> bool:
    if snapshot is None:
        return False
    return snapshot.get("status") == "skipped" and snapshot.get("reason") == "not_trading_day"


def _write_market_diagnostics(output_dir: Path, market: str, snapshot: dict[str, Any]) -> Path:
    diagnostics_dir = output_dir / "diagnostics" / market.lower()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = diagnostics_dir / "snapshot-failure.json"
    payload: dict[str, Any] = {
        "market": market.upper(),
        "status": snapshot.get("status"),
        "failed_symbols": snapshot.get("failed_symbols", []),
        "row_count": snapshot.get("row_count"),
        "warnings": snapshot.get("warnings", []),
        "failure_diagnostics": snapshot.get("failure_diagnostics", {}),
    }
    for key in ("reason", "run_id", "existing_run_id"):
        if key in snapshot:
            payload[key] = snapshot[key]
    diagnostics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return diagnostics_path


def _refresh_static_daily_prices(*, as_of_date: date, market: str | None = None) -> dict[str, Any]:
    service = StaticDailyPriceRefreshService(
        session_factory=SessionLocal,
        price_cache=get_price_cache(),
        fetcher=BulkDataFetcher(),
        batch_size_for_market=_static_daily_price_refresh_batch_size,
    )
    return service.refresh(as_of_date=as_of_date, market=market)


def _generate_trading_dates(
    start_date: date,
    end_date: date,
    *,
    market: str = STATIC_DEFAULT_MARKET,
) -> list[date]:
    normalized_market = (market or STATIC_DEFAULT_MARKET).upper()
    calendar_service = get_market_calendar_service()
    trading_dates: list[date] = []
    current = start_date
    while current <= end_date:
        if calendar_service.is_trading_day(normalized_market, current):
            trading_dates.append(current)
        current += timedelta(days=1)
    return trading_dates


def _ensure_group_rank_history(*, as_of_date: date, market: str = "US") -> dict[str, Any]:
    """Backfill recent group-rank history so 1W/1M/3M deltas can be rendered."""
    normalized_market = (market or "US").upper()
    start_date = as_of_date - timedelta(days=STATIC_GROUP_HISTORY_LOOKBACK_DAYS)
    desired_dates = _generate_trading_dates(
        start_date,
        as_of_date,
        market=normalized_market,
    )

    with SessionLocal() as db:
        if not hasattr(db, "query"):
            return {
                "status": "skipped",
                "market": normalized_market,
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "missing_dates": 0,
                "processed": 0,
                "errors": 0,
                "reason": "session_factory_stub",
            }
        existing_dates = {
            record_date
            for record_date, in db.query(IBDGroupRank.date)
            .filter(
                IBDGroupRank.date >= start_date,
                IBDGroupRank.date <= as_of_date,
                IBDGroupRank.market == normalized_market,
            )
            .distinct()
            .all()
        }
        missing_dates = [calc_date for calc_date in desired_dates if calc_date not in existing_dates]
        if not missing_dates:
            print(
                f"[static-groups:{normalized_market}] Existing rankings already cover "
                f"{len(desired_dates):,} trading dates through {as_of_date}.",
                flush=True,
            )
            return {
                "status": "skipped",
                "market": normalized_market,
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "missing_dates": 0,
                "processed": 0,
                "errors": 0,
            }

        print(
            f"[static-groups:{normalized_market}] Backfilling {len(missing_dates):,} missing trading dates "
            f"from {missing_dates[0]} to {missing_dates[-1]} before publishing {as_of_date}.",
            flush=True,
        )
        try:
            stats = get_group_rank_service().fill_gaps_optimized(
                db, missing_dates, market=normalized_market,
            )
        except Exception as exc:
            print(
                f"[static-groups:{normalized_market}] Group-rank backfill failed: {exc}",
                flush=True,
            )
            return {
                "status": "errored",
                "market": normalized_market,
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "missing_dates": len(missing_dates),
                "processed": 0,
                "errors": len(missing_dates),
                "error": str(exc),
            }
        stats.update(
            {
                "status": "completed",
                "market": normalized_market,
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "missing_dates": len(missing_dates),
            }
        )
        return stats


def _ensure_breadth_history(
    *,
    as_of_date: date,
    min_trading_days: int = STATIC_BREADTH_HISTORY_MIN_TRADING_DAYS,
) -> dict[str, Any]:
    """Backfill recent breadth history so static snapshots include multi-day context."""
    start_date = as_of_date - timedelta(days=STATIC_BREADTH_HISTORY_LOOKBACK_DAYS)
    desired_dates = _generate_trading_dates(start_date, as_of_date)
    target_dates = desired_dates[-min_trading_days:] if min_trading_days > 0 else desired_dates
    if not target_dates:
        return {
            "status": "skipped",
            "as_of_date": as_of_date.isoformat(),
            "lookback_start_date": start_date.isoformat(),
            "target_trading_days": 0,
            "recomputed_dates": 0,
        }

    with SessionLocal() as db:
        existing_dates = {
            record_date
            for record_date, in db.query(MarketBreadth.date)
            .filter(
                MarketBreadth.date >= target_dates[0],
                MarketBreadth.date <= as_of_date,
                MarketBreadth.market == "US",
            )
            .all()
        }
        missing_dates = [calc_date for calc_date in target_dates if calc_date not in existing_dates]
        recompute_dates = sorted(set(missing_dates + [as_of_date]))

        if len(recompute_dates) == 1 and as_of_date in existing_dates:
            print(
                f"[static-breadth] Existing breadth history already covers the last "
                f"{len(target_dates)} trading days through {as_of_date}.",
                flush=True,
            )
            return {
                "status": "skipped",
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "target_trading_days": len(target_dates),
                "missing_dates": 0,
                "recomputed_dates": 0,
            }

        print(
            f"[static-breadth] Recomputing {len(recompute_dates)} dates "
            f"({len(missing_dates)} missing) to ensure {len(target_dates)} trading-day history "
            f"through {as_of_date}.",
            flush=True,
        )
        stats = BreadthCalculatorService(db, get_price_cache()).backfill_range(
            start_date=recompute_dates[0],
            end_date=recompute_dates[-1],
            trading_dates=recompute_dates,
            cache_only=True,
        )
        stats.update(
            {
                "status": "completed",
                "as_of_date": as_of_date.isoformat(),
                "lookback_start_date": start_date.isoformat(),
                "target_trading_days": len(target_dates),
                "missing_dates": len(missing_dates),
                "recomputed_dates": len(recompute_dates),
            }
        )
        return stats


def _run_daily_refresh(
    *,
    market: str | None = None,
    skip_universe_refresh: bool = False,
    skip_fundamentals_refresh: bool = False,
    build_mode: Literal["price_delta", "full"] = STATIC_BUILD_MODE_PRICE_DELTA,
    hydrate_published_snapshot: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    from app.interfaces.tasks.feature_store_tasks import (
        _enrich_feature_run_with_ibd_metadata,
        build_daily_snapshot,
    )
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.universe_tasks import refresh_stock_universe

    warnings: list[str] = []

    selected_markets = (market,) if market is not None else STATIC_EXPORT_MARKETS
    as_of_by_market: dict[str, date] = {
        selected_market: _resolve_latest_completed_trading_date(selected_market)
        for selected_market in selected_markets
    }

    with disable_serialized_data_fetch_lock(), disable_serialized_market_workload():
        results: dict[str, Any] = {}
        if not skip_universe_refresh:
            universe_kwargs = {"market": market} if market is not None else {}
            results["universe_refresh"] = refresh_stock_universe.run(**universe_kwargs)

        if not skip_fundamentals_refresh:
            fundamentals_kwargs = {"market": market} if market is not None else {}
            results["fundamentals_refresh"] = refresh_all_fundamentals.run(**fundamentals_kwargs)

        if build_mode == STATIC_BUILD_MODE_FULL and hydrate_published_snapshot:
            provider_snapshot_service = get_provider_snapshot_service()
            with SessionLocal() as db:
                results["fundamentals_hydrate"] = provider_snapshot_service.hydrate_all_published_snapshots(
                    db,
                    allow_yahoo_hydration=False,
                )

        with SessionLocal() as db:
            results["ibd_seed_refresh"] = {
                "csv_path": str(_tracked_ibd_csv_path()),
                "loaded": IBDIndustryService.load_from_csv(db, csv_path=_tracked_ibd_csv_path()),
            }

        # Price refresh is per-market so each market's staleness check uses
        # its own calendar's latest session — avoids treating an HK-traded
        # day as stale because NYSE was closed (or vice-versa).
        price_refresh_results: dict[str, Any] = {}
        for selected_market in selected_markets:
            price_refresh_results[selected_market] = _refresh_static_daily_prices(
                as_of_date=as_of_by_market[selected_market],
                market=selected_market,
            )
        results["price_refresh"] = (
            price_refresh_results[selected_markets[0]]
            if market is not None
            else price_refresh_results
        )

        feature_snapshots: dict[str, Any] = {}
        for selected_market in selected_markets:
            market_as_of = as_of_by_market[selected_market]
            market_result = build_daily_snapshot.run(
                as_of_date_str=market_as_of.isoformat(),
                static_daily_mode=True,
                universe_name=f"market:{selected_market.lower()}",
                market=selected_market,
                publish_pointer_key=_market_pointer_key(selected_market),
                ignore_runtime_market_gate=True,
            )
            feature_snapshots[selected_market] = market_result

        results["feature_snapshots"] = feature_snapshots

        group_rank_history: dict[str, Any] = {}
        for selected_market in selected_markets:
            snapshot = feature_snapshots.get(selected_market, {})
            if not _snapshot_publishable(snapshot):
                group_rank_history[selected_market] = {
                    "status": "skipped",
                    "market": selected_market,
                    "reason": "snapshot_not_ready",
                }
                continue
            group_rank_history[selected_market] = _ensure_group_rank_history(
                as_of_date=as_of_by_market[selected_market],
                market=selected_market,
            )
        results["group_rank_history_backfill"] = group_rank_history

        # Re-enrich feature runs after the IBDGroupRank backfill above.
        # build_daily_snapshot's inner enrichment runs *before* group ranks
        # for `as_of_date` are populated, so US rows would otherwise carry
        # `details_json["ibd_group_rank"] = None`.
        #
        # Only re-enrich when the backfill above actually succeeded
        # (status "completed" — fresh ranks were written — or "skipped" —
        # existing rows already cover ``as_of_date``). For any other
        # status (e.g. "errored") the IBDGroupRank table is still missing
        # rows, so calling the enricher would overwrite previously valid
        # ``ibd_group_rank`` values with ``None`` — particularly harmful
        # when ``build_daily_snapshot`` returned "already_published" and
        # the existing run carries good ranks from an earlier successful
        # refresh.
        ibd_metadata_refresh: dict[str, Any] = {}
        for selected_market in selected_markets:
            snapshot = feature_snapshots.get(selected_market, {})
            if not _snapshot_publishable(snapshot):
                ibd_metadata_refresh[selected_market] = {
                    "status": "skipped",
                    "market": selected_market,
                    "reason": "snapshot_not_ready",
                }
                continue
            backfill_status = (group_rank_history.get(selected_market) or {}).get("status")
            if backfill_status not in ("completed", "skipped"):
                ibd_metadata_refresh[selected_market] = {
                    "status": "skipped",
                    "market": selected_market,
                    "reason": f"group_rank_backfill_{backfill_status or 'missing'}",
                }
                continue
            feature_run_id = (
                snapshot.get("run_id") or snapshot.get("existing_run_id")
            )
            if feature_run_id is None:
                ibd_metadata_refresh[selected_market] = {
                    "status": "skipped",
                    "market": selected_market,
                    "reason": "no_run_id",
                }
                continue
            ibd_metadata_refresh[selected_market] = _enrich_feature_run_with_ibd_metadata(
                feature_run_id=feature_run_id,
                ranking_date=as_of_by_market[selected_market],
            )
        results["ibd_metadata_refresh"] = ibd_metadata_refresh

        for snapshot_market, snapshot in feature_snapshots.items():
            if snapshot_market == STATIC_DEFAULT_MARKET:
                continue
            if _snapshot_publishable(snapshot):
                continue
            status = snapshot.get("status")
            reason = snapshot.get("reason")
            message = f"Static export market {snapshot_market} snapshot returned status {status!r}"
            if reason:
                message += f" ({reason})."
            else:
                message += "."
            warnings.append(message)

        if STATIC_DEFAULT_MARKET in feature_snapshots:
            default_snapshot = feature_snapshots.get(STATIC_DEFAULT_MARKET, {})
            default_snapshot_status = default_snapshot.get("status")
            default_snapshot_ready = _snapshot_publishable(default_snapshot)
            default_run_id = (
                default_snapshot.get("run_id")
                or default_snapshot.get("existing_run_id")
            )
            if default_snapshot_ready and default_run_id is not None:
                _upsert_feature_run_pointer(
                    pointer_key="latest_published",
                    run_id=default_run_id,
                )
                results["default_market_pointer"] = {
                    "market": STATIC_DEFAULT_MARKET,
                    "pointer_key": "latest_published",
                    "run_id": default_run_id,
                }
            elif default_run_id is not None:
                warnings.append(
                    f"{STATIC_DEFAULT_MARKET} feature snapshot returned status "
                    f"{default_snapshot_status!r}; 'latest_published' was not updated."
                )
            else:
                warnings.append(
                    f"No {STATIC_DEFAULT_MARKET} feature snapshot produced a run id; "
                    "'latest_published' was not updated."
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
        "--market",
        choices=STATIC_EXPORT_MARKETS,
        help="Limit refresh/build/export to one market.",
    )
    parser.add_argument(
        "--combine-artifacts-dir",
        help="Combine previously exported market artifacts from this directory into one static-data bundle.",
    )
    parser.add_argument(
        "--fallback-artifacts-dir",
        help="Optional previous-run market artifacts directory used to fill markets missing from --combine-artifacts-dir.",
    )
    parser.add_argument(
        "--build-mode",
        choices=(STATIC_BUILD_MODE_PRICE_DELTA, STATIC_BUILD_MODE_FULL),
        default=STATIC_BUILD_MODE_PRICE_DELTA,
        help="Refresh mode to use before static export. price_delta is the optimized default.",
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

    if args.combine_artifacts_dir and args.refresh_daily:
        raise SystemExit("--combine-artifacts-dir cannot be used together with --refresh-daily")
    if args.combine_artifacts_dir and args.market:
        raise SystemExit("--combine-artifacts-dir cannot be used together with --market")
    if args.fallback_artifacts_dir and not args.combine_artifacts_dir:
        raise SystemExit("--fallback-artifacts-dir requires --combine-artifacts-dir")

    refresh_warnings: list[str] = []
    selected_market_non_publishable_snapshot: dict[str, Any] | None = None
    if args.combine_artifacts_dir:
        result = StaticSiteExportService.combine_market_artifacts(
            Path(args.combine_artifacts_dir),
            Path(args.output_dir),
            fallback_artifacts_dir=(
                Path(args.fallback_artifacts_dir)
                if args.fallback_artifacts_dir
                else None
            ),
            clean=not args.no_clean,
        )
    else:
        prepare_runtime()

        if args.refresh_daily:
            refresh_results, refresh_warnings = _run_daily_refresh(
                market=args.market,
                skip_universe_refresh=args.skip_universe_refresh,
                skip_fundamentals_refresh=args.skip_fundamentals_refresh,
                build_mode=args.build_mode,
                hydrate_published_snapshot=args.hydrate_published_snapshot,
            )
            print("Daily refresh complete:")
            for name, result_item in refresh_results.items():
                print(f"  - {name}: {result_item}")
            for warning in refresh_warnings:
                print(f"  - warning: {warning}")

            selected_market_non_publishable_snapshot = _selected_market_non_publishable_snapshot(
                refresh_results,
                args.market,
            )
            if _snapshot_skipped_not_trading_day(selected_market_non_publishable_snapshot):
                print(
                    f"Static site export skipped for market {args.market} because it is not a trading day."
                )
                return STATIC_EXPORT_SKIPPED_EXIT_CODE

        service = StaticSiteExportService(SessionLocal)
        try:
            result = service.export(
                Path(args.output_dir),
                clean=not args.no_clean,
                markets=((args.market,) if args.market else None),
                write_manifest=args.market is None,
            )
        except NoPublishedStaticMarketArtifact:
            if (
                args.market is not None
                and args.refresh_daily
                and selected_market_non_publishable_snapshot is not None
            ):
                _write_market_diagnostics(
                    Path(args.output_dir),
                    args.market,
                    selected_market_non_publishable_snapshot,
                )
                print(
                    f"Static site export skipped for market {args.market}; "
                    "no current artifact was produced, diagnostics were uploaded, "
                    "and the combine job can use fallback artifacts."
                )
                return STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE
            raise

    print("Static site export complete:")
    print(f"  - output_dir: {result.output_dir}")
    print(f"  - generated_at: {result.generated_at}")
    print(f"  - as_of_date: {result.as_of_date}")
    for warning in (*refresh_warnings, *result.warnings):
        print(f"  - warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
