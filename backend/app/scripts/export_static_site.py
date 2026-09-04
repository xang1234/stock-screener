"""CLI for building the read-only static-site data bundle."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from app.config import settings
from app.database import SessionLocal
from app.domain.markets import market_registry
from app.domain.markets.catalog import get_market_catalog
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.models.feature_store import FeatureRunPointer
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.ibd_industry_service import IBDIndustryService
from app.services.group_rank_history_backfill_service import (
    DEFAULT_CALENDAR_DAY_GROUP_RANK_HISTORY_LOOKBACK_DAYS,
    GroupRankHistoryBackfillResult,
    GroupRankHistoryBackfillService,
    GroupRankHistoryBackfillStatus,
)
from app.services.benchmark_cache_service import BenchmarkFallbackPolicy
from app.services.benchmark_resolution import BenchmarkResolution
from app.services.market_exposure_service import EXPOSURE_BACKFILL_DAYS
from app.services.static_daily_price_refresh_service import (
    StaticDailyPriceRefreshService,
    static_daily_price_refresh_batch_size as _static_daily_price_refresh_batch_size,
)
from app.services.static_breadth_eligibility import (
    classify_static_breadth_eligibility,
)
from app.services.static_breadth_history_coordinator import (
    StaticBreadthHistoryCoordinator,
    StaticBreadthHistoryRequest,
)
from app.services.static_breadth_contributor_metadata_contract import (
    build_static_breadth_contributor_metadata_plan,
)
from app.services.static_breadth_contributor_metadata_finalizer import (
    StaticBreadthContributorMetadataFinalizer,
)
from app.services.static_site_export_service import (
    NoPublishedStaticMarketArtifact,
    STATIC_SITE_SCHEMA_VERSION,
    StaticSiteExportService,
)
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGRollingHistoryExportSession,
)
from app.services.static_group_snapshot_coordinator import (
    build_static_group_snapshot_coordinator,
)
from app.services.static_rrg_history_contract import StaticRRGHistoryBundleError
from app.services.market_rs_result_contract import (
    MARKET_RS_REASON_BENCHMARK_ADJUSTED_ANCHOR_MISSING,
)
from app.services.static_market_publish_policy import (
    OPTIONAL_STATIC_MARKETS,
    StaticMarketRsArtifactState,
    classify_static_market_rs_artifact_result,
    collect_static_no_current_artifact_failures,
)
from app.tasks.data_fetch_lock import disable_serialized_data_fetch_lock
from app.tasks.workload_coordination import disable_serialized_market_workload
from app.wiring.bootstrap import (
    get_benchmark_cache,
    get_group_rank_snapshot_coordinator,
    get_market_calendar_service,
    get_price_cache,
    get_provider_snapshot_service,
)


STATIC_BREADTH_HISTORY_MIN_TRADING_DAYS = 20
STATIC_BREADTH_HISTORY_LOOKBACK_DAYS = 90
STATIC_BREADTH_RATIO_RECOMPUTE_TRADING_DAYS = 10
STATIC_BUILD_MODE_PRICE_DELTA = "price_delta"
STATIC_BUILD_MODE_FULL = "full"
STATIC_EXPORT_MARKETS = market_registry.supported_market_codes()
STATIC_DEFAULT_MARKET = "US"
STATIC_EXPOSURE_PRIMARY_ONLY_BENCHMARK_MARKETS = frozenset({"US"})
STATIC_EXPORT_SKIPPED_EXIT_CODE = 78
STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE = 79
STATIC_RS_BENCHMARK_HYDRATION_PERIOD = "2y"
STATIC_RS_BENCHMARK_HYDRATION_ATTEMPTS = 2
STATIC_RS_BENCHMARK_RESOLUTION_EXCEPTION = "benchmark_resolution_exception"
SUPPORTED_RS_FORMULA_VERSIONS = frozenset(
    {BALANCED_RS_FORMULA_VERSION, LEGACY_RS_FORMULA_VERSION}
)
STATIC_NO_CURRENT_ARTIFACT_EXIT_MESSAGES = {
    "group_rank_backfill_not_ready": "group-rank history backfill was not ready",
    "market_exposure_not_ready": "exposure was not stored",
    "market_rs_not_ready": "Market RS was not ready",
}


def _default_rs_formula_policy() -> dict[str, str]:
    """Return the independently overridable publication policy per market."""
    return {
        market: BALANCED_RS_FORMULA_VERSION for market in STATIC_EXPORT_MARKETS
    }


def _parse_rs_formula_overrides_json(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object keyed by market")

    overrides: dict[str, str] = {}
    for raw_market, raw_formula in parsed.items():
        market = str(raw_market).strip().upper()
        formula = str(raw_formula).strip()
        if market not in STATIC_EXPORT_MARKETS:
            raise argparse.ArgumentTypeError(f"unsupported market {market!r}")
        if formula not in SUPPORTED_RS_FORMULA_VERSIONS:
            raise argparse.ArgumentTypeError(
                f"unsupported RS formula {formula!r} for market {market}"
            )
        overrides[market] = formula
    return overrides


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


def _static_exposure_benchmark_fallback_policy(market: str) -> BenchmarkFallbackPolicy:
    normalized_market = market.upper()
    if normalized_market in STATIC_EXPOSURE_PRIMARY_ONLY_BENCHMARK_MARKETS:
        return BenchmarkFallbackPolicy.PRIMARY_ONLY
    return BenchmarkFallbackPolicy.ALLOW


def _compute_static_market_exposure(*, as_of_date: date, market: str) -> dict[str, Any]:
    from app.services.market_exposure_service import refresh_market_exposure_for_date

    normalized_market = market.upper()
    with SessionLocal() as db:
        return refresh_market_exposure_for_date(
            db,
            normalized_market,
            as_of_date,
            benchmark_fallback_policy=_static_exposure_benchmark_fallback_policy(normalized_market),
        )


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


def _market_rs_not_ready_warning(
    *,
    market: str,
    as_of_date: date,
    result: dict[str, Any],
) -> str:
    reason = result.get("reason_code") or result.get("status") or "unknown"
    return (
        f"Static export market {market} Market RS not ready "
        f"for {as_of_date.isoformat()}: {reason}."
    )


def _market_rs_not_ready_snapshot(
    *,
    market: str,
    as_of_date: date,
    result: dict[str, Any],
    warning: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": "market_rs_not_ready",
        "market": market,
        "as_of_date": as_of_date.isoformat(),
        "failure_diagnostics": {
            "reason_code": result.get("reason_code"),
            "diagnostics": result.get("diagnostics") or {},
        },
        "warnings": [warning],
    }


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


def _run_static_options_refresh(source_run_id: int) -> dict[str, Any]:
    from app.interfaces.tasks.options_analytics_tasks import refresh_options_analytics

    try:
        return refresh_options_analytics.run(
            source_run_id=source_run_id,
            market="US",
        )
    except Exception as exc:
        return {
            "status": "failed",
            "reason_codes": ["options_refresh_failed"],
            "error": str(exc),
        }


def _write_market_diagnostics(output_dir: Path, market: str, snapshot: Mapping[str, Any]) -> Path:
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


def _no_current_artifact_exit_message(
    *,
    market: str | None,
    failed_markets: tuple[str, ...],
    reasons: tuple[str, ...],
) -> str:
    market_label = market or ", ".join(failed_markets)
    generic_detail = "one or more market artifacts were not current"
    unique_reasons = set(reasons)
    if len(unique_reasons) == 1:
        detail = STATIC_NO_CURRENT_ARTIFACT_EXIT_MESSAGES.get(
            reasons[0],
            generic_detail,
        )
    else:
        detail = generic_detail
    return (
        f"Static site export skipped for market {market_label}; "
        f"{detail}, diagnostics were uploaded, "
        "and the combine job can use fallback artifacts."
    )


def _refresh_static_daily_prices(*, as_of_date: date, market: str | None = None) -> dict[str, Any]:
    service = StaticDailyPriceRefreshService(
        session_factory=SessionLocal,
        price_cache=get_price_cache(),
        fetcher=BulkDataFetcher(),
        batch_size_for_market=_static_daily_price_refresh_batch_size,
        breadth_history_price_lookback_days=EXPOSURE_BACKFILL_DAYS,
    )
    return service.refresh(
        as_of_date=as_of_date,
        market=market,
        ensure_static_history=True,
    )


def _benchmark_resolution_exception(exc: Exception) -> BenchmarkResolution:
    return BenchmarkResolution(
        bundle=None,
        error=STATIC_RS_BENCHMARK_RESOLUTION_EXCEPTION,
        diagnostics={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )


def _resolve_static_rs_benchmark_bundle(
    benchmark_cache: Any,
    *,
    market: str,
    as_of_date: date,
    force_refresh: bool,
) -> BenchmarkResolution:
    try:
        return benchmark_cache.resolve_benchmark_bundle(
            market=market,
            period=STATIC_RS_BENCHMARK_HYDRATION_PERIOD,
            force_refresh=force_refresh,
            fallback_policy=BenchmarkFallbackPolicy.ALLOW,
            required_as_of_date=as_of_date,
        )
    except Exception as exc:  # pragma: no cover - provider/cache variability
        print(
            f"[static-rs] Benchmark resolution for {market} failed "
            f"for {as_of_date.isoformat()}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return _benchmark_resolution_exception(exc)


def _resolve_static_rs_benchmark_anchors(*, market: str, as_of_date: date) -> Any:
    benchmark_cache = get_benchmark_cache()
    last_resolution: Any = None
    for attempt in range(STATIC_RS_BENCHMARK_HYDRATION_ATTEMPTS):
        last_resolution = _resolve_static_rs_benchmark_bundle(
            benchmark_cache,
            market=market,
            as_of_date=as_of_date,
            force_refresh=True,
        )
        if last_resolution.bundle is not None:
            return last_resolution
        if attempt + 1 < STATIC_RS_BENCHMARK_HYDRATION_ATTEMPTS:
            print(
                f"[static-rs] Benchmark anchor for {market} is missing or stale "
                f"for {as_of_date.isoformat()}; retrying benchmark candidates.",
                flush=True,
            )
    cached_resolution = _resolve_static_rs_benchmark_bundle(
        benchmark_cache,
        market=market,
        as_of_date=as_of_date,
        force_refresh=False,
    )
    if cached_resolution.bundle is not None:
        return cached_resolution
    if (
        getattr(last_resolution, "error", None)
        == STATIC_RS_BENCHMARK_RESOLUTION_EXCEPTION
    ):
        return last_resolution
    if (
        getattr(cached_resolution, "error", None)
        == STATIC_RS_BENCHMARK_RESOLUTION_EXCEPTION
    ):
        return cached_resolution
    last_statuses = tuple(getattr(last_resolution, "candidate_statuses", ()) or ())
    cached_statuses = tuple(getattr(cached_resolution, "candidate_statuses", ()) or ())
    if last_statuses or cached_statuses:
        return BenchmarkResolution(
            bundle=None,
            candidate_statuses=tuple((*last_statuses, *cached_statuses)),
            error=getattr(cached_resolution, "error", None)
            or getattr(last_resolution, "error", None),
        )
    return last_resolution


def _benchmark_resolution_candidates(resolution: Any) -> tuple[str, ...]:
    bundle = getattr(resolution, "bundle", None)
    candidate_symbols = getattr(bundle, "candidate_symbols", None)
    if candidate_symbols:
        return tuple(str(symbol) for symbol in candidate_symbols if str(symbol))
    return ()


def _date_from_iso(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _latest_static_rs_benchmark_backed_as_of_date(
    result: Mapping[str, Any],
    *,
    requested_as_of_date: date,
) -> date | None:
    if (
        result.get("reason_code")
        != MARKET_RS_REASON_BENCHMARK_ADJUSTED_ANCHOR_MISSING
    ):
        return None
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    if _date_from_iso(diagnostics.get("date")) != requested_as_of_date:
        return None
    candidates = diagnostics.get("benchmark_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None

    latest_dates = [
        latest_date
        for candidate in candidates
        if isinstance(candidate, Mapping)
        for latest_date in (_date_from_iso(candidate.get("latest_date")),)
        if latest_date is not None and latest_date <= requested_as_of_date
    ]
    if not latest_dates:
        return None
    latest_backed_date = max(latest_dates)
    if latest_backed_date >= requested_as_of_date:
        return None
    return latest_backed_date


def _hydrate_remaining_static_rs_benchmarks(
    *,
    market: str,
    as_of_date: date,
    resolution: Any,
) -> None:
    bundle = getattr(resolution, "bundle", None)
    selected_symbol = getattr(bundle, "benchmark_symbol", None)
    if selected_symbol is None:
        return

    benchmark_cache = get_benchmark_cache()
    remaining_symbols = tuple(
        symbol
        for symbol in _benchmark_resolution_candidates(resolution)
        if symbol != selected_symbol
    )
    for benchmark_symbol in remaining_symbols:
        try:
            benchmark_cache.fetch_and_cache_benchmark(
                benchmark_symbol=benchmark_symbol,
                market=market,
                period=STATIC_RS_BENCHMARK_HYDRATION_PERIOD,
                required_as_of_date=as_of_date,
            )
        except Exception as exc:  # pragma: no cover - provider/cache variability
            print(
                f"[static-rs] Fallback benchmark hydration for {market} "
                f"{benchmark_symbol} failed for {as_of_date.isoformat()}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


def _prepare_balanced_static_rs(*, market: str, as_of_date: date) -> dict[str, Any]:
    """Build the exact canonical snapshot and select it in this private build DB."""
    from app.tasks.market_rs_tasks import calculate_market_rs_snapshot

    normalized_market = market.upper()
    benchmark_resolution = _resolve_static_rs_benchmark_anchors(
        market=normalized_market,
        as_of_date=as_of_date,
    )
    if benchmark_resolution.bundle is None:
        return {
            "status": "failed",
            "market": normalized_market,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": BALANCED_RS_FORMULA_VERSION,
            "reason_code": MARKET_RS_REASON_BENCHMARK_ADJUSTED_ANCHOR_MISSING,
            "diagnostics": benchmark_resolution.error_payload(
                market=normalized_market,
                as_of_date=as_of_date,
            ),
            "market_rs_run_id": None,
        }

    result = calculate_market_rs_snapshot.run(
        market=normalized_market,
        calculation_date=as_of_date.isoformat(),
        formula_version=BALANCED_RS_FORMULA_VERSION,
        rebuild_incompatible=True,
    )
    if (
        isinstance(result, dict)
        and result.get("status") == "failed"
        and result.get("reason_code")
        == MARKET_RS_REASON_BENCHMARK_ADJUSTED_ANCHOR_MISSING
    ):
        _hydrate_remaining_static_rs_benchmarks(
            market=normalized_market,
            as_of_date=as_of_date,
            resolution=benchmark_resolution,
        )
        result = calculate_market_rs_snapshot.run(
            market=normalized_market,
            calculation_date=as_of_date.isoformat(),
            formula_version=BALANCED_RS_FORMULA_VERSION,
            rebuild_incompatible=True,
        )
    artifact_state = classify_static_market_rs_artifact_result(
        result,
        market=normalized_market,
        as_of_date=as_of_date,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    if artifact_state is StaticMarketRsArtifactState.NO_CURRENT_ARTIFACT:
        if not isinstance(result, Mapping):
            raise RuntimeError(
                f"Balanced Market RS preparation returned a non-mapping no-current "
                f"artifact result for {normalized_market} on {as_of_date.isoformat()}: "
                f"{result}"
            )
        return {**result, "market_rs_run_id": None}
    if artifact_state is not StaticMarketRsArtifactState.READY:
        raise RuntimeError(
            f"Balanced Market RS preparation failed for {normalized_market} "
            f"on {as_of_date.isoformat()}: {result}"
        )

    with SessionLocal() as db:
        MarketRsRunRepository().activate_formula(
            db,
            market=normalized_market,
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )
        db.commit()
    return result


def _prepare_static_rs_formula(
    *,
    market: str,
    as_of_date: date,
    formula_version: str,
) -> dict[str, Any]:
    if formula_version == BALANCED_RS_FORMULA_VERSION:
        return _prepare_balanced_static_rs(market=market, as_of_date=as_of_date)
    if formula_version != LEGACY_RS_FORMULA_VERSION:
        raise ValueError(f"Unsupported static RS formula: {formula_version}")

    normalized_market = market.upper()
    with SessionLocal() as db:
        MarketRsRunRepository().activate_formula(
            db,
            market=normalized_market,
            formula_version=LEGACY_RS_FORMULA_VERSION,
        )
        db.commit()
    return {
        "status": "selected",
        "market": normalized_market,
        "as_of_date": as_of_date.isoformat(),
        "formula_version": LEGACY_RS_FORMULA_VERSION,
        "market_rs_run_id": None,
    }


def _generate_trading_dates(
    start_date: date,
    end_date: date,
    *,
    market: str = STATIC_DEFAULT_MARKET,
) -> list[date]:
    normalized_market = (market or STATIC_DEFAULT_MARKET).upper()
    return get_market_calendar_service().trading_days(
        normalized_market,
        start_date,
        end_date,
    )


def _ensure_group_rank_history(
    *,
    as_of_date: date,
    market: str = "US",
    formula_version: str,
) -> GroupRankHistoryBackfillResult:
    """Backfill recent group-rank history so 1W/1M/3M deltas can be rendered."""
    calendar_service = get_market_calendar_service()
    group_snapshot_coordinator = (
        build_static_group_snapshot_coordinator(
            calendar_service=calendar_service,
        )
        if formula_version == BALANCED_RS_FORMULA_VERSION
        else get_group_rank_snapshot_coordinator()
    )
    return GroupRankHistoryBackfillService(
        session_factory=SessionLocal,
        calendar_service=calendar_service,
        group_snapshot_coordinator=group_snapshot_coordinator,
    ).backfill(
        as_of_date=as_of_date,
        market=market,
        formula_version=formula_version,
    )


def _static_breadth_ready_for_exposure(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("error"):
        return False
    return result.get("status") in {"completed", "skipped"}


def _finalize_static_breadth_contributor_metadata(
    *,
    market: str,
    directory: Path,
    source_status: str,
) -> dict[str, Any]:
    normalized_status = str(source_status or "").strip().lower()
    if normalized_status == "failed":
        raise RuntimeError(
            f"Static breadth contributor metadata restore failed for {market}; "
            "the market artifact is unsafe to publish."
        )
    plan = build_static_breadth_contributor_metadata_plan(
        market=market,
        directory=directory,
    )
    with SessionLocal() as db:
        report = StaticBreadthContributorMetadataFinalizer(db).finalize(
            market=market,
            source_path=plan.source_path,
            output_path=plan.output_path,
            source_status=normalized_status,
        )
    return {
        "status": "completed",
        **report.as_dict(),
        "asset_name": plan.asset_name,
        "output_path": str(plan.output_path),
    }


def _run_daily_refresh(
    *,
    market: str | None = None,
    skip_universe_refresh: bool = False,
    skip_fundamentals_refresh: bool = False,
    build_mode: Literal["price_delta", "full"] = STATIC_BUILD_MODE_PRICE_DELTA,
    hydrate_published_snapshot: bool = False,
    rs_formula_version: str = BALANCED_RS_FORMULA_VERSION,
    rs_formula_version_by_market: Mapping[str, str] | None = None,
    breadth_contributor_metadata_dir: Path | None = None,
    breadth_contributor_metadata_restore_status: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    from app.interfaces.tasks.feature_store_tasks import (
        _enrich_feature_run_with_ibd_metadata,
        build_daily_snapshot,
    )
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.universe_tasks import refresh_stock_universe

    warnings: list[str] = []

    if breadth_contributor_metadata_dir is not None and market is None:
        raise ValueError(
            "Breadth contributor metadata finalization requires one selected market."
        )
    selected_markets = (market,) if market is not None else STATIC_EXPORT_MARKETS
    formula_by_market = {
        selected_market: (
            rs_formula_version_by_market.get(selected_market, rs_formula_version)
            if rs_formula_version_by_market is not None
            else rs_formula_version
        )
        for selected_market in selected_markets
    }
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

        # Static CI uses a fresh, private database on every run. Migrations seed
        # its formula pointer to legacy for rollback safety, so explicitly build
        # and select balanced RS before any Feature, Group, or RRG consumer runs.
        market_rs_results: dict[str, Any] = {}
        for selected_market in selected_markets:
            market_as_of = as_of_by_market[selected_market]
            market_rs_result = _prepare_static_rs_formula(
                market=selected_market,
                as_of_date=market_as_of,
                formula_version=formula_by_market[selected_market],
            )
            benchmark_backed_as_of = _latest_static_rs_benchmark_backed_as_of_date(
                market_rs_result,
                requested_as_of_date=market_as_of,
            )
            if benchmark_backed_as_of is not None:
                warnings.append(
                    f"Static export market {selected_market} using benchmark-backed "
                    f"as-of date {benchmark_backed_as_of.isoformat()} because "
                    f"benchmarks were unavailable for {market_as_of.isoformat()}."
                )
                as_of_by_market[selected_market] = benchmark_backed_as_of
                market_rs_result = _prepare_static_rs_formula(
                    market=selected_market,
                    as_of_date=benchmark_backed_as_of,
                    formula_version=formula_by_market[selected_market],
                )
            market_rs_results[selected_market] = market_rs_result
        results["market_rs"] = market_rs_results
        market_rs_artifact_states = {
            selected_market: classify_static_market_rs_artifact_result(
                market_rs_results.get(selected_market),
                market=selected_market,
                as_of_date=as_of_by_market[selected_market],
                formula_version=formula_by_market[selected_market],
            )
            for selected_market in selected_markets
        }
        hard_market_rs_failures = {
            selected_market: market_rs_results.get(selected_market)
            for selected_market in selected_markets
            if market_rs_artifact_states[selected_market]
            is StaticMarketRsArtifactState.HARD_FAILURE
        }
        if hard_market_rs_failures:
            details = "; ".join(
                f"Static Market RS failed hard for {failed_market} "
                f"on {as_of_by_market[failed_market].isoformat()}: {failed_result}"
                for failed_market, failed_result in hard_market_rs_failures.items()
            )
            raise RuntimeError(
                details
            )

        market_rs_no_current_artifact_warnings = {
            selected_market: _market_rs_not_ready_warning(
                market=selected_market,
                as_of_date=as_of_by_market[selected_market],
                result=market_rs_results[selected_market],
            )
            for selected_market in selected_markets
            if market_rs_artifact_states[selected_market]
            is StaticMarketRsArtifactState.NO_CURRENT_ARTIFACT
        }
        warnings.extend(market_rs_no_current_artifact_warnings.values())

        supports_breadth_by_market = {
            selected_market: get_market_catalog()
            .get(selected_market)
            .capabilities.breadth
            for selected_market in selected_markets
        }
        breadth_history: dict[str, Any] = {}
        for selected_market in selected_markets:
            if not supports_breadth_by_market[selected_market]:
                breadth_history[selected_market] = {
                    "status": "skipped",
                    "reason": "market_breadth_unsupported",
                    "market": selected_market,
                    "as_of_date": as_of_by_market[selected_market].isoformat(),
                }
                continue
            if (
                market_rs_artifact_states[selected_market]
                is StaticMarketRsArtifactState.NO_CURRENT_ARTIFACT
            ):
                breadth_history[selected_market] = {
                    "status": "skipped",
                    "reason": "market_rs_not_ready",
                    "market": selected_market,
                    "as_of_date": as_of_by_market[selected_market].isoformat(),
                }
                continue
            market_as_of = as_of_by_market[selected_market]
            try:
                breadth_history[selected_market] = _ensure_breadth_history(
                    as_of_date=market_as_of,
                    market=selected_market,
                    min_trading_days=0,
                    lookback_days=EXPOSURE_BACKFILL_DAYS,
                )
            except Exception as exc:
                breadth_history[selected_market] = {
                    "status": "errored",
                    "market": selected_market,
                    "as_of_date": market_as_of.isoformat(),
                    "error": str(exc),
                    "exception_type": exc.__class__.__name__,
                }
                warnings.append(
                    f"Static export market {selected_market} breadth history "
                    f"failed for {market_as_of.isoformat()}: {exc}"
                )
        results["breadth_history"] = breadth_history

        market_exposure: dict[str, Any] = {}
        for selected_market in selected_markets:
            market_as_of = as_of_by_market[selected_market]
            if not supports_breadth_by_market[selected_market]:
                market_exposure[selected_market] = {
                    "status": "skipped",
                    "reason": "market_breadth_unsupported",
                    "market": selected_market,
                    "date": market_as_of.isoformat(),
                }
                continue
            if (
                market_rs_artifact_states[selected_market]
                is StaticMarketRsArtifactState.NO_CURRENT_ARTIFACT
            ):
                market_exposure[selected_market] = {
                    "status": "skipped",
                    "reason": "market_rs_not_ready",
                    "market": selected_market,
                    "date": market_as_of.isoformat(),
                }
                continue
            if not _static_breadth_ready_for_exposure(breadth_history.get(selected_market)):
                market_exposure[selected_market] = {
                    "status": "skipped",
                    "reason": "market_breadth_not_ready",
                    "error": "market_breadth_not_ready",
                    "market": selected_market,
                    "date": market_as_of.isoformat(),
                    "breadth_history": breadth_history.get(selected_market),
                }
                warnings.append(
                    f"Static export market {selected_market} exposure not stored "
                    f"for {market_as_of.isoformat()}: market_breadth_not_ready."
                )
                continue
            try:
                exposure_result = _compute_static_market_exposure(
                    as_of_date=market_as_of,
                    market=selected_market,
                )
            except Exception as exc:  # pragma: no cover - defensive diagnostics path
                exposure_result = {
                    "error": str(exc),
                    "market": selected_market,
                    "date": market_as_of.isoformat(),
                }
            market_exposure[selected_market] = exposure_result
            if isinstance(exposure_result, dict) and exposure_result.get("error"):
                warnings.append(
                    f"Static export market {selected_market} exposure not stored "
                    f"for {market_as_of.isoformat()}: {exposure_result['error']}."
                )
            history_seed = (
                exposure_result.get("history_seed")
                if isinstance(exposure_result, dict)
                else None
            )
            if isinstance(history_seed, dict) and history_seed.get("error"):
                warnings.append(
                    f"Static export market {selected_market} exposure history seed skipped: "
                    f"{history_seed['error']}."
                )
        results["market_exposure"] = market_exposure

        feature_snapshots: dict[str, Any] = {}
        for selected_market in selected_markets:
            market_as_of = as_of_by_market[selected_market]
            market_rs_warning = market_rs_no_current_artifact_warnings.get(selected_market)
            if market_rs_warning is not None:
                feature_snapshots[selected_market] = _market_rs_not_ready_snapshot(
                    market=selected_market,
                    as_of_date=market_as_of,
                    result=market_rs_results[selected_market],
                    warning=market_rs_warning,
                )
                continue
            exposure_result = market_exposure.get(selected_market)
            if isinstance(exposure_result, dict) and exposure_result.get("error"):
                exposure_warning = (
                    f"Static export market {selected_market} exposure not stored "
                    f"for {market_as_of.isoformat()}: {exposure_result['error']}."
                )
                feature_snapshots[selected_market] = {
                    "status": "skipped",
                    "reason": "market_exposure_not_ready",
                    "market": selected_market,
                    "as_of_date": market_as_of.isoformat(),
                    "failure_diagnostics": {
                        "date": exposure_result.get("date") or market_as_of.isoformat(),
                        "error": exposure_result["error"],
                    },
                    "warnings": [exposure_warning],
                }
                continue
            market_result = build_daily_snapshot.run(
                as_of_date_str=market_as_of.isoformat(),
                static_daily_mode=True,
                universe_name=f"market:{selected_market.lower()}",
                market=selected_market,
                publish_pointer_key=_market_pointer_key(selected_market),
                ignore_runtime_market_gate=True,
                rs_formula_version_override=formula_by_market[selected_market],
                skip_ibd_metadata_enrichment=True,
            )
            feature_snapshots[selected_market] = market_result

        results["feature_snapshots"] = feature_snapshots

        # build_daily_snapshot hydrates broad historical prices in static CI.
        # Group-rank history must run after that hydration step, then metadata
        # enrichment is replayed so rows pick up the newly stored ranks.
        group_rank_history: dict[str, GroupRankHistoryBackfillResult] = {}
        group_rank_history_report: dict[str, dict[str, Any]] = {}
        for selected_market in selected_markets:
            market_as_of = as_of_by_market[selected_market]
            if (
                market_rs_artifact_states[selected_market]
                is StaticMarketRsArtifactState.NO_CURRENT_ARTIFACT
            ):
                skipped = GroupRankHistoryBackfillResult(
                    status=GroupRankHistoryBackfillStatus.SKIPPED,
                    market=selected_market,
                    as_of_date=market_as_of,
                    lookback_start_date=market_as_of,
                    reason="market_rs_not_ready",
                )
                group_rank_history[selected_market] = skipped
                group_rank_history_report[selected_market] = skipped.as_dict()
                continue
            snapshot = feature_snapshots.get(selected_market, {})
            if not _snapshot_publishable(snapshot):
                skipped = GroupRankHistoryBackfillResult(
                    status=GroupRankHistoryBackfillStatus.SKIPPED,
                    market=selected_market,
                    as_of_date=market_as_of,
                    lookback_start_date=(
                        market_as_of
                        - timedelta(
                            days=DEFAULT_CALENDAR_DAY_GROUP_RANK_HISTORY_LOOKBACK_DAYS
                        )
                    ),
                    reason="snapshot_not_ready",
                )
                group_rank_history[selected_market] = skipped
                group_rank_history_report[selected_market] = skipped.as_dict()
                continue
            backfill = _ensure_group_rank_history(
                as_of_date=market_as_of,
                market=selected_market,
                formula_version=formula_by_market[selected_market],
            )
            group_rank_history[selected_market] = backfill
            group_rank_history_report[selected_market] = backfill.as_dict()
            snapshot = feature_snapshots.get(selected_market, {})
            metadata_refresh = (
                snapshot.get("metadata_refresh") if isinstance(snapshot, dict) else None
            )
            if (
                not backfill.ready_for_enrichment
                and isinstance(snapshot, dict)
                and snapshot.get("status") == "published"
                and isinstance(metadata_refresh, dict)
                and metadata_refresh.get("reason") == "deferred"
            ):
                warning = (
                    f"Static export market {selected_market} group-rank history "
                    f"backfill not ready for {market_as_of.isoformat()}: "
                    f"{backfill.status.value}."
                )
                existing_diagnostics = snapshot.get("failure_diagnostics")
                failure_diagnostics = (
                    dict(existing_diagnostics) if isinstance(existing_diagnostics, dict) else {}
                )
                failure_diagnostics["group_rank_history_backfill"] = backfill.as_dict()
                existing_warnings = snapshot.get("warnings")
                snapshot_warnings = (
                    list(existing_warnings) if isinstance(existing_warnings, list) else []
                )
                snapshot_warnings.append(warning)
                feature_snapshots[selected_market] = {
                    **snapshot,
                    "status": "quarantined",
                    "reason": "group_rank_backfill_not_ready",
                    "market": selected_market,
                    "as_of_date": market_as_of.isoformat(),
                    "warnings": snapshot_warnings,
                    "failure_diagnostics": failure_diagnostics,
                }
                warnings.append(warning)
        results["group_rank_history_backfill"] = group_rank_history_report

        # Re-enrich feature runs after the IBDGroupRank backfill above.
        # build_daily_snapshot's inner enrichment runs *before* group ranks
        # for `as_of_date` are populated, so rows would otherwise carry
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
            backfill = group_rank_history[selected_market]
            if not backfill.ready_for_enrichment:
                ibd_metadata_refresh[selected_market] = {
                    "status": "skipped",
                    "market": selected_market,
                    "reason": f"group_rank_backfill_{backfill.status.value}",
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

        if breadth_contributor_metadata_dir is not None:
            contributor_metadata: dict[str, Any] = {}
            for selected_market in selected_markets:
                if not supports_breadth_by_market[selected_market]:
                    contributor_metadata[selected_market] = {
                        "status": "skipped",
                        "market": selected_market,
                        "reason": "market_breadth_unsupported",
                    }
                    continue
                snapshot = feature_snapshots.get(selected_market, {})
                if not _snapshot_publishable(snapshot):
                    contributor_metadata[selected_market] = {
                        "status": "skipped",
                        "market": selected_market,
                        "reason": "snapshot_not_ready",
                    }
                    continue
                contributor_metadata[selected_market] = (
                    _finalize_static_breadth_contributor_metadata(
                        market=selected_market,
                        directory=breadth_contributor_metadata_dir,
                        source_status=(
                            breadth_contributor_metadata_restore_status or ""
                        ),
                    )
                )
            results["breadth_contributor_metadata"] = contributor_metadata

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
                if settings.options_analytics_enabled:
                    options_result = _run_static_options_refresh(int(default_run_id))
                    results["options_analytics"] = options_result
                    if options_result.get("status") != "published":
                        warnings.append(
                            "Static US Options Analytics did not publish; "
                            "last-good options may be used independently."
                        )
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


def main(argv: Sequence[str] | None = None) -> int:
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
        "--options-artifacts-dir",
        help="Optional current options directory selected independently in combine mode.",
    )
    parser.add_argument(
        "--fallback-options-artifacts-dir",
        help="Optional last-good options directory selected independently in combine mode.",
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
    parser.add_argument(
        "--rrg-history-dir",
        help="Optional directory holding the market's rolling RRG history state.",
    )
    parser.add_argument(
        "--breadth-contributor-metadata-dir",
        help=(
            "Optional directory holding the market's rolling breadth contributor "
            "metadata state."
        ),
    )
    parser.add_argument(
        "--breadth-contributor-metadata-restore-status",
        choices=("restored", "missing", "failed"),
        default=os.environ.get("BREADTH_CONTRIBUTOR_METADATA_RESTORE_STATUS"),
        help="Result of restoring the rolling breadth contributor metadata asset.",
    )
    parser.add_argument(
        "--rs-formula-version",
        choices=(BALANCED_RS_FORMULA_VERSION, LEGACY_RS_FORMULA_VERSION),
        help="RS formula override for a single --market export.",
    )
    parser.add_argument(
        "--rs-formula-overrides-json",
        type=_parse_rs_formula_overrides_json,
        default={},
        metavar="JSON",
        help=(
            "Per-market formula overrides, for example "
            "'{\"HK\":\"legacy-linear-v1\"}'."
        ),
    )
    args = parser.parse_args(argv)

    if args.combine_artifacts_dir and args.refresh_daily:
        raise SystemExit("--combine-artifacts-dir cannot be used together with --refresh-daily")
    if args.combine_artifacts_dir and args.market:
        raise SystemExit("--combine-artifacts-dir cannot be used together with --market")
    if args.fallback_artifacts_dir and not args.combine_artifacts_dir:
        raise SystemExit("--fallback-artifacts-dir requires --combine-artifacts-dir")
    if (
        args.options_artifacts_dir or args.fallback_options_artifacts_dir
    ) and not args.combine_artifacts_dir:
        raise SystemExit("options artifact directories require --combine-artifacts-dir")
    if args.rrg_history_dir and not args.market:
        raise SystemExit("--rrg-history-dir requires --market")
    if args.combine_artifacts_dir and args.rrg_history_dir:
        raise SystemExit("--rrg-history-dir cannot be used while combining artifacts")
    if args.breadth_contributor_metadata_dir and not args.market:
        raise SystemExit("--breadth-contributor-metadata-dir requires --market")
    if (
        args.breadth_contributor_metadata_dir
        and not args.breadth_contributor_metadata_restore_status
    ):
        raise SystemExit(
            "--breadth-contributor-metadata-restore-status is required when "
            "--breadth-contributor-metadata-dir is used"
        )
    if args.breadth_contributor_metadata_dir and not args.refresh_daily:
        raise SystemExit(
            "--breadth-contributor-metadata-dir requires --refresh-daily"
        )
    if args.rs_formula_version and (args.combine_artifacts_dir or not args.market):
        raise SystemExit("--rs-formula-version is limited to single-market exports")

    rs_formula_policy = _default_rs_formula_policy()
    rs_formula_policy.update(args.rs_formula_overrides_json)
    if args.rs_formula_version:
        rs_formula_policy[args.market] = args.rs_formula_version

    refresh_warnings: list[str] = []
    selected_market_non_publishable_snapshot: dict[str, Any] | None = None
    if args.combine_artifacts_dir:
        options_combine_kwargs: dict[str, Path] = {}
        if args.options_artifacts_dir:
            options_combine_kwargs["options_artifacts_dir"] = Path(
                args.options_artifacts_dir
            )
        if args.fallback_options_artifacts_dir:
            options_combine_kwargs["fallback_options_artifacts_dir"] = Path(
                args.fallback_options_artifacts_dir
            )
        result = StaticSiteExportService.combine_market_artifacts(
            Path(args.combine_artifacts_dir),
            Path(args.output_dir),
            fallback_artifacts_dir=(
                Path(args.fallback_artifacts_dir)
                if args.fallback_artifacts_dir
                else None
            ),
            clean=not args.no_clean,
            rs_formula_version_overrides=rs_formula_policy,
            # Last-good artifacts can predate the balanced rollout. Only an
            # explicit operator override constrains a fallback publication.
            fallback_rs_formula_version_overrides=args.rs_formula_overrides_json,
            optional_markets=OPTIONAL_STATIC_MARKETS,
            **options_combine_kwargs,
        )
    else:
        prepare_runtime()

        if args.refresh_daily:
            refresh_results, daily_refresh_warnings = _run_daily_refresh(
                market=args.market,
                skip_universe_refresh=args.skip_universe_refresh,
                skip_fundamentals_refresh=args.skip_fundamentals_refresh,
                build_mode=args.build_mode,
                hydrate_published_snapshot=args.hydrate_published_snapshot,
                rs_formula_version_by_market=rs_formula_policy,
                breadth_contributor_metadata_dir=(
                    Path(args.breadth_contributor_metadata_dir)
                    if args.breadth_contributor_metadata_dir
                    else None
                ),
                breadth_contributor_metadata_restore_status=(
                    args.breadth_contributor_metadata_restore_status
                ),
            )
            refresh_warnings.extend(daily_refresh_warnings)
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
            no_current_artifact_failures = collect_static_no_current_artifact_failures(
                refresh_results,
                market=args.market,
            )
            if no_current_artifact_failures:
                for failure in no_current_artifact_failures:
                    _write_market_diagnostics(
                        Path(args.output_dir),
                        failure.market,
                        failure.snapshot,
                    )
                print(
                    _no_current_artifact_exit_message(
                        market=args.market,
                        failed_markets=tuple(
                            failure.market for failure in no_current_artifact_failures
                        ),
                        reasons=tuple(
                            failure.reason for failure in no_current_artifact_failures
                        ),
                    )
                )
                return STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE

        rrg_history_session = (
            StaticGroupsRRGRollingHistoryExportSession(
                schema_version=STATIC_SITE_SCHEMA_VERSION,
                market=args.market,
                directory=Path(args.rrg_history_dir),
            )
            if args.rrg_history_dir
            else None
        )

        service = StaticSiteExportService(
            SessionLocal,
            rrg_payload_source=rrg_history_session,
        )
        try:
            result = service.export(
                Path(args.output_dir),
                clean=not args.no_clean,
                markets=((args.market,) if args.market else None),
                write_manifest=args.market is None,
                rs_formula_version_overrides=rs_formula_policy,
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

        if rrg_history_session is not None:
            bootstrap_result = rrg_history_session.bootstrap_result
            if bootstrap_result is not None:
                print(f"Static RRG bootstrap: {bootstrap_result.as_dict()}")
            refresh_warnings.extend(rrg_history_session.warnings)
            try:
                history_stats = rrg_history_session.persist(
                    exported_as_of_date=date.fromisoformat(result.as_of_date),
                )
                if history_stats is not None:
                    print(f"Updated rolling RRG history: {history_stats}")
            except StaticRRGHistoryBundleError as exc:
                refresh_warnings.append(f"Rolling RRG history was not persisted: {exc}")

    print("Static site export complete:")
    print(f"  - output_dir: {result.output_dir}")
    print(f"  - generated_at: {result.generated_at}")
    print(f"  - as_of_date: {result.as_of_date}")
    for warning in (*refresh_warnings, *result.warnings):
        print(f"  - warning: {warning}")
    return 0


def _ensure_breadth_history(
    *,
    as_of_date: date,
    market: str = STATIC_DEFAULT_MARKET,
    min_trading_days: int = STATIC_BREADTH_HISTORY_MIN_TRADING_DAYS,
    lookback_days: int = STATIC_BREADTH_HISTORY_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Backfill recent breadth history through the canonical coordinator."""
    coordinator = StaticBreadthHistoryCoordinator(
        session_factory=SessionLocal,
        trading_dates=lambda start, end, normalized_market: _generate_trading_dates(
            start,
            end,
            market=normalized_market,
        ),
        eligibility_classifier=classify_static_breadth_eligibility,
        calculator_factory=lambda db, price_cache, normalized_market: (
            BreadthCalculatorService(
                db,
                price_cache,
                market=normalized_market,
            )
        ),
        price_cache_factory=get_price_cache,
        message_sink=lambda message: print(
            f"[static-breadth] {message}",
            flush=True,
        ),
    )
    return coordinator.ensure(
        StaticBreadthHistoryRequest(
            market=market,
            as_of_date=as_of_date,
            min_trading_days=min_trading_days,
            lookback_days=lookback_days,
        )
    ).as_dict()


if __name__ == "__main__":
    raise SystemExit(main())
