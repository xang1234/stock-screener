"""BuildDailyFeatureSnapshotUseCase — builds a complete feature snapshot for a trading day.

This use case contains the business rules for building a daily feature
snapshot:
  1. Validate as_of_date is a trading day for the requested market
  2. Start a feature run (status=RUNNING)
  3. Resolve universe of active symbols and save snapshot
  4. Scan each symbol via StockScanner in chunks (with progress + cancellation)
  5. Upsert feature rows after each chunk (checkpoint)
  6. Mark run COMPLETED with stats
  7. Delegate DQ evaluation and publish/quarantine to PublishFeatureRunUseCase
  8. Emit completion ProgressEvent

The use case depends ONLY on domain ports — never on SQLAlchemy, Celery,
Redis, or any other infrastructure.

Zero Celery imports.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Iterator, Protocol, Sequence

import pandas_market_calendars as mcal

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.feature_store.models import (
    RATING_TO_INT,
    FeatureRowWrite,
    RunStats,
    RunStatus,
    RunType,
)
from app.domain.feature_store.quality import DQInputs, DQThresholds
from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
    hash_universe_symbols,
)
from app.domain.scanning.models import ProgressEvent
from app.domain.scanning.ports import (
    CancellationToken,
    ProgressSink,
    StockDataProvider,
    StockScanner,
)
from app.services.bootstrap_cache_coverage import (
    BOOTSTRAP_CACHE_ONLY_MIN_COVERAGE,
    MISSING_SYMBOL_PREVIEW_LIMIT,
    evaluate_bootstrap_cache_coverage,
)
from app.utils.symbol_support import split_supported_price_symbols
from app.use_cases.feature_store.publish_run import (
    PublishFeatureRunUseCase,
    PublishRunCommand,
)

logger = logging.getLogger(__name__)

# Singleton calendar instance (thread-safe, immutable schedule data).
_nyse_calendar = mcal.get_calendar("NYSE")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_us_trading_day(dt: date) -> bool:
    """Check if *dt* is a valid US trading day using the NYSE calendar."""
    schedule = _nyse_calendar.schedule(start_date=dt, end_date=dt)
    return len(schedule) > 0


def _chunked(seq: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield successive chunks of *size* from *seq*."""
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _map_orchestrator_to_feature_row(
    symbol: str,
    as_of_date: date,
    result_dict: dict,
) -> FeatureRowWrite:
    """Map ScanOrchestrator dict output to a FeatureRowWrite domain object.

    Pure function — no I/O, fully deterministic.

    Stores the full ``result_dict`` as details so that JSON paths like
    ``$.rs_rating`` and ``$.minervini_score`` resolve correctly in the
    feature store query builder.
    """
    rating = result_dict.get("rating")
    if rating == "Insufficient Data":
        overall_rating = None
    elif rating is None:
        overall_rating = None
    else:
        overall_rating = RATING_TO_INT.get(rating, RATING_TO_INT["Watch"])

    return FeatureRowWrite(
        symbol=symbol,
        as_of_date=as_of_date,
        composite_score=result_dict.get("composite_score"),
        overall_rating=overall_rating,
        passes_count=result_dict.get("screeners_passed"),
        details=result_dict,
    )


def _resolve_result_status(result_dict: object) -> str:
    """Normalize legacy and new scanner payloads into snapshot persistence states."""
    if not isinstance(result_dict, dict):
        return "error"

    explicit_status = result_dict.get("result_status")
    if explicit_status in {"ok", "insufficient_history", "error"}:
        return explicit_status

    if "error" in result_dict:
        return "error"

    if result_dict.get("rating") == "Insufficient Data":
        return "insufficient_history"

    return "ok"


def _serialize_universe_definition(universe_def: object) -> dict[str, object]:
    """Best-effort JSON-safe snapshot of the requested universe definition."""
    model_dump = getattr(universe_def, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")

    result: dict[str, object] = {}
    for key in ("type", "market", "exchange", "index", "symbols", "allow_inactive_symbols"):
        value = getattr(universe_def, key, None)
        if value is None:
            continue
        result[key] = getattr(value, "value", value)
    return result


# ── Command (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildDailySnapshotCommand:
    """Immutable value object describing the snapshot to build."""

    as_of_date: date
    screener_names: list[str]
    universe_def: object  # UniverseSpec or similar
    criteria: dict = field(default_factory=dict)
    composite_method: str = "weighted_average"
    chunk_size: int = 50
    correlation_id: str | None = None
    code_version: str | None = None
    exclude_unsupported_price_symbols: bool = False
    batch_only_prices: bool = False
    batch_only_fundamentals: bool = False
    require_bulk_prefetch: bool = False
    static_parallel_workers: int = 1
    static_chunk_size: int | None = None
    bootstrap_cache_only_if_covered: bool = False
    bootstrap_coverage_report: dict | None = None
    publish_pointer_key: str = "latest_published"
    market: str = "US"

    # DQ thresholds (overridable per-invocation)
    dq_thresholds: DQThresholds = field(default_factory=DQThresholds)

    def __post_init__(self) -> None:
        normalized_market = (self.market or "US").strip().upper()
        object.__setattr__(self, "market", normalized_market)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.static_parallel_workers < 1:
            raise ValueError("static_parallel_workers must be >= 1")
        if self.static_chunk_size is not None and self.static_chunk_size < 1:
            raise ValueError("static_chunk_size must be >= 1 when provided")


# ── Result (output) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildDailySnapshotResult:
    """What the use case returns to the caller."""

    run_id: int
    status: str  # published | quarantined | failed
    total_symbols: int
    processed_symbols: int
    failed_symbols: int
    skipped_symbols: int
    dq_passed: bool
    warnings: tuple[str, ...]
    row_count: int
    duration_seconds: float


class BootstrapCacheCoverageInsufficient(RuntimeError):
    """Raised when bootstrap scan must wait for cache coverage before scanning."""

    def __init__(self, market: str, coverage_report: dict) -> None:
        self.market = market
        self.coverage_report = dict(coverage_report)
        price_ratio = float(self.coverage_report.get("price_coverage_ratio") or 0.0)
        fundamentals_ratio = float(
            self.coverage_report.get("fundamentals_coverage_ratio") or 0.0
        )
        super().__init__(
            "bootstrap cache coverage insufficient for "
            f"{market}: price={price_ratio:.3f}, fundamentals={fundamentals_ratio:.3f}"
        )


@dataclass
class _SnapshotRunProgress:
    """Mutable counters shared with best-effort failure handling."""

    attempted_symbols: int = 0
    failed_symbols: int = 0
    passed_symbols: int = 0


class MarketCalendarPort(Protocol):
    def is_trading_day(self, market: str, day: date | None = None) -> bool:
        ...


# ── Use Case ─────────────────────────────────────────────────────────────


class BuildDailyFeatureSnapshotUseCase:
    """Build a complete feature snapshot for a trading day.

    The constructor accepts infrastructure collaborators through ports.
    ``execute()`` receives a fresh UoW per invocation.
    """

    def __init__(
        self,
        scanner: StockScanner,
        data_provider: StockDataProvider | None = None,
        market_calendar: MarketCalendarPort | None = None,
    ) -> None:
        self._scanner = scanner
        self._data_provider = data_provider
        self._market_calendar = market_calendar

    def execute(
        self,
        uow: UnitOfWork,
        cmd: BuildDailySnapshotCommand,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> BuildDailySnapshotResult:
        """Run the full snapshot build lifecycle inside a single UoW."""
        # ── Validate trading day ────────────────────────────────
        if self._market_calendar is not None:
            is_trading_day = self._market_calendar.is_trading_day(cmd.market, cmd.as_of_date)
        elif cmd.market == "US":
            is_trading_day = _is_us_trading_day(cmd.as_of_date)
        else:
            raise ValidationError(f"No market calendar configured for {cmd.market}")
        if not is_trading_day:
            raise ValidationError(
                f"as_of_date {cmd.as_of_date} is not a {cmd.market} trading day"
            )

        with uow:
            symbols = uow.universe.resolve_symbols(cmd.universe_def)
            if not symbols:
                raise ValidationError("Universe resolved to zero symbols")
            run_warnings: list[str] = []
            skipped_symbols: list[str] = []
            bootstrap_gate_report: dict | None = None
            if cmd.bootstrap_cache_only_if_covered:
                supported_symbols, unsupported_symbols = split_supported_price_symbols(symbols)
                bootstrap_gate_report = dict(cmd.bootstrap_coverage_report or {})
                if not bootstrap_gate_report and supported_symbols:
                    session = getattr(uow, "session", None)
                    if session is None:
                        raise RuntimeError(
                            "Bootstrap cache-only coverage gate requires a SQLAlchemy session "
                            "or bootstrap_coverage_report"
                        )
                    bootstrap_gate_report = evaluate_bootstrap_cache_coverage(
                        session,
                        market=cmd.market,
                        symbols=supported_symbols,
                        as_of_date=cmd.as_of_date,
                    )
                elif not bootstrap_gate_report:
                    bootstrap_gate_report = {"eligible": False}
                eligible = bool(bootstrap_gate_report.get("eligible"))
                bootstrap_gate_report.update(
                    {
                        "threshold": BOOTSTRAP_CACHE_ONLY_MIN_COVERAGE,
                        "mode": (
                            "cache_only"
                            if eligible
                            else "waiting_for_cache_coverage"
                        ),
                        "unsupported_skipped_count": (
                            len(unsupported_symbols) if eligible else 0
                        ),
                        "unsupported_symbols_preview": (
                            unsupported_symbols[:MISSING_SYMBOL_PREVIEW_LIMIT]
                            if eligible
                            else []
                        ),
                    }
                )
                if eligible:
                    if not supported_symbols:
                        raise ValidationError(
                            "Universe resolved to zero symbols after excluding unsupported Yahoo price symbols"
                        )
                    cmd = replace(
                        cmd,
                        exclude_unsupported_price_symbols=True,
                        batch_only_prices=True,
                        batch_only_fundamentals=True,
                        require_bulk_prefetch=True,
                    )
                    symbols = supported_symbols
                    skipped_symbols = unsupported_symbols
                    if skipped_symbols:
                        run_warnings.append(
                            "Skipped unsupported Yahoo price symbols in bootstrap cache-only snapshot: "
                            f"{len(skipped_symbols)}"
                        )
                else:
                    raise BootstrapCacheCoverageInsufficient(
                        cmd.market,
                        bootstrap_gate_report,
                    )
            elif cmd.exclude_unsupported_price_symbols:
                symbols, skipped_symbols = split_supported_price_symbols(symbols)
                if skipped_symbols:
                    run_warnings.append(
                        "Skipped unsupported Yahoo price symbols in static daily snapshot: "
                        f"{len(skipped_symbols)}"
                    )
                    logger.info(
                        "Run will skip %d unsupported Yahoo price symbols before hashing: %s",
                        len(skipped_symbols),
                        ", ".join(skipped_symbols[:10]),
                    )
                if not symbols:
                    raise ValidationError(
                        "Universe resolved to zero symbols after excluding unsupported Yahoo price symbols"
                    )
            total = len(symbols)

            signature_payload = build_scan_signature_payload(
                universe_type=getattr(cmd.universe_def, "type", "all"),
                screeners=cmd.screener_names,
                composite_method=cmd.composite_method,
                criteria=cmd.criteria,
            )
            run_config = {
                **signature_payload,
                "signature": signature_payload,
                "universe": _serialize_universe_definition(cmd.universe_def),
                "market": cmd.market,
                "publish_pointer_key": cmd.publish_pointer_key,
            }
            if bootstrap_gate_report is not None:
                run_config["bootstrap_cache_only_gate"] = bootstrap_gate_report

            # Start run BEFORE the try-except so run_id is available
            # for best-effort error handling.
            run = uow.feature_runs.start_run(
                as_of_date=cmd.as_of_date,
                run_type=RunType.DAILY_SNAPSHOT,
                code_version=cmd.code_version,
                universe_hash=hash_universe_symbols(symbols),
                input_hash=hash_scan_signature(signature_payload),
                config_json=run_config,
                correlation_id=cmd.correlation_id,
            )
            run_id = run.id
            uow.commit()
            logger.info(
                "Started feature run %d for %s", run_id, cmd.as_of_date
            )
            run_started_at = time.monotonic()
            progress_state = _SnapshotRunProgress()

            try:
                return self._run(
                    uow,
                    run_id,
                    cmd,
                    progress,
                    cancel,
                    symbols,
                    tuple(run_warnings),
                    len(skipped_symbols),
                    progress_state,
                )
            except Exception as exc:
                logger.exception("Feature run %d failed", run_id)
                try:
                    uow.rollback()
                    current_run = uow.feature_runs.get_run(run_id)
                    if current_run.status == RunStatus.RUNNING:
                        attempted_symbols = min(
                            progress_state.attempted_symbols,
                            total,
                        )
                        failed_symbols = min(
                            progress_state.failed_symbols,
                            attempted_symbols,
                        )
                        stats = RunStats(
                            total_symbols=total,
                            processed_symbols=max(
                                attempted_symbols - failed_symbols,
                                0,
                            ),
                            failed_symbols=failed_symbols,
                            duration_seconds=round(
                                time.monotonic() - run_started_at,
                                2,
                            ),
                            passed_symbols=None,
                        )
                        uow.feature_runs.mark_failed(
                            run_id,
                            stats,
                            warnings=(
                                f"Snapshot build failed: {exc.__class__.__name__}: {exc}",
                            ),
                        )
                        uow.commit()
                except Exception:
                    logger.exception(
                        "Best-effort failure transition failed for feature run %d",
                        run_id,
                    )
                raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        uow: UnitOfWork,
        run_id: int,
        cmd: BuildDailySnapshotCommand,
        progress: ProgressSink,
        cancel: CancellationToken,
        symbols: list[str],
        run_warnings: tuple[str, ...],
        skipped_symbols: int,
        progress_state: _SnapshotRunProgress,
    ) -> BuildDailySnapshotResult:
        start_time = time.monotonic()
        effective_chunk_size = (
            cmd.static_chunk_size
            if cmd.require_bulk_prefetch and cmd.static_chunk_size is not None
            else cmd.chunk_size
        )

        # ── 1. Resolve universe ─────────────────────────────────
        total = len(symbols)
        logger.info("Resolved %d symbols for run %d", total, run_id)

        # ── 2. Save universe snapshot ───────────────────────────
        uow.feature_store.save_run_universe_symbols(run_id, symbols)
        uow.commit()

        # ── 3. Scan symbols in chunks ───────────────────────────
        all_rows: list[FeatureRowWrite] = []
        merged_requirements = None
        bulk_prefetch_enabled = self._data_provider is not None

        if cmd.require_bulk_prefetch and not bulk_prefetch_enabled:
            raise RuntimeError(
                "Static daily snapshot requires a bulk-capable data provider"
            )

        if bulk_prefetch_enabled:
            get_requirements = getattr(self._scanner, "get_merged_requirements", None)
            if callable(get_requirements):
                try:
                    merged_requirements = get_requirements(
                        cmd.screener_names,
                        cmd.criteria,
                    )
                    logger.info(
                        "Run %d: pre-merged data requirements for batch processing",
                        run_id,
                    )
                except Exception as exc:
                    if cmd.require_bulk_prefetch:
                        raise RuntimeError(
                            "Static daily snapshot failed to prepare merged bulk requirements"
                        ) from exc
                    logger.warning(
                        "Run %d: failed to pre-merge data requirements; "
                        "falling back to per-symbol fetch",
                        run_id,
                        exc_info=True,
                    )
                    bulk_prefetch_enabled = False
            else:
                if cmd.require_bulk_prefetch:
                    raise RuntimeError(
                        "Static daily snapshot requires scanner bulk requirements support"
                    )
                bulk_prefetch_enabled = False

        for chunk in _chunked(symbols, effective_chunk_size):
            # 3a — Cancellation gate
            if cancel.is_cancelled():
                duration = time.monotonic() - start_time
                stats = RunStats(
                    total_symbols=total,
                    processed_symbols=(
                        progress_state.attempted_symbols
                        - progress_state.failed_symbols
                    ),
                    failed_symbols=progress_state.failed_symbols,
                    duration_seconds=round(duration, 2),
                    passed_symbols=progress_state.passed_symbols,
                )
                uow.feature_runs.mark_completed(
                    run_id, stats, warnings=("Cancelled by user",)
                )
                uow.commit()
                logger.info(
                    "Run %d cancelled at %d/%d",
                    run_id,
                    progress_state.attempted_symbols,
                    total,
                )
                return BuildDailySnapshotResult(
                    run_id=run_id,
                    status=RunStatus.COMPLETED.value,
                    total_symbols=total,
                    processed_symbols=progress_state.attempted_symbols,
                    failed_symbols=progress_state.failed_symbols,
                    skipped_symbols=skipped_symbols,
                    dq_passed=False,
                    warnings=(*run_warnings, "Cancelled by user"),
                    row_count=uow.feature_store.count_by_run_id(run_id),
                    duration_seconds=round(duration, 2),
                )

            # 3b — Scan each symbol in the chunk
            pre_fetched_data: dict[str, object] = {}
            cache_only_symbol_data = (
                cmd.batch_only_prices
                or cmd.batch_only_fundamentals
                or cmd.require_bulk_prefetch
            )
            if bulk_prefetch_enabled and merged_requirements is not None:
                try:
                    pre_fetched_data = self._data_provider.prepare_data_bulk(
                        [symbol.upper() for symbol in chunk],
                        merged_requirements,
                        allow_partial=True,
                        batch_only_prices=cmd.batch_only_prices,
                        batch_only_fundamentals=cmd.batch_only_fundamentals,
                    )
                except Exception as exc:
                    if cmd.require_bulk_prefetch:
                        raise RuntimeError(
                            "Cache-only snapshot bulk prefetch failed; refusing per-symbol fallback"
                        ) from exc
                    logger.warning(
                        "Run %d: bulk data fetch failed for chunk; "
                        "falling back to per-symbol fetch",
                        run_id,
                        exc_info=True,
                    )
                    bulk_prefetch_enabled = False
                    pre_fetched_data = {}

            chunk_rows: list[FeatureRowWrite] = []
            def _scan_symbol(symbol: str) -> tuple[str, FeatureRowWrite | None, bool]:
                sym = symbol.upper()
                try:
                    if cache_only_symbol_data and sym not in pre_fetched_data:
                        return sym, None, False
                    scan_kwargs: dict[str, object] = {}
                    if merged_requirements is not None:
                        scan_kwargs["pre_merged_requirements"] = merged_requirements
                    if sym in pre_fetched_data:
                        scan_kwargs["pre_fetched_data"] = pre_fetched_data[sym]
                    result = self._scanner.scan_stock_multi(
                        symbol=sym,
                        screener_names=cmd.screener_names,
                        criteria=cmd.criteria,
                        composite_method=cmd.composite_method,
                        **scan_kwargs,
                    )
                    result_status = _resolve_result_status(result)
                    if result and result_status != "error":
                        row = _map_orchestrator_to_feature_row(
                            sym, cmd.as_of_date, result
                        )
                        return sym, row, bool(result.get("passes_template"))
                    return sym, None, False
                except Exception:
                    logger.debug(
                        "Error scanning %s in run %d",
                        sym,
                        run_id,
                        exc_info=True,
                    )
                    return sym, None, False

            outcomes_by_symbol: dict[str, tuple[FeatureRowWrite | None, bool]] = {}
            if (
                cmd.require_bulk_prefetch
                and cmd.static_parallel_workers > 1
                and len(chunk) > 1
            ):
                max_workers = min(cmd.static_parallel_workers, len(chunk))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_scan_symbol, symbol): symbol for symbol in chunk
                    }
                    for future in as_completed(futures):
                        sym, row, passed = future.result()
                        outcomes_by_symbol[sym] = (row, passed)
            else:
                for symbol in chunk:
                    sym, row, passed = _scan_symbol(symbol)
                    outcomes_by_symbol[sym] = (row, passed)

            for symbol in chunk:
                row, passed = outcomes_by_symbol.get(symbol.upper(), (None, False))
                if row is not None:
                    chunk_rows.append(row)
                    if passed:
                        progress_state.passed_symbols += 1
                else:
                    progress_state.failed_symbols += 1
                progress_state.attempted_symbols += 1

            # 3c — Persist chunk (checkpoint)
            if chunk_rows:
                uow.feature_store.upsert_snapshot_rows(run_id, chunk_rows)
                all_rows.extend(chunk_rows)
            uow.commit()

            # 3d — Progress reporting
            elapsed = time.monotonic() - start_time
            throughput = (
                progress_state.attempted_symbols / elapsed if elapsed > 0 else 0.0
            )
            remaining = total - progress_state.attempted_symbols
            eta = remaining / throughput if throughput > 0 else None

            progress.emit(
                ProgressEvent(
                    current=progress_state.attempted_symbols,
                    total=total,
                    passed=progress_state.passed_symbols,
                    failed=progress_state.failed_symbols,
                    throughput=(
                        round(throughput, 2) if throughput > 0 else None
                    ),
                    eta_seconds=round(eta) if eta is not None else None,
                )
            )

        # ── 4. Mark completed ───────────────────────────────────
        duration = time.monotonic() - start_time
        stats = RunStats(
            total_symbols=total,
            processed_symbols=(
                progress_state.attempted_symbols - progress_state.failed_symbols
            ),
            failed_symbols=progress_state.failed_symbols,
            duration_seconds=round(duration, 2),
            passed_symbols=progress_state.passed_symbols,
        )
        uow.feature_runs.mark_completed(run_id, stats)
        uow.commit()
        logger.info(
            "Run %d completed: %d processed, %d failed, %.1fs",
            run_id,
            progress_state.attempted_symbols,
            progress_state.failed_symbols,
            duration,
        )

        # ── 5. Delegate DQ + publish to PublishFeatureRunUseCase ──
        actual_count = uow.feature_store.count_by_run_id(run_id)
        nulls = sum(1 for r in all_rows if r.composite_score is None)
        scores = tuple(
            r.composite_score for r in all_rows if r.composite_score is not None
        )
        ratings = tuple(
            r.overall_rating for r in all_rows if r.overall_rating is not None
        )
        result_syms = tuple(r.symbol for r in all_rows)

        dq_inputs = DQInputs(
            expected_row_count=total,
            actual_row_count=actual_count,
            null_score_count=nulls,
            total_row_count=len(all_rows),
            scores=scores,
            ratings=ratings,
            universe_symbols=tuple(symbols),
            result_symbols=result_syms,
        )

        publish_cmd = PublishRunCommand(
            run_id=run_id,
            dq_inputs=dq_inputs,
            pointer_key=cmd.publish_pointer_key,
            dq_thresholds=cmd.dq_thresholds,
        )
        pub_result = PublishFeatureRunUseCase().execute(uow, publish_cmd)

        # ── 6. Completion progress ─────────────────────────────
        progress.emit(
            ProgressEvent(
                current=total,
                total=total,
                passed=progress_state.passed_symbols,
                failed=progress_state.failed_symbols,
            )
        )

        return BuildDailySnapshotResult(
            run_id=run_id,
            status=pub_result.status,
            total_symbols=total,
            processed_symbols=progress_state.attempted_symbols,
            failed_symbols=progress_state.failed_symbols,
            skipped_symbols=skipped_symbols,
            dq_passed=pub_result.dq_passed,
            warnings=(*run_warnings, *pub_result.warnings),
            row_count=actual_count,
            duration_seconds=round(duration, 2),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BootstrapCacheCoverageInsufficient",
    "BuildDailyFeatureSnapshotUseCase",
    "BuildDailySnapshotCommand",
    "BuildDailySnapshotResult",
]
