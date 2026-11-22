"""RunBulkScanUseCase — orchestrates the full scan execution lifecycle.

This use case contains the business rules for executing a bulk scan:
  1. Load scan record and determine checkpoint (resume support)
  2. Mark scan as running
  3. Submit each chunk of symbols to the batch runner; while it computes,
     collect the previous chunk, persist it (acts as checkpoint) and report
     progress via ProgressSink
  4. Check CancellationToken before the first chunk and after each persisted
     chunk except the last; a chunk already submitted when the scan is
     cancelled is dropped
  5. Handle completion / failure status transitions

The use case depends ONLY on domain ports — never on SQLAlchemy, Celery,
Redis, or any other infrastructure.

Zero Celery imports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator, Sequence

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.uow import UnitOfWork
from app.domain.markets.symbol_suffixes import market_symbol_suffix_registry
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.domain.scanning.models import ProgressEvent, ScanStatus
from app.domain.scanning.ports import (
    CancellationToken,
    MarketRsReader,
    MarketRsResolution,
    PendingScanBatch,
    ProgressSink,
    SerialStockScanBatchRunner,
    StockDataProvider,
    StockScanBatchRunner,
    StockScanBatchRunnerFactory,
    StockScanCall,
    StockScanner,
    StockScanOutcome,
    serial_stock_scan_batch_runner,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunked(seq: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield successive chunks of *size* from *seq*."""
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _should_persist_result(result: object) -> bool:
    """Persist visible result rows; skip only hard errors."""
    if not isinstance(result, dict):
        return False

    explicit_status = result.get("result_status")
    if explicit_status in {"ok", "insufficient_history"}:
        return True
    if explicit_status == "error":
        return False

    if "error" in result:
        return False

    return result.get("rating") != "Insufficient Data"


# ── Command (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunBulkScanCommand:
    """Immutable value object describing the scan to execute.

    ``parallel_workers`` is how many processes may compute a chunk's scan
    results. Only chunks whose data is fully prefetched (always true for
    cache-only scans) are spread across them; 1 computes in-process.
    """

    scan_id: str
    symbols: list[str]
    criteria: dict = field(default_factory=dict)
    chunk_size: int = 50
    correlation_id: str | None = None
    cache_only: bool = False
    parallel_workers: int = 1

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")


# ── Result (output) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunBulkScanResult:
    """What the use case returns to the caller."""

    scan_id: str
    status: str  # completed | cancelled | failed
    total_scanned: int
    passed: int
    failed: int


# ── Use Case ─────────────────────────────────────────────────────────────


class RunBulkScanUseCase:
    """Execute a bulk stock scan: load → run → persist → progress → done.

    The constructor accepts infrastructure collaborators through ports.
    ``execute()`` receives a fresh UoW per invocation (same pattern as
    :class:`CreateScanUseCase`).
    """

    def __init__(
        self,
        scanner: StockScanner,
        data_provider: StockDataProvider | None = None,
        market_rs_reader: MarketRsReader | None = None,
        scan_batch_runner_factory: StockScanBatchRunnerFactory = (
            serial_stock_scan_batch_runner
        ),
    ) -> None:
        self._scanner = scanner
        self._data_provider = data_provider
        self._market_rs_reader = market_rs_reader
        self._scan_batch_runner_factory = scan_batch_runner_factory

    def execute(
        self,
        uow: UnitOfWork,
        cmd: RunBulkScanCommand,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> RunBulkScanResult:
        """Run the full scan lifecycle inside a single UoW."""
        with uow:
            # ── Load scan record ──────────────────────────────────────
            scan = uow.scans.get_by_scan_id(cmd.scan_id)
            if scan is None:
                raise EntityNotFoundError("Scan", cmd.scan_id)

            try:
                return self._run(uow, scan, cmd, progress, cancel)
            except Exception:
                # Best-effort: mark as failed so the UI knows
                try:
                    uow.rollback()
                    uow.scans.update_status(
                        cmd.scan_id, ScanStatus.FAILED.value
                    )
                    uow.commit()
                except Exception:
                    logger.warning(
                        "Could not mark scan %s as failed", cmd.scan_id, exc_info=True
                    )
                raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        uow: UnitOfWork,
        scan: object,
        cmd: RunBulkScanCommand,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> RunBulkScanResult:
        total = len(cmd.symbols)
        screener_names: list[str] = getattr(scan, "screener_types", None) or [
            "minervini"
        ]
        composite_method: str = (
            getattr(scan, "composite_method", None) or "weighted_average"
        )

        merged_requirements = None
        if self._data_provider is not None:
            get_requirements = getattr(self._scanner, "get_merged_requirements", None)
            if callable(get_requirements):
                try:
                    merged_requirements = get_requirements(screener_names, cmd.criteria)
                    logger.info(
                        "Scan %s: pre-merged data requirements for batch processing",
                        cmd.scan_id,
                    )
                except Exception:
                    logger.warning(
                        "Scan %s: failed to pre-merge data requirements; falling back to per-symbol fetch",
                        cmd.scan_id,
                        exc_info=True,
                    )

        # ── Checkpoint: skip already-persisted results ────────────
        already_done = uow.scan_results.count_by_scan_id(cmd.scan_id)
        remaining_symbols = cmd.symbols[already_done:]
        default_market = str(
            getattr(scan, "universe_market", None) or "US"
        ).strip().upper()
        rs_publication_by_market: dict[
            str,
            tuple[str, int | None, date | None],
        ] = {}

        if already_done > 0:
            logger.info(
                "Resuming scan %s from checkpoint %d/%d",
                cmd.scan_id,
                already_done,
                total,
            )
            if self._market_rs_reader is not None:
                audits = uow.scan_results.list_rs_audits_by_scan_id(cmd.scan_id)
                if len(audits) != already_done:
                    raise RuntimeError(
                        f"Cannot resume scan {cmd.scan_id}: persisted result count "
                        "does not match its Market RS audit rows"
                    )
                for audit in audits:
                    if not audit.formula_version:
                        raise RuntimeError(
                            f"Cannot resume scan {cmd.scan_id}: {audit.symbol} has "
                            "no persisted Market RS publication identity"
                        )
                    if (
                        audit.formula_version == BALANCED_RS_FORMULA_VERSION
                        and audit.run_id is None
                    ):
                        raise RuntimeError(
                            f"Cannot resume scan {cmd.scan_id}: {audit.symbol} has "
                            "no persisted canonical Market RS run ID"
                        )
                    market = (
                        market_symbol_suffix_registry.market_for_symbol(audit.symbol)
                        or default_market
                    ).strip().upper()
                    publication = (
                        audit.formula_version,
                        audit.run_id,
                        None,
                    )
                    pinned = rs_publication_by_market.get(market)
                    if pinned is not None and pinned[:2] != publication[:2]:
                        raise RuntimeError(
                            "Cannot resume scan "
                            f"{cmd.scan_id}: persisted Market RS publications "
                            f"disagree for {market}: {pinned[:2]!r} and "
                            f"{publication[:2]!r}"
                        )
                    rs_publication_by_market[market] = publication

        # ── Mark running ──────────────────────────────────────────
        uow.scans.update_status(
            cmd.scan_id,
            ScanStatus.RUNNING.value,
            total_stocks=total,
        )
        uow.commit()

        # ── Process chunks ────────────────────────────────────────
        with self._scan_batch_runner_factory(
            self._scanner,
            cmd.parallel_workers if len(remaining_symbols) > 1 else 1,
        ) as compute_runner:
            return self._process_chunks(
                uow,
                scan,
                cmd,
                progress,
                cancel,
                remaining_symbols=remaining_symbols,
                already_done=already_done,
                total=total,
                screener_names=screener_names,
                composite_method=composite_method,
                merged_requirements=merged_requirements,
                default_market=default_market,
                rs_publication_by_market=rs_publication_by_market,
                compute_runner=compute_runner,
            )

    def _process_chunks(
        self,
        uow: UnitOfWork,
        scan: object,
        cmd: RunBulkScanCommand,
        progress: ProgressSink,
        cancel: CancellationToken,
        *,
        remaining_symbols: Sequence[str],
        already_done: int,
        total: int,
        screener_names: list[str],
        composite_method: str,
        merged_requirements: object | None,
        default_market: str,
        rs_publication_by_market: dict[str, tuple[str, int | None, date | None]],
        compute_runner: StockScanBatchRunner,
    ) -> RunBulkScanResult:
        in_process_runner = SerialStockScanBatchRunner(self._scanner)
        processed = already_done
        # Restore passed count from the scan record so resume doesn't
        # lose previously-accumulated passes.
        passed = getattr(scan, "passed_stocks", 0) or 0
        failed = 0
        start_time = time.monotonic()

        def _interpret(
            sym: str,
            outcome: StockScanOutcome,
        ) -> tuple[dict | None, bool, bool]:
            if outcome.error is not None:
                logger.debug(
                    "Error scanning %s in scan %s",
                    sym,
                    cmd.scan_id,
                    exc_info=outcome.error,
                )
                return (None, False, False)
            result = outcome.result
            passed_flag = (
                bool(result.get("passes_template"))
                if isinstance(result, dict)
                else False
            )
            return (result, passed_flag, _should_persist_result(result))

        def _finish(
            calls: list[StockScanCall | None],
            batch: PendingScanBatch,
        ) -> None:
            """Collect a submitted chunk, persist it (the checkpoint), report progress."""
            nonlocal processed, passed, failed
            chunk_results: list[tuple[str, dict]] = []
            scan_outcomes = iter(batch())
            for call in calls:
                if call is None:
                    failed += 1
                    processed += 1
                    continue
                result, passed_flag, persistable = _interpret(
                    call.symbol,
                    next(scan_outcomes),
                )
                if persistable and result is not None:
                    chunk_results.append((call.symbol, result))
                    if passed_flag:
                        passed += 1
                else:
                    failed += 1
                processed += 1

            if chunk_results:
                uow.scan_results.persist_orchestrator_results(
                    cmd.scan_id, chunk_results
                )
            uow.commit()

            elapsed = time.monotonic() - start_time
            throughput = (
                (processed - already_done) / elapsed if elapsed > 0 else 0.0
            )
            remaining = total - processed
            eta = remaining / throughput if throughput > 0 else None

            progress.emit(
                ProgressEvent(
                    current=processed,
                    total=total,
                    passed=passed,
                    failed=failed,
                    throughput=round(throughput, 2) if throughput > 0 else None,
                    eta_seconds=round(eta) if eta is not None else None,
                )
            )

        def _cancelled() -> RunBulkScanResult:
            uow.scans.update_status(
                cmd.scan_id,
                ScanStatus.CANCELLED.value,
                passed_stocks=passed,
            )
            uow.commit()
            logger.info(
                "Scan %s cancelled at %d/%d", cmd.scan_id, processed, total
            )
            return RunBulkScanResult(
                scan_id=cmd.scan_id,
                status=ScanStatus.CANCELLED.value,
                total_scanned=processed,
                passed=passed,
                failed=failed,
            )

        if cancel.is_cancelled():
            return _cancelled()

        # The chunk submitted last computes while the parent finishes the one
        # before it. Commits stay in chunk order, so the resume checkpoint holds.
        # A cancel drops the batch submitted just before it: the runner's exit
        # cancels its queued work and waits for the part already dispatched.
        pending: tuple[list[StockScanCall | None], PendingScanBatch] | None = None
        for chunk in _chunked(remaining_symbols, cmd.chunk_size):
            pre_fetched_data = {}
            if self._data_provider is not None and merged_requirements is not None:
                try:
                    pre_fetched_data = self._data_provider.prepare_data_bulk(
                        [symbol.upper() for symbol in chunk],
                        merged_requirements,
                        allow_partial=True,
                        batch_only_prices=cmd.cache_only,
                        batch_only_fundamentals=cmd.cache_only,
                        cache_only=cmd.cache_only,
                    )
                except Exception:
                    logger.warning(
                        "Scan %s: bulk data fetch failed for chunk; falling back to per-symbol fetch",
                        cmd.scan_id,
                        exc_info=True,
                    )
                    pre_fetched_data = {}

            chunk_rs_resolution_by_symbol: dict[str, MarketRsResolution] = {}
            if self._market_rs_reader is not None:
                symbols_by_market: dict[str, list[str]] = {}
                for symbol in chunk:
                    normalized_symbol = str(symbol).upper()
                    stock_data = pre_fetched_data.get(normalized_symbol)
                    stock_market = str(
                        getattr(stock_data, "market", None) or ""
                    ).strip().upper()
                    data_market = stock_market or (
                        market_symbol_suffix_registry.market_for_symbol(
                            normalized_symbol
                        )
                        or default_market
                    ).strip().upper()
                    symbols_by_market.setdefault(data_market, []).append(
                        normalized_symbol
                    )
                for data_market, market_symbols in symbols_by_market.items():
                    pinned_publication = rs_publication_by_market.get(data_market)
                    resolution = self._market_rs_reader.get(
                        market=data_market,
                        symbols=tuple(market_symbols),
                        as_of_date=(
                            pinned_publication[2]
                            if pinned_publication is not None
                            else None
                        ),
                        formula_version=(
                            pinned_publication[0]
                            if pinned_publication is not None
                            else None
                        ),
                        run_id=(
                            pinned_publication[1]
                            if pinned_publication is not None
                            else None
                        ),
                    )
                    resolved_publication = (
                        resolution.formula_version,
                        resolution.run_id,
                        resolution.as_of_date,
                    )
                    publication_changed = pinned_publication is not None and (
                        resolved_publication[:2] != pinned_publication[:2]
                        or (
                            pinned_publication[2] is not None
                            and resolved_publication[2] != pinned_publication[2]
                        )
                    )
                    if publication_changed:
                        raise RuntimeError(
                            f"Market RS publication changed for {data_market} "
                            f"during scan {cmd.scan_id}: expected "
                            f"{pinned_publication!r}, got {resolved_publication!r}"
                        )
                    if pinned_publication is None or pinned_publication[2] is None:
                        rs_publication_by_market[data_market] = resolved_publication
                    market_data = {
                        symbol: pre_fetched_data[symbol]
                        for symbol in market_symbols
                        if symbol in pre_fetched_data
                    }
                    if self._data_provider is not None and market_data:
                        self._data_provider.apply_market_rs_resolution(
                            market_data,
                            resolution,
                        )
                    for symbol in market_symbols:
                        chunk_rs_resolution_by_symbol[symbol] = resolution

            def _call_for(symbol: str) -> StockScanCall | None:
                sym = symbol.upper()
                # cache_only invariant: the bulk prefetch is the only sanctioned
                # data source. Any path that leaves a symbol out of pre_fetched_data
                # (bulk raised, merged_requirements is None, data provider is None)
                # must not fall through to per-symbol prepare_data() which can still
                # hit live APIs.
                if cmd.cache_only and sym not in pre_fetched_data:
                    return None
                scan_kwargs: dict[str, object] = {
                    "screener_names": screener_names,
                    "criteria": cmd.criteria,
                    "composite_method": composite_method,
                }
                if merged_requirements is not None:
                    scan_kwargs["pre_merged_requirements"] = merged_requirements
                if (
                    sym not in pre_fetched_data
                    and sym in chunk_rs_resolution_by_symbol
                ):
                    scan_kwargs["market_rs_resolution"] = (
                        chunk_rs_resolution_by_symbol[sym]
                    )
                if sym in pre_fetched_data:
                    scan_kwargs["pre_fetched_data"] = pre_fetched_data[sym]
                return StockScanCall(symbol=sym, kwargs=scan_kwargs)

            calls = [_call_for(symbol) for symbol in chunk]
            prefetch_covers_chunk = bool(pre_fetched_data) and all(
                symbol.upper() in pre_fetched_data for symbol in chunk
            )
            # Symbols outside the prefetch fetch their own data (live
            # providers); keep those serial so they respect rate limits.
            runner = (
                compute_runner
                if cmd.cache_only or prefetch_covers_chunk
                else in_process_runner
            )
            submitted = (
                calls,
                runner.submit_batch([call for call in calls if call is not None]),
            )
            if pending is not None:
                _finish(*pending)
                if cancel.is_cancelled():
                    return _cancelled()
            pending = submitted

        if pending is not None:
            _finish(*pending)

        # ── Completed ─────────────────────────────────────────────
        uow.scans.update_status(
            cmd.scan_id,
            ScanStatus.COMPLETED.value,
            passed_stocks=passed,
        )
        uow.commit()

        logger.info(
            "Scan %s completed: %d scanned, %d passed, %d failed",
            cmd.scan_id,
            processed,
            passed,
            failed,
        )

        return RunBulkScanResult(
            scan_id=cmd.scan_id,
            status=ScanStatus.COMPLETED.value,
            total_scanned=processed,
            passed=passed,
            failed=failed,
        )
