"""CreateScanUseCase — orchestrates scan creation with idempotency.

This use case owns the business rules for creating a new scan:
  1. Check idempotency key (return existing scan if duplicate)
  2. Resolve universe symbols via the UniverseRepository port
  3. Persist the Scan record via ScanRepository
  4. Dispatch the background scan task via TaskDispatcher

The use case depends ONLY on domain ports (abstract interfaces) —
never on SQLAlchemy, Celery, or any other infrastructure.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.custom_criteria_compiler import (
    CompiledCustomCriteria,
    compile_custom_criteria,
)
from app.domain.scanning.errors import SingleActiveScanViolation
from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
    hash_universe_symbols,
)
from app.schemas.universe import UniverseType
from app.domain.scanning.ports import TaskDispatcher

FreshnessChecker = Callable[[Iterable[str]], Optional[dict]]

logger = logging.getLogger(__name__)


# Per-screener scores belonging to screeners the user didn't request.
# The covering snapshot may have computed them, but they have no meaning
# in this scan's context and would clutter result columns / sort options.
_COMPILE_PATH_DROP_KEYS: tuple[str, ...] = (
    "minervini_score",
    "canslim_score",
    "ipo_score",
    "volume_breakthrough_score",
    "composite_reason",
)


def _normalize_compile_details(details: dict) -> dict:
    """Strip stale-criteria score fields from a compile-path row's details.

    Called per row before passing through ``persist_orchestrator_results``.
    The compile path is reached only when the exact-signature lookup
    missed, so ``custom_score`` / ``composite_score`` / ``rating`` /
    ``passes_template`` from the covering snapshot reflect *some other*
    criteria set; persisting them as-is would misrank matches in the UI.
    Replace those fields with values that describe what the compile path
    actually guarantees: every persisted row passed every user-specified
    per-field filter at the SQL gate.

    Per-symbol *facts* (price, RS rating, MA alignment, sector, growth,
    sparklines, setup-engine outputs) are left untouched — they're inputs
    to the user's filter, not derived from custom criteria — so users keep
    seeing the snapshot's factual columns; only the score / rating /
    pass-template metadata gets normalised.

    The override dict is constructed fresh inside this function so that
    mutable values (notably ``screeners_run``'s list) are not shared by
    reference across rows.
    """
    if isinstance(details, dict):
        normalized = dict(details)
    else:
        normalized = {}
    for key in _COMPILE_PATH_DROP_KEYS:
        normalized.pop(key, None)
    normalized.update({
        "custom_score": 100.0,
        "composite_score": 100.0,
        "rating": "Strong Buy",
        "passes_template": True,
        # Fresh list per call — never shared between rows or invocations.
        "screeners_run": ["custom"],
        "screeners_passed": 1,
        "screeners_total": 1,
        "composite_method": "weighted_average",
    })
    return normalized


# ── Command (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CreateScanCommand:
    """Immutable value object describing what the caller wants to create."""

    # Opaque universe definition — passed through to UniverseRepository
    universe_def: object

    # Pre-computed universe metadata (router extracts from UniverseDefinition)
    universe_label: str
    universe_key: str
    universe_type: str
    universe_market: str | None = None
    universe_exchange: str | None = None
    universe_index: str | None = None
    universe_symbols: list[str] | None = None

    # Screener configuration
    screeners: list[str] = field(default_factory=lambda: ["minervini"])
    composite_method: str = "weighted_average"
    criteria: dict | None = None

    # Idempotency
    idempotency_key: str | None = None

    # Provenance — determines cache_only policy in the runner.
    # Manual scans must run cache-only (no yfinance/Finviz fallback); bootstrap
    # and other internal scans populate the cache and therefore must allow
    # live fetch. See scan_tasks._run_bulk_scan_via_use_case.
    trigger_source: str = "manual"


# ── Result (output) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CreateScanResult:
    """What the use case returns to the caller."""

    scan_id: str
    status: str
    total_stocks: int
    is_duplicate: bool
    feature_run_id: int | None = None


@dataclass(frozen=True)
class ActiveScanConflict:
    """Structured metadata describing the active scan blocking a new request."""

    scan_id: str
    status: str
    trigger_source: str
    total_stocks: int
    started_at: datetime | None


class ActiveScanConflictError(RuntimeError):
    """Raised when a user tries to queue a second scan while one is active."""

    def __init__(self, active_scan: ActiveScanConflict) -> None:
        super().__init__(f"Scan {active_scan.scan_id} is already {active_scan.status}")
        self.active_scan = active_scan

    def to_dict(self) -> dict[str, object]:
        return {
            "code": "scan_already_active",
            "message": "Another scan is already queued or running.",
            "active_scan": {
                "scan_id": self.active_scan.scan_id,
                "status": self.active_scan.status,
                "trigger_source": self.active_scan.trigger_source,
                "total_stocks": self.active_scan.total_stocks,
                "started_at": (
                    self.active_scan.started_at.isoformat()
                    if self.active_scan.started_at is not None
                    else None
                ),
            },
        }


class StaleMarketDataError(RuntimeError):
    """Raised when the resolved universe includes symbols with stale cached prices.

    Runs inside the use case AFTER idempotency resolution and universe symbol
    resolution, so:
      - idempotent retries return the existing scan without being blocked by
        current market freshness
      - the check is scoped to the actual scan universe, not every market-wide
        symbol
    """

    def __init__(self, detail: dict) -> None:
        super().__init__(detail.get("message", "market data is stale"))
        self.detail = detail

    def to_dict(self) -> dict:
        return dict(self.detail)


# ── Use Case ─────────────────────────────────────────────────────────────


class CreateScanUseCase:
    """Create a scan record, resolve its universe, and dispatch execution."""

    def __init__(
        self,
        dispatcher: TaskDispatcher,
        *,
        freshness_checker: FreshnessChecker | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._freshness_checker = freshness_checker

    def _attempt_compile_path(
        self,
        uow: UnitOfWork,
        cmd: CreateScanCommand,
        symbols: list[str],
    ) -> tuple[object, list[tuple[str, dict]]] | None:
        """Try to serve the scan from a published feature run via compiled criteria.

        Returns ``(run, results)`` on success — caller persists *results*
        as scan_results and marks the scan completed. Returns ``None`` to
        signal "fall back to async chunked compute" for any reason
        (criteria not fully representable, no covering run, infra error).

        Failures are logged but never raised so a misconfigured feature
        store can never block scan creation.

        Correctness note — score gate. By definition this path runs only
        when the exact-signature lookup missed, so the covering run was
        produced under different custom-screener criteria. ``custom_score``
        in the stored details was computed against *that* run's filters,
        not the user's, so reusing it as a pass gate would yield results
        that disagree with async. We therefore only use the compile path
        when the compiler can prove the SQL hard gate is equivalent to
        CustomScanner's weighted ``min_score`` pass semantics.
        """
        try:
            compiled: CompiledCustomCriteria = compile_custom_criteria(
                cmd.criteria,
                screeners=cmd.screeners,
                universe_market=cmd.universe_market,
            )
        except Exception:
            logger.warning(
                "Criteria compile raised — falling back to async path",
                exc_info=True,
            )
            return None

        if not compiled.is_fully_representable:
            logger.debug(
                "Compile path skipped: %d unrepresentable criteria keys",
                len(compiled.unrepresentable_keys),
            )
            return None
        if compiled.score_field is None:
            # ``score_field`` is set only for ``screeners == ["custom"]``.
            # Multi-screener composites would need extra logic to derive
            # a consistent pass test; defer those to the async path.
            return None
        if not compiled.hard_gate_equivalent:
            logger.debug(
                "Compile path skipped: criteria are not hard-gate equivalent",
            )
            return None
        if not compiled.filter_spec.range_filters \
                and not compiled.filter_spec.categorical_filters \
                and not compiled.filter_spec.boolean_filters:
            # Without per-field thresholds the compile path would return
            # the entire covering universe — meaningless as a "scan".
            # Defer to async so the scanner produces real scores.
            return None

        try:
            run = uow.feature_runs.find_latest_published_covering(
                symbols=symbols,
                market=cmd.universe_market,
            )
        except Exception:
            logger.warning(
                "Covering feature run lookup failed — falling back to async path",
                exc_info=True,
            )
            return None

        if run is None:
            logger.debug(
                "Compile path skipped: no covering published feature run for market=%s",
                cmd.universe_market,
            )
            return None

        try:
            results = uow.feature_store.query_run_details(
                run.id,
                compiled.filter_spec,
                symbols=symbols,
            )
        except Exception:
            logger.warning(
                "Feature-store details query failed for run %s — falling back to async path",
                run.id,
                exc_info=True,
            )
            return None

        return run, results

    @staticmethod
    def _raise_active_scan_conflict(active_scan: object) -> None:
        raise ActiveScanConflictError(
            ActiveScanConflict(
                scan_id=active_scan.scan_id,
                status=active_scan.status,
                trigger_source=getattr(active_scan, "trigger_source", "manual") or "manual",
                total_stocks=active_scan.total_stocks or 0,
                started_at=getattr(active_scan, "started_at", None),
            )
        )

    def execute(self, uow: UnitOfWork, cmd: CreateScanCommand) -> CreateScanResult:
        """Run the use case.

        Opens a UoW context, persists the Scan, commits (so the Celery
        worker can read it), dispatches the task, then stores the task ID.
        """
        with uow:
            # ── Idempotency check ────────────────────────────────────
            if cmd.idempotency_key is not None:
                existing = uow.scans.get_by_idempotency_key(cmd.idempotency_key)
                if existing is not None:
                    return CreateScanResult(
                        scan_id=existing.scan_id,
                        status=existing.status,
                        total_stocks=existing.total_stocks or 0,
                        is_duplicate=True,
                        feature_run_id=getattr(existing, "feature_run_id", None),
                    )

            active_scan = uow.scans.get_active_scan()
            if active_scan is not None:
                self._raise_active_scan_conflict(active_scan)

            # ── Resolve universe symbols ─────────────────────────────
            symbols = uow.universe.resolve_symbols(cmd.universe_def)
            if not symbols:
                raise ValidationError(
                    f"No symbols found for universe '{cmd.universe_label}'. "
                    "Try refreshing the universe."
                )

            # ── Staleness gate (scoped to resolved symbols) ──────────
            # Runs after idempotency + active-scan + symbol resolution so:
            #   - duplicate idempotent retries return the existing scan above
            #   - 'all'-universe scans get checked across every resolved market
            #   - unrelated market-wide symbol issues don't block narrow scans
            if self._freshness_checker is not None:
                staleness_detail = self._freshness_checker(symbols)
                if staleness_detail is not None:
                    raise StaleMarketDataError(staleness_detail)

            feature_run_id = None
            instant_match = None
            compile_outcome: tuple[object, list[tuple[str, dict]]] | None = None
            should_attempt_instant = cmd.universe_type in {
                UniverseType.ALL.value,
                UniverseType.MARKET.value,
            }
            signature_payload = build_scan_signature_payload(
                universe_type=cmd.universe_type,
                screeners=cmd.screeners,
                composite_method=cmd.composite_method,
                criteria=cmd.criteria,
            )
            input_hash = hash_scan_signature(signature_payload)
            universe_hash = hash_universe_symbols(symbols)

            if should_attempt_instant:
                try:
                    instant_match = uow.feature_runs.find_latest_published_exact(
                        input_hash=input_hash,
                        universe_hash=universe_hash,
                    )
                    if instant_match is not None:
                        feature_run_id = instant_match.id
                        logger.info(
                            "Creating instant snapshot-backed scan from feature run %d",
                            instant_match.id,
                        )
                    else:
                        logger.info("No exact published feature run match — checking compile path")
                except Exception:
                    logger.warning(
                        "Exact feature run lookup failed — proceeding to compile path",
                        exc_info=True,
                    )

            # Feature-store-first compile path: when no exact match exists
            # but the criteria compile cleanly into queryable fields, serve
            # the scan from a published feature run as a single SQL query
            # instead of dispatching async chunked compute.
            #
            # Restricted to ALL / MARKET universes so the mixed-market vs
            # single-market policy that drives volume/market-cap unit
            # semantics is unambiguous: ALL is mixed-market, MARKET pins a
            # single market explicitly. INDEX / CUSTOM / TEST universes
            # have ``universe_market = None`` even when their resolved
            # symbols all live in one market, so the compiler would treat
            # them as mixed-market (USD columns) while async derives the
            # mode from resolved symbols and may use native units. Defer
            # those to async to avoid silent unit mismatches.
            if instant_match is None and should_attempt_instant:
                compile_outcome = self._attempt_compile_path(uow, cmd, symbols)

            # ── Create scan record ───────────────────────────────────
            scan_id = str(uuid.uuid4())
            try:
                scan = uow.scans.create(
                    scan_id=scan_id,
                    criteria=cmd.criteria or {},
                    universe=cmd.universe_label,
                    universe_key=cmd.universe_key,
                    universe_type=cmd.universe_type,
                    universe_market=cmd.universe_market,
                    universe_exchange=cmd.universe_exchange,
                    universe_index=cmd.universe_index,
                    universe_symbols=cmd.universe_symbols,
                    screener_types=cmd.screeners,
                    composite_method=cmd.composite_method,
                    total_stocks=len(symbols),
                    passed_stocks=0,
                    status="queued",
                    trigger_source=cmd.trigger_source,
                    task_id=None,
                    idempotency_key=cmd.idempotency_key,
                    feature_run_id=feature_run_id,
                )
            except SingleActiveScanViolation:
                uow.rollback()
                active_scan = uow.scans.get_active_scan()
                if active_scan is not None:
                    self._raise_active_scan_conflict(active_scan)
                raise

            if instant_match is not None:
                uow.scans.update_status(
                    scan_id,
                    "completed",
                    total_stocks=len(symbols),
                    passed_stocks=instant_match.stats.passed_symbols
                    if instant_match.stats and instant_match.stats.passed_symbols is not None
                    else 0,
                )
                uow.commit()
                return CreateScanResult(
                    scan_id=scan_id,
                    status="completed",
                    total_stocks=len(symbols),
                    is_duplicate=False,
                    feature_run_id=feature_run_id,
                )

            if compile_outcome is not None:
                source_run, results = compile_outcome
                # Normalise stale-criteria score / rating / pass fields
                # before persisting — the covering run was produced under
                # different criteria, so its custom_score / composite_score
                # / rating would mislead users sorting by them. See
                # ``_normalize_compile_details`` for the rationale.
                normalized_results = [
                    (symbol, _normalize_compile_details(details))
                    for symbol, details in results
                ]
                if normalized_results:
                    uow.scan_results.persist_orchestrator_results(
                        scan_id, normalized_results
                    )
                uow.scans.update_status(
                    scan_id,
                    "completed",
                    total_stocks=len(symbols),
                    passed_stocks=len(normalized_results),
                )
                uow.commit()
                logger.info(
                    "Scan %s completed instantly via compile path (run=%s, %d/%d passing)",
                    scan_id,
                    getattr(source_run, "id", None),
                    len(results),
                    len(symbols),
                )
                return CreateScanResult(
                    scan_id=scan_id,
                    status="completed",
                    total_stocks=len(symbols),
                    is_duplicate=False,
                    feature_run_id=None,
                )

            # Commit so the scan row is visible to the Celery worker.
            uow.commit()

            # ── Dispatch background task ─────────────────────────────
            try:
                task_id = self._dispatcher.dispatch_scan(
                    scan_id,
                    symbols,
                    cmd.criteria or {},
                    market=cmd.universe_market,
                )
            except Exception:
                scan.status = "failed"
                uow.commit()
                raise

            scan.task_id = task_id
            uow.commit()

        return CreateScanResult(
            scan_id=scan_id,
            status="queued",
            total_stocks=len(symbols),
            is_duplicate=False,
            feature_run_id=feature_run_id,
        )
