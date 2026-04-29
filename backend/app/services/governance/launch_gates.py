"""ASIA v2 launch-gate runner (bead asia.11.1).

Aggregates the nine gates defined in ``docs/asia/asia_v2_launch_gate_charter.md``
into one deterministic pass/fail artifact. Gates that have evidence in the
repo (runbook, drill record, flag matrix, migration rehearsal, telemetry
event log) self-check; gates that depend on externally-produced reports
(load test output, multilingual QA snapshot) accept an injected evidence
path or report ``MISSING_EVIDENCE`` so reviewers can distinguish "the test
wasn't run" from "the test failed".

Output contract (matches bead 10.4):
- ``data/governance/launch_gates/YYYY-MM-DD.{json,md,sha256}``
- ``content_hash``: SHA-256 over canonical compact JSON (semantic integrity)
- ``.sha256`` sidecar: SHA-256 of raw .json file bytes (file integrity for
  ``sha256sum -c``)

The aggregate verdict is **PASS** only if every hard gate is PASS. A
``MISSING_EVIDENCE`` on any hard gate yields **NO_GO**, distinct from
**FAIL** (which means a check was attempted and breached).
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ...utils.datetime_utils import as_aware_utc
from .signed_artifact import compute_content_hash


REPORT_SCHEMA_VERSION = 2
CHARTER_VERSION = "1.0"  # ties to docs/asia/asia_v2_launch_gate_charter.md
GATE_RUNNER_VERSION = "1.2"


class GateStatus:
    """Distinct from FAIL: MISSING_EVIDENCE means 'we didn't try' so reviewers
    can reach for the right next action (run the test) rather than treat it
    as a regression."""
    PASS = "pass"
    FAIL = "fail"
    MISSING_EVIDENCE = "missing_evidence"
    INFORMATIONAL = "informational"


class GateVerdict:
    PASS = "pass"          # every hard gate PASS
    NO_GO = "no_go"        # any hard gate MISSING_EVIDENCE
    FAIL = "fail"          # any hard gate FAIL


@dataclass
class GateResult:
    gate_id: str
    name: str
    severity: str  # "hard" | "informational"
    status: str
    detail: str
    evidence_paths: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LaunchGateReport:
    report_schema_version: int
    charter_version: str
    runner_version: str
    generated_at: str
    verdict: str
    hard_gate_count: int
    hard_passed: int
    hard_failed: int
    hard_missing_evidence: int
    gates: List[GateResult]
    enabled_markets: List[str] = field(default_factory=list)
    target_market: Optional[str] = None
    execution_mode: Optional[str] = None
    provenance_note: Optional[str] = None
    content_hash: Optional[str] = None


# Each gate is a Callable[[GateContext], GateResult]. The context bundles the
# project root and externally-injected evidence paths so checks stay pure.

@dataclass
class GateContext:
    project_root: Path
    # Operators attach evidence paths produced outside the repo (load test,
    # multilingual QA harness, US parity regression run) here so the gate
    # check can read them. Keys are gate IDs; values are file paths.
    external_evidence: Dict[str, str] = field(default_factory=dict)
    # Injectable wall-clock for tests.
    now: Optional[datetime] = None
    enabled_markets: tuple[str, ...] = ()
    target_market: Optional[str] = None

    def now_utc(self) -> datetime:
        return as_aware_utc(self.now) or datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _file_exists(path: Path) -> bool:
    return path.is_file()


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _doc_path(ctx: GateContext, name: str) -> Path:
    return ctx.project_root / "docs" / "asia" / name


def _normalize_markets(markets: Optional[List[str]] = None) -> tuple[str, ...]:
    from ...tasks.market_queues import SUPPORTED_MARKETS

    allowed = set(SUPPORTED_MARKETS)
    normalized: List[str] = []
    for raw in markets or list(SUPPORTED_MARKETS):
        market = str(raw).strip().upper()
        if market not in allowed:
            raise ValueError(
                f"Unsupported market {raw!r}; expected one of {sorted(allowed)}."
            )
        if market not in normalized:
            normalized.append(market)
    if not normalized:
        raise ValueError("enabled_markets must not be empty.")
    return tuple(normalized)


def _latest_rows_by_market(rows: List[object], markets: tuple[str, ...]) -> Dict[str, object]:
    latest: Dict[str, object] = {}
    for row in rows:
        market = getattr(row, "market", None)
        if market is not None and market not in markets:
            continue
        if market is None:
            # Test doubles may omit market because they only exercise payload parsing.
            market = "__unknown__"
        current = latest.get(market)
        recorded_at = getattr(row, "recorded_at", None)
        current_at = getattr(current, "recorded_at", None) if current is not None else None
        if current is None or (recorded_at is not None and current_at is not None and recorded_at > current_at):
            latest[market] = row
    return latest


def _missing_enabled_markets(latest_rows: Dict[str, object], enabled_markets: tuple[str, ...]) -> list[str]:
    if not enabled_markets or "__unknown__" in latest_rows:
        return []
    return sorted(set(enabled_markets).difference(latest_rows.keys()))


def _check_kr_taxonomy_coverage(ctx: GateContext, db=None) -> GateResult | None:
    if "KR" not in ctx.enabled_markets:
        return None
    try:
        from ...models.stock_universe import StockUniverse
        from ...services.market_taxonomy_service import get_market_taxonomy_service

        active_rows = (
            db.query(StockUniverse.symbol, StockUniverse.exchange)
            .filter(StockUniverse.market == "KR", StockUniverse.active_filter())
            .all()
        )
        if not active_rows:
            return GateResult(
                gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
                status=GateStatus.MISSING_EVIDENCE,
                detail="KR is enabled but no active KR universe rows are available for taxonomy coverage.",
                metrics={"enabled_markets": list(ctx.enabled_markets), "market": "KR"},
            )

        taxonomy = get_market_taxonomy_service()
        sector_group_count = 0
        sub_industry_count = 0
        missing_symbols: list[str] = []
        for symbol, exchange in active_rows:
            entry = taxonomy.get(symbol, market="KR", exchange=exchange)
            has_sector_group = bool(entry and entry.sector and entry.industry_group)
            has_sub_industry = bool(entry and entry.sub_industry)
            if has_sector_group:
                sector_group_count += 1
            else:
                missing_symbols.append(symbol)
            if has_sub_industry:
                sub_industry_count += 1

        total = len(active_rows)
        sector_group_ratio = sector_group_count / total
        sub_industry_ratio = sub_industry_count / total
        metrics = {
            "enabled_markets": list(ctx.enabled_markets),
            "market": "KR",
            "active_symbols": total,
            "sector_industry_group_coverage": round(sector_group_ratio, 4),
            "sub_industry_coverage": round(sub_industry_ratio, 4),
            "sector_industry_group_threshold": 0.95,
            "sub_industry_threshold": 0.85,
            "missing_symbols_sample": missing_symbols[:25],
            "missing_symbols_truncated": len(missing_symbols) > 25,
        }
        if sector_group_ratio < 0.95 or sub_industry_ratio < 0.85:
            return GateResult(
                gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
                status=GateStatus.FAIL,
                detail=(
                    "KR taxonomy coverage below launch thresholds: "
                    f"sector+industry_group={sector_group_ratio:.2%} (<95%) or "
                    f"sub_industry={sub_industry_ratio:.2%} (<85%)."
                ),
                metrics=metrics,
            )
        return None
    except Exception as exc:
        return GateResult(
            gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"KR taxonomy coverage check failed: {exc}",
            metrics={"enabled_markets": list(ctx.enabled_markets), "market": "KR"},
        )


def _drill_age_days(text: str, ctx: GateContext) -> Optional[int]:
    """Pull the ISO date out of a runbook drill record header line.

    Drill records lead with ``# ASIA v2 Runbook Drill Record — YYYY-MM-DD``.
    Returns days between that date and ``ctx.now_utc()``.
    """
    import re
    m = re.search(r"Drill Record — (\d{4}-\d{2}-\d{2})", text)
    if not m:
        return None
    try:
        drill_date = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return max(0, (ctx.now_utc() - drill_date).days)


# ---------------------------------------------------------------------------
# Individual gate checks
# ---------------------------------------------------------------------------
def _check_g1_schema(ctx: GateContext) -> GateResult:
    """G1 — Schema/Contract Readiness.

    Self-check: most recent migration rehearsal report under docs/asia/.
    Glob-based so a new dated report (e.g. pre-canary rehearsal) is
    picked up automatically without editing the gate. Failures here
    block all downstream gates.
    """
    # Match both the original E2/ST3 rehearsal and the broader E11/ST2
    # full-chain rehearsal (bead 11.2). G1 prefers the newest by date
    # SUFFIX (filename ends in _YYYY-MM-DD.md) — sorting by full filename
    # would put e11 before e2 alphabetically and pick the wrong report.
    candidates = list(
        (ctx.project_root / "docs" / "asia").glob(
            "asia_v2_e*_*_migration_rehearsal_report_*.md"
        )
    )

    def _date_key(p: Path) -> str:
        m = re.search(r"_(\d{4}-\d{2}-\d{2})\.md$", p.name)
        return m.group(1) if m else ""

    reports = sorted(candidates, key=_date_key, reverse=True)
    if not reports:
        return GateResult(
            gate_id="G1", name="Schema/Contract Readiness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="No migration rehearsal report found in docs/asia/",
        )
    path = reports[0]
    text = _read_text(path) or ""
    text_lower = text.lower()
    # Accept the report if it asserts no-data-loss OR logs multiple "Success"
    # rows in its results table (≥3 catches both upgrade and downgrade rehearsals).
    has_no_loss_assertion = (
        "no data-loss" in text_lower or "complete without data-loss" in text_lower
    )
    success_count = text.count("Success")
    if has_no_loss_assertion or success_count >= 3:
        detail = (
            "Migration rehearsal report present with no-data-loss assertion."
            if has_no_loss_assertion
            else f"Migration rehearsal report present with {success_count} Success rows in results."
        )
        return GateResult(
            gate_id="G1", name="Schema/Contract Readiness", severity="hard",
            status=GateStatus.PASS,
            detail=detail,
            evidence_paths=[str(path)],
            metrics={"success_markers": success_count},
        )
    return GateResult(
        gate_id="G1", name="Schema/Contract Readiness", severity="hard",
        status=GateStatus.MISSING_EVIDENCE,
        detail=(
            "Rehearsal report exists but lacks either a no-data-loss assertion "
            f"or ≥3 Success rows (found {success_count})."
        ),
        evidence_paths=[str(path)],
        metrics={"success_markers": success_count},
    )


def _check_g2_universe(ctx: GateContext, db=None) -> GateResult:
    """G2 — Universe Integrity and Freshness.

    DB-backed self-check: query the latest ``universe_drift`` telemetry
    event per market and assert max ratio < 0.15 (critical threshold per
    alert_thresholds.py). If no DB session is available, falls back to
    MISSING_EVIDENCE rather than guessing.
    """
    if db is None:
        return GateResult(
            gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="No DB session passed to gate runner; cannot read universe_drift telemetry.",
        )
    try:
        from ...models.market_telemetry import MarketTelemetryEvent
        from ...services.telemetry.schema import MetricKey
        rows = (
            db.query(MarketTelemetryEvent)
            .filter(MarketTelemetryEvent.metric_key == MetricKey.UNIVERSE_DRIFT)
            .filter(MarketTelemetryEvent.recorded_at >= ctx.now_utc() - timedelta(days=2))
            .all()
        )
        latest_rows = _latest_rows_by_market(rows, ctx.enabled_markets)
        if not latest_rows:
            return GateResult(
                gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
                status=GateStatus.MISSING_EVIDENCE,
                detail=(
                    "No universe_drift events in the last 2 days for enabled markets "
                    f"{list(ctx.enabled_markets)}."
                ),
            )
        missing_markets = _missing_enabled_markets(latest_rows, ctx.enabled_markets)
        if missing_markets:
            return GateResult(
                gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
                status=GateStatus.MISSING_EVIDENCE,
                detail=(
                    "Missing universe_drift telemetry in the last 2 days for enabled markets "
                    f"{missing_markets}."
                ),
                metrics={
                    "enabled_markets": list(ctx.enabled_markets),
                    "markets_with_evidence": sorted(latest_rows.keys()),
                    "missing_markets": missing_markets,
                },
            )

        worst = 0.0
        worst_market = None
        for market, r in latest_rows.items():
            if market == "__unknown__":
                market = None
            p = r.payload or {}
            prior = p.get("prior_size") or 0
            delta = p.get("delta") or 0
            if prior > 0:
                ratio = abs(float(delta)) / float(prior)
                if ratio > worst:
                    worst = ratio
                    worst_market = market
    except Exception as exc:
        return GateResult(
            gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"DB query failed: {exc}",
        )

    if worst >= 0.15:
        return GateResult(
            gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
            status=GateStatus.FAIL,
            detail=(
                f"Worst universe drift ratio in last 2d = {worst:.3f} "
                f"(>= 0.15 critical) across {list(ctx.enabled_markets)}."
            ),
            metrics={
                "enabled_markets": list(ctx.enabled_markets),
                "worst_drift_ratio": worst,
                "worst_market": worst_market,
            },
        )
    kr_taxonomy_result = _check_kr_taxonomy_coverage(ctx, db=db)
    if kr_taxonomy_result is not None:
        return kr_taxonomy_result
    return GateResult(
        gate_id="G2", name="Universe Integrity and Freshness", severity="hard",
        status=GateStatus.PASS,
        detail=(
            f"Worst universe drift ratio in last 2d = {worst:.3f} (< 0.15) "
            f"across {list(ctx.enabled_markets)}."
        ),
        metrics={
            "enabled_markets": list(ctx.enabled_markets),
            "worst_drift_ratio": worst,
            "worst_market": worst_market,
        },
    )


def _check_g3_benchmark(ctx: GateContext) -> GateResult:
    """G3 — Benchmark/Calendar Correctness.

    Self-check: the benchmark_registry_service must map each supported market to
    its index symbol (no SPY leakage). This is the same invariant exercised
    by unit tests; the gate runner re-asserts it at evidence-capture time.
    """
    try:
        from ...services.benchmark_registry_service import benchmark_registry
        from ...services.market_calendar_service import MarketCalendarService
    except Exception as exc:
        return GateResult(
            gate_id="G3", name="Benchmark/Calendar Correctness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Benchmark/calendar dependency import failed: {exc}",
        )

    expected = {
        "US": "SPY",
        "HK": "^HSI",
        "IN": "^NSEI",
        "JP": "^N225",
        "KR": "^KS11",
        "TW": "^TWII",
    }
    expected_calendar_ids = {
        "US": "XNYS",
        "HK": "XHKG",
        "IN": "XNSE",
        "JP": "XTKS",
        "KR": "XKRX",
        "TW": "XTAI",
    }
    markets = ctx.enabled_markets
    benchmark_mismatches = {}
    calendar_id_mismatches = {}
    weekend_regressions = []
    sunday_regressions = {}

    service = MarketCalendarService()
    saturday = date(2026, 4, 11)
    sunday_utc = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
    expected_last_completed = date(2026, 4, 10)

    for market in markets:
        try:
            got = benchmark_registry.get_primary_symbol(market)
        except Exception as exc:
            benchmark_mismatches[market] = f"lookup failed: {exc}"
            continue
        if got != expected[market]:
            benchmark_mismatches[market] = f"expected {expected[market]}, got {got}"
        try:
            calendar_id = service.calendar_id(market)
            if calendar_id != expected_calendar_ids[market]:
                calendar_id_mismatches[market] = (
                    f"expected {expected_calendar_ids[market]}, got {calendar_id}"
                )
            if service.is_trading_day(market, saturday):
                weekend_regressions.append(market)
            last_completed = service.last_completed_trading_day(market, now=sunday_utc)
            if last_completed != expected_last_completed:
                sunday_regressions[market] = str(last_completed)
        except Exception as exc:
            return GateResult(
                gate_id="G3", name="Benchmark/Calendar Correctness", severity="hard",
                status=GateStatus.MISSING_EVIDENCE,
                detail=f"Calendar invariant check failed to execute: {exc}",
                metrics={"enabled_markets": list(markets)},
            )

    if benchmark_mismatches or calendar_id_mismatches or weekend_regressions or sunday_regressions:
        return GateResult(
            gate_id="G3", name="Benchmark/Calendar Correctness", severity="hard",
            status=GateStatus.FAIL,
            detail=(
                "Benchmark/calendar regressions detected: "
                f"benchmark={benchmark_mismatches}, "
                f"calendar_ids={calendar_id_mismatches}, "
                f"weekend_sessions={weekend_regressions}, "
                f"sunday_rollover={sunday_regressions}"
            ),
            metrics={
                "enabled_markets": list(markets),
                "benchmark_mismatches": benchmark_mismatches,
                "calendar_id_mismatches": calendar_id_mismatches,
                "weekend_session_regressions": weekend_regressions,
                "sunday_rollover_regressions": sunday_regressions,
            },
        )
    return GateResult(
        gate_id="G3", name="Benchmark/Calendar Correctness", severity="hard",
        status=GateStatus.PASS,
        detail=(
            "Benchmark registry and calendar invariants passed for "
            f"{list(markets)}."
        ),
        metrics={
            "checked_markets": list(markets),
            "saturday_probe": str(saturday),
            "sunday_rollover_probe": str(sunday_utc.date()),
        },
    )


def _check_g4_fundamentals(ctx: GateContext, db=None) -> GateResult:
    """G4 — Fundamentals Data Quality.

    DB-backed: read latest completeness_distribution event per market;
    fail if any market's 0-25 bucket fraction >= 0.50 (critical threshold).
    """
    if db is None:
        return GateResult(
            gate_id="G4", name="Fundamentals Data Quality", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="No DB session passed; cannot read completeness telemetry.",
        )
    try:
        from ...models.market_telemetry import MarketTelemetryEvent
        from ...services.telemetry.schema import MetricKey, low_completeness_ratio
        from ...domain.scanning.models import ScanResultItemDomain
        from ...infra.db.repositories.feature_store_repo import _unpack_feature_joined_row
        from ...infra.db.repositories.scan_result_repo import _unpack_joined_row
        from ...schemas.scanning import ScanResultItem
        from ...services.field_capability_registry import (
            REASON_CODE_NON_US_GAP,
            REASON_CODE_POLICY_EXCLUDED,
        )
        from ...services.growth_cadence_service import (
            BASIS_COMPARABLE_YOY,
            BASIS_UNAVAILABLE,
            CADENCE_INSUFFICIENT,
            CADENCE_SEMIANNUAL,
            REASON_COMPARABLE_YOY_FALLBACK,
            REASON_INSUFFICIENT_HISTORY,
        )

        rows = (
            db.query(MarketTelemetryEvent)
            .filter(MarketTelemetryEvent.metric_key == MetricKey.COMPLETENESS_DISTRIBUTION)
            .all()
        )
        latest_rows = _latest_rows_by_market(rows, ctx.enabled_markets)
        missing_markets = _missing_enabled_markets(latest_rows, ctx.enabled_markets)
        if missing_markets:
            return GateResult(
                gate_id="G4", name="Fundamentals Data Quality", severity="hard",
                status=GateStatus.MISSING_EVIDENCE,
                detail=(
                    "Missing completeness_distribution telemetry for enabled markets "
                    f"{missing_markets}."
                ),
                metrics={
                    "enabled_markets": list(ctx.enabled_markets),
                    "markets_with_evidence": sorted(latest_rows.keys()),
                    "missing_markets": missing_markets,
                },
            )
        worst = 0.0
        worst_market = None
        for market, row in latest_rows.items():
            if market == "__unknown__":
                market = None
            ratio = low_completeness_ratio(row.payload or {})
            if ratio is not None and (worst_market is None or ratio > worst):
                worst, worst_market = ratio, market

        sample_market = ctx.target_market or next(
            (market for market in ctx.enabled_markets if market != "US"),
            "HK",
        )

        fields = getattr(ScanResultItem, "model_fields", None) or getattr(ScanResultItem, "__fields__", {})
        required_fields = {"field_availability", "growth_reporting_cadence", "growth_metric_basis"}
        missing_fields = sorted(required_fields.difference(fields))
        if missing_fields:
            raise AssertionError(f"scan schema missing transparency fields: {missing_fields}")

        unavailable_row = (
            object(),
            "Synthetic Co",
            sample_market,
            "TEST",
            "USD",
            1_000_000.0,
            100_000.0,
            None,
            None,
            None,
            CADENCE_INSUFFICIENT,
            BASIS_UNAVAILABLE,
        )
        _, unavailable_joined = _unpack_joined_row(unavailable_row)
        availability = unavailable_joined["field_availability"] or {}
        for field_name in ("institutional_ownership", "insider_ownership", "short_interest"):
            entry = availability.get(field_name) or {}
            if entry.get("status") != "unsupported":
                raise AssertionError(f"{field_name} did not surface unsupported status")
            if entry.get("reason_code") not in (REASON_CODE_POLICY_EXCLUDED, REASON_CODE_NON_US_GAP):
                raise AssertionError(f"{field_name} surfaced unexpected reason code {entry.get('reason_code')!r}")
        if (availability.get("eps_growth_qq") or {}).get("reason_code") != REASON_INSUFFICIENT_HISTORY:
            raise AssertionError("eps_growth_qq missing insufficient-history reason code")

        computed_row = (
            object(),
            "Synthetic Co",
            sample_market,
            "TEST",
            "USD",
            1_000_000.0,
            100_000.0,
            None,
            None,
            None,
            CADENCE_SEMIANNUAL,
            BASIS_COMPARABLE_YOY,
        )
        _, computed_joined = _unpack_joined_row(computed_row)
        computed_entry = (computed_joined["field_availability"] or {}).get("eps_growth_qq") or {}
        if computed_entry.get("status") != "computed":
            raise AssertionError("eps_growth_qq comparable-period fallback did not surface computed status")
        if computed_entry.get("reason_code") != REASON_COMPARABLE_YOY_FALLBACK:
            raise AssertionError("eps_growth_qq comparable-period fallback reason code missing")

        _, feature_joined = _unpack_feature_joined_row(unavailable_row)
        if "institutional_ownership" not in (feature_joined["field_availability"] or {}):
            raise AssertionError("feature-store transparency contract dropped ownership availability")

        domain_item = ScanResultItemDomain(
            symbol="0700.HK",
            composite_score=80.0,
            rating="Buy",
            current_price=410.0,
            screener_outputs={},
            screeners_run=["minervini"],
            composite_method="weighted_average",
            screeners_passed=1,
            screeners_total=1,
            extended_fields={
                "field_availability": computed_joined["field_availability"],
                "growth_reporting_cadence": CADENCE_SEMIANNUAL,
                "growth_metric_basis": BASIS_COMPARABLE_YOY,
            },
        )
        response = ScanResultItem.from_domain(domain_item)
        if response.field_availability != computed_joined["field_availability"]:
            raise AssertionError("HTTP schema mapping dropped field_availability")
        if response.growth_reporting_cadence != CADENCE_SEMIANNUAL:
            raise AssertionError("HTTP schema mapping dropped growth_reporting_cadence")
        if response.growth_metric_basis != BASIS_COMPARABLE_YOY:
            raise AssertionError("HTTP schema mapping dropped growth_metric_basis")
    except AssertionError as exc:
        return GateResult(
            gate_id="G4", name="Fundamentals Data Quality", severity="hard",
            status=GateStatus.FAIL,
            detail=f"Transparency contract failed: {exc}",
            metrics={"enabled_markets": list(ctx.enabled_markets)},
        )
    except Exception as exc:
        return GateResult(
            gate_id="G4", name="Fundamentals Data Quality", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"DB query failed: {exc}",
        )

    if worst_market is None:
        return GateResult(
            gate_id="G4", name="Fundamentals Data Quality", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=(
                "No completeness_distribution events found for enabled markets "
                f"{list(ctx.enabled_markets)}."
            ),
        )
    if worst >= 0.50:
        return GateResult(
            gate_id="G4", name="Fundamentals Data Quality", severity="hard",
            status=GateStatus.FAIL,
            detail=(
                f"Market {worst_market} 0-25 completeness bucket = {worst:.2f} "
                f"(>= 0.50) across {list(ctx.enabled_markets)}."
            ),
            metrics={
                "enabled_markets": list(ctx.enabled_markets),
                "transparency_sample_market": sample_market,
                "worst_market": worst_market,
                "worst_low_bucket_ratio": worst,
            },
        )
    return GateResult(
        gate_id="G4", name="Fundamentals Data Quality", severity="hard",
        status=GateStatus.PASS,
        detail=(
            f"Worst market completeness 0-25 bucket = {worst:.2f} (< 0.50) on "
            f"{worst_market}; transparency contract passed for {sample_market}."
        ),
        metrics={
            "enabled_markets": list(ctx.enabled_markets),
            "transparency_sample_market": sample_market,
            "worst_market": worst_market,
            "worst_low_bucket_ratio": worst,
        },
    )


def _check_g5_multilingual(ctx: GateContext) -> GateResult:
    """G5 — Multilingual Extraction Quality.

    External evidence required: ``ctx.external_evidence['G5']`` should
    point to the latest QA harness summary (precision/recall/false-positive)
    for zh/ja/zh-TW. Without it, MISSING_EVIDENCE.
    """
    ev = ctx.external_evidence.get("G5")
    if not ev:
        return GateResult(
            gate_id="G5", name="Multilingual Extraction Quality", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="QA harness report path not provided. Pass via external_evidence['G5'].",
        )
    path = Path(ev)
    text = _read_text(path)
    if text is None:
        return GateResult(
            gate_id="G5", name="Multilingual Extraction Quality", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"QA harness report not readable at {ev}",
            evidence_paths=[ev],
        )
    # Caller is expected to have already validated thresholds; the runner
    # asserts the shape and propagates a pass/fail flag if present.
    try:
        data = json.loads(text)
        precision = float(data.get("precision", 0))
        recall = float(data.get("recall", 0))
        fpr = float(data.get("false_positive_rate", 1.0))
    except Exception as exc:
        return GateResult(
            gate_id="G5", name="Multilingual Extraction Quality", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"QA harness report not JSON-parseable: {exc}",
            evidence_paths=[ev],
        )
    if precision < 0.85 or recall < 0.75 or fpr > 0.10:
        return GateResult(
            gate_id="G5", name="Multilingual Extraction Quality", severity="hard",
            status=GateStatus.FAIL,
            detail=(
                f"Threshold miss: precision={precision:.3f} (>=0.85), "
                f"recall={recall:.3f} (>=0.75), fpr={fpr:.3f} (<=0.10)."
            ),
            evidence_paths=[ev],
            metrics={"precision": precision, "recall": recall, "false_positive_rate": fpr},
        )
    return GateResult(
        gate_id="G5", name="Multilingual Extraction Quality", severity="hard",
        status=GateStatus.PASS,
        detail=(
            f"QA: precision={precision:.3f}, recall={recall:.3f}, fpr={fpr:.3f} "
            f"(thresholds: 0.85 / 0.75 / 0.10)."
        ),
        evidence_paths=[ev],
        metrics={"precision": precision, "recall": recall, "false_positive_rate": fpr},
    )


def _check_g6_parity(ctx: GateContext) -> GateResult:
    """G6 — US Parity and Non-US Scan Correctness.

    External evidence: ``ctx.external_evidence['G6']`` is a JSON path to
    the consolidated regression report (us_parity + non_us_correctness).
    """
    return _check_external_pass_fail(
        ctx, gate_id="G6", name="US Parity and Non-US Scan Correctness",
        required_keys=("us_parity_pass", "non_us_correctness_pass"),
    )


def _check_g7_performance(ctx: GateContext) -> GateResult:
    """G7 — Performance and Stability.

    External evidence: load/soak + fault-injection report. Required JSON
    fields: scan_create_p95_ms, scan_failure_rate, market_isolation_pass.
    """
    ev = ctx.external_evidence.get("G7")
    if not ev:
        return GateResult(
            gate_id="G7", name="Performance and Stability", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="Load/soak + fault-injection report path not provided.",
        )
    path = Path(ev)
    text = _read_text(path)
    if text is None:
        return GateResult(
            gate_id="G7", name="Performance and Stability", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Performance report not readable at {ev}",
            evidence_paths=[ev],
        )
    try:
        data = json.loads(text)
        p95 = float(data["scan_create_p95_ms"])
        failure = float(data["scan_failure_rate"])
        # Require a real JSON boolean, not a truthy coercion.
        if not isinstance(data.get("market_isolation_pass"), bool):
            raise TypeError(f"market_isolation_pass must be a JSON boolean, got {type(data.get('market_isolation_pass')).__name__!r}")
        isolation = data["market_isolation_pass"]
    except Exception as exc:
        return GateResult(
            gate_id="G7", name="Performance and Stability", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Performance report missing required fields: {exc}",
            evidence_paths=[ev],
        )
    if p95 > 1500 or failure > 0.01 or not isolation:
        return GateResult(
            gate_id="G7", name="Performance and Stability", severity="hard",
            status=GateStatus.FAIL,
            detail=(
                f"Threshold miss: p95={p95:.0f}ms (<=1500), "
                f"failure={failure:.3f} (<=0.01), isolation={isolation}."
            ),
            evidence_paths=[ev],
            metrics={"p95_ms": p95, "failure_rate": failure, "isolation": isolation},
        )
    return GateResult(
        gate_id="G7", name="Performance and Stability", severity="hard",
        status=GateStatus.PASS,
        detail=(
            f"p95={p95:.0f}ms, failure_rate={failure:.3f}, "
            f"market_isolation={isolation}."
        ),
        evidence_paths=[ev],
        metrics={"p95_ms": p95, "failure_rate": failure, "isolation": isolation},
    )


def _check_g8_observability(ctx: GateContext) -> GateResult:
    """G8 — Observability and Operations Readiness.

    Self-check: runbook doc + most recent drill record both exist, and the
    drill is at most 14 days old. Anything older means the runbook hasn't
    been exercised against the current state of the system.
    """
    runbook = _doc_path(ctx, "asia_v2_operator_runbooks.md")
    if not _file_exists(runbook):
        return GateResult(
            gate_id="G8", name="Observability and Operations Readiness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="Operator runbook missing.",
            evidence_paths=[str(runbook)],
        )

    drills = sorted(
        (ctx.project_root / "docs" / "asia").glob("asia_v2_runbook_drill_*.md"),
        reverse=True,
    )
    if not drills:
        return GateResult(
            gate_id="G8", name="Observability and Operations Readiness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="No runbook drill record found.",
            evidence_paths=[str(runbook)],
        )

    latest = drills[0]
    text = _read_text(latest) or ""
    age = _drill_age_days(text, ctx)
    if age is None:
        return GateResult(
            gate_id="G8", name="Observability and Operations Readiness", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Drill record {latest.name} has no parseable date in header.",
            evidence_paths=[str(runbook), str(latest)],
        )
    if age > 14:
        return GateResult(
            gate_id="G8", name="Observability and Operations Readiness", severity="hard",
            status=GateStatus.FAIL,
            detail=f"Most recent drill ({latest.name}) is {age} days old (>14d stale).",
            evidence_paths=[str(runbook), str(latest)],
            metrics={"drill_age_days": age},
        )
    return GateResult(
        gate_id="G8", name="Observability and Operations Readiness", severity="hard",
        status=GateStatus.PASS,
        detail=f"Runbook present; drill {latest.name} is {age} days old (<=14d).",
        evidence_paths=[str(runbook), str(latest)],
        metrics={"drill_age_days": age},
    )


def _check_g9_rollback(ctx: GateContext) -> GateResult:
    """G9 — Rollback Control Validation.

    Self-check: the flag matrix doc must be present and reference all
    market-scoped kill switches expected by the runbook.
    """
    matrix = _doc_path(ctx, "asia_v2_flag_matrix_and_rollback_runbook.md")
    if not _file_exists(matrix):
        return GateResult(
            gate_id="G9", name="Rollback Control Validation", severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail="Flag matrix document missing.",
            evidence_paths=[str(matrix)],
        )
    text = _read_text(matrix) or ""
    required_flags = (
        "asia_master_enabled",
        "asia_market_hk_enabled",
        "asia_market_jp_enabled",
        "asia_market_tw_enabled",
        "asia_universe_apply_destructive_enabled",
        "asia_reconciliation_quarantine_enforced",
    )
    missing = [f for f in required_flags if f not in text]
    if missing:
        return GateResult(
            gate_id="G9", name="Rollback Control Validation", severity="hard",
            status=GateStatus.FAIL,
            detail=f"Flag matrix missing kill switches: {missing}",
            evidence_paths=[str(matrix)],
            metrics={"missing_flags": missing},
        )
    return GateResult(
        gate_id="G9", name="Rollback Control Validation", severity="hard",
        status=GateStatus.PASS,
        detail=f"Flag matrix references all {len(required_flags)} required kill switches.",
        evidence_paths=[str(matrix)],
        metrics={"checked_flags": list(required_flags)},
    )


def _check_external_pass_fail(
    ctx: GateContext, *, gate_id: str, name: str, required_keys: tuple,
) -> GateResult:
    """Generic check for external JSON evidence with boolean pass keys.

    Used by gates whose evidence is a small JSON file the operator
    attaches (e.g. {"us_parity_pass": true, "non_us_correctness_pass": true}).
    """
    ev = ctx.external_evidence.get(gate_id)
    if not ev:
        return GateResult(
            gate_id=gate_id, name=name, severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Evidence path not provided for {gate_id}.",
        )
    path = Path(ev)
    text = _read_text(path)
    if text is None:
        return GateResult(
            gate_id=gate_id, name=name, severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Evidence file not readable at {ev}",
            evidence_paths=[ev],
        )
    try:
        data = json.loads(text)
    except Exception as exc:
        return GateResult(
            gate_id=gate_id, name=name, severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Evidence not JSON-parseable: {exc}",
            evidence_paths=[ev],
        )
    # Require actual JSON booleans, not truthy coercion. bool("false") is True
    # in Python, so a malformed evidence file would incorrectly pass the gate.
    malformed = [k for k in required_keys if not isinstance(data.get(k), bool)]
    if malformed:
        return GateResult(
            gate_id=gate_id, name=name, severity="hard",
            status=GateStatus.MISSING_EVIDENCE,
            detail=f"Evidence keys must be JSON booleans, got non-bool values: {malformed}",
            evidence_paths=[ev],
            metrics={k: data.get(k) for k in required_keys},
        )
    failed_keys = [k for k in required_keys if not data[k]]
    if failed_keys:
        return GateResult(
            gate_id=gate_id, name=name, severity="hard",
            status=GateStatus.FAIL,
            detail=f"Evidence reports failures: {failed_keys}",
            evidence_paths=[ev],
            metrics={k: data.get(k) for k in required_keys},
        )
    return GateResult(
        gate_id=gate_id, name=name, severity="hard",
        status=GateStatus.PASS,
        detail=f"All required keys true: {list(required_keys)}",
        evidence_paths=[ev],
        metrics={k: data.get(k) for k in required_keys},
    )


# Gate dispatch table — ordered by charter ID. Each entry takes ctx and
# (optionally) db; the runner injects whichever the check accepts.
_GATES: List[Callable] = [
    _check_g1_schema,
    _check_g2_universe,
    _check_g3_benchmark,
    _check_g4_fundamentals,
    _check_g5_multilingual,
    _check_g6_parity,
    _check_g7_performance,
    _check_g8_observability,
    _check_g9_rollback,
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_gates(
    *,
    project_root: Path,
    db=None,
    external_evidence: Optional[Dict[str, str]] = None,
    now: Optional[datetime] = None,
    enabled_markets: Optional[List[str]] = None,
    target_market: Optional[str] = None,
    execution_mode: Optional[str] = None,
    provenance_note: Optional[str] = None,
) -> LaunchGateReport:
    """Execute every gate in charter order and aggregate into one report.

    ``db`` is optional; gates that require it (G2, G4) report
    MISSING_EVIDENCE if it's None. ``external_evidence`` keys are gate IDs
    (G5/G6/G7) pointing at on-disk JSON evidence. ``now`` is injectable
    for deterministic tests.
    """
    ctx = GateContext(
        project_root=project_root,
        external_evidence=dict(external_evidence or {}),
        now=now,
        enabled_markets=_normalize_markets(enabled_markets),
        target_market=target_market.strip().upper() if target_market else None,
    )
    if ctx.target_market and ctx.target_market not in ctx.enabled_markets:
        raise ValueError(
            f"target_market {ctx.target_market!r} must be included in enabled_markets "
            f"{list(ctx.enabled_markets)}."
        )

    results: List[GateResult] = []
    for check in _GATES:
        # Two-arg gates take db; one-arg gates don't. Detected via signature
        # inspection rather than registering metadata — keeps adding a new
        # gate to a one-line edit of _GATES.
        sig = inspect.signature(check)
        if "db" in sig.parameters:
            results.append(check(ctx, db=db))
        else:
            results.append(check(ctx))

    hard = [r for r in results if r.severity == "hard"]
    hard_passed = sum(1 for r in hard if r.status == GateStatus.PASS)
    hard_failed = sum(1 for r in hard if r.status == GateStatus.FAIL)
    hard_missing = sum(1 for r in hard if r.status == GateStatus.MISSING_EVIDENCE)

    if hard_failed > 0:
        verdict = GateVerdict.FAIL
    elif hard_missing > 0:
        verdict = GateVerdict.NO_GO
    else:
        verdict = GateVerdict.PASS

    # Normalize evidence_paths to repo-relative paths before serialising.
    # Absolute workstation-local paths make the signed artifact non-portable
    # and prevent reproducible verification on other machines.
    for r in results:
        r.evidence_paths = [
            _make_relative(p, project_root) for p in r.evidence_paths
        ]

    report = LaunchGateReport(
        report_schema_version=REPORT_SCHEMA_VERSION,
        charter_version=CHARTER_VERSION,
        runner_version=GATE_RUNNER_VERSION,
        enabled_markets=list(ctx.enabled_markets),
        target_market=ctx.target_market,
        execution_mode=execution_mode,
        provenance_note=provenance_note,
        generated_at=ctx.now_utc().isoformat(),
        verdict=verdict,
        hard_gate_count=len(hard),
        hard_passed=hard_passed,
        hard_failed=hard_failed,
        hard_missing_evidence=hard_missing,
        gates=results,
    )
    report.content_hash = _content_hash(report)
    return report


def _make_relative(path_str: str, root: Path) -> str:
    """Convert an absolute path to a repo-relative string if possible.

    Falls back to the original string for paths that can't be relativised
    (e.g. external evidence on a different drive on Windows).
    """
    try:
        return str(Path(path_str).relative_to(root))
    except ValueError:
        return path_str


def _content_hash(report: LaunchGateReport) -> str:
    return compute_content_hash(asdict(report))


def render_json(report: LaunchGateReport) -> str:
    """Return canonical JSON (sorted, indented for human review)."""
    return json.dumps(asdict(report), sort_keys=True, indent=2, default=str)


def render_markdown(report: LaunchGateReport) -> str:
    """Operator-facing rendering. Verdict + per-gate table."""
    lines: List[str] = []
    lines.append(f"# ASIA v2 Launch Gate Report — {report.generated_at}")
    lines.append("")
    lines.append(f"- Charter version: {report.charter_version}")
    lines.append(f"- Runner version: {report.runner_version}")
    lines.append(f"- Report schema version: {report.report_schema_version}")
    lines.append(f"- Enabled markets: {report.enabled_markets}")
    if report.target_market:
        lines.append(f"- Target market: {report.target_market}")
    if report.execution_mode:
        lines.append(f"- Execution mode: {report.execution_mode}")
    if report.provenance_note:
        lines.append(f"- Provenance: {report.provenance_note}")
    lines.append(f"- **Verdict: {report.verdict.upper()}**")
    lines.append(
        f"- Hard gates: {report.hard_gate_count} total · "
        f"{report.hard_passed} pass · {report.hard_failed} fail · "
        f"{report.hard_missing_evidence} missing evidence"
    )
    lines.append(f"- Content hash (SHA-256): `{report.content_hash}`")
    lines.append("")
    lines.append("## Per-gate results")
    lines.append("")
    lines.append("| Gate | Name | Status | Detail |")
    lines.append("|---|---|---|---|")
    for r in report.gates:
        # Pipe-escape detail so a stray "|" doesn't break the table.
        safe_detail = r.detail.replace("|", "\\|")
        lines.append(f"| {r.gate_id} | {r.name} | **{r.status}** | {safe_detail} |")
    lines.append("")
    lines.append("---")
    lines.append(f"Content hash (SHA-256): `{report.content_hash}`")
    lines.append("")
    return "\n".join(lines)
