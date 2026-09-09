"""Health scorecard + regression gate for IBD classification bundles.

Pure functions (no DB, no network): given the freshly-built bundle payload and,
optionally, the previous week's payload, produce a per-market health report —
coverage, tier mix, a confidence histogram, the embedding-model fingerprint, and
a week-over-week churn diff — and evaluate it against configurable thresholds.

The report is published as the ``ibd-classification-health-<market>.json`` release
asset next to the bundle/manifest. The gate runs at build time so a regression is
blocked before the new bundle is uploaded; the prior good ``-latest`` manifest then
stays referenced and the static-site build falls back to it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HEALTH_REPORT_SCHEMA_VERSION = 1

HISTOGRAM_BINS = [
    "[0.0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)",
    "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]",
]


def confidence_histogram(rows: Iterable[dict]) -> dict[str, int]:
    """Bucket assignment confidences into ten 0.1-width bins plus a null bucket.

    crosswalk/embedding rows carry a float confidence; llm rows carry None.
    """
    hist: dict[str, int] = {b: 0 for b in HISTOGRAM_BINS}
    hist["null"] = 0
    for row in rows:
        confidence = row.get("confidence")
        if confidence is None:
            hist["null"] += 1
            continue
        if confidence >= 1.0:
            hist[HISTOGRAM_BINS[-1]] += 1
            continue
        # Nudge by a tiny epsilon before truncating so a float just shy of a
        # tenth (e.g. an upstream-computed 0.6 stored as 0.5999999999999999)
        # doesn't land one bucket low.
        idx = int(confidence * 10 + 1e-9)
        idx = max(0, min(idx, 9))
        hist[HISTOGRAM_BINS[idx]] += 1
    return hist


def diff_classifications(prev_rows: Iterable[dict], new_rows: Iterable[dict]) -> dict:
    """Compare two weeks of classifications keyed by symbol.

    ``churn_pct`` is the share of symbols *present in both weeks* whose industry
    group changed — the signal for "did classifications shift unexpectedly".
    Symbols added/removed across weeks are reported separately so a normal
    universe refresh isn't mistaken for churn.
    """
    prev = {r["symbol"]: r.get("industry_group") for r in prev_rows}
    new = {r["symbol"]: r.get("industry_group") for r in new_rows}
    prev_keys, new_keys = set(prev), set(new)
    common = prev_keys & new_keys
    changed = sorted(s for s in common if prev[s] != new[s])
    compared = len(common)
    return {
        "compared": compared,
        "added": len(new_keys - prev_keys),
        "removed": len(prev_keys - new_keys),
        "changed_group": len(changed),
        "churn_pct": round(100.0 * len(changed) / compared, 2) if compared else 0.0,
        "changed_examples": [
            {"symbol": s, "prev": prev[s], "new": new[s]} for s in changed[:50]
        ],
    }


def build_health_report(
    *,
    payload: dict[str, Any],
    prev_payload: dict[str, Any] | None,
    embedding_model: str | None,
) -> dict:
    """Assemble the per-market health report from a fresh payload (+ prior week)."""
    rows = payload.get("classifications", [])
    diff = None
    if prev_payload is not None:
        diff = diff_classifications(prev_payload.get("classifications", []), rows)
        diff["prev_as_of_date"] = prev_payload.get("as_of_date")
        diff["prev_source_revision"] = prev_payload.get("source_revision")
    return {
        "schema_version": HEALTH_REPORT_SCHEMA_VERSION,
        "market": payload.get("market"),
        "as_of_date": payload.get("as_of_date"),
        "generated_at": payload.get("generated_at"),
        "model_id": payload.get("model_id"),
        "embedding_model": embedding_model,
        "summary": payload.get("summary", {}),
        "confidence_histogram": confidence_histogram(rows),
        "diff": diff,
    }


def health_asset_name(market: str) -> str:
    return f"ibd-classification-health-{market.lower()}.json"


def write_health_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_health_report(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class GateResult:
    passed: bool
    mode: str
    breaches: list[str]


def evaluate_gate(
    report: dict,
    *,
    max_churn_pct: float,
    min_coverage_pct: float,
    mode: str,
) -> GateResult:
    """Evaluate coverage + churn thresholds.

    mode="off"     -> never blocks, no breaches reported.
    mode="warn"    -> breaches reported, but passed is always True.
    mode="enforce" -> passed is False when any threshold is breached.
    """
    if mode == "off":
        return GateResult(passed=True, mode=mode, breaches=[])

    breaches: list[str] = []
    # Coerce a missing/None/non-numeric coverage to 0.0 so the gate fails safe
    # (a coverage breach) on a malformed report rather than raising TypeError.
    coverage = (report.get("summary") or {}).get("coverage_pct")
    coverage = coverage if isinstance(coverage, (int, float)) else 0.0
    if coverage < min_coverage_pct:
        breaches.append(f"coverage {coverage}% < min {min_coverage_pct}%")

    summary = report.get("summary") or {}
    attempts = summary.get("llm_attempts")
    successes = summary.get("llm_successes")
    if (
        report.get("model_id")
        and isinstance(attempts, (int, float))
        and attempts > 0
        and successes == 0
    ):
        failures = summary.get("llm_failures", 0)
        parse_failures = summary.get("llm_parse_failures", 0)
        breaches.append(
            "configured model produced zero successful LLM classifications "
            f"({attempts} attempts, {failures} provider failures, "
            f"{parse_failures} invalid responses)"
        )

    diff = report.get("diff")
    if diff is not None:
        churn = diff.get("churn_pct", 0.0)
        if churn > max_churn_pct:
            breaches.append(f"churn {churn}% > max {max_churn_pct}%")

    passed = (not breaches) if mode == "enforce" else True
    return GateResult(passed=passed, mode=mode, breaches=breaches)
