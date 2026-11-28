"""Durable, fail-closed benchmark results used by taxonomy publication."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from app.models.economic_taxonomy import TaxonomyVersion
from app.models.economic_taxonomy_runtime import TaxonomyBenchmarkResult
from app.services.economic_taxonomy_processor import SYSTEM_ACTOR
from app.utils.file_hashing import canonical_json_sha256


def register_verified_benchmark(
    session: Session,
    *,
    report: Mapping[str, Any],
    verified_by: str,
) -> TaxonomyBenchmarkResult:
    payload = dict(report)
    if not payload.get("passed"):
        raise ValueError("benchmark_result_not_passed")
    taxonomy_hash = str(payload.get("taxonomy_hash") or "").strip()
    policy_bundle = str(payload.get("policy_bundle") or "").strip()
    fixture_version = payload.get("fixture_version")
    if not taxonomy_hash:
        raise ValueError("benchmark_taxonomy_hash_required")
    if not policy_bundle:
        raise ValueError("benchmark_policy_bundle_required")
    if not isinstance(fixture_version, int):
        raise TypeError("benchmark_fixture_version_required")
    if not verified_by.strip():
        raise ValueError("benchmark_verifier_required")

    report_hash = canonical_json_sha256(payload)
    existing = session.scalar(
        select(TaxonomyBenchmarkResult).where(
            TaxonomyBenchmarkResult.taxonomy_semantic_hash == taxonomy_hash,
            TaxonomyBenchmarkResult.policy_bundle == policy_bundle,
            TaxonomyBenchmarkResult.report_hash == report_hash,
        )
    )
    if existing is not None:
        return existing
    result = TaxonomyBenchmarkResult(
        taxonomy_semantic_hash=taxonomy_hash,
        policy_bundle=policy_bundle,
        fixture_version=fixture_version,
        report=payload,
        report_hash=report_hash,
        passed=True,
        verified_by=verified_by.strip(),
    )
    session.add(result)
    session.flush()
    return result


# Shadow results reach no reader. Every other mode, rollback included, can
# serve the current processing taxonomy.
BENCHMARK_EXEMPT_MODES = frozenset({"shadow"})


def benchmark_anchor_version(session: Session, version_id) -> TaxonomyVersion:
    """Return the nearest ancestor-or-self version the processor did not create.

    Processor-derived versions inherit the benchmark of this anchor. A seed or a
    reviewed operation becomes a new anchor that needs its own benchmark.
    """
    lineage = (
        select(
            TaxonomyVersion.id,
            TaxonomyVersion.parent_version_id,
            TaxonomyVersion.created_by,
        )
        .where(TaxonomyVersion.id == version_id)
        .cte("anchor_lineage", recursive=True)
    )
    parent = aliased(TaxonomyVersion)
    lineage = lineage.union(
        select(parent.id, parent.parent_version_id, parent.created_by)
        .join(lineage, parent.id == lineage.c.parent_version_id)
        .where(lineage.c.created_by == SYSTEM_ACTOR)
    )
    anchor = session.scalar(
        select(TaxonomyVersion)
        .join(lineage, TaxonomyVersion.id == lineage.c.id)
        .where(lineage.c.created_by != SYSTEM_ACTOR)
    )
    if anchor is None or not anchor.semantic_hash:
        raise LookupError("benchmark_anchor_missing")
    return anchor


def find_verified_benchmark(
    session: Session,
    *,
    taxonomy_semantic_hash: str,
    policy_bundle: str,
) -> dict[str, Any]:
    result = session.scalar(
        select(TaxonomyBenchmarkResult)
        .where(
            TaxonomyBenchmarkResult.taxonomy_semantic_hash
            == taxonomy_semantic_hash,
            TaxonomyBenchmarkResult.policy_bundle == policy_bundle,
            TaxonomyBenchmarkResult.passed.is_(True),
        )
        .order_by(
            TaxonomyBenchmarkResult.created_at.desc(),
            TaxonomyBenchmarkResult.id.desc(),
        )
        .limit(1)
    )
    if result is None:
        raise LookupError("benchmark_result_missing")
    return verify_benchmark_reference(
        session,
        {
            "benchmark_result_id": str(result.id),
            "report_hash": result.report_hash,
        },
        taxonomy_semantic_hash=taxonomy_semantic_hash,
        policy_bundle=policy_bundle,
    )


def verify_benchmark_reference(
    session: Session,
    reference: Mapping[str, Any],
    *,
    taxonomy_semantic_hash: str,
    policy_bundle: str,
) -> dict[str, Any]:
    raw_id = reference.get("benchmark_result_id")
    if not raw_id:
        raise LookupError("benchmark_result_missing")
    result = session.get(TaxonomyBenchmarkResult, UUID(str(raw_id)))
    if result is None:
        raise LookupError("benchmark_result_missing")
    if canonical_json_sha256(result.report) != result.report_hash:
        raise ValueError("benchmark_result_hash_mismatch")
    expected_report_hash = reference.get("report_hash")
    if expected_report_hash and expected_report_hash != result.report_hash:
        raise ValueError("benchmark_result_hash_mismatch")
    if (
        result.report.get("taxonomy_hash") != taxonomy_semantic_hash
        or result.report.get("policy_bundle") != policy_bundle
        or result.report.get("passed") is not True
    ):
        raise ValueError("benchmark_result_contract_mismatch")
    return {
        **dict(result.report),
        "benchmark_result_id": str(result.id),
        "report_hash": result.report_hash,
        "verified_by": result.verified_by,
    }
