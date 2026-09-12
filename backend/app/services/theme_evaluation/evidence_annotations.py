"""Validate all human annotations before changing a review packet."""

from typing import Literal

from pydantic import AwareDatetime

from .records import SHA, Record


class EvidenceAnnotation(Record):
    bundle_id: SHA
    preparation_id: SHA
    entry_id: SHA
    result_id: SHA | None
    input_sha256: SHA
    source_text_sha256: SHA
    decision: Literal["accept", "exclude", "hold"]
    reviewer: str
    reviewed_at: AwareDatetime
    reason: str
    scope: Literal["claim", "evidence"] = "claim"
    significance: Literal["cosmetic", "material", "unknown"] = "unknown"


def apply_annotations(assessment, annotations):
    validated = [EvidenceAnnotation.model_validate(row) for row in annotations]
    seen = set()
    claimed = set()
    for row in validated:
        entry = assessment["entries"].get(row.entry_id)
        if (
            row.bundle_id != assessment["bundle_id"]
            or row.preparation_id != assessment["preparation_id"]
            or entry is None
            or row.result_id != entry["result_id"]
            or row.input_sha256 != entry["input_sha256"]
            or row.source_text_sha256 != entry["source_text_sha256"]
        ):
            raise ValueError("annotation_evidence_mismatch")
        if row.entry_id in seen:
            raise ValueError("duplicate_annotation")
        seen.add(row.entry_id)
        if not row.reviewer.strip() or not row.reason.strip():
            raise ValueError("annotation_reason_and_reviewer_required")
        affected = (
            {row.entry_id}
            if row.scope == "claim"
            else {
                key
                for key, candidate in assessment["entries"].items()
                if candidate["source_id"] == entry["source_id"]
                and candidate["result_id"] == entry["result_id"]
                and candidate["stage"] == entry["stage"]
            }
        )
        if claimed.intersection(affected):
            raise ValueError("annotation_scope_overlap")
        claimed.update(affected)
        if row.significance != "unknown" and entry["issue_code"] != "image_uncertainty":
            raise ValueError("annotation_significance_requires_image_claim")
    for row in validated:
        entry = assessment["entries"][row.entry_id]
        entry["manual_decision"] = row.model_dump(mode="json")
        targets = (
            [entry]
            if row.scope == "claim"
            else [
                candidate
                for candidate in assessment["entries"].values()
                if candidate["source_id"] == entry["source_id"]
                and candidate["result_id"] == entry["result_id"]
                and candidate["stage"] == entry["stage"]
            ]
        )
        for target in targets:
            target["reviewer_disposition"] = row.decision
            if row.decision == "hold":
                target["severity"] = "hold"
        if row.decision == "hold":
            entry["severity"] = "hold"
        elif row.significance == "cosmetic":
            entry["severity"] = "info"
        elif row.significance == "material":
            entry["severity"] = "hold"
        # A manual note never rewrites deterministic selection or extraction flags.
