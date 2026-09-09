"""Reproducible evidence findings; eligibility never constitutes approval."""

from .bundle import IntegrityError, canonical_bytes, sha256
from .evidence_findings import findings
from .evidence_followups import unprepared_reference_entries
from .preparation_store import _atomic
from .reference_routing import destination_identity
from .translation_selection_store import selection_for_preparation

POLICY = "evidence-assessment-v1"
PRIORITY = {"hold": 0, "review": 1, "info": 2}


def build_assessment(base, store, preparation_id, *, annotations=()):
    manifest = store.load(base, preparation_id)
    bundle = store.validate(base, manifest)
    try:
        selection_id, sidecar = selection_for_preparation(base, store, preparation_id)
        decisions = {d.document_id: d for d in sidecar.decisions}
    except IntegrityError as exc:
        if str(exc) != "missing_translation_selection":
            raise
        selection_id, decisions = None, {}
    entries = {}
    destinations = set()
    image_inputs, image_outputs = set(), set()
    for binding in manifest.current_bindings:
        result = store.load_result(binding.result_id)
        root_text = binding.stage == "text" and binding.parent_result_id is None
        decision = decisions.get(binding.source_id) if root_text else None
        candidates = (
            [r for r in (decision.x_result_id, decision.kimi_result_id) if r]
            if decision
            else [binding.result_id]
        )
        selected = decision.selected_result_id if decision else binding.result_id
        if result.stage == "article":
            url = (
                result.payload.final_url
                if result.payload
                else result.source_url or result.request.destination_url
            )
            if url:
                destinations.add(
                    destination_identity(
                        url, result.payload.canonical_url if result.payload else None
                    )
                )
        if result.stage == "image":
            if result.has_input:
                image_inputs.add(result.request.input_sha256)
            if result.payload:
                image_outputs.add(result.request.input_sha256)
        for index, (code, severity, eligible, explanation, action) in enumerate(
            findings(result, decision, root_text)
        ):
            entry_id = sha256(
                canonical_bytes(
                    {
                        "binding_id": binding.binding_id,
                        "issue_index": index,
                        "policy_version": POLICY,
                    }
                )
            )
            entries[entry_id] = {
                "binding_id": binding.binding_id,
                "source_id": binding.source_id,
                "source_kind": binding.source_kind,
                "stage": binding.stage,
                "result_id": binding.result_id,
                "input_sha256": result.request.input_sha256,
                "source_text_sha256": binding.source_text_sha256,
                "candidate_result_ids": candidates,
                "selected_result_id": selected,
                "eligible": eligible,
                "issue_code": code,
                "severity": severity,
                "explanation": explanation,
                "next_action": action,
                "claim": explanation if code == "image_uncertainty" else "",
                "legacy_status": result.status,
                "legacy_warnings": result.warnings,
                "input_locator": binding.input_locator,
                "parent_result_id": binding.parent_result_id,
                "manual_decision": None,
                "reviewer_disposition": None,
            }
    bound_refs = {
        b.source_id for b in manifest.current_bindings if b.stage == "article"
    }
    for ref in bundle.followups:
        if ref.reference_id not in bound_refs and ref.candidate_url:
            article = next(
                (d for d in bundle.documents if d.document_id == ref.article_id), None
            )
            destinations.add(
                destination_identity(article.url if article else ref.candidate_url)
            )
    for doc in bundle.documents:
        completeness = doc.source_metadata.text_complete
        if doc.capture_status == "partial" or (
            doc.kind == "post" and completeness is not True
        ):
            code = (
                "original_incomplete"
                if completeness is False or doc.capture_status == "partial"
                else "original_completeness_unknown"
            )
            entry_id = sha256(
                canonical_bytes(
                    {
                        "document_id": doc.document_id,
                        "source_hash": doc.text_sha256,
                        "code": code,
                        "policy_version": POLICY,
                    }
                )
            )
            entries[entry_id] = {
                "binding_id": None,
                "source_id": doc.document_id,
                "source_kind": "document",
                "stage": "original",
                "result_id": None,
                "input_sha256": doc.text_sha256,
                "source_text_sha256": doc.text_sha256,
                "candidate_result_ids": [],
                "selected_result_id": None,
                "eligible": False,
                "issue_code": code,
                "severity": "review",
                "explanation": "Original capture completeness is "
                + (
                    "unknown."
                    if completeness is None and doc.capture_status != "partial"
                    else "partial."
                ),
                "next_action": "Inspect original capture; unknown does not establish truncation.",
                "claim": "",
                "legacy_status": doc.capture_status,
                "legacy_warnings": doc.source_metadata.incomplete_text_reasons,
                "input_locator": doc.url,
                "parent_result_id": None,
                "manual_decision": None,
                "reviewer_disposition": None,
            }
    entries.update(unprepared_reference_entries(bundle, manifest))
    assessment = {
        "schema_version": 1,
        "policy_version": POLICY,
        "bundle_id": base.name,
        "preparation_id": preparation_id,
        "selection_id": selection_id,
        "evidence_review": "pending",
        "extraction": "awaiting_evidence_approval",
        "coverage": {
            "selected_posts": len(
                {
                    d.document_id
                    for d in bundle.documents
                    if d.kind == "post" and not d.reference_only
                }
            ),
            "reference_records": len(bundle.followups),
            "distinct_destinations": len(destinations),
            "image_references": sum(
                b.stage == "image" for b in manifest.current_bindings
            ),
            "image_inputs": len(image_inputs),
            "image_outputs": len(image_outputs),
            "original_incomplete": sum(
                d.capture_status == "partial"
                or d.source_metadata.text_complete is False
                for d in bundle.documents
            ),
            "original_completeness_unknown": sum(
                d.kind == "post"
                and d.capture_status != "partial"
                and d.source_metadata.text_complete is None
                for d in bundle.documents
            ),
        },
        "entries": entries,
    }
    if annotations:
        from .evidence_annotations import apply_annotations

        apply_annotations(assessment, annotations)
    assessment["entries"] = dict(
        sorted(
            entries.items(), key=lambda item: (PRIORITY[item[1]["severity"]], item[0])
        )
    )
    return assessment


def save_assessment(base, store, assessment):
    annotations = [
        entry["manual_decision"]
        for entry in assessment["entries"].values()
        if entry["manual_decision"]
    ]
    expected = build_assessment(
        base, store, assessment["preparation_id"], annotations=annotations
    )
    if assessment != expected:
        raise ValueError("assessment_evidence_mismatch")
    raw = canonical_bytes(assessment)
    digest = sha256(raw)
    _atomic(store._path("assessments", digest, ".json"), raw)
    return digest


def load_assessment(base, store, preparation_id, assessment_id):
    import json

    assessment = json.loads(store._read("assessments", assessment_id, ".json"))
    if assessment["preparation_id"] != preparation_id:
        raise ValueError("assessment_preparation_mismatch")
    annotations = [
        e["manual_decision"]
        for e in assessment["entries"].values()
        if e["manual_decision"]
    ]
    if assessment != build_assessment(
        base, store, preparation_id, annotations=annotations
    ):
        raise ValueError("assessment_evidence_mismatch")
    return assessment
