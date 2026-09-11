"""Freeze reviewed evidence inputs without changing acquisition or quality records."""

from collections import defaultdict
from dataclasses import asdict

from pydantic import AwareDatetime, Field

from .bundle import load_bundle, sha256
from .evidence_assessment import load_assessment
from .extraction_records import make_input
from .multilingual_v2 import assess_language
from .preparation_results import ArticleResult, ImageResult, TextResult
from .quantity_display import POLICY_VERSION, normalize_quantities
from .records import SHA, Record


class ExtractionApproval(Record):
    bundle_id: SHA
    preparation_id: SHA
    assessment_id: SHA
    reviewer: str = Field(min_length=1)
    approved_at: AwareDatetime
    reason: str = Field(min_length=1)
    accepted_result_ids: list[SHA] = Field(default_factory=list)


def _translation(result):
    if not isinstance(result, TextResult) or result.payload.target_language != "en":
        return None
    if not result.payload.segments or any(
        s.translated is None for s in result.payload.segments
    ):
        return None
    return "\n\n".join(s.translated for s in result.payload.segments)


def build_extraction_inputs(base, store, preparation_id, assessment_id, approval):
    """Material holds win over approval; partial captures retain explicit limitations."""
    approval = ExtractionApproval.model_validate(approval)
    if (approval.bundle_id, approval.preparation_id, approval.assessment_id) != (
        base.name,
        preparation_id,
        assessment_id,
    ):
        raise ValueError("approval_evidence_mismatch")
    if not approval.reviewer.strip() or not approval.reason.strip():
        raise ValueError("approval_attribution_required")
    assessment = load_assessment(base, store, preparation_id, assessment_id)
    manifest = store.load(base, preparation_id)
    bundle = load_bundle(base)
    docs = {d.document_id: d for d in bundle.documents}
    entries = list(assessment["entries"].values())
    by_binding = defaultdict(list)
    for entry in entries:
        by_binding[entry.get("binding_id")].append(entry)
    held_sources = {
        e["source_id"]
        for e in entries
        if e["severity"] == "hold" or e["reviewer_disposition"] in {"hold", "exclude"}
    }
    known = {b.result_id for b in manifest.current_bindings}
    if not set(approval.accepted_result_ids).issubset(known):
        raise ValueError("approval_result_not_current")
    inputs, exclusions = [], []

    def exclude(source_id, reason, result_id=None):
        exclusions.append(
            {"source_id": source_id, "result_id": result_id, "reason": reason}
        )

    def eligible(binding):
        findings = by_binding[binding.binding_id]
        return bool(findings) and all(
            e["severity"] != "hold"
            and e["reviewer_disposition"] not in {"hold", "exclude"}
            and (e["eligible"] or binding.result_id in approval.accepted_result_ids)
            for e in findings
        )

    def admit(
        doc,
        *,
        source_id,
        text,
        kind,
        result_ids,
        available_at,
        url=None,
        title=None,
        original=None,
        warnings=(),
    ):
        view = normalize_quantities(text)
        if not text.strip() or view.issues:
            exclude(
                source_id,
                "empty_or_ambiguous_quantity_input",
                result_ids[-1] if result_ids else None,
            )
            return
        inputs.append(
            make_input(
                source_id=source_id,
                source_kind=doc.kind,
                source_url=url or doc.url,
                title=title or doc.title,
                text=view.text,
                language="en",
                published_at=doc.published_at,
                available_at=available_at,
                original_text_sha256=sha256(
                    (original if original is not None else doc.text).encode()
                ),
                result_ids=result_ids,
                input_kind=kind,
                normalization_policy=POLICY_VERSION if view.quantities else None,
                normalization=asdict(view) if view.quantities else None,
                warnings=list(dict.fromkeys(warnings)),
            )
        )

    roots = defaultdict(list)
    children = defaultdict(list)
    for binding in manifest.current_bindings:
        if binding.parent_result_id:
            children[(binding.source_id, binding.parent_result_id)].append(binding)
        else:
            roots[binding.source_id].append(binding)

    for doc in bundle.documents:
        if doc.reference_only:
            exclude(doc.document_id, "reviewer_reference_only")
            continue
        if doc.document_id in held_sources:
            exclude(doc.document_id, "material_evidence_hold")
            continue
        warnings = [
            e["issue_code"]
            for e in entries
            if e["source_id"] == doc.document_id and e["stage"] == "original"
        ]
        if doc.capture_status == "partial":
            warnings.append("captured_partial")
        if assess_language(doc.text, doc.original_language).action == "identity":
            admit(
                doc,
                source_id=doc.document_id,
                text=doc.text,
                kind="original",
                result_ids=[],
                available_at=doc.retrieved_at,
                warnings=warnings,
            )
            continue
        chosen = None
        for binding in roots[doc.document_id]:
            if binding.stage != "text" or not eligible(binding):
                continue
            selected = next(
                (
                    e["selected_result_id"]
                    for e in by_binding[binding.binding_id]
                    if e["selected_result_id"]
                ),
                None,
            )
            if selected:
                result = store.load_result(selected)
                translated = _translation(result)
                if translated:
                    chosen = selected, result, translated
                    break
        if chosen:
            rid, result, translated = chosen
            admit(
                doc,
                source_id=doc.document_id,
                text=translated,
                kind="translation",
                result_ids=[rid],
                available_at=max(doc.retrieved_at, result.created_at),
                warnings=warnings,
            )
        else:
            exclude(doc.document_id, "english_translation_unresolved")

    references = {r.reference_id: r for r in bundle.followups}
    for source_id, bindings in roots.items():
        for binding in bindings:
            if binding.stage == "text":
                continue
            ref = references.get(source_id)
            doc = docs.get(ref.post_id if ref else source_id)
            if (
                doc is None
                or doc.reference_only
                or source_id in held_sources
                or doc.document_id in held_sources
            ):
                exclude(
                    source_id, "parent_evidence_unavailable_or_held", binding.result_id
                )
                continue
            result = store.load_result(binding.result_id)
            if not eligible(binding) or not result.source_text.strip():
                exclude(source_id, "derivative_evidence_unresolved", binding.result_id)
                continue
            translated = None
            for child in children[(source_id, binding.result_id)]:
                if eligible(child):
                    child_result = store.load_result(child.result_id)
                    text = _translation(child_result)
                    if text:
                        translated = child, child_result, text
                        break
            if translated is None:
                exclude(
                    source_id, "derivative_translation_unresolved", binding.result_id
                )
                continue
            child, child_result, text = translated
            if isinstance(result, ImageResult):
                kind, url, title = (
                    "image_transcription",
                    doc.url,
                    doc.title + " — image transcription",
                )
            elif isinstance(result, ArticleResult):
                kind, url, title = (
                    "article",
                    result.payload.final_url,
                    result.payload.title,
                )
            else:
                continue
            admit(
                doc,
                source_id=source_id,
                text=text,
                kind=kind,
                url=url,
                title=title,
                result_ids=[binding.result_id, child.result_id],
                original=result.source_text,
                available_at=max(
                    doc.retrieved_at, result.created_at, child_result.created_at
                ),
                warnings=[*result.warnings, "derivative_not_independent_corroboration"],
            )
    inputs.sort(key=lambda item: (item.available_at, item.source_id, item.input_id))
    return inputs, {
        "bundle_id": base.name,
        "preparation_id": preparation_id,
        "assessment_id": assessment_id,
        "approval": approval.model_dump(mode="json"),
        "admission_policy": "reviewed-input-v1",
        "availability_mode": bundle.mode,
        "extraction_status": "pending",
        "selected_documents": len(bundle.documents),
        "admitted_inputs": len(inputs),
        "exclusions": exclusions,
        "limitations": [
            *bundle.limitations,
            "Captured text only; completeness gaps are not repaired.",
            "Supplemental historical posts and synthetic cases are not in this baseline.",
            "No ranked performance or historical speed claims from this pilot.",
        ],
    }
