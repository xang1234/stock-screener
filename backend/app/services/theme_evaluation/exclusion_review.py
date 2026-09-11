"""Readable exclusions joined to the exact archived evidence, without new decisions."""

from .bundle import load_bundle
from .evidence_assessment import load_assessment
from .preparation_results import ArticleResult, ImageResult, TextResult

EXCLUSION_FIELDS = [
    "reason_detail",
    "evidence_kind",
    "title",
    "source_url",
    "original_text",
    "translated_text",
    "translation_status",
    "review_findings",
    "content_status",
    "parent_post_url",
    "parent_post_text",
    "source_language",
    "published_at",
    "captured_at",
    "translation_result_ids",
    "source_id",
    "result_id",
    "reason",
]
_REASONS = {
    "english_translation_unresolved": "No English translation met the admission requirements. Available attempts are shown for review.",
    "material_evidence_hold": "A material evidence or translation issue is still on hold. This source was excluded from the baseline.",
    "derivative_evidence_unresolved": "The article or image evidence has unresolved quality, completeness or uncertainty findings.",
    "derivative_translation_unresolved": "The article or image text has no English preparation that met the admission requirements.",
    "parent_evidence_unavailable_or_held": "The parent source is unavailable, reference-only, or on hold, so this attached evidence was excluded too.",
    "empty_or_ambiguous_quantity_input": "The proposed input was empty or contained an ambiguous quantity requiring review.",
    "reviewer_reference_only": "This item is supplemental reviewer context, not a detector input.",
}


def _translated_text(result):
    return "\n\n".join(
        segment.translated
        if segment.translated is not None
        else "[Translation unavailable for this segment]"
        for segment in result.payload.segments
    )


def exclusion_rows(run, *, base=None, store=None):
    """Preserve one row per frozen exclusion; source/translation text is never truncated."""
    frozen = run["manifest"]
    exclusions = frozen.get("exclusions", [])
    if (base is None) != (store is None):
        raise ValueError("exclusion_context_requires_bundle_and_store")
    if base is None:
        return [
            {
                **row,
                "reason_detail": _REASONS.get(row["reason"], row["reason"]),
                "content_status": "Evidence context not supplied; render with --bundle and --store.",
            }
            for row in exclusions
        ]
    if base.name != frozen["bundle_id"]:
        raise ValueError("exclusion_context_bundle_mismatch")
    assessment = load_assessment(
        base, store, frozen["preparation_id"], frozen["assessment_id"]
    )
    preparation = store.load(base, frozen["preparation_id"])
    bundle = load_bundle(base)
    docs = {doc.document_id: doc for doc in bundle.documents}
    refs = {ref.reference_id: ref for ref in bundle.followups}
    findings = list(assessment["entries"].values())
    results = {}

    def result_for(rid):
        if rid not in results:
            results[rid] = store.load_result(rid)
        return results[rid]

    rows = []
    for exclusion in exclusions:
        source_id, rid = exclusion["source_id"], exclusion.get("result_id")
        ref = refs.get(source_id)
        doc = docs.get(ref.post_id if ref else source_id)
        bound = [b for b in preparation.bindings if b.source_id == source_id]
        if doc is None or (rid and not any(b.result_id == rid for b in bound)):
            raise ValueError("exclusion_context_source_mismatch")
        result = result_for(rid) if rid else None
        relevant = [
            e
            for e in findings
            if e["source_id"] == source_id
            and (
                e["result_id"] == rid
                or (rid and e.get("parent_result_id") == rid)
                or (
                    not rid
                    and (
                        e["severity"] == "hold"
                        or (
                            e["stage"] in ("original", "text")
                            and not e.get("parent_result_id")
                        )
                    )
                )
            )
        ]
        if exclusion["reason"] == "parent_evidence_unavailable_or_held" and ref:
            relevant.extend(
                e
                for e in findings
                if e["source_id"] == doc.document_id
                and (
                    e["severity"] == "hold"
                    or e["reviewer_disposition"] in ("hold", "exclude")
                )
            )
        selected_ids = {
            e["selected_result_id"] for e in relevant if e["selected_result_id"]
        }
        candidate_ids = []
        if isinstance(result, TextResult):
            candidate_ids.append(rid)
        elif rid:
            candidate_ids.extend(
                b.result_id
                for b in preparation.current_bindings
                if b.source_id == source_id
                and b.stage == "text"
                and b.parent_result_id == rid
            )
        else:
            root_text_ids = {
                b.result_id
                for b in bound
                if b.stage == "text" and b.parent_result_id is None
            }
            candidate_ids.extend(
                b.result_id
                for b in preparation.current_bindings
                if b.result_id in root_text_ids
            )
            candidate_ids.extend(
                candidate
                for e in relevant
                for candidate in e.get("candidate_result_ids", [])
                if candidate in root_text_ids
            )
        candidate_ids = list(dict.fromkeys(candidate_ids))
        translations, statuses = [], []
        for candidate_id in candidate_ids:
            candidate = result_for(candidate_id)
            if (
                not isinstance(candidate, TextResult)
                or candidate.payload.target_language != "en"
            ):
                raise ValueError("exclusion_translation_context_mismatch")
            role = (
                "selected candidate"
                if candidate_id in selected_ids
                else "other saved candidate"
            )
            label = f"{candidate.request.provider}/{candidate.request.model or 'none'}; {candidate.status}; {role}"
            statuses.append(label)
            translations.append(
                f"[{label}; result {candidate_id}]\n{_translated_text(candidate)}"
            )
        url, title = doc.url, doc.title
        if isinstance(result, ArticleResult):
            url = (
                result.payload.final_url
                if result.payload
                else result.source_url
                or result.request.destination_url
                or (ref.candidate_url if ref else None)
                or doc.url
            )
            title = (
                result.payload.title
                if result.payload
                else (ref.reference_text if ref else doc.title)
            )
        elif isinstance(result, ImageResult):
            url = result.source_url or doc.url
        original = result.source_text if result is not None else doc.text
        language = (
            result.payload.source_language
            if isinstance(result, TextResult)
            else (
                result_for(candidate_ids[0]).payload.source_language
                if candidate_ids
                else doc.original_language
                if result is None
                else None
            )
        )
        rows.append(
            {
                **exclusion,
                "reason_detail": _REASONS.get(exclusion["reason"], exclusion["reason"]),
                "evidence_kind": result.stage if result is not None else doc.kind,
                "title": title,
                "source_url": url,
                "original_text": original,
                "translated_text": "\n\n".join(translations),
                "translation_status": "\n".join(statuses)
                or "No saved English preparation found.",
                "review_findings": "\n\n".join(
                    dict.fromkeys(
                        f"{e['severity']}: {e['issue_code']} — {e['explanation']}"
                        for e in relevant
                    )
                ),
                "content_status": (result.status if result else doc.capture_status)
                if original
                else "No text captured for this item.",
                "parent_post_url": doc.url,
                "parent_post_text": doc.text,
                "source_language": language or "unknown",
                "published_at": doc.published_at.isoformat()
                if doc.published_at
                else "",
                "captured_at": result.created_at.isoformat()
                if result
                else doc.retrieved_at.isoformat(),
                "translation_result_ids": "; ".join(candidate_ids),
            }
        )
    return rows
