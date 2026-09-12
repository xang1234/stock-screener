"""Preserve every reference and expose bounded linked-post follow-up work."""

from dataclasses import asdict

from .bundle import canonical_bytes, sha256
from .reference_routing import build_linked_post_manifest, route_reference


def reference_manifest(bundle, manifest, store):
    bindings = {
        b.source_id: b for b in manifest.current_bindings if b.stage == "article"
    }
    routes, rows = [], []
    for ref in bundle.followups:
        binding = bindings.get(ref.reference_id)
        result = store.load_result(binding.result_id) if binding else None
        original_url = manifest.handoff.references.get(
            ref.reference_id, ref.candidate_url
        )
        final_url = result.source_url if result and result.source_url else original_url
        if original_url and final_url:
            routes.append(
                route_reference(
                    reference_id=ref.reference_id,
                    source_post_id=ref.post_id,
                    original_url=original_url,
                    final_url=final_url,
                    canonical_url=result.payload.canonical_url
                    if result and result.payload
                    else None,
                )
            )
        rows.append(
            {
                "reference_id": ref.reference_id,
                "source_post_id": ref.post_id,
                "original_url": original_url,
                "final_url": final_url,
                "result_id": binding.result_id if binding else None,
                "article_id": ref.article_id,
                "original_disposition": ref.status,
                "investment_related": ref.investment_related,
            }
        )
    linked = build_linked_post_manifest(
        routes,
        base_post_ids={
            d.document_id
            for d in bundle.documents
            if d.kind == "post" and d.document_id.removeprefix("post:").isdigit()
        },
    )
    return rows, asdict(linked)


def unprepared_reference_entries(bundle, manifest):
    bound = {b.source_id for b in manifest.current_bindings if b.stage == "article"}
    docs = {doc.document_id: doc for doc in bundle.documents}
    entries = {}
    for ref in bundle.followups:
        if ref.reference_id in bound:
            continue
        article = docs.get(ref.article_id)
        if article:
            code = (
                "native_article_partial"
                if article.capture_status == "partial"
                else "native_article_captured"
            )
            explanation = f"Existing captured article {article.document_id}; completeness {article.capture_status}."
        else:
            code = "reference_not_prepared"
            explanation = f"Reference has original disposition {ref.status}; investment relevance {ref.investment_related}."
        digest = docs[ref.post_id].text_sha256
        key = sha256(
            canonical_bytes(
                {"reference_id": ref.reference_id, "source_hash": digest, "code": code}
            )
        )
        entries[key] = {
            "binding_id": None,
            "source_id": ref.reference_id,
            "source_kind": "reference",
            "stage": "article",
            "result_id": None,
            "input_sha256": article.text_sha256 if article else digest,
            "source_text_sha256": digest,
            "candidate_result_ids": [],
            "selected_result_id": None,
            "eligible": False,
            "issue_code": code,
            "severity": "info"
            if ref.investment_related == "no"
            or ref.status in {"not_article", "skipped_noninvestment"}
            else "review",
            "explanation": explanation,
            "next_action": "Inspect original article or reference; preserve the relationship in coverage.",
            "claim": "",
            "legacy_status": ref.status,
            "legacy_warnings": [],
            "input_locator": ref.candidate_url,
            "parent_result_id": None,
            "manual_decision": None,
            "reviewer_disposition": None,
        }
    return entries
