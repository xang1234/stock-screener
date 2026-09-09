"""Bind recovered references without collapsing original request identities."""

import json
from dataclasses import asdict

from .article_stage import ReferenceInput, recover_references
from .bundle import sha256
from .preparation_records import PreparationBinding
from .preparation_results import ArticleRequest, ArticleResult
from .reference_routing import build_linked_post_manifest


def prepare_articles(bundle, store, state, handoff, *, fetcher, allow_network):
    references = [
        ref
        for ref in bundle.followups
        if not ref.article_id
        and ref.investment_related != "no"
        and ref.status not in {"not_article", "skipped_noninvestment"}
    ]
    inputs = [
        ReferenceInput(
            ref.reference_id,
            ref.post_id,
            handoff.references.get(ref.reference_id, ref.candidate_url),
        )
        for ref in references
        if handoff.references.get(ref.reference_id, ref.candidate_url)
    ]
    downloads = 0

    def fetch(url):
        nonlocal downloads
        if not allow_network:
            raise ValueError("network_disabled")
        downloads += 1
        return fetcher(url)

    outcome = recover_references(inputs, fetcher=fetch)
    documents = {doc.document_id: doc for doc in bundle.documents}
    routes = {route.reference_id: route for route in outcome.routes}
    for ref in references:
        url = handoff.references.get(ref.reference_id, ref.candidate_url)
        capture = outcome.capture_for(ref.reference_id)
        if capture:
            digest = store.save_asset(capture.response.body)
            result = ArticleResult(
                request=ArticleRequest(
                    input_sha256=digest,
                    destination_url=url,
                    provider="local",
                    model=None,
                    policy_version="article-v2",
                ),
                payload=capture.recovery,
                assets=[digest],
                source_url=capture.recovery.final_url,
                created_at=capture.recovery.retrieved_at,
            )
        else:
            code = (
                outcome.assessment_for(ref.reference_id)
                if url
                else "article_destination_unknown"
            )
            if code == "fetch_failed" and not allow_network:
                code = "network_disabled"
            route = routes.get(ref.reference_id)
            result = ArticleResult(
                request=ArticleRequest(
                    input_sha256=sha256((url or ref.reference_text).encode()),
                    destination_url=url,
                    provider="local",
                    model=None,
                    policy_version="article-v2",
                ),
                source_url=route.final_url if route else url,
                failure_reasons=[code],
            )
        rid = store.save_result(result)
        state.record(
            PreparationBinding(
                stage="article",
                source_kind="reference",
                source_id=ref.reference_id,
                source_text_sha256=documents[ref.post_id].text_sha256,
                result_id=rid,
            )
        )
    linked = build_linked_post_manifest(
        outcome.routes,
        base_post_ids={
            doc.document_id
            for doc in bundle.documents
            if doc.kind == "post" and doc.document_id.removeprefix("post:").isdigit()
        },
    )
    return {
        "article_download_request_count": downloads,
        "linked_posts": json.loads(json.dumps(asdict(linked))),
        "reference_routes": [asdict(r) for r in outcome.routes],
    }
