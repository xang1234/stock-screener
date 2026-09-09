"""Run the actual pipeline with deterministic external adapters."""

import json

import pytest
from app.services.theme_evaluation.article_intake import (
    apply_followups,
    propose_references,
)
from app.services.theme_evaluation.bundle import seal_bundle
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.public_fetch import PublicResponse
from app.services.theme_evaluation.records import Bundle


def test_article_redirect_duplicates_keep_requests_and_distinct_coverage(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import build_assessment

    b = Bundle.model_validate(
        bundle(
            documents=[
                document(
                    text="https://t.co/one https://t.co/two https://x.com/a/status/9"
                )
            ]
        )
    )
    b = apply_followups(b, propose_references(b), [])
    base = seal_bundle(tmp_path, b)
    store = PreparationStore(tmp_path / "store")
    calls = []

    def fetch(url):
        calls.append(url)
        return PublicResponse(
            b"<article><p>Revenue increased 10% this quarter.</p></article>",
            "https://publisher.example/story",
            "text/html",
        )

    stats = {}
    pid = prepare(
        base,
        store,
        Handoff(bundle_id=base.name),
        stages=["article"],
        fetcher=fetch,
        allow_network=True,
        run_summary=stats,
    )
    results = [
        store.load_result(b.result_id) for b in store.load(base, pid).current_bindings
    ]
    assert {r.request.destination_url for r in results} == {
        r.candidate_url for r in b.followups
    }
    assert all(r.request.policy_version == "article-v2" for r in results)
    assert len(calls) == 2
    assert stats["linked_posts"]["post_ids"] == ["9"]
    assert stats["article_download_request_count"] == 2
    assessment = build_assessment(base, store, pid)
    assert assessment["coverage"]["reference_records"] == 3
    assert assessment["coverage"]["distinct_destinations"] == 2
    assert (
        len([e for e in assessment["entries"].values() if e["stage"] == "article"]) >= 3
    )


def test_image_retry_history_and_run_counts(tmp_path, bundle, document):
    from app.services.theme_evaluation.preparation_failures import PreparationFailure
    from test_preparation_pipeline import Vision, png

    img = tmp_path / "image.png"
    img.write_bytes(png())

    class Retry(Vision):
        def describe_image(self, data, mime_type):
            if self.calls == 0:
                self.calls += 1
                raise PreparationFailure("model_timeout")
            return super().describe_image(data, mime_type)

    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(), document(document_id="post:2")])
        ),
    )
    store = PreparationStore(tmp_path / "store")
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            id: {
                "source_text_sha256": document()["text_sha256"],
                "local_images": [str(img)],
            }
            for id in ["post:1", "post:2"]
        },
    )
    stats = {}
    pid = prepare(
        base, store, handoff, stages=["image"], vision=Retry(), run_summary=stats
    )
    assert len(store.load(base, pid).bindings) == 4
    assert len({b.result_id for b in store.load(base, pid).bindings}) == 2
    assert stats["image_model_request_count"] == 2
    assert stats["image_download_request_count"] == 0


def test_opt_in_v2_retains_nonlinguistic_and_cli_reports_selection(
    tmp_path, bundle, document, capsys
):
    from app.services.theme_evaluation.evidence_assessment import build_assessment
    from app.services.theme_evaluation.preparation_cli import main

    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(
                documents=[document(text="📈 https://t.co/abc", original_language=None)]
            )
        ),
    )
    output = tmp_path / "store"
    assert (
        main(
            [
                "prepare",
                "--bundle",
                str(base),
                "--output-root",
                str(output),
                "--stages",
                "text",
                "--text-policy",
                "v2",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["selection_id"]
    assert data["translation_model_request_count"] == 0
    assert data["elapsed_seconds"] >= 0
    store = PreparationStore(output)
    result = store.load_result(
        store.load(base, data["preparation_id"]).current_bindings[0].result_id
    )
    assert result.payload.source_language == "zxx"
    assert result.source_text == "📈 https://t.co/abc"
    assert result.request.policy_version.startswith("text-language-efficient-v2+")
    entries = build_assessment(base, store, data["preparation_id"])["entries"].values()
    text = [e for e in entries if e["stage"] == "text"]
    assert [e["issue_code"] for e in text] == ["translation_not_needed"]
    assert text[0]["severity"] == "info"


def test_full_browser_capture_needs_completeness_basis_before_writes(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.bundle import sha256
    from app.services.theme_evaluation.preparation_pipeline import import_articles

    b = Bundle.model_validate(
        bundle(documents=[document(text="https://publisher.example/story")])
    )
    b = apply_followups(b, propose_references(b), [])
    base = seal_bundle(tmp_path, b)
    store = PreparationStore(tmp_path / "store")
    row = {
        "reference_id": b.followups[0].reference_id,
        "destination_url": b.followups[0].candidate_url,
        "article": {
            "title": "Story",
            "text": "Body",
            "final_url": b.followups[0].candidate_url,
            "method": "browser_import",
            "capture_status": "full",
            "response_sha256": sha256(b"Body"),
            "retrieved_at": "2026-09-10T00:00:00Z",
            "match_basis": "Viewed URL and matching title",
        },
    }
    with pytest.raises(ValueError, match="completeness_basis"):
        import_articles(base, store, Handoff(bundle_id=base.name), [row])
    assert not store.root.exists()


def test_native_article_is_not_refetched_and_preserves_body_language(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import build_assessment

    b = Bundle.model_validate(
        bundle(
            documents=[
                document(text="https://x.com/i/article/99"),
                document(
                    document_id="article:99",
                    kind="article",
                    text="A complete English paragraph about expanding revenue.",
                    original_language="en",
                    capture_status="partial",
                    url="https://x.com/i/article/99",
                ),
            ]
        )
    )
    refs = propose_references(b)
    refs[0].article_id = "article:99"
    refs[0].status = "partial"
    refs[0].attempted_at = b.documents[0].retrieved_at
    refs[0].match_basis = "Viewed native article identifier"
    refs[0].evidence_urls = ["https://x.com/i/article/99"]
    b.followups = refs
    base = seal_bundle(tmp_path, b)

    def forbidden(*args, **kwargs):
        pytest.fail("Native captured article was refetched or translated")

    class Translator:
        provider = "test"
        model = "test"
        policy_version = "test-v1"
        __call__ = forbidden

    store = PreparationStore(tmp_path / "store")
    pid = prepare(
        base,
        store,
        Handoff(bundle_id=base.name),
        stages=["article", "text"],
        translator=Translator(),
        fetcher=forbidden,
        allow_network=True,
        text_policy="v2",
    )
    assert not any(b.stage == "article" for b in store.load(base, pid).bindings)
    article = next(
        store.load_result(b.result_id)
        for b in store.load(base, pid).current_bindings
        if b.source_id == "article:99"
    )
    assert article.payload.source_language == "en"
    assert article.payload.segments[0].status == "identity"
    assert any(
        e["issue_code"] == "native_article_partial"
        for e in build_assessment(base, store, pid)["entries"].values()
    )
