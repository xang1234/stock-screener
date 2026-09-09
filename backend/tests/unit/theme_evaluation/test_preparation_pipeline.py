"""Exercise the real preparation/review workflow with only external calls replaced."""

from io import BytesIO

import pytest
from app.services.theme_evaluation.bundle import seal_bundle
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.records import Bundle
from PIL import Image


def api():
    from app.services.theme_evaluation import preparation_pipeline, preparation_review

    return preparation_pipeline, preparation_review


class Vision:
    provider = "opencode-go"
    model = "kimi-k2.6"
    policy_version = "image-v1"

    def __init__(self):
        self.calls = 0

    def describe_image(self, data, mime_type):
        self.calls += 1
        return {
            "transcription": "매출 100억원",
            "observations": ["Revenue table"],
            "image_type": "table",
            "uncertainties": [],
        }


def png():
    out = BytesIO()
    Image.new("RGB", (8, 8), "white").save(out, format="PNG")
    return out.getvalue()


def test_images_reuse_processing_and_review_contains_originals(
    bundle, document, tmp_path
):
    pipeline, review = api()
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(), document(document_id="post:2")])
        ),
    )
    raw_base = (base / "bundle.json").read_bytes()
    img = tmp_path / "test.png"
    img.write_bytes(png())
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            key: {
                "source_text_sha256": document()["text_sha256"],
                "local_images": [str(img)],
            }
            for key in ["post:1", "post:2"]
        },
    )
    store = PreparationStore(tmp_path / "prepared")
    vision = Vision()
    pid = pipeline.prepare(
        base, store, handoff, stages=["image", "text"], vision=vision
    )
    assert vision.calls == 1
    manifest = store.load(base, pid)
    image_bindings = [
        b
        for b in manifest.bindings
        if store.load_result(b.result_id).request.stage == "image"
    ]
    assert len(image_bindings) == 2
    assert image_bindings[0].result_id == image_bindings[1].result_id
    review.render_preparation(base, store, pid, tmp_path / "review")
    report = (tmp_path / "review" / "evidence.md").read_text()
    assert "매출 100억원" in report
    assert (
        "translator_unavailable"
        in (tmp_path / "review" / "preparations.csv").read_text()
    )
    assert "awaiting_evidence_approval" in report
    assert len(list((tmp_path / "review" / "images").iterdir())) == 1
    assert (base / "bundle.json").read_bytes() == raw_base


def test_invalid_handoff_blocks_processing_before_any_call(bundle, tmp_path):
    pipeline, _ = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    handoff = Handoff(
        bundle_id=base.name, documents={"missing": {"source_text_sha256": "a" * 64}}
    )
    vision = Vision()
    with pytest.raises(ValueError):
        pipeline.prepare(
            base,
            PreparationStore(tmp_path / "out"),
            handoff,
            stages=["image"],
            vision=vision,
        )
    assert vision.calls == 0


def test_network_disabled_records_article_followup_without_fetch(bundle, tmp_path):
    pipeline, _ = api()
    from app.services.theme_evaluation.article_intake import (
        apply_followups,
        propose_references,
    )

    b = Bundle.model_validate(bundle())
    b.documents[0].text += " https://publisher.example/article"
    from app.services.theme_evaluation.bundle import sha256

    b.documents[0].text_sha256 = sha256(b.documents[0].text.encode())
    b = apply_followups(b, propose_references(b), [])
    base = seal_bundle(tmp_path, b)

    def fetch(*args, **kwargs):
        pytest.fail("offline run attempted a fetch")

    store = PreparationStore(tmp_path / "out")
    pid = pipeline.prepare(
        base, store, Handoff(bundle_id=base.name), stages=["article"], fetcher=fetch
    )
    manifest = store.load(base, pid)
    results = [store.load_result(b.result_id) for b in manifest.bindings]
    assert results
    assert any("network_disabled" in r.warnings for r in results)


def test_article_import_rejects_wrong_destination(bundle, tmp_path):
    pipeline, _ = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    with pytest.raises(ValueError):
        pipeline.import_articles(
            base,
            PreparationStore(tmp_path / "out"),
            Handoff(bundle_id=base.name),
            [{"reference_id": "unknown", "destination_url": "https://wrong.example"}],
        )


def test_resolved_article_import_clears_active_followups_but_keeps_history(
    bundle, document, tmp_path
):
    pipeline, review = api()
    from datetime import datetime, timezone

    from app.services.theme_evaluation.article_intake import (
        apply_followups,
        propose_references,
    )
    from app.services.theme_evaluation.bundle import sha256

    b = Bundle.model_validate(
        bundle(documents=[document(text="New orders https://publisher.example/story")])
    )
    b = apply_followups(b, propose_references(b), [])
    base = seal_bundle(tmp_path, b)
    handoff = Handoff(bundle_id=base.name)
    store = PreparationStore(tmp_path / "out")
    first = pipeline.prepare(base, store, handoff, stages=["article"])
    body = "Full recovered article."
    row = {
        "reference_id": b.followups[0].reference_id,
        "destination_url": "https://publisher.example/story",
        "article": {
            "title": "Recovered",
            "text": body,
            "final_url": "https://publisher.example/story",
            "method": "browser_import",
            "capture_status": "full",
            "response_sha256": sha256(body.encode()),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "match_basis": "Same canonical URL and title.",
            "completeness_basis": "Reviewer inspected the complete body through its end.",
        },
    }
    second = pipeline.import_articles(base, store, handoff, [row], prior_id=first)
    assert len(store.load(base, second).bindings) == 2
    summary = review.render_preparation(base, store, second, tmp_path / "review")
    assert summary["gaps"] == 0
    import csv

    with (tmp_path / "review" / "article_followups.csv").open() as handle:
        assert list(csv.DictReader(handle)) == []


def test_browser_import_requires_actual_capture_time():
    pipeline, _ = api()
    with pytest.raises(ValueError, match="capture_time"):
        pipeline.ArticleImport.model_validate(
            {
                "reference_id": "ref",
                "destination_url": "https://example.com",
                "article": {
                    "title": "story",
                    "text": "full text",
                    "final_url": "https://example.com",
                    "method": "browser_import",
                    "capture_status": "full",
                    "response_sha256": "a" * 64,
                    "match_basis": "same URL",
                },
            }
        )


def test_returning_to_cached_image_makes_that_version_current(
    bundle, document, tmp_path
):
    pipeline, _ = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    image_path = tmp_path / "image.png"
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            "post:1": {
                "source_text_sha256": document()["text_sha256"],
                "local_images": [str(image_path)],
            }
        },
    )
    store = PreparationStore(tmp_path / "out")
    vision = Vision()
    prior = None
    for color in ("white", "black", "white"):
        raw = BytesIO()
        Image.new("RGB", (8, 8), color).save(raw, format="PNG")
        image_path.write_bytes(raw.getvalue())
        prior = pipeline.prepare(
            base, store, handoff, stages=["image"], vision=vision, prior_id=prior
        )
    manifest = store.load(base, prior)
    active = manifest.current_bindings
    from app.services.theme_evaluation.bundle import sha256

    assert len(active) == 1
    current = store.load_result(active[0].result_id)
    assert current.request.input_sha256 == sha256(image_path.read_bytes())
    assert vision.calls == 2


def test_changed_image_invalidates_translation_and_does_not_process_history(
    bundle, document, tmp_path
):
    pipeline, review = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    img = tmp_path / "image.png"
    img.write_bytes(png())
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            "post:1": {
                "source_text_sha256": document()["text_sha256"],
                "local_images": [str(img)],
            }
        },
    )

    class ChangingVision(Vision):
        def describe_image(self, data, mime_type):
            result = super().describe_image(data, mime_type)
            result["transcription"] = "매출 증가" if self.calls == 1 else "수출 증가"
            return result

    class Translator:
        provider = "test"
        model = "test"
        policy_version = "text-v1"

        def __init__(self):
            self.inputs = []

        def __call__(self, text, source, target):
            self.inputs.append(text)
            return "Revenue growth"

    store = PreparationStore(tmp_path / "out")
    vision = ChangingVision()
    first = pipeline.prepare(
        base, store, handoff, stages=["image", "text"], vision=vision
    )
    old = store.load(base, first)
    old_image = next(b.result_id for b in old.current_bindings if b.stage == "image")
    Image.new("RGB", (8, 8), "black").save(img, format="PNG")
    changed = pipeline.prepare(
        base, store, handoff, stages=["image"], vision=vision, prior_id=first
    )
    assert not any(
        b.parent_result_id == old_image
        for b in store.load(base, changed).current_bindings
    )
    translator = Translator()
    translated = pipeline.prepare(
        base, store, handoff, stages=["text"], translator=translator, prior_id=changed
    )
    assert translator.inputs == ["수출 증가"]
    manifest = store.load(base, translated)
    assert any(b.parent_result_id == old_image for b in manifest.bindings)
    assert not any(b.parent_result_id == old_image for b in manifest.current_bindings)
    summary = review.render_preparation(base, store, translated, tmp_path / "review")
    assert summary["gaps"] == 0


def test_changed_image_with_missing_provider_does_not_retain_old_pixels(
    bundle, document, tmp_path
):
    pipeline, _ = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    img = tmp_path / "image.png"
    img.write_bytes(png())
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            "post:1": {
                "source_text_sha256": document()["text_sha256"],
                "local_images": [str(img)],
            }
        },
    )
    store = PreparationStore(tmp_path / "out")
    first = pipeline.prepare(base, store, handoff, stages=["image"], vision=Vision())
    Image.new("RGB", (8, 8), "black").save(img, format="PNG")
    second = pipeline.prepare(base, store, handoff, stages=["image"], prior_id=first)
    current = store.load_result(store.load(base, second).current_bindings[0].result_id)
    from app.services.theme_evaluation.bundle import sha256

    assert current.request.input_sha256 == sha256(img.read_bytes())
    assert current.status == "unavailable"


def test_shared_image_result_does_not_keep_translation_for_changed_attachment(
    bundle, document, tmp_path
):
    pipeline, _ = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    images = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in images:
        path.write_bytes(png())
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            "post:1": {
                "source_text_sha256": document()["text_sha256"],
                "local_images": [str(p) for p in images],
            }
        },
    )
    store = PreparationStore(tmp_path / "out")
    first = pipeline.prepare(
        base, store, handoff, stages=["image", "text"], vision=Vision()
    )
    Image.new("RGB", (8, 8), "black").save(images[0], format="PNG")
    second = pipeline.prepare(
        base, store, handoff, stages=["image"], vision=Vision(), prior_id=first
    )
    children = [
        b for b in store.load(base, second).current_bindings if b.parent_result_id
    ]
    assert [b.input_locator for b in children] == [str(images[1])]
