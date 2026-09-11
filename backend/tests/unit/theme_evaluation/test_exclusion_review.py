"""Excluded evidence must be understandable without resolving opaque IDs."""

import csv
from io import BytesIO

import pytest
from PIL import Image

from app.services.theme_evaluation.bundle import seal_bundle
from app.services.theme_evaluation.evidence_assessment import (
    build_assessment,
    save_assessment,
)
from app.services.theme_evaluation.extraction_capture import save_extraction_run
from app.services.theme_evaluation.extraction_review import render_extractions
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.records import Bundle


def freeze(base, store, pid, exclusions, tmp_path):
    aid = save_assessment(base, store, build_assessment(base, store, pid))
    return save_extraction_run(
        tmp_path,
        manifest={
            "bundle_id": base.name,
            "preparation_id": pid,
            "assessment_id": aid,
            "exclusions": exclusions,
        },
        inputs=[],
        records=[],
    )


class Translator:
    provider = "test"
    model = "test"
    policy_version = "test-v1"

    def __call__(self, text, source, target):
        return "=Translated " + text


def test_root_exclusion_has_full_source_translation_url_and_no_frozen_changes(
    bundle, document, tmp_path
):
    source = "매출 증가\n" + "긴 설명 " * 1500
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(
                documents=[
                    document(
                        text=source, original_language="ko", title="=untrusted title"
                    )
                ]
            )
        ),
    )
    store = PreparationStore(tmp_path / "store")
    pid = prepare(
        base,
        store,
        Handoff(bundle_id=base.name),
        stages=["text"],
        translator=Translator(),
    )
    run = freeze(
        base,
        store,
        pid,
        [
            {
                "source_id": "post:1",
                "result_id": None,
                "reason": "material_evidence_hold",
            }
        ],
        tmp_path,
    )
    before = {
        p: p.read_bytes()
        for root in (base, store.root, run)
        for p in root.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "review"
    render_extractions(run, output, base=base, store=store)
    rows = list(csv.DictReader((output / "exclusions.csv").open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["original_text"] == source
    assert (
        "Translated" in row["translated_text"] and "긴 설명" in row["translated_text"]
    )
    assert row["source_url"] == "https://example.com/post/1"
    assert row["title"] == "'=untrusted title"
    assert row["reason_detail"] and row["translation_result_ids"]
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_image_rows_use_their_own_transcription_and_translation(
    bundle, document, tmp_path
):
    class Vision:
        provider = "test"
        model = "test"
        policy_version = "test-v1"
        calls = 0

        def describe_image(self, raw, mime_type):
            self.calls += 1
            return {
                "transcription": f"도표 {self.calls}",
                "observations": [],
                "uncertainties": ["Unclear axis"],
                "image_type": "chart",
            }

    paths = []
    for color in ("white", "black"):
        buffer = BytesIO()
        Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
        path = tmp_path / (color + ".png")
        path.write_bytes(buffer.getvalue())
        paths.append(str(path))
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    store = PreparationStore(tmp_path / "store")
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            "post:1": {
                "source_text_sha256": document()["text_sha256"],
                "local_images": paths,
            }
        },
    )
    pid = prepare(
        base,
        store,
        handoff,
        stages=["image", "text"],
        vision=Vision(),
        translator=Translator(),
    )
    images = [b for b in store.load(base, pid).current_bindings if b.stage == "image"]
    run = freeze(
        base,
        store,
        pid,
        [
            {
                "source_id": b.source_id,
                "result_id": b.result_id,
                "reason": "derivative_evidence_unresolved",
            }
            for b in images
        ],
        tmp_path,
    )
    render_extractions(run, tmp_path / "review", base=base, store=store)
    rows = list(csv.DictReader((tmp_path / "review/exclusions.csv").open()))
    for row in rows:
        assert row["evidence_kind"] == "image"
        assert row["original_text"] in ("도표 1", "도표 2")
        assert row["original_text"] in row["translated_text"]
        other = "도표 2" if row["original_text"] == "도표 1" else "도표 1"
        assert other not in row["translated_text"]
        assert row["parent_post_text"] == "A supplier reports new orders."
        assert "Unclear axis" in row["review_findings"]


def test_missing_article_keeps_post_context_separate_in_cli_export(
    bundle, document, tmp_path
):
    from app.services.theme_evaluation.article_intake import (
        apply_followups,
        propose_references,
    )
    from app.services.theme_evaluation.extraction_cli import main

    post_text = "New orders https://publisher.example/story"
    evidence = Bundle.model_validate(bundle(documents=[document(text=post_text)]))
    evidence = apply_followups(evidence, propose_references(evidence), [])
    base = seal_bundle(tmp_path, evidence)
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["article"])
    binding = store.load(base, pid).current_bindings[0]
    run = freeze(
        base,
        store,
        pid,
        [
            {
                "source_id": binding.source_id,
                "result_id": binding.result_id,
                "reason": "derivative_evidence_unresolved",
            }
        ],
        tmp_path,
    )
    output = tmp_path / "review"
    assert (
        main(
            [
                "review",
                "--run",
                str(run),
                "--bundle",
                str(base),
                "--store",
                str(store.root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    row = next(csv.DictReader((output / "exclusions.csv").open()))
    assert row["evidence_kind"] == "article"
    assert row["source_url"] == "https://publisher.example/story"
    assert row["original_text"] == ""
    assert row["parent_post_text"] == post_text
    assert row["content_status"] == "No text captured for this item."
    assert row["source_language"] == "unknown"
    assert row["review_findings"]


def test_wrong_context_is_rejected_before_writing_output(bundle, document, tmp_path):
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])
    run = freeze(base, store, pid, [], tmp_path)
    wrong = seal_bundle(
        tmp_path,
        Bundle.model_validate(bundle(documents=[document(text="Wrong corpus")])),
    )
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="exclusion_context_bundle_mismatch"):
        render_extractions(run, output, base=wrong, store=store)
    assert not output.exists()
