"""Saved-evidence assessment and atomic annotation boundary regressions."""

import json
from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import seal_bundle, sha256
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.records import Bundle


def prepared(tmp_path, bundle, document):
    original = "718만주 1.9조원에 매수"
    doc = document(
        text=original,
        original_language="ko",
        source_metadata={
            "x_translation": {
                "status": "captured",
                "text": "Samsung",
                "post_id": "1",
                "original_text_sha256": "sha256:" + sha256(original.encode()),
                "source_language": "ko",
                "target_language": "en",
                "captured_at": datetime.now(timezone.utc),
                "provider": "X translation",
                "display_mode": "on_demand",
                "failure_reason": None,
            }
        },
    )
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle(documents=[doc])))
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])
    return base, store, pid


def test_saved_fragment_failed_fallback_is_material_and_immutable(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import (
        build_assessment,
        load_assessment,
        save_assessment,
    )

    base, store, pid = prepared(tmp_path, bundle, document)
    original = (base / "bundle.json").read_bytes()
    assessment = build_assessment(base, store, pid)
    assert assessment["bundle_id"] == base.name
    assert assessment["preparation_id"] == pid
    entries = list(assessment["entries"].values())
    selected = next(e for e in entries if e["stage"] == "text")
    assert selected["eligible"] is False
    assert selected["severity"] == "hold"
    assert len(selected["candidate_result_ids"]) == 2
    aid = save_assessment(base, store, assessment)
    assert load_assessment(base, store, pid, aid) == assessment
    assert store.load(base, pid).extraction == "awaiting_evidence_approval"
    assert (base / "bundle.json").read_bytes() == original


def test_review_packet_has_priority_files_and_annotations_are_atomic(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import build_assessment
    from app.services.theme_evaluation.preparation_review import render_preparation

    base, store, pid = prepared(tmp_path, bundle, document)
    assessment = build_assessment(base, store, pid)
    entry_id, entry = next(
        (k, e) for k, e in assessment["entries"].items() if e["stage"] == "text"
    )
    decision = {
        "bundle_id": base.name,
        "preparation_id": pid,
        "entry_id": entry_id,
        "result_id": entry["result_id"],
        "input_sha256": entry["input_sha256"],
        "source_text_sha256": entry["source_text_sha256"],
        "decision": "exclude",
        "reviewer": "analyst",
        "reviewed_at": "2026-09-10T00:00:00Z",
        "reason": "Omitted transaction",
        "scope": "evidence",
    }
    output = tmp_path / "review"
    render_preparation(base, store, pid, output, annotations=[decision])
    for name in [
        "START_HERE.md",
        "translation-review.md",
        "translation-review.csv",
        "article-review.csv",
        "image-review.csv",
        "assessment.json",
    ]:
        assert (output / name).is_file()
    packet = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert (
        json.loads((output / "assessment.json").read_bytes())["entries"][entry_id][
            "manual_decision"
        ]["decision"]
        == "exclude"
    )
    bad = {**decision, "input_sha256": "f" * 64}
    with pytest.raises(ValueError, match="annotation_evidence_mismatch"):
        render_preparation(base, store, pid, output, annotations=[bad])
    assert packet == {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert store.load(base, pid).extraction == "awaiting_evidence_approval"


def test_unknown_image_significance_is_review_not_keyword_downgraded(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import build_assessment
    from test_preparation_pipeline import Vision, png

    image = tmp_path / "image.png"
    image.write_bytes(png())

    class Uncertain(Vision):
        def describe_image(self, data, mime_type):
            return {
                "transcription": "",
                "observations": ["Table"],
                "image_type": "table",
                "uncertainties": ["Cosmetic label or investment number unreadable"],
            }

    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    store = PreparationStore(tmp_path / "store")
    handoff = Handoff(
        bundle_id=base.name,
        documents={
            "post:1": {
                "source_text_sha256": document()["text_sha256"],
                "local_images": [str(image)],
            }
        },
    )
    pid = prepare(base, store, handoff, stages=["image"], vision=Uncertain())
    entries = build_assessment(base, store, pid)["entries"].values()
    entry = next(e for e in entries if e["stage"] == "image")
    assert entry["severity"] == "review"
    assert entry["claim"] == "Cosmetic label or investment number unreadable"


def test_annotations_exclude_whole_evidence_and_never_unlock_accept(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import (
        build_assessment,
        load_assessment,
        save_assessment,
    )

    base, store, pid = prepared(tmp_path, bundle, document)
    assessment = build_assessment(base, store, pid)
    key, entry = next(
        (k, e) for k, e in assessment["entries"].items() if e["stage"] == "text"
    )
    annotation = {
        "bundle_id": base.name,
        "preparation_id": pid,
        "entry_id": key,
        "result_id": entry["result_id"],
        "input_sha256": entry["input_sha256"],
        "source_text_sha256": entry["source_text_sha256"],
        "decision": "exclude",
        "reviewer": "reviewer",
        "reviewed_at": "2026-09-10T00:00:00Z",
        "reason": "Unsupported transaction",
        "scope": "evidence",
    }
    excluded = build_assessment(base, store, pid, annotations=[annotation])
    assert all(
        e["reviewer_disposition"] == "exclude"
        for e in excluded["entries"].values()
        if e["stage"] == "text"
    )
    accepted = build_assessment(
        base, store, pid, annotations=[{**annotation, "decision": "accept"}]
    )
    assert not any(
        e["eligible"] for e in accepted["entries"].values() if e["stage"] == "text"
    )
    assert accepted["extraction"] == "awaiting_evidence_approval"
    held = build_assessment(
        base, store, pid, annotations=[{**annotation, "decision": "hold"}]
    )
    assert all(
        e["severity"] == "hold"
        for e in held["entries"].values()
        if e["stage"] == "text"
    )
    assert (
        load_assessment(base, store, pid, save_assessment(base, store, excluded))
        == excluded
    )


def test_packet_links_and_linked_manifest_are_reproducible(tmp_path, bundle, document):
    import re

    from app.services.theme_evaluation.article_intake import (
        apply_followups,
        propose_references,
    )
    from app.services.theme_evaluation.preparation_review import render_preparation

    b = Bundle.model_validate(
        bundle(documents=[document(text="https://x.com/a/status/9")])
    )
    b = apply_followups(b, propose_references(b), [])
    base = seal_bundle(tmp_path, b)
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["article"])
    output = tmp_path / "review"
    summary = render_preparation(base, store, pid, output)
    linked = json.loads((output / "linked-post-manifest.json").read_bytes())
    assert linked["post_ids"] == ["9"]
    assert summary["linked_post_references"] == 1
    assert summary["linked_posts_queued"] == 1
    assert linked["entries"][0]["disposition"] == "queued"
    for path in output.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not link.startswith(("https://", "http://", "#")):
                assert (path.parent / link.split("#")[0]).exists(), (path, link)


def test_conflicting_annotations_rejected_before_report_creation(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import build_assessment
    from app.services.theme_evaluation.preparation_review import render_preparation

    base, store, pid = prepared(tmp_path, bundle, document)
    entries = [
        (k, e)
        for k, e in build_assessment(base, store, pid)["entries"].items()
        if e["stage"] == "text"
    ]
    annotations = [
        {
            "bundle_id": base.name,
            "preparation_id": pid,
            "entry_id": k,
            "result_id": e["result_id"],
            "input_sha256": e["input_sha256"],
            "source_text_sha256": e["source_text_sha256"],
            "reviewer": "reviewer",
            "reviewed_at": "2026-09-10T00:00:00Z",
            "reason": "Reviewed",
            "scope": "evidence",
            "decision": "exclude" if i == 0 else "accept",
        }
        for i, (k, e) in enumerate(entries[:2])
    ]
    assert len(annotations) == 2
    with pytest.raises(ValueError, match="annotation_scope_overlap"):
        render_preparation(
            base, store, pid, tmp_path / "review", annotations=annotations
        )
    assert not (tmp_path / "review").exists()


def test_material_omission_precedes_reviewer_classified_cosmetic_claim(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.bundle import load_bundle
    from app.services.theme_evaluation.evidence_assessment import build_assessment
    from app.services.theme_evaluation.preparation_review import render_preparation
    from app.services.theme_evaluation.public_fetch import PublicResponse
    from test_preparation_pipeline import Vision, png

    base, _, _ = prepared(tmp_path, bundle, document)
    b = load_bundle(base)
    b.documents[0].source_metadata.image_urls = ["https://example.com/image.png"]
    base = seal_bundle(tmp_path, b)
    store = PreparationStore(tmp_path / "combined")

    class Cosmetic(Vision):
        def describe_image(self, data, mime_type):
            return {
                "transcription": "",
                "observations": ["Chart"],
                "image_type": "chart",
                "uncertainties": ["Background logo shape is unclear"],
            }

    pid = prepare(
        base,
        store,
        Handoff(bundle_id=base.name),
        stages=["text", "image"],
        vision=Cosmetic(),
        allow_network=True,
        fetcher=lambda url, **kwargs: PublicResponse(png(), url, "image/png"),
    )
    assessment = build_assessment(base, store, pid)
    key, image = next(
        (key, entry)
        for key, entry in assessment["entries"].items()
        if entry["stage"] == "image"
    )
    annotation = {
        "bundle_id": base.name,
        "preparation_id": pid,
        "entry_id": key,
        "result_id": image["result_id"],
        "input_sha256": image["input_sha256"],
        "source_text_sha256": image["source_text_sha256"],
        "decision": "accept",
        "reviewer": "analyst",
        "reviewed_at": "2026-09-10T00:00:00Z",
        "reason": "Logo shape does not affect the investment claim",
        "significance": "cosmetic",
    }
    reviewed = build_assessment(base, store, pid, annotations=[annotation])
    entries = list(reviewed["entries"].values())
    assert entries[0]["stage"] == "text" and entries[0]["severity"] == "hold"
    assert reviewed["entries"][key]["severity"] == "info"
    output = tmp_path / "packet"
    render_preparation(base, store, pid, output, annotations=[annotation])
    detailed = (output / "evidence.md").read_text()
    assert "Warnings:" in detailed and "Samsung" in detailed
    assert "Background logo shape is unclear" in detailed


def test_incomplete_derived_translation_is_not_eligible():
    from app.services.theme_evaluation.evidence_findings import findings
    from app.services.theme_evaluation.multilingual_preparation import (
        TextPreparation,
        TranslationSegment,
    )
    from app.services.theme_evaluation.preparation_results import (
        TextRequest,
        TextResult,
    )

    original = "매출 증가\n\n매출 상승"
    result = TextResult(
        request=TextRequest(
            input_sha256=sha256(original.encode()),
            provider="test",
            model=None,
            policy_version="text-v1",
            language="ko",
        ),
        payload=TextPreparation(
            source_language="ko",
            supplied_language="ko",
            target_language="en",
            segments=[
                TranslationSegment(
                    original="매출 증가",
                    translated="Revenue increased.",
                    status="translated",
                ),
                TranslationSegment(
                    original="\n\n매출 상승",
                    translated=None,
                    status="unavailable",
                    failure_reason="translation_segment_failed",
                ),
            ],
        ),
    )
    assert not any(eligible for _, _, eligible, _, _ in findings(result, None, False))


@pytest.mark.parametrize(
    "complete,expected_code,incomplete,unknown",
    [
        (None, "original_completeness_unknown", 0, 1),
        (False, "original_incomplete", 1, 0),
        (True, None, 0, 0),
    ],
)
def test_imported_post_completeness_is_not_inferred_from_legacy_status(
    tmp_path, xui_payloads, complete, expected_code, incomplete, unknown
):
    from app.services.theme_evaluation.evidence_assessment import build_assessment
    from app.services.theme_evaluation.xui_intake import import_xui

    for payload in xui_payloads.values():
        payload["items"][0].update(
            text="An investment note with captured text. " * 20,
            text_complete=complete,
            text_source="note_tweet",
            incomplete_text_reasons=[],
        )
    captured = import_xui(
        xui_payloads,
        captured_at=datetime(2026, 9, 1, 10, tzinfo=timezone.utc),
        max_posts_per_source=50,
        requested_limit=50,
        mode="controlled",
    )
    base = seal_bundle(tmp_path, captured)
    original = (base / "bundle.json").read_bytes()
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["image"])
    assessment = build_assessment(base, store, pid)
    originals = [e for e in assessment["entries"].values() if e["stage"] == "original"]
    assert [e["issue_code"] for e in originals] == (
        [expected_code] if expected_code else []
    )
    assert assessment["coverage"]["original_incomplete"] == incomplete
    assert assessment["coverage"]["original_completeness_unknown"] == unknown
    if complete is None:
        assert captured.documents[0].capture_status == "partial"
        assert originals[0]["legacy_status"] == "partial"
        assert "unknown" in originals[0]["explanation"]
    assert (base / "bundle.json").read_bytes() == original
    assert store.load(base, pid).extraction == "awaiting_evidence_approval"


def test_corrected_completeness_policy_preserves_archived_assessment_validation(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.evidence_assessment import (
        build_assessment,
        load_assessment,
        save_assessment,
    )

    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(
                documents=[
                    document(
                        capture_status="partial",
                        source_metadata={"text_complete": None},
                    )
                ]
            )
        ),
    )
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["image"])
    legacy = build_assessment(base, store, pid, policy_version="evidence-assessment-v1")
    legacy_id = save_assessment(base, store, legacy)
    legacy_bytes = store._read("assessments", legacy_id, ".json")
    corrected = build_assessment(base, store, pid)
    assert corrected["policy_version"] == "evidence-assessment-v2"
    assert legacy["coverage"]["original_incomplete"] == 1
    assert corrected["coverage"]["original_incomplete"] == 0
    assert corrected["coverage"]["original_completeness_unknown"] == 1
    assert load_assessment(base, store, pid, legacy_id) == legacy
    assert store._read("assessments", legacy_id, ".json") == legacy_bytes


def test_archived_assessment_load_and_save_do_not_rediscover_selection(
    tmp_path, bundle, document, monkeypatch
):
    from app.services.theme_evaluation import evidence_assessment

    base, store, pid = prepared(tmp_path, bundle, document)
    assessment = evidence_assessment.build_assessment(base, store, pid)
    aid = evidence_assessment.save_assessment(base, store, assessment)

    def forbidden(*args, **kwargs):
        pytest.fail("Archived assessment used latest selection discovery")

    monkeypatch.setattr(evidence_assessment, "selection_for_preparation", forbidden)
    assert evidence_assessment.load_assessment(base, store, pid, aid) == assessment
    assert evidence_assessment.save_assessment(base, store, assessment) == aid


def test_archived_missing_selection_stays_missing_after_sidecar_is_added(
    tmp_path, bundle
):
    from app.services.theme_evaluation.evidence_assessment import (
        build_assessment,
        load_assessment,
        save_assessment,
    )
    from app.services.theme_evaluation.preparation_records import PreparationManifest

    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    store = PreparationStore(tmp_path / "store")
    handoff = Handoff(bundle_id=base.name)
    pid = store.seal(base, PreparationManifest(bundle_id=base.name, handoff=handoff))
    archived = build_assessment(base, store, pid)
    assert archived["selection_id"] is None
    aid = save_assessment(base, store, archived)
    assert prepare(base, store, handoff, stages=["image"], prior_id=pid) == pid
    # A missing prior sidecar remains missing by design; explicitly add an empty sidecar.
    from app.services.theme_evaluation.translation_selection_store import (
        save_translation_selection,
    )

    save_translation_selection(base, store, pid, [])
    assert build_assessment(base, store, pid)["selection_id"] is not None
    assert load_assessment(base, store, pid, aid) == archived


def test_v1_assessment_remains_exact_when_v2_selection_coexists(tmp_path, bundle):
    from app.services.theme_evaluation.evidence_assessment import (
        build_assessment,
        load_assessment,
        save_assessment,
    )
    from app.services.theme_evaluation.translation_selection_store import (
        save_translation_selection,
        selection_for_preparation,
    )

    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])
    decisions = selection_for_preparation(base, store, pid)[1].decisions
    v1_id = save_translation_selection(
        base, store, pid, decisions, policy_version="translation-quality-v1"
    )
    archived = build_assessment(base, store, pid, selection_id=v1_id)
    aid = save_assessment(base, store, archived)
    original_bytes = store._read("assessments", aid, ".json")
    v2_id = save_translation_selection(
        base, store, pid, decisions, policy_version="translation-quality-v2"
    )
    latest = build_assessment(base, store, pid)
    assert latest["selection_id"] == v2_id != v1_id
    selected = next(e for e in latest["entries"].values() if e["stage"] == "text")
    assert "translation-quality-v2" in selected["explanation"]
    assert load_assessment(base, store, pid, aid) == archived
    assert save_assessment(base, store, archived) == aid
    assert store._read("assessments", aid, ".json") == original_bytes
