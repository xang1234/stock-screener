import copy
from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest
from app.services.theme_evaluation.bundle import canonical_bytes, seal_bundle
from app.services.theme_evaluation.evidence_assessment import (
    build_assessment,
    save_assessment,
)
from app.services.theme_evaluation.extraction_intake import build_extraction_inputs
from app.services.theme_evaluation.extraction_records import make_input
from app.services.theme_evaluation.grounding import (
    prepare_grounding,
    validate_grounding,
)
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.records import Bundle


def _approval(base, preparation_id, assessment_id):
    return {
        "bundle_id": base.name,
        "preparation_id": preparation_id,
        "assessment_id": assessment_id,
        "reviewer": "reviewer",
        "approved_at": "2026-09-02T00:00:00Z",
        "reason": "Reviewed evidence is suitable for this limited extraction.",
        "accepted_result_ids": [],
    }


def _frozen_run(bundle, tmp_path):
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle))
    store = PreparationStore(tmp_path / "store")
    preparation_id = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])
    assessment_id = save_assessment(
        base, store, build_assessment(base, store, preparation_id)
    )
    approval = _approval(base, preparation_id, assessment_id)
    inputs, manifest = build_extraction_inputs(
        base, store, preparation_id, assessment_id, approval
    )
    return (
        base,
        store,
        {
            "run_id": "f" * 64,
            "manifest": manifest,
            "inputs": inputs,
            "records": [],
        },
    )


def _digest(packet):
    unsigned = {key: value for key, value in packet.items() if key != "digest"}
    return sha256(canonical_bytes(unsigned)).hexdigest()


def _input(source_id, kind, text, *, available_at, url=None, source_kind="post"):
    return make_input(
        source_id=source_id,
        source_kind=source_kind,
        source_url=url or "https://example.com/" + source_id.replace(":", "/"),
        title=source_id,
        text=text,
        language="en",
        published_at=available_at,
        available_at=available_at,
        original_text_sha256=sha256(text.encode()).hexdigest(),
        result_ids=[],
        input_kind=kind,
        normalization_policy=None,
        warnings=["captured_partial"] if kind == "image_transcription" else [],
    )


def _company(symbol, name):
    return {
        "symbol": symbol,
        "name": name,
        "identity_source": "active_universe",
        "sector": "Technology",
        "industry": "Information Technology Services",
        "business_description": "Provides cloud computing infrastructure.",
        "profile_source": {"industry": "finviz"},
        "profile_as_of": {"industry": "2026-09-01T00:00:00Z"},
        "profile_status": "available",
    }


def test_prepare_binds_approved_inputs_and_current_context(bundle, tmp_path):
    base, store, run = _frozen_run(bundle(), tmp_path)
    as_of = datetime(2026, 9, 2, tzinfo=timezone.utc)

    packet = prepare_grounding(
        run,
        base,
        store,
        {
            run["inputs"][0].input_id: {
                "companies": [],
                "warnings": ["identity_unknown"],
            }
        },
        as_of,
    )

    assert packet["run_id"] == run["run_id"]
    assert set(packet["contexts"]) == {run["inputs"][0].input_id}
    context = validate_grounding(
        packet, run["inputs"], run_id=run["run_id"], manifest=run["manifest"]
    )
    assert context[run["inputs"][0].input_id].context_available_at == as_of
    assert context[run["inputs"][0].input_id].warnings == ["identity_unknown"]
    wrong_manifest = {**run["manifest"], "bundle_id": "0" * 64}
    with pytest.raises(ValueError, match="grounding_packet_manifest_mismatch"):
        validate_grounding(
            packet,
            run["inputs"],
            run_id=run["run_id"],
            manifest=wrong_manifest,
        )


def test_prepare_does_not_import_excluded_or_unrelated_image(
    bundle, document, tmp_path, monkeypatch
):
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(
                documents=[
                    document(document_id="post:1"),
                    document(document_id="post:2"),
                ]
            )
        ),
    )
    stamp = datetime(2026, 9, 1, tzinfo=timezone.utc)
    post_one = _input("post:1", "original", "Post one", available_at=stamp)
    post_two = _input("post:2", "original", "Post two", available_at=stamp)
    image_two = _input("post:2", "image_transcription", "Image two", available_at=stamp)
    inputs = [post_one, post_two, image_two]
    approval = _approval(base, "a" * 64, "b" * 64)
    manifest = {
        "bundle_id": base.name,
        "preparation_id": "a" * 64,
        "assessment_id": "b" * 64,
        "approval": approval,
        "admitted_inputs": len(inputs),
        "exclusions": [
            {
                "source_id": "post:held",
                "reason": "material_evidence_hold",
                "result_id": None,
            }
        ],
    }
    # Admission comparison is the security boundary here; the relationship test
    # supplies an already-reviewed fixture without invoking image OCR.
    monkeypatch.setattr(
        "app.services.theme_evaluation.grounding.build_extraction_inputs",
        lambda *args: (inputs, {**manifest}),
    )
    run = {"run_id": "c" * 64, "manifest": manifest, "inputs": inputs, "records": []}

    packet = prepare_grounding(
        run, base, PreparationStore(tmp_path / "store"), {}, stamp
    )

    assert packet["contexts"][post_one.input_id]["evidence"] == []
    image_context = packet["contexts"][image_two.input_id]
    assert [(e["input_id"], e["relation"]) for e in image_context["evidence"]] == [
        (post_two.input_id, "parent_post")
    ]

    excluded = copy.deepcopy(run)
    excluded["inputs"] = [
        *inputs,
        _input("post:held", "original", "Held", available_at=stamp),
    ]
    with pytest.raises(ValueError, match="grounding_admission_mismatch"):
        prepare_grounding(
            excluded, base, PreparationStore(tmp_path / "store"), {}, stamp
        )


def test_validation_rejects_tampered_evidence_bytes_url_and_availability(
    bundle, document, tmp_path, monkeypatch
):
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(bundle(documents=[document(document_id="post:1")])),
    )
    stamp = datetime(2026, 9, 1, tzinfo=timezone.utc)
    post = _input("post:1", "original", "Post", available_at=stamp)
    image = _input(
        "post:1", "image_transcription", "Attached image", available_at=stamp
    )
    inputs = [post, image]
    approval = _approval(base, "a" * 64, "b" * 64)
    manifest = {
        "bundle_id": base.name,
        "preparation_id": "a" * 64,
        "assessment_id": "b" * 64,
        "approval": approval,
        "admitted_inputs": 2,
        "exclusions": [],
    }
    monkeypatch.setattr(
        "app.services.theme_evaluation.grounding.build_extraction_inputs",
        lambda *args: (inputs, {**manifest}),
    )
    run = {"run_id": "d" * 64, "manifest": manifest, "inputs": inputs, "records": []}
    packet = prepare_grounding(
        run, base, PreparationStore(tmp_path / "store"), {}, stamp
    )
    for field, value in (
        ("text", "old bytes"),
        ("source_url", "https://wrong.example/evidence"),
        ("available_at", (stamp + timedelta(days=1)).isoformat()),
    ):
        tampered = copy.deepcopy(packet)
        changed = tampered["contexts"][post.input_id]["evidence"][0]
        changed[field] = value
        if field == "text":
            changed["text_sha256"] = sha256(value.encode()).hexdigest()
            changed["original_text_sha256"] = sha256(value.encode()).hexdigest()
        tampered["digest"] = _digest(tampered)
        with pytest.raises(ValueError):
            validate_grounding(
                tampered, inputs, run_id=run["run_id"], manifest=run["manifest"]
            )


def test_prepare_rejects_future_primary(bundle, tmp_path):
    base, store, run = _frozen_run(bundle(), tmp_path)
    with pytest.raises(ValueError, match="grounding_primary_not_yet_available"):
        prepare_grounding(
            run,
            base,
            store,
            {},
            datetime(2026, 8, 1, tzinfo=timezone.utc),
        )


def test_validation_rechecks_exact_linked_article_relation(
    bundle, document, tmp_path, monkeypatch
):
    stamp = datetime(2026, 9, 1, tzinfo=timezone.utc)
    article_one = document(
        document_id="article:one",
        kind="article",
        text="Linked article",
        url="https://example.com/article/one",
    )
    article_two = document(
        document_id="article:two",
        kind="article",
        text="Unrelated article",
        url="https://example.com/article/two",
    )
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(
                documents=[
                    document(document_id="post:1"),
                    document(document_id="post:2"),
                    article_one,
                    article_two,
                ],
                followups=[
                    {
                        "reference_id": "ref:one",
                        "post_id": "post:1",
                        "reference_text": "https://example.com/article/one",
                        "candidate_url": "https://example.com/article/one",
                        "investment_related": "yes",
                        "screen_reason": "Reviewed article.",
                        "screen_version": "manual-v1",
                        "status": "resolved",
                        "article_id": "article:one",
                        "attempted_at": stamp,
                        "lookup_method": "reviewed",
                        "match_basis": "Exact persisted article ID.",
                        "evidence_urls": ["https://example.com/post/1"],
                        "error_code": None,
                    },
                    {
                        "reference_id": "ref:two",
                        "post_id": "post:2",
                        "reference_text": "https://example.com/article/two",
                        "candidate_url": "https://example.com/article/two",
                        "investment_related": "yes",
                        "screen_reason": "Reviewed article.",
                        "screen_version": "manual-v1",
                        "status": "resolved",
                        "article_id": "article:two",
                        "attempted_at": stamp,
                        "lookup_method": "reviewed",
                        "match_basis": "Exact persisted article ID.",
                        "evidence_urls": ["https://example.com/post/2"],
                        "error_code": None,
                    },
                ],
            )
        ),
    )
    post = _input("post:1", "original", "Post", available_at=stamp)
    linked = _input(
        "article:one",
        "original",
        "Linked article",
        available_at=stamp,
        source_kind="article",
        url="https://example.com/article/one",
    )
    unrelated = _input(
        "article:two",
        "original",
        "Unrelated article",
        available_at=stamp,
        source_kind="article",
        url="https://example.com/article/two",
    )
    inputs = [post, linked, unrelated]
    approval = _approval(base, "a" * 64, "b" * 64)
    manifest = {
        "bundle_id": base.name,
        "preparation_id": "a" * 64,
        "assessment_id": "b" * 64,
        "approval": approval,
        "admitted_inputs": len(inputs),
        "exclusions": [],
    }
    monkeypatch.setattr(
        "app.services.theme_evaluation.grounding.build_extraction_inputs",
        lambda *args: (inputs, {**manifest}),
    )
    run = {"run_id": "e" * 64, "manifest": manifest, "inputs": inputs, "records": []}
    packet = prepare_grounding(
        run, base, PreparationStore(tmp_path / "store"), {}, stamp
    )
    assert [e["input_id"] for e in packet["contexts"][post.input_id]["evidence"]] == [
        linked.input_id
    ]

    tampered = copy.deepcopy(packet)
    evidence = tampered["contexts"][post.input_id]["evidence"][0]
    evidence.update(
        input_id=unrelated.input_id,
        source_id=unrelated.source_id,
        source_url=unrelated.source_url,
        input_kind=unrelated.input_kind,
        text=unrelated.text,
        original_text_sha256=sha256(unrelated.text.encode()).hexdigest(),
        text_sha256=sha256(unrelated.text.encode()).hexdigest(),
    )
    tampered["digest"] = _digest(tampered)
    with pytest.raises(ValueError, match="grounding_article_relation_mismatch"):
        validate_grounding(
            tampered,
            inputs,
            run_id=run["run_id"],
            manifest=run["manifest"],
            base=base,
        )


def test_image_inherits_only_its_admitted_parent_company_context(
    bundle, document, tmp_path, monkeypatch
):
    """Sparse OCR may use its reviewed parent profile, never another post's."""
    stamp = datetime(2026, 9, 1, tzinfo=timezone.utc)
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(
                documents=[
                    document(document_id="post:nbis"),
                    document(document_id="post:other"),
                ]
            )
        ),
    )
    parent = _input("post:nbis", "original", "Building for META $NBIS", available_at=stamp)
    image = _input("post:nbis", "image_transcription", "Conference slide", available_at=stamp)
    unrelated = _input("post:other", "original", "Other company $OTHER", available_at=stamp)
    inputs = [parent, image, unrelated]
    approval = _approval(base, "a" * 64, "b" * 64)
    manifest = {
        "bundle_id": base.name,
        "preparation_id": "a" * 64,
        "assessment_id": "b" * 64,
        "approval": approval,
        "admitted_inputs": len(inputs),
        "exclusions": [],
    }
    monkeypatch.setattr(
        "app.services.theme_evaluation.grounding.build_extraction_inputs",
        lambda *args: (inputs, {**manifest}),
    )
    run = {"run_id": "f" * 64, "manifest": manifest, "inputs": inputs, "records": []}

    packet = prepare_grounding(
        run,
        base,
        PreparationStore(tmp_path / "store"),
        {
            parent.input_id: {"companies": [_company("NBIS", "Nebius Group")], "warnings": []},
            unrelated.input_id: {"companies": [_company("OTHER", "Other Corp")], "warnings": []},
        },
        stamp,
    )

    image_context = packet["contexts"][image.input_id]
    assert image_context["companies"] == [_company("NBIS", "Nebius Group")]
    assert image_context["warnings"] == [
        f"company_context_inherited_from_parent:{parent.input_id}"
    ]
    assert [(item["input_id"], item["relation"]) for item in image_context["evidence"]] == [
        (parent.input_id, "parent_post")
    ]
    validate_grounding(packet, inputs, run_id=run["run_id"], manifest=manifest)


def test_validation_rejects_image_context_that_claims_a_different_parent_profile(
    bundle, document, tmp_path, monkeypatch
):
    stamp = datetime(2026, 9, 1, tzinfo=timezone.utc)
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(bundle(documents=[document(document_id="post:nbis")])),
    )
    parent = _input("post:nbis", "original", "Building for META $NBIS", available_at=stamp)
    image = _input("post:nbis", "image_transcription", "Conference slide", available_at=stamp)
    inputs = [parent, image]
    approval = _approval(base, "a" * 64, "b" * 64)
    manifest = {
        "bundle_id": base.name,
        "preparation_id": "a" * 64,
        "assessment_id": "b" * 64,
        "approval": approval,
        "admitted_inputs": len(inputs),
        "exclusions": [],
    }
    monkeypatch.setattr(
        "app.services.theme_evaluation.grounding.build_extraction_inputs",
        lambda *args: (inputs, {**manifest}),
    )
    run = {"run_id": "f" * 64, "manifest": manifest, "inputs": inputs, "records": []}
    packet = prepare_grounding(
        run,
        base,
        PreparationStore(tmp_path / "store"),
        {parent.input_id: {"companies": [_company("NBIS", "Nebius Group")], "warnings": []}},
        stamp,
    )
    tampered = copy.deepcopy(packet)
    tampered["contexts"][image.input_id]["companies"] = [_company("OTHER", "Other Corp")]
    tampered["digest"] = _digest(tampered)

    with pytest.raises(
        ValueError, match="grounding_inherited_company_context_mismatch"
    ):
        validate_grounding(tampered, inputs, run_id=run["run_id"], manifest=manifest)
