"""Behavioral regressions from the evidence-preparation quality review."""

from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import IntegrityError, seal_bundle, sha256
from app.services.theme_evaluation.multilingual_preparation import TextPreparation
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_results import (
    ImageRequest,
    ImageResult,
    TextRequest,
)
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.preparation_translation_import import (
    import_translations,
)
from app.services.theme_evaluation.records import Bundle


def translated_source(bundle, document, tmp_path):
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(text="매출 증가", original_language="ko")])
        ),
    )
    store = PreparationStore(tmp_path / "out")
    handoff = Handoff(bundle_id=base.name)
    pid = prepare(base, store, handoff, stages=["text"])
    original = store.load(base, pid).bindings[0]
    row = {
        "result_id": original.result_id,
        "source_text_sha256": original.source_text_sha256,
        "translations": ["Revenue growth"],
        "provider": "human",
        "model": "human",
        "policy_version": "manual-v1",
        "generated_at": datetime(2026, 9, 8, tzinfo=timezone.utc),
    }
    return base, store, handoff, pid, row


def test_offline_retry_retains_imported_translation(bundle, document, tmp_path):
    base, store, handoff, pid, row = translated_source(bundle, document, tmp_path)
    imported = import_translations(base, store, pid, [row])
    retried = prepare(base, store, handoff, stages=["text"], prior_id=imported)
    manifest = store.load(base, retried)
    current = [store.load_result(b.result_id) for b in manifest.current_bindings]
    assert len(current) == 1
    assert current[0].status == "success"
    assert current[0].request.provider == "human"


def test_foreign_identity_cannot_be_constructed_as_valid_text():
    with pytest.raises(ValueError, match="identity_requires_matching_language"):
        TextPreparation(
            source_language="ko",
            supplied_language="ko",
            target_language="en",
            segments=[{"original": "매출", "translated": "매출", "status": "identity"}],
            language_warnings=[],
        )


def test_import_does_not_resegment_stored_source(
    bundle, document, tmp_path, monkeypatch
):
    base, store, _, pid, row = translated_source(bundle, document, tmp_path)

    def changed_segmentation(*args, **kwargs):
        raise AssertionError("import must use stored segments")

    monkeypatch.setattr(
        "app.services.theme_evaluation.multilingual_preparation.segment_text",
        changed_segmentation,
    )
    result = import_translations(base, store, pid, [row])
    assert (
        store.load_result(store.load(base, result).bindings[-1].result_id).status
        == "success"
    )


def test_unrelated_asset_corruption_does_not_break_cache_miss(tmp_path):
    store = PreparationStore(tmp_path)
    asset = store.save_asset(b"image bytes")
    request = ImageRequest(
        input_sha256=asset, provider="test", model="test", policy_version="v1"
    )
    result = ImageResult(
        request=request,
        assets=[asset],
        payload={
            "transcription": "100",
            "observations": [],
            "image_type": "table",
            "uncertainties": [],
        },
    )
    rid = store.save_result(result)
    (tmp_path / "assets" / asset).write_bytes(b"corrupted")
    other = TextRequest(
        input_sha256=sha256(b"new text"),
        provider="test",
        model="test",
        policy_version="v1",
    )
    assert store.cached(other) is None
    with pytest.raises(IntegrityError):
        store.load_result(rid)

    with pytest.raises(IntegrityError):
        store.verify_all()


def test_selection_is_independent_of_history_order(bundle, document, tmp_path):
    import csv

    from app.services.theme_evaluation.preparation_records import PreparationManifest
    from app.services.theme_evaluation.preparation_review import render_preparation

    base, store, _, pid, row = translated_source(bundle, document, tmp_path)
    imported = import_translations(base, store, pid, [row])
    manifest = store.load(base, imported)
    reordered = PreparationManifest(
        **{**manifest.model_dump(), "bindings": list(reversed(manifest.bindings))}
    )
    pid = store.seal(base, reordered)
    render_preparation(base, store, pid, tmp_path / "review")
    with (tmp_path / "review" / "preparations.csv").open() as handle:
        current = [r for r in csv.DictReader(handle) if r["version"] == "current"]
    assert len(current) == 1
    assert current[0]["provider"] == "human"


def test_multisegment_import_preserves_stored_alignment_and_explicit_gaps(
    bundle, document, tmp_path, monkeypatch
):
    from app.services.theme_evaluation.multilingual_preparation import prepare_text
    from app.services.theme_evaluation.preparation_records import PreparationBinding
    from app.services.theme_evaluation.preparation_results import TextResult
    from app.services.theme_evaluation.preparation_state import PreparationState

    text = "매출 증가\n\n고용 증가\r\n\r\n\n\n수출 증가"
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(text=text, original_language="ko")])
        ),
    )
    store = PreparationStore(tmp_path / "out")
    source = prepare_text(text, language="ko", max_chars=7)
    request = TextRequest(
        input_sha256=sha256(text.encode()),
        language="ko",
        max_chars=7,
        provider="local",
        model=None,
        policy_version="text-v1",
    )
    rid = store.save_result(TextResult(request=request, payload=source))
    state = PreparationState(base, store, Handoff(bundle_id=base.name))
    state.record(
        PreparationBinding(
            stage="text",
            source_kind="document",
            source_id="post:1",
            source_text_sha256=request.input_sha256,
            result_id=rid,
        )
    )
    pid = store.seal(base, state.manifest)
    translations = [
        s.original if s.status == "identity" else f"Translation {i}"
        for i, s in enumerate(source.segments)
    ]
    translations[-1] = None

    def changed_algorithm(*args, **kwargs):
        pytest.fail("imports must preserve stored language and segmentation")

    monkeypatch.setattr(
        "app.services.theme_evaluation.multilingual_preparation.segment_text",
        changed_algorithm,
    )
    monkeypatch.setattr(
        "app.services.theme_evaluation.multilingual_preparation.detect_language",
        changed_algorithm,
    )
    row = {
        "result_id": rid,
        "source_text_sha256": request.input_sha256,
        "translations": translations,
        "provider": "human",
        "model": "human",
        "policy_version": "manual-v1",
        "generated_at": datetime(2026, 9, 8, tzinfo=timezone.utc),
    }
    imported = import_translations(base, store, pid, [row])
    result = store.load_result(store.load(base, imported).current_bindings[0].result_id)
    assert result.status == "partial"
    assert [s.original for s in result.payload.segments] == [
        s.original for s in source.segments
    ]
    assert [s.translated for s in result.payload.segments] == translations
    assert result.payload.supplied_language == "ko"
    assert import_translations(base, store, imported, [row]) == imported
