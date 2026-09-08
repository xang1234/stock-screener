"""Catch stale evidence bindings, tampering and incorrect cache reuse."""

from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import (
    IntegrityError,
    load_bundle,
    seal_bundle,
    sha256,
)
from app.services.theme_evaluation.multilingual_preparation import prepare_text
from app.services.theme_evaluation.preparation_records import (
    Handoff,
    PreparationBinding,
    PreparationManifest,
)
from app.services.theme_evaluation.preparation_results import (
    ImageRequest,
    ImageResult,
    TextRequest,
    TextResult,
)
from app.services.theme_evaluation.preparation_store import (
    PreparationStore,
    validate_handoff,
)
from app.services.theme_evaluation.records import Bundle


def image_result(store):
    asset = store.save_asset(b"original pixels")
    request = ImageRequest(
        input_sha256=asset,
        provider="opencode-go",
        model="kimi-k2.6",
        policy_version="image-v1",
    )
    return ImageResult(
        request=request,
        payload={
            "transcription": "100",
            "observations": [],
            "image_type": "table",
            "uncertainties": [],
        },
        assets=[asset],
        created_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
    )


def manifest_for(base, binding):
    return PreparationManifest(
        bundle_id=base.name,
        handoff=Handoff(bundle_id=base.name),
        bindings=[binding],
        current={binding.slot_id: binding.binding_id},
    )


def test_handoff_rejects_unknown_and_stale_documents(bundle, tmp_path):
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    for key, digest in [("missing", "a" * 64), ("post:1", "a" * 64)]:
        mapping = Handoff(
            bundle_id=base.name, documents={key: {"source_text_sha256": digest}}
        )
        with pytest.raises(ValueError):
            validate_handoff(base, mapping)


def test_success_cache_preserves_timestamp_and_checks_assets(tmp_path):
    store = PreparationStore(tmp_path)
    result = image_result(store)
    result_id = store.save_result(result)
    cached = store.cached(result.request)
    assert cached[0] == result_id
    assert cached[1].created_at.year == 2026
    for field, value in [("model", "changed"), ("policy_version", "image-v2")]:
        assert store.cached(result.request.model_copy(update={field: value})) is None
    (tmp_path / "assets" / result.assets[0]).write_bytes(b"corrupt")
    with pytest.raises(IntegrityError):
        store.cached(result.request)


def test_partial_results_are_retryable_and_tampering_detected(tmp_path):
    store = PreparationStore(tmp_path)
    request = TextRequest(
        input_sha256=sha256(b"text"),
        provider="none",
        model=None,
        policy_version="text-v1",
    )
    result_id = store.save_result(
        TextResult(request=request, payload=prepare_text("text"))
    )
    assert store.cached(request) is None
    (tmp_path / "results" / (result_id + ".json")).write_text("{}")
    with pytest.raises(IntegrityError):
        store.load_result(result_id)


def test_manifest_rejects_stale_binding_and_keeps_base_unchanged(bundle, tmp_path):
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    original = (base / "bundle.json").read_bytes()
    store = PreparationStore(tmp_path / "prepared")
    request = TextRequest(
        input_sha256=sha256(b"text"),
        provider="none",
        model=None,
        policy_version="text-v1",
    )
    rid = store.save_result(TextResult(request=request, payload=prepare_text("text")))
    binding = PreparationBinding(
        stage="text",
        source_kind="document",
        source_id="post:1",
        source_text_sha256="a" * 64,
        result_id=rid,
    )
    with pytest.raises(ValueError, match="source"):
        store.seal(base, manifest_for(base, binding))
    assert (base / "bundle.json").read_bytes() == original


def test_malformed_image_and_missing_translation_are_rejected_at_construction(tmp_path):
    store = PreparationStore(tmp_path)
    valid = image_result(store)
    with pytest.raises(ValueError):
        ImageResult(request=valid.request, payload={"transcription": "100"})
    request = TextRequest(
        input_sha256=sha256("売上".encode()),
        language="ja",
        provider="test",
        model="test",
        policy_version="v1",
    )
    with pytest.raises(ValueError, match="translation_segment_missing"):
        TextResult(
            request=request,
            payload={
                "source_language": "ja",
                "supplied_language": "ja",
                "target_language": "en",
                "segments": [
                    {"original": "売上", "translated": None, "status": "translated"}
                ],
            },
        )


def test_image_cannot_be_attached_to_unrelated_post(bundle, tmp_path):
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    store = PreparationStore(tmp_path / "out")
    rid = store.save_result(image_result(store))
    doc = load_bundle(base).documents[0]
    binding = PreparationBinding(
        stage="image",
        source_kind="document",
        source_id=doc.document_id,
        source_text_sha256=doc.text_sha256,
        result_id=rid,
    )
    with pytest.raises(ValueError, match="image"):
        store.seal(base, manifest_for(base, binding))


def test_verification_repairs_index_after_interrupted_write(tmp_path):
    store = PreparationStore(tmp_path)
    result = image_result(store)
    rid = store.save_result(result)
    index = next((tmp_path / "cache").glob("*/*"))
    index.unlink()
    assert store.cached(result.request) is None
    store.verify_all()
    assert store.cached(result.request)[0] == rid


def test_cache_rejects_entry_for_wrong_request(tmp_path):
    store = PreparationStore(tmp_path)
    first = image_result(store)
    first_id = store.save_result(first)
    second = first.model_copy(
        update={"request": first.request.model_copy(update={"model": "other"})}
    )
    second_id = store.save_result(second)
    index = next(
        path for path in (tmp_path / "cache").glob("*/*") if path.name == first_id
    )
    index.rename(index.parent / second_id)
    with pytest.raises(IntegrityError, match="cache_mismatch"):
        store.cached(first.request)
    with pytest.raises(IntegrityError, match="cache_mismatch"):
        store.verify_all()
