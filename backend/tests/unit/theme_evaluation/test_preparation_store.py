"""Catch stale evidence bindings, tampering and incorrect cache reuse."""

from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import IntegrityError, seal_bundle, sha256
from app.services.theme_evaluation.records import Bundle


def api():
    from app.services.theme_evaluation import preparation_records, preparation_store

    return preparation_records, preparation_store


def test_handoff_rejects_unknown_and_stale_documents(bundle, tmp_path):
    r, s = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    for key, digest in [("missing", "a" * 64), ("post:1", "a" * 64)]:
        mapping = r.Handoff(
            bundle_id=base.name, documents={key: {"source_text_sha256": digest}}
        )
        with pytest.raises(ValueError):
            s.validate_handoff(base, mapping)


def test_success_cache_preserves_timestamp_and_checks_assets(tmp_path):
    r, s = api()
    store = s.PreparationStore(tmp_path)
    asset = store.save_asset(b"original pixels")
    request = r.PreparationRequest(
        stage="image",
        input_sha256=asset,
        provider="opencode-go",
        model="kimi-k2.6",
        policy_version="image-v1",
    )
    result = r.PreparationResult(
        request=request,
        status="success",
        payload={
            "transcription": "100",
            "observations": [],
            "image_type": "table",
            "uncertainties": [],
        },
        assets=[asset],
        created_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
    )
    result_id = store.save_result(result)
    cached = store.cached(request)
    assert cached[0] == result_id
    assert cached[1].created_at.year == 2026
    assert store.cached(request.model_copy(update={"model": "changed"})) is None
    assert (
        store.cached(request.model_copy(update={"policy_version": "image-v2"})) is None
    )
    (tmp_path / "assets" / asset).write_bytes(b"corrupt")
    with pytest.raises(IntegrityError):
        store.load_result(result_id)


def test_partial_results_are_retryable_and_tampering_detected(tmp_path):
    r, s = api()
    store = s.PreparationStore(tmp_path)
    request = r.PreparationRequest(
        stage="text",
        input_sha256=sha256(b"text"),
        provider="none",
        model=None,
        policy_version="text-v1",
    )
    result_id = store.save_result(
        r.PreparationResult(
            request=request,
            status="unavailable",
            payload={
                "source_language": "und",
                "supplied_language": None,
                "target_language": "en",
                "segments": [
                    {"original": "text", "translated": None, "status": "unavailable"}
                ],
                "status": "unavailable",
                "warnings": ["missing_translation"],
            },
            warnings=["missing_translation"],
        )
    )
    assert store.cached(request) is None
    (tmp_path / "results" / (result_id + ".json")).write_text("{}")
    with pytest.raises(IntegrityError):
        store.load_result(result_id)


def test_manifest_rejects_stale_binding_and_keeps_base_unchanged(bundle, tmp_path):
    r, s = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    original = (base / "bundle.json").read_bytes()
    store = s.PreparationStore(tmp_path / "prepared")
    mapping = r.Handoff(bundle_id=base.name)
    request = r.PreparationRequest(
        stage="text",
        input_sha256="a" * 64,
        provider="none",
        model=None,
        policy_version="text-v1",
    )
    rid = store.save_result(
        r.PreparationResult(request=request, status="unavailable", warnings=["missing"])
    )
    binding = r.PreparationBinding(
        source_kind="document",
        source_id="post:1",
        source_text_sha256="a" * 64,
        result_id=rid,
    )
    manifest = r.PreparationManifest(
        bundle_id=base.name, handoff=mapping, bindings=[binding]
    )
    with pytest.raises(ValueError, match="source"):
        store.seal(base, manifest)
    assert (base / "bundle.json").read_bytes() == original


def test_malformed_success_and_missing_text_segment_are_rejected(tmp_path):
    r, s = api()
    store = s.PreparationStore(tmp_path)
    for stage, payload in [
        ("image", {"transcription": "100"}),
        (
            "text",
            {
                "source_language": "ja",
                "supplied_language": "ja",
                "target_language": "en",
                "segments": [
                    {"original": "売上", "translated": None, "status": "unavailable"}
                ],
                "status": "success",
                "warnings": [],
            },
        ),
    ]:
        result = r.PreparationResult(
            request=r.PreparationRequest(
                stage=stage,
                input_sha256=sha256("売上".encode()),
                provider="test",
                model="test",
                policy_version="v1",
            ),
            status="success",
            payload=payload,
        )
        with pytest.raises(ValueError):
            store.save_result(result)


def test_image_cannot_be_attached_to_unrelated_post(bundle, tmp_path):
    r, s = api()
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    store = s.PreparationStore(tmp_path / "out")
    asset = store.save_asset(b"pixels")
    rid = store.save_result(
        r.PreparationResult(
            request=r.PreparationRequest(
                stage="image",
                input_sha256=asset,
                provider="test",
                model="test",
                policy_version="v1",
            ),
            status="success",
            assets=[asset],
            payload={
                "transcription": "100",
                "observations": [],
                "image_type": "table",
                "uncertainties": [],
            },
        )
    )
    from app.services.theme_evaluation.bundle import load_bundle

    doc = load_bundle(base).documents[0]
    binding = r.PreparationBinding(
        source_kind="document",
        source_id=doc.document_id,
        source_text_sha256=doc.text_sha256,
        result_id=rid,
    )
    with pytest.raises(ValueError, match="image"):
        store.seal(
            base,
            r.PreparationManifest(
                bundle_id=base.name,
                handoff=r.Handoff(bundle_id=base.name),
                bindings=[binding],
            ),
        )
