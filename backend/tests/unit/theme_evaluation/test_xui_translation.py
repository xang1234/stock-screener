"""X-rendered translation binding, selection, and preparation contracts."""

import copy
from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import load_bundle, seal_bundle, sha256
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.xui_intake import import_xui

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)


def payloads_with_translation(payloads):
    for payload in payloads.values():
        row = payload["items"][0]
        row.update(
            text="매출 10% 증가\n\n전망 유지",
            lang="ko",
            text_complete=True,
            x_translation={
                "status": "captured",
                "text": "Revenue increased 10%.\n\nOutlook unchanged.",
                "post_id": "1",
                "original_text_sha256": "sha256:"
                + sha256("매출 10% 증가\n\n전망 유지".encode()),
                "source_language": "ko",
                "target_language": "en",
                "captured_at": NOW.isoformat(),
                "provider": "X translation",
                "display_mode": "on_demand",
                "failure_reason": None,
            },
        )
    return payloads


def ingest(payloads):
    return import_xui(
        payloads, captured_at=NOW, max_posts_per_source=50, mode="controlled"
    )


def test_captured_translation_is_preferred_without_model_call(xui_payloads, tmp_path):
    bundle = ingest(payloads_with_translation(xui_payloads))
    base = seal_bundle(tmp_path, bundle)
    assert (
        load_bundle(base)
        .documents[0]
        .source_metadata.x_translation.text.startswith("Revenue")
    )

    class ForbiddenTranslator:
        provider = "test"
        model = "test"
        policy_version = "test"

        def __call__(self, *args):
            pytest.fail("Captured X translation must not call a model")

    store = PreparationStore(tmp_path / "prepared")
    pid = prepare(
        base,
        store,
        Handoff(bundle_id=base.name),
        stages=["text"],
        translator=ForbiddenTranslator(),
    )
    result = store.load_result(store.load(base, pid).current_bindings[0].result_id)
    assert result.request.provider == "X translation"
    assert result.created_at == NOW
    assert result.source_text == "매출 10% 증가\n\n전망 유지"
    assert (
        "".join(s.translated for s in result.payload.segments)
        == "Revenue increased 10%.\n\nOutlook unchanged."
    )


@pytest.mark.parametrize(
    "change",
    [
        {"post_id": "2"},
        {"original_text_sha256": "sha256:" + "0" * 64},
        {"status": "capture_failed"},
        {"target_language": "fr"},
        {"text": "   "},
    ],
)
def test_invalid_x_translation_cannot_enter_bundle(xui_payloads, change):
    payloads = payloads_with_translation(xui_payloads)
    for p in payloads.values():
        p["items"][0]["x_translation"].update(change)
    with pytest.raises(ValueError):
        ingest(payloads)


def test_translation_from_second_list_wins_over_failed_first_capture(xui_payloads):
    payloads = payloads_with_translation(xui_payloads)
    first = payloads["1986290701492232693"]["items"][0]["x_translation"]
    first.update(status="unavailable", text=None, failure_reason="control_missing")
    assert (
        ingest(payloads).documents[0].source_metadata.x_translation.status == "captured"
    )


def test_failed_x_translation_preserves_reason_and_uses_fallback(
    xui_payloads, tmp_path
):
    payloads = payloads_with_translation(xui_payloads)
    for p in payloads.values():
        p["items"][0]["x_translation"].update(
            status="capture_failed", text=None, failure_reason="original_mismatch"
        )
    bundle = ingest(payloads)
    assert (
        bundle.documents[0].source_metadata.x_translation.failure_reason
        == "original_mismatch"
    )
    base = seal_bundle(tmp_path, bundle)

    class Translator:
        provider = "fallback"
        model = "test"
        policy_version = "test"

        def __call__(self, text, source, target):
            return "Fallback English"

    store = PreparationStore(tmp_path / "prepared")
    pid = prepare(
        base,
        store,
        Handoff(bundle_id=base.name),
        stages=["text"],
        translator=Translator(),
    )
    result = store.load_result(store.load(base, pid).current_bindings[0].result_id)
    assert result.request.provider == "fallback"
    assert all(s.translated == "Fallback English" for s in result.payload.segments)


def test_new_x_capture_does_not_reuse_previous_translation(xui_payloads, tmp_path):
    payloads = payloads_with_translation(xui_payloads)
    store = PreparationStore(tmp_path / "prepared")
    for translated in ("Revenue increased 10%.", "Revenue rose by 10%."):
        new = copy.deepcopy(payloads)
        for p in new.values():
            p["items"][0]["x_translation"]["text"] = translated
        base = seal_bundle(tmp_path, ingest(new))
        pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])
        result = store.load_result(store.load(base, pid).current_bindings[0].result_id)
        assert result.payload.segments[0].translated == translated
