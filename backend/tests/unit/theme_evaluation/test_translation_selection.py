"""Translation candidate selection and immutable sidecar contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import IntegrityError, seal_bundle, sha256
from app.services.theme_evaluation.multilingual_preparation import (
    TextPreparation,
    TranslationSegment,
)
from app.services.theme_evaluation.preparation_records import (
    Handoff,
    PreparationBinding,
    PreparationManifest,
)
from app.services.theme_evaluation.preparation_results import TextRequest, TextResult
from app.services.theme_evaluation.preparation_state import PreparationState
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.records import Bundle
from app.services.theme_evaluation.translation_selection import select_translation
from app.services.theme_evaluation.translation_selection_store import (
    load_translation_selection,
    save_translation_selection,
    selection_for_preparation,
    selection_record,
)

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)


def candidate(
    original,
    translated,
    *,
    provider,
    segments=None,
    target_language="en",
):
    if segments is None:
        segments = [
            TranslationSegment(
                original=original,
                translated=translated,
                status="translated" if translated is not None else "unavailable",
                failure_reason=None if translated is not None else "translation_failed",
            )
        ]
    return TextResult(
        request=TextRequest(
            input_sha256=sha256(original.encode()),
            provider=provider,
            model=None,
            policy_version=(
                "x-rendered-v1" if provider == "X translation" else "candidate-v1"
            ),
            language="ko",
            target_language=target_language,
            method="translation_import" if provider == "X translation" else "prepare",
        ),
        payload=TextPreparation(
            source_language="ko",
            supplied_language="ko",
            target_language=target_language,
            segments=segments,
        ),
        created_at=NOW,
    )


def sealed_candidates(tmp_path, bundle, document, x_text, kimi_text):
    original = "718만주 1.9조원에 매수"
    doc = document(
        text=original,
        original_language="ko",
        source_metadata={
            "x_translation": {
                "status": "captured",
                "text": x_text,
                "post_id": "1",
                "original_text_sha256": "sha256:" + sha256(original.encode()),
                "source_language": "ko",
                "target_language": "en",
                "captured_at": NOW,
                "provider": "X translation",
                "display_mode": "on_demand",
                "failure_reason": None,
            }
        },
    )
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(bundle(documents=[doc])),
    )
    store = PreparationStore(tmp_path / "prepared")
    x_result = candidate(original, x_text, provider="X translation")
    kimi_result = candidate(original, kimi_text, provider="opencode-go")
    x_id = store.save_result(x_result)
    kimi_id = store.save_result(kimi_result)
    bindings = [
        PreparationBinding(
            stage="text",
            source_kind="document",
            source_id="post:1",
            source_text_sha256=sha256(original.encode()),
            result_id=result_id,
        )
        for result_id in (x_id, kimi_id)
    ]
    manifest = PreparationManifest(
        bundle_id=base.name,
        handoff=Handoff(bundle_id=base.name),
        bindings=bindings,
        current={bindings[-1].slot_id: bindings[-1].binding_id},
    )
    preparation_id = store.seal(base, manifest)
    return base, store, preparation_id, x_result, x_id, kimi_result, kimi_id


def test_selector_uses_adequate_x_and_is_frozen():
    original = "매출 10% 증가"
    x_result = candidate(original, "Revenue increased 10%", provider="X translation")

    decision = select_translation(original, "ko", x_result, None)

    assert decision.selected_provider == "X translation"
    assert decision.selected_candidate == "x"
    assert decision.eligible is True
    assert decision.assessment.policy_version == "translation-quality-v2"
    assert decision.assessment.disposition == "use"
    with pytest.raises(FrozenInstanceError):
        decision.eligible = False


def test_selector_keeps_ambiguous_x_pending_review_without_second_candidate():
    original = "삼성전자는 새로운 공장을 건설할 예정이라고 오늘 공식 발표했습니다"
    x_result = candidate(original, "New factory", provider="X translation")

    decision = select_translation(original, "ko", x_result, None)

    assert decision.selected_candidate == "x"
    assert decision.eligible is False
    assert decision.assessment.disposition == "review"


@pytest.mark.parametrize(
    "kimi_text", [None, "Bought 7.18 million shares for 1.8 trillion won"]
)
def test_rejected_x_cannot_become_eligible_when_kimi_has_a_blocker(kimi_text):
    original = "718만주 1.9조원에 매수"
    x_result = candidate(original, "Samsung", provider="X translation")
    kimi_result = candidate(original, kimi_text, provider="opencode-go")

    decision = select_translation(original, "ko", x_result, kimi_result)

    assert decision.selected_provider is None
    assert decision.selected_candidate is None
    assert decision.eligible is False
    assert decision.assessment.disposition == "fallback"
    assert any(issue.severity == "blocker" for issue in decision.issues)


def test_incomplete_multisegment_candidate_is_never_eligible():
    original = "매출 증가\n\n고용 증가"
    partial = candidate(
        original,
        None,
        provider="opencode-go",
        segments=[
            TranslationSegment(
                original="매출 증가\n\n",
                translated="Revenue grew\n\n",
                status="translated",
            ),
            TranslationSegment(
                original="고용 증가",
                translated=None,
                status="unavailable",
                failure_reason="translation_failed",
            ),
        ],
    )

    decision = select_translation(original, "ko", None, partial)

    assert decision.selected_result is None
    assert decision.eligible is False
    assert decision.assessment.disposition == "fallback"


def test_selector_rejects_candidate_bound_to_different_source():
    result = candidate("다른 원문", "Different source", provider="X translation")
    with pytest.raises(ValueError, match="translation_candidate_source_mismatch"):
        select_translation("매출 10% 증가", "ko", result, None)


@pytest.mark.parametrize(
    "original, expected_disposition, expected_eligible",
    [("", "review", False), ("   ", "review", False)],
)
def test_empty_or_whitespace_document_gets_an_explicit_identity_decision(
    tmp_path,
    bundle,
    document,
    original,
    expected_disposition,
    expected_eligible,
):
    from app.services.theme_evaluation.preparation_pipeline import prepare

    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(text=original, original_language=None)])
        ),
    )
    store = PreparationStore(tmp_path / "prepared")

    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])

    _, sidecar = selection_for_preparation(base, store, pid)
    decision = sidecar.decisions[0]
    assert decision.kimi_result_id is not None
    assert decision.selected_result_id == decision.kimi_result_id
    assert decision.disposition == expected_disposition
    assert decision.eligible is expected_eligible


def test_absent_candidates_are_explicitly_unresolved():
    decision = select_translation("", None, None, None)

    assert decision.selected_result is None
    assert decision.eligible is False
    assert decision.assessment.disposition == "fallback"
    assert [issue.code for issue in decision.issues] == [
        "translation_candidate_missing"
    ]


def test_non_english_candidate_cannot_be_selected_or_persisted(
    tmp_path, bundle, document
):
    original = "점유율 10%"
    french = candidate(
        original,
        "Part de marché 10%",
        provider="opencode-go",
        target_language="fr",
    )
    with pytest.raises(ValueError, match="translation_candidate_target_mismatch"):
        select_translation(original, "ko", None, french)

    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(text=original, original_language="ko")])
        ),
    )
    store = PreparationStore(tmp_path / "prepared")
    result_id = store.save_result(french)
    binding = PreparationBinding(
        stage="text",
        source_kind="document",
        source_id="post:1",
        source_text_sha256=sha256(original.encode()),
        result_id=result_id,
    )
    manifest = PreparationManifest(
        bundle_id=base.name,
        handoff=Handoff(bundle_id=base.name),
        bindings=[binding],
        current={binding.slot_id: binding.binding_id},
    )
    pid = store.seal(base, manifest)
    forged = {
        "document_id": "post:1",
        "source_text_sha256": sha256(original.encode()),
        "x_result_id": None,
        "kimi_result_id": result_id,
        "selected_result_id": result_id,
        "eligible": True,
        "disposition": "use",
        "issues": [],
    }

    with pytest.raises(ValueError, match="translation_candidate_target_mismatch"):
        save_translation_selection(base, store, pid, [forged])


def test_no_translator_candidate_is_explicitly_unresolved(tmp_path, bundle, document):
    from app.services.theme_evaluation.preparation_pipeline import prepare

    original = "매출 증가"
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(text=original, original_language="ko")])
        ),
    )
    store = PreparationStore(tmp_path / "prepared")

    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])

    _, sidecar = selection_for_preparation(base, store, pid)
    record = sidecar.decisions[0]
    assert record.x_result_id is None
    assert record.kimi_result_id is not None
    assert record.selected_result_id is None
    assert record.eligible is False
    assert record.disposition == "fallback"


def test_english_article_identity_does_not_call_translator(tmp_path, bundle, document):
    from app.services.theme_evaluation.preparation_pipeline import prepare

    original = "Revenue increased 10%."
    article = document(
        document_id="article:1",
        kind="article",
        text=original,
        original_language="en",
    )
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(
                documents=[document(original_language="en"), article],
                followups=[
                    {
                        "reference_id": "ref:1",
                        "post_id": "post:1",
                        "reference_text": "https://example.com/article",
                        "candidate_url": "https://example.com/article",
                        "investment_related": "yes",
                        "screen_reason": "linked article",
                        "screen_version": "test-v1",
                        "status": "resolved",
                        "article_id": "article:1",
                        "attempted_at": NOW,
                        "lookup_method": "fixture",
                        "match_basis": "exact URL",
                        "evidence_urls": ["https://example.com/article"],
                        "error_code": None,
                    }
                ],
            )
        ),
    )

    class ForbiddenTranslator:
        provider = "opencode-go"
        model = "kimi-k2.6"
        policy_version = "translation-v3"

        def __call__(self, *args):
            pytest.fail("English identity preparation must not call the model")

    store = PreparationStore(tmp_path / "prepared")
    pid = prepare(
        base,
        store,
        Handoff(bundle_id=base.name),
        stages=["text"],
        translator=ForbiddenTranslator(),
    )

    _, sidecar = selection_for_preparation(base, store, pid)
    record = next(
        decision
        for decision in sidecar.decisions
        if decision.document_id == "article:1"
    )
    assert record.selected_result_id == record.kimi_result_id
    assert record.eligible is True
    assert (
        store.load_result(record.selected_result_id).payload.segments[0].status
        == "identity"
    )


def test_legacy_current_x_is_not_eligible_after_failed_kimi(tmp_path, bundle, document):
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
                "captured_at": NOW,
                "provider": "X translation",
                "display_mode": "on_demand",
                "failure_reason": None,
            }
        },
    )
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(bundle(documents=[doc])),
    )
    store = PreparationStore(tmp_path / "prepared")
    handoff = Handoff(bundle_id=base.name)
    state = PreparationState(base, store, handoff)
    x_result = candidate(original, "Samsung", provider="X translation")
    kimi_result = candidate(original, None, provider="opencode-go")
    x_id = store.save_result(x_result)
    kimi_id = store.save_result(kimi_result)
    for result_id in (x_id, kimi_id):
        state.record(
            PreparationBinding(
                stage="text",
                source_kind="document",
                source_id="post:1",
                source_text_sha256=sha256(original.encode()),
                result_id=result_id,
            )
        )
    pid = store.seal(base, state.manifest)
    choice = select_translation(original, "ko", x_result, kimi_result)
    sidecar_id = save_translation_selection(
        base, store, pid, [selection_record("post:1", x_id, kimi_id, choice)]
    )

    current = state.manifest.current_bindings[0]
    assert current.result_id == x_id
    decision = load_translation_selection(base, store, pid, sidecar_id).decisions[0]
    assert decision.selected_result_id is None
    assert decision.eligible is False


def test_sidecar_round_trip_is_content_addressed_and_bound_to_exact_inputs(
    tmp_path, bundle, document
):
    base, store, pid, x_result, x_id, kimi_result, kimi_id = sealed_candidates(
        tmp_path,
        bundle,
        document,
        "Samsung",
        "Bought 7.18 million shares for 1.9 trillion won",
    )
    choice = select_translation(x_result.source_text, "ko", x_result, kimi_result)
    entry = selection_record("post:1", x_id, kimi_id, choice)

    sidecar_id = save_translation_selection(base, store, pid, [entry])
    loaded_id, loaded = selection_for_preparation(base, store, pid)

    assert loaded_id == sidecar_id
    assert loaded == load_translation_selection(base, store, pid, sidecar_id)
    assert loaded.schema_version == 1
    assert loaded.policy_version == "translation-quality-v2"
    assert loaded.bundle_id == base.name
    assert loaded.preparation_id == pid
    assert loaded.decisions[0].source_text_sha256 == sha256(
        x_result.source_text.encode()
    )
    assert loaded.decisions[0].x_result_id == x_id
    assert loaded.decisions[0].kimi_result_id == kimi_id
    assert loaded.decisions[0].selected_result_id == kimi_id
    assert loaded.decisions[0].eligible is True
    assert (store.root / "selection-decisions" / f"{sidecar_id}.json").is_file()


def test_sidecar_rejects_forged_eligibility_and_selected_candidate_mismatch(
    tmp_path, bundle, document
):
    base, store, pid, x_result, x_id, kimi_result, kimi_id = sealed_candidates(
        tmp_path, bundle, document, "Samsung", None
    )
    choice = select_translation(x_result.source_text, "ko", x_result, kimi_result)
    entry = selection_record("post:1", x_id, kimi_id, choice).model_dump(mode="json")

    entry.update(eligible=True, selected_result_id=x_id)
    with pytest.raises(ValueError, match="selection_decision_inconsistent"):
        save_translation_selection(base, store, pid, [entry])


def test_sidecar_rejects_bound_result_that_is_not_the_document_x_capture(
    tmp_path, bundle, document
):
    original = "718만주 1.9조원"
    actual_text = "7.18 million shares, 1.9 trillion won"
    doc = document(
        text=original,
        original_language="ko",
        source_metadata={
            "x_translation": {
                "status": "captured",
                "text": actual_text,
                "post_id": "1",
                "original_text_sha256": "sha256:" + sha256(original.encode()),
                "source_language": "ko",
                "target_language": "en",
                "captured_at": NOW,
                "provider": "X translation",
                "display_mode": "on_demand",
                "failure_reason": None,
            }
        },
    )
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle(documents=[doc])))
    store = PreparationStore(tmp_path / "prepared")
    forged = candidate(
        original,
        "1.9 trillion won for 7.18 million shares",
        provider="X translation",
    )
    forged_id = store.save_result(forged)
    binding = PreparationBinding(
        stage="text",
        source_kind="document",
        source_id="post:1",
        source_text_sha256=sha256(original.encode()),
        result_id=forged_id,
    )
    manifest = PreparationManifest(
        bundle_id=base.name,
        handoff=Handoff(bundle_id=base.name),
        bindings=[binding],
        current={binding.slot_id: binding.binding_id},
    )
    pid = store.seal(base, manifest)
    choice = select_translation(original, "ko", forged, None)

    with pytest.raises(ValueError, match="selection_x_capture_mismatch"):
        save_translation_selection(
            base,
            store,
            pid,
            [selection_record("post:1", forged_id, None, choice)],
        )


def test_sidecar_requires_a_decision_for_every_root_document_translation(
    tmp_path, bundle, document
):
    base, store, pid, *_ = sealed_candidates(
        tmp_path,
        bundle,
        document,
        "Samsung",
        "Bought 7.18 million shares for 1.9 trillion won",
    )

    with pytest.raises(ValueError, match="selection_decision_coverage_mismatch"):
        save_translation_selection(base, store, pid, [])


def test_sidecar_load_rejects_wrong_preparation_id(tmp_path, bundle, document):
    base, store, pid, x_result, x_id, kimi_result, kimi_id = sealed_candidates(
        tmp_path,
        bundle,
        document,
        "Samsung",
        "Bought 7.18 million shares for 1.9 trillion won",
    )
    choice = select_translation(x_result.source_text, "ko", x_result, kimi_result)
    sidecar_id = save_translation_selection(
        base, store, pid, [selection_record("post:1", x_id, kimi_id, choice)]
    )

    with pytest.raises(ValueError, match="selection_preparation_mismatch"):
        load_translation_selection(base, store, "0" * 64, sidecar_id)


def test_nontext_stage_preserves_missing_sidecar_from_legacy_preparation(
    tmp_path, bundle, document
):
    from app.services.theme_evaluation.preparation_pipeline import prepare

    base, store, pid, *_ = sealed_candidates(
        tmp_path,
        bundle,
        document,
        "Samsung",
        "Bought 7.18 million shares for 1.9 trillion won",
    )
    handoff = store.load(base, pid).handoff

    next_pid = prepare(base, store, handoff, stages=["image"], prior_id=pid)

    assert store.load(base, next_pid).bindings == store.load(base, pid).bindings
    with pytest.raises(IntegrityError, match="missing_translation_selection"):
        selection_for_preparation(base, store, next_pid)


def test_v1_and_v2_sidecars_coexist_and_discovery_prefers_latest_policy(
    tmp_path, bundle, document
):
    base, store, pid, x_result, x_id, kimi_result, kimi_id = sealed_candidates(
        tmp_path,
        bundle,
        document,
        "Samsung",
        "Bought 7.18 million shares for 1.9 trillion won",
    )

    v1_choice = select_translation(
        x_result.source_text,
        "ko",
        x_result,
        kimi_result,
        policy_version="translation-quality-v1",
    )
    v1_id = save_translation_selection(
        base,
        store,
        pid,
        [selection_record("post:1", x_id, kimi_id, v1_choice)],
        policy_version="translation-quality-v1",
    )
    v2_choice = select_translation(
        x_result.source_text,
        "ko",
        x_result,
        kimi_result,
        policy_version="translation-quality-v2",
    )
    v2_id = save_translation_selection(
        base,
        store,
        pid,
        [selection_record("post:1", x_id, kimi_id, v2_choice)],
    )

    selected_id, selected = selection_for_preparation(base, store, pid)
    archived_id, archived = selection_for_preparation(
        base,
        store,
        pid,
        policy_version="translation-quality-v1",
    )
    assert (selected_id, selected.policy_version) == (
        v2_id,
        "translation-quality-v2",
    )
    assert (archived_id, archived.policy_version) == (
        v1_id,
        "translation-quality-v1",
    )
    assert load_translation_selection(base, store, pid, v1_id) == archived
