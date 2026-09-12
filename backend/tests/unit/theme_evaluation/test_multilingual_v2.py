"""Opt-in language preparation avoids calls without discarding source bytes."""

import json
from collections import Counter
from pathlib import Path

from app.services.theme_evaluation.bundle import sha256
from app.services.theme_evaluation.multilingual_v2 import (
    assess_language,
    preparation_cache_policy,
    prepare_text_v2,
    segment_text_v2,
)
from app.services.theme_evaluation.preparation_results import TextRequest, TextResult
from app.services.theme_evaluation.preparation_store import PreparationStore


def test_original_bytes_survive_segmentation():
    from app.services.theme_evaluation.multilingual_preparation import segment_text

    original = "売上高は1.9兆円。\r\n\r\n매출 718억원.\n\n$NVDA 전망 유지"
    assert "".join(segment_text(original)) == original


def test_v2_packs_paragraphs_without_changing_source_bytes():
    original = "売上高は1.9兆円。\r\n\r\n매출 718억원.\n\n$NVDA 전망 유지"

    segments = segment_text_v2(original, max_chars=28)

    assert "".join(segments) == original
    assert segments == ["売上高は1.9兆円。\r\n\r\n매출 718억원.\n\n", "$NVDA 전망 유지"]
    assert all(len(segment) <= 28 for segment in segments)


def test_v2_long_paragraph_splitting_stays_bounded():
    original = "長" * 8001

    segments = segment_text_v2(original)

    assert [len(segment) for segment in segments] == [4000, 4000, 1]
    assert "".join(segments) == original


def test_valid_english_metadata_avoids_translation_call():
    calls = []

    result = prepare_text_v2(
        "Revenue rose 12% for Product 2.",
        language="en-US",
        translator=lambda *args: calls.append(args),
    )

    assert calls == []
    assert result.source_language == "en"
    assert result.supplied_language == "en-US"
    assert result.segments[0].status == "identity"
    assert result.status == "success"


def test_contradictory_script_metadata_translates_under_detected_language_for_review():
    calls = []

    def translate(text, source, target):
        calls.append((text, source, target))
        return "Revenue increased."

    result = prepare_text_v2(
        "売上高は増加。", language="en", translator=translate
    )

    assert calls == [("売上高は増加。", "ja", "en")]
    assert result.source_language == "ja"
    assert result.supplied_language == "en"
    assert "language_metadata_conflict" in result.warnings
    assert result.status == "needs_review"


def test_han_only_without_provenance_is_not_confidently_chinese():
    calls = []

    result = prepare_text_v2(
        "半導體收入增長。",
        translator=lambda text, source, target: calls.append(source) or "Revenue grew.",
    )

    assert calls == ["und"]
    assert result.source_language == "und"
    assert result.supplied_language is None
    assert "han_language_ambiguous" in result.warnings
    assert "language_metadata_conflict" not in result.warnings


def test_observed_chinese_metadata_resolves_han_ambiguity_without_replacement():
    decision = assess_language("台積電收入增長。", "zh-TW")

    assert decision.source_language == "zh"
    assert decision.supplied_language == "zh-TW"
    assert decision.action == "translate"
    assert "han_language_ambiguous" not in decision.warnings


def test_url_and_emoji_only_content_is_retained_as_nonlinguistic_without_call():
    original = "https://t.co/AbC123 🚀📈"

    result = prepare_text_v2(
        original,
        translator=lambda *args: (_ for _ in ()).throw(
            AssertionError("translator must not be called")
        ),
    )

    assert result.source_language == "zxx"
    assert result.supplied_language is None
    assert "".join(segment.original for segment in result.segments) == original
    assert result.segments[0].status == "unavailable"
    assert result.segments[0].failure_reason == "nonlinguistic_content"
    assert "nonlinguistic_content" in result.warnings


def test_nonlinguistic_source_does_not_become_english_from_metadata():
    decision = assess_language("https://example.test/a 🚀", "en")

    assert decision.source_language == "zxx"
    assert decision.supplied_language == "en"
    assert decision.action == "retain_original"
    assert decision.warnings == [
        "nonlinguistic_content",
        "language_metadata_conflict",
    ]


def test_linguistic_source_overrides_conflicting_zxx_but_preserves_metadata():
    calls = []

    result = prepare_text_v2(
        "삼성전자 매출 증가",
        language="zxx",
        translator=lambda text, source, target: calls.append(source)
        or "Samsung Electronics revenue increased.",
    )

    assert calls == ["ko"]
    assert result.source_language == "ko"
    assert result.supplied_language == "zxx"
    assert "language_metadata_conflict" in result.warnings
    assert result.status == "needs_review"


def test_v2_cache_policy_namespaces_language_and_translator_versions():
    class Translator:
        policy_version = "translation-v3"

    assert (
        preparation_cache_policy(Translator())
        == "text-language-efficient-v2+translation-v3"
    )
    assert (
        preparation_cache_policy(None)
        == "text-language-efficient-v2+translator-unavailable"
    )


def test_mixed_language_prose_remains_translatable():
    calls = []
    original = "Market update: 売上は増加, 하지만 전망은 flat."

    result = prepare_text_v2(
        original,
        translator=lambda text, source, target: calls.append(
            (text, source, target)
        )
        or "Market update: revenue increased, but the outlook is flat.",
    )

    assert calls == [(original, "mixed", "en")]
    assert result.source_language == "mixed"
    assert "mixed_scripts" in result.warnings
    assert result.segments[0].status == "translated"


def test_translation_failure_only_marks_its_exact_stored_segment():
    original = "第一。\n\n第二。\n\n第三。"

    def translate(segment, source, target):
        if segment == "第二。\n\n":
            raise RuntimeError("private detail")
        return {"第一。\n\n": "First.\n\n", "第三。": "Third."}[segment]

    result = prepare_text_v2(
        original, language="ja", translator=translate, max_chars=7
    )

    assert "".join(segment.original for segment in result.segments) == original
    assert [segment.status for segment in result.segments] == [
        "translated",
        "unavailable",
        "translated",
    ]
    assert result.segments[1].failure_reason == "translation_segment_failed"
    assert "private detail" not in result.model_dump_json()
    assert result.status == "partial"


def test_language_challenge_has_twenty_labeled_synthetic_cases():
    path = Path(__file__).parent / "fixtures" / "language_challenge.json"
    challenge = json.loads(path.read_text(encoding="utf-8"))

    assert challenge["provenance"]["kind"] == "synthetic"
    assert challenge["provenance"]["live_model_validated"] is False
    assert len(challenge["cases"]) == 20
    assert Counter(case["category"] for case in challenge["cases"]) == {
        "korean": 4,
        "japanese": 4,
        "chinese": 4,
        "mixed": 4,
        "english_nonlinguistic": 4,
    }
    assert all(
        case["synthetic"]
        and case["label"]
        and case["expected_meaning"]
        and "expected_quantity_facts" in case
        for case in challenge["cases"]
    )

    for case in challenge["cases"]:
        decision = assess_language(case["text"], case["source_language"])
        assert decision.action == case["preparation_expectation"], case["id"]


def test_v2_text_result_round_trips_and_uses_distinct_cache_namespace(tmp_path):
    original = "Revenue rose 12%."
    request = TextRequest(
        input_sha256=sha256(original.encode()),
        provider="local",
        model=None,
        policy_version=preparation_cache_policy(None),
        language="en",
        target_language="en",
        max_chars=4000,
    )
    result = TextResult(
        request=request,
        payload=prepare_text_v2(original, language="en"),
    )
    store = PreparationStore(tmp_path)

    result_id = store.save_result(result)
    loaded = store.load_result(result_id)

    assert loaded == result
    assert store.cached(request) == (result_id, result)
    assert store.cached(
        request.model_copy(update={"policy_version": "translation-v3"})
    ) is None
    assert set(loaded.payload.model_dump()) == {
        "source_language",
        "supplied_language",
        "target_language",
        "segments",
        "language_warnings",
    }
