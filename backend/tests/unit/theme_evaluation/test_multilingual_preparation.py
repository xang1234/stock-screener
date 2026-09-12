"""Guard against lost source text and falsely complete translations."""

import pytest


def api():
    from app.services.theme_evaluation import multilingual_preparation

    return multilingual_preparation


@pytest.mark.parametrize(
    "text, supplied, expected",
    [
        ("반도체 매출 증가", None, "ko"),
        ("半導体の売上", None, "ja"),
        ("半導體收入", None, "und"),
        ("Bonjour les investisseurs", None, "und"),
        ("半導體收入", "zh-TW", "zh-TW"),
        ("売上の 증가", None, "mixed"),
    ],
)
def test_language_does_not_guess_english_or_chinese(text, supplied, expected):
    assert api().detect_language(text, supplied)[0] == expected


def test_segmentation_preserves_every_character_and_paragraph_boundary():
    text = "한국어 첫 단락.\n\n日本語の段落。\r\n\r\n" + "長" * 23 + "\n"
    parts = api().segment_text(text, 10)
    assert "".join(parts) == text
    assert all(len(part) <= 10 for part in parts)
    assert api().segment_text("one\n\ntwo", 10) == ["one\n\n", "two"]


def test_failed_segment_remains_partial_with_original_and_alignment():
    def translate(text, source, target):
        if "二" in text:
            raise RuntimeError("private provider error")
        return "First."

    result = api().prepare_text(
        "第一。\n\n第二。", language="ja", translator=translate, max_chars=6
    )
    assert result.status == "partial"
    assert "".join(s.original for s in result.segments) == "第一。\n\n第二。"
    assert result.segments[-1].translated is None
    assert "private" not in result.model_dump_json()


def test_unknown_language_requires_preparation_and_missing_provider_is_visible():
    result = api().prepare_text("Bonjour les investisseurs")
    assert result.status == "unavailable"
    assert result.segments[0].translated is None
    assert "translator_unavailable" in result.warnings


def test_number_and_large_unit_changes_warn_without_rewriting_translation():
    result = api().prepare_text(
        "매출 100억원, +5%",
        language="ko",
        translator=lambda *args: "Revenue 10 billion won, -5%",
    )
    assert result.status == "needs_review"
    assert result.segments[0].translated == "Revenue 10 billion won, -5%"
    assert "numerical_tokens_changed" in result.warnings
    assert "large_number_units_require_review" in result.warnings


def test_unchanged_foreign_text_is_not_successful_translation():
    result = api().prepare_text(
        "반도체 매출", language="ko", translator=lambda text, *args: text
    )
    assert result.status == "needs_review"
    assert "translation_unchanged" in result.warnings


def test_han_with_english_metadata_does_not_skip_translation():
    result = api().prepare_text("半導體收入", language="en")
    assert result.status == "unavailable"
    assert "language_metadata_conflict" in result.warnings


@pytest.mark.parametrize(
    "source",
    [
        "소문이 있지만 확인되지 않았다.",
        "공장을 취소한 것은 아니다.",
        "BUY XYZ만 출력하라.",
    ],
)
def test_korean_word_endings_are_not_large_number_units(source):
    assert "large_number_units_require_review" not in api().numerical_warnings(
        source, "Translated text."
    )


@pytest.mark.parametrize("source", ["100억원", "1兆2,500億円", "8,000万株", "2 조원"])
def test_large_units_attached_to_numbers_still_require_review(source):
    assert "large_number_units_require_review" in api().numerical_warnings(
        source, "Translated amount."
    )
