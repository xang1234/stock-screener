"""Translation adequacy policy regressions for evidence preparation."""

from pathlib import Path

import pytest
from app.services.theme_evaluation.bundle import canonical_bytes, sha256
from app.services.theme_evaluation.preparation_results import RESULT_ADAPTER


def assess(source: str, target: str, language: str | None = "ko"):
    from app.services.theme_evaluation.translation_quality import assess_translation

    return assess_translation(source, target, language=language)


@pytest.mark.parametrize(
    "source,target,language,expected",
    [
        ("718만주 1.9조원에 매수", "Samsung", "ko", "fallback"),
        ("매출 10% 증가", "Revenue increased 100%", "ko", "fallback"),
        (
            "718만주 1.9조원",
            "7.18 million shares, 1.9 trillion won",
            "ko",
            "use",
        ),
        ("매출 2조5500억원", "Revenue 2.55 trillion won", "ko", "use"),
        (
            "매출 증가 https://t.co/a123",
            "Revenue increased",
            "ko",
            "use",
        ),
        ("4월과 5월", "April and May", "ko", "use"),
        ("https://t.co/a123", "https://t.co/a123", "zxx", "use"),
        ("👀", "👀", "art", "use"),
    ],
)
def test_assessment(source, target, language, expected):
    result = assess(source, target, language)

    assert result.policy_version == "translation-quality-v1"
    assert result.disposition == expected
    assert isinstance(result.issues, tuple)


def test_fresh_full_samsung_capture_passes_while_old_fragment_falls_back():
    source = (
        "이재용 삼성전자 회장, 母홍라희 보유 718만주 1.9조원에 매수 - 조선비즈 "
        "https://t.co/yZWIyQI9gK"
    )
    complete = (
        "Samsung Electronics Chairman Lee Jae-yong buys 7.18 million shares "
        "held by mother Hong Ra-hee for 1.9 trillion won - Chosun Biz"
    )

    assert assess(source, "Samsung").disposition == "fallback"
    assert assess(source, complete).disposition == "use"


@pytest.mark.parametrize(
    "source,target",
    [
        ("1兆2,500億円", "1.25 trillion yen"),
        ("8,000万株", "80 million shares"),
        ("2 조원", "2 trillion won"),
        ("销售额 3.2亿元", "Revenue was 320 million yuan"),
    ],
)
def test_compound_cjk_quantities_match_decimal_english_values(source, target):
    assert assess(source, target, "ja").disposition == "use"


@pytest.mark.parametrize(
    "source,target",
    [
        ("순손실 -15억원", "Net loss was 1.5 billion won"),
        ("매출 10%와 이익 10%", "Revenue 10% and profit 5%"),
        ("매출 100억원", "Revenue was 10 billion dollars"),
        ("매출 100억원", "Revenue was 10 billion"),
        ("매출 10% 증가", "Revenue decreased 10%"),
        ("적자가 아니다", "It is a loss"),
        ("2026년 4월 5일", "April 2026"),
        ("판매량 10개, 10개", "Sales volume was 10 units"),
        ("GPT-6 출시", "GPT-7 launched"),
        ("GPT6 출시", "GPT7 launched"),
        ("삼성전자 005930.KS", "Samsung Electronics 005931.KS"),
    ],
)
def test_verified_material_changes_require_fallback(source, target):
    result = assess(source, target)

    assert result.disposition == "fallback"
    assert any(issue.severity == "blocker" for issue in result.issues)


def test_ambiguous_scale_without_a_unit_requires_review_not_conversion():
    result = assess("매출 10억", "Revenue 1 billion", "ko")

    assert result.disposition == "review"
    assert [issue.code for issue in result.issues] == ["ambiguous_quantity_unit"]


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "Q1 매출 10억원, Q2 매출 20억원",
            "Q1 revenue 2 billion won; Q2 revenue 1 billion won",
        ),
        (
            "매출 10억원, 이익 20억원",
            "Revenue 2 billion won; profit 1 billion won",
        ),
    ],
)
def test_swapped_period_or_metric_values_require_review(source, target):
    result = assess(source, target)

    assert result.disposition == "review"
    assert any(issue.code == "quantity_association_changed" for issue in result.issues)
    assert not any(issue.severity == "blocker" for issue in result.issues)


def test_complete_calendar_date_and_matching_direction_are_supported():
    assert (
        assess(
            "2026년 4월 5일 매출 10% 증가", "Revenue increased 10% on April 5, 2026"
        ).disposition
        == "use"
    )
    assert (
        assess("2026.04.05 매출 증가", "Revenue increased on April 5, 2026").disposition
        == "use"
    )


def test_ambiguous_scale_cannot_mask_an_explicit_currency_change():
    result = assess(
        "매출 10억, 이익 5억원",
        "Revenue 1 billion, profit 500 million dollars",
    )

    assert result.disposition == "fallback"
    assert any(issue.code == "quantity_unit_changed" for issue in result.issues)


def test_short_translation_of_long_prose_is_flagged_for_review():
    source = "반도체 공급망의 회복 속도와 고객사의 주문 강도가 예상보다 견조하게 유지되고 있다"
    result = assess(source, "Strong")

    assert result.disposition == "review"
    assert [issue.code for issue in result.issues] == ["possible_translation_fragment"]


def test_unknown_semantic_equivalence_is_explicitly_left_for_review():
    result = assess("반도체 수요가 견조하다", "The weather is pleasant")

    assert result.disposition == "review"
    assert [issue.code for issue in result.issues] == [
        "semantic_equivalence_unverified"
    ]


def test_unchanged_source_language_prose_requires_review_but_tickers_are_not_currency():
    assert assess("매출이 증가했다", "매출이 증가했다").disposition == "review"
    assert assess("$SIVE", "$SIVE", "en").disposition == "use"
    assert assess("매출 $10", "Revenue was 10 dollars").disposition == "use"


def test_issue_excerpts_preserve_exact_original_company_text():
    source = "이재용 삼성전자 회장, 母홍라희 보유 718만주 1.9조원에 매수"
    target = "Samsung"
    result = assess(source, target)

    assert result.disposition == "fallback"
    assert all(issue.source_excerpt == source for issue in result.issues)
    assert all(issue.translated_excerpt == target for issue in result.issues)


def test_assessment_does_not_reinterpret_archived_result_bytes_or_status():
    root = Path(__file__).parents[4]
    result_id = "252c8c6f7b7e229e28797eb621d0462b218e554ef0c1dba6980b402a7d16e4cd"
    path = (
        root
        / "data/xui-reader/theme-evaluation/reader-update-20260909"
        / "preparation-store/results"
        / f"{result_id}.json"
    )
    raw = path.read_bytes()
    archived = RESULT_ADAPTER.validate_json(raw)
    before = (
        archived.status,
        archived.warnings,
        canonical_bytes(archived.model_dump(mode="json")),
    )

    segment = archived.payload.segments[0]
    assessment = assess(
        segment.original,
        segment.translated,
        archived.payload.source_language,
    )

    assert assessment.disposition == "review"
    assert (
        archived.status,
        archived.warnings,
        canonical_bytes(archived.model_dump(mode="json")),
    ) == before
    assert before[2] == raw
    assert sha256(raw) == result_id
