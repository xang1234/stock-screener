"""Translation adequacy policy regressions for evidence preparation."""

from pathlib import Path

import pytest
from app.services.theme_evaluation.bundle import canonical_bytes, sha256
from app.services.theme_evaluation.preparation_results import RESULT_ADAPTER


def assess(
    source: str,
    target: str,
    language: str | None = "ko",
    *,
    policy_version: str = "translation-quality-v2",
):
    from app.services.theme_evaluation.translation_quality import assess_translation

    return assess_translation(
        source,
        target,
        language=language,
        policy_version=policy_version,
    )


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

    assert result.policy_version == "translation-quality-v2"
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


def test_compound_cjk_quantity_includes_unscaled_final_component():
    assert assess("1만5000주", "15 thousand shares").disposition == "use"
    assert assess("1만5000주", "10 thousand shares").disposition == "fallback"


def test_sign_is_preserved_before_or_after_currency_symbol():
    assert assess("매출 -$10", "Revenue was -10 dollars").disposition == "use"
    assert assess("매출 $-10", "Revenue was -10 dollars").disposition == "use"
    assert assess("매출 -$10", "Revenue was $10").disposition == "fallback"


def test_korean_particles_do_not_break_explicit_quantity_units():
    assert (
        assess("매출 10%와 이익 5%", "Revenue 10% and profit 5%").disposition == "use"
    )
    assert assess("매출 100원에 도달", "Revenue reached 100 won").disposition == "use"


@pytest.mark.parametrize(
    "source,target",
    [
        ("순손실 -15억원", "Net loss was 1.5 billion won"),
        ("매출 10%와 이익 10%", "Revenue 10% and profit 5%"),
        ("매출 100억원", "Revenue was 10 billion dollars"),
        ("매출 100억원", "Revenue was 10 billion"),
        ("매출 10% 증가", "Revenue decreased 10%"),
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


def test_unanchored_lexical_polarity_differences_require_review():
    french = assess(
        "La note est là pour m’abattre… Le mec est sur un tapis volant.",
        "The grade is there to bring me down… The guy is on a flying carpet.",
        "fr",
    )
    korean = assess(
        "AI 반도체 인프라 투자는 쉽게 멈추기 어렵습니다.",
        "It is hard to easily halt AI semiconductor infrastructure investment.",
        "ko",
    )

    assert french.disposition == "review"
    assert korean.disposition == "review"
    assert assess("적자가 아니다", "It is a loss").disposition == "review"
    assert any(issue.code == "direction_unanchored" for issue in french.issues)
    assert not any(
        issue.severity == "blocker" for issue in (*french.issues, *korean.issues)
    )


def test_value_bound_negation_change_remains_a_blocker():
    result = assess("매출 10% 증가하지 않았다", "Revenue increased 10%")

    assert result.disposition == "fallback"
    assert any(issue.code == "polarity_association_changed" for issue in result.issues)


def test_v1_preserves_archived_global_polarity_blockers():
    result = assess(
        "La note est là pour m’abattre.",
        "The grade is there to bring me down.",
        "fr",
        policy_version="translation-quality-v1",
    )

    assert result.policy_version == "translation-quality-v1"
    assert result.disposition == "fallback"
    assert [issue.code for issue in result.issues] == ["direction_changed"]


def test_european_decimal_percentages_match_dot_decimal_transcription():
    source = "Sivers ▲ 8,72 %\nCorning ▲ 1,40 %\nAXT ▲ 2,73 %"
    target = "Sivers ▲ 8.72 %\nCorning ▲ 1.40 %\nAXT ▲ 2.73 %"

    locale_unknown = assess(source, target, "und")
    assert locale_unknown.disposition == "review"
    assert not any(issue.severity == "blocker" for issue in locale_unknown.issues)
    assert not any(
        issue.code.startswith("quantity_") for issue in locale_unknown.issues
    )
    assert (
        assess(
            source,
            target,
            "und",
            policy_version="translation-quality-v1",
        ).disposition
        == "fallback"
    )
    assert assess("Volume 1,000%", "Volume 1.000%", "en").disposition == "fallback"


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


def test_temporal_rules_do_not_invent_a_month_or_hide_explicit_units():
    assert assess("매출이 증가할 수 있다", "Revenue may increase").disposition == "use"
    titlecase_modal = assess("회사는 증가할 수 있다", "May increase")
    assert titlecase_modal.disposition == "review"
    assert not any(
        issue.code == "temporal_value_changed" for issue in titlecase_modal.issues
    )
    assert assess("2026주", "2026 shares").disposition == "use"
    assert assess("2026주", "2026 won").disposition == "fallback"


def test_metric_bound_direction_reversal_requires_fallback():
    result = assess(
        "매출 10% 증가, 이익 20% 감소",
        "Revenue decreased 10%; profit increased 20%",
    )

    assert result.disposition == "fallback"
    assert any(issue.code == "polarity_association_changed" for issue in result.issues)


def test_each_quantity_retains_its_nearby_direction_within_one_metric():
    result = assess(
        "매출 10% 증가, 20% 감소",
        "Revenue decreased 10%, increased 20%",
    )

    assert result.disposition == "fallback"
    assert any(issue.code == "polarity_association_changed" for issue in result.issues)


@pytest.mark.parametrize(
    "source,target,language",
    [
        (
            "영업이익은 -12.5% 감소했고 비용은 ₩320억원 늘었다.",
            "Operating profit decreased by -12.5%, and expenses rose by ₩320억원.",
            "ko",
        ),
        (
            "매출 2조원 돌파, 실적 대박.\n\n근데 가이던스는 3% 하향이라 추격매수는 ㄴㄴ.",
            (
                "Revenue surpassed 2조원, a blockbuster performance.\n\n"
                "But guidance was lowered by 3%, so chasing buys is a no-go."
            ),
            "ko",
        ),
        (
            "利益率は−8.4%で、8,000万株を消却する。",
            "The profit margin is -8.4%, and it will write off 8,000万株.",
            "ja",
        ),
        (
            "受注は前年比+25%。\r\n\r\nでも材料出尽くしで、今から買うのは微妙。",
            (
                "Orders are up 25% year-on-year.\n\n"
                "But with the positive news already priced in, buying from here is iffy."
            ),
            "ja",
        ),
        (
            "宁德时代季度收入为人民币718亿元，同比增长19%。",
            "CATL quarterly revenue was RMB 718亿, up 19% year on year.",
            "zh",
        ),
        (
            "台積電毛利率變動-2.3個百分點，資本支出為320億美元。",
            (
                "TSMC gross margin changed by -2.3 percentage points, and capital "
                "expenditure was 320億 US dollars."
            ),
            "zh",
        ),
        (
            "小米集团（1810.HK）称SU7 Ultra订单达到10万台。",
            "Xiaomi Group (1810.HK) stated that SU7 Ultra orders reached 10万 units.",
            "zh",
        ),
        (
            "订单增长30%，基本面很顶。\n\n不过股价已涨80%，别上头。",
            (
                "Order growth is 30%, and the fundamentals are very strong.\n\n"
                "However, the stock price has already risen 80%, so don't get carried away."
            ),
            "zh",
        ),
        (
            "HBM4 ramp looks good, 하지만 수율은 아직 65% 수준.",
            "HBM4 ramp looks good, but yield is still at around 65%.",
            "ko",
        ),
    ],
    ids=[
        "korean-direction",
        "korean-negation-slang",
        "japanese-unicode-minus",
        "japanese-explicit-plus",
        "simplified-chinese-currency",
        "traditional-chinese-percentage-points",
        "chinese-product-and-count",
        "chinese-script-adjacent-percentages",
        "mixed-korean-percent-suffix",
    ],
)
def test_saved_multilingual_challenge_equivalences_are_recognized(
    source, target, language
):
    assert assess(source, target, language).disposition == "use"


def test_retained_quoted_quantity_with_an_english_gloss_requires_review():
    result = assess(
        "会社は『收入增长20%』と説明したが、株価は-4%。",
        'The company explained "收入增长20%" (revenue growth of 20%), '
        "but the stock price was -4%.",
        "ja",
    )

    assert result.disposition == "review"
    assert any(issue.code == "quantity_repeated_in_gloss" for issue in result.issues)


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "매출 10억원 이익 20억원",
            "Revenue 2 billion won and profit 1 billion won",
        ),
        (
            "Q1 매출 10억원 Q2 매출 20억원",
            "Q1 revenue 2 billion won and Q2 revenue 1 billion won",
        ),
    ],
)
def test_nearby_associations_do_not_depend_on_punctuation(source, target):
    result = assess(source, target)

    assert result.disposition == "review"
    assert any(issue.code == "quantity_association_changed" for issue in result.issues)


def test_normalized_unchanged_foreign_prose_requires_review():
    assert assess("매출이 증가했다", " 매출이 증가했다 ").disposition == "review"
    assert assess("매출 증가 https://t.co/a123", "매출 증가").disposition == "review"


def test_date_relationships_are_not_flattened_into_component_bags():
    result = assess(
        "2026년 4월 5일과 5월 6일",
        "May 5, 2026 and April 6",
    )

    assert result.disposition == "review"
    assert any(issue.code == "date_association_changed" for issue in result.issues)

    year_swap = assess(
        "2025년 4월, 2026년 5월",
        "April 2026 and May 2025",
    )
    assert year_swap.disposition == "review"
    assert any(issue.code == "date_association_changed" for issue in year_swap.issues)


def test_thousands_separator_is_not_treated_as_clause_punctuation():
    assert assess("매출 1000원", "Revenue 1,000 won").disposition == "use"


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
