"""Quantity display must not change scale, currency, provenance or prose."""

import pytest


def normalize(text):
    from app.services.theme_evaluation import quantity_display

    return quantity_display.normalize_quantities(text)


@pytest.mark.parametrize(
    ("source", "display", "value", "unit"),
    [
        ("718만주", "7.18 million shares", "7180000", "shares"),
        ("1.9조원", "KRW 1.9 trillion", "1900000000000", "KRW"),
        ("₩320억원", "KRW 32 billion", "32000000000", "KRW"),
        ("2조원", "KRW 2 trillion", "2000000000000", "KRW"),
        ("1兆2,500億円", "JPY 1.25 trillion", "1250000000000", "JPY"),
        ("RMB 718亿", "CNY 71.8 billion", "71800000000", "CNY"),
        ("1조 9,000억원", "KRW 1.9 trillion", "1900000000000", "KRW"),
        ("−3.2억원", "KRW -320 million", "-320000000", "KRW"),
        ("+2億円", "JPY +200 million", "200000000", "JPY"),
        ("0万株", "0 shares", "0", "shares"),
        ("1.000001万股", "10000.01 shares", "10000.01", "shares"),
        ("2亿美元", "USD 200 million", "200000000", "USD"),
    ],
)
def test_exact_quantities(source, display, value, unit):
    result = normalize("About " + source + " was reported.")
    assert result.text == "About " + display + " was reported."
    assert not result.issues
    assert len(result.quantities) == 1
    quantity = result.quantities[0]
    assert (quantity.original, quantity.value, quantity.unit) == (source, value, unit)
    assert result.original[quantity.start : quantity.end] == source
    assert normalize(result.text).text == result.text


@pytest.mark.parametrize(
    "source",
    [
        "2조",
        "¥2亿",
        "$2億",
        "USD 2억원",
        "RMB 2億円",
        "1,9조원",
        "1.2.3억원",
        "1億2兆円",
        "1万万株",
        "2억원%",
    ],
)
def test_ambiguous_or_malformed_quantities_are_unchanged_and_flagged(source):
    result = normalize("Value: " + source + ".")
    assert result.text == result.original
    assert result.issues
    assert not result.quantities


def test_preserves_urls_identifiers_names_and_semantic_roles():
    text = (
        "삼성전자 rose by $30.62, not to $30.62; about 2조원. "
        "https://example.com/2조원 @test2조원 ID2조원"
    )
    result = normalize(text)
    assert result.text == text.replace("about 2조원", "about KRW 2 trillion")
    assert len(result.quantities) == 1
    assert (
        result.original[result.quantities[0].start : result.quantities[0].end]
        == "2조원"
    )


def test_repeated_amounts_keep_independent_spans_and_exact_precision():
    text = "1억원 then 1억원; 12345678901234567890123456789만원"
    result = normalize(text)
    assert [q.value for q in result.quantities] == [
        "100000000",
        "100000000",
        "123456789012345678901234567890000",
    ]
    assert [text[q.start : q.end] for q in result.quantities] == [
        "1억원",
        "1억원",
        "12345678901234567890123456789만원",
    ]


@pytest.mark.parametrize(
    "source",
    [
        "2조원3억원",
        "2억-3만원",
        "2억원+3만원",
        "2억원 / 3만원",
        "1億円元",
        "2억원원",
        "￥2億元",
        "2億元",
    ],
)
def test_does_not_salvage_connected_or_conflicting_expressions(source):
    result = normalize(source)
    assert result.text == source
    assert not result.quantities
    assert result.issues


@pytest.mark.parametrize(
    "url",
    [
        "www.example.com/2조원",
        "ftp://example.com/2조원",
        "example.com/2조원",
        "2조원@example.com",
    ],
)
def test_preserves_other_url_and_email_forms(url):
    result = normalize(url + " costs 2조원.")
    assert result.text == url + " costs KRW 2 trillion."
    assert len(result.quantities) == 1


def test_prefix_survives_whitespace_after_sign():
    result = normalize("USD + 2억원")
    assert result.text == result.original
    assert result.issues[0].code == "conflicting_quantity_units"
    assert not result.quantities
    assert normalize("KRW + 2조원").text == "KRW +2 trillion"


def test_explicit_chinese_currency_suffix():
    result = normalize("RMB 2亿人民币 and 2亿人民币")
    assert result.text == "CNY 200 million and CNY 200 million"
    assert [q.value for q in result.quantities] == ["200000000", "200000000"]


def test_currency_prefix_is_never_left_outside_normalized_span():
    assert normalize("円2億円").text == "JPY 200 million"
    for source in ("USD KRW 2조원", "JPY 円2億円"):
        result = normalize(source)
        assert result.text == source
        assert not result.quantities
        assert result.issues
