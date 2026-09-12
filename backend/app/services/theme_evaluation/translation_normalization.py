"""Normalize the finite token classes covered by translation-quality policies."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeVar

_Label = TypeVar("_Label")


@dataclass(frozen=True)
class Quantity:
    value: Decimal
    unit: str
    start: int
    end: int

    @property
    def key(self) -> tuple[Decimal, str]:
        return self.value.normalize(), self.unit


@dataclass(frozen=True)
class NormalizedText:
    text: str
    temporal: Counter[tuple[str, int]]
    dates: Counter[tuple[int | None, int | None, int | None]]
    identifiers: Counter[str]
    quantities: tuple[Quantity, ...]
    metrics: frozenset[str]
    directions: frozenset[str]
    negated: bool

    @property
    def has_supported_semantics(self) -> bool:
        return bool(
            self.quantities
            or self.temporal
            or (self.metrics and (self.directions or self.negated))
        )


_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_HANDLE = re.compile(r"(?<![\w$])@[A-Za-z0-9_]+\b")
_CJK_SCALES = {
    "万": Decimal("1e4"),
    "萬": Decimal("1e4"),
    "만": Decimal("1e4"),
    "億": Decimal("1e8"),
    "亿": Decimal("1e8"),
    "억": Decimal("1e8"),
    "兆": Decimal("1e12"),
    "조": Decimal("1e12"),
}
_ENGLISH_SCALES = {
    None: Decimal(1),
    "thousand": Decimal("1e3"),
    "million": Decimal("1e6"),
    "billion": Decimal("1e9"),
    "trillion": Decimal("1e12"),
}
_UNIT_ALIASES = {
    "원": "KRW",
    "won": "KRW",
    "krw": "KRW",
    "₩": "KRW",
    "円": "JPY",
    "yen": "JPY",
    "jpy": "JPY",
    "¥": "JPY",
    "元": "CNY",
    "yuan": "CNY",
    "rmb": "CNY",
    "cny": "CNY",
    "dollar": "USD",
    "dollars": "USD",
    "usd": "USD",
    "$": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "€": "EUR",
    "pound": "GBP",
    "pounds": "GBP",
    "gbp": "GBP",
    "£": "GBP",
    "株": "shares",
    "주": "shares",
    "share": "shares",
    "shares": "shares",
    "stock": "shares",
    "stocks": "shares",
    "개": "count",
    "unit": "count",
    "units": "count",
    "台": "count",
    "%": "percent",
    "percent": "percent",
    "percentage point": "percentage_point",
    "percentage points": "percentage_point",
    "個百分點": "percentage_point",
    "个百分点": "percentage_point",
    "個百分点": "percentage_point",
    "个百分點": "percentage_point",
    "人民币": "CNY",
    "美元": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
}
_CJK_QUANTITY = re.compile(
    r"(?:(?P<prefix>人民币|RMB|CNY|USD)\s*)?"
    r"(?P<amount>[+−-]?\d[\d,.]*(?:\s*(?:兆|億|亿|万|萬|억|조|만)\s*[\d,.]*)+)"
    r"\s*(?P<unit>US\s+dollars?|美元|원|円|元|株|주|개|台|units?)?",
    re.IGNORECASE,
)
_PLAIN_QUANTITY = re.compile(
    r"(?<![A-Za-z0-9_.$€£¥₩+−-])"
    r"(?P<head>(?:[+−-]\s*[$€£¥₩]?|[$€£¥₩]\s*[+−-]?))?\s*"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|trillion)?\s*"
    r"(?P<unit>percentage\s+points?|[個个]百分[點点]|percent|%|won|krw|yen|jpy|"
    r"yuan|rmb|cny|dollars?|usd|euros?|eur|pounds?|gbp|shares?|stocks?|units?|"
    r"원|円|元|株|주|개|台)?"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]+(?:[-.]?\d+)+(?:\.[A-Za-z]+)?|"
    r"\d+(?:\.\d+)*\.[A-Za-z][A-Za-z0-9]*)(?![A-Za-z0-9_])"
)
_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            (),
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sept", "sep"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        )
    )
    for name in names
}
_MONTH_NAME = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?,?",
    re.IGNORECASE,
)
_METRICS = {
    "revenue": re.compile(
        r"매출|売上|(?:销售|銷售)额?|营收|營收|收入|\b(?:revenue|sales)\b",
        re.IGNORECASE,
    ),
    "profit": re.compile(
        r"영업이익|순이익|이익|営業利益|利益|利润|利潤|\b(?:profit|earnings)\b",
        re.IGNORECASE,
    ),
    "loss": re.compile(r"손실|적자|損失|亏损|虧損|\b(?:loss|losses)\b", re.IGNORECASE),
}
_DIRECTIONS = {
    "increase": re.compile(
        r"증가|상승|급증|성장|확대|올랐|늘(?:었|어|어난|어나|고|며|다)|"
        r"上昇|増加|增长|增長|涨|漲|\+\s*\d|"
        r"\b(?:increase[ds]?|increasing|rose|risen|grew|growth|up)\b",
        re.IGNORECASE,
    ),
    "decrease": re.compile(
        r"감소|하락|급감|축소|내렸|低下|減少|下降|"
        r"\b(?:decrease[ds]?|decreasing|fell|fallen|decline[ds]?|down)\b",
        re.IGNORECASE,
    ),
}
_NEGATION = re.compile(
    r"아니(?:다|며|고|라고)|않(?:다|았다|는다|고)|없(?:다|었다)|"
    r"ㄴㄴ|ではない|ない|不是|没有|沒有|"
    r"\b(?:not|no|never|without|isn't|wasn't|doesn't|didn't)\b",
    re.IGNORECASE,
)
_QUARTER = re.compile(
    r"\bQ([1-4])\b|(?:第\s*)?([1-4])\s*(?:분기|四半期|季度)", re.IGNORECASE
)
_EUROPEAN_DECIMAL_PERCENT = re.compile(
    r"(?<![\d,.])(?P<integer>[+−-]?\d+),(?P<fraction>\d{1,2})(?=\s*%)"
)


def _clean(text: str) -> str:
    return _HANDLE.sub(" ", _URL.sub(" ", text))


def normalize_decimal_percentages(text: str) -> str:
    """Canonicalize comma decimals only when directly typed as percentages."""
    return _EUROPEAN_DECIMAL_PERCENT.sub(
        lambda match: f"{match.group('integer')}.{match.group('fraction')}",
        text,
    )


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(
        start < other_end and end > other_start for other_start, other_end in spans
    )


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", "").replace("−", "-"))


def _year_has_quantity_unit(text: str, start: int, end: int) -> bool:
    before = text[:start]
    after = text[end:]
    return bool(
        re.search(r"[$€£¥₩]\s*$", before)
        or re.match(
            r"\s*(?:%|won|krw|yen|jpy|yuan|rmb|cny|dollars?|usd|euros?|eur|"
            r"pounds?|gbp|shares?|stocks?|units?|원|円|元|株|주|개)",
            after,
            re.IGNORECASE,
        )
    )


def _temporal_tokens(
    text: str,
) -> tuple[Counter, Counter, list[tuple[int, int]]]:
    tokens: Counter[tuple[str, int]] = Counter()
    date_parts: list[tuple[int | None, int | None, int | None]] = []
    spans: list[tuple[int, int]] = []

    for match in re.finditer(
        r"(?<!\d)((?:19|20)\d{2})([-/.])(1[0-2]|0?[1-9])\2(3[01]|[12]\d|0?[1-9])(?!\d)",
        text,
    ):
        tokens[("year", int(match.group(1)))] += 1
        tokens[("month", int(match.group(3)))] += 1
        tokens[("day", int(match.group(4)))] += 1
        date_parts.append(
            (int(match.group(1)), int(match.group(3)), int(match.group(4)))
        )
        spans.append(match.span())
    for match in re.finditer(r"(?<!\d)(\d{4})\s*[年년]", text):
        tokens[("year", int(match.group(1)))] += 1
        spans.append(match.span())
    for match in re.finditer(r"(?<!\d)(1[0-2]|0?[1-9])\s*[月월]", text):
        tokens[("month", int(match.group(1)))] += 1
        spans.append(match.span())
    for match in re.finditer(r"(?<!\d)(3[01]|[12]\d|0?[1-9])\s*[日일]", text):
        tokens[("day", int(match.group(1)))] += 1
        spans.append(match.span())
    for match in _QUARTER.finditer(text):
        tokens[("quarter", int(match.group(1) or match.group(2)))] += 1
        spans.append(match.span())

    for match in _MONTH_NAME.finditer(text):
        after = re.match(
            r"\s*(3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?\b",
            text[match.end() :],
            re.IGNORECASE,
        )
        before = re.search(
            r"\b(3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?\s*$",
            text[: match.start()],
            re.IGNORECASE,
        )
        year_after_context = re.match(r"\s*,?\s*(?:19|20)\d{2}\b", text[match.end() :])
        year_before_context = re.search(r"\b(?:19|20)\d{2}\s*$", text[: match.start()])
        other_month = any(
            other.span() != match.span() for other in _MONTH_NAME.finditer(text)
        )
        if match.group(1).lower() == "may" and not (
            after or before or year_after_context or year_before_context or other_month
        ):
            continue
        month = _MONTHS[match.group(1).lower()]
        tokens[("month", month)] += 1
        spans.append(match.span())
        day = None
        if after:
            start = match.end() + after.start()
            end = match.end() + after.end()
            day = int(after.group(1))
            tokens[("day", day)] += 1
            spans.append((start, end))
        if before and day is None:
            day = int(before.group(1))
            tokens[("day", day)] += 1
            spans.append(before.span())
        tail_start = after.end() if after else 0
        year_after = re.match(
            r"\s*,?\s*((?:19|20)\d{2})\b",
            text[match.end() + tail_start :],
        )
        year_before = re.search(r"\b((?:19|20)\d{2})\s*$", text[: match.start()])
        year = int(year_after.group(1)) if year_after else None
        if year is None and year_before:
            year = int(year_before.group(1))
        date_parts.append((year, month, day))

    for match in re.finditer(
        r"(?:(?<!\d)((?:19|20)\d{2})\s*[年년]\s*)?"
        r"(1[0-2]|0?[1-9])\s*[月월]"
        r"(?:\s*(3[01]|[12]\d|0?[1-9])\s*[日일])?",
        text,
    ):
        date_parts.append(
            (
                int(match.group(1)) if match.group(1) else None,
                int(match.group(2)),
                int(match.group(3)) if match.group(3) else None,
            )
        )

    for match in re.finditer(r"(?<![\d.])(?:19|20)\d{2}(?![\d.])", text):
        if not _overlaps(*match.span(), spans) and not _year_has_quantity_unit(
            text, *match.span()
        ):
            tokens[("year", int(match.group()))] += 1
            spans.append(match.span())

    years = {
        value for (kind, value), count in tokens.items() if kind == "year" and count
    }
    if len(years) == 1:
        shared_year = next(iter(years))
        date_parts = [
            (year if year is not None else shared_year, month, day)
            for year, month, day in date_parts
        ]
    dates = Counter(date_parts)
    if not dates:
        dates.update(
            (year, None, None)
            for (kind, year), count in tokens.items()
            if kind == "year"
            for _ in range(count)
        )
    return tokens, dates, spans


def _identifiers(
    text: str, occupied: list[tuple[int, int]]
) -> tuple[Counter, list[tuple[int, int]]]:
    values: Counter[str] = Counter()
    spans: list[tuple[int, int]] = []
    for match in _IDENTIFIER.finditer(text):
        if _overlaps(*match.span(), occupied) or re.fullmatch(
            r"Q[1-4]", match.group(), re.IGNORECASE
        ):
            continue
        values[match.group().casefold()] += 1
        spans.append(match.span())
    return values, spans


def _cjk_quantity(match: re.Match[str]) -> Quantity:
    raw = match.group("amount")
    sign = Decimal(-1) if raw.startswith(("-", "−")) else Decimal(1)
    pairs = list(re.finditer(r"(\d[\d,.]*)\s*(兆|億|亿|万|萬|억|조|만)", raw))
    value = sum(
        (
            _decimal(component.group(1)) * _CJK_SCALES[component.group(2)]
            for component in pairs
        ),
        Decimal(0),
    )
    tail = raw[pairs[-1].end() :].strip()
    if re.search(r"\d", tail):
        value += _decimal(tail)
    explicit_unit = match.group("unit") or match.group("prefix")
    unit_text = (
        re.sub(r"\s+", " ", explicit_unit.strip().lower()) if explicit_unit else None
    )
    unit = _UNIT_ALIASES[unit_text] if unit_text else "ambiguous"
    return Quantity(sign * value, unit, match.start(), match.end())


def _quantities(text: str, occupied: list[tuple[int, int]]) -> tuple[Quantity, ...]:
    quantities: list[Quantity] = []
    spans = list(occupied)
    for match in _CJK_QUANTITY.finditer(text):
        if not _overlaps(*match.span(), spans):
            quantities.append(_cjk_quantity(match))
            spans.append(match.span())
    for match in _PLAIN_QUANTITY.finditer(text):
        if _overlaps(*match.span(), spans):
            continue
        head = match.group("head") or ""
        prefix_match = re.search(r"[$€£¥₩]", head)
        prefix = prefix_match.group() if prefix_match else None
        unit_text = re.sub(
            r"\s+", " ", (match.group("unit") or prefix or "number").lower()
        )
        unit = _UNIT_ALIASES.get(unit_text, "number")
        scale = _ENGLISH_SCALES[(match.group("scale") or "").lower() or None]
        sign = Decimal(-1) if "-" in head or "−" in head else Decimal(1)
        quantities.append(
            Quantity(
                sign * _decimal(match.group("number")) * scale,
                unit,
                match.start(),
                match.end(),
            )
        )
        spans.append(match.span())
    return tuple(sorted(quantities, key=lambda item: item.start))


def normalize_text(text: str) -> NormalizedText:
    cleaned = _clean(text)
    temporal, dates, temporal_spans = _temporal_tokens(cleaned)
    identifiers, identifier_spans = _identifiers(cleaned, temporal_spans)
    quantities = _quantities(cleaned, [*temporal_spans, *identifier_spans])
    return NormalizedText(
        text=cleaned,
        temporal=temporal,
        dates=dates,
        identifiers=identifiers,
        quantities=quantities,
        metrics=frozenset(
            name for name, pattern in _METRICS.items() if pattern.search(cleaned)
        ),
        directions=frozenset(
            name for name, pattern in _DIRECTIONS.items() if pattern.search(cleaned)
        ),
        negated=bool(_NEGATION.search(cleaned)),
    )


def quantity_counter(
    quantities: tuple[Quantity, ...], *, include_ambiguous: bool = True
) -> Counter:
    return Counter(
        quantity.key
        for quantity in quantities
        if include_ambiguous or quantity.unit != "ambiguous"
    )


def ambiguous_values(quantities: tuple[Quantity, ...]) -> Counter:
    return Counter(
        quantity.value.normalize()
        for quantity in quantities
        if quantity.unit == "ambiguous"
    )


def _markers(patterns: dict[str, re.Pattern], text: str) -> list[tuple[int, int, str]]:
    return sorted(
        (match.start(), match.end(), name)
        for name, pattern in patterns.items()
        for match in pattern.finditer(text)
    )


def _nearest_label(
    markers: list[tuple[int, int, _Label]], position: int
) -> tuple[int, int, _Label] | None:
    preceding = [marker for marker in markers if marker[0] <= position]
    if preceding:
        return preceding[-1]
    return markers[0] if markers else None


def quantity_associations(text: str) -> Counter:
    associations: Counter[
        tuple[str | None, int | None, Decimal, str, str | None, bool]
    ] = Counter()
    normalized = normalize_text(text)
    metrics = _markers(_METRICS, normalized.text)
    directions = _markers(_DIRECTIONS, normalized.text)
    quarters = [
        (match.start(), match.end(), int(match.group(1) or match.group(2)))
        for match in _QUARTER.finditer(normalized.text)
    ]

    for quantity in normalized.quantities:
        metric_marker = _nearest_label(metrics, quantity.start)
        metric = metric_marker[2] if metric_marker else None
        region_start = metric_marker[0] if metric_marker else 0
        region_end = next(
            (
                marker[0]
                for marker in metrics
                if metric_marker and marker[0] > metric_marker[0]
            ),
            len(normalized.text),
        )
        local_directions = [
            marker for marker in directions if region_start <= marker[0] < region_end
        ]
        direction_marker = min(
            local_directions,
            key=lambda marker: min(
                abs(marker[0] - quantity.end), abs(quantity.start - marker[1])
            ),
            default=None,
        )
        direction = direction_marker[2] if direction_marker else None
        quarter_marker = _nearest_label(quarters, quantity.start)
        quarter = quarter_marker[2] if quarter_marker else None
        negated = bool(_NEGATION.search(normalized.text[region_start:region_end]))
        if metric is None and quarter is None:
            continue
        associations[(metric, quarter, *quantity.key, direction, negated)] += 1
    return associations


__all__ = [
    "NormalizedText",
    "Quantity",
    "ambiguous_values",
    "normalize_text",
    "quantity_associations",
    "quantity_counter",
]
