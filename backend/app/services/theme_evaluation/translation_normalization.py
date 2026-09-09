"""Normalize the finite token classes covered by translation-quality-v1."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal


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
    "%": "percent",
    "percent": "percent",
}
_CJK_QUANTITY = re.compile(
    r"[+−-]?\d[\d,.]*(?:\s*(?:兆|億|亿|万|萬|억|조|만)\s*[\d,.]*)+"
    r"\s*(?:원|円|元|株|주|개)?"
)
_PLAIN_QUANTITY = re.compile(
    r"(?<![\w.])(?P<prefix>[$€£¥₩])?\s*"
    r"(?P<number>[+−-]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|trillion)?\s*"
    r"(?P<unit>percent|%|won|krw|yen|jpy|yuan|rmb|cny|dollars?|usd|"
    r"euros?|eur|pounds?|gbp|shares?|stocks?|units?|원|円|元|株|주|개)?"
    r"(?![\w.])",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(
    r"(?<![\w])(?:[A-Za-z]+(?:[-.]?\d+)+(?:\.[A-Za-z]+)?|"
    r"\d+(?:\.\d+)*\.[A-Za-z][A-Za-z0-9]*)(?![\w])"
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
        r"증가|상승|급증|성장|확대|올랐|上昇|増加|增长|增長|"
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
    r"ではない|ない|不是|没有|沒有|\b(?:not|no|never|without|isn't|wasn't|doesn't|didn't)\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return _HANDLE.sub(" ", _URL.sub(" ", text))


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(
        start < other_end and end > other_start for other_start, other_end in spans
    )


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", "").replace("−", "-"))


def _temporal_tokens(text: str) -> tuple[Counter, list[tuple[int, int]]]:
    tokens: Counter[tuple[str, int]] = Counter()
    spans: list[tuple[int, int]] = []

    for match in re.finditer(
        r"(?<!\d)((?:19|20)\d{2})([-/.])(1[0-2]|0?[1-9])\2(3[01]|[12]\d|0?[1-9])(?!\d)",
        text,
    ):
        tokens[("year", int(match.group(1)))] += 1
        tokens[("month", int(match.group(3)))] += 1
        tokens[("day", int(match.group(4)))] += 1
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
    for match in re.finditer(
        r"\bQ([1-4])\b|(?:第\s*)?([1-4])\s*(?:분기|四半期|季度)",
        text,
        re.IGNORECASE,
    ):
        tokens[("quarter", int(match.group(1) or match.group(2)))] += 1
        spans.append(match.span())

    for match in _MONTH_NAME.finditer(text):
        tokens[("month", _MONTHS[match.group(1).lower()])] += 1
        spans.append(match.span())
        after = re.match(
            r"\s*(3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?\b",
            text[match.end() :],
            re.IGNORECASE,
        )
        if after:
            start = match.end() + after.start()
            end = match.end() + after.end()
            tokens[("day", int(after.group(1)))] += 1
            spans.append((start, end))
        before = re.search(
            r"\b(3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?\s*$",
            text[: match.start()],
            re.IGNORECASE,
        )
        if before:
            tokens[("day", int(before.group(1)))] += 1
            spans.append(before.span())

    for match in re.finditer(r"(?<![\d.])(?:19|20)\d{2}(?![\d.])", text):
        if not _overlaps(*match.span(), spans):
            tokens[("year", int(match.group()))] += 1
            spans.append(match.span())
    return tokens, spans


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
    raw = match.group()
    sign = Decimal(-1) if raw.startswith(("-", "−")) else Decimal(1)
    pairs = re.findall(r"(\d[\d,.]*)\s*(兆|億|亿|万|萬|억|조|만)", raw)
    value = sum(
        (_decimal(number) * _CJK_SCALES[scale] for number, scale in pairs), Decimal(0)
    )
    unit_match = re.search(r"(원|円|元|株|주|개)\s*$", raw)
    unit = _UNIT_ALIASES[unit_match.group(1)] if unit_match else "ambiguous"
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
        prefix = match.group("prefix")
        unit_text = (match.group("unit") or prefix or "number").lower()
        unit = _UNIT_ALIASES.get(unit_text, "number")
        scale = _ENGLISH_SCALES[(match.group("scale") or "").lower() or None]
        quantities.append(
            Quantity(
                _decimal(match.group("number")) * scale,
                unit,
                match.start(),
                match.end(),
            )
        )
        spans.append(match.span())
    return tuple(sorted(quantities, key=lambda item: item.start))


def normalize_text(text: str) -> NormalizedText:
    cleaned = _clean(text)
    temporal, temporal_spans = _temporal_tokens(cleaned)
    identifiers, identifier_spans = _identifiers(cleaned, temporal_spans)
    quantities = _quantities(cleaned, [*temporal_spans, *identifier_spans])
    return NormalizedText(
        text=cleaned,
        temporal=temporal,
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


def quantity_associations(text: str) -> Counter:
    associations: Counter[tuple[str | None, int | None, Decimal, str]] = Counter()
    for clause in re.split(r"[,;；，\n]+", text):
        normalized = normalize_text(clause)
        metric = (
            next(iter(normalized.metrics)) if len(normalized.metrics) == 1 else None
        )
        quarters = [
            value
            for (kind, value), count in normalized.temporal.items()
            if kind == "quarter"
            for _ in range(count)
        ]
        quarter = quarters[0] if len(quarters) == 1 else None
        if metric is None and quarter is None:
            continue
        for quantity in normalized.quantities:
            associations[(metric, quarter, *quantity.key)] += 1
    return associations


__all__ = [
    "NormalizedText",
    "Quantity",
    "ambiguous_values",
    "normalize_text",
    "quantity_associations",
    "quantity_counter",
]
