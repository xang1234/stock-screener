"""Conservative CJK quantity rendering; never rewrites saved model evidence.

Offsets are Python character offsets into ``original``, not normalized text.
Values are exact base-unit decimal strings, with no exchange-rate conversion.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
from itertools import pairwise

POLICY_VERSION = "quantity-display-v1"
_SCALES = {"万": 4, "萬": 4, "만": 4, "億": 8, "亿": 8, "억": 8, "兆": 12, "조": 12}
_UNITS = {
    "원": "KRW",
    "₩": "KRW",
    "krw": "KRW",
    "won": "KRW",
    "円": "JPY",
    "jpy": "JPY",
    "yen": "JPY",
    "人民币": "CNY",
    "人民幣": "CNY",
    "元": "CNY",
    "rmb": "CNY",
    "cny": "CNY",
    "yuan": "CNY",
    "usd": "USD",
    "美元": "USD",
    "eur": "EUR",
    "€": "EUR",
    "gbp": "GBP",
    "£": "GBP",
    "주": "shares",
    "株": "shares",
    "股": "shares",
    "shares": "shares",
    "개": "count",
    "台": "count",
    "units": "count",
}
_PREFIX = r"人民币|人民幣|美元|円|원|元|RMB|CNY|KRW|JPY|USD|EUR|GBP|₩|¥|￥|\$|€|£"
_SUFFIX = (
    r"人民币|人民幣|美元|원|円|元|株|주|股|개|台|shares\b|units\b|won\b|yen\b|yuan\b"
)
_SCALE = "万萬만億亿억兆조"
# Broad candidate capture prevents salvaging a valid-looking substring from an
# invalid compound amount. Validation below is deliberately stricter.
_CANDIDATE = re.compile(
    rf"(?<![A-Za-z0-9_.,+−$¥￥₩€£-])"
    rf"(?P<sign>[+−-]?)\s*(?:(?P<prefix>{_PREFIX})\s*)?"
    rf"(?P<inner_sign>[+−-]?)\s*"
    rf"(?P<amount>[0-9][0-9,.]*(?:\s*[{_SCALE}])+"
    rf"(?:\s*[0-9][0-9,.]*(?:\s*[{_SCALE}])*)*)"
    rf"(?:\s*(?P<unit>{_SUFFIX}))?",
    re.IGNORECASE,
)
_NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
_COMPONENT = re.compile(rf"({_NUMBER})\s*([{_SCALE}])")
_PROTECTED = re.compile(
    r"[a-z][a-z0-9+.-]*://\S+|www\.\S+|"
    r"[\w.-]+\.[a-z]{2,}/\S*|[^\s@]+@[^\s@]+|@\w+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DisplayQuantity:
    original: str
    start: int
    end: int
    value: str
    unit: str
    display: str


@dataclass(frozen=True)
class QuantityIssue:
    original: str
    start: int
    end: int
    code: str


@dataclass(frozen=True)
class QuantityDisplay:
    policy_version: str
    source_text_sha256: str
    original: str
    text: str
    quantities: tuple[DisplayQuantity, ...]
    issues: tuple[QuantityIssue, ...]


def _decimal_string(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _amount(raw: str) -> Decimal:
    value, previous, position = Decimal(0), 13, 0
    for part in _COMPONENT.finditer(raw):
        if raw[position : part.start()].strip():
            raise ValueError("unsupported_quantity_syntax")
        scale = _SCALES[part[2]]
        if scale >= previous:
            raise ValueError("unsupported_quantity_scale_order")
        value += Decimal(part[1].replace(",", "")) * Decimal(10) ** scale
        previous, position = scale, part.end()
    if position == 0 or raw[position:].strip():
        raise ValueError("unsupported_quantity_syntax")
    return value


def _unit(prefix: str, suffix: str) -> str:
    head, tail = _UNITS.get(prefix.lower()), _UNITS.get(suffix.lower())
    if prefix == "元" or (suffix == "元" and head != "CNY"):
        raise ValueError("ambiguous_quantity_unit")
    if prefix in ("$", "¥", "￥"):
        # These symbols alone do not identify a currency.
        allowed = {"USD"} if prefix == "$" else {"JPY", "CNY"}
        if tail not in allowed:
            raise ValueError("ambiguous_quantity_unit")
    if head and tail and head != tail:
        raise ValueError("conflicting_quantity_units")
    if not (head or tail):
        raise ValueError("ambiguous_quantity_unit")
    return head or tail


def _display(value: Decimal, unit: str, sign: str) -> str:
    magnitude, label = value, ""
    for exponent, name in ((12, "trillion"), (9, "billion"), (6, "million")):
        if abs(value) >= Decimal(10) ** exponent:
            magnitude, label = value / Decimal(10) ** exponent, " " + name
            break
    amount = ("+" if sign == "+" else "") + _decimal_string(magnitude) + label
    if unit in ("shares", "count"):
        return amount + (" shares" if unit == "shares" else " units")
    return unit + " " + amount


def _connected_candidates(text, matches):
    """Do not salvage individual numbers from touching amounts or expressions."""
    blocked = set()
    for left, right in pairwise(matches):
        gap = text[left.end() : right.start()]
        connector = gap.strip()
        if (
            not gap
            or (not connector and (right["sign"] or right["inner_sign"]))
            or (
                connector and re.fullmatch(r"[+−~〜～/–—원円元株주股개台-]+", connector)
            )
        ):
            blocked.update((left.start(), right.start()))
    return blocked


def normalize_quantities(text: str) -> QuantityDisplay:
    """Normalize only explicit, supported amounts; retain ambiguous text and flags."""
    protected = [match.span() for match in _PROTECTED.finditer(text)]
    quantities, issues, pieces, cursor = [], [], [], 0
    matches = list(_CANDIDATE.finditer(text))
    connected = _connected_candidates(text, matches)
    for match in matches:
        start, end = match.span()
        # The optional leading sign permits whitespace: keep it out of the span.
        start += len(match[0]) - len(match[0].lstrip())
        if any(start < right and end > left for left, right in protected):
            continue
        original = text[start:end]
        try:
            if re.search(
                rf"(?<![A-Za-z])(?:{_PREFIX})\s*$", text[:start], re.IGNORECASE
            ):
                raise ValueError("unsupported_quantity_prefix")
            if match.start() in connected:
                raise ValueError("unsupported_quantity_expression")
            if re.match(
                r"[%A-Za-z0-9_万萬만億亿억兆조원円元株주股개台₩¥￥$€£]", text[end:]
            ):
                raise ValueError("unsupported_quantity_suffix")
            if match["sign"] and match["inner_sign"]:
                raise ValueError("unsupported_quantity_sign")
            unit = _unit(match["prefix"] or "", match["unit"] or "")
            sign = match["sign"] or match["inner_sign"]
            # Precision grows with the input, so large coefficients cannot round.
            with localcontext() as context:
                context.prec = len(match["amount"]) + 32
                value = _amount(match["amount"])
                if sign in ("-", "−"):
                    value = -value
                display = _display(value, unit, sign)
                quantity = DisplayQuantity(
                    original, start, end, _decimal_string(value), unit, display
                )
        except ValueError as error:
            issues.append(QuantityIssue(original, start, end, str(error)))
            continue
        quantities.append(quantity)
        pieces.extend((text[cursor:start], display))
        cursor = end
    pieces.append(text[cursor:])
    return QuantityDisplay(
        POLICY_VERSION,
        sha256(text.encode()).hexdigest(),
        text,
        "".join(pieces),
        tuple(quantities),
        tuple(issues),
    )
