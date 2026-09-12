"""Evidence-aware safeguards for ambiguous market symbols.

Some exchange-traded commodity roots share a short symbol with an equity in the
stock universe.  A cashtag is not enough to select the equity when the nearby
source text explicitly names the matching commodity contract.  This module
models those contract roots and their commodity terms; it is not a list of
equities to suppress.
"""

from __future__ import annotations

import re

# CME/COMEX/NYMEX roots that can be written as a social cashtag.  A root is
# treated as a commodity only when its corresponding commodity is also named
# close to that cashtag in the source.
COMMODITY_FUTURES_TERMS: dict[str, tuple[str, ...]] = {
    "GC": ("gold",),
    "SI": ("silver",),
    "HG": ("copper", "high grade copper"),
    "PL": ("platinum",),
    "PA": ("palladium",),
    "CL": ("crude oil", "wti", "brent"),
    "NG": ("natural gas",),
    "HO": ("heating oil",),
    "RB": ("gasoline", "rbob"),
    "ZC": ("corn",),
    "ZW": ("wheat",),
    "ZS": ("soybean", "soybeans"),
}

_CONTEXT_WINDOW = 96
_LEGAL_COMPANY_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "plc",
        "nv",
        "sa",
        "ag",
        "holdings",
        "holding",
        "group",
    }
)


def is_commodity_futures_reference(symbol: str, text: str | None) -> bool:
    """Whether ``$symbol`` is explicitly used as its matching commodity root.

    The function deliberately needs both a cashtag and a nearby commodity word.
    This retains ordinary explicit equity references, including companies whose
    symbols happen to overlap a futures root.
    """
    normalized = str(symbol or "").strip().upper().lstrip("$")
    terms = COMMODITY_FUTURES_TERMS.get(normalized)
    if not terms or not text:
        return False

    pattern = rf"(?<![A-Za-z0-9_$])\${re.escape(normalized)}(?![A-Za-z0-9.\-])"
    symbol_matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not symbol_matches:
        return False

    for term in terms:
        for commodity_match in re.finditer(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        ):
            if any(
                _within_context_window(symbol_match, commodity_match)
                for symbol_match in symbol_matches
            ):
                return True
    return False


def has_explicit_equity_identity(
    symbol: str,
    text: str | None,
    company_name: str | None,
) -> bool:
    """Whether the source explicitly identifies the active-universe equity.

    A matching full issuer name is sufficient.  A legal-suffix-normalized name
    is also accepted only when it immediately precedes the issuer's exact
    parenthetical cashtag, such as ``Hamilton Insurance ($HG)``.  This is a
    narrow override for a real equity reference near commodity discussion.
    """
    normalized = str(symbol or "").strip().upper().lstrip("$")
    if not text or not company_name or not normalized:
        return False

    full_name = _normalize_company_name(company_name)
    if not full_name:
        return False
    normalized_text = _normalize_company_name(text)
    if _has_exact_words(normalized_text, full_name):
        return True

    base_name = _strip_legal_company_suffixes(full_name)
    if not base_name:
        return False
    parenthetical = re.compile(rf"\(\s*\${re.escape(normalized)}\s*\)", re.IGNORECASE)
    for match in parenthetical.finditer(text):
        preceding = _normalize_company_name(text[: match.start()])
        if preceding.endswith(base_name) and _has_exact_words(preceding, base_name):
            return True
    return False


def _within_context_window(left: re.Match[str], right: re.Match[str]) -> bool:
    return (
        left.start() - _CONTEXT_WINDOW <= right.end()
        and right.start() <= left.end() + _CONTEXT_WINDOW
    )


def _normalize_company_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _strip_legal_company_suffixes(name: str) -> str:
    tokens = name.split()
    while tokens and tokens[-1] in _LEGAL_COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _has_exact_words(text: str, value: str) -> bool:
    return f" {value} " in f" {text} "
