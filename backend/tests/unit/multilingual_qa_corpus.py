"""Multilingual QA golden corpus (T7.6).

Hand-curated reference set for the deterministic stages of the
multilingual extraction pipeline:

- T7.2 language detection    (``detect_language``)
- T7.4 CJK alias resolution  (``resolve_alias``)
- T7.5 ticker normalization  (``normalize_extracted_ticker``)

Each corpus deliberately mixes three classes of case:

1. **Positive** — mappings that must succeed. Drives recall.
2. **Negative / adversarial** — mappings that must NOT fire. Drives
   precision (the asymmetric risk: a false positive silently links
   unrelated content to a ticker, poisoning downstream theme stats).
3. **Ambiguous** — cases where the resolver is allowed to bail out
   to ``METHOD_NONE`` rather than guess.

The LLM-dependent stages (theme extraction, non-identity translation)
are **out of scope** for this harness — they cannot be golden-tested
deterministically and instead rely on production drop-path logs (T7.5)
plus manual spot-check for QA.

Versioning
----------
Any change to a corpus entry must bump ``CORPUS_VERSION``. The
harness test ``test_corpus_version_is_pinned`` catches accidental
semantic drift during review.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

CORPUS_VERSION: str = "2026.04.13.1"


# ---------------------------------------------------------------------------
# Alias resolver corpus
# ---------------------------------------------------------------------------


class AliasCase(NamedTuple):
    """One hand-labeled alias-resolution golden item.

    ``expected_canonical`` of ``None`` means the resolver MUST NOT
    produce a canonical symbol for this input (``METHOD_NONE``) —
    guessing wrong is a precision violation.
    """

    query: str
    hint_market: Optional[str]
    expected_canonical: Optional[str]
    tag: str  # "positive" | "negative" | "adversarial"


ALIAS_CORPUS: list[AliasCase] = [
    # --- Positive: Hong Kong ----------------------------------------
    AliasCase("Tencent",          None, "0700.HK", "positive"),
    AliasCase("Tencent Holdings", None, "0700.HK", "positive"),
    AliasCase("騰訊",               None, "0700.HK", "positive"),
    AliasCase("腾讯",               None, "0700.HK", "positive"),  # Simplified
    AliasCase("HSBC",              None, "0005.HK", "positive"),
    AliasCase("滙豐",               None, "0005.HK", "positive"),
    AliasCase("汇丰",               None, "0005.HK", "positive"),  # Simplified
    AliasCase("Alibaba",           None, "9988.HK", "positive"),
    AliasCase("阿里巴巴",            None, "9988.HK", "positive"),
    AliasCase("0700.HK",           None, "0700.HK", "positive"),  # Passthrough
    AliasCase("700.HK",            None, "0700.HK", "positive"),  # Padded
    AliasCase("700",               "HK", "0700.HK", "positive"),  # Normalized

    # --- Positive: Japan --------------------------------------------
    AliasCase("Sony",              None, "6758.T", "positive"),
    AliasCase("ソニー",              None, "6758.T", "positive"),
    AliasCase("ｿﾆｰ",                None, "6758.T", "positive"),  # Halfwidth
    AliasCase("SoftBank",          None, "9984.T", "positive"),
    AliasCase("ソフトバンク",         None, "9984.T", "positive"),
    AliasCase("Toyota",            None, "7203.T", "positive"),
    AliasCase("トヨタ",              None, "7203.T", "positive"),
    AliasCase("6758.T",            None, "6758.T", "positive"),  # Passthrough

    # --- Positive: Taiwan -------------------------------------------
    AliasCase("TSMC",              None, "2330.TW", "positive"),
    AliasCase("台積電",             None, "2330.TW", "positive"),
    AliasCase("臺積電",             None, "2330.TW", "positive"),  # Alt traditional
    AliasCase("Foxconn",           None, "2317.TW", "positive"),
    AliasCase("鴻海",               None, "2317.TW", "positive"),

    # --- Negative: unknown US companies (should fall to METHOD_NONE)
    AliasCase("Apple",             None, None, "negative"),
    AliasCase("Microsoft",         None, None, "negative"),
    AliasCase("Amazon",            None, None, "negative"),
    AliasCase("Nvidia",            None, None, "negative"),

    # --- Adversarial: ambiguous / noise / over-reach guards ---------
    AliasCase("",                  None, None, "adversarial"),   # empty
    AliasCase("   ",               None, None, "adversarial"),   # whitespace
    AliasCase("700",               None, None, "adversarial"),   # bare numeric, no hint
    AliasCase("12",                "HK", None, "adversarial"),   # too short for any market
    AliasCase("AI",                None, None, "adversarial"),   # acronym false positive
    AliasCase("CEO",               None, None, "adversarial"),   # role abbreviation
    AliasCase("Sony Music",        None, None, "adversarial"),   # subsidiary — must not over-reach
    AliasCase("iPhone",            None, None, "adversarial"),   # product, not a ticker
    AliasCase("半導体",             None, None, "adversarial"),   # generic industry term
    AliasCase("銀行",               None, None, "adversarial"),   # generic industry term
]


# ---------------------------------------------------------------------------
# Language detection corpus
# ---------------------------------------------------------------------------


class LanguageCase(NamedTuple):
    text: str
    expected_language: str
    tag: str


LANGUAGE_CORPUS: list[LanguageCase] = [
    # --- English financial prose ------------------------------------
    LanguageCase("The Federal Reserve raised rates by 25 basis points.", "en", "positive"),
    LanguageCase("Nvidia posts record data-center revenue this quarter.", "en", "positive"),
    LanguageCase("GLP-1 weight-loss drugs continue to drive pharmaceutical sales.", "en", "positive"),

    # --- Japanese (kana-containing) ---------------------------------
    LanguageCase("日経平均は続伸、半導体が牽引した。", "ja", "positive"),
    LanguageCase("ソニーグループが好決算を発表した。", "ja", "positive"),
    LanguageCase("トヨタ自動車の生産台数が増加。", "ja", "positive"),
    LanguageCase("ｿﾆｰの新製品発表", "ja", "positive"),  # halfwidth katakana

    # --- Chinese simplified (mainland) ------------------------------
    LanguageCase("腾讯控股发布季度财报，业绩超预期。", "zh", "positive"),
    LanguageCase("阿里巴巴宣布云业务重组。", "zh", "positive"),

    # --- Chinese traditional (HK/TW) --------------------------------
    LanguageCase("恆生指數創下新高，科技股領漲。", "zh", "positive"),
    LanguageCase("台積電宣布建設新廠。", "zh", "positive"),

    # --- Adversarial / edge cases -----------------------------------
    LanguageCase("",                              "und", "adversarial"),  # empty
    LanguageCase("   \n\t  ",                     "und", "adversarial"),  # whitespace
    LanguageCase("123 456.78 !!!",                "und", "adversarial"),  # digits/punct only
    LanguageCase("Нефтяные запасы выросли",       "und", "adversarial"),  # Cyrillic
    LanguageCase("النفط يرتفع",                    "und", "adversarial"),  # Arabic
    LanguageCase("The 日経 index rallied today on strong earnings.", "en", "adversarial"),  # mixed, Latin dominant
]


# ---------------------------------------------------------------------------
# Ticker normalizer corpus (T7.5 pure normalization, pre-universe)
# ---------------------------------------------------------------------------


class NormalizerCase(NamedTuple):
    raw: object  # object to allow non-string adversarial inputs
    expected_canonical: Optional[str]
    tag: str


NORMALIZER_CORPUS: list[NormalizerCase] = [
    # --- Positive: canonical passthrough ----------------------------
    NormalizerCase("NVDA",     "NVDA",     "positive"),
    NormalizerCase("AAPL",     "AAPL",     "positive"),
    NormalizerCase("$AAPL",    "AAPL",     "positive"),   # dollar strip
    NormalizerCase("nvda",     "NVDA",     "positive"),   # casefold
    NormalizerCase("0700.HK",  "0700.HK",  "positive"),
    NormalizerCase("6758.T",   "6758.T",   "positive"),
    NormalizerCase("2330.TW",  "2330.TW",  "positive"),

    # --- Positive: CJK aliases --------------------------------------
    NormalizerCase("Tencent",  "0700.HK",  "positive"),
    NormalizerCase("ソニー",    "6758.T",   "positive"),
    NormalizerCase("台積電",    "2330.TW",  "positive"),

    # --- Adversarial: non-string input must drop, not crash ---------
    NormalizerCase(None,                   None, "adversarial"),
    NormalizerCase(123,                    None, "adversarial"),
    NormalizerCase({"nested": "x"},        None, "adversarial"),
    NormalizerCase("",                     None, "adversarial"),
    NormalizerCase("   ",                  None, "adversarial"),
]


__all__ = [
    "CORPUS_VERSION",
    "AliasCase",
    "LanguageCase",
    "NormalizerCase",
    "ALIAS_CORPUS",
    "LANGUAGE_CORPUS",
    "NORMALIZER_CORPUS",
]
