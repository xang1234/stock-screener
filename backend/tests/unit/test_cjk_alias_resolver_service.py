"""Unit tests for the deterministic CJK alias resolver (T7.4).

The test corpus doubles as the bead acceptance evidence: each case in
``GOLDEN_CORPUS`` represents a format we must handle, with the expected
method tag proving that the right resolution path fired. Pinning the
method is intentional — if a later refactor makes (say) ``Tencent``
resolve via ``alias_folded`` instead of ``alias_exact``, a reviewer
will notice because the golden row will have to be touched.
"""

from __future__ import annotations

from typing import Optional

import pytest

from app.services.cjk_alias_resolver_service import (
    METHOD_ALIAS_EXACT,
    METHOD_ALIAS_FOLDED,
    METHOD_NONE,
    METHOD_SYMBOL_NORMALIZED,
    METHOD_SYMBOL_PASSTHROUGH,
    POLICY_VERSION,
    SUPPORTED_MARKETS,
    AliasResolution,
    describe_policy,
    fold_key,
    nfkc,
    policy_version,
    resolve_alias,
)


# ---------------------------------------------------------------------------
# Golden corpus: (query, hint_market, expected_symbol, expected_market, method)
# ---------------------------------------------------------------------------

POSITIVE_CORPUS = [
    # --- Symbol passthrough: already canonical ------------------------
    ("0700.HK", None, "0700.HK", "HK", METHOD_SYMBOL_PASSTHROUGH),
    ("6758.T",  None, "6758.T",  "JP", METHOD_SYMBOL_PASSTHROUGH),
    ("2330.TW", None, "2330.TW", "TW", METHOD_SYMBOL_PASSTHROUGH),
    # HK codes may arrive un-padded with suffix; passthrough pads to 4.
    ("700.HK",  None, "0700.HK", "HK", METHOD_SYMBOL_PASSTHROUGH),
    # Lowercased suffix normalizes upstream (NFKC casefold via .upper()).
    ("0700.hk", None, "0700.HK", "HK", METHOD_SYMBOL_PASSTHROUGH),

    # --- Symbol normalized: bare numeric + hint -----------------------
    ("700",     "HK", "0700.HK", "HK", METHOD_SYMBOL_NORMALIZED),
    ("0700",    "HK", "0700.HK", "HK", METHOD_SYMBOL_NORMALIZED),
    ("6758",    "JP", "6758.T",  "JP", METHOD_SYMBOL_NORMALIZED),
    ("2330",    "TW", "2330.TW", "TW", METHOD_SYMBOL_NORMALIZED),
    # Fullwidth digits are NFKC-normalized to ASCII.
    ("６７５８", "JP", "6758.T",  "JP", METHOD_SYMBOL_NORMALIZED),

    # --- Alias exact: NFKC-literal match ------------------------------
    ("Tencent",       None, "0700.HK", "HK", METHOD_ALIAS_EXACT),
    ("Sony",          None, "6758.T",  "JP", METHOD_ALIAS_EXACT),
    ("TSMC",          None, "2330.TW", "TW", METHOD_ALIAS_EXACT),
    ("ソニー",          None, "6758.T",  "JP", METHOD_ALIAS_EXACT),
    ("台積電",          None, "2330.TW", "TW", METHOD_ALIAS_EXACT),
    ("臺積電",          None, "2330.TW", "TW", METHOD_ALIAS_EXACT),  # Alt traditional
    ("騰訊",           None, "0700.HK", "HK", METHOD_ALIAS_EXACT),
    ("腾讯",           None, "0700.HK", "HK", METHOD_ALIAS_EXACT),  # Simplified
    # Halfwidth katakana → fullwidth via NFKC → matches "ソニー".
    ("ｿﾆｰ",            None, "6758.T",  "JP", METHOD_ALIAS_EXACT),

    # --- Alias folded: casefold + punctuation strip -------------------
    ("TENCENT",          None, "0700.HK", "HK", METHOD_ALIAS_FOLDED),
    ("tencent holdings", None, "0700.HK", "HK", METHOD_ALIAS_FOLDED),
    ("Sony-Group",       None, "6758.T",  "JP", METHOD_ALIAS_FOLDED),
    # Exact entry "Hon Hai Precision" is in the dict verbatim → EXACT, not FOLDED.
    ("Hon Hai Precision",None, "2317.TW", "TW", METHOD_ALIAS_EXACT),
    # Folded-only variant: hyphenated, upper-case — exercises fold_key.
    ("HON-HAI PRECISION",None, "2317.TW", "TW", METHOD_ALIAS_FOLDED),
    # Interpunct (・ U+30FB) is in the strip set.
    ("ソニー・グループ",    None, "6758.T",  "JP", METHOD_ALIAS_FOLDED),
]

NEGATIVE_CORPUS = [
    # Empty / whitespace: fast path to METHOD_NONE.
    "",
    "   ",
    "\n\t",
    # Genuinely unknown companies: the LLM's job, not ours.
    "Some Random Company",
    "未知の会社",
    # Bare numeric with NO hint_market: cannot know the exchange.
    # (Passed separately below with hint=None to prove we don't guess.)
]


class TestNormalizationPrimitives:
    """Pure helpers — pin the exact transforms the resolver relies on."""

    def test_nfkc_folds_halfwidth_katakana(self):
        # Halfwidth katakana (U+FF66+) → fullwidth via NFKC.
        assert nfkc("ｿﾆｰ") == "ソニー"

    def test_nfkc_folds_fullwidth_digits(self):
        assert nfkc("６７５８") == "6758"

    def test_nfkc_preserves_han(self):
        # NFKC is script-preserving for CJK Unified Ideographs.
        assert nfkc("騰訊") == "騰訊"
        assert nfkc("腾讯") == "腾讯"
        # Traditional ≠ Simplified even under NFKC — that's the design,
        # we rely on explicit alias entries for simplified/traditional.
        assert nfkc("騰訊") != nfkc("腾讯")

    def test_nfkc_strips_whitespace(self):
        assert nfkc("  Sony  ") == "Sony"

    def test_nfkc_handles_none_safely(self):
        assert nfkc(None) == ""  # type: ignore[arg-type]

    def test_fold_key_casefolds(self):
        assert fold_key("Tencent") == fold_key("TENCENT")
        assert fold_key("TSMC") == fold_key("tsmc")

    def test_fold_key_strips_common_punctuation(self):
        assert fold_key("Sony-Group") == fold_key("Sony Group")
        assert fold_key("Sony.Group") == fold_key("SonyGroup")
        # CJK middle dot (U+30FB) is in the strip set — "ソニー・グループ"
        # folds to the same key as "ソニーグループ".
        assert fold_key("ソニー・グループ") == fold_key("ソニーグループ")


class TestPositiveResolutions:
    """Parametric run of the golden positive corpus."""

    @pytest.mark.parametrize(
        "query,hint_market,expected_symbol,expected_market,expected_method",
        POSITIVE_CORPUS,
    )
    def test_resolves(
        self,
        query: str,
        hint_market: Optional[str],
        expected_symbol: str,
        expected_market: str,
        expected_method: str,
    ):
        result = resolve_alias(query, hint_market=hint_market)
        assert isinstance(result, AliasResolution)
        assert result.query == query
        assert result.canonical_symbol == expected_symbol, (
            f"{query!r} → {result.canonical_symbol!r}, expected {expected_symbol!r}"
        )
        assert result.market == expected_market
        assert result.method == expected_method
        assert result.policy_version == POLICY_VERSION


class TestNegativeResolutions:
    """Unresolved queries must explicitly carry METHOD_NONE, not a guess."""

    @pytest.mark.parametrize("query", NEGATIVE_CORPUS)
    def test_returns_method_none(self, query: str):
        result = resolve_alias(query)
        assert result.method == METHOD_NONE
        assert result.canonical_symbol is None
        assert result.market is None
        assert result.policy_version == POLICY_VERSION

    def test_bare_numeric_without_hint_is_unresolved(self):
        # "700" alone could be HK/JP/TW. We refuse to guess — the LLM
        # (or an upstream market-hint detector) has to disambiguate.
        result = resolve_alias("700")
        assert result.method == METHOD_NONE

    def test_jp_tw_require_four_digit_minimum(self):
        # JP/TW codes are essentially always 4 digits. Refusing shorter
        # tokens keeps numeric noise (prices, years) out of the pipeline.
        for short in ("7", "70", "700"):
            for market in ("JP", "TW"):
                result = resolve_alias(short, hint_market=market)
                assert result.method == METHOD_NONE, (
                    f"{short!r} + {market} should not resolve"
                )

    def test_hk_rejects_one_and_two_digit_tokens(self):
        # HK canonical codes zero-pad to 4 digits, so in principle "5"
        # → "0005.HK" (HSBC). In practice 1-2 digit tokens are too
        # ambiguous (prices, indices) so the resolver only accepts
        # 3-5 digit inputs for the normalized path.
        for short in ("5", "55"):
            result = resolve_alias(short, hint_market="HK")
            assert result.method == METHOD_NONE

    def test_hk_three_digit_code_pads_with_hint(self):
        # With an explicit HK hint, "700" is unambiguous enough to
        # resolve to the canonical zero-padded form.
        result = resolve_alias("700", hint_market="HK")
        assert result.method == METHOD_SYMBOL_NORMALIZED
        assert result.canonical_symbol == "0700.HK"


class TestPrecedence:
    """The ordering of rules is load-bearing — pin it."""

    def test_symbol_passthrough_beats_alias(self):
        # If a row happens to carry "2330.TW" we must take the symbol
        # passthrough path — not accidentally fall into the alias table
        # via some coincidental folded-key collision.
        result = resolve_alias("2330.TW")
        assert result.method == METHOD_SYMBOL_PASSTHROUGH

    def test_symbol_normalized_beats_alias(self):
        # "2330" + TW hint must resolve deterministically via the
        # numeric rule, not via any alias collision.
        result = resolve_alias("2330", hint_market="TW")
        assert result.method == METHOD_SYMBOL_NORMALIZED

    def test_exact_beats_folded_when_both_would_match(self):
        # "Sony" is a verbatim alias entry; it must not degrade to the
        # folded path even though folded would also match.
        result = resolve_alias("Sony")
        assert result.method == METHOD_ALIAS_EXACT


class TestDeterminism:
    """Same input must always give the same output — no caches, no clocks."""

    def test_repeated_call_produces_identical_resolution(self):
        first = resolve_alias("ソニーグループ")
        for _ in range(20):
            nth = resolve_alias("ソニーグループ")
            assert nth == first


class TestPolicySurface:
    def test_policy_version_accessor(self):
        assert policy_version() == POLICY_VERSION

    def test_describe_policy_exposes_corpus_shape(self):
        snap = describe_policy()
        assert snap["policy_version"] == POLICY_VERSION
        assert set(snap["supported_markets"]) == SUPPORTED_MARKETS
        assert snap["corpus_size"] >= 6  # representative seed, not exhaustive
        assert snap["alias_count"] >= snap["corpus_size"]
        assert METHOD_NONE in snap["methods"]
        assert METHOD_SYMBOL_PASSTHROUGH in snap["methods"]


class TestGoldenCorpusCoverage:
    """Meta-test: make sure the corpus exercises every resolution method."""

    def test_every_method_is_covered_by_positive_corpus(self):
        methods_in_corpus = {row[4] for row in POSITIVE_CORPUS}
        # METHOD_NONE is covered by NEGATIVE_CORPUS; the other four
        # must each have at least one positive golden row.
        assert methods_in_corpus == {
            METHOD_SYMBOL_PASSTHROUGH,
            METHOD_SYMBOL_NORMALIZED,
            METHOD_ALIAS_EXACT,
            METHOD_ALIAS_FOLDED,
        }

    def test_every_market_is_covered(self):
        markets_in_corpus = {row[3] for row in POSITIVE_CORPUS}
        assert markets_in_corpus == {"HK", "JP", "TW"}
