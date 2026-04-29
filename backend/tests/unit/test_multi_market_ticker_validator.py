"""Unit tests for multi_market_ticker_validator (T7.5).

The module is pure — no DB required — so we exercise every drop
reason and every accept path via direct calls. The DB-dependent
universe check lives on ThemeExtractionService and is covered by the
existing integration tests for that service.
"""

from __future__ import annotations

import logging

import pytest

from app.services.multi_market_ticker_validator import (
    POLICY_VERSION,
    REASON_EMPTY,
    REASON_UNRESOLVABLE,
    TICKER_SHAPE_RE,
    NormalizedTicker,
    describe_policy,
    log_drop,
    normalize_extracted_ticker,
)


class TestNormalizeExtractedTicker:
    """The T7.5 prompt contract: what the validator accepts must match
    what the prompt promises. Pin every supported format so a prompt
    drift can't silently open a drop path."""

    # --- T7.4 CJK resolver passthrough ------------------------------

    def test_us_ticker_falls_through_to_security_master(self):
        result = normalize_extracted_ticker("NVDA")
        assert result.canonical == "NVDA"
        assert result.reason is None

    def test_us_ticker_is_uppercased(self):
        # SecurityMaster.normalize_symbol does .upper() — preserves the
        # existing US behaviour unchanged.
        result = normalize_extracted_ticker("nvda")
        assert result.canonical == "NVDA"

    def test_us_ticker_strips_dollar_prefix(self):
        # SecurityMaster drops a leading "$" that Twitter-style mentions add.
        result = normalize_extracted_ticker("$AAPL")
        assert result.canonical == "AAPL"

    def test_hk_canonical_symbol_passthrough(self):
        result = normalize_extracted_ticker("0700.HK")
        assert result.canonical == "0700.HK"

    def test_hk_unpadded_code_pads_to_four_digits(self):
        # The CJK resolver pads "700.HK" → "0700.HK" via passthrough.
        result = normalize_extracted_ticker("700.HK")
        assert result.canonical == "0700.HK"

    def test_jp_canonical_symbol_passthrough(self):
        assert normalize_extracted_ticker("6758.T").canonical == "6758.T"

    def test_tw_canonical_symbol_passthrough(self):
        assert normalize_extracted_ticker("2330.TW").canonical == "2330.TW"

    def test_two_canonical_symbol_passthrough(self):
        # TPEx (Taiwanese over-the-counter) uses .TWO.
        assert normalize_extracted_ticker("1234.TWO").canonical == "1234.TWO"

    # --- CJK alias lookups ------------------------------------------

    def test_cjk_english_alias_resolves(self):
        assert normalize_extracted_ticker("Tencent").canonical == "0700.HK"
        assert normalize_extracted_ticker("Sony").canonical == "6758.T"
        assert normalize_extracted_ticker("TSMC").canonical == "2330.TW"

    def test_cjk_native_script_alias_resolves(self):
        assert normalize_extracted_ticker("騰訊").canonical == "0700.HK"
        assert normalize_extracted_ticker("ソニー").canonical == "6758.T"
        assert normalize_extracted_ticker("台積電").canonical == "2330.TW"

    def test_halfwidth_katakana_resolves(self):
        # NFKC folds ｿﾆｰ → ソニー; the alias index catches the fullwidth form.
        assert normalize_extracted_ticker("ｿﾆｰ").canonical == "6758.T"

    def test_simplified_han_resolves(self):
        # Simplified/Traditional are separate entries in the seed corpus.
        assert normalize_extracted_ticker("腾讯").canonical == "0700.HK"

    # --- Drop reasons -----------------------------------------------

    def test_empty_string_drops_with_reason_empty(self):
        result = normalize_extracted_ticker("")
        assert result.canonical is None
        assert result.reason == REASON_EMPTY

    def test_whitespace_drops_with_reason_empty(self):
        assert normalize_extracted_ticker("   ").reason == REASON_EMPTY
        assert normalize_extracted_ticker("\n\t").reason == REASON_EMPTY

    def test_non_string_drops_with_reason_empty(self):
        # LLM output is JSON and can contain None, numbers, nested dicts.
        # We treat any non-string as an empty drop rather than raise.
        assert normalize_extracted_ticker(None).reason == REASON_EMPTY
        assert normalize_extracted_ticker(123).reason == REASON_EMPTY
        assert normalize_extracted_ticker({"nested": "junk"}).reason == REASON_EMPTY

    def test_returns_normalized_ticker_dataclass(self):
        result = normalize_extracted_ticker("NVDA")
        assert isinstance(result, NormalizedTicker)
        assert result.raw == "NVDA"
        assert result.canonical == "NVDA"
        assert result.reason is None


class TestTickerShapeRegex:
    """The shape regex is the second gate — canonical forms must match."""

    @pytest.mark.parametrize("shape_ok", [
        "NVDA", "AAPL", "GOOGL", "A",
        "0700.HK", "0005.HK", "12345.HK",
        "6758.T", "9984.T",
        "2330.TW", "2330.TWO",
        "BRK.B", "BF.B",  # US share-class tickers
    ])
    def test_canonical_shapes_match(self, shape_ok: str):
        assert TICKER_SHAPE_RE.match(shape_ok), f"{shape_ok!r} should match"

    @pytest.mark.parametrize("shape_bad", [
        "",
        "NVDA NVDA",   # space
        "nvda",        # lowercase
        "$NVDA",       # leading $
        ".HK",         # suffix only
        "12345678901234",  # too long
    ])
    def test_malformed_shapes_reject(self, shape_bad: str):
        assert not TICKER_SHAPE_RE.match(shape_bad), f"{shape_bad!r} should not match"


class TestLogDrop:
    def test_known_reason_emits_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="app.services.multi_market_ticker_validator"):
            log_drop(raw="junk", canonical=None, reason=REASON_EMPTY)
        assert any("ticker_dropped" in r.getMessage() for r in caplog.records)
        assert any("reason=empty_input" in r.getMessage() for r in caplog.records)

    def test_unknown_reason_emits_warning_and_still_logs_debug(self, caplog):
        # Defense: a typo'd reason tag must NOT silently disappear; it
        # should promote to warning so the caller notices. But the
        # structured debug line must still be emitted so ops dashboards
        # don't lose visibility into the mention itself.
        with caplog.at_level(logging.DEBUG, logger="app.services.multi_market_ticker_validator"):
            log_drop(raw="x", canonical=None, reason="bogus_reason")
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert any(
            r.levelno == logging.DEBUG and "reason=bogus_reason" in r.getMessage()
            for r in caplog.records
        )


class TestPolicySurface:
    def test_policy_version_accessor(self):
        snap = describe_policy()
        assert snap["policy_version"] == POLICY_VERSION

    def test_accepted_suffixes_match_bead_contract(self):
        snap = describe_policy()
        # The extraction prompt in theme_extraction_service.py promises
        # callers these exact suffixes — keep pinned so a prompt
        # change forces a policy-version bump here too.
        assert snap["accepted_suffixes"] == [
            ".HK",
            ".NS",
            ".BO",
            ".T",
            ".KS",
            ".KQ",
            ".TW",
            ".TWO",
        ]

    def test_drop_reasons_include_every_category(self):
        snap = describe_policy()
        # Every drop path in _clean_tickers must be represented; if a
        # new bucket is added, update both _DROP_REASONS and this test.
        assert "empty_input" in snap["drop_reasons"]
        assert "in_false_positive_list" in snap["drop_reasons"]
        assert "unresolvable_to_canonical" in snap["drop_reasons"]
        assert "invalid_shape" in snap["drop_reasons"]
        assert "not_in_active_universe" in snap["drop_reasons"]
        assert "active_universe_empty" in snap["drop_reasons"]


class TestPromptValidatorSynchronization:
    """Meta-test: the acceptance criterion on bead asia.7.5 says
    'prompt contract and validator contract must be synchronized.' We
    read the prompt constant directly and assert it mentions the same
    suffixes the validator advertises."""

    def test_prompt_mentions_every_accepted_suffix(self):
        # Use word-boundary regex rather than substring `in` because the
        # suffixes are prefix-nested: ``.T`` is a substring of ``.TW``
        # and ``.TWO``, so ``".T" in prompt`` would false-pass even if
        # the prompt dropped the literal ``.T`` spec while keeping ``.TW``.
        import re
        from app.services.theme_extraction_service import EXTRACTION_SYSTEM_PROMPT
        for suffix in [".HK", ".NS", ".BO", ".T", ".KS", ".KQ", ".TW", ".TWO"]:
            pattern = re.compile(re.escape(suffix) + r"\b")
            assert pattern.search(EXTRACTION_SYSTEM_PROMPT), (
                f"Prompt must document {suffix!r} as a literal token "
                f"(not as a prefix of a longer suffix) to stay in sync "
                f"with the validator contract"
            )

    def test_prompt_still_covers_us_markets(self):
        from app.services.theme_extraction_service import EXTRACTION_SYSTEM_PROMPT
        assert "NYSE" in EXTRACTION_SYSTEM_PROMPT
        assert "NASDAQ" in EXTRACTION_SYSTEM_PROMPT

    def test_synchronization_check_rejects_prefix_false_positive(self):
        # Regression: the previous naive ``suffix in prompt`` test would
        # pass if ``.T`` was only present as a substring of ``.TW``.
        # This test confirms the word-boundary regex correctly fails a
        # prompt that has ``.TW`` but no standalone ``.T``.
        import re
        synthetic_prompt = (
            "Taiwan (TWSE): 4-digit code with ``.TW`` suffix, e.g. 2330.TW"
        )
        # Naive substring would wrongly claim .T is documented:
        assert ".T" in synthetic_prompt  # proves the old style false-positives
        # Word-boundary regex correctly says no:
        assert not re.search(r"\.T\b", synthetic_prompt)
