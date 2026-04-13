"""Unit tests for deterministic language detection (T7.2)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.language_detection_service import (
    LANGUAGE_EN,
    LANGUAGE_JA,
    LANGUAGE_UNKNOWN,
    LANGUAGE_ZH,
    POLICY_VERSION,
    SUPPORTED_LANGUAGES,
    describe_policy,
    detect_and_cache_language,
    detect_language,
    policy_version,
)


class TestDetectLanguagePure:
    """Pure-function classification; no I/O, fully deterministic."""

    def test_english_text(self):
        assert detect_language(
            "Nvidia reported record data-center revenue this quarter."
        ) == LANGUAGE_EN

    def test_japanese_with_kana(self):
        assert detect_language("日経平均は続伸、半導体が牽引した。") == LANGUAGE_JA

    def test_japanese_katakana_only(self):
        # Katakana alone is still unambiguously Japanese.
        assert detect_language("ソニー・グループ") == LANGUAGE_JA

    def test_chinese_traditional(self):
        # No kana, all Han — Traditional Chinese (HK/TW bucket).
        assert detect_language("恆生指數創下新高，科技股領漲。") == LANGUAGE_ZH

    def test_english_with_loanword_stays_english(self):
        # One Japanese loanword citation must not flip the classifier.
        # English article > 20 letters with one Han char → still "en".
        assert detect_language(
            "The 日経 index rallied today on strong semiconductor earnings."
        ) == LANGUAGE_EN

    def test_empty_and_whitespace(self):
        assert detect_language("") == LANGUAGE_UNKNOWN
        assert detect_language(None) == LANGUAGE_UNKNOWN
        assert detect_language("   \n\t  ") == LANGUAGE_UNKNOWN

    def test_digits_and_punctuation_only(self):
        # No alphabetic characters → undetermined (not English).
        assert detect_language("123 456.78 !!!") == LANGUAGE_UNKNOWN

    def test_deterministic_repeat(self):
        # Same input must always give the same output — pin it.
        text = "混合 English and 日本語 text with some 中文 sprinkled in."
        results = {detect_language(text) for _ in range(10)}
        assert len(results) == 1

    def test_all_results_are_supported_languages(self):
        for text in (
            "hello world",
            "こんにちは",
            "你好世界",
            "",
            "123",
        ):
            assert detect_language(text) in SUPPORTED_LANGUAGES


class TestDetectAndCacheLanguage:
    """DB-aware wrapper: the column IS the cache."""

    def _row(self, *, title=None, content=None, source_language=None):
        return SimpleNamespace(
            title=title, content=content, source_language=source_language,
        )

    def test_detects_and_persists_when_missing(self):
        row = self._row(title="日経平均、続伸", content="半導体が上昇した。")
        result = detect_and_cache_language(row)
        assert result == LANGUAGE_JA
        assert row.source_language == LANGUAGE_JA

    def test_returns_cached_value_without_recomputing(self):
        # Cached value MUST be returned as-is even if the text disagrees —
        # that's the whole point of "idempotent" for this bead. A stale
        # cache from a previous policy version should be flushed via
        # force_refresh, not silently overridden.
        row = self._row(
            title="English only", content=None, source_language=LANGUAGE_JA,
        )
        result = detect_and_cache_language(row)
        assert result == LANGUAGE_JA
        assert row.source_language == LANGUAGE_JA

    def test_force_refresh_recomputes(self):
        row = self._row(
            title="English only here", content=None, source_language="xx",
        )
        result = detect_and_cache_language(row, force_refresh=True)
        assert result == LANGUAGE_EN
        assert row.source_language == LANGUAGE_EN

    def test_idempotent_second_call(self):
        # The acceptance contract: two calls in a row must produce the
        # same result, and the second must not mutate state. Use kana so
        # the classification is unambiguous (a kanji-only string shares
        # glyphs with Chinese and can't be disambiguated from script).
        row = self._row(title="日経平均、続伸", content="半導体が牽引した。")
        first = detect_and_cache_language(row)
        original = row.source_language
        second = detect_and_cache_language(row)
        assert first == second == LANGUAGE_JA
        assert row.source_language == original  # unchanged

    def test_handles_missing_content(self):
        row = self._row(title=None, content=None)
        result = detect_and_cache_language(row)
        assert result == LANGUAGE_UNKNOWN
        assert row.source_language == LANGUAGE_UNKNOWN

    def test_title_alone_is_sufficient(self):
        row = self._row(title="ソニーが好決算", content=None)
        assert detect_and_cache_language(row) == LANGUAGE_JA

    def test_content_alone_is_sufficient(self):
        row = self._row(title=None, content="Big tech rallied on earnings.")
        assert detect_and_cache_language(row) == LANGUAGE_EN


class TestPolicySurface:
    def test_policy_version_accessor(self):
        assert policy_version() == POLICY_VERSION

    def test_describe_policy_shape(self):
        snap = describe_policy()
        assert snap["policy_version"] == POLICY_VERSION
        assert set(snap["supported_languages"]) == SUPPORTED_LANGUAGES
        assert 0 < snap["kana_threshold"] < snap["cjk_threshold"] < 1
