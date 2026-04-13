"""Unit tests for the translation stage (T7.3)."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Optional

import pytest

from app.services.translation_service import (
    CONFIDENCE_DOWNGRADE_THRESHOLD,
    PROVIDER_IDENTITY,
    PROVIDER_UNAVAILABLE,
    POLICY_VERSION,
    TranslationQuote,
    TranslationService,
    describe_policy,
    identity_metadata,
    policy_version,
    select_extraction_text,
    should_downgrade_for_translation,
    translation_confidence,
    unavailable_metadata,
)


def _row(
    *,
    id: int = 1,
    title: Optional[str] = None,
    content: Optional[str] = None,
    source_language: Optional[str] = None,
    translated_title: Optional[str] = None,
    translated_content: Optional[str] = None,
    translation_metadata: Optional[dict] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        title=title,
        content=content,
        source_language=source_language,
        translated_title=translated_title,
        translated_content=translated_content,
        translation_metadata=translation_metadata,
    )


def _make_translator(
    *,
    title_confidence: float = 0.9,
    content_confidence: float = 0.85,
    provider: str = "deepl",
    model: str = "deepl-v1",
):
    """Deterministic fake translator for tests."""
    def translator(text, source, target):
        confidence = title_confidence if len(text) < 50 else content_confidence
        translated = f"[{source}->{target}] {text}"
        quote = TranslationQuote(
            source_language=source,
            target_language=target,
            provider=provider,
            model=model,
            confidence=confidence,
            translated_at=date.today(),
        )
        return translated, quote
    return translator


class TestIdentityPath:
    """English source == English target: mark as identity, copy originals."""

    def test_identity_copies_originals(self):
        svc = TranslationService(_make_translator(), target_language="en")
        row = _row(
            title="Nvidia posts record quarter",
            content="Revenue jumped 30%.",
            source_language="en",
        )
        meta = svc.translate_content_item(row)
        assert meta["provider"] == PROVIDER_IDENTITY
        assert meta["source_language"] == "en"
        assert meta["target_language"] == "en"
        assert meta["confidence"] == 1.0
        assert row.translated_title == "Nvidia posts record quarter"
        assert row.translated_content == "Revenue jumped 30%."


class TestSuccessfulTranslation:
    def test_persists_translated_text_and_metadata(self):
        svc = TranslationService(_make_translator(), target_language="en")
        row = _row(
            title="日経平均", content="半導体が牽引。", source_language="ja",
        )
        meta = svc.translate_content_item(row)
        assert meta["provider"] == "deepl"
        assert meta["source_language"] == "ja"
        assert meta["target_language"] == "en"
        assert row.translated_title.startswith("[ja->en]")
        assert row.translated_content.startswith("[ja->en]")

    def test_combined_confidence_uses_weaker_signal(self):
        # Title confidence 0.9, content confidence 0.60 → merged = 0.60.
        # QA scoring should gate on the weaker half of the row.
        svc = TranslationService(
            _make_translator(title_confidence=0.9, content_confidence=0.60),
            target_language="en",
        )
        row = _row(
            title="短い", content="長めの本文" * 20, source_language="ja",
        )
        meta = svc.translate_content_item(row)
        assert meta["confidence"] == 0.60


class TestIdempotency:
    """Bead acceptance: retries must not duplicate successful translation work."""

    def test_second_call_after_success_is_noop(self):
        calls = []

        def counting_translator(text, src, tgt):
            calls.append((text, src, tgt))
            quote = TranslationQuote(
                source_language=src, target_language=tgt, provider="deepl",
                model="v1", confidence=0.95, translated_at=date.today(),
            )
            return f"[{text}]", quote

        svc = TranslationService(counting_translator, target_language="en")
        row = _row(title="日経", content="半導体", source_language="ja")
        svc.translate_content_item(row)
        first_call_count = len(calls)

        # Second call on the same row must skip translation entirely.
        svc.translate_content_item(row)
        assert len(calls) == first_call_count

    def test_unavailable_retries_on_next_call(self):
        """Transient failures must not permanently mark the row un-translatable."""
        fail_count = {"n": 0}

        def flaky_translator(text, src, tgt):
            if fail_count["n"] == 0:
                fail_count["n"] += 1
                raise RuntimeError("transient")
            quote = TranslationQuote(
                source_language=src, target_language=tgt, provider="deepl",
                model="v1", confidence=0.9, translated_at=date.today(),
            )
            return f"[{text}]", quote

        svc = TranslationService(flaky_translator, target_language="en")
        row = _row(title="日経", content="半導体", source_language="ja")

        svc.translate_content_item(row)
        assert row.translation_metadata["provider"] == PROVIDER_UNAVAILABLE
        assert row.translated_title is None

        svc.translate_content_item(row)  # retry
        assert row.translation_metadata["provider"] == "deepl"
        assert row.translated_title == "[日経]"

    def test_force_refresh_overrides_cache(self):
        calls = []

        def counting_translator(text, src, tgt):
            calls.append(text)
            quote = TranslationQuote(
                source_language=src, target_language=tgt, provider="deepl",
                model="v1", confidence=0.9, translated_at=date.today(),
            )
            return f"v{len(calls)}[{text}]", quote

        svc = TranslationService(counting_translator, target_language="en")
        row = _row(title="日経", content="半導体", source_language="ja")
        svc.translate_content_item(row)
        assert row.translated_title == "v1[日経]"

        svc.translate_content_item(row, force_refresh=True)
        assert row.translated_title == "v3[日経]"  # calls 3 and 4 ran

    def test_different_target_language_retranslates(self):
        svc_en = TranslationService(_make_translator(), target_language="en")
        svc_ja = TranslationService(_make_translator(), target_language="ja")
        row = _row(title="恒生指數", content="科技股領漲", source_language="zh")

        svc_en.translate_content_item(row)
        assert row.translation_metadata["target_language"] == "en"

        # Asking for ja now must re-translate (different target).
        svc_ja.translate_content_item(row)
        assert row.translation_metadata["target_language"] == "ja"
        assert row.translated_title.startswith("[zh->ja]")


class TestFailureAtomicity:
    def test_translator_exception_writes_unavailable_and_clears_translated(self):
        def failing(text, src, tgt):
            raise ConnectionError("upstream down")

        svc = TranslationService(failing, target_language="en")
        # Pre-seed translated fields to prove they get cleared.
        row = _row(
            title="日経", content="半導体", source_language="ja",
            translated_title="stale", translated_content="stale",
        )
        meta = svc.translate_content_item(row)
        assert meta["provider"] == PROVIDER_UNAVAILABLE
        assert "upstream down" in meta["reason"]
        assert row.translated_title is None
        assert row.translated_content is None

    def test_unavailable_reason_is_truncated(self):
        def huge_reason(text, src, tgt):
            raise RuntimeError("x" * 1000)

        svc = TranslationService(huge_reason, target_language="en")
        row = _row(title="a", content="b", source_language="ja")
        meta = svc.translate_content_item(row)
        # Reason string includes "RuntimeError: " prefix then xs, capped at 200.
        assert len(meta["reason"]) == 200


class TestMissingSourceLanguage:
    def test_raises_when_language_not_detected(self):
        svc = TranslationService(_make_translator())
        row = _row(title="未知", content="未知", source_language=None)
        with pytest.raises(ValueError, match="detect_and_cache_language"):
            svc.translate_content_item(row)


class TestSelectExtractionText:
    def test_prefers_translated_when_available(self):
        row = _row(
            title="日経", content="半導体",
            source_language="ja",
            translated_title="Nikkei", translated_content="Semiconductors",
            translation_metadata={
                "provider": "deepl", "source_language": "ja",
                "target_language": "en", "confidence": 0.9,
            },
        )
        title, content, lang = select_extraction_text(row)
        assert title == "Nikkei"
        assert content == "Semiconductors"
        assert lang == "en"

    def test_identity_returns_target_language(self):
        row = _row(
            title="Hello", content="World",
            source_language="en",
            translated_title="Hello", translated_content="World",
            translation_metadata=identity_metadata("en", "en"),
        )
        title, content, lang = select_extraction_text(row)
        assert title == "Hello"
        assert lang == "en"

    def test_unavailable_falls_back_to_original(self):
        row = _row(
            title="日経", content="半導体",
            source_language="ja",
            translated_title=None, translated_content=None,
            translation_metadata=unavailable_metadata("ja", "en", "network"),
        )
        title, content, lang = select_extraction_text(row)
        assert title == "日経"
        assert content == "半導体"
        assert lang == "ja"  # flags the caller that this is un-translated original

    def test_no_metadata_returns_original(self):
        row = _row(title="日経", content="半導体", source_language="ja")
        title, content, lang = select_extraction_text(row)
        assert (title, content, lang) == ("日経", "半導体", "ja")


class TestShouldDowngradeForTranslation:
    def test_unavailable_triggers_downgrade(self):
        row = _row(translation_metadata=unavailable_metadata("ja", "en", "boom"))
        assert should_downgrade_for_translation(row) is True

    def test_high_confidence_does_not_downgrade(self):
        row = _row(translation_metadata={
            "provider": "deepl", "source_language": "ja", "target_language": "en",
            "confidence": 0.95,
        })
        assert should_downgrade_for_translation(row) is False

    def test_below_threshold_downgrades(self):
        row = _row(translation_metadata={
            "provider": "deepl", "source_language": "ja", "target_language": "en",
            "confidence": CONFIDENCE_DOWNGRADE_THRESHOLD - 0.05,
        })
        assert should_downgrade_for_translation(row) is True

    def test_identity_never_downgrades(self):
        row = _row(translation_metadata=identity_metadata("en", "en"))
        assert should_downgrade_for_translation(row) is False

    def test_no_metadata_returns_false(self):
        # Not-yet-translated is not this function's call; the extraction
        # pipeline is expected to have run translation first.
        row = _row()
        assert should_downgrade_for_translation(row) is False


class TestTranslationConfidence:
    def test_returns_float(self):
        row = _row(translation_metadata={"confidence": 0.8})
        assert translation_confidence(row) == 0.8

    def test_handles_none_and_missing(self):
        assert translation_confidence(_row()) is None
        assert translation_confidence(_row(translation_metadata={})) is None
        assert translation_confidence(
            _row(translation_metadata={"confidence": None})
        ) is None


class TestPolicySurface:
    def test_version_accessor(self):
        assert policy_version() == POLICY_VERSION

    def test_describe_policy_pins_threshold(self):
        snap = describe_policy()
        assert snap["policy_version"] == POLICY_VERSION
        assert snap["default_target_language"] == "en"
        assert snap["confidence_downgrade_threshold"] == 0.70
        assert snap["identity_provider"] == PROVIDER_IDENTITY
        assert snap["unavailable_provider"] == PROVIDER_UNAVAILABLE
