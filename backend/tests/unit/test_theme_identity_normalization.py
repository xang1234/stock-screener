"""Tests for theme identity normalization utilities and integration hooks."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.services.theme_extraction_service import ThemeExtractionService
from app.services.theme_identity_normalization import (
    KEY_REGEX,
    MAX_KEY_LENGTH,
    canonical_theme_key,
    display_theme_name,
)
from app.services.theme_merging_service import ThemeMergingService

ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = ROOT / "docs" / "theme_identity" / "adr_e2_canonical_key_normalization_v1.corpus.json"


def test_canonical_theme_key_matches_policy_corpus():
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        assert canonical_theme_key(case["raw"]) == case["expected_key"], case["id"]


def test_canonical_theme_key_idempotent_and_regex_safe():
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        key = canonical_theme_key(case["raw"])
        assert canonical_theme_key(key) == key
        assert KEY_REGEX.fullmatch(key), case["id"]
        assert len(key) <= MAX_KEY_LENGTH, case["id"]


def test_near_collision_examples_stay_distinct():
    left = canonical_theme_key("AI data center buildout")
    right = canonical_theme_key("AI datacenter buildout")
    assert left == "ai_data_center_buildout"
    assert right == "ai_datacenter_buildout"
    assert left != right


def test_overflow_key_uses_deterministic_hash_suffix():
    raw = " ".join(["infrastructure"] * 20)
    key = canonical_theme_key(raw)
    assert len(key) <= MAX_KEY_LENGTH
    assert re.fullmatch(r"[a-z0-9_]+_[0-9a-f]{8}", key)


def test_display_theme_name_formats_acronyms_and_numeric_tokens():
    assert display_theme_name("AI infrastructure") == "AI Infrastructure"
    assert display_theme_name("GLP-1 weight loss") == "GLP-1 Weight Loss"
    assert display_theme_name("5g infrastructure") == "5G Infrastructure"
    assert display_theme_name("   the of for to   ") == "Unknown Theme"


def test_plus_normalization_respects_token_context():
    assert canonical_theme_key("C++ tooling") == "c_plus_plus_tooling"
    assert canonical_theme_key("+AI infrastructure") == "ai_infrastructure"
    assert canonical_theme_key("AI+Infrastructure") == "ai_plus_infrastructure"


def test_canonical_theme_key_normalizes_diacritics_and_numeric_grouping():
    assert canonical_theme_key("Café Robotics 1,000 GPUs") == "cafe_robotic_1000_gpu"


def test_extraction_service_normalize_theme_uses_legacy_map_and_fallback():
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    assert service._normalize_theme("AI infra") == "AI Infrastructure"
    assert service._normalize_theme("Quantum tech") == "Quantum Computing"
    assert service._normalize_theme("A.I. datacenter buildout") == "AI Datacenter Buildout"


def test_extract_from_content_filters_empty_and_unknown_themes():
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.provider = "litellm"
    service.gemini_client = None
    service._rate_limit = lambda: None
    service._clean_tickers = lambda _tickers, *, source_text=None: []
    service._try_generate_litellm = lambda _prompt: json.dumps(
        [
            {"theme": "   ", "tickers": [], "sentiment": "neutral", "confidence": 0.5, "excerpt": ""},
            {"theme": "the of for to", "tickers": [], "sentiment": "neutral", "confidence": 0.5, "excerpt": ""},
            {"theme": "AI infrastructure", "tickers": [], "sentiment": "bullish", "confidence": 0.9, "excerpt": "x"},
        ]
    )
    content_item = SimpleNamespace(
        content="body",
        source_name="src",
        source_type="news",
        source_language=None,
        published_at=datetime.utcnow(),
        title="title",
    )

    mentions = service.extract_from_content(content_item, verify_claims=False)
    assert len(mentions) == 1
    assert mentions[0]["theme"] == "AI infrastructure"


def test_merging_service_preserves_valid_suggested_name_text():
    service = ThemeMergingService.__new__(ThemeMergingService)
    assert service._normalize_suggested_name("  GLP-1 Weight Loss Drugs  ") == "GLP-1 Weight Loss Drugs"
    assert service._normalize_suggested_name("the of for to") is None


@pytest.mark.parametrize("name", ["Nvidia", "Tesla", "Tencent", "Sony"])
def test_unambiguous_single_word_issuer_alias(name):
    from app.services.theme_extraction_service import ThemeExtractionService
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    text = f"Analysts expect {name} to benefit from stronger demand."
    normalized = f" {service._normalize_company_text(text)} "
    assert service._text_contains_company_alias(text, normalized, name.lower())


@pytest.mark.parametrize("location", ["title", "attachment"])
def test_commodity_guard_reads_all_extraction_evidence(location):
    import json
    from datetime import datetime, timezone
    from hashlib import sha256
    from types import SimpleNamespace

    from app.services.theme_extraction_service import ThemeExtractionService
    from app.services.theme_grounding_context import GroundingContext, GroundingEvidence

    text = "Copper $HG is rising."
    digest = sha256(text.encode()).hexdigest()
    context = GroundingContext(evidence=[] if location == "title" else [GroundingEvidence(
        input_id="a" * 64, source_id="1", source_url="https://example.com/chart",
        input_kind="image_observation", relation="attached_image", text=text,
        text_sha256=digest, original_text_sha256=digest, available_at=datetime.now(timezone.utc))])
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.provider, service.db = "litellm", None
    service._valid_tickers = {"HG"}
    service._rate_limit = lambda: None
    service._resolve_company_name_tickers = lambda *args: []
    service._try_generate_litellm = lambda *args, **kwargs: json.dumps([{
        "theme": "Copper", "tickers": ["HG"], "confidence": .8,
        "sentiment": "bullish", "excerpt": text}])
    item = SimpleNamespace(title=text if location == "title" else "", content="",
        source_name="Test", source_type="news", source_language="en", published_at=None)
    result = service.extract_from_content(item, verify_claims=False, grounding_context=context)
    assert result[0]["tickers"] == []
