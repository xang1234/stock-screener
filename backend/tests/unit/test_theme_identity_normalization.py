"""Tests for theme identity normalization utilities and integration hooks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.services.theme_extraction_service import ThemeExtractionService
from app.services.theme_identity_normalization import (
    KEY_REGEX,
    MAX_KEY_LENGTH,
    canonical_theme_key,
    display_theme_name,
)


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


def test_extraction_service_normalize_theme_uses_legacy_map_and_fallback():
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    assert service._normalize_theme("AI infra") == "AI Infrastructure"
    assert service._normalize_theme("Quantum tech") == "Quantum Computing"
    assert service._normalize_theme("A.I. datacenter buildout") == "AI Datacenter Buildout"
