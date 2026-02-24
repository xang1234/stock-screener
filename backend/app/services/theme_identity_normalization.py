"""Deterministic theme identity normalization helpers.

This module implements the v1 policy in
docs/theme_identity/adr_e2_canonical_key_normalization_v1.md.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable


KEY_REGEX = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
MAX_KEY_LENGTH = 96
TRUNCATE_PREFIX_LENGTH = 80
TRUNCATE_HASH_LENGTH = 8
UNKNOWN_THEME_KEY = "unknown_theme"

_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "from",
}

_PLURAL_EXCEPTIONS = {"gas", "as", "us", "esg", "saas", "loss", "plus"}

_TOKEN_MAP = {
    "ai": "ai",
    "ml": "ml",
    "llm": "llm",
    "glp": "glp1",
    "glp1": "glp1",
    "gpu": "gpu",
    "gpus": "gpu",
    "ev": "ev",
    "evs": "ev",
    "ipo": "ipo",
    "ipos": "ipo",
    "etf": "etf",
    "etfs": "etf",
}

_DISPLAY_TOKEN_MAP = {
    "ai": "AI",
    "ml": "ML",
    "llm": "LLM",
    "glp1": "GLP-1",
    "gpu": "GPU",
    "ev": "EV",
    "ipo": "IPO",
    "etf": "ETF",
    "esg": "ESG",
    "us": "US",
    "gaap": "GAAP",
}


def canonical_theme_key(raw_theme: str) -> str:
    """Return deterministic canonical key for a raw theme string."""
    text = _normalize_text(raw_theme)
    text = _apply_pre_replacements(text)
    text = _replace_separators(text)
    tokens = _tokenize(text)
    tokens = _combine_letter_acronyms(tokens)
    tokens = [_normalize_token(token) for token in tokens]
    tokens = [token for token in tokens if token and token not in _STOPWORDS]

    if not tokens:
        return UNKNOWN_THEME_KEY

    key = "_".join(tokens)
    key = _truncate_key_if_needed(key)
    if not KEY_REGEX.fullmatch(key):
        return UNKNOWN_THEME_KEY
    return key


def display_theme_name(raw_theme: str) -> str:
    """Return UI-friendly display name derived from canonical key."""
    key = canonical_theme_key(raw_theme)
    if key == UNKNOWN_THEME_KEY:
        return "Unknown Theme"

    display_tokens = []
    for token in key.split("_"):
        mapped = _DISPLAY_TOKEN_MAP.get(token)
        if mapped is not None:
            display_tokens.append(mapped)
            continue

        if _is_alnum_token(token):
            display_tokens.append(_format_alnum_token(token))
            continue

        display_tokens.append(token.capitalize())

    return " ".join(display_tokens)


def _normalize_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _apply_pre_replacements(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\bglp[\s\-]+1\b", "glp1", text)
    text = re.sub(r"\ba/s\b", "as", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text


def _replace_separators(text: str) -> str:
    if not text:
        return text
    text = text.replace("&", " and ")
    text = text.replace("+", " plus ")
    text = re.sub(r"[\/\\|\-_\.,:;\(\)\[\]\{\}'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [token for token in text.split(" ") if token]


def _combine_letter_acronyms(tokens: Iterable[str]) -> list[str]:
    merged: list[str] = []
    src = list(tokens)
    i = 0
    while i < len(src):
        if i + 1 < len(src) and src[i] == "a" and src[i + 1] == "i":
            merged.append("ai")
            i += 2
            continue
        if i + 1 < len(src) and src[i] == "m" and src[i + 1] == "l":
            merged.append("ml")
            i += 2
            continue
        if i + 2 < len(src) and src[i] == "l" and src[i + 1] == "l" and src[i + 2] == "m":
            merged.append("llm")
            i += 3
            continue
        merged.append(src[i])
        i += 1
    return merged


def _normalize_token(token: str) -> str:
    token = _TOKEN_MAP.get(token, token)
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and token not in _PLURAL_EXCEPTIONS:
        return token[:-1]
    return token


def _truncate_key_if_needed(key: str) -> str:
    if len(key) <= MAX_KEY_LENGTH:
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:TRUNCATE_HASH_LENGTH]
    return f"{key[:TRUNCATE_PREFIX_LENGTH]}_{digest}"


def _is_alnum_token(token: str) -> bool:
    return any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token)


def _format_alnum_token(token: str) -> str:
    return "".join(ch.upper() if ch.isalpha() else ch for ch in token)
