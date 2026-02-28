"""Extraction contracts."""

from .base import Extractor
from .fixture_sanitizer import FixtureSanitizationResult, sanitize_fixture_file, sanitize_fixture_html
from .normalize import TweetNormalizationResult, TweetNormalizer
from .selectors import (
    DEFAULT_SELECTOR_PACK,
    SelectorPack,
    SelectorPackResolution,
    default_selector_pack,
    resolve_selector_pack,
)
from .tweets import PrimaryFallbackTweetExtractor, TweetExtractionResult

__all__ = [
    "DEFAULT_SELECTOR_PACK",
    "Extractor",
    "FixtureSanitizationResult",
    "PrimaryFallbackTweetExtractor",
    "SelectorPack",
    "SelectorPackResolution",
    "TweetNormalizationResult",
    "TweetNormalizer",
    "TweetExtractionResult",
    "sanitize_fixture_file",
    "sanitize_fixture_html",
    "default_selector_pack",
    "resolve_selector_pack",
]
