"""Extraction contracts."""

from .base import Extractor
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
    "PrimaryFallbackTweetExtractor",
    "SelectorPack",
    "SelectorPackResolution",
    "TweetNormalizationResult",
    "TweetNormalizer",
    "TweetExtractionResult",
    "default_selector_pack",
    "resolve_selector_pack",
]
