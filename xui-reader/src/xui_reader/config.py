"""Shared configuration contracts for xui-reader."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import SourceRef


@dataclass(frozen=True)
class AppConfig:
    default_profile: str = "default"
    debug: bool = False


@dataclass(frozen=True)
class BrowserConfig:
    headless: bool = False
    block_resources: bool = True
    navigation_timeout_seconds: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    app: AppConfig = field(default_factory=AppConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    sources: tuple[SourceRef, ...] = ()


def default_config() -> RuntimeConfig:
    return RuntimeConfig()
