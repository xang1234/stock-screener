"""xui_reader package scaffold."""

from .config import (
    AppConfig,
    BrowserConfig,
    RuntimeConfig,
    config_to_dict,
    default_config,
    init_default_config,
    load_runtime_config,
    resolve_config_path,
)
from .models import Checkpoint, SourceRef, TweetItem

__all__ = [
    "AppConfig",
    "BrowserConfig",
    "Checkpoint",
    "RuntimeConfig",
    "SourceRef",
    "TweetItem",
    "config_to_dict",
    "default_config",
    "init_default_config",
    "load_runtime_config",
    "resolve_config_path",
]

__version__ = "0.1.0"
