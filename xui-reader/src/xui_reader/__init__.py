"""xui_reader package scaffold."""

from .config import AppConfig, BrowserConfig, RuntimeConfig, default_config
from .models import Checkpoint, SourceRef, TweetItem

__all__ = [
    "AppConfig",
    "BrowserConfig",
    "Checkpoint",
    "RuntimeConfig",
    "SourceRef",
    "TweetItem",
    "default_config",
]

__version__ = "0.1.0"
