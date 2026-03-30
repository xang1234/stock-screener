"""
Z.AI API Key Manager - Handles multiple API keys with rate-limit-aware rotation.

Behavior:
- On rate limit: Continue retrying with the SAME key (exponential backoff)
- On the NEXT API call (new request): 50% chance to rotate to a different key
- This ensures the API never sees identical parameters from a different key
"""
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_COOLDOWN_SECONDS = 60.0
MAX_COOLDOWN_SECONDS = 300.0
ROTATION_PROBABILITY = 0.5

# Module-level singleton storage (not in dataclass)
_zai_key_manager_instance: Optional["ZAIKeyManager"] = None
_zai_key_manager_lock = threading.Lock()


@dataclass
class KeyState:
    """State tracking for a single API key."""

    key: str
    request_count: int = 0
    rate_limit_count: int = 0
    rate_limited_until: float = 0.0
    last_used: float = 0.0

    @property
    def is_rate_limited(self) -> bool:
        """Check if key is currently rate limited."""
        return time.time() < self.rate_limited_until

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in cooldown (0 if not rate limited)."""
        remaining = self.rate_limited_until - time.time()
        return max(0.0, remaining)


@dataclass
class ZAIKeyManager:
    """
    Manages multiple Z.AI API keys with rate-limit-aware rotation.

    Thread-safe singleton that:
    - Tracks multiple keys with usage stats
    - Rotates keys probabilistically AFTER rate limits (not during retry)
    - Provides stats for monitoring
    """

    keys: List[str] = field(default_factory=list)
    _key_states: Dict[str, KeyState] = field(default_factory=dict)
    _current_index: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _should_rotate_next: bool = False
    _last_rate_limited_key: Optional[str] = None

    def __post_init__(self):
        """Initialize key states."""
        for key in self.keys:
            if key not in self._key_states:
                self._key_states[key] = KeyState(key=key)

    @classmethod
    def get_instance(cls, keys: Optional[List[str]] = None) -> "ZAIKeyManager":
        """Get or create the singleton instance."""
        global _zai_key_manager_instance

        if _zai_key_manager_instance is None:
            with _zai_key_manager_lock:
                if _zai_key_manager_instance is None:
                    if keys is None:
                        keys = _get_keys_from_settings()

                    _zai_key_manager_instance = cls(keys=keys or [])
                    if keys:
                        logger.info("ZAIKeyManager initialized with %s key(s)", len(keys))
                    else:
                        logger.debug("ZAIKeyManager initialized with no keys")
        return _zai_key_manager_instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        global _zai_key_manager_instance
        with _zai_key_manager_lock:
            _zai_key_manager_instance = None

    def get_key(self) -> Optional[str]:
        """Get the current API key, potentially rotating after a rate limit."""
        if not self.keys:
            return None

        with self._lock:
            if self._should_rotate_next and len(self.keys) > 1:
                if random.random() < ROTATION_PROBABILITY:
                    self._rotate_key()
                    logger.info(
                        "Rotated Z.AI key to index %s (50%% chance triggered after rate limit)",
                        self._current_index,
                    )
                else:
                    logger.debug(
                        "Staying with Z.AI key index %s (50%% chance - no rotation)",
                        self._current_index,
                    )
                self._should_rotate_next = False
                self._last_rate_limited_key = None

            key = self.keys[self._current_index]
            state = self._key_states[key]
            state.request_count += 1
            state.last_used = time.time()
            return key

    def _rotate_key(self):
        """Rotate to the next available key (internal, must hold lock)."""
        if len(self.keys) <= 1:
            return

        original_index = self._current_index

        for _ in range(len(self.keys)):
            self._current_index = (self._current_index + 1) % len(self.keys)
            key = self.keys[self._current_index]
            state = self._key_states[key]

            if not state.is_rate_limited:
                return

        best_index = original_index
        best_cooldown = float("inf")

        for i, key in enumerate(self.keys):
            state = self._key_states[key]
            if state.cooldown_remaining < best_cooldown:
                best_cooldown = state.cooldown_remaining
                best_index = i

        self._current_index = best_index
        logger.warning(
            "All Z.AI keys rate limited. Using key index %s (cooldown: %.1fs remaining)",
            best_index,
            best_cooldown,
        )

    def report_rate_limit(self, key: str, retry_after: Optional[float] = None) -> None:
        """Report that a key hit a rate limit."""
        if key not in self._key_states:
            logger.warning("Unknown Z.AI key reported rate limit: %s...", key[:8])
            return

        with self._lock:
            state = self._key_states[key]
            state.rate_limit_count += 1

            cooldown = retry_after or DEFAULT_COOLDOWN_SECONDS
            cooldown = min(cooldown, MAX_COOLDOWN_SECONDS)
            state.rate_limited_until = time.time() + cooldown

            self._should_rotate_next = True
            self._last_rate_limited_key = key

            logger.info(
                "Z.AI key %s... rate limited for %.1fs. Next request may rotate (50%% chance). Total rate limits for this key: %s",
                key[:8],
                cooldown,
                state.rate_limit_count,
            )

    def get_stats(self) -> Dict:
        """Get statistics about key usage."""
        with self._lock:
            stats = {
                "total_keys": len(self.keys),
                "current_index": self._current_index,
                "pending_rotation": self._should_rotate_next,
                "keys": [],
            }

            for i, key in enumerate(self.keys):
                state = self._key_states[key]
                stats["keys"].append(
                    {
                        "index": i,
                        "key_prefix": key[:8] + "..." if len(key) > 8 else key,
                        "request_count": state.request_count,
                        "rate_limit_count": state.rate_limit_count,
                        "is_rate_limited": state.is_rate_limited,
                        "cooldown_remaining": round(state.cooldown_remaining, 1),
                        "is_current": i == self._current_index,
                    }
                )

            return stats

    def __len__(self) -> int:
        """Return number of keys."""
        return len(self.keys)


def _get_keys_from_settings() -> List[str]:
    """
    Get Z.AI API keys from settings with fallbacks.

    Tries multiple sources in order:
    1. settings.zai_api_keys_list property (if exists)
    2. settings.zai_api_keys string (comma-separated)
    3. settings.zai_api_key (single key)
    4. ZAI_API_KEYS environment variable
    5. ZAI_API_KEY environment variable
    """

    keys: List[str] = []

    try:
        from ...config import settings

        if hasattr(settings, "zai_api_keys_list"):
            keys = settings.zai_api_keys_list
        elif hasattr(settings, "zai_api_keys") and settings.zai_api_keys:
            keys = [k.strip() for k in settings.zai_api_keys.split(",") if k.strip()]
        elif hasattr(settings, "zai_api_key") and settings.zai_api_key:
            keys = [settings.zai_api_key]
    except Exception as exc:
        logger.warning("Could not load Z.AI keys from settings: %s", exc)

    if not keys:
        env_keys = os.environ.get("ZAI_API_KEYS", "")
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        elif os.environ.get("ZAI_API_KEY"):
            keys = [os.environ["ZAI_API_KEY"]]

    return keys


def get_zai_key_manager() -> ZAIKeyManager:
    """Get the singleton ZAIKeyManager instance."""
    return ZAIKeyManager.get_instance()
