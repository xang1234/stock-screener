"""
Groq API Key Manager - Handles multiple API keys with rate-limit-aware rotation.

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
DEFAULT_COOLDOWN_SECONDS = 60.0   # If no retry_after provided
MAX_COOLDOWN_SECONDS = 300.0     # Cap at 5 minutes
ROTATION_PROBABILITY = 0.5       # 50% chance to switch on next call after rate limit


@dataclass
class KeyState:
    """State tracking for a single API key."""
    key: str
    request_count: int = 0
    rate_limit_count: int = 0
    rate_limited_until: float = 0.0  # Unix timestamp
    last_used: float = 0.0  # Unix timestamp

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
class GroqKeyManager:
    """
    Manages multiple Groq API keys with rate-limit-aware rotation.

    Thread-safe manager that:
    - Tracks multiple keys with usage stats
    - Rotates keys probabilistically AFTER rate limits (not during retry)
    - Provides stats for monitoring

    Usage:
        key = manager.get_key()
        # ... make API call ...
        # On rate limit:
        manager.report_rate_limit(key, retry_after=60.0)
        # Continue retrying with same key!
        # Next NEW request will potentially use a different key
    """
    keys: List[str] = field(default_factory=list)
    _key_states: Dict[str, KeyState] = field(default_factory=dict)
    _current_index: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _should_rotate_next: bool = False  # Flag to rotate on next get_key()
    _last_rate_limited_key: Optional[str] = None

    def __post_init__(self):
        """Initialize key states."""
        for key in self.keys:
            if key not in self._key_states:
                self._key_states[key] = KeyState(key=key)

    def get_key(self) -> Optional[str]:
        """
        Get the current API key, potentially rotating after a rate limit.

        Returns:
            API key string, or None if no keys configured
        """
        if not self.keys:
            return None

        with self._lock:
            # Check if we should rotate due to previous rate limit
            if self._should_rotate_next and len(self.keys) > 1:
                if random.random() < ROTATION_PROBABILITY:
                    self._rotate_key()
                    logger.info(
                        f"Rotated to key index {self._current_index} "
                        f"(50% chance triggered after rate limit)"
                    )
                else:
                    logger.debug(
                        f"Staying with key index {self._current_index} "
                        f"(50% chance - no rotation)"
                    )
                self._should_rotate_next = False
                self._last_rate_limited_key = None

            # Get current key
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

        # Try to find a non-rate-limited key
        for _ in range(len(self.keys)):
            self._current_index = (self._current_index + 1) % len(self.keys)
            key = self.keys[self._current_index]
            state = self._key_states[key]

            if not state.is_rate_limited:
                return  # Found a good key

        # All keys are rate limited - pick the one with shortest cooldown
        best_index = original_index
        best_cooldown = float('inf')

        for i, key in enumerate(self.keys):
            state = self._key_states[key]
            if state.cooldown_remaining < best_cooldown:
                best_cooldown = state.cooldown_remaining
                best_index = i

        self._current_index = best_index
        logger.warning(
            f"All keys rate limited. Using key index {best_index} "
            f"(cooldown: {best_cooldown:.1f}s remaining)"
        )

    def report_rate_limit(
        self,
        key: str,
        retry_after: Optional[float] = None
    ) -> None:
        """
        Report that a key hit a rate limit.

        This marks the key as rate-limited and flags for potential rotation
        on the NEXT get_key() call (not immediately).

        Args:
            key: The API key that hit the rate limit
            retry_after: Seconds until rate limit resets (from server)
        """
        if key not in self._key_states:
            logger.warning(f"Unknown key reported rate limit: {key[:8]}...")
            return

        with self._lock:
            state = self._key_states[key]
            state.rate_limit_count += 1

            # Calculate cooldown
            cooldown = retry_after or DEFAULT_COOLDOWN_SECONDS
            cooldown = min(cooldown, MAX_COOLDOWN_SECONDS)
            state.rate_limited_until = time.time() + cooldown

            # Flag for potential rotation on next NEW request
            self._should_rotate_next = True
            self._last_rate_limited_key = key

            logger.info(
                f"Key {key[:8]}... rate limited for {cooldown:.1f}s. "
                f"Next request may rotate (50% chance). "
                f"Total rate limits for this key: {state.rate_limit_count}"
            )

    def get_stats(self) -> Dict:
        """
        Get statistics about key usage.

        Returns:
            Dict with key usage stats
        """
        with self._lock:
            stats = {
                "total_keys": len(self.keys),
                "current_index": self._current_index,
                "pending_rotation": self._should_rotate_next,
                "keys": []
            }

            for i, key in enumerate(self.keys):
                state = self._key_states[key]
                key_stats = {
                    "index": i,
                    "key_prefix": key[:8] + "..." if len(key) > 8 else key,
                    "request_count": state.request_count,
                    "rate_limit_count": state.rate_limit_count,
                    "is_rate_limited": state.is_rate_limited,
                    "cooldown_remaining": round(state.cooldown_remaining, 1),
                    "is_current": i == self._current_index,
                }
                stats["keys"].append(key_stats)

            return stats

    def __len__(self) -> int:
        """Return number of keys."""
        return len(self.keys)


def _get_keys_from_settings() -> List[str]:
    """
    Get Groq API keys from settings with fallbacks.

    Tries multiple sources in order:
    1. settings.groq_api_keys_list property (if exists)
    2. settings.groq_api_keys string (comma-separated)
    3. settings.groq_api_key (single key)
    4. GROQ_API_KEYS environment variable
    5. GROQ_API_KEY environment variable
    """
    keys = []

    try:
        from ...config import settings

        # Try the list property first
        if hasattr(settings, 'groq_api_keys_list'):
            keys = settings.groq_api_keys_list
        # Fallback: parse groq_api_keys string manually
        elif hasattr(settings, 'groq_api_keys') and settings.groq_api_keys:
            keys = [k.strip() for k in settings.groq_api_keys.split(",") if k.strip()]
        # Fallback: single key
        elif hasattr(settings, 'groq_api_key') and settings.groq_api_key:
            keys = [settings.groq_api_key]
    except Exception as e:
        logger.warning(f"Could not load keys from settings: {e}")

    # Final fallback: environment variables
    if not keys:
        env_keys = os.environ.get("GROQ_API_KEYS", "")
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        elif os.environ.get("GROQ_API_KEY"):
            keys = [os.environ["GROQ_API_KEY"]]

    return keys


# Module-level convenience function
def get_groq_key_manager() -> GroqKeyManager:
    """Get the process-wide GroqKeyManager instance from bootstrap wiring."""
    from ...wiring.bootstrap import get_groq_key_manager as _provider

    return _provider()
