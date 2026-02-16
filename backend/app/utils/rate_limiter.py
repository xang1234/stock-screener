"""
Rate limiter utility for API calls.
Implements token bucket algorithm to respect API rate limits.
"""
import time
import threading
from typing import Dict
from collections import deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, rate: float, per: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            rate: Number of requests allowed
            per: Time period in seconds (default: 1 second)

        Example:
            # 1 request per second
            limiter = RateLimiter(rate=1, per=1.0)

            # 25 requests per day
            limiter = RateLimiter(rate=25, per=86400)
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = threading.Lock()

        logger.info(f"Rate limiter initialized: {rate} requests per {per} seconds")

    def wait_if_needed(self) -> float:
        """
        Wait if rate limit would be exceeded.

        Returns:
            Time waited in seconds
        """
        with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current

            # Add tokens based on time passed
            self.allowance += time_passed * (self.rate / self.per)

            # Cap at rate limit
            if self.allowance > self.rate:
                self.allowance = self.rate

            # If we have tokens, use one
            if self.allowance >= 1.0:
                self.allowance -= 1.0
                return 0.0

            # Not enough tokens - calculate wait time
            wait_time = (1.0 - self.allowance) * (self.per / self.rate)

            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            time.sleep(wait_time)

            self.allowance = 0.0
            return wait_time


class ServiceRateLimiters:
    """Manage rate limiters for different services."""

    def __init__(self):
        """Initialize rate limiters for all services."""
        self._limiters: Dict[str, RateLimiter] = {}

    def get_limiter(self, service: str, rate: float, per: float = 1.0) -> RateLimiter:
        """
        Get or create rate limiter for a service.

        Args:
            service: Service name
            rate: Number of requests allowed
            per: Time period in seconds

        Returns:
            RateLimiter instance
        """
        if service not in self._limiters:
            self._limiters[service] = RateLimiter(rate=rate, per=per)

        return self._limiters[service]

    def wait_for_service(self, service: str) -> float:
        """
        Wait for service rate limit if needed.

        Args:
            service: Service name

        Returns:
            Time waited in seconds
        """
        if service not in self._limiters:
            logger.warning(f"No rate limiter configured for {service}")
            return 0.0

        return self._limiters[service].wait_if_needed()


class DailyQuotaTracker:
    """
    Track daily API quota usage.
    Useful for services with daily limits like Alpha Vantage (25 req/day).
    """

    def __init__(self, daily_limit: int):
        """
        Initialize daily quota tracker.

        Args:
            daily_limit: Maximum requests per day
        """
        self.daily_limit = daily_limit
        self.requests: deque = deque()  # Timestamps of requests
        self.lock = threading.Lock()

        logger.info(f"Daily quota tracker initialized: {daily_limit} requests/day")

    def _cleanup_old_requests(self):
        """Remove requests older than 24 hours."""
        cutoff = time.time() - 86400  # 24 hours ago

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def can_make_request(self) -> bool:
        """
        Check if request can be made within daily quota.

        Returns:
            True if under quota, False otherwise
        """
        with self.lock:
            self._cleanup_old_requests()
            return len(self.requests) < self.daily_limit

    def get_remaining_quota(self) -> int:
        """
        Get remaining requests for today.

        Returns:
            Number of requests remaining
        """
        with self.lock:
            self._cleanup_old_requests()
            return self.daily_limit - len(self.requests)

    def record_request(self):
        """Record a new request."""
        with self.lock:
            self.requests.append(time.time())

    def wait_if_quota_exceeded(self) -> float:
        """
        Wait if daily quota is exceeded.

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        with self.lock:
            self._cleanup_old_requests()

            if len(self.requests) < self.daily_limit:
                return 0.0

            # Calculate time until oldest request expires
            oldest_request = self.requests[0]
            wait_time = 86400 - (time.time() - oldest_request)

            if wait_time > 0:
                logger.warning(
                    f"Daily quota exceeded ({len(self.requests)}/{self.daily_limit}). "
                    f"Waiting {wait_time/3600:.1f} hours"
                )
                # Don't actually wait - just return wait time
                # In production, you might queue the request
                return wait_time

            return 0.0


# Global rate limiter instances
rate_limiters = ServiceRateLimiters()

# DEPRECATED: Use app.services.rate_limiter.rate_limiter instead.
# These in-process limiters are kept only for backward compatibility with
# alphavantage_limiter (still used by data_fetcher.py for Alpha Vantage).
# yfinance_limiter is no longer used — all yfinance rate limiting goes through
# the Redis-backed distributed limiter in app.services.rate_limiter.
yfinance_limiter = rate_limiters.get_limiter("yfinance", rate=1, per=1.0)  # DEPRECATED
alphavantage_limiter = rate_limiters.get_limiter("alphavantage", rate=5, per=60.0)  # 5 req/min
alphavantage_quota = DailyQuotaTracker(daily_limit=25)  # 25 req/day
