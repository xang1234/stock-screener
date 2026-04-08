"""Shared typed exceptions for critical backend reliability paths."""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for typed service exceptions with stable error codes."""

    error_code = "service_error"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        if error_code:
            self.error_code = error_code


class AuthTokenError(ServiceError):
    """Base class for server-auth session token validation failures."""

    error_code = "auth_token_error"


class AuthTokenDecodeError(AuthTokenError):
    """Raised when a session token cannot be decoded/parsing fails."""

    error_code = "auth_token_decode_error"


class AuthTokenSignatureError(AuthTokenError):
    """Raised when token signature validation fails."""

    error_code = "auth_token_signature_error"


class AuthTokenExpiredError(AuthTokenError):
    """Raised when session token has expired."""

    error_code = "auth_token_expired"


class ProviderRateLimitServiceError(ServiceError):
    """Raised when provider requests are throttled beyond retry budget."""

    error_code = "provider_rate_limited"


class ProviderQuotaServiceError(ServiceError):
    """Raised when provider daily/allocated quota is exhausted."""

    error_code = "provider_quota_exhausted"


class ThemeMergeConflictError(ServiceError):
    """Raised when merge execution conflicts with current merge state."""

    error_code = "theme_merge_conflict"


class CacheRefreshError(ServiceError):
    """Raised when cache refresh/update workflow fails in critical hot paths."""

    error_code = "cache_refresh_failed"
