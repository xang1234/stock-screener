"""Scanning-domain exceptions shared across use cases and repositories."""

from __future__ import annotations


class SingleActiveScanViolation(RuntimeError):
    """Raised when persistence detects a second queued/running scan."""


class DuplicateIdempotencyKey(RuntimeError):
    """Raised when persistence detects a scan already holding the idempotency key."""
