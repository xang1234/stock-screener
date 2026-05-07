"""Helpers for retrying Celery tasks through transient database startup states."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.exc import DBAPIError

TRANSIENT_DATABASE_RETRY_MAX_RETRIES = 12
TRANSIENT_DATABASE_RETRY_BASE_SECONDS = 5
TRANSIENT_DATABASE_RETRY_MAX_SECONDS = 60

_TRANSIENT_DATABASE_MESSAGES = (
    "database system is not yet accepting connections",
    "consistent recovery state has not been yet reached",
    "database system is starting up",
    "database system is in recovery mode",
    "the database system is shutting down",
    "connection refused",
    "could not connect to server",
    "server closed the connection unexpectedly",
    "terminating connection due to administrator command",
    "connection reset by peer",
)


def is_transient_database_error(exc: Exception) -> bool:
    """Return true for DB connection errors that should be retried by Celery."""
    if not isinstance(exc, DBAPIError):
        return False
    if getattr(exc, "connection_invalidated", False):
        return True
    orig = getattr(exc, "orig", None)
    message = str(orig if orig is not None else exc).lower()
    return any(fragment in message for fragment in _TRANSIENT_DATABASE_MESSAGES)


def raise_if_transient_database_error(exc: Exception) -> None:
    """Propagate transient DB errors out of task bodies so wrappers can retry."""
    if is_transient_database_error(exc):
        raise exc


def retry_transient_database_error(
    task: Any,
    task_name: str,
    exc: Exception,
    *,
    logger: logging.Logger,
    max_retries: int = TRANSIENT_DATABASE_RETRY_MAX_RETRIES,
) -> None:
    """Schedule a Celery retry when ``exc`` is a transient DB connection failure."""
    if task is None or not hasattr(task, "retry"):
        return
    if not is_transient_database_error(exc):
        return

    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    countdown = min(
        TRANSIENT_DATABASE_RETRY_BASE_SECONDS * (2 ** retries),
        TRANSIENT_DATABASE_RETRY_MAX_SECONDS,
    )
    logger.warning(
        "Transient database connection error in %s: %s. Retrying in %ss "
        "(attempt %s/%s).",
        task_name,
        exc,
        countdown,
        retries + 1,
        max_retries,
    )
    raise task.retry(exc=exc, countdown=countdown, max_retries=max_retries)
