"""DB-polling implementation of the CancellationToken port.

Checks the scan's status column to detect user-initiated cancellation.
Uses a single reusable session with ``expire_all()`` to get fresh reads
without the overhead of opening/closing a session per check (avoids
hundreds of session cycles on a 9000-stock scan).

Fail-open: returns ``False`` on DB errors so the scan continues
rather than being silently killed by an infrastructure glitch.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import sessionmaker

from app.domain.scanning.ports import CancellationToken
from app.models.scan_result import Scan

logger = logging.getLogger(__name__)


class DbCancellationToken(CancellationToken):
    """Poll the database for scan cancellation."""

    def __init__(self, session_factory: sessionmaker, scan_id: str) -> None:
        self._session = session_factory()
        self._scan_id = scan_id

    def is_cancelled(self) -> bool:
        try:
            # Invalidate the identity map so the next query hits the DB
            self._session.expire_all()
            status = (
                self._session.query(Scan.status)
                .filter(Scan.scan_id == self._scan_id)
                .scalar()
            )
            return status == "cancelled"
        except Exception:
            logger.warning(
                "Failed to check cancellation for scan %s",
                self._scan_id,
                exc_info=True,
            )
            # The session is reused for the whole scan, and an aborted
            # transaction rejects every later query until it is rolled back.
            try:
                self._session.rollback()
            except Exception:
                logger.warning(
                    "Failed to roll back cancellation session for scan %s",
                    self._scan_id,
                    exc_info=True,
                )
            return False  # fail-open: don't cancel on error

    def close(self) -> None:
        """Close the reusable session. Call in a finally block."""
        try:
            self._session.close()
        except Exception:
            logger.debug("Error closing cancellation session", exc_info=True)
