"""Join the economic-taxonomy writer fence for authoritative research writes.

Lock order stays ``writer fence -> domain rows``. Research issuer links,
holds, assessments and decisions take the *shared* side of the same advisory
lock the economic publisher takes exclusively, so a generation cutoff never
observes a half-written research commit. No provider or network call may
happen inside this context.

Unlike ``producer_write`` this does not require a ``TaxonomyAuthority`` row:
deployments that never enabled the economic taxonomy still record research,
and there is no publisher to race with them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.economic_taxonomy_fence import (
    _SQLITE_FENCE,
    ECONOMIC_TAXONOMY_FENCE_KEY,
)


@contextmanager
def research_write(session: Session) -> Iterator[None]:
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock_shared(:key)"),
            {"key": ECONOMIC_TAXONOMY_FENCE_KEY},
        )
        yield
        return
    with _SQLITE_FENCE:
        yield
