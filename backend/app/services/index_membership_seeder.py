"""Seed ``stock_universe_index_membership`` rows from a constituent CSV.

Factored out of ``scripts/seed_index_memberships.py`` so the upsert logic is
importable (for tests, for management tasks) while the script stays a thin
argparse wrapper. CSV schema: ``symbol,name`` with a header row.

Upsert policy: INSERT on new (symbol, index_name); UPDATE ``as_of_date`` +
``source`` when the pair already exists. Rows whose symbols drop out of the
CSV are NOT deleted — if a quarterly rebalance removes a constituent, the
operator must re-run with a fresh CSV AND prune stale rows separately.
This keeps the seed path idempotent and side-effect-scoped.

Symbol normalization goes through :func:`symbol_format.normalize_symbol`
so seeded symbols fail-fast on invalid shape (lowercase, ``$``-prefix,
non-ASCII, >20 chars) rather than silently landing malformed rows that
would never match scan filters.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from ..models.stock_universe import StockUniverseIndexMembership
from .symbol_format import normalize_symbol

logger = logging.getLogger(__name__)


@dataclass
class SeedCounts:
    """Result of a seed run. Counts sum to total CSV rows processed."""

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0


def seed_from_csv(
    session: Session,
    csv_path: Path,
    *,
    index_name: str,
    as_of_date: str,
    source: str = "seed_v1",
    dry_run: bool = False,
) -> SeedCounts:
    """Upsert membership rows for ``index_name`` from ``csv_path``.

    Idempotent: re-running with the same CSV and as_of_date produces an
    all-``unchanged`` counts result. Bumping ``as_of_date`` or ``source``
    with the same symbol set produces an all-``updated`` result. Rows
    whose ``symbol`` fails :func:`normalize_symbol` (blank, malformed,
    too long) land in ``skipped`` rather than raising.

    Returns a :class:`SeedCounts` summary. Commits only when ``dry_run``
    is False; dry-run still walks the full CSV and reports what would
    change.
    """
    counts = SeedCounts()
    normalized_index = index_name.strip().upper()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "symbol" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV {csv_path} missing required 'symbol' header; "
                f"got headers={reader.fieldnames}"
            )
        for row in reader:
            symbol = normalize_symbol(row.get("symbol"))
            if symbol is None:
                counts.skipped += 1
                continue

            existing = (
                session.query(StockUniverseIndexMembership)
                .filter_by(symbol=symbol, index_name=normalized_index)
                .one_or_none()
            )
            if existing is None:
                if not dry_run:
                    session.add(
                        StockUniverseIndexMembership(
                            symbol=symbol,
                            index_name=normalized_index,
                            as_of_date=as_of_date,
                            source=source,
                        )
                    )
                counts.added += 1
            elif existing.as_of_date != as_of_date or existing.source != source:
                if not dry_run:
                    existing.as_of_date = as_of_date
                    existing.source = source
                counts.updated += 1
            else:
                counts.unchanged += 1

    if not dry_run:
        session.commit()
    logger.info(
        "index membership seed: index=%s source=%s as_of=%s "
        "added=%d updated=%d unchanged=%d skipped=%d dry_run=%s",
        normalized_index,
        source,
        as_of_date,
        counts.added,
        counts.updated,
        counts.unchanged,
        counts.skipped,
        dry_run,
    )
    return counts
