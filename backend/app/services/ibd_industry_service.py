"""Service for loading and managing IBD Industry Group data.

The ``market`` parameter on the read helpers routes non-US markets through
``MarketTaxonomyService``, which loads per-market classification CSVs at
runtime. US continues to read from the ``ibd_industry_groups`` DB table that
``load_from_csv`` populates, so legacy callers are unaffected.
"""
import csv
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import delete
from ..config import settings
from ..config.settings import get_project_root
from ..models.industry import IBDIndustryGroup

logger = logging.getLogger(__name__)


def _market_taxonomy_service():
    """Return the process-wide taxonomy service.

    A second instance here would retry on its own deadline and parse every CSV
    a second time, so both the retry state and the loaded data must be shared.
    """
    from .market_taxonomy_service import get_market_taxonomy_service

    return get_market_taxonomy_service()


def _market_has_curated_taxonomy(market: str) -> bool:
    """True when ``market`` ships a committed classification taxonomy (HK/JP/TW/IN).

    CA/DE/SG/MY register empty taxonomy buckets, so they fall through to the
    classifier-populated ``ibd_industry_groups`` table instead.
    """
    try:
        return bool(_market_taxonomy_service().groups_for_market(market))
    except Exception:  # noqa: BLE001 — fail toward the DB path
        return False


def _classifier_group_memberships(db: Session, market: str) -> dict[str, list[str]]:
    rows = (
        db.query(
            IBDIndustryGroup.industry_group,
            IBDIndustryGroup.symbol,
        )
        .filter(IBDIndustryGroup.market == market)
        .order_by(
            IBDIndustryGroup.industry_group,
            IBDIndustryGroup.symbol,
        )
        .all()
    )
    memberships: dict[str, list[str]] = {}
    for industry_group, symbol in rows:
        memberships.setdefault(industry_group, []).append(symbol)
    return memberships


def _classifier_symbols_for_group(
    db: Session,
    *,
    market: str,
    industry_group: str,
) -> list[str]:
    rows = (
        db.query(IBDIndustryGroup.symbol)
        .filter(
            IBDIndustryGroup.market == market,
            IBDIndustryGroup.industry_group == industry_group,
        )
        .order_by(IBDIndustryGroup.symbol)
        .all()
    )
    return list(dict.fromkeys(row.symbol for row in rows))


def _merge_classifier_gap_memberships(
    db: Session,
    *,
    market: str,
    taxonomy_memberships: dict[str, list[str]],
) -> dict[str, list[str]]:
    memberships = {
        group: list(dict.fromkeys(symbols))
        for group, symbols in taxonomy_memberships.items()
    }
    taxonomy_symbols = {
        symbol
        for symbols in memberships.values()
        for symbol in symbols
    }
    for group, symbols in _classifier_group_memberships(db, market).items():
        for symbol in symbols:
            if symbol in taxonomy_symbols:
                continue
            memberships.setdefault(group, []).append(symbol)
    return memberships


def _merge_classifier_gap_symbols(
    db: Session,
    *,
    market: str,
    industry_group: str,
    taxonomy_symbols: list[str],
    taxonomy_service,
) -> list[str]:
    symbols = list(dict.fromkeys(taxonomy_symbols))
    for symbol in _classifier_symbols_for_group(
        db,
        market=market,
        industry_group=industry_group,
    ):
        entry = taxonomy_service.get(symbol, market=market)
        if entry is not None and entry.industry_group:
            continue
        symbols.append(symbol)
    return list(dict.fromkeys(symbols))


class IBDIndustryService:
    """Manage IBD Industry Group data"""

    @staticmethod
    def resolve_tracked_csv_path(csv_path: str | Path | None = None) -> Path:
        """Resolve the tracked IBD CSV, falling back to the repo copy when needed."""
        normalized_csv_path = csv_path
        if isinstance(normalized_csv_path, str) and not normalized_csv_path.strip():
            normalized_csv_path = None

        settings_path = Path(settings.ibd_industry_csv_path)
        configured_path = Path(normalized_csv_path) if normalized_csv_path is not None else settings_path
        if configured_path.exists():
            return configured_path

        fallback_path = get_project_root() / "data" / "IBD_industry_group.csv"
        should_fallback = normalized_csv_path is None or configured_path == settings_path
        if should_fallback and fallback_path.exists() and fallback_path != configured_path:
            logger.warning(
                "Configured IBD industry CSV path %s is missing; falling back to tracked CSV at %s",
                configured_path,
                fallback_path,
            )
            return fallback_path

        if normalized_csv_path is not None:
            raise FileNotFoundError(f"IBD industry CSV file not found at {configured_path}")

        return configured_path

    # Provenance values that are authoritative and must not be clobbered by the
    # hybrid classifier or its bundle import.
    AUTHORITATIVE_SOURCES = ("csv", "manual")

    @staticmethod
    def load_from_csv(db: Session, csv_path: str = None) -> int:
        """
        Load IBD industry groups from the curated CSV (US, ``source='csv'``).

        The CSV is the authoritative seed layer. To preserve classifier-derived
        rows (``source in {crosswalk, embedding, llm}``) and human overrides
        (``source='manual'``) across reloads, this only clears ``csv`` rows rather
        than the whole table, and drops classifier rows for any symbol the CSV now
        claims (CSV wins) while leaving ``manual`` rows untouched.

        Args:
            db: Database session
            csv_path: Path to CSV file (defaults to data/IBD_industry_group.csv)

        Returns:
            Number of records loaded
        """
        csv_path = IBDIndustryService.resolve_tracked_csv_path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"IBD industry CSV file not found at {csv_path}")

        logger.info(f"Loading IBD industry groups from {csv_path}")

        # Parse the full CSV up front so we know which symbols the curated file
        # claims before we mutate the table.
        parsed: dict[str, str] = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) != 2:
                    continue
                symbol, industry_group = row
                symbol = symbol.strip().upper()
                industry_group = industry_group.strip()
                if not symbol or not industry_group:
                    continue
                parsed[symbol] = industry_group  # last write wins on dup symbol

        csv_symbols = list(parsed.keys())

        try:
            # 1. Clear previously-loaded CSV rows (not classifier/manual rows).
            db.execute(
                delete(IBDIndustryGroup).where(IBDIndustryGroup.source == "csv")
            )

            # 2. Drop classifier rows for symbols the CSV now claims (CSV is
            #    authoritative); keep human ``manual`` rows. Chunked to respect
            #    SQLite's bound-parameter limit.
            for start in range(0, len(csv_symbols), 500):
                chunk = csv_symbols[start:start + 500]
                db.execute(
                    delete(IBDIndustryGroup).where(
                        IBDIndustryGroup.symbol.in_(chunk),
                        IBDIndustryGroup.source.notin_(
                            IBDIndustryService.AUTHORITATIVE_SOURCES
                        ),
                    )
                )
            # No intermediate commit: the whole delete-and-reload runs in one
            # transaction so a failure rolls back to the prior consistent state
            # rather than leaving the table partially emptied. The protected
            # query below still sees the deletes via the session's autoflush.
            logger.info("Cleared existing CSV-sourced IBD industry data")

            # 3. Symbols still present are ``manual`` overrides — never replace them.
            protected: set[str] = set()
            for start in range(0, len(csv_symbols), 500):
                chunk = csv_symbols[start:start + 500]
                rows = db.query(IBDIndustryGroup.symbol).filter(
                    IBDIndustryGroup.symbol.in_(chunk)
                ).all()
                protected.update(r.symbol for r in rows)
            if protected:
                logger.info(
                    "Preserving %d manual IBD overrides over CSV values", len(protected)
                )

            loaded = 0
            batch = []
            for symbol, industry_group in parsed.items():
                if symbol in protected:
                    continue
                batch.append({
                    'symbol': symbol,
                    'industry_group': industry_group,
                    'market': 'US',
                    'source': 'csv',
                })
                if len(batch) >= 500:
                    # bulk_insert_mappings emits the INSERT immediately within the
                    # open transaction; the single commit below makes the whole
                    # delete+reload atomic.
                    db.bulk_insert_mappings(IBDIndustryGroup, batch)
                    loaded += len(batch)
                    logger.info(f"Loaded {loaded} IBD industry mappings...")
                    batch = []

            if batch:
                db.bulk_insert_mappings(IBDIndustryGroup, batch)
                loaded += len(batch)

            db.commit()  # single commit → delete+reload is atomic
            logger.info(f"Successfully loaded {loaded} IBD industry group mappings")
            return loaded

        except Exception as e:
            logger.error(f"Error loading IBD industry data: {e}", exc_info=True)
            db.rollback()
            raise

    @staticmethod
    def get_industry_group(db: Session, symbol: str) -> str:
        """
        Get IBD industry group for a symbol.

        Args:
            db: Database session
            symbol: Stock symbol

        Returns:
            Industry group name or None if not found
        """
        try:
            record = db.query(IBDIndustryGroup).filter(
                IBDIndustryGroup.symbol == symbol.upper()
            ).first()
            return record.industry_group if record else None
        except Exception as e:
            logger.error(f"Error getting industry group for {symbol}: {e}")
            return None

    @staticmethod
    def get_bulk_industry_groups(db: Session, symbols: list) -> dict:
        """
        Bulk fetch IBD industry groups for multiple symbols (performance optimization).

        Args:
            db: Database session
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol -> industry group name
        """
        if not symbols:
            return {}

        try:
            # Normalize symbols to uppercase
            symbols_upper = [s.upper() for s in symbols]

            # Single bulk query instead of N individual queries
            records = db.query(IBDIndustryGroup).filter(
                IBDIndustryGroup.symbol.in_(symbols_upper)
            ).all()

            # Return as dict for O(1) lookup
            return {record.symbol: record.industry_group for record in records}

        except Exception as e:
            logger.error(f"Error bulk fetching industry groups: {e}")
            return {}

    @staticmethod
    def get_group_symbols(db: Session, industry_group: str, *, market: str | None = None) -> list:
        """Get all symbols in an industry group.

        Markets with a committed taxonomy use it as the authoritative first
        layer, then fill missing symbols from the classifier-populated
        ``ibd_industry_groups`` table. Taxonomy-less markets read the
        classifier table directly. All DB reads are filtered by ``market`` so
        a group's membership never leaks symbols across markets.
        """
        normalized = (market or "US").upper()
        try:
            if normalized != "US" and _market_has_curated_taxonomy(normalized):
                taxonomy_service = _market_taxonomy_service()
                taxonomy_symbols = taxonomy_service.symbols_for_group(
                    normalized,
                    industry_group,
                )
                return _merge_classifier_gap_symbols(
                    db,
                    market=normalized,
                    industry_group=industry_group,
                    taxonomy_symbols=taxonomy_symbols,
                    taxonomy_service=taxonomy_service,
                )
            return _classifier_symbols_for_group(
                db,
                market=normalized,
                industry_group=industry_group,
            )
        except Exception as e:
            logger.error(
                "Error getting symbols for group %s in market %s: %s",
                industry_group,
                normalized,
                e,
            )
            return []

    @staticmethod
    def get_group_memberships(
        db: Session,
        *,
        market: str | None = None,
    ) -> dict[str, list[str]]:
        """Load all symbol memberships for one market without per-group queries."""
        normalized = (market or "US").upper()
        if normalized != "US" and _market_has_curated_taxonomy(normalized):
            try:
                taxonomy_memberships = _market_taxonomy_service().group_symbols_for_market(
                    normalized
                )
                return _merge_classifier_gap_memberships(
                    db,
                    market=normalized,
                    taxonomy_memberships=taxonomy_memberships,
                )
            except Exception as exc:
                logger.error(
                    "Error loading bulk group memberships for market %s: %s",
                    normalized,
                    exc,
                    exc_info=True,
                )
                raise
        return _classifier_group_memberships(db, normalized)

    @staticmethod
    def get_all_groups(db: Session, *, market: str | None = None) -> list:
        """Get list of all unique industry groups for a market.

        Markets with a committed taxonomy use taxonomy groups first, then add
        classifier-only groups for symbols absent from that taxonomy. US and
        taxonomy-less markets read the classifier-populated
        ``ibd_industry_groups`` table, scoped to ``market``.
        """
        normalized = (market or "US").upper()
        try:
            return sorted(
                IBDIndustryService.get_group_memberships(
                    db,
                    market=normalized,
                )
            )
        except Exception as e:
            logger.error(f"Error getting all industry groups: {e}", exc_info=True)
            raise
