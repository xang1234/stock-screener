"""
Universe resolver — centralized symbol resolution for scan universes.

Replaces inline symbol resolution in create_scan() with a single service
that maps UniverseDefinition → list of stock symbols. Used by the API,
Celery tasks, and any future scan triggers.
"""
import logging
from typing import Any, List, Optional

from sqlalchemy.orm import Session

from ..schemas.universe import UniverseDefinition, UniverseType
from ..services.stock_universe_service import stock_universe_service

logger = logging.getLogger(__name__)


def normalize_universe_definition(universe_def: Any) -> UniverseDefinition:
    """Coerce legacy or serialized universe payloads to UniverseDefinition."""
    if isinstance(universe_def, UniverseDefinition):
        return universe_def

    if isinstance(universe_def, str):
        legacy = universe_def.strip()
        if legacy.lower() == "active":
            return UniverseDefinition(type=UniverseType.ALL)
        return UniverseDefinition.from_legacy(legacy)

    if isinstance(universe_def, dict):
        if "type" in universe_def:
            return UniverseDefinition.model_validate(universe_def)

        for key in ("name", "universe", "value"):
            legacy = universe_def.get(key)
            if isinstance(legacy, str):
                return normalize_universe_definition(legacy)

        raise ValueError(
            "Unsupported universe definition dict; expected {'type': ...} "
            "or a legacy name field like {'name': 'active'}"
        )

    raise ValueError(
        f"Unsupported universe definition type: {type(universe_def).__name__}"
    )


def resolve_symbols(
    db: Session,
    universe_def: Any,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Resolve a UniverseDefinition to a list of stock symbols.

    Args:
        db: Database session
        universe_def: Typed or normalized universe definition
        limit: Optional max number of symbols to return

    Returns:
        List of uppercase stock symbol strings
    """
    universe_def = normalize_universe_definition(universe_def)
    t = universe_def.type

    if t == UniverseType.ALL:
        return stock_universe_service.get_active_symbols(
            db, exchange=None, sp500_only=False, limit=limit
        )

    elif t == UniverseType.EXCHANGE:
        return stock_universe_service.get_active_symbols(
            db, exchange=universe_def.exchange.value, sp500_only=False, limit=limit
        )

    elif t == UniverseType.INDEX:
        return stock_universe_service.get_active_symbols(
            db, exchange=None, sp500_only=True, limit=limit
        )

    elif t in (UniverseType.CUSTOM, UniverseType.TEST):
        symbols = universe_def.symbols
        if not universe_def.allow_inactive_symbols:
            filtered = stock_universe_service.filter_active_symbols(db, symbols)
            filtered_set = set(filtered)
            dropped = [symbol for symbol in symbols if symbol not in filtered_set]
            if dropped:
                logger.warning(
                    "Dropped %d inactive or unknown symbols from %s universe: %s",
                    len(dropped),
                    t.value,
                    ", ".join(dropped[:10]),
                )
            symbols = filtered
        if limit is not None:
            return symbols[:limit]
        return symbols

    else:
        raise ValueError(f"Unknown universe type: {t}")


def resolve_count(
    db: Session,
    universe_def: Any,
) -> int:
    """
    Get the count of symbols in a universe without fetching the full list.

    For CUSTOM/TEST, this is cheap (len of symbols list).
    For ALL/EXCHANGE/INDEX, this queries the DB.

    Args:
        db: Database session
        universe_def: Typed or normalized universe definition

    Returns:
        Number of symbols in the universe
    """
    universe_def = normalize_universe_definition(universe_def)
    if universe_def.type in (UniverseType.CUSTOM, UniverseType.TEST):
        return len(resolve_symbols(db, universe_def))
    return len(resolve_symbols(db, universe_def))
