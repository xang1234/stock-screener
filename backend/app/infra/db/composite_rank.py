"""Best-first composite ORDER BY: the SQL form of ``scoring.is_core_scored``."""

from __future__ import annotations

from sqlalchemy import case, or_


def core_scored_first(model):
    """0 for rows Minervini or CANSLIM scored, 1 for IPO-only and unscored rows."""
    return case(
        (or_(model.minervini_score.is_not(None), model.canslim_score.is_not(None)), 0),
        else_=1,
    )


def composite_rank_order(model) -> tuple:
    """Best-first composite ORDER BY clauses for scan-result or feature rows."""
    return core_scored_first(model), model.composite_score.desc().nullslast()
