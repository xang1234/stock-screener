"""Sector Phase 2 participation rates."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def get_sector_phase_health(db: Session) -> Dict[str, Dict]:
    try:
        from ..models.stock import StockTechnical, StockIndustry
        rows = (
            db.query(StockTechnical.stage, StockIndustry.sector)
            .join(StockIndustry, StockTechnical.symbol == StockIndustry.symbol)
            .all()
        )
    except Exception as e:
        logger.warning("sector_phase_health query failed: %s", e)
        return {}

    counts: Dict[str, Dict] = defaultdict(lambda: {"phase2": 0, "total": 0})
    for stage, sector in rows:
        if sector:
            counts[sector]["total"] += 1
            if stage == 2:
                counts[sector]["phase2"] += 1

    return {
        s: {
            "pct_phase2": round(c["phase2"] / c["total"], 3) if c["total"] > 0 else 0,
            "phase2_count": c["phase2"],
            "total_count": c["total"],
            "is_healthy": (c["phase2"] / c["total"]) >= 0.20 if c["total"] > 0 else False,
        }
        for s, c in counts.items()
    }
