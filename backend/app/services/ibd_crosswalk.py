"""Data-derived GICS/sector → IBD industry-group crosswalk.

The curated ``data/IBD_industry_group.csv`` gives ``symbol → IBD group`` for ~10k
US names. Those same symbols carry GICS classification (``stock_industry``) and
source sector/industry (``stock_universe``). By majority-voting the IBD group
within each GICS sub-industry (and, as a fallback, each ``sector||industry`` and
each ``sector``), we learn a deterministic mapping that generalises for free to
any stock — new US names and foreign markets — that shares those attributes.

The vote logic here is pure and unit-tested; ``scripts/build_ibd_crosswalk.py``
feeds it from the database and writes ``data/ibd_crosswalk.json``. At runtime the
classifier loads that JSON via :class:`IBDCrosswalk` and consults it as the first,
free tier of the classification cascade.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CROSSWALK_SCHEMA_VERSION = 1

# Keys for the three lookup tiers, most specific first.
TIER_SUBINDUSTRY = "gics_subindustry"
TIER_SECTOR_INDUSTRY = "sector_industry"
TIER_SECTOR = "sector"

_SECTOR_INDUSTRY_SEP = "||"


def _norm(value: Optional[str]) -> str:
    return (value or "").strip()


def _pick_majority(votes: dict[str, int]) -> tuple[str, int, int]:
    """Return (group, group_votes, total_votes) for the winning group.

    Ties broken by highest votes then lexicographically smallest group name, so
    the crosswalk is fully deterministic regardless of insertion order.
    """
    total = sum(votes.values())
    group = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return group, votes[group], total


def build_crosswalk(
    *,
    symbol_to_group: dict[str, str],
    symbol_to_subindustry: dict[str, str] | None = None,
    symbol_to_sector_industry: dict[str, tuple[str, str]] | None = None,
    generated_at: str | None = None,
) -> dict:
    """Build the crosswalk dict from ground-truth symbol→group labels.

    Args:
        symbol_to_group: authoritative IBD group per symbol (from the curated CSV).
        symbol_to_subindustry: GICS sub-industry per symbol (``stock_industry``).
        symbol_to_sector_industry: ``(sector, industry)`` per symbol (``stock_universe``).
        generated_at: ISO timestamp stamped into the artifact (caller-supplied so
            the build stays deterministic / replayable).
    """
    symbol_to_subindustry = symbol_to_subindustry or {}
    symbol_to_sector_industry = symbol_to_sector_industry or {}

    sub_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sec_ind_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sec_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for symbol, group in symbol_to_group.items():
        group = _norm(group)
        if not group:
            continue

        sub = _norm(symbol_to_subindustry.get(symbol))
        if sub:
            sub_votes[sub][group] += 1

        sector, industry = symbol_to_sector_industry.get(symbol, ("", ""))
        sector, industry = _norm(sector), _norm(industry)
        if sector and industry:
            sec_ind_votes[f"{sector}{_SECTOR_INDUSTRY_SEP}{industry}"][group] += 1
        if sector:
            sec_votes[sector][group] += 1

    def _resolve(table: dict[str, dict[str, int]]) -> dict[str, dict]:
        resolved: dict[str, dict] = {}
        for key in sorted(table):
            group, votes, total = _pick_majority(table[key])
            resolved[key] = {
                "group": group,
                "votes": votes,
                "total": total,
                "share": round(votes / total, 4) if total else 0.0,
            }
        return resolved

    return {
        "schema_version": CROSSWALK_SCHEMA_VERSION,
        "generated_at": generated_at,
        TIER_SUBINDUSTRY: _resolve(sub_votes),
        TIER_SECTOR_INDUSTRY: _resolve(sec_ind_votes),
        TIER_SECTOR: _resolve(sec_votes),
    }


@dataclass
class CrosswalkHit:
    group: str
    confidence: float
    method: str  # one of TIER_* constants


@dataclass
class CrosswalkResolution:
    """Both readings of one lookup, from a single pass over the tiers:

    - ``strict``: the most-specific tier clearing the confidence thresholds (a
      high-confidence deterministic match), or ``None``.
    - ``plurality``: the most-specific tier with *any* votes (the best-effort
      guess used as a free fallback for foreign markets), or ``None``.
    """
    strict: Optional[CrosswalkHit]
    plurality: Optional[CrosswalkHit]


class IBDCrosswalk:
    """Loads a crosswalk artifact and resolves stocks to IBD groups."""

    def __init__(self, data: dict):
        self._sub = data.get(TIER_SUBINDUSTRY, {})
        self._sec_ind = data.get(TIER_SECTOR_INDUSTRY, {})
        self._sec = data.get(TIER_SECTOR, {})

    @classmethod
    def load(cls, path: str | Path) -> "IBDCrosswalk":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def resolve(
        self,
        *,
        sub_industry: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        min_share: float = 0.6,
        min_votes: int = 3,
    ) -> CrosswalkResolution:
        """Resolve a stock in one most-specific-first pass over the tiers.

        Returns both the strict hit (first tier clearing ``min_share``/``min_votes``)
        and the plurality hit (first tier with any votes), so the caller gets the
        confident match and the best-effort fallback without walking the tiers twice.
        """
        sub = _norm(sub_industry)
        sector = _norm(sector)
        industry = _norm(industry)

        candidates: list[tuple[dict, str, str]] = []
        if sub:
            candidates.append((self._sub, sub, TIER_SUBINDUSTRY))
        if sector and industry:
            candidates.append(
                (self._sec_ind, f"{sector}{_SECTOR_INDUSTRY_SEP}{industry}", TIER_SECTOR_INDUSTRY)
            )
        if sector:
            candidates.append((self._sec, sector, TIER_SECTOR))

        strict: Optional[CrosswalkHit] = None
        plurality: Optional[CrosswalkHit] = None
        for table, key, method in candidates:
            entry = table.get(key)
            if not entry or entry["votes"] < 1:
                continue
            hit = CrosswalkHit(group=entry["group"], confidence=entry["share"], method=method)
            if plurality is None:
                plurality = hit
            if strict is None and entry["share"] >= min_share and entry["votes"] >= min_votes:
                strict = hit
        return CrosswalkResolution(strict=strict, plurality=plurality)

    def lookup(
        self,
        *,
        sub_industry: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        min_share: float = 0.6,
        min_votes: int = 3,
    ) -> Optional[CrosswalkHit]:
        """The strict deterministic match (first tier clearing the thresholds), or
        ``None``. Thin accessor over :meth:`resolve` for callers that only want the
        confident hit."""
        return self.resolve(
            sub_industry=sub_industry, sector=sector, industry=industry,
            min_share=min_share, min_votes=min_votes,
        ).strict
