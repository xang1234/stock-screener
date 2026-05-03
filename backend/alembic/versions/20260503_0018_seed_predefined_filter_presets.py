"""Seed predefined filter presets (Minervini, CANSLIM, etc.) for live deployments.

Mirrors the predefined screens that the static site exposes via
``app.services.preset_screens.PRESET_SCREENS`` so that Docker / live deployments
ship with the same starter set out of the box. Users can rename, edit, or delete
these rows like any other ``filter_presets`` entry — Alembic only runs this
migration once, so deletions stick across restarts.

Two static-site presets are intentionally excluded:

* ``movers_9m`` — uses ``minVolume`` interpreted as dollar volume; on the live
  site ``min_volume`` is share count, so the same numeric value is meaningless.
* ``club_97`` — relies on ``pctDay`` / ``pctWeek`` / ``pctMonth`` percentile
  fields that exist only in the static-site row schema.

Definitions are embedded inline so the migration is self-contained and stable
even if ``preset_screens.py`` is later refactored.
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from alembic import op


revision = "20260503_0018"
down_revision = "20260426_0017"
branch_labels = None
depends_on = None


# Empty filter shape mirroring frontend/src/features/scan/defaultFilters.js
# `buildDefaultScanFilters()`. Stored full-shape so seeded presets behave
# identically to user-saved presets when loaded into the FilterPanel.
def _empty_filter_shape() -> dict[str, Any]:
    range_filter = {"min": None, "max": None}
    return {
        "symbolSearch": "",
        "stage": None,
        "ratings": [],
        "ibdIndustries": {"values": [], "mode": "include"},
        "gicsSectors": {"values": [], "mode": "include"},
        "minVolume": None,
        "minMarketCap": None,
        "marketCapUsd": dict(range_filter),
        "advUsd": dict(range_filter),
        "markets": [],
        "compositeScore": dict(range_filter),
        "minerviniScore": dict(range_filter),
        "canslimScore": dict(range_filter),
        "ipoScore": dict(range_filter),
        "customScore": dict(range_filter),
        "volBreakthroughScore": dict(range_filter),
        "seSetupScore": dict(range_filter),
        "seDistanceToPivot": dict(range_filter),
        "seBbSqueeze": dict(range_filter),
        "seVolumeVs50d": dict(range_filter),
        "seSetupReady": None,
        "seRsLineNewHigh": None,
        "rsRating": dict(range_filter),
        "rs1m": dict(range_filter),
        "rs3m": dict(range_filter),
        "rs12m": dict(range_filter),
        "epsRating": dict(range_filter),
        "price": dict(range_filter),
        "adrPercent": dict(range_filter),
        "epsGrowth": dict(range_filter),
        "salesGrowth": dict(range_filter),
        "vcpScore": dict(range_filter),
        "vcpPivot": dict(range_filter),
        "vcpDetected": None,
        "vcpReady": None,
        "maAlignment": None,
        "passesTemplate": None,
        "perfDay": dict(range_filter),
        "perfWeek": dict(range_filter),
        "perfMonth": dict(range_filter),
        "perf3m": dict(range_filter),
        "perf6m": dict(range_filter),
        "gapPercent": dict(range_filter),
        "volumeSurge": dict(range_filter),
        "ema10Distance": dict(range_filter),
        "ema20Distance": dict(range_filter),
        "ema50Distance": dict(range_filter),
        "week52HighDistance": dict(range_filter),
        "week52LowDistance": dict(range_filter),
        "ipoAfter": None,
        "beta": dict(range_filter),
        "betaAdjRs": dict(range_filter),
    }


# (name, description, sparse_filter_overrides, sort_by, sort_order)
SEEDED_PRESETS: list[tuple[str, str, dict[str, Any], str, str]] = [
    (
        "Minervini Trend Template",
        "Stage 2 uptrend stocks passing the 8-point trend template.",
        {
            "minerviniScore": {"min": 70, "max": None},
            "stage": 2,
            "maAlignment": True,
            "rsRating": {"min": 70, "max": None},
        },
        "minervini_score",
        "desc",
    ),
    (
        "CANSLIM",
        "William O'Neil growth screen: strong earnings, RS, and institutional demand.",
        {
            "canslimScore": {"min": 70, "max": None},
            "epsGrowth": {"min": 25, "max": None},
            "rsRating": {"min": 80, "max": None},
        },
        "canslim_score",
        "desc",
    ),
    (
        "VCP Setups",
        "Volatility contraction patterns with Minervini trend confirmation.",
        {
            "vcpDetected": True,
            "minerviniScore": {"min": 50, "max": None},
        },
        "vcp_score",
        "desc",
    ),
    (
        "Volume Breakthrough",
        "Record-volume breakouts across 1-year and 5-year lookbacks.",
        {
            "volBreakthroughScore": {"min": 33, "max": None},
        },
        "volume_breakthrough_score",
        "desc",
    ),
    (
        "Episodic Pivot",
        "Qullamaggie-style gap-ups on massive volume.",
        {
            "gapPercent": {"min": 10, "max": None},
            "volumeSurge": {"min": 2.0, "max": None},
            "rsRating": {"min": 70, "max": None},
        },
        "gap_percent",
        "desc",
    ),
    (
        "Momentum Leaders",
        "Top performers over 3-6 months in confirmed Stage 2 uptrends.",
        {
            "perf3m": {"min": 30, "max": None},
            "perf6m": {"min": 80, "max": None},
            "rsRating": {"min": 85, "max": None},
            "stage": 2,
        },
        "perf_6m",
        "desc",
    ),
    (
        "Oliver Kell Growth",
        "Growth stocks near highs with strong earnings and sales acceleration.",
        {
            "price": {"min": 20, "max": None},
            "epsGrowth": {"min": 25, "max": None},
            "salesGrowth": {"min": 15, "max": None},
            "rsRating": {"min": 80, "max": None},
            "week52HighDistance": {"min": -15, "max": None},
        },
        "composite_score",
        "desc",
    ),
    (
        "RS Power Play",
        "Elite relative strength leaders in Stage 2 uptrends.",
        {
            "rsRating": {"min": 90, "max": None},
            "rs3m": {"min": 85, "max": None},
            "stage": 2,
        },
        "rs_rating",
        "desc",
    ),
    (
        "New Highs + Volume",
        "Stocks at or near 52-week highs with above-average volume.",
        {
            "week52HighDistance": {"min": -5, "max": None},
            "volumeSurge": {"min": 1.3, "max": None},
            "rsRating": {"min": 70, "max": None},
        },
        "week_52_high_distance",
        "desc",
    ),
    (
        "Growth Rockets",
        "Triple-digit EPS and sales growth with strong relative strength.",
        {
            "epsGrowth": {"min": 40, "max": None},
            "salesGrowth": {"min": 30, "max": None},
            "rsRating": {"min": 70, "max": None},
        },
        "eps_growth_qq",
        "desc",
    ),
    (
        "Tight Setups",
        "Low-volatility bases with VCP characteristics in strong uptrends.",
        {
            "adrPercent": {"min": None, "max": 4},
            "rsRating": {"min": 80, "max": None},
            "vcpScore": {"min": 30, "max": None},
            "stage": 2,
        },
        "vcp_score",
        "desc",
    ),
    (
        "Recent IPOs",
        "High-scoring recent IPOs with strong early price action.",
        {
            "ipoScore": {"min": 50, "max": None},
        },
        "ipo_score",
        "desc",
    ),
    (
        "4% Daily Gainers",
        "Stocks advancing 4%+ intraday — quick broad-market momentum scan.",
        {
            "perfDay": {"min": 4, "max": None},
        },
        "price_change_1d",
        "desc",
    ),
    (
        "20% Weekly Movers",
        "Stocks up 20%+ over the past 5 sessions.",
        {
            "perfWeek": {"min": 20, "max": None},
        },
        "perf_week",
        "desc",
    ),
]


def _filter_presets_table() -> sa.Table:
    return sa.table(
        "filter_presets",
        sa.column("id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("description", sa.Text),
        sa.column("filters", sa.Text),
        sa.column("sort_by", sa.String),
        sa.column("sort_order", sa.String),
        sa.column("position", sa.Integer),
    )


def _build_filters_payload(overrides: dict[str, Any]) -> str:
    payload = _empty_filter_shape()
    payload.update(overrides)
    return json.dumps(payload)


def upgrade() -> None:
    bind = op.get_bind()
    table = _filter_presets_table()

    existing_names = {
        row[0]
        for row in bind.execute(sa.select(table.c.name)).fetchall()
    }
    next_position = bind.execute(
        sa.select(sa.func.coalesce(sa.func.max(table.c.position), -1))
    ).scalar_one() + 1

    rows_to_insert: list[dict[str, Any]] = []
    for name, description, overrides, sort_by, sort_order in SEEDED_PRESETS:
        if name in existing_names:
            continue
        rows_to_insert.append(
            {
                "name": name,
                "description": description,
                "filters": _build_filters_payload(overrides),
                "sort_by": sort_by,
                "sort_order": sort_order,
                "position": next_position,
            }
        )
        next_position += 1

    if rows_to_insert:
        op.bulk_insert(table, rows_to_insert)


def downgrade() -> None:
    table = _filter_presets_table()
    seeded_names = [name for name, *_ in SEEDED_PRESETS]
    op.execute(
        table.delete().where(table.c.name.in_(seeded_names))
    )
