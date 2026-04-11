"""Default screening presets for the static site.

Each preset defines a named filter+sort combination inspired by popular
stock screening methodologies.  The backend uses these at export time to
determine which symbols need chart data.  The frontend reads the serialized
preset definitions from the scan manifest and applies them client-side.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
# ``filters`` uses *backend row field names* (snake_case) for server-side
# application in the static-site export pipeline.
#
# ``ui_filters`` uses *frontend filter key names* (camelCase) so the
# manifest payload can be applied directly via ``setFilters``.
# ---------------------------------------------------------------------------

def _ipo_cutoff_iso() -> str:
    """Return ISO date string for 2 years ago (IPO Leaders preset)."""
    return (date.today() - timedelta(days=730)).isoformat()


def _build_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": "minervini",
            "name": "Minervini Template",
            "description": (
                "Mark Minervini's trend template: Stage 2 uptrend, "
                "moving average alignment, RS Rating > 70, passes template checks."
            ),
            "filters": {
                "passes_template": True,
                "stage": 2,
                "ma_alignment": True,
                "rs_rating": {"min": 70},
            },
            "ui_filters": {
                "passesTemplate": True,
                "stage": 2,
                "maAlignment": True,
                "rsRating": {"min": 70, "max": None},
            },
            "sort_field": "minervini_score",
            "sort_order": "desc",
        },
        {
            "id": "canslim",
            "name": "CANSLIM Leaders",
            "description": (
                "William O'Neil's CAN SLIM: current quarterly EPS growth > 25%, "
                "strong relative strength, Stage 2 uptrend."
            ),
            "filters": {
                "eps_growth_qq": {"min": 25},
                "rs_rating": {"min": 70},
                "stage": 2,
            },
            "ui_filters": {
                "epsGrowth": {"min": 25, "max": None},
                "rsRating": {"min": 70, "max": None},
                "stage": 2,
            },
            "sort_field": "canslim_score",
            "sort_order": "desc",
        },
        {
            "id": "momentum_leaders",
            "name": "Momentum Leaders",
            "description": (
                "Pure relative-strength momentum screen: RS Rating > 90, "
                "3-month RS > 80, Stage 2 uptrend."
            ),
            "filters": {
                "rs_rating": {"min": 90},
                "rs_rating_3m": {"min": 80},
                "stage": 2,
            },
            "ui_filters": {
                "rsRating": {"min": 90, "max": None},
                "rs3m": {"min": 80, "max": None},
                "stage": 2,
            },
            "sort_field": "rs_rating",
            "sort_order": "desc",
        },
        {
            "id": "volume_breakout",
            "name": "Volume Breakouts",
            "description": (
                "Dan Zanger-style volume surge breakouts: volume surge > 1.5x, "
                "RS Rating > 70, Stage 2 uptrend."
            ),
            "filters": {
                "volume_surge": {"min": 1.5},
                "rs_rating": {"min": 70},
                "stage": 2,
            },
            "ui_filters": {
                "volumeSurge": {"min": 1.5, "max": None},
                "rsRating": {"min": 70, "max": None},
                "stage": 2,
            },
            "sort_field": "volume_breakthrough_score",
            "sort_order": "desc",
        },
        {
            "id": "ep_gapper",
            "name": "Episodic Pivots",
            "description": (
                "Qullamaggie-style episodic pivots: gap-up > 4% with "
                "volume surge > 2x and RS Rating > 60."
            ),
            "filters": {
                "gap_percent": {"min": 4},
                "volume_surge": {"min": 2},
                "rs_rating": {"min": 60},
            },
            "ui_filters": {
                "gapPercent": {"min": 4, "max": None},
                "volumeSurge": {"min": 2, "max": None},
                "rsRating": {"min": 60, "max": None},
            },
            "sort_field": "gap_percent",
            "sort_order": "desc",
        },
        {
            "id": "tight_setups",
            "name": "Tight Setups",
            "description": (
                "Oliver Kell-style tight bases near pivot: Setup Engine ready, "
                "RS Rating > 80, Stage 2, MA alignment."
            ),
            "filters": {
                "se_setup_ready": True,
                "rs_rating": {"min": 80},
                "stage": 2,
                "ma_alignment": True,
            },
            "ui_filters": {
                "seSetupReady": True,
                "rsRating": {"min": 80, "max": None},
                "stage": 2,
                "maAlignment": True,
            },
            "sort_field": "se_setup_score",
            "sort_order": "desc",
        },
        {
            "id": "vcp",
            "name": "VCP Setups",
            "description": (
                "Volatility Contraction Pattern: VCP detected, "
                "Stage 2 uptrend, RS Rating > 70."
            ),
            "filters": {
                "vcp_detected": True,
                "stage": 2,
                "rs_rating": {"min": 70},
            },
            "ui_filters": {
                "vcpDetected": True,
                "stage": 2,
                "rsRating": {"min": 70, "max": None},
            },
            "sort_field": "vcp_score",
            "sort_order": "desc",
        },
        {
            "id": "ipo_leaders",
            "name": "IPO Leaders",
            "description": (
                "Recent IPOs (last 2 years) with strong relative strength "
                "(RS Rating > 60)."
            ),
            "filters": {
                "ipo_date": {"after": _ipo_cutoff_iso()},
                "rs_rating": {"min": 60},
            },
            "ui_filters": {
                "ipoAfter": _ipo_cutoff_iso(),
                "rsRating": {"min": 60, "max": None},
            },
            "sort_field": "ipo_score",
            "sort_order": "desc",
        },
        {
            "id": "near_highs",
            "name": "Near 52-Week High",
            "description": (
                "Stocks within 5% of their 52-week high with "
                "RS Rating > 70 and Stage 2 uptrend."
            ),
            "filters": {
                "week_52_high_distance": {"min": -5},
                "rs_rating": {"min": 70},
                "stage": 2,
            },
            "ui_filters": {
                "week52HighDistance": {"min": -5, "max": None},
                "rsRating": {"min": 70, "max": None},
                "stage": 2,
            },
            "sort_field": "composite_score",
            "sort_order": "desc",
        },
        {
            "id": "rs_line_new_high",
            "name": "RS Line New High",
            "description": (
                "Stocks with relative strength line at a new high "
                "(O'Neil-style confirmation) and RS Rating > 70."
            ),
            "filters": {
                "se_rs_line_new_high": True,
                "rs_rating": {"min": 70},
            },
            "ui_filters": {
                "seRsLineNewHigh": True,
                "rsRating": {"min": 70, "max": None},
            },
            "sort_field": "rs_rating",
            "sort_order": "desc",
        },
    ]


SCREENING_PRESETS: list[dict[str, Any]] = _build_presets()


# ---------------------------------------------------------------------------
# Row filtering helpers (operate on serialized row dicts)
# ---------------------------------------------------------------------------

def _row_matches_range(value: Any, spec: dict[str, Any]) -> bool:
    """Check if a numeric value satisfies a {min, max} range spec."""
    if value is None:
        return False
    if spec.get("min") is not None and value < spec["min"]:
        return False
    if spec.get("max") is not None and value > spec["max"]:
        return False
    return True


def _row_passes_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True if *row* passes all *filters*."""
    for key, spec in filters.items():
        # IPO date filter (special case)
        if key == "ipo_date":
            after = spec.get("after") if isinstance(spec, dict) else None
            if after and (not row.get("ipo_date") or row["ipo_date"] < after):
                return False
            continue

        value = row.get(key)

        # Range filter ({min, max} dict)
        if isinstance(spec, dict):
            if not _row_matches_range(value, spec):
                return False
        # Boolean filter
        elif isinstance(spec, bool):
            if bool(value) != spec:
                return False
        # Equality filter (e.g. stage == 2)
        else:
            if value != spec:
                return False

    return True


def apply_preset_to_rows(
    rows: list[dict[str, Any]],
    preset: dict[str, Any],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Filter and sort *rows* according to *preset*, returning the top *limit*."""
    filtered = [row for row in rows if _row_passes_filters(row, preset["filters"])]

    sort_field = preset["sort_field"]
    descending = preset.get("sort_order", "desc") == "desc"

    filtered.sort(
        key=lambda r: (r.get(sort_field) is not None, r.get(sort_field) or 0),
        reverse=descending,
    )

    return filtered[:limit]


def collect_preset_chart_symbols(
    rows: list[dict[str, Any]],
    presets: list[dict[str, Any]] | None = None,
    *,
    per_preset_limit: int = 50,
) -> set[str]:
    """Return the union of top-N symbols across all presets."""
    if presets is None:
        presets = SCREENING_PRESETS

    symbols: set[str] = set()
    for preset in presets:
        top_rows = apply_preset_to_rows(rows, preset, limit=per_preset_limit)
        symbols.update(row["symbol"] for row in top_rows if row.get("symbol"))
    return symbols


def serialize_presets_for_manifest(
    rows: list[dict[str, Any]],
    presets: list[dict[str, Any]] | None = None,
    *,
    per_preset_limit: int = 50,
) -> list[dict[str, Any]]:
    """Build the preset metadata block for the scan manifest.

    Each entry includes the preset definition (with ``ui_filters`` for
    frontend consumption) and the ordered list of top symbols for that preset.
    """
    if presets is None:
        presets = SCREENING_PRESETS

    result: list[dict[str, Any]] = []
    for preset in presets:
        top_rows = apply_preset_to_rows(rows, preset, limit=per_preset_limit)
        result.append({
            "id": preset["id"],
            "name": preset["name"],
            "description": preset["description"],
            "filters": preset["ui_filters"],
            "sort": {
                "field": preset["sort_field"],
                "order": preset.get("sort_order", "desc"),
            },
            "top_symbols": [row["symbol"] for row in top_rows if row.get("symbol")],
        })
    return result
