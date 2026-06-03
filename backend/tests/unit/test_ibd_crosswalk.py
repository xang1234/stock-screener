"""Unit tests for the data-derived GICS/sector → IBD crosswalk (pure vote logic)."""
from app.services.ibd_crosswalk import (
    TIER_SECTOR,
    TIER_SECTOR_INDUSTRY,
    TIER_SUBINDUSTRY,
    IBDCrosswalk,
    build_crosswalk,
)


def _build(**kwargs):
    return build_crosswalk(**kwargs)


def test_majority_vote_per_subindustry():
    cw = _build(
        symbol_to_group={"A": "Computers-Large", "B": "Computers-Large", "C": "Computers-Small"},
        symbol_to_subindustry={"A": "Hardware", "B": "Hardware", "C": "Hardware"},
    )
    entry = cw[TIER_SUBINDUSTRY]["Hardware"]
    assert entry["group"] == "Computers-Large"
    assert entry["votes"] == 2
    assert entry["total"] == 3
    assert entry["share"] == round(2 / 3, 4)


def test_tie_break_is_deterministic_alphabetical():
    # 1 vote each → tie broken by lexicographically smallest group name.
    cw = _build(
        symbol_to_group={"A": "Zeta-Group", "B": "Alpha-Group"},
        symbol_to_subindustry={"A": "X", "B": "X"},
    )
    assert cw[TIER_SUBINDUSTRY]["X"]["group"] == "Alpha-Group"


def test_lookup_prefers_subindustry_then_sector_industry_then_sector():
    cw = IBDCrosswalk(_build(
        symbol_to_group={f"S{i}": "Specific-Group" for i in range(5)},
        symbol_to_subindustry={f"S{i}": "SubX" for i in range(5)},
        symbol_to_sector_industry={f"S{i}": ("Tech", "Software") for i in range(5)},
    ))
    # Sub-industry tier wins when present.
    hit = cw.lookup(sub_industry="SubX", sector="Tech", industry="Software")
    assert hit is not None
    assert hit.method == TIER_SUBINDUSTRY
    assert hit.group == "Specific-Group"

    # Falls back to sector+industry when sub-industry unknown.
    hit2 = cw.lookup(sub_industry="Unknown", sector="Tech", industry="Software")
    assert hit2.method == TIER_SECTOR_INDUSTRY

    # Falls back to sector alone when industry unknown.
    hit3 = cw.lookup(sector="Tech")
    assert hit3.method == TIER_SECTOR


def test_lookup_returns_none_below_thresholds():
    cw = IBDCrosswalk(_build(
        symbol_to_group={"A": "G1", "B": "G2", "C": "G3"},  # 1 vote each, share 1/3
        symbol_to_subindustry={"A": "Mixed", "B": "Mixed", "C": "Mixed"},
    ))
    # share 0.33 < default 0.6 → no confident hit.
    assert cw.lookup(sub_industry="Mixed") is None
    # Even with low share threshold, min_votes=3 not met for the winning group (1).
    assert cw.lookup(sub_industry="Mixed", min_share=0.1, min_votes=3) is None
    # Relaxing both yields the deterministic winner.
    hit = cw.lookup(sub_industry="Mixed", min_share=0.1, min_votes=1)
    assert hit is not None and hit.group == "G1"


def test_unknown_key_returns_none():
    cw = IBDCrosswalk(_build(symbol_to_group={"A": "G"}, symbol_to_subindustry={"A": "K"}))
    assert cw.lookup(sub_industry="DOES-NOT-EXIST") is None
    assert cw.lookup() is None  # no attributes → nothing to resolve


def test_resolve_returns_strict_and_plurality_from_different_tiers():
    # Sub-industry is a 1–1 split (plurality, but below strict thresholds); the
    # sector+industry tier is decisive. One pass must yield both, from different tiers.
    cw = IBDCrosswalk(_build(
        symbol_to_group={"M0": "G1", "M1": "G2", **{f"T{i}": "Strict-Group" for i in range(5)}},
        symbol_to_subindustry={"M0": "SubMixed", "M1": "SubMixed"},
        symbol_to_sector_industry={f"T{i}": ("Tech", "Software") for i in range(5)},
    ))
    res = cw.resolve(sub_industry="SubMixed", sector="Tech", industry="Software")

    # plurality = most-specific tier with any data = the 1–1 sub-industry winner
    assert res.plurality is not None
    assert res.plurality.method == TIER_SUBINDUSTRY
    assert res.plurality.group == "G1"
    assert res.plurality.confidence == 0.5
    # strict skips the sub-industry (below thresholds) and lands on sector+industry
    assert res.strict is not None
    assert res.strict.method == TIER_SECTOR_INDUSTRY
    assert res.strict.group == "Strict-Group"
    # lookup() is exactly resolve().strict
    assert cw.lookup(
        sub_industry="SubMixed", sector="Tech", industry="Software"
    ).method == TIER_SECTOR_INDUSTRY


def test_resolve_none_when_no_tier_has_data():
    cw = IBDCrosswalk(_build(symbol_to_group={"A": "G"}, symbol_to_subindustry={"A": "K"}))
    res = cw.resolve(sub_industry="UNKNOWN")
    assert res.strict is None and res.plurality is None
