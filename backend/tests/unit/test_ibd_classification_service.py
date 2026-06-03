"""Unit tests for the hybrid IBD classifier cascade.

Uses a deterministic fake embedding engine (orthogonal one-hot vectors keyed off
group keywords) so the crosswalk → embedding → LLM tiers can be exercised without
loading sentence-transformers or calling an API.
"""
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDIndustryGroup
from app.models.stock_universe import StockUniverse
from app.services.ibd_classification_service import (
    SOURCE_CROSSWALK,
    SOURCE_EMBEDDING,
    SOURCE_LLM,
    IBDClassificationService,
    StockContext,
    _LLMTier,
)
from app.services.ibd_crosswalk import IBDCrosswalk, build_crosswalk

DIM = 3
_KEYWORDS = {"software": 0, "drug": 1, "energy": 2}
_NEUTRAL = np.ones(DIM) / np.sqrt(DIM)


class FakeEngine:
    """encode() → one-hot for a recognised keyword, neutral vector otherwise,
    or None for the sentinel 'NOVEC' (simulates an un-embeddable stock)."""

    def __init__(self):
        self.calls = 0

    def encode(self, text: str):
        self.calls += 1
        lowered = (text or "").lower()
        if "novec" in lowered:
            return None
        for kw, idx in _KEYWORDS.items():
            if kw in lowered:
                v = np.zeros(DIM)
                v[idx] = 1.0
                return v
        return _NEUTRAL.copy()

    @staticmethod
    def cosine_similarity(left, right) -> float:
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / denom) if denom else 0.0


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_taxonomy(session):
    # Curated US groups (centroids derive from these names; no US universe rows,
    # so centroids are name-only — the foreign-market scenario).
    for sym, grp in [
        ("AAPL", "Computers-Software"),
        ("JNJ", "Medical-Drugs"),
        ("XOM", "Energy-Oil"),
    ]:
        session.add(IBDIndustryGroup(symbol=sym, industry_group=grp, market="US", source="csv"))
    session.commit()


def _add_universe(session, symbol, market, name="", sector="", industry=""):
    session.add(StockUniverse(
        symbol=symbol, name=name, market=market, sector=sector, industry=industry, is_active=True
    ))
    session.commit()


def test_crosswalk_tier_short_circuits():
    session = _make_session()
    _seed_taxonomy(session)
    _add_universe(session, "D05.SG", "SG", name="DBS", sector="Financials", industry="Banks")
    crosswalk = IBDCrosswalk(build_crosswalk(
        symbol_to_group={f"B{i}": "Banks-Money Center" for i in range(5)},
        symbol_to_sector_industry={f"B{i}": ("Financials", "Banks") for i in range(5)},
    ))
    svc = IBDClassificationService(crosswalk=crosswalk, embedding_engine=FakeEngine())

    result = svc.classify_market(session, "SG")

    assert len(result.assignments) == 1
    a = result.assignments[0]
    assert a.source == SOURCE_CROSSWALK
    assert a.industry_group == "Banks-Money Center"
    assert a.confidence == 1.0


def test_embedding_tier_attaches_above_threshold():
    session = _make_session()
    _seed_taxonomy(session)
    _add_universe(session, "0700.HK", "HK", name="Tencent", sector="Tech", industry="Software")
    svc = IBDClassificationService(crosswalk=None, embedding_engine=FakeEngine())

    result = svc.classify_market(session, "HK")

    a = result.assignments[0]
    assert a.source == SOURCE_EMBEDDING
    assert a.industry_group == "Computers-Software"
    assert a.confidence is not None and a.confidence > 0.99


def test_llm_tiebreaker_invoked_below_threshold():
    session = _make_session()
    _seed_taxonomy(session)
    # Neutral text → best cosine ~0.577; high attach threshold forces the LLM.
    _add_universe(session, "MIXED.SG", "SG", name="Conglomerate Holdings")
    captured = {}

    def fake_llm(text, shortlist):
        captured["shortlist"] = shortlist
        return shortlist[-1]  # pick a deterministic, non-top candidate

    svc = IBDClassificationService(
        crosswalk=None, embedding_engine=FakeEngine(),
        llm_tiebreaker=fake_llm, llm_model_id="test/model", attach_threshold=0.9,
    )

    result = svc.classify_market(session, "SG")

    a = result.assignments[0]
    assert a.source == SOURCE_LLM
    assert a.model_id == "test/model"
    assert a.industry_group == captured["shortlist"][-1]
    assert a.industry_group in {"Computers-Software", "Medical-Drugs", "Energy-Oil"}


def test_authoritative_symbols_are_skipped():
    session = _make_session()
    _seed_taxonomy(session)
    _add_universe(session, "RELIANCE.NS", "IN", name="Reliance", sector="Energy", industry="Oil")
    # Human override for the same symbol → must be skipped, counted authoritative.
    session.add(IBDIndustryGroup(
        symbol="RELIANCE.NS", industry_group="Energy-Oil", market="IN", source="manual"
    ))
    session.commit()
    svc = IBDClassificationService(crosswalk=None, embedding_engine=FakeEngine())

    result = svc.classify_market(session, "IN")

    assert result.assignments == []
    assert result.candidates == 0
    assert result.skipped_authoritative == 1


def test_unresolved_when_no_match_and_no_llm():
    session = _make_session()
    _seed_taxonomy(session)
    _add_universe(session, "NOVEC.SG", "SG", name="NOVEC unembeddable")
    svc = IBDClassificationService(crosswalk=None, embedding_engine=FakeEngine())

    result = svc.classify_market(session, "SG")

    assert result.assignments == []
    assert result.unresolved == ["NOVEC.SG"]
    assert result.summary()["coverage_pct"] == 0.0


def _counting_llm(return_value=None):
    calls = {"n": 0}

    def _llm(text, shortlist):
        calls["n"] += 1
        return shortlist[-1] if return_value is None else return_value

    return _llm, calls


def test_deadline_disables_llm_and_falls_back_to_deterministic():
    session = _make_session()
    _seed_taxonomy(session)
    # Two neutral SG stocks → both would hit the LLM (threshold 0.9), but the
    # deadline trips before the 2nd, so it must fall back to a free soft attach.
    # Distinct sectors → distinct LLM-cache keys, so the 2nd isn't served from cache.
    _add_universe(session, "AAA.SG", "SG", name="Alpha Holdings", sector="SectorA")
    _add_universe(session, "BBB.SG", "SG", name="Beta Holdings", sector="SectorB")
    llm, calls = _counting_llm()
    svc = IBDClassificationService(
        crosswalk=None, embedding_engine=FakeEngine(),
        llm_tiebreaker=llm, llm_model_id="test/model", attach_threshold=0.9,
    )

    # The deadline trips only after the first real LLM call: the 1st stock gets the
    # LLM, the 2nd is past the deadline and must fall back to a free soft attach.
    # Keyed on call count (not clock-call order) so it's robust to bookkeeping reads.
    clock = lambda: 0.0 if calls["n"] == 0 else 100.0  # noqa: E731
    result = svc.classify_market(
        session, "SG", deadline_seconds=10, clock=clock, soft_attach=True,
    )

    assert calls["n"] == 1                      # only the first symbol reached the LLM
    assert result.deadline_hit is True
    assert result.llm_calls == 1
    assert len(result.assignments) == 2         # both still classified (coverage kept)
    sources = sorted(a.source for a in result.assignments)
    assert SOURCE_LLM in sources
    assert any(a.method == "centroid_nn_soft" for a in result.assignments)
    assert result.summary()["partial"] is True
    assert result.summary()["processed"] == 2


def test_max_llm_calls_budget_caps_calls():
    session = _make_session()
    _seed_taxonomy(session)
    # Distinct sectors → distinct LLM-cache keys, so the budget (not the cache) is
    # what forces the fallback here.
    for sym, sec in [("AAA.SG", "S1"), ("BBB.SG", "S2"), ("CCC.SG", "S3")]:
        _add_universe(session, sym, "SG", name=f"{sym} Holdings", sector=sec)
    llm, calls = _counting_llm()
    svc = IBDClassificationService(
        crosswalk=None, embedding_engine=FakeEngine(),
        llm_tiebreaker=llm, llm_model_id="test/model", attach_threshold=0.9,
    )

    result = svc.classify_market(session, "SG", max_llm_calls=1, soft_attach=True)

    assert calls["n"] == 1                      # budget capped the paid tier
    assert result.llm_calls == 1
    assert len(result.assignments) == 3         # the other two soft-attached for free
    assert sum(1 for a in result.assignments if a.method == "centroid_nn_soft") == 2
    # Budget exhaustion is reported distinctly from the deadline alarm.
    assert result.llm_budget_exhausted is True
    assert result.summary()["llm_budget_exhausted"] is True
    assert result.summary()["partial"] is False  # budget cap is by-design, not "partial"


def test_llm_cache_dedupes_same_industry_calls():
    session = _make_session()
    _seed_taxonomy(session)
    # Two neutral stocks with identical industry attributes → identical shortlist
    # → one shared LLM decision; the second is a free cache hit.
    _add_universe(session, "AAA.SG", "SG", name="Alpha Holdings", sector="Finance", industry="Banks")
    _add_universe(session, "BBB.SG", "SG", name="Beta Holdings", sector="Finance", industry="Banks")
    llm, calls = _counting_llm()
    svc = IBDClassificationService(
        crosswalk=None, embedding_engine=FakeEngine(),
        llm_tiebreaker=llm, llm_model_id="test/model", attach_threshold=0.9,
    )

    result = svc.classify_market(session, "SG", soft_attach=True)

    assert calls["n"] == 1                      # one network call serves both
    assert result.llm_calls == 1
    assert result.llm_cache_hits == 1
    assert len(result.assignments) == 2
    assert all(a.source == SOURCE_LLM for a in result.assignments)
    assert result.summary()["llm_cache_hits"] == 1


def test_llm_cache_hit_bypasses_budget():
    session = _make_session()
    _seed_taxonomy(session)
    # All three share one industry+shortlist; with a budget of 1 the single distinct
    # call is spent once and the other two are served free from cache — so all three
    # keep full LLM coverage instead of being soft-attached.
    for sym in ("AAA.SG", "BBB.SG", "CCC.SG"):
        _add_universe(session, sym, "SG", name=f"{sym} Co", sector="Finance", industry="Banks")
    llm, calls = _counting_llm()
    svc = IBDClassificationService(
        crosswalk=None, embedding_engine=FakeEngine(),
        llm_tiebreaker=llm, llm_model_id="test/model", attach_threshold=0.9,
    )

    result = svc.classify_market(session, "SG", max_llm_calls=1, soft_attach=True)

    assert calls["n"] == 1
    assert result.llm_calls == 1
    assert result.llm_cache_hits == 2
    assert len(result.assignments) == 3
    assert all(a.source == SOURCE_LLM for a in result.assignments)  # none soft-attached


def test_soft_attach_uses_relaxed_crosswalk_plurality():
    session = _make_session()
    _seed_taxonomy(session)
    _add_universe(session, "D05.SG", "SG", name="DBS", sector="Financials", industry="Banks")
    # A 1-1 split -> strict lookup (0.6/3) fails, but the relaxed plurality wins.
    crosswalk = IBDCrosswalk(build_crosswalk(
        symbol_to_group={"B0": "Banks-Money Center", "B1": "Banks-Regional"},
        symbol_to_sector_industry={
            "B0": ("Financials", "Banks"), "B1": ("Financials", "Banks"),
        },
    ))
    svc = IBDClassificationService(
        crosswalk=crosswalk, embedding_engine=FakeEngine(), attach_threshold=0.9,
    )

    result = svc.classify_market(session, "SG", soft_attach=True)

    a = result.assignments[0]
    assert a.source == SOURCE_CROSSWALK
    assert a.industry_group == "Banks-Money Center"   # deterministic tie-break
    assert a.confidence == 0.5                          # actual vote share
    assert a.method == "sector_industry_plurality"


def test_soft_attach_falls_back_to_embedding_top1():
    session = _make_session()
    _seed_taxonomy(session)
    # Neutral name (no keyword) → best cosine ~0.577, below the 0.9 threshold, no
    # crosswalk and no LLM → soft attach to the nearest centroid (lexicographic
    # tie-break across the equal-distance one-hot centroids → Computers-Software).
    _add_universe(session, "0700.HK", "HK", name="Generic Holdings")
    svc = IBDClassificationService(
        crosswalk=None, embedding_engine=FakeEngine(), attach_threshold=0.9,
    )

    result = svc.classify_market(session, "HK", soft_attach=True)

    a = result.assignments[0]
    assert a.source == SOURCE_EMBEDDING
    assert a.method == "centroid_nn_soft"
    assert a.industry_group == "Computers-Software"


def test_soft_attach_off_preserves_unresolved():
    session = _make_session()
    _seed_taxonomy(session)
    _add_universe(session, "MIXED.SG", "SG", name="Conglomerate Holdings")
    svc = IBDClassificationService(
        crosswalk=None, embedding_engine=FakeEngine(), attach_threshold=0.9,
    )

    result = svc.classify_market(session, "SG")  # soft_attach defaults False

    assert result.assignments == []
    assert result.unresolved == ["MIXED.SG"]


def test_progress_callback_receives_counts():
    session = _make_session()
    _seed_taxonomy(session)
    _add_universe(session, "0700.HK", "HK", name="Tencent Software")
    seen = []
    svc = IBDClassificationService(crosswalk=None, embedding_engine=FakeEngine())

    svc.classify_market(session, "HK", progress_every=1, progress_callback=seen.append)

    assert seen and seen[-1]["processed"] == 1
    assert seen[-1]["market"] == "HK"
    assert "by_source" in seen[-1]


def test_canonical_taxonomy_excludes_machine_derived_rows():
    session = _make_session()
    _seed_taxonomy(session)  # US csv groups
    # Machine-derived rows (any market) must NOT widen the taxonomy — otherwise a
    # bad guess becomes a "real" group and grows its own centroid (feedback loop).
    session.add(IBDIndustryGroup(
        symbol="X.SG", industry_group="Bogus-Embedding-Group",
        market="SG", source="embedding", confidence=0.3,
    ))
    session.add(IBDIndustryGroup(
        symbol="Y.SG", industry_group="Bogus-LLM-Group", market="SG", source="llm",
    ))
    session.commit()

    svc = IBDClassificationService(crosswalk=None, embedding_engine=None)
    taxonomy = svc.canonical_taxonomy(session)

    assert "Bogus-Embedding-Group" not in taxonomy
    assert "Bogus-LLM-Group" not in taxonomy
    assert set(taxonomy) == {"Computers-Software", "Medical-Drugs", "Energy-Oil"}


def test_canonical_taxonomy_includes_manual_foreign_group():
    session = _make_session()
    _seed_taxonomy(session)  # US csv groups
    # A human-curated (manual) row for a genuinely foreign industry extends the
    # ONE shared namespace, regardless of market — the escape hatch for industries
    # the US CSV doesn't cover (e.g. a CN-specific group).
    session.add(IBDIndustryGroup(
        symbol="600519.SS", industry_group="Beverages-Baijiu",
        market="CN", source="manual",
    ))
    session.commit()

    svc = IBDClassificationService(crosswalk=None, embedding_engine=None)
    taxonomy = svc.canonical_taxonomy(session)

    assert "Beverages-Baijiu" in taxonomy
    assert set(taxonomy) == {
        "Computers-Software", "Medical-Drugs", "Energy-Oil", "Beverages-Baijiu",
    }


# ---- _LLMTier (the rationed/deduped paid tier), unit-tested without a DB --------

def _tier_tiebreaker():
    calls = {"n": 0}

    def tb(text, shortlist):
        calls["n"] += 1
        return shortlist[0]

    return tb, calls


def test_llm_tier_caches_and_caps_budget():
    tb, calls = _tier_tiebreaker()
    tier = _LLMTier(tb, model_id="m", max_calls=1, deadline_seconds=None, clock=lambda: 0.0)
    a = StockContext("A", "SG", sector="Fin", industry="Banks")
    b = StockContext("B", "SG", sector="Fin", industry="Banks")   # same key as a
    c = StockContext("C", "SG", sector="Energy", industry="Oil")  # distinct key

    assert tier.choose(a, ["G1", "G2"]) == "G1"   # miss → one paid call
    assert tier.choose(b, ["G1", "G2"]) == "G1"   # same ranked shortlist → free cache hit
    assert tier.choose(c, ["G3"]) is None         # budget spent → decline
    assert tier.calls == 1 and tier.cache_hits == 1
    assert tier.budget_exhausted is True
    assert tier.choose(b, ["G1", "G2"]) == "G1"   # cached → still served over budget


def test_llm_tier_reordered_shortlist_is_a_distinct_prompt():
    # The LLM prompt renders candidates in order, so a reordered shortlist is a
    # different prompt and must NOT reuse the cached answer.
    tb, calls = _tier_tiebreaker()  # returns shortlist[0]
    tier = _LLMTier(tb, model_id="m", max_calls=5, deadline_seconds=None, clock=lambda: 0.0)
    a = StockContext("A", "SG", sector="Fin", industry="Banks")

    assert tier.choose(a, ["G1", "G2"]) == "G1"   # miss → call (top candidate G1)
    assert tier.choose(a, ["G2", "G1"]) == "G2"   # reordered → separate call (top G2)
    assert calls["n"] == 2
    assert tier.cache_hits == 0


def test_llm_tier_declines_past_deadline_but_serves_cache():
    tb, calls = _tier_tiebreaker()
    now = {"v": 0.0}
    tier = _LLMTier(tb, model_id="m", max_calls=None, deadline_seconds=10, clock=lambda: now["v"])
    a = StockContext("A", "SG", sector="Fin", industry="Banks")
    b = StockContext("B", "SG", sector="Energy", industry="Oil")

    assert tier.choose(a, ["G1"]) == "G1"     # under deadline → call
    now["v"] = 100.0                          # past deadline
    assert tier.choose(b, ["G2"]) is None     # uncached → declined
    assert tier.deadline_reached is True
    assert tier.calls == 1
    assert tier.choose(a, ["G1"]) == "G1"     # cached → served even past deadline
    assert tier.cache_hits == 1


def test_llm_tier_caches_none_result():
    # A "no match" (LLM declined / hallucinated off-list) is cached too, so the same
    # industry+shortlist is never re-queried.
    calls = {"n": 0}

    def tb(text, shortlist):
        calls["n"] += 1
        return None

    tier = _LLMTier(tb, model_id="m", max_calls=5, deadline_seconds=None, clock=lambda: 0.0)
    a = StockContext("A", "SG", sector="Fin", industry="Banks")

    assert tier.choose(a, ["G1"]) is None     # miss → one call
    assert tier.choose(a, ["G1"]) is None     # same key → cache hit, not a retry
    assert calls["n"] == 1
    assert tier.cache_hits == 1


def test_llm_tier_raising_tiebreaker_is_charged_and_cached():
    # A raising tiebreaker must not break the run, must consume budget hard enough
    # to block the next key (the ceiling holds under errors), and its (None) outcome
    # is cached so the same key isn't retried.
    calls = {"n": 0}

    def tb(text, shortlist):
        calls["n"] += 1
        raise RuntimeError("provider down")

    tier = _LLMTier(tb, model_id="m", max_calls=1, deadline_seconds=None, clock=lambda: 0.0)
    a = StockContext("A", "SG", sector="Fin", industry="Banks")
    b = StockContext("B", "SG", sector="Energy", industry="Oil")  # distinct key

    assert tier.choose(a, ["G1"]) is None     # swallowed → no match
    assert tier.calls == 1                    # budget charged despite the raise
    assert tier.budget_exhausted is True      # the raise spent the only slot
    assert tier.choose(a, ["G1"]) is None     # same key → cache hit, not retried
    assert tier.cache_hits == 1
    assert tier.choose(b, ["G2"]) is None     # distinct key, budget spent → declined
    assert calls["n"] == 1                    # tiebreaker invoked exactly once total


def test_llm_tier_without_tiebreaker_declines():
    tier = _LLMTier(None, model_id=None, max_calls=None, deadline_seconds=None, clock=lambda: 0.0)
    a = StockContext("A", "SG", sector="Fin", industry="Banks")
    assert tier.choose(a, ["G1"]) is None
    assert tier.choose(a, []) is None
    assert tier.calls == 0 and tier.cache_hits == 0
