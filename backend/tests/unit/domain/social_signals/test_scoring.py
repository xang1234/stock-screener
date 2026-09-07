"""Hand-derived policy examples; no providers, clocks, databases, or models."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from math import log1p

import pytest

from app.domain.social_signals import records

NOW = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)


def post(pid="1", author="alice", source="a", age=1, **kwargs):
    values = dict(provider="xui", provider_post_id=pid, source_id=source,
                  text="$NVDA thesis", url=f"https://x.com/{author}/status/{pid}",
                  author_handle=author, created_at=NOW - timedelta(hours=age),
                  observed_at=NOW, likes=0, reposts=0, replies=0)
    values.update(kwargs)
    return records.SocialPostRecord(**values)


def evidence(posts=(), key="US:NVDA", **kwargs):
    values = dict(candidate_key=key, canonical_symbol=key.split(":")[1],
                  market=key.split(":")[0], posts=tuple(posts),
                  enabled_source_ids=("a", "b", "c"), history_complete=True)
    values.update(kwargs)
    return records.SocialEvidenceInput(**values)


def score(row, days=14):
    from app.domain.social_signals.scoring import score_social_candidates
    return score_social_candidates((row,), days, NOW)[0]


def components(row):
    return dict(row.components)


def test_exact_formulas_and_missing_top_level_portion():
    from app.domain.social_signals.scoring import engagement_value, queue_score, recency_score
    assert engagement_value(likes=10, reposts=2, replies=4, quotes=2, bookmarks=1, views=1000) == pytest.approx(log1p(26))
    assert engagement_value(likes=0, reposts=0, replies=0) == 0
    assert engagement_value(likes=10, reposts=None, replies=4) is None
    assert recency_score(age_hours=48) == Decimal("50.0000")
    assert queue_score(social=80, confirmation=50) == Decimal("68.0000")
    assert queue_score(social=80, confirmation=None) is None
    assert queue_score(social=None, confirmation=50) is None


def test_midrank_percentiles_and_linear_winsorization():
    from app.domain.social_signals.scoring import midrank_percentile, percentile_threshold
    assert midrank_percentile(10, (10, 10, 30, 40)) == Decimal("25.0000")
    assert midrank_percentile(7, (7,)) == Decimal("50.0000")
    assert midrank_percentile(7, ()) is None
    assert percentile_threshold((0, 10, 20, 30), Decimal("0.95")) == Decimal("28.50")


def test_duplicate_post_retains_memberships_without_cross_list_credit():
    row = score(evidence((post(source="a"), post(source="b"))))
    assert row.mention_count == 1
    assert row.post_memberships == (("xui:1", ("a", "b")),)
    assert row.observed_list_count == 2
    assert components(row)["cross_list"].value == 0


@pytest.mark.parametrize("sources", [("a", "b"), ("b", "c"), ("a", "b", "c")])
def test_distinct_posts_from_any_two_enabled_sources_earn_full_binary_credit(sources):
    row = score(evidence(tuple(post(str(i), str(i), source) for i, source in enumerate(sources))))
    assert components(row)["cross_list"].value == 100


def test_disabled_source_does_not_earn_mentions_or_list_credit():
    row = score(evidence((post(), post("2", "bob", "disabled"))))
    assert row.mention_count == 1
    assert row.observed_list_count == 1
    assert components(row)["cross_list"].value == 0


def test_rolling_author_cap_is_not_a_calendar_day_cap():
    row = score(evidence(tuple(post(str(i), age=30-i*4) for i in range(5))))
    assert row.mention_count == 3
    assert ("xui:3", "author_24h_cap") in row.exclusions
    assert ("xui:4", "author_24h_cap") in row.exclusions
    later = score(evidence(tuple(post(str(i), age=49-i*8) for i in range(5))))
    assert later.mention_count == 5  # exactly 24h apart falls outside (t-24h, t]


def test_author_cap_carries_in_saved_posts_before_fourteen_day_boundary():
    row = score(evidence(tuple(post(str(i), age=340-i) for i in range(5))))
    assert row.mention_count == 0  # three weighted posts at 340/339/338h already fill cap
    assert ("xui:4", "author_24h_cap") in row.exclusions


def test_reposts_empty_quotes_copied_claims_and_canonical_urls_do_not_inflate():
    row = score(evidence((
        post("1", canonical_url="https://example.com/claim", canonical_claim_key="claim"),
        post("2", "b", is_repost=True),
        post("3", "c", quoted_text="original", has_new_thesis=False),
        post("4", "d", canonical_url="https://example.com/claim"),
        post("5", "e", canonical_claim_key="claim"),
        post("6", "f", quoted_text="original", has_new_thesis=True),
    )))
    assert row.mention_count == 2
    assert len(row.exclusions) == 4


def test_future_observations_and_publications_are_excluded_even_with_ingress_skew():
    row = score(evidence((post(), post("2", created_at=NOW+timedelta(minutes=1)),
                          post("3", observed_at=NOW+timedelta(minutes=1)))))
    assert row.mention_count == 1
    assert len(row.exclusions) == 2


def test_acceleration_requires_explicit_history_and_window_boundaries_are_rolling():
    row = score(evidence((post(age=24), post("2", "b", age=24.01),
                          post("3", "c", age=336), post("4", "d", age=336.01))), 1)
    assert row.mention_count == 1
    assert row.acceleration == Decimal("2") / (Decimal("2") / 13 + 1)
    warm = score(evidence((post(),), history_complete=False, coverage_reasons=("post_cap_reached",)))
    assert components(warm)["acceleration"].value is None
    assert "post_cap_reached" in components(warm)["acceleration"].reasons
    assert warm.social_score is not None


def test_no_qualifying_posts_returns_null_not_synthetic_observed_zero():
    row = score(evidence((post(is_repost=True),)))
    assert row.social_score is None
    assert all(component.value is None for _, component in row.components)


def test_social_weights_renormalize_only_available_components():
    complete = score(evidence((post(age=48),)))
    assert complete.social_score == Decimal("45.8333")
    warm = score(evidence((post(age=0, likes=None), post("2", "bob", "b", age=0, likes=None)), history_complete=False))
    assert warm.social_score == Decimal("70.0000")  # (50*15 + 100*5 + 100*5)/25


def test_partial_engagement_preserves_coverage_and_missing_is_not_zero():
    row = score(evidence((post(), post("2", "b", likes=None))))
    c = components(row)["engagement"]
    assert c.value == 50
    assert (c.observed_count, c.input_count) == (1, 2)
    assert "missing_engagement_metrics" in c.reasons


def test_market_cohort_at_twenty_and_global_fallback_below_twenty():
    from app.domain.social_signals.scoring import score_social_candidates
    us = tuple(evidence((post(str(i)),), key=f"US:S{i:02}") for i in range(20))
    hk = evidence((post("hk1"), post("hk2", "b")), key="HK:00001")
    rows = score_social_candidates(us + (hk,), 14, NOW)
    assert rows[0].normalization_scope == "global_fallback"  # canonical candidate-key order
    by_key = {r.candidate_key: r for r in rows}
    assert by_key["US:S00"].normalization_scope == "market"
    assert components(by_key["US:S00"])["authors"].value == 50
    assert components(by_key["HK:00001"])["authors"].value == Decimal("97.6190")
    fallback = score_social_candidates(us[:19] + (hk,), 14, NOW)
    assert {r.normalization_scope for r in fallback} == {"global_fallback"}
    assert components({r.candidate_key: r for r in fallback}["US:S00"])["authors"].value == Decimal("47.5000")


def test_candidates_without_window_mentions_do_not_trigger_market_normalization():
    from app.domain.social_signals.scoring import score_social_candidates
    us = tuple(evidence((post(str(i)),), key=f"US:S{i:02}") for i in range(19))
    absent = evidence((post("old", age=200),), key="US:OLD")
    hk = evidence((post("hk"), post("hk2", "bob")), key="HK:00001")
    rows = score_social_candidates(us + (absent, hk), 1, NOW)
    by_key = {r.candidate_key: r for r in rows}
    assert by_key["US:S00"].normalization_scope == "global_fallback"
    assert components(by_key["US:S00"])["authors"].value == Decimal("47.5000")


def test_winsorization_is_per_post_before_candidate_summation():
    from app.domain.social_signals.scoring import score_social_candidates
    rows = score_social_candidates((evidence((post(likes=0),), key="US:A"),
                                    evidence((post("2", likes=99),), key="US:B")), 14, NOW)
    assert rows[1].engagement_sum == pytest.approx(0.95 * log1p(99))


def test_context_and_unresolved_are_outside_cohort_and_replay_is_byte_identical():
    from app.domain.social_signals.scoring import score_social_candidates
    inputs = (evidence((post(),)), evidence((post("2"),), key="US:SPY", security_kind="broad_etf"),
              evidence((post("3"),), key="US:UNKNOWN", resolved=False))
    first = score_social_candidates(inputs, 14, NOW)
    second = score_social_candidates(tuple(replace(r, posts=tuple(reversed(r.posts))) for r in reversed(inputs)), 14, NOW)
    assert len(first) == 1
    assert components(first[0])["authors"].value == 50
    serialize = lambda rows: json.dumps([asdict(r) for r in rows], default=str, sort_keys=True).encode()
    assert serialize(first) == serialize(second)


def confirmation(**kwargs):
    return records.ConfirmationInput(candidate_key="US:NVDA", market="US", observed_at=NOW, **kwargs)


def test_confirmation_exact_weights_partial_rs_and_group_formula():
    from app.domain.social_signals.scoring import score_confirmation
    row = score_confirmation(confirmation(setup_score=Decimal(80), rs_rating_1m=Decimal(50),
                                         rs_rating_3m=Decimal(100), group_rank=2, market_group_count=5))
    assert row.value == Decimal("79.2857")  # (80*20 + 80*10 + 75*5)/35
    assert (row.available_weight, row.total_weight) == (35, 40)
    partial = score_confirmation(confirmation(rs_rating_1m=Decimal(70)))
    assert partial.value == 70
    assert "missing_rs_rating_3m" in dict(partial.components)["rs"].reasons
    assert score_confirmation(confirmation(group_rank=1, market_group_count=1)).value is None


def test_theme_component_coverage_thresholds_highest_selection_and_tie_break():
    from app.domain.social_signals.scoring import score_confirmation
    themes = (
        theme("z", (80, 100, 90)),
        theme("a", (90, 90, 90)),
        theme("bad", (100, 100, None), counts=(2, 6, 0)),
    )
    row = score_confirmation(confirmation(theme_confirmations=themes, market_benchmark="SPY"))
    assert row.value == 90
    assert dict(row.components)["theme"].selected_key == "a"
    assert "theme:bad:basket_rs_vs_benchmark:insufficient_coverage" in dict(row.components)["theme"].reasons
    assert score_confirmation(confirmation(theme_confirmations=(themes[2],), market_benchmark="SPY")).value is None


def theme(key, values=(90, None, None), counts=(7, 7, 7), market="US", benchmark="SPY"):
    keys = ("basket_rs_vs_benchmark", "avg_rs_rating", "pct_above_50ma")
    return records.ThemeMarketEvidence(key, market, NOW.date(), benchmark, "basket-1", 10,
        tuple(zip(keys, (Decimal(v) if v is not None else None for v in values))),
        tuple(zip(keys, counts)), ())


def test_non_us_theme_requires_matching_market_benchmark():
    from app.domain.social_signals.scoring import score_confirmation
    hk = records.ConfirmationInput("HK:00001", "HK", NOW, market_benchmark="HSI",
        theme_confirmations=(theme("wrong", market="HK"),
                             theme("right", market="HK", benchmark="HSI")))
    assert score_confirmation(hk).value == 90
    assert dict(score_confirmation(hk).components)["theme"].selected_key == "right"


def test_theme_ignores_social_fields_and_missing_history_preserves_null():
    from app.domain.social_signals.scoring import score_confirmation
    frozen = theme("social-only", values=(None, None, None))
    frozen = replace(frozen, components=frozen.components + (("mentions_1d", Decimal(100)), ("momentum_score", Decimal(100))))
    assert score_confirmation(confirmation(theme_confirmations=(frozen,), market_benchmark="SPY")).value is None


def test_immutable_theme_evidence_rejects_mutable_mappings_and_impossible_counts():
    with pytest.raises(TypeError, match="immutable_tuple"):
        replace(theme("a"), components={"avg_rs_rating": Decimal(100)})
    with pytest.raises(ValueError, match="invalid_measured_company_count"):
        replace(theme("a"), measured_company_counts=(("avg_rs_rating", 11),))


def snapshot(key, social, confirmation_value=None, age=1, state="watch"):
    from app.domain.social_signals.scoring import queue_score
    return records.SocialSnapshotRecord("run", key, key.split(":")[1], key.split(":")[0], state,
        Decimal(social) if social is not None else None,
        Decimal(confirmation_value) if confirmation_value is not None else None,
        queue_score(social=social, confirmation=confirmation_value), (), (),
        NOW-timedelta(hours=age), key.split(":")[1], key)


def test_rank_modes_nulls_recency_symbol_and_candidate_key_ties():
    from app.domain.social_signals.scoring import rank_snapshots
    rows = (snapshot("US:B", 80), snapshot("US:A", 80), snapshot("US:C", 80, age=0),
            snapshot("US:D", 40, 50), snapshot("US:E", None), snapshot("US:SPY", 100, 100, state="context"))
    assert [r.symbol for r in rank_snapshots(rows, "pure_social")] == ["C", "A", "B", "D", "E"]
    assert [r.symbol for r in rank_snapshots(rows, "blended")] == ["D", "C", "A", "B", "E"]
    twins = (replace(rows[1], candidate_key="US:A:2"), replace(rows[1], candidate_key="US:A:1"))
    assert [r.candidate_key for r in rank_snapshots(twins, "blended")] == ["US:A:1", "US:A:2"]


@pytest.mark.parametrize("window", [0, 2, 30])
def test_invalid_scoring_window_is_rejected(window):
    from app.domain.social_signals.scoring import score_social_candidates
    with pytest.raises(ValueError, match="invalid_window"):
        score_social_candidates((), window, NOW)
