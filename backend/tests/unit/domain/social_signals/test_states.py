from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.domain.social_signals import records


def state_input(**changes):
    values = dict(resolved=True, active=True, market="US", security_kind="stock",
                  feature_fresh=True, market_fresh=True, liquidity_eligible=True,
                  setup_ready=True, setup_score=Decimal(80), market_exposure=Decimal(65))
    values.update(changes)
    return records.SignalStateInput(**values)


def test_market_posture_changes_state_without_entering_any_numeric_score():
    from app.domain.social_signals.states import classify_signal_state
    from app.domain.social_signals.scoring import queue_score, score_confirmation, score_social_candidates
    inp = state_input()
    now = datetime(2026, 9, 6, tzinfo=timezone.utc)
    post = records.SocialPostRecord("xui", "1", "a", "$NVDA thesis", "https://x.com/a/status/1", "a",
                                    now-timedelta(hours=1), now, likes=10, reposts=2, replies=1)
    evidence = records.SocialEvidenceInput("US:NVDA", "NVDA", "US", (post,), ("a", "b"), True)
    technical = records.ConfirmationInput("US:NVDA", "US", now, setup_score=Decimal(80))
    def scored():
        social = score_social_candidates((evidence,), 14, now)[0].social_score
        confirmation = score_confirmation(technical).value
        return social, confirmation, queue_score(social=social, confirmation=confirmation)
    before = scored()
    assert classify_signal_state(inp).state == "actionable"
    assert classify_signal_state(replace(inp, market_exposure=Decimal(30))).state == "risk_off"
    assert scored() == before
    assert classify_signal_state(replace(inp, market_exposure=Decimal(50))).state == "actionable"


@pytest.mark.parametrize(("changes", "reason"), [
    ({"active": False}, "inactive_security"),
    ({"market": "GB"}, "unsupported_market"),
    ({"feature_fresh": False}, "stale_features"),
    ({"feature_fresh": None}, "unknown_feature_freshness"),
    ({"market_fresh": False}, "stale_market"),
    ({"market_fresh": None}, "unknown_market_freshness"),
    ({"liquidity_eligible": False}, "liquidity_rejected"),
    ({"liquidity_eligible": None}, "missing_liquidity"),
    ({"setup_ready": False}, "setup_not_ready"),
    ({"setup_ready": None}, "missing_setup_readiness"),
    ({"setup_score": None}, "missing_setup_score"),
    ({"market_exposure": None}, "missing_market_exposure"),
])
def test_required_confirmation_failures_are_watch_even_in_risk_off_market(changes, reason):
    from app.domain.social_signals.states import classify_signal_state
    values = dict(market_exposure=Decimal(30))
    values.update(changes)
    result = classify_signal_state(state_input(**values))
    assert result.state == "watch"
    assert reason in result.reasons


def test_broad_etf_and_macro_are_context_thematic_etf_is_eligible_unresolved_stays_unresolved():
    from app.domain.social_signals.states import classify_signal_state
    for kind in ("broad_etf", "macro"):
        assert classify_signal_state(state_input(security_kind=kind)).state == "context"
    assert classify_signal_state(state_input(security_kind="thematic_etf")).state == "actionable"
    assert classify_signal_state(state_input(resolved=False, market=None)).state == "unresolved"
