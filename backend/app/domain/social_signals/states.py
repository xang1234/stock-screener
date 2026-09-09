"""Signal eligibility from saved checks; Market posture never changes scores."""

from decimal import Decimal

from .records import SUPPORTED_MARKETS, SignalStateDecision, SignalStateInput


def classify_signal_state(input: SignalStateInput) -> SignalStateDecision:
    if input.security_kind in {"broad_etf", "macro"}:
        return SignalStateDecision("context", ("market_context",))
    if not input.resolved:
        return SignalStateDecision("unresolved", ("unresolved_security",))
    reasons = []
    if not input.active:
        reasons.append("inactive_security")
    if input.market not in SUPPORTED_MARKETS:
        reasons.append("unsupported_market")
    for value, missing, failed in (
        (input.feature_fresh, "unknown_feature_freshness", "stale_features"),
        (input.market_fresh, "unknown_market_freshness", "stale_market"),
        (input.liquidity_eligible, "missing_liquidity", "liquidity_rejected"),
        (input.setup_ready, "missing_setup_readiness", "setup_not_ready"),
    ):
        if value is not True:
            reasons.append(missing if value is None else failed)
    for value, name in ((input.setup_score, "setup_score"), (input.market_exposure, "market_exposure")):
        if value is None:
            reasons.append(f"missing_{name}")
        elif not Decimal(str(value)).is_finite() or not 0 <= value <= 100:
            reasons.append(f"invalid_{name}")
    if reasons:
        return SignalStateDecision("watch", tuple(reasons))
    if input.market_exposure < 50:
        return SignalStateDecision("risk_off", ("market_exposure_below_50",))
    return SignalStateDecision("actionable")
