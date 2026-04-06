"""Registry and resolution helpers for app-wide strategy profiles."""

from __future__ import annotations

from copy import deepcopy

from app.domain.scanning.defaults import get_default_scan_profile
from app.schemas.app_runtime import ScanDefaultsResponse
from app.schemas.strategy_profile import (
    StrategyProfileDetail,
    StrategyProfileDigestConfig,
    StrategyProfileListResponse,
    StrategyProfileStockActionConfig,
    StrategyProfileStewardshipConfig,
)

DEFAULT_PROFILE = "default"


class StrategyProfileService:
    """Resolve immutable strategy-profile overlays."""

    def __init__(self) -> None:
        default_scan_defaults = get_default_scan_profile()
        self._registry: dict[str, StrategyProfileDetail] = {
            "default": StrategyProfileDetail(
                profile="default",
                label="Default",
                description="Balanced leadership and risk posture using the house scan defaults.",
                scan_defaults=ScanDefaultsResponse(**default_scan_defaults),
                digest=StrategyProfileDigestConfig(
                    leader_min_composite_score=70.0,
                    leader_limit=5,
                    leader_sort="composite_score",
                    theme_sort="momentum_score",
                    section_order=["market", "leaders", "themes", "validation", "watchlists", "risks"],
                    weak_validation_positive_rate_floor=0.5,
                    weak_validation_avg_return_floor=0.0,
                ),
                stewardship=StrategyProfileStewardshipConfig(
                    status_priority=["exit_risk", "deteriorating", "strengthening", "unchanged", "missing_from_run"],
                ),
                stock_action=StrategyProfileStockActionConfig(
                    offense_sizing_guidance="full",
                    balanced_sizing_guidance="half",
                    defense_sizing_guidance="probe",
                    earnings_caution_days=14,
                    earnings_imminent_days=5,
                    preferred_setups=["Stage 2 breakouts", "Tight pullbacks", "High RS leaders"],
                    strengths_title="Top Strengths",
                    weaknesses_title="Top Weaknesses",
                    summary_emphasis="balanced",
                ),
            ),
            "growth": StrategyProfileDetail(
                profile="growth",
                label="Growth",
                description="Higher growth thresholds with earnings and revenue acceleration emphasized.",
                scan_defaults=ScanDefaultsResponse(
                    **{
                        **deepcopy(default_scan_defaults),
                        "criteria": {
                            **deepcopy(default_scan_defaults["criteria"]),
                            "custom_filters": {
                                **deepcopy(default_scan_defaults["criteria"]["custom_filters"]),
                                "rs_rating_min": 80,
                                "eps_growth_min": 30,
                                "sales_growth_min": 25,
                                "min_score": 75,
                            },
                        },
                    }
                ),
                digest=StrategyProfileDigestConfig(
                    leader_min_composite_score=74.0,
                    leader_limit=5,
                    leader_sort="growth_then_score",
                    theme_sort="basket_return_1m",
                    section_order=["market", "leaders", "validation", "themes", "watchlists", "risks"],
                    weak_validation_positive_rate_floor=0.48,
                    weak_validation_avg_return_floor=0.0,
                ),
                stewardship=StrategyProfileStewardshipConfig(
                    deteriorating_score_delta_max=-6.0,
                    deteriorating_rs_delta_max=-10.0,
                    strengthening_score_delta_min=4.0,
                    strengthening_rs_delta_min=6.0,
                    status_priority=["strengthening", "deteriorating", "unchanged", "exit_risk", "missing_from_run"],
                ),
                stock_action=StrategyProfileStockActionConfig(
                    offense_sizing_guidance="full",
                    balanced_sizing_guidance="half",
                    defense_sizing_guidance="probe",
                    earnings_caution_days=10,
                    earnings_imminent_days=4,
                    preferred_setups=["Earnings-supported breakouts", "Accelerating growth leaders", "Tight bases"],
                    strengths_title="Growth Drivers",
                    weaknesses_title="Growth Drags",
                    summary_emphasis="growth",
                ),
            ),
            "momentum": StrategyProfileDetail(
                profile="momentum",
                label="Momentum",
                description="Favor relative-strength leadership, velocity, and continuation setups.",
                scan_defaults=ScanDefaultsResponse(
                    **{
                        **deepcopy(default_scan_defaults),
                        "criteria": {
                            **deepcopy(default_scan_defaults["criteria"]),
                            "custom_filters": {
                                **deepcopy(default_scan_defaults["criteria"]["custom_filters"]),
                                "rs_rating_min": 90,
                                "eps_growth_min": 20,
                                "sales_growth_min": 15,
                                "min_score": 78,
                            },
                        },
                    }
                ),
                digest=StrategyProfileDigestConfig(
                    leader_min_composite_score=75.0,
                    leader_limit=5,
                    leader_sort="rs_then_score",
                    theme_sort="mention_velocity",
                    section_order=["market", "leaders", "themes", "watchlists", "validation", "risks"],
                    weak_validation_positive_rate_floor=0.5,
                    weak_validation_avg_return_floor=0.1,
                ),
                stewardship=StrategyProfileStewardshipConfig(
                    deteriorating_score_delta_max=-7.0,
                    deteriorating_rs_delta_max=-8.0,
                    strengthening_score_delta_min=4.0,
                    strengthening_rs_delta_min=5.0,
                    status_priority=["strengthening", "unchanged", "deteriorating", "exit_risk", "missing_from_run"],
                ),
                stock_action=StrategyProfileStockActionConfig(
                    offense_sizing_guidance="full",
                    balanced_sizing_guidance="half",
                    defense_sizing_guidance="probe",
                    earnings_caution_days=9,
                    earnings_imminent_days=3,
                    preferred_setups=["Breakouts on volume", "RS leaders near highs", "High-tight continuation"],
                    strengths_title="Momentum Strengths",
                    weaknesses_title="Momentum Risks",
                    summary_emphasis="momentum",
                ),
            ),
            "risk_off": StrategyProfileDetail(
                profile="risk_off",
                label="Risk Off",
                description="Conservative overlays that prioritize defense, validation quality, and risk controls.",
                scan_defaults=ScanDefaultsResponse(
                    **{
                        **deepcopy(default_scan_defaults),
                        "screeners": ["minervini", "custom", "setup_engine"],
                        "criteria": {
                            **deepcopy(default_scan_defaults["criteria"]),
                            "custom_filters": {
                                **deepcopy(default_scan_defaults["criteria"]["custom_filters"]),
                                "rs_rating_min": 80,
                                "eps_growth_min": 20,
                                "sales_growth_min": 15,
                                "min_score": 80,
                            },
                        },
                    }
                ),
                digest=StrategyProfileDigestConfig(
                    leader_min_composite_score=82.0,
                    leader_limit=3,
                    leader_sort="composite_score",
                    theme_sort="momentum_score",
                    section_order=["market", "risks", "validation", "leaders", "themes", "watchlists"],
                    weak_validation_positive_rate_floor=0.55,
                    weak_validation_avg_return_floor=0.5,
                ),
                stewardship=StrategyProfileStewardshipConfig(
                    exit_score_max=60.0,
                    exit_score_delta_max=-10.0,
                    defense_earnings_exit_window_days=10,
                    deteriorating_score_delta_max=-4.0,
                    deteriorating_rs_delta_max=-6.0,
                    strengthening_score_delta_min=6.0,
                    strengthening_rs_delta_min=10.0,
                    status_priority=["exit_risk", "deteriorating", "unchanged", "strengthening", "missing_from_run"],
                ),
                stock_action=StrategyProfileStockActionConfig(
                    offense_sizing_guidance="half",
                    balanced_sizing_guidance="probe",
                    defense_sizing_guidance="avoid",
                    earnings_caution_days=12,
                    earnings_imminent_days=5,
                    preferred_setups=["Defensive leaders", "Tight risk-defined pullbacks", "Quality setups only"],
                    strengths_title="Defensive Strengths",
                    weaknesses_title="Risk Flags",
                    summary_emphasis="defense",
                ),
            ),
        }

    def list_profiles(self) -> StrategyProfileListResponse:
        """Return all known strategy profiles."""

        profiles = [self.get_profile(name) for name in sorted(self._registry.keys())]
        return StrategyProfileListResponse(profiles=profiles)

    def get_profile(self, profile: str | None = None) -> StrategyProfileDetail:
        """Return a copy of the requested profile or fall back to default."""

        resolved = self._registry.get((profile or DEFAULT_PROFILE).lower(), self._registry[DEFAULT_PROFILE])
        return resolved.model_copy(deep=True)
