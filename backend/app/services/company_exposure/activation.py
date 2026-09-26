"""Staged activation for company-exposure research (plan Task 27; spec §18).

Each privilege is a separate stage. This build installs only the US
verify-only shadow slice; every later stage reports ``not_installed`` even
when configuration asks for it, so a setting can never enable behaviour
whose gates have not shipped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.domain.company_exposure.contracts import ResearchMode
from app.services.company_exposure.config import ExposureRuntimeConfig

SHADOW_VERIFY_US = "shadow_verify_us"
STAGES = (
    SHADOW_VERIFY_US,
    "shadow_verify_all_markets",
    "generation_reads",
    "automatic_admission",
    "bounded_discovery",
    "paid_search",
    "classifier_grounding",
)
INSTALLED_STAGES = frozenset({SHADOW_VERIFY_US})


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    stage: str
    allowed: bool
    reasons: tuple[str, ...] = ()


def storage_writable(path: str | Path) -> bool:
    target = Path(path)
    return target.is_dir() and os.access(target, os.W_OK | os.X_OK)


def evaluate(
    stage: str, config: ExposureRuntimeConfig, *, check_storage: bool = False
) -> ActivationDecision:
    if stage not in STAGES:
        raise ValueError(f"unknown_stage:{stage}")
    if stage not in INSTALLED_STAGES:
        return ActivationDecision(stage, False, ("not_installed",))
    reasons = []
    if config.research_mode == ResearchMode.DISABLED:
        reasons.append("research_disabled")
    elif config.research_mode == ResearchMode.LIVE:
        # Live publication is not part of this slice; shadow is required.
        reasons.append("live_mode_not_installed")
    if not config.route_approved("text"):
        reasons.append("text_route_not_enabled")
    if not config.subscription_key_present:
        reasons.append("subscription_credentials_missing")
    if config.daily_request_limit is None:
        reasons.append("allocation_not_configured")
    if not config.sec_user_agent.strip():
        reasons.append("sec_user_agent_not_configured")
    if check_storage and not storage_writable(config.document_store):
        reasons.append("storage_not_writable")
    return ActivationDecision(stage, not reasons, tuple(reasons))


def evaluate_all(
    config: ExposureRuntimeConfig, *, check_storage: bool = False
) -> list[ActivationDecision]:
    return [evaluate(stage, config, check_storage=check_storage) for stage in STAGES]


__all__ = (
    "STAGES",
    "ActivationDecision",
    "evaluate",
    "evaluate_all",
    "storage_writable",
)
