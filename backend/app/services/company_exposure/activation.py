"""Activation check for the installed stage (plan Task 27; spec §18).

This build installs one stage, the US verify-only shadow slice. The check
lists every missing prerequisite; later stages are not part of this build.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.domain.company_exposure.contracts import ResearchMode
from app.services.company_exposure.config import ExposureRuntimeConfig

SHADOW_VERIFY_US = "shadow_verify_us"


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    stage: str
    allowed: bool
    reasons: tuple[str, ...] = ()


def storage_writable(path: str | Path) -> bool:
    target = Path(path)
    return target.is_dir() and os.access(target, os.W_OK | os.X_OK)


def evaluate(
    config: ExposureRuntimeConfig, *, check_storage: bool = False
) -> ActivationDecision:
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
    return ActivationDecision(SHADOW_VERIFY_US, not reasons, tuple(reasons))
