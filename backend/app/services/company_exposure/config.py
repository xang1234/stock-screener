"""Runtime configuration for company-exposure research.

Values come from application settings (environment). Every default is the
non-spending one: research disabled, paid search off, subscription routes
not approved and no local allocation. Nothing here reads or exposes secret
values; only their presence is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domain.company_exposure.contracts import (
    LLMBillingMode,
    ResearchLimits,
    ResearchMode,
)

SUBSCRIPTION_PROVIDER = "opencode-go"
SUBSCRIPTION_MODEL = "kimi-k2.6"
SUBSCRIPTION_MODEL_IDENTITY = f"{SUBSCRIPTION_PROVIDER}/{SUBSCRIPTION_MODEL}"
TEXT = "text"
VISION = "vision"


@dataclass(frozen=True, slots=True)
class ExposureRuntimeConfig:
    research_mode: ResearchMode = ResearchMode.DISABLED
    paid_search_enabled: bool = False
    search_provider: str = "none"
    llm_billing_mode: LLMBillingMode = LLMBillingMode.SUBSCRIPTION
    text_route_enabled: bool = False
    vision_route_enabled: bool = False
    daily_request_limit: int | None = None
    daily_token_limit: int | None = None
    allocation_timezone: str = "UTC"
    subscription_key_present: bool = False
    document_store: str = "data/exposure-evidence"
    storage_max_bytes: int = ResearchLimits().storage_max_bytes
    storage_min_free_bytes: int = ResearchLimits().storage_min_free_bytes
    sec_user_agent: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "research_mode", ResearchMode(self.research_mode))
        object.__setattr__(
            self, "llm_billing_mode", LLMBillingMode(self.llm_billing_mode)
        )
        try:
            ZoneInfo(self.allocation_timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("invalid_allocation_timezone") from None
        for name in ("daily_request_limit", "daily_token_limit"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def limits(self) -> ResearchLimits:
        return ResearchLimits(
            research_mode=self.research_mode,
            paid_search_enabled=self.paid_search_enabled,
            llm_billing_mode=self.llm_billing_mode,
            storage_max_bytes=self.storage_max_bytes,
            storage_min_free_bytes=self.storage_min_free_bytes,
            daily_request_allocation=self.daily_request_limit,
            daily_token_allocation=self.daily_token_limit,
        )

    def route_approved(self, capability: str) -> bool:
        if capability == TEXT:
            return self.text_route_enabled
        if capability == VISION:
            return self.vision_route_enabled
        return False

    def allocation_period(self, at: datetime) -> tuple[str, datetime]:
        """The daily allocation period containing ``at`` and its end (UTC)."""

        if at.tzinfo is None:
            raise ValueError("allocation period requires a timezone-aware time")
        zone = ZoneInfo(self.allocation_timezone)
        local = at.astimezone(zone)
        start = datetime.combine(local.date(), time.min, tzinfo=zone)
        return local.date().isoformat(), start + timedelta(days=1)

    def public_status(self) -> dict:
        """Operator-visible configuration; secret values are never included."""

        return {
            "research_mode": self.research_mode.value,
            "paid_search_enabled": self.paid_search_enabled,
            "search_provider": self.search_provider,
            "llm_billing_mode": self.llm_billing_mode.value,
            "subscription_route": SUBSCRIPTION_MODEL_IDENTITY,
            "subscription_key_present": self.subscription_key_present,
            "text_route_enabled": self.text_route_enabled,
            "vision_route_enabled": self.vision_route_enabled,
            "daily_request_limit": self.daily_request_limit,
            "daily_token_limit": self.daily_token_limit,
            "allocation_timezone": self.allocation_timezone,
            "provider_remaining_allowance": "unknown",
            "sec_user_agent_configured": bool(self.sec_user_agent.strip()),
        }


def load_config(settings=None) -> ExposureRuntimeConfig:
    if settings is None:
        from app.config import settings as app_settings

        settings = app_settings
    return ExposureRuntimeConfig(
        research_mode=getattr(settings, "exposure_research_mode", "disabled"),
        paid_search_enabled=bool(getattr(settings, "exposure_paid_search_enabled", False)),
        search_provider=getattr(settings, "exposure_search_provider", "none"),
        llm_billing_mode=getattr(settings, "exposure_llm_billing_mode", "subscription"),
        text_route_enabled=bool(getattr(settings, "exposure_llm_text_route_enabled", False)),
        vision_route_enabled=bool(
            getattr(settings, "exposure_llm_vision_route_enabled", False)
        ),
        daily_request_limit=getattr(settings, "exposure_llm_daily_request_limit", None),
        daily_token_limit=getattr(settings, "exposure_llm_daily_token_limit", None),
        allocation_timezone=getattr(settings, "exposure_allocation_timezone", "UTC"),
        subscription_key_present=bool(
            (getattr(settings, "opencode_go_api_key", "") or "").strip()
        ),
        document_store=getattr(settings, "exposure_document_store", "data/exposure-evidence"),
        storage_max_bytes=getattr(
            settings, "exposure_storage_max_bytes", ResearchLimits().storage_max_bytes
        ),
        storage_min_free_bytes=getattr(
            settings,
            "exposure_storage_min_free_bytes",
            ResearchLimits().storage_min_free_bytes,
        ),
        sec_user_agent=getattr(settings, "exposure_sec_user_agent", "") or "",
    )
