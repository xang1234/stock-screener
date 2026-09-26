"""Company-exposure research settings (``EXPOSURE_*`` environment variables).

Kept apart from the application ``Settings`` so the feature's configuration
lives with its own types. Research is disabled by default and a credential
alone never enables spending (docs/runbooks/company-exposure-map.md).
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.company_exposure.contracts import LLMBillingMode, ResearchMode

GIB = 1024 * 1024 * 1024


class ExposureSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EXPOSURE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    research_mode: ResearchMode = ResearchMode.DISABLED
    paid_search_enabled: bool = False
    search_provider: str = "none"
    llm_billing_mode: LLMBillingMode = LLMBillingMode.SUBSCRIPTION
    # Operator-approved OpenCode Go routes; both off until explicitly enabled.
    llm_text_route_enabled: bool = False
    llm_vision_route_enabled: bool = False
    # Local allocation ceilings; no guessed default. Unset blocks dispatch.
    llm_daily_request_limit: int | None = None
    llm_daily_token_limit: int | None = None
    allocation_timezone: str = "UTC"
    document_store: str = "data/exposure-evidence"
    storage_max_bytes: int = 5 * GIB
    storage_min_free_bytes: int = GIB
    # Identifying User-Agent required by SEC fair-access policy.
    sec_user_agent: str = ""

    @field_validator("llm_daily_request_limit", "llm_daily_token_limit", mode="before")
    @classmethod
    def _blank_limit_is_unset(cls, value):
        # Compose passes "" for unset limits; unset means "not configured".
        if isinstance(value, str) and not value.strip():
            return None
        return value
