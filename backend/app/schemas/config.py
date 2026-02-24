"""Pydantic schemas for application configuration API endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class LLMModelUpdate(BaseModel):
    """Request body for updating LLM model selection."""

    model_id: str
    use_case: str = "extraction"  # "extraction" or "merge"


class OllamaSettings(BaseModel):
    """Request body for updating Ollama settings."""

    api_base: str


class LLMConfigResponse(BaseModel):
    """Response for LLM configuration."""

    extraction: dict
    merge: dict
    ollama_status: str
    ollama_api_base: str
    available_models: list


class ThemePolicyMatcherConfig(BaseModel):
    match_default_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fuzzy_attach_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fuzzy_review_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fuzzy_ambiguity_margin: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    embedding_attach_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    embedding_review_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    embedding_ambiguity_margin: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ThemePolicyLifecycleConfig(BaseModel):
    promotion_min_mentions_7d: Optional[int] = Field(default=None, ge=0, le=200)
    promotion_min_source_diversity_7d: Optional[int] = Field(default=None, ge=0, le=50)
    promotion_min_avg_confidence_30d: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    promotion_min_persistence_days: Optional[int] = Field(default=None, ge=0, le=60)
    dormancy_inactivity_days: Optional[int] = Field(default=None, ge=1, le=365)
    dormancy_min_mentions_30d: Optional[int] = Field(default=None, ge=0, le=500)
    dormancy_min_silence_days: Optional[int] = Field(default=None, ge=0, le=365)
    reactivation_min_mentions_7d: Optional[int] = Field(default=None, ge=0, le=200)
    reactivation_min_source_diversity_7d: Optional[int] = Field(default=None, ge=0, le=50)
    reactivation_min_avg_confidence_30d: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relationship_subset_overlap_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relationship_related_jaccard_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relationship_min_overlap_constituents: Optional[int] = Field(default=None, ge=1, le=200)


class ThemePolicyUpdateRequest(BaseModel):
    pipeline: str = Field(..., pattern=r"^(technical|fundamental)$")
    matcher: Optional[ThemePolicyMatcherConfig] = None
    lifecycle: Optional[ThemePolicyLifecycleConfig] = None
    note: Optional[str] = None
    mode: str = Field(default="preview", pattern=r"^(preview|stage|apply)$")


class ThemePolicyRevertRequest(BaseModel):
    pipeline: str = Field(..., pattern=r"^(technical|fundamental)$")
    version_id: str
    note: Optional[str] = None


class ThemePolicyVersionSummary(BaseModel):
    version_id: str
    pipeline: str
    updated_at: str
    updated_by: str
    note: Optional[str] = None


class ThemePolicyConfigResponse(BaseModel):
    pipeline: str
    active_version_id: Optional[str] = None
    active_updated_at: Optional[str] = None
    active_updated_by: Optional[str] = None
    defaults: dict
    overrides: dict
    effective: dict
    staged: Optional[dict] = None
    history: list[ThemePolicyVersionSummary]


class ThemePolicyUpdateResponse(BaseModel):
    status: str
    pipeline: str
    mode: str
    version_id: Optional[str] = None
    preview_effective: dict
    diff_keys: list[str]
