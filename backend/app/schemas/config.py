"""Pydantic schemas for application configuration API endpoints."""

from pydantic import BaseModel


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
