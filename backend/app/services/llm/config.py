"""LLM model registry and use-case presets for sanctioned providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific provider model."""

    model_id: str
    temperature: float = 0.3
    max_tokens: int = 4000
    top_p: float = 1.0
    extra_params: Dict = field(default_factory=dict)


@dataclass
class ModelPreset:
    """Preset configuration for a use case (chatbot, research, extraction)."""

    primary: ModelConfig
    fallbacks: List[ModelConfig] = field(default_factory=list)


# Supported cloud models
MINIMAX_M27 = ModelConfig(
    model_id="minimax/MiniMax-M2.7",
    temperature=0.2,
    max_tokens=4000,
)

ZAI_GLM_47_FLASH = ModelConfig(
    model_id="openai/glm-4.7-flash",
    temperature=0.2,
    max_tokens=4000,
)

GROQ_QWEN3_32B = ModelConfig(
    model_id="groq/qwen/qwen3-32b",
    temperature=0.6,
    max_tokens=8000,
    top_p=0.95,
)

GROQ_LLAMA_70B = ModelConfig(
    model_id="groq/llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=8000,
)

GROQ_LLAMA_8B = ModelConfig(
    model_id="groq/llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=4000,
)


# Use-case presets
CHATBOT_PRESET = ModelPreset(
    primary=GROQ_QWEN3_32B,
    fallbacks=[GROQ_LLAMA_70B],
)

RESEARCH_PRESET = ModelPreset(
    primary=GROQ_QWEN3_32B,
    fallbacks=[GROQ_LLAMA_70B],
)

EXTRACTION_PRESET = ModelPreset(
    primary=MINIMAX_M27,
    fallbacks=[ZAI_GLM_47_FLASH],
)

MERGE_PRESET = ModelPreset(
    primary=MINIMAX_M27,
    fallbacks=[ZAI_GLM_47_FLASH],
)

REPORT_PRESET = ModelPreset(
    primary=GROQ_QWEN3_32B,
    fallbacks=[GROQ_LLAMA_70B],
)

COMPRESSION_PRESET = ModelPreset(
    primary=GROQ_LLAMA_70B,
    fallbacks=[GROQ_LLAMA_8B],
)


PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "zai": "ZAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


AVAILABLE_MODELS = [
    {"id": "minimax/MiniMax-M2.7", "name": "MiniMax M2.7 (Minimax)", "provider": "minimax", "category": "cloud"},
    {"id": "openai/glm-4.7-flash", "name": "GLM-4.7-Flash (Z.AI)", "provider": "zai", "category": "cloud"},
    {"id": "groq/qwen/qwen3-32b", "name": "Qwen 3 32B (Groq)", "provider": "groq", "category": "cloud"},
    {"id": "groq/llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Groq)", "provider": "groq", "category": "cloud"},
    {"id": "groq/llama-3.1-8b-instant", "name": "Llama 3.1 8B (Groq)", "provider": "groq", "category": "cloud"},
]


SUPPORTED_MODELS_BY_USE_CASE: dict[str, set[str]] = {
    "chatbot": {
        "groq/qwen/qwen3-32b",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
    },
    "research": {
        "groq/qwen/qwen3-32b",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
    },
    "extraction": {
        "minimax/MiniMax-M2.7",
        "openai/glm-4.7-flash",
    },
    "merge": {
        "minimax/MiniMax-M2.7",
        "openai/glm-4.7-flash",
    },
}


DEFAULT_MODEL_BY_USE_CASE: dict[str, str] = {
    "chatbot": "groq/qwen/qwen3-32b",
    "research": "groq/qwen/qwen3-32b",
    "extraction": "minimax/MiniMax-M2.7",
    "merge": "minimax/MiniMax-M2.7",
}


def get_model_by_id(model_id: str) -> Optional[Dict]:
    """Get model info by ID."""
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model
    return None


def get_models_by_provider(provider: str) -> List[Dict]:
    """Get all models for a specific provider."""
    return [m for m in AVAILABLE_MODELS if m["provider"] == provider]


def get_models_by_category(category: str) -> List[Dict]:
    """Get all models by category."""
    return [m for m in AVAILABLE_MODELS if m["category"] == category]


def get_preset_for_use_case(use_case: str) -> ModelPreset:
    """Get model preset for the selected use case."""
    presets = {
        "chatbot": CHATBOT_PRESET,
        "research": RESEARCH_PRESET,
        "extraction": EXTRACTION_PRESET,
        "merge": MERGE_PRESET,
        "report": REPORT_PRESET,
        "compression": COMPRESSION_PRESET,
    }
    return presets.get(use_case, CHATBOT_PRESET)


def get_fallback_chain(preset: ModelPreset) -> List[str]:
    """Get ordered fallback chain for a model preset."""
    models = [preset.primary.model_id]
    for fallback in preset.fallbacks:
        models.append(fallback.model_id)
    return models


def get_model_params(model_config: ModelConfig, **overrides) -> Dict:
    """Build LiteLLM params for a model config."""
    params = {
        "model": model_config.model_id,
        "temperature": model_config.temperature,
        "max_tokens": model_config.max_tokens,
    }
    if model_config.top_p != 1.0:
        params["top_p"] = model_config.top_p
    params.update(model_config.extra_params)
    params.update(overrides)
    return params


def is_model_supported_for_use_case(*, model_id: str, use_case: str) -> bool:
    """Return True when model is sanctioned for the requested use case."""
    allowed = SUPPORTED_MODELS_BY_USE_CASE.get(use_case)
    if not allowed:
        return False
    return model_id in allowed
