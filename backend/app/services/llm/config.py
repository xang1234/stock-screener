"""
LLM Configuration - Model configs and fallback chains for LiteLLM.

Defines provider-prefixed models and fallback strategies for different use cases.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str  # LiteLLM format: provider/model
    temperature: float = 0.3
    max_tokens: int = 4000
    top_p: float = 1.0
    # Provider-specific parameters
    extra_params: Dict = field(default_factory=dict)


@dataclass
class ModelPreset:
    """Preset configuration for a use case (chatbot, research, extraction)."""
    primary: ModelConfig
    fallbacks: List[ModelConfig] = field(default_factory=list)


# =============================================================================
# Model Definitions with LiteLLM provider prefixes
# =============================================================================

# Ollama Models (Local) - Use ollama_chat/ prefix for chat completions
OLLAMA_QWEN3_4B = ModelConfig(
    model_id="ollama_chat/qwen3:4b",
    temperature=0.2,
    max_tokens=2000,
)

OLLAMA_QWEN3_14B = ModelConfig(
    model_id="ollama_chat/qwen3:14b",
    temperature=0.2,
    max_tokens=2000,
)

OLLAMA_LLAMA3_8B = ModelConfig(
    model_id="ollama_chat/llama3.1:8b",
    temperature=0.2,
    max_tokens=2000,
)

OLLAMA_MISTRAL_7B = ModelConfig(
    model_id="ollama_chat/mistral:7b",
    temperature=0.2,
    max_tokens=2000,
)

OLLAMA_QWEN25_32B = ModelConfig(
    model_id="ollama_chat/qwen2.5:32b",
    temperature=0.2,
    max_tokens=2000,
)

# Groq Models - Qwen 3 32B (Primary: 128K context, tool use support)
GROQ_QWEN3_32B = ModelConfig(
    model_id="groq/qwen/qwen3-32b",
    temperature=0.6,  # Recommended for Qwen reasoning
    max_tokens=8000,
    top_p=0.95,
)

# Groq Models - Llama 3.3 70B (Fallback: strong general purpose)
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

GROQ_MIXTRAL = ModelConfig(
    model_id="groq/mixtral-8x7b-32768",
    temperature=0.3,
    max_tokens=8000,
)

# DeepSeek Models
DEEPSEEK_CHAT = ModelConfig(
    model_id="deepseek/deepseek-chat",
    temperature=0.3,
    max_tokens=8000,
)

DEEPSEEK_REASONER = ModelConfig(
    model_id="deepseek/deepseek-reasoner",
    temperature=0.3,
    max_tokens=8000,
)

# Together AI Models
TOGETHER_LLAMA_70B = ModelConfig(
    model_id="together_ai/meta-llama/Llama-3-70b-chat-hf",
    temperature=0.3,
    max_tokens=4000,
)

TOGETHER_QWEN_72B = ModelConfig(
    model_id="together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    temperature=0.3,
    max_tokens=8000,
)

# OpenRouter Models (access to multiple providers)
OPENROUTER_CLAUDE_SONNET = ModelConfig(
    model_id="openrouter/anthropic/claude-3.5-sonnet",
    temperature=0.3,
    max_tokens=8000,
)

OPENROUTER_GPT4O = ModelConfig(
    model_id="openrouter/openai/gpt-4o",
    temperature=0.3,
    max_tokens=8000,
)


# =============================================================================
# Use Case Presets - Qwen 3 32B as Primary
# =============================================================================

# Chatbot with tool calling - needs strong reasoning and tool support
CHATBOT_PRESET = ModelPreset(
    primary=GROQ_QWEN3_32B,
    fallbacks=[
        GROQ_LLAMA_70B,
        GROQ_MIXTRAL,
    ]
)

# Research agents - needs good reasoning for planning and synthesis
RESEARCH_PRESET = ModelPreset(
    primary=GROQ_QWEN3_32B,
    fallbacks=[
        GROQ_LLAMA_70B,
        DEEPSEEK_CHAT,
    ]
)

# Theme extraction - needs structured output (JSON)
EXTRACTION_PRESET = ModelPreset(
    primary=GROQ_QWEN3_32B,
    fallbacks=[
        GROQ_LLAMA_70B,
        GROQ_MIXTRAL,
    ]
)

# Report writing - needs long context and good prose
REPORT_PRESET = ModelPreset(
    primary=GROQ_QWEN3_32B,
    fallbacks=[
        GROQ_LLAMA_70B,
        DEEPSEEK_CHAT,
    ]
)

# Compression/summarization - fast, cost-effective
COMPRESSION_PRESET = ModelPreset(
    primary=GROQ_LLAMA_70B,
    fallbacks=[
        GROQ_LLAMA_8B,
        GROQ_MIXTRAL,
    ]
)

# Local extraction preset - for use with Ollama
LOCAL_EXTRACTION_PRESET = ModelPreset(
    primary=OLLAMA_QWEN3_14B,
    fallbacks=[
        OLLAMA_LLAMA3_8B,
        OLLAMA_MISTRAL_7B,
    ]
)


# =============================================================================
# Provider Environment Variable Mapping
# =============================================================================

PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together_ai": "TOGETHER_API_KEY",  # Also: TOGETHERAI_API_KEY
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "OLLAMA_API_BASE",  # Default: http://localhost:11434
}


# =============================================================================
# Available Models Registry - for UI selection
# =============================================================================

AVAILABLE_MODELS = [
    # Cloud models (Groq)
    {"id": "groq/llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Groq)", "provider": "groq", "category": "cloud"},
    {"id": "groq/qwen/qwen3-32b", "name": "Qwen 3 32B (Groq)", "provider": "groq", "category": "cloud"},
    {"id": "groq/llama-3.1-8b-instant", "name": "Llama 3.1 8B (Groq)", "provider": "groq", "category": "cloud"},
    {"id": "groq/mixtral-8x7b-32768", "name": "Mixtral 8x7B (Groq)", "provider": "groq", "category": "cloud"},

    # Cloud models (DeepSeek)
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat", "provider": "deepseek", "category": "cloud"},
    {"id": "deepseek/deepseek-reasoner", "name": "DeepSeek Reasoner", "provider": "deepseek", "category": "cloud"},

    # Local models (Ollama)
    {"id": "ollama_chat/qwen3:4b", "name": "Qwen 3 4B (Local)", "provider": "ollama", "category": "local"},
    {"id": "ollama_chat/qwen3:14b", "name": "Qwen 3 14B (Local)", "provider": "ollama", "category": "local"},
    {"id": "ollama_chat/qwen2.5:32b", "name": "Qwen 2.5 32B (Local)", "provider": "ollama", "category": "local"},
    {"id": "ollama_chat/llama3.1:8b", "name": "Llama 3.1 8B (Local)", "provider": "ollama", "category": "local"},
    {"id": "ollama_chat/mistral:7b", "name": "Mistral 7B (Local)", "provider": "ollama", "category": "local"},
]


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
    """Get all models by category (cloud/local)."""
    return [m for m in AVAILABLE_MODELS if m["category"] == category]


def get_preset_for_use_case(use_case: str) -> ModelPreset:
    """Get the model preset for a given use case."""
    presets = {
        "chatbot": CHATBOT_PRESET,
        "research": RESEARCH_PRESET,
        "extraction": EXTRACTION_PRESET,
        "report": REPORT_PRESET,
        "compression": COMPRESSION_PRESET,
    }
    return presets.get(use_case, CHATBOT_PRESET)


def get_fallback_chain(preset: ModelPreset) -> List[str]:
    """Get the list of model IDs for fallback."""
    models = [preset.primary.model_id]
    for fallback in preset.fallbacks:
        models.append(fallback.model_id)
    return models


def get_model_params(model_config: ModelConfig, **overrides) -> Dict:
    """Get parameters for a model, applying any overrides."""
    params = {
        "model": model_config.model_id,
        "temperature": model_config.temperature,
        "max_tokens": model_config.max_tokens,
    }
    if model_config.top_p != 1.0:
        params["top_p"] = model_config.top_p

    # Add extra params (provider-specific)
    params.update(model_config.extra_params)

    # Apply overrides
    params.update(overrides)

    return params
