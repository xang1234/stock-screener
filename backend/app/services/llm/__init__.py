"""
LLM Service Package - Unified LLM interface using LiteLLM.

This package provides a unified interface for interacting with multiple LLM
providers (Groq, DeepSeek, Together AI, OpenRouter) with automatic retry,
fallback handling, and both sync/async support.

Usage:
    from app.services.llm import LLMService

    # Create service for a specific use case
    llm = LLMService(use_case="chatbot")

    # Async completion
    response = await llm.completion(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[...],  # Optional tool definitions
    )

    # Extract content
    content = LLMService.extract_content(response)
    tool_calls = LLMService.extract_tool_calls(response)

    # Streaming (must await first, then iterate)
    stream = await llm.completion(messages=[...], stream=True)
    async for chunk in stream:
        print(chunk)

    # Sync completion (blocking)
    response = llm.completion_sync(messages=[...])
"""

from .llm_service import (
    LLMService,
    LLMError,
    LLMRateLimitError,
    LLMQuotaExceededError,
    LLMContextWindowError,
    quick_completion,
)
from .groq_key_manager import (
    GroqKeyManager,
    get_groq_key_manager,
)
from .zai_key_manager import (
    ZAIKeyManager,
    get_zai_key_manager,
)
from .config import (
    ModelConfig,
    ModelPreset,
    get_preset_for_use_case,
    get_model_params,
    get_fallback_chain,
    get_model_by_id,
    get_models_by_provider,
    get_models_by_category,
    AVAILABLE_MODELS,
    CHATBOT_PRESET,
    RESEARCH_PRESET,
    EXTRACTION_PRESET,
    REPORT_PRESET,
    COMPRESSION_PRESET,
    LOCAL_EXTRACTION_PRESET,
    GROQ_QWEN3_32B,
    GROQ_LLAMA_70B,
    GROQ_MIXTRAL,
    DEEPSEEK_CHAT,
    OLLAMA_QWEN3_14B,
    OLLAMA_LLAMA3_8B,
    OLLAMA_MISTRAL_7B,
    OLLAMA_QWEN25_32B,
)

__all__ = [
    # Main service
    "LLMService",
    # Key manager
    "GroqKeyManager",
    "get_groq_key_manager",
    "ZAIKeyManager",
    "get_zai_key_manager",
    # Exceptions
    "LLMError",
    "LLMRateLimitError",
    "LLMQuotaExceededError",
    "LLMContextWindowError",
    # Quick helper
    "quick_completion",
    # Config classes
    "ModelConfig",
    "ModelPreset",
    # Config functions
    "get_preset_for_use_case",
    "get_model_params",
    "get_fallback_chain",
    "get_model_by_id",
    "get_models_by_provider",
    "get_models_by_category",
    # Available models registry
    "AVAILABLE_MODELS",
    # Presets
    "CHATBOT_PRESET",
    "RESEARCH_PRESET",
    "EXTRACTION_PRESET",
    "REPORT_PRESET",
    "COMPRESSION_PRESET",
    "LOCAL_EXTRACTION_PRESET",
    # Cloud model configs
    "GROQ_QWEN3_32B",
    "GROQ_LLAMA_70B",
    "GROQ_MIXTRAL",
    "DEEPSEEK_CHAT",
    # Local model configs (Ollama)
    "OLLAMA_QWEN3_14B",
    "OLLAMA_LLAMA3_8B",
    "OLLAMA_MISTRAL_7B",
    "OLLAMA_QWEN25_32B",
]
