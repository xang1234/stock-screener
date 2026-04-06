"""
LLM Service - Unified interface wrapping LiteLLM for multi-provider support.

Provides:
- Async and sync completion methods
- Automatic fallback handling between models
- Streaming support
- Retry logic with exponential backoff
- Support for Groq, Z.AI, DeepSeek, Together AI, OpenRouter, and more
"""
import asyncio
import logging
import os
import random
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
from litellm import acompletion, completion
from litellm.exceptions import (
    RateLimitError,
    ContextWindowExceededError,
    APIConnectionError,
    APIError,
)

from ...config import settings
from .config import (
    ModelPreset,
    get_preset_for_use_case,
    get_model_params,
    get_fallback_chain,
)
from .groq_key_manager import GroqKeyManager, get_groq_key_manager
from .zai_key_manager import ZAIKeyManager, get_zai_key_manager

logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.drop_params = True  # Drop unsupported params instead of erroring
litellm.set_verbose = False  # Set to True for debugging

_ZAI_API_BASE_DEFAULT = "https://api.z.ai/api/paas/v4"


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded after retries."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaExceededError(LLMRateLimitError):
    """Raised when provider hard quota is exhausted (for example Groq TPD)."""


class LLMContextWindowError(LLMError):
    """Raised when request exceeds context window."""
    def __init__(self, message: str, tokens_used: Optional[int] = None, token_limit: Optional[int] = None):
        super().__init__(message)
        self.tokens_used = tokens_used
        self.token_limit = token_limit


class LLMService:
    """
    Unified LLM service wrapping LiteLLM for multi-provider support.

    Usage:
        llm = LLMService(use_case="chatbot")
        response = await llm.completion(messages=[...], tools=[...])

        # Or with streaming
        stream = await llm.completion(messages=[...], stream=True)
        async for chunk in stream:
            print(chunk)
    """

    def __init__(
        self,
        use_case: str = "chatbot",
        preset: Optional[ModelPreset] = None,
    ):
        """
        Initialize LLM service.

        Args:
            use_case: One of "chatbot", "research", "extraction", "report", "compression"
            preset: Optional custom ModelPreset (overrides use_case)
        """
        self.preset = preset or get_preset_for_use_case(use_case)
        self._setup_api_keys()
        self._groq_key_manager = get_groq_key_manager()
        self._zai_key_manager = get_zai_key_manager()

    def _setup_api_keys(self):
        """Set up API keys from settings or environment."""
        # Groq
        groq_key = getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        # Z.AI
        zai_keys = list(getattr(settings, "zai_api_keys_list", []) or [])
        if zai_keys:
            os.environ["ZAI_API_KEYS"] = ",".join(zai_keys)
            if len(zai_keys) == 1:
                os.environ["ZAI_API_KEY"] = zai_keys[0]
        else:
            zai_key = getattr(settings, "zai_api_key", None) or os.environ.get("ZAI_API_KEY")
            if zai_key:
                os.environ["ZAI_API_KEY"] = zai_key
        zai_base = getattr(settings, "zai_api_base", None) or os.environ.get("ZAI_API_BASE")
        if zai_base:
            os.environ["ZAI_API_BASE"] = zai_base

        # DeepSeek
        deepseek_key = getattr(settings, "deepseek_api_key", None) or os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            os.environ["DEEPSEEK_API_KEY"] = deepseek_key

        # Together AI
        together_key = getattr(settings, "together_api_key", None) or os.environ.get("TOGETHER_API_KEY")
        if together_key:
            os.environ["TOGETHER_API_KEY"] = together_key
            os.environ["TOGETHERAI_API_KEY"] = together_key

        # OpenRouter
        openrouter_key = getattr(settings, "openrouter_api_key", None) or os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key

        # Minimax — cache resolved values for per-request use
        self._minimax_api_key = getattr(settings, "minimax_api_key", None) or os.environ.get("MINIMAX_API_KEY")
        self._minimax_api_base = (
            getattr(settings, "minimax_api_base", None)
            or os.environ.get("MINIMAX_API_BASE")
            or "https://api.minimax.io/v1"
        )
        if self._minimax_api_key:
            os.environ["MINIMAX_API_KEY"] = self._minimax_api_key
        elif self.preset.primary.model_id.startswith("minimax/"):
            logger.warning(
                "MINIMAX_API_KEY not set — extraction will fall back to next provider in chain"
            )

    async def completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        allow_fallbacks: bool = True,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
        num_retries: int = 3,
        **kwargs
    ) -> Any:
        """
        Async completion with automatic fallbacks.

        Args:
            messages: List of message dicts with role and content
            model: Override model (uses preset primary if not specified)
            tools: List of tool definitions for function calling
            tool_choice: "auto", "none", or specific tool
            temperature: Override temperature
            max_tokens: Override max tokens
            response_format: Response format (e.g., {"type": "json_object"})
            stream: Return streaming generator if True
            num_retries: Number of retries per model
            **kwargs: Additional parameters

        Returns:
            ChatCompletion or AsyncGenerator if streaming
        """
        if stream:
            return self._completion_stream(
                messages=messages,
                model=model,
                allow_fallbacks=allow_fallbacks,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                num_retries=num_retries,
                **kwargs
            )

        # Build parameters from preset
        model_config = self.preset.primary
        params = get_model_params(model_config)

        # Apply overrides
        if model:
            params["model"] = model

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        params["messages"] = messages

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        if response_format:
            params["response_format"] = response_format

        # Build fallback list
        fallback_models = self._resolve_fallback_models(
            primary_model=str(params["model"]),
            allow_fallbacks=allow_fallbacks,
        )

        return await self._call_with_fallbacks(
            params=params,
            fallbacks=fallback_models,
            num_retries=num_retries,
        )

    async def _call_with_fallbacks(
        self,
        params: Dict,
        fallbacks: List[str],
        num_retries: int = 3,
    ) -> Any:
        """Execute completion with fallback handling."""
        all_models = [params["model"]] + fallbacks
        last_error = None

        for index, model in enumerate(all_models):
            model_params = dict(params)
            model_params["model"] = model
            try:
                return await self._call_with_retry(model_params, num_retries)
            except LLMQuotaExceededError as e:
                # Fail fast when quota is exhausted and all remaining fallbacks
                # are on the same provider (e.g., Groq-only extraction presets).
                remaining_models = all_models[index + 1 :]
                if self._should_fail_fast_on_quota(model, remaining_models):
                    raise
                logger.warning(f"Model {model} quota exhausted: {e}")
                last_error = e
                continue
            except LLMContextWindowError:
                raise
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                last_error = e
                continue

        raise LLMError(f"All models failed. Last error: {last_error}")

    @staticmethod
    def _is_zai_model(model: str) -> bool:
        """Return True when a model should route through the Z.AI OpenAI-compatible endpoint."""
        return model.startswith("openai/glm-")

    @staticmethod
    def _is_minimax_model(model: str) -> bool:
        """Return True when a model should route through the Minimax endpoint."""
        return model.startswith("minimax/")

    def _resolve_fallback_models(self, *, primary_model: str, allow_fallbacks: bool) -> List[str]:
        """Resolve fallback models for a request, excluding duplicates of the active model."""
        if not allow_fallbacks:
            return []
        return [
            candidate
            for candidate in get_fallback_chain(self.preset)[1:]
            if candidate != primary_model
        ]

    def _get_provider_key_manager(
        self,
        model: str,
    ) -> tuple[Optional[str], Optional[GroqKeyManager | ZAIKeyManager]]:
        """Return the managed provider label and key manager for a model."""
        if model.startswith("groq/"):
            manager = getattr(self, "_groq_key_manager", None)
            if manager is None:
                manager = get_groq_key_manager()
            return "groq", manager
        if self._is_zai_model(model):
            manager = getattr(self, "_zai_key_manager", None)
            if manager is None:
                manager = get_zai_key_manager()
            return "zai", manager
        return None, None

    def _apply_provider_overrides(
        self,
        params: Dict[str, Any],
    ) -> tuple[Optional[str], Optional[str], Optional[GroqKeyManager | ZAIKeyManager]]:
        """Apply per-provider request overrides for the selected model."""
        model = str(params.get("model", "") or "")
        provider_name, key_manager = self._get_provider_key_manager(model)
        provider_key = None

        is_zai = self._is_zai_model(model)
        is_minimax = self._is_minimax_model(model)

        if key_manager and len(key_manager) > 0:
            provider_key = key_manager.get_key()
        elif is_zai:
            provider_key = getattr(settings, "zai_api_key", None) or os.environ.get("ZAI_API_KEY")
        elif is_minimax:
            provider_key = self._minimax_api_key
            provider_name = "minimax"

        if provider_key:
            params["api_key"] = provider_key

        if is_zai:
            zai_base = (
                getattr(settings, "zai_api_base", None)
                or os.environ.get("ZAI_API_BASE")
                or _ZAI_API_BASE_DEFAULT
            )
            params["api_base"] = zai_base
        elif is_minimax:
            params["api_base"] = self._minimax_api_base

        return provider_name, provider_key, key_manager

    def _report_provider_rate_limit(
        self,
        provider_name: Optional[str],
        provider_key: Optional[str],
        key_manager: Optional[GroqKeyManager | ZAIKeyManager],
        retry_after: Optional[float],
    ) -> None:
        """Report a provider rate limit when the request used a managed key."""
        if not provider_name or not provider_key or key_manager is None:
            return
        key_manager.report_rate_limit(provider_key, retry_after)

    async def _call_with_retry(
        self,
        params: Dict,
        num_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> Any:
        """Execute completion with exponential backoff retry."""
        last_error = None
        model = params.get("model", "")

        # For managed providers, get a key ONCE for this entire request.
        # The key stays the same for all retries (rotation happens on NEXT request).
        provider_name, provider_key, key_manager = self._apply_provider_overrides(params)

        for attempt in range(num_retries + 1):
            try:
                response = await acompletion(**params)
                return response

            except ContextWindowExceededError as e:
                raise LLMContextWindowError(str(e))

            except RateLimitError as e:
                last_error = e

                retry_after = self._extract_retry_after(e)
                self._report_provider_rate_limit(
                    provider_name,
                    provider_key,
                    key_manager,
                    retry_after,
                )

                if self._is_provider_quota_exhausted(e):
                    raise LLMQuotaExceededError(
                        f"Provider quota exhausted: {e}",
                        retry_after=retry_after,
                    )

                if attempt >= num_retries:
                    raise LLMRateLimitError(f"Rate limit exceeded after {num_retries} retries: {e}")

                delay = self._calculate_delay(attempt, base_delay, max_delay, e)
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{num_retries + 1}). Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

            except APIConnectionError as e:
                last_error = e
                if attempt >= num_retries:
                    raise LLMError(f"Connection error after {num_retries} retries: {e}")

                delay = self._calculate_delay(attempt, base_delay, max_delay / 2)
                logger.warning(f"Connection error (attempt {attempt + 1}/{num_retries + 1}). Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

            except APIError as e:
                error_msg = str(e).lower()

                # Check for context window errors
                if "context" in error_msg or "too long" in error_msg or "token" in error_msg:
                    raise LLMContextWindowError(str(e))

                # Check for rate limit
                if "rate" in error_msg or "429" in str(e):
                    last_error = e

                    retry_after = self._extract_retry_after(e)
                    self._report_provider_rate_limit(
                        provider_name,
                        provider_key,
                        key_manager,
                        retry_after,
                    )

                    if self._is_provider_quota_exhausted(e):
                        raise LLMQuotaExceededError(
                            f"Provider quota exhausted: {e}",
                            retry_after=retry_after,
                        )

                    if attempt >= num_retries:
                        raise LLMRateLimitError(f"Rate limit exceeded after {num_retries} retries: {e}")

                    delay = self._calculate_delay(attempt, base_delay, max_delay, e)
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{num_retries + 1}). Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue

                # Other API errors
                last_error = e
                if attempt >= num_retries:
                    raise LLMError(f"API error after {num_retries} retries: {e}")

                delay = self._calculate_delay(attempt, base_delay, max_delay / 2)
                logger.warning(f"API error (attempt {attempt + 1}/{num_retries + 1}). Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

            except Exception as e:
                last_error = e
                if attempt >= num_retries:
                    raise LLMError(f"Unexpected error after {num_retries} retries: {e}")

                delay = self._calculate_delay(attempt, base_delay, max_delay / 2)
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{num_retries + 1}). Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        raise LLMError(f"Unexpected error: {last_error}")

    def _should_fail_fast_on_quota(self, model: str, remaining_models: List[str]) -> bool:
        """Return True when quota exhaustion means remaining fallbacks are unlikely to help."""
        if not model.startswith("groq/"):
            return False
        if not remaining_models:
            return True
        return all(candidate.startswith("groq/") for candidate in remaining_models)

    def _is_provider_quota_exhausted(self, error: Exception) -> bool:
        """Detect hard quota exhaustion signals that should bypass retries."""
        error_text = str(error).lower()
        quota_markers = (
            "tokens per day",
            "tpd",
            "quota exceeded",
            "insufficient_quota",
            "billing hard limit",
            "need more tokens",
        )
        return any(marker in error_text for marker in quota_markers)

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after value from error message."""
        error_str = str(error).lower()
        retry_match = re.search(r'try again in (\d+(?:\.\d+)?)\s*(?:s|second)', error_str)
        if retry_match:
            return float(retry_match.group(1))
        retry_match2 = re.search(r'retry after (\d+(?:\.\d+)?)\s*(ms|s|second)', error_str)
        if retry_match2:
            value = float(retry_match2.group(1))
            unit = retry_match2.group(2)
            if unit == "ms":
                value = value / 1000
            return value
        return None

    def _calculate_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
        error: Optional[Exception] = None,
    ) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(base_delay * (2 ** attempt), max_delay)

        # Try to extract retry-after from error
        if error:
            error_str = str(error).lower()
            retry_match = re.search(r'try again in (\d+(?:\.\d+)?)\s*(?:s|second)', error_str)
            if retry_match:
                server_delay = float(retry_match.group(1))
                if server_delay <= max_delay:
                    delay = max(delay, server_delay)

        # Add jitter (0-10%)
        jitter = delay * 0.1 * random.random()
        return delay + jitter

    async def _completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        allow_fallbacks: bool = True,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        num_retries: int = 3,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        """Streaming completion with fallbacks."""
        model_config = self.preset.primary
        params = get_model_params(model_config)

        if model:
            params["model"] = model

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        params["messages"] = messages
        params["stream"] = True

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        if response_format:
            params["response_format"] = response_format

        all_models = [params["model"]] + self._resolve_fallback_models(
            primary_model=str(params["model"]),
            allow_fallbacks=allow_fallbacks,
        )
        last_error = None

        for current_model in all_models:
            current_params = dict(params)
            current_params["model"] = current_model

            provider_name, provider_key, key_manager = self._apply_provider_overrides(current_params)

            try:
                response = await acompletion(**current_params)
                async for chunk in response:
                    yield chunk
                return

            except Exception as e:
                if provider_key and ("rate" in str(e).lower() or "429" in str(e)):
                    retry_after = self._extract_retry_after(e)
                    self._report_provider_rate_limit(
                        provider_name,
                        provider_key,
                        key_manager,
                        retry_after,
                    )

                logger.warning(f"Streaming model {current_model} failed: {e}")
                last_error = e
                continue

        raise LLMError(f"All streaming models failed. Last error: {last_error}")

    def completion_sync(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        allow_fallbacks: bool = True,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        num_retries: int = 3,
        **kwargs
    ) -> Any:
        """
        Synchronous completion (blocking).

        Use this for sync contexts where async isn't available.
        """
        import time as time_module

        model_config = self.preset.primary
        params = get_model_params(model_config)

        if model:
            params["model"] = model

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        params["messages"] = messages

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        if response_format:
            params["response_format"] = response_format

        all_models = [params["model"]] + self._resolve_fallback_models(
            primary_model=str(params["model"]),
            allow_fallbacks=allow_fallbacks,
        )
        last_error = None

        for current_model in all_models:
            current_params = dict(params)
            current_params["model"] = current_model

            provider_name, provider_key, key_manager = self._apply_provider_overrides(current_params)

            for attempt in range(num_retries + 1):
                try:
                    return completion(**current_params)
                except RateLimitError as e:
                    retry_after = self._extract_retry_after(e)
                    self._report_provider_rate_limit(
                        provider_name,
                        provider_key,
                        key_manager,
                        retry_after,
                    )

                    if attempt >= num_retries:
                        last_error = e
                        break
                    delay = self._calculate_delay(attempt, 1.0, 60.0, e)
                    logger.warning(f"Rate limit hit. Retrying in {delay:.1f}s...")
                    time_module.sleep(delay)
                except APIError as e:
                    if "rate" in str(e).lower() or "429" in str(e):
                        retry_after = self._extract_retry_after(e)
                        self._report_provider_rate_limit(
                            provider_name,
                            provider_key,
                            key_manager,
                            retry_after,
                        )
                        if self._is_provider_quota_exhausted(e):
                            raise LLMQuotaExceededError(
                                f"Provider quota exhausted: {e}",
                                retry_after=retry_after,
                            )
                        if attempt >= num_retries:
                            last_error = e
                            break
                        delay = self._calculate_delay(attempt, 1.0, 60.0, e)
                        logger.warning(f"Rate limit hit. Retrying in {delay:.1f}s...")
                        time_module.sleep(delay)
                        continue
                    last_error = e
                    break
                except Exception as e:
                    last_error = e
                    break

        raise LLMError(f"All models failed. Last error: {last_error}")

    @staticmethod
    def extract_content(response: Any) -> str:
        """Extract text content from response."""
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            return message.content or ""
        return ""

    @staticmethod
    def extract_tool_calls(response: Any) -> List[Any]:
        """Extract tool calls from response."""
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return message.tool_calls
        return []

    @staticmethod
    def extract_reasoning(response: Any) -> Optional[str]:
        """Extract reasoning content from response (for models that support it)."""
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'reasoning') and message.reasoning:
                return message.reasoning
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                return message.reasoning_content
        return None


# Backward compatibility: helper function for simple completions
async def quick_completion(
    messages: List[Dict],
    use_case: str = "chatbot",
    **kwargs
) -> str:
    """Quick helper for simple completions."""
    llm = LLMService(use_case=use_case)
    response = await llm.completion(messages=messages, **kwargs)
    return LLMService.extract_content(response)
