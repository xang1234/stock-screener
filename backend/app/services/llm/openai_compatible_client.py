"""Env-driven, provider-agnostic LLM tiebreaker for IBD classification.

The IBD classifier is delivered as a GitHub Action where the operator wants to
point at *any* OpenAI-compatible LLM by setting environment variables — exactly
the pattern in ZhuLinsen/daily_stock_analysis (OPENAI_BASE_URL/MODEL/API_KEY).
Switching providers is therefore an env-var edit, not a code change:

    IBD_LLM_API_BASE   e.g. https://api.deepseek.com/v1
    IBD_LLM_API_KEY    the provider key
    IBD_LLM_MODEL      e.g. deepseek-chat  (or a provider-prefixed id like minimax/MiniMax-M2.7)
    IBD_LLM_TEMPERATURE / IBD_LLM_MAX_TOKENS  (optional)

LiteLLM routes any OpenAI-compatible endpoint via ``model="openai/<name>" +
api_base + api_key`` — the same mechanism the repo already uses for Z.AI.

When ``IBD_LLM_*`` is unset, this falls back to the sanctioned in-repo path
(``LLMService(use_case="ibd_classification")``), so governance is preserved for
default deployments while the GitHub Action stays freely swappable. The generic
override is scoped to this classifier and does not touch the allowlist used by
chatbot/extraction/etc.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an IBD industry-group classifier. Given a company's descriptive text "
    "and a list of candidate industry groups, choose the single group that best "
    "fits the company. Respond with ONLY the exact group name from the list, with "
    "no explanation, punctuation, or extra words."
)


def _render_prompt(text: str, shortlist: list[str]) -> str:
    listing = "\n".join(f"- {g}" for g in shortlist)
    return (
        f"Company: {text}\n\n"
        f"Candidate industry groups:\n{listing}\n\n"
        "Answer with exactly one group name from the list above."
    )


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def env_model_id() -> Optional[str]:
    return _env("IBD_LLM_MODEL")


def _resolve_litellm_model(model: str, *, has_api_base: bool) -> str:
    """Provider-prefix the model for LiteLLM.

    The deciding signal is whether a custom ``api_base`` is configured, not
    whether the id contains a slash:
    - already ``openai/``-prefixed → unchanged;
    - custom OpenAI-compatible endpoint (``has_api_base``) → force the ``openai/``
      prefix so a slashed gateway model id like ``org/model`` (vLLM/LM Studio/HF
      names) routes through ``api_base`` instead of being misread as a
      ``<provider>/...`` route;
    - otherwise a slashed id is a native LiteLLM provider route (e.g.
      ``groq/llama-3.3-70b``) and a bare name gets the ``openai/`` prefix.
    """
    if model.startswith("openai/"):
        return model
    if has_api_base:
        return f"openai/{model}"
    return model if "/" in model else f"openai/{model}"


def match_choice(response_text: str, shortlist: list[str]) -> Optional[str]:
    """Map a model response to one of the shortlisted group names.

    Tolerates models that echo extra text: exact (case-insensitive) match first,
    then the longest shortlist entry contained in the response (longest-first so
    'Computers-Software' wins over a substring 'Computers')."""
    if not response_text:
        return None
    text = response_text.strip()
    lowered = text.lower()
    by_lower = {g.lower(): g for g in shortlist}
    if lowered in by_lower:
        return by_lower[lowered]
    for group in sorted(shortlist, key=len, reverse=True):
        if group.lower() in lowered:
            return group
    return None


class OpenAICompatibleTiebreaker:
    """Calls an OpenAI-compatible endpoint to pick one group from a shortlist."""

    def __init__(
        self,
        *,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        temperature: float = 0.1,
        max_tokens: int = 200,
        timeout: float = 30.0,
        complete_fn: Optional[Callable[..., str]] = None,
    ):
        self.model = model
        self.litellm_model = _resolve_litellm_model(model, has_api_base=bool(api_base))
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._complete_fn = complete_fn  # injectable for tests

    @property
    def model_id(self) -> str:
        return self.model

    def _litellm_params(self, user_prompt: str) -> dict:
        params = {
            "model": self.litellm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Bound every call: the classifier's runtime deadline is only checked
            # between symbols, so without this a single hung request could run past
            # the job cap. Worst-case overrun is one timeout, not unbounded.
            "timeout": self.timeout,
        }
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _complete(self, user_prompt: str) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(user_prompt)
        import litellm

        response = litellm.completion(**self._litellm_params(user_prompt))
        return response["choices"][0]["message"]["content"] or ""

    def __call__(self, text: str, shortlist: list[str]) -> Optional[str]:
        if not shortlist:
            return None
        try:
            raw = self._complete(_render_prompt(text, shortlist))
        except Exception as exc:  # noqa: BLE001 — never let a bad call break the batch
            logger.warning("IBD LLM tiebreaker call failed: %s", exc)
            return None
        return match_choice(raw, shortlist)


def build_ibd_tiebreaker() -> tuple[Optional[Callable[[str, list[str]], Optional[str]]], Optional[str]]:
    """Return ``(tiebreaker, model_id)`` for the IBD classifier.

    Precedence:
    1. Env-driven OpenAI-compatible path when ``IBD_LLM_MODEL`` is set.
    2. Sanctioned ``LLMService(use_case="ibd_classification")`` when a provider key
       is configured.
    3. ``(None, None)`` — the classifier then runs free-only (crosswalk + embedding).
    """
    model = env_model_id()
    if model:
        temperature = float(_env("IBD_LLM_TEMPERATURE") or 0.1)
        max_tokens = int(_env("IBD_LLM_MAX_TOKENS") or 200)
        timeout = float(_env("IBD_LLM_TIMEOUT") or 30.0)
        tb = OpenAICompatibleTiebreaker(
            model=model,
            api_base=_env("IBD_LLM_API_BASE"),
            api_key=_env("IBD_LLM_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        logger.info("IBD tiebreaker: env-driven OpenAI-compatible model %s", model)
        return tb, tb.model_id

    sanctioned = _build_sanctioned_tiebreaker()
    if sanctioned is not None:
        return sanctioned
    logger.info("IBD tiebreaker: none configured; classifier runs free-only")
    return None, None


def _sanctioned_key_available() -> bool:
    """True when a key for the ibd_classification preset (Minimax/Z.AI) exists.

    Guards against returning a tiebreaker that would fail one API call per residual
    stock when no provider is configured — in that case the classifier should run
    free-only (crosswalk + embedding) instead.
    """
    try:
        from ...config import settings
    except Exception:  # noqa: BLE001
        return False
    if (getattr(settings, "minimax_api_key", "") or "").strip():
        return True
    try:
        if settings.zai_api_keys_list:
            return True
    except Exception:  # noqa: BLE001
        pass
    return bool((getattr(settings, "zai_api_key", "") or "").strip())


def _build_sanctioned_tiebreaker():
    """Wrap the in-repo LLMService preset as a tiebreaker, if a key is available."""
    if not _sanctioned_key_available():
        return None
    try:
        from .llm_service import LLMService
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sanctioned LLMService unavailable: %s", exc)
        return None
    try:
        service = LLMService(use_case="ibd_classification")
    except Exception as exc:  # noqa: BLE001
        logger.info("No sanctioned LLM provider for IBD classification: %s", exc)
        return None

    model_id = service.preset.primary.model_id

    # Reuse the single tiebreaker implementation; only the transport differs.
    def _complete(prompt: str) -> str:
        response = service.completion_sync(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        return LLMService.extract_content(response)

    tiebreaker = OpenAICompatibleTiebreaker(
        model=model_id, api_base=None, api_key=None, complete_fn=_complete
    )
    return tiebreaker, model_id
