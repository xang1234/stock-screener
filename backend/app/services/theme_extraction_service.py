"""
Theme Extraction Service

Uses LLMService (via LiteLLM) to extract themes and tickers from unstructured content.
This is the core intelligence layer that identifies market themes from text.
Falls back to Google Gemini if LiteLLM providers are unavailable.
"""
import json
import logging
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import and_, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..infra.db.repositories.theme_alias_repo import SqlThemeAliasRepository
from ..domain.theme_matching import MatchDecision, MatchThresholdConfig
from ..models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ThemeAlias,
    ThemeMention,
    ThemeCluster,
    ThemeConstituent,
)
from ..config import settings
from .llm import LLMService, LLMError
from .theme_identity_normalization import UNKNOWN_THEME_KEY, canonical_theme_key, display_theme_name

# Optional Gemini import for fallback
# Suppress Pydantic warnings from google-genai library (third-party issue)
import warnings
warnings.filterwarnings("ignore", message="Field name .* shadows an attribute in parent")

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

logger = logging.getLogger(__name__)


class ThemeExtractionParseError(Exception):
    """Raised when model output cannot be parsed into the extraction schema."""


_LEGACY_THEME_CANONICAL_NAME_MAP = {
    # AI themes
    "ai_infrastructure": "AI Infrastructure",
    "ai_infra": "AI Infrastructure",
    "ai_datacenter": "AI Infrastructure",
    "ai_data_center": "AI Infrastructure",
    "ai_chip": "AI Semiconductors",
    "ai_semiconductor": "AI Semiconductors",
    # Healthcare
    "glp1": "GLP-1 Weight Loss",
    "weight_loss_drug": "GLP-1 Weight Loss",
    "obesity_drug": "GLP-1 Weight Loss",
    # Energy
    "nuclear_power": "Nuclear Energy",
    "nuclear_renaissance": "Nuclear Energy",
    "uranium": "Nuclear Energy",
    # Defense
    "defense_drone": "Defense & Drones",
    "military_drone": "Defense & Drones",
    "defense_technology": "Defense & Drones",
    # Quantum
    "quantum": "Quantum Computing",
    "quantum_tech": "Quantum Computing",
    # Manufacturing
    "reshoring": "Nearshoring & Reshoring",
    "nearshoring": "Nearshoring & Reshoring",
    "onshoring": "Nearshoring & Reshoring",
    # Crypto
    "bitcoin": "Crypto & Bitcoin",
    "cryptocurrency": "Crypto & Bitcoin",
    "bitcoin_mining": "Bitcoin Miners",
    "crypto_mining": "Bitcoin Miners",
}


# System prompt for theme extraction
EXTRACTION_SYSTEM_PROMPT = """You are a financial analyst AI that extracts market themes and stock mentions from text.

Your task is to identify:
1. MARKET THEMES - Investment narratives, sector trends, or thematic plays mentioned in the text
   Examples: "AI infrastructure", "GLP-1 weight loss drugs", "nuclear energy renaissance",
   "defense drones", "quantum computing", "nearshoring/reshoring", "datacenter power demand"

2. STOCK TICKERS - US-listed stock symbols mentioned or clearly implied
   - Only include actual tradeable US stock tickers (NYSE, NASDAQ)
   - Convert company names to tickers when confident (e.g., "Nvidia" -> "NVDA")
   - Do NOT include ETFs, indices, or non-US stocks unless they're ADRs

3. SENTIMENT - The author's view on each theme (bullish, bearish, neutral)

4. CONFIDENCE - Your confidence in the extraction (0.0 to 1.0)

IMPORTANT RULES:
- Only extract themes that are INVESTMENT-RELATED (not general news topics)
- A theme must be actionable - something an investor could trade on
- Group related concepts into canonical theme names
- If no clear themes are present, return an empty list
- Be conservative - only extract high-confidence mentions
- Prefer specific themes over vague ones ("AI chip demand" > "technology")
"""

EXTRACTION_USER_PROMPT = """Extract market themes and stock tickers from this content.

Source: {source_name} ({source_type})
Published: {published_at}

Title: {title}

Content:
{content}

---

Return a JSON array of theme mentions. Each mention should have:
- theme: string (the market theme/narrative)
- tickers: array of strings (US stock tickers related to this theme in this content)
- sentiment: string ("bullish", "bearish", or "neutral")
- confidence: float (0.0 to 1.0)
- excerpt: string (relevant quote from content, max 200 chars)

Example output:
[
  {{
    "theme": "AI Infrastructure",
    "tickers": ["NVDA", "AVGO", "MRVL"],
    "sentiment": "bullish",
    "confidence": 0.9,
    "excerpt": "The buildout of AI datacenters is accelerating faster than expected..."
  }}
]

If no investment themes are found, return an empty array: []

Return ONLY the JSON array, no other text."""


class ThemeExtractionService:
    """Service for extracting themes from content using LLMService or Google Gemini API"""

    # Gemini models (fallback if LiteLLM providers not available)
    GEMINI_MODELS = [
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.5-flash",
    ]
    PROCESSABLE_STATUSES = {"pending", "failed_retryable"}
    MATCH_THRESHOLD_CONFIG = MatchThresholdConfig(
        version="match-v1",
        default_threshold=1.0,
        pipeline_overrides={
            "technical": 1.0,
            "fundamental": 1.0,
        },
        source_type_overrides={},
        pipeline_source_type_overrides={},
    )
    ALIAS_SOURCE_TRUST = {
        "manual": 1.0,
        "correlation": 0.85,
        "backfill": 0.7,
        "llm_extraction": 0.45,
    }
    ALIAS_AUTO_ATTACH_MIN_SCORE = 0.70
    ALIAS_AUTO_ATTACH_MIN_EVIDENCE = 2
    FUZZY_ATTACH_THRESHOLD_DEFAULT = 0.90
    FUZZY_REVIEW_THRESHOLD_DEFAULT = 0.78
    FUZZY_AMBIGUITY_MARGIN_DEFAULT = 0.04
    FUZZY_ATTACH_THRESHOLD_PIPELINE_OVERRIDES = {
        "technical": 0.91,
        "fundamental": 0.88,
    }
    FUZZY_ATTACH_THRESHOLD_SOURCE_TYPE_OVERRIDES = {
        "news": 0.89,
        "substack": 0.91,
    }
    FUZZY_ATTACH_THRESHOLD_PIPELINE_SOURCE_TYPE_OVERRIDES = {}
    FUZZY_REVIEW_THRESHOLD_PIPELINE_OVERRIDES = {}
    FUZZY_REVIEW_THRESHOLD_SOURCE_TYPE_OVERRIDES = {}
    FUZZY_REVIEW_THRESHOLD_PIPELINE_SOURCE_TYPE_OVERRIDES = {}
    FUZZY_AMBIGUITY_MARGIN_PIPELINE_OVERRIDES = {}
    FUZZY_AMBIGUITY_MARGIN_SOURCE_TYPE_OVERRIDES = {}
    FUZZY_AMBIGUITY_MARGIN_PIPELINE_SOURCE_TYPE_OVERRIDES = {}

    def __init__(self, db: Session, pipeline: str = "technical"):
        self.db = db
        self.pipeline = pipeline
        self.pipeline_config = None
        self._load_pipeline_config()

        self.llm = None
        self.gemini_client = None
        self.provider = None  # 'litellm' or 'gemini'
        self.configured_model = None  # Model ID from settings
        self._load_configured_model()
        self._init_client()
        self._load_reprocessing_config()

        # Known ticker patterns for validation
        self.ticker_pattern = re.compile(r'^[A-Z]{1,5}$')

        # Cache of valid tickers (loaded from universe)
        self._valid_tickers: Optional[set] = None

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 0.5 seconds for most providers

    def _load_pipeline_config(self):
        """Load pipeline-specific configuration"""
        try:
            from ..config.pipeline_config import get_pipeline_config
            self.pipeline_config = get_pipeline_config(self.pipeline)
            logger.info(f"Loaded pipeline config for: {self.pipeline}")
        except Exception as e:
            logger.warning(f"Could not load pipeline config: {e}. Using defaults.")
            self.pipeline_config = None

    def _load_configured_model(self):
        """Load configured model from database settings"""
        try:
            from ..models.app_settings import AppSetting
            setting = self.db.query(AppSetting).filter(AppSetting.key == "llm_extraction_model").first()
            if setting:
                self.configured_model = setting.value
                logger.info(f"Using configured extraction model: {self.configured_model}")

                # Set OLLAMA_API_BASE if using Ollama model
                if self.configured_model.startswith("ollama"):
                    ollama_setting = self.db.query(AppSetting).filter(AppSetting.key == "ollama_api_base").first()
                    if ollama_setting:
                        os.environ["OLLAMA_API_BASE"] = ollama_setting.value
                        logger.info(f"Set OLLAMA_API_BASE to: {ollama_setting.value}")
            else:
                self.configured_model = None
        except Exception as e:
            logger.warning(f"Could not load configured model: {e}")
            self.configured_model = None

    def _load_reprocessing_config(self):
        """Load reprocessing settings from database."""
        try:
            from ..models.app_settings import AppSetting
            setting = self.db.query(AppSetting).filter(
                AppSetting.key == "reprocessing_max_age_days"
            ).first()
            self.max_age_days = int(setting.value) if setting else 30
        except Exception:
            self.max_age_days = 30

    def _init_client(self):
        """Initialize LLM client - try LLMService first, then Gemini"""
        # Try LLMService first (supports multiple providers via LiteLLM)
        try:
            self.llm = LLMService(use_case="extraction")
            self.provider = "litellm"
            self._min_request_interval = 0.5
            logger.info("LLMService initialized for theme extraction")
            return
        except Exception as e:
            logger.warning(f"LLMService initialization failed: {e}")

        # Fallback to Gemini
        gemini_api_key = getattr(settings, "gemini_api_key", None) or getattr(settings, "google_api_key", None)
        if not gemini_api_key:
            gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if gemini_api_key and GEMINI_AVAILABLE:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            self.provider = "gemini"
            self._min_request_interval = 15.0  # Gemini has lower rate limit
            logger.info("Gemini API client initialized for theme extraction (fallback)")
            return

        logger.warning("No LLM provider available - theme extraction will be disabled")

    def _get_valid_tickers(self) -> set:
        """Get set of valid tickers from stock universe"""
        if self._valid_tickers is None:
            from ..models.stock_universe import StockUniverse
            tickers = self.db.query(StockUniverse.symbol).all()
            self._valid_tickers = {t[0] for t in tickers}
        return self._valid_tickers

    def _validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid"""
        if not self.ticker_pattern.match(ticker):
            return False

        valid_tickers = self._get_valid_tickers()
        if valid_tickers and ticker not in valid_tickers:
            return False

        return True

    def _clean_tickers(self, tickers: list) -> list:
        """Filter and clean ticker list"""
        cleaned = []
        for t in tickers:
            t = t.upper().strip()
            # Remove common false positives
            if t in {"A", "I", "AI", "CEO", "CFO", "IPO", "ETF", "NYSE", "SEC", "FDA", "GDP", "CPI"}:
                continue
            if self._validate_ticker(t):
                cleaned.append(t)
        return list(set(cleaned))  # Dedupe

    def _rate_limit(self):
        """Apply rate limiting between API calls"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _try_generate_litellm(self, prompt: str) -> str:
        """Try to generate content with LLMService (LiteLLM)"""
        import asyncio

        system_prompt = self._get_system_prompt()

        # Use configured model if set
        model_override = self.configured_model if self.configured_model else None

        async def _call():
            response = await self.llm.completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=model_override,  # Use configured model or preset default
                temperature=0.2,
                max_tokens=2000,
            )
            return LLMService.extract_content(response)

        # Run async in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context - create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _call())
                return future.result()
        else:
            return asyncio.run(_call())

    def _try_generate_gemini(self, prompt: str, model: str) -> str:
        """Try to generate content with Gemini API"""
        system_prompt = self._get_system_prompt()
        response = self.gemini_client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=system_prompt + "\n\n" + prompt)]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2000,
            )
        )
        return response.text.strip()

    def _get_system_prompt(self) -> str:
        """Get the system prompt, optionally with pipeline-specific additions"""
        base_prompt = EXTRACTION_SYSTEM_PROMPT

        if self.pipeline_config and self.pipeline_config.extraction_prompt_additions:
            # Add pipeline-specific instructions
            additions = self.pipeline_config.extraction_prompt_additions
            examples = self.pipeline_config.theme_examples
            examples_str = ", ".join(f'"{e}"' for e in examples[:5]) if examples else ""

            pipeline_section = f"""

PIPELINE: {self.pipeline_config.display_name.upper()}
{additions}

Example themes for this pipeline: {examples_str}
"""
            return base_prompt + pipeline_section

        return base_prompt

    def extract_from_content(self, content_item: ContentItem) -> list[dict]:
        """
        Extract themes from a single content item using LLM (Groq or Gemini)

        Returns list of extracted theme mentions
        """
        if not self.provider:
            logger.error("No LLM client initialized")
            return []

        # Truncate content if too long
        content = content_item.content or ""
        if len(content) > 10000:
            content = content[:10000] + "... [truncated]"

        prompt = EXTRACTION_USER_PROMPT.format(
            source_name=content_item.source_name or "Unknown",
            source_type=content_item.source_type or "Unknown",
            published_at=content_item.published_at.isoformat() if content_item.published_at else "Unknown",
            title=content_item.title or "",
            content=content,
        )

        try:
            # Apply rate limiting
            self._rate_limit()

            # Try to generate based on provider
            response_text = None
            last_error = None

            if self.provider == "litellm":
                try:
                    response_text = self._try_generate_litellm(prompt)
                except Exception as e:
                    logger.warning(f"LLMService failed: {e}")
                    last_error = e
                    # Fallback to Gemini if available
                    if self.gemini_client:
                        self.provider = "gemini"

            if self.provider == "gemini" and response_text is None:
                for model in self.GEMINI_MODELS:
                    try:
                        response_text = self._try_generate_gemini(prompt, model)
                        break
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate_limit" in error_str.lower():
                            wait_time = 15
                            logger.warning(f"Rate limited on {model}, waiting {wait_time}s before trying next model...")
                            time.sleep(wait_time)
                            last_error = e
                            continue
                        else:
                            raise

            if response_text is None:
                raise last_error or Exception("All providers exhausted")

            # Extract JSON from response
            # Handle Qwen3 thinking tags first
            if "<think>" in response_text and "</think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()

            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            # Strip whitespace
            response_text = response_text.strip()

            # Debug log if response seems problematic
            if not response_text or not response_text.startswith('['):
                logger.warning(f"Unexpected LLM response format: {response_text[:200] if response_text else 'EMPTY'}")

            mentions = json.loads(response_text)

            # Validate and clean extractions
            cleaned_mentions = []
            for mention in mentions:
                if not mention.get("theme"):
                    continue
                raw_theme = mention["theme"].strip()
                if not raw_theme:
                    continue
                if canonical_theme_key(raw_theme) == UNKNOWN_THEME_KEY:
                    continue

                # Clean tickers
                tickers = self._clean_tickers(mention.get("tickers", []))

                cleaned_mentions.append({
                    "theme": raw_theme,
                    "tickers": tickers,
                    "sentiment": mention.get("sentiment", "neutral"),
                    "confidence": min(1.0, max(0.0, float(mention.get("confidence", 0.5)))),
                    "excerpt": (mention.get("excerpt", ""))[:500],  # Limit excerpt length
                })

            return cleaned_mentions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {response_text[:500] if response_text else 'EMPTY'}")
            raise ThemeExtractionParseError(f"Failed to parse LLM response: {e}") from e
        except Exception as e:
            logger.error(f"LLM extraction failed (will retry): {e}")
            raise  # Re-raise so process_batch() marks extraction_error

    def _extract_and_store_mentions(self, content_item: ContentItem) -> int:
        """
        Extract and persist mentions for a content item.

        Returns number of theme mentions created.
        """
        mentions = self.extract_from_content(content_item)

        mention_count = 0
        for mention_data in mentions:
            cluster, decision = self._resolve_cluster_match(mention_data, source_type=content_item.source_type)

            theme_mention = ThemeMention(
                content_item_id=content_item.id,
                theme_cluster_id=cluster.id,
                match_method=decision.method,
                match_score=decision.score,
                match_threshold=decision.threshold,
                threshold_version=decision.threshold_version,
                match_fallback_reason=decision.fallback_reason,
                best_alternative_cluster_id=decision.best_alternative_cluster_id,
                best_alternative_score=decision.best_alternative_score,
                match_score_margin=decision.score_margin,
                source_type=content_item.source_type,
                source_name=content_item.source_name,
                raw_theme=mention_data["theme"],
                canonical_theme=self._normalize_theme(mention_data["theme"]),
                pipeline=self.pipeline,
                tickers=mention_data["tickers"],
                ticker_count=len(mention_data["tickers"]),
                sentiment=mention_data["sentiment"],
                confidence=mention_data["confidence"],
                excerpt=mention_data["excerpt"],
                mentioned_at=content_item.published_at,
            )
            self.db.add(theme_mention)
            mention_count += 1

            self._update_theme_constituents(mention_data, cluster)

        return mention_count

    def _get_match_threshold_config(self) -> MatchThresholdConfig:
        config = getattr(self, "match_threshold_config", None)
        if config is None:
            # Avoid cross-request mutation by cloning class-level defaults.
            config = deepcopy(self.MATCH_THRESHOLD_CONFIG)
            self.match_threshold_config = config
        return config

    def _alias_quality_score(self, alias: ThemeAlias) -> float:
        """Blend source trust, confidence, and evidence for Stage B auto-attach."""
        source = (alias.source or "").strip().lower()
        trust_score = float(self.ALIAS_SOURCE_TRUST.get(source, 0.5))
        confidence_score = max(0.0, min(1.0, float(alias.confidence or 0.0)))
        evidence_count = max(1, int(alias.evidence_count or 1))
        evidence_score = min(1.0, evidence_count / 4.0)
        score = (0.5 * trust_score) + (0.3 * confidence_score) + (0.2 * evidence_score)
        return max(0.0, min(1.0, score))

    def _can_auto_attach_alias(self, alias: ThemeAlias) -> bool:
        """Gate exact alias-key attachment to reduce low-trust alias poisoning."""
        evidence_count = max(1, int(alias.evidence_count or 1))
        if evidence_count < self.ALIAS_AUTO_ATTACH_MIN_EVIDENCE:
            return False
        return self._alias_quality_score(alias) >= self.ALIAS_AUTO_ATTACH_MIN_SCORE

    def _resolve_threshold_with_overrides(
        self,
        *,
        default: float,
        pipeline: str,
        source_type: str | None,
        pipeline_overrides: dict[str, float],
        source_type_overrides: dict[str, float],
        pipeline_source_type_overrides: dict[str, dict[str, float]],
    ) -> float:
        normalized_pipeline = (pipeline or "").strip().lower()
        normalized_source_type = (source_type or "").strip().lower()
        pipeline_source = pipeline_source_type_overrides.get(normalized_pipeline, {})
        if normalized_source_type and normalized_source_type in pipeline_source:
            return float(pipeline_source[normalized_source_type])
        if normalized_pipeline in pipeline_overrides:
            return float(pipeline_overrides[normalized_pipeline])
        if normalized_source_type and normalized_source_type in source_type_overrides:
            return float(source_type_overrides[normalized_source_type])
        return float(default)

    def _resolve_fuzzy_thresholds(self, source_type: str | None) -> tuple[float, float, float]:
        attach_threshold = self._resolve_threshold_with_overrides(
            default=self.FUZZY_ATTACH_THRESHOLD_DEFAULT,
            pipeline=self.pipeline,
            source_type=source_type,
            pipeline_overrides=self.FUZZY_ATTACH_THRESHOLD_PIPELINE_OVERRIDES,
            source_type_overrides=self.FUZZY_ATTACH_THRESHOLD_SOURCE_TYPE_OVERRIDES,
            pipeline_source_type_overrides=self.FUZZY_ATTACH_THRESHOLD_PIPELINE_SOURCE_TYPE_OVERRIDES,
        )
        review_threshold = self._resolve_threshold_with_overrides(
            default=self.FUZZY_REVIEW_THRESHOLD_DEFAULT,
            pipeline=self.pipeline,
            source_type=source_type,
            pipeline_overrides=self.FUZZY_REVIEW_THRESHOLD_PIPELINE_OVERRIDES,
            source_type_overrides=self.FUZZY_REVIEW_THRESHOLD_SOURCE_TYPE_OVERRIDES,
            pipeline_source_type_overrides=self.FUZZY_REVIEW_THRESHOLD_PIPELINE_SOURCE_TYPE_OVERRIDES,
        )
        ambiguity_margin = self._resolve_threshold_with_overrides(
            default=self.FUZZY_AMBIGUITY_MARGIN_DEFAULT,
            pipeline=self.pipeline,
            source_type=source_type,
            pipeline_overrides=self.FUZZY_AMBIGUITY_MARGIN_PIPELINE_OVERRIDES,
            source_type_overrides=self.FUZZY_AMBIGUITY_MARGIN_SOURCE_TYPE_OVERRIDES,
            pipeline_source_type_overrides=self.FUZZY_AMBIGUITY_MARGIN_PIPELINE_SOURCE_TYPE_OVERRIDES,
        )
        review_threshold = min(review_threshold, attach_threshold)
        ambiguity_margin = max(0.0, ambiguity_margin)
        return attach_threshold, review_threshold, ambiguity_margin

    def _normalize_lexical_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
        return normalized

    def _fuzzy_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return float(SequenceMatcher(a=left, b=right).ratio())

    def _cluster_fuzzy_score(
        self,
        *,
        raw_alias: str,
        canonical_key: str,
        canonical_theme: str,
        cluster: ThemeCluster,
    ) -> float:
        query_terms = {
            self._normalize_lexical_text(raw_alias),
            self._normalize_lexical_text(canonical_theme),
            self._normalize_lexical_text(canonical_key.replace("_", " ")),
        }
        query_terms = {term for term in query_terms if term}
        if not query_terms:
            return 0.0

        variants = {
            self._normalize_lexical_text(cluster.display_name or ""),
            self._normalize_lexical_text(cluster.name or ""),
            self._normalize_lexical_text((cluster.canonical_key or "").replace("_", " ")),
        }
        aliases = cluster.aliases if isinstance(cluster.aliases, list) else []
        variants.update(self._normalize_lexical_text(str(alias)) for alias in aliases)
        variants = {variant for variant in variants if variant}
        if not variants:
            return 0.0

        best = 0.0
        for query in query_terms:
            for variant in variants:
                similarity = self._fuzzy_similarity(query, variant)
                if similarity > best:
                    best = similarity
        return best

    def process_content_item(self, content_item: ContentItem) -> int:
        """
        Backward-compatible single-item processing API.

        This method retains legacy commit semantics for direct callers. Batch
        orchestration uses pipeline-state-aware transactional handling instead.
        """
        processed, mention_count = self._process_item_transactional(content_item.id)
        if not processed:
            logger.info(
                "Skipped content item %s for pipeline %s due to non-processable state",
                content_item.id,
                self.pipeline,
            )
            return 0

        logger.info(f"Extracted {mention_count} theme mentions from content item {content_item.id}")
        return mention_count

    def _load_pipeline_state(self, content_item_id: int) -> Optional[ContentItemPipelineState]:
        """Load per-pipeline processing state for a content item."""
        return self.db.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == content_item_id,
            ContentItemPipelineState.pipeline == self.pipeline,
        ).first()

    def _claim_item_for_processing(self, item_id: int) -> bool:
        """
        Atomically claim a content item for this pipeline.

        Returns True only when this worker successfully transitions state to
        in_progress and can safely proceed with extraction.
        """
        now = datetime.utcnow()
        update_count = self.db.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item_id,
            ContentItemPipelineState.pipeline == self.pipeline,
            ContentItemPipelineState.status.in_(list(self.PROCESSABLE_STATUSES)),
        ).update(
            {
                ContentItemPipelineState.status: "in_progress",
                ContentItemPipelineState.attempt_count: func.coalesce(ContentItemPipelineState.attempt_count, 0) + 1,
                ContentItemPipelineState.last_attempt_at: now,
                ContentItemPipelineState.error_code: None,
                ContentItemPipelineState.error_message: None,
            },
            synchronize_session=False,
        )
        if update_count == 1:
            self.db.commit()
            return True
        self.db.rollback()

        existing = self._load_pipeline_state(item_id)
        if existing:
            return False

        try:
            self.db.add(
                ContentItemPipelineState(
                    content_item_id=item_id,
                    pipeline=self.pipeline,
                    status="in_progress",
                    attempt_count=1,
                    last_attempt_at=now,
                )
            )
            self.db.commit()
            return True
        except IntegrityError:
            self.db.rollback()
            return False

    def _classify_failure_status(self, error: Exception) -> str:
        """Classify failures into retryable or terminal pipeline-state buckets."""
        if isinstance(error, ThemeExtractionParseError):
            return "failed_retryable"

        error_text = str(error).lower()
        terminal_markers = (
            "invalid api key",
            "authentication",
            "unauthorized",
            "forbidden",
            "403",
            "quota exceeded",
            "billing",
            "context length",
            "request too large",
            "413",
        )
        if any(marker in error_text for marker in terminal_markers):
            return "failed_terminal"
        return "failed_retryable"

    def _failure_code(self, error: Exception) -> str:
        """Best-effort normalized error code for observability."""
        name = error.__class__.__name__.lower()
        if isinstance(error, ThemeExtractionParseError):
            return "llm_response_parse_error"
        if isinstance(error, LLMError):
            return f"llm_{name}"
        if "json" in name:
            return "json_decode_error"
        return name

    def _process_item_transactional(self, item_id: int) -> tuple[bool, int]:
        """
        Process one item with transactional state transitions.

        Returns (processed_flag, mention_count). If processed_flag is False,
        the item was skipped because its pipeline state is not eligible.
        """
        item = self.db.query(ContentItem).filter(ContentItem.id == item_id).first()
        if not item:
            return False, 0

        if not self._claim_item_for_processing(item_id):
            return False, 0

        try:
            mention_count = self._extract_and_store_mentions(item)
            state = self._load_pipeline_state(item_id)
            if state is None:
                self.db.rollback()
                return False, 0

            processed_at = datetime.utcnow()
            state.status = "processed"
            state.processed_at = processed_at
            state.error_code = None
            state.error_message = None

            # Compatibility writes (global fields remain populated during cutover)
            item.is_processed = True
            item.processed_at = processed_at
            item.extraction_error = None

            self.db.commit()
            return True, mention_count
        except Exception as error:
            self.db.rollback()

            item_for_failure = self.db.query(ContentItem).filter(ContentItem.id == item_id).first()
            if item_for_failure is None:
                return False, 0

            failure_state = self._load_pipeline_state(item_id)
            if failure_state is None:
                failure_state = ContentItemPipelineState(
                    content_item_id=item_id,
                    pipeline=self.pipeline,
                    status="in_progress",
                    attempt_count=1,
                    last_attempt_at=datetime.utcnow(),
                )
                self.db.add(failure_state)

            failure_state.status = self._classify_failure_status(error)
            failure_state.last_attempt_at = datetime.utcnow()
            failure_state.error_code = self._failure_code(error)
            failure_state.error_message = str(error)[:4000]
            failure_state.processed_at = None

            # Compatibility writes
            if isinstance(error, ThemeExtractionParseError):
                item_for_failure.is_processed = False
                item_for_failure.processed_at = None
            else:
                item_for_failure.is_processed = True
                item_for_failure.processed_at = datetime.utcnow()
            item_for_failure.extraction_error = str(error)

            self.db.commit()
            raise

    def _normalize_theme(self, theme: str) -> str:
        """
        Normalize theme name to canonical form

        This helps cluster similar themes:
        - "AI infrastructure" = "AI Infra" = "AI Infrastructure buildout"
        - "GLP-1" = "GLP1" = "Weight loss drugs"
        """
        key = canonical_theme_key(theme)
        if key in _LEGACY_THEME_CANONICAL_NAME_MAP:
            return _LEGACY_THEME_CANONICAL_NAME_MAP[key]
        return display_theme_name(theme)

    def _get_or_create_cluster(self, mention_data: dict) -> ThemeCluster:
        """Backward-compatible wrapper returning only the resolved cluster."""
        cluster, _ = self._resolve_cluster_match(mention_data)
        return cluster

    def _resolve_cluster_match(
        self,
        mention_data: dict,
        source_type: str | None = None,
    ) -> tuple[ThemeCluster, MatchDecision]:
        """Resolve cluster assignment plus a typed decision payload."""
        raw_alias = (mention_data.get("theme") or "").strip()
        canonical_key = canonical_theme_key(raw_alias)
        canonical_theme = self._normalize_theme(raw_alias)
        alias_repo = SqlThemeAliasRepository(self.db)
        threshold_config = self._get_match_threshold_config()
        threshold = threshold_config.resolve_threshold(
            pipeline=self.pipeline,
            source_type=source_type,
        )
        threshold_version = threshold_config.version

        cluster = None
        fallback_reason = None
        method = "create_new_cluster"
        score = 0.0
        best_alternative_cluster_id = None
        best_alternative_score = None
        blocked_alias_key_for_counter: str | None = None
        fuzzy_secondary_cluster_id: int | None = None
        fuzzy_secondary_score: float | None = None

        # Stage A: pipeline+canonical_key exact matching (indexed, low-latency gate).
        if cluster is None:
            cluster = self.db.query(ThemeCluster).filter(
                ThemeCluster.canonical_key == canonical_key,
                ThemeCluster.pipeline == self.pipeline,
                ThemeCluster.is_active == True,
            ).first()
            if cluster is not None:
                method = "exact_canonical_key"
                score = 1.0

        # Stage B: alias-key exact matching when canonical lookup misses.
        alias_match: ThemeAlias | None = None
        if cluster is None and canonical_key != UNKNOWN_THEME_KEY:
            alias_match = alias_repo.find_exact(pipeline=self.pipeline, alias_key=canonical_key)
            if alias_match:
                alias_score = self._alias_quality_score(alias_match)
                cluster = self.db.query(ThemeCluster).filter(
                    ThemeCluster.id == alias_match.theme_cluster_id,
                    ThemeCluster.pipeline == self.pipeline,
                    ThemeCluster.is_active == True,
                ).first()
                if cluster is None:
                    best_alternative_cluster_id = alias_match.theme_cluster_id
                    best_alternative_score = alias_score
                    alias_match.is_active = False
                    if fallback_reason is None:
                        fallback_reason = "alias_target_inactive"
                elif not self._can_auto_attach_alias(alias_match):
                    best_alternative_cluster_id = alias_match.theme_cluster_id
                    best_alternative_score = alias_score
                    blocked_alias_key_for_counter = alias_match.alias_key
                    cluster = None
                    if fallback_reason is None:
                        fallback_reason = "alias_match_below_auto_attach_threshold"
                else:
                    method = "exact_alias_key"
                    score = 1.0
        if cluster is None:
            inactive_cluster = self.db.query(ThemeCluster).filter(
                ThemeCluster.canonical_key == canonical_key,
                ThemeCluster.pipeline == self.pipeline,
                ThemeCluster.is_active == False,
            ).first()
            if inactive_cluster is not None:
                inactive_cluster.is_active = True
                inactive_cluster.last_seen_at = datetime.utcnow()
                cluster = inactive_cluster
                method = "exact_canonical_key"
                score = 1.0
                if fallback_reason is None:
                    fallback_reason = "reactivated_inactive_canonical_match"
        if cluster is None:
            cluster = self.db.query(ThemeCluster).filter(
                ThemeCluster.display_name == canonical_theme,
                ThemeCluster.pipeline == self.pipeline,
                ThemeCluster.is_active == True,
            ).first()
            if cluster is not None:
                method = "exact_display_name"
                score = 1.0
        if cluster is None and canonical_key != UNKNOWN_THEME_KEY:
            fuzzy_candidates: list[tuple[ThemeCluster, float]] = []
            candidate_clusters = self.db.query(ThemeCluster).filter(
                ThemeCluster.pipeline == self.pipeline,
                ThemeCluster.is_active == True,
            ).all()
            for candidate in candidate_clusters:
                candidate_score = self._cluster_fuzzy_score(
                    raw_alias=raw_alias,
                    canonical_key=canonical_key,
                    canonical_theme=canonical_theme,
                    cluster=candidate,
                )
                if candidate_score > 0.0:
                    fuzzy_candidates.append((candidate, candidate_score))

            if fuzzy_candidates:
                fuzzy_candidates.sort(key=lambda item: (item[1], -(item[0].id or 0)), reverse=True)
                top_cluster, top_score = fuzzy_candidates[0]
                if len(fuzzy_candidates) > 1:
                    second_cluster, second_score = fuzzy_candidates[1]
                    fuzzy_secondary_cluster_id = second_cluster.id
                    fuzzy_secondary_score = second_score
                attach_threshold, review_threshold, ambiguity_margin = self._resolve_fuzzy_thresholds(source_type)
                fuzzy_margin = top_score - (fuzzy_secondary_score or 0.0)
                if top_score >= attach_threshold and fuzzy_margin >= ambiguity_margin:
                    cluster = top_cluster
                    method = "fuzzy_lexical"
                    score = float(top_score)
                    threshold = float(attach_threshold)
                    best_alternative_cluster_id = fuzzy_secondary_cluster_id
                    best_alternative_score = fuzzy_secondary_score
                elif top_score >= review_threshold:
                    best_alternative_cluster_id = top_cluster.id
                    best_alternative_score = float(top_score)
                    threshold = float(attach_threshold)
                    if fuzzy_margin < ambiguity_margin:
                        fallback_reason = "fuzzy_ambiguous_review"
                    else:
                        fallback_reason = "fuzzy_low_confidence_review"

        if not cluster:
            cluster = ThemeCluster(
                canonical_key=canonical_key,
                display_name=canonical_theme,
                name=canonical_theme,
                aliases=[mention_data["theme"]],
                pipeline=self.pipeline,  # Set pipeline from service instance
                discovery_source="llm_extraction",
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
                is_emerging=True,
            )
            self.db.add(cluster)
            self.db.flush()  # Get ID

            logger.info(f"Discovered new theme cluster: {canonical_theme} (pipeline={self.pipeline})")
            method = "create_new_cluster"
            score = 0.0
            if fallback_reason is None:
                fallback_reason = "no_existing_cluster_match"

        else:
            # Preserve analyst-managed display labels once a cluster exists.
            if not cluster.display_name:
                cluster.display_name = canonical_theme
            if not cluster.name:
                cluster.name = cluster.display_name
            cluster.last_seen_at = datetime.utcnow()
            # Add alias if new
            if cluster.aliases is None:
                cluster.aliases = []
            if raw_alias and raw_alias not in cluster.aliases:
                cluster.aliases = cluster.aliases + [raw_alias]

        if raw_alias and canonical_key != UNKNOWN_THEME_KEY:
            if blocked_alias_key_for_counter is not None:
                alias_repo.record_counter_evidence(
                    pipeline=self.pipeline,
                    alias_key=blocked_alias_key_for_counter,
                    seen_at=datetime.utcnow(),
                )
            else:
                alias_repo.record_observation(
                    theme_cluster_id=cluster.id,
                    pipeline=self.pipeline,
                    alias_text=raw_alias,
                    source="llm_extraction",
                    confidence=float(mention_data.get("confidence") or 0.5),
                    seen_at=datetime.utcnow(),
                )

        score_margin = None
        if best_alternative_score is not None:
            score_margin = float(score) - float(best_alternative_score)

        decision = MatchDecision(
            selected_cluster_id=cluster.id,
            method=method,
            score=float(score),
            threshold=float(threshold),
            threshold_version=threshold_version,
            fallback_reason=fallback_reason,
            best_alternative_cluster_id=best_alternative_cluster_id,
            best_alternative_score=best_alternative_score,
            score_margin=score_margin,
        )
        return cluster, decision

    def _update_theme_constituents(self, mention_data: dict, cluster: ThemeCluster):
        """Update theme-to-ticker mapping based on mention"""
        # Update constituents
        for ticker in mention_data["tickers"]:
            constituent = self.db.query(ThemeConstituent).filter(
                ThemeConstituent.theme_cluster_id == cluster.id,
                ThemeConstituent.symbol == ticker
            ).first()

            if not constituent:
                constituent = ThemeConstituent(
                    theme_cluster_id=cluster.id,
                    symbol=ticker,
                    source="llm_extraction",
                    confidence=mention_data["confidence"],
                    mention_count=1,
                    first_mentioned_at=datetime.utcnow(),
                    last_mentioned_at=datetime.utcnow(),
                )
                self.db.add(constituent)
            else:
                constituent.mention_count += 1
                constituent.last_mentioned_at = datetime.utcnow()
                # Update confidence (weighted average)
                constituent.confidence = (
                    constituent.confidence * 0.8 + mention_data["confidence"] * 0.2
                )

    def _get_pipeline_source_ids(self) -> list[int]:
        """Get IDs of content sources assigned to this pipeline."""
        from ..models.theme import ContentSource
        pipeline_source_ids = []
        sources = self.db.query(ContentSource).filter(
            ContentSource.is_active == True
        ).all()
        for source in sources:
            source_pipelines = source.pipelines or ["technical", "fundamental"]
            if isinstance(source_pipelines, str):
                try:
                    source_pipelines = json.loads(source_pipelines)
                except Exception:
                    source_pipelines = ["technical", "fundamental"]
            if self.pipeline in source_pipelines:
                pipeline_source_ids.append(source.id)
        return pipeline_source_ids

    def process_batch(self, limit: int = 50, item_ids: list[int] = None) -> dict:
        """
        Process a batch of unprocessed content items

        Filters content items to only those from sources assigned to this pipeline.
        If item_ids is provided, only processes those specific items (used by reprocessing).

        Returns summary of processing results
        """
        pipeline_source_ids = self._get_pipeline_source_ids()
        state_alias = ContentItemPipelineState
        query = self.db.query(ContentItem.id).outerjoin(
            state_alias,
            and_(
                state_alias.content_item_id == ContentItem.id,
                state_alias.pipeline == self.pipeline,
            ),
        )

        if pipeline_source_ids:
            query = query.filter(ContentItem.source_id.in_(pipeline_source_ids))

        if item_ids is not None:
            query = query.filter(ContentItem.id.in_(item_ids))
            query = query.filter(
                or_(
                    state_alias.id.is_(None),
                    state_alias.status.in_(list(self.PROCESSABLE_STATUSES)),
                )
            )
        else:
            query = query.filter(
                or_(
                    state_alias.id.is_(None),
                    state_alias.status.in_(list(self.PROCESSABLE_STATUSES)),
                )
            )
        query = query.order_by(ContentItem.published_at.desc())
        if item_ids is None:
            query = query.limit(limit)

        item_id_rows = query.all()
        process_item_ids = [row[0] for row in item_id_rows]

        results = {
            "processed": 0,
            "total_mentions": 0,
            "errors": 0,
            "new_themes": [],
            "pipeline": self.pipeline,
        }

        for item_id in process_item_ids:
            try:
                processed, mention_count = self._process_item_transactional(item_id)
                if processed:
                    results["processed"] += 1
                    results["total_mentions"] += mention_count
            except Exception as e:
                logger.error(f"Error processing item {item_id}: {e}")
                results["errors"] += 1

        # Get newly discovered themes
        recent_themes = self.db.query(ThemeCluster).filter(
            ThemeCluster.first_seen_at >= datetime.utcnow().replace(hour=0, minute=0, second=0),
            ThemeCluster.pipeline == self.pipeline,
        ).all()
        results["new_themes"] = [t.name for t in recent_themes]

        # Auto-update metrics for all themes so rankings are current
        if results["processed"] > 0:
            try:
                from .theme_discovery_service import ThemeDiscoveryService
                discovery_service = ThemeDiscoveryService(self.db, pipeline=self.pipeline)
                metrics_result = discovery_service.update_all_theme_metrics()
                results["metrics_updated"] = metrics_result.get("themes_updated", 0)
                logger.info(f"Auto-updated metrics for {results['metrics_updated']} themes")
            except Exception as e:
                logger.error(f"Error auto-updating theme metrics: {e}")
                results["metrics_updated"] = 0

        return results

    def reprocess_failed_items(self, limit: int = 500) -> dict:
        """
        Reprocess content items that previously failed extraction for this pipeline.

        Finds pipeline-state rows in failed_retryable, resets them to pending,
        then delegates to process_batch() for retry.

        Returns dict with reprocessing statistics.
        """
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=self.max_age_days)
        pipeline_source_ids = self._get_pipeline_source_ids()

        # Find pipeline-scoped retryable failures eligible for retry
        query = self.db.query(ContentItem).join(
            ContentItemPipelineState,
            and_(
                ContentItemPipelineState.content_item_id == ContentItem.id,
                ContentItemPipelineState.pipeline == self.pipeline,
            ),
        ).filter(
            ContentItemPipelineState.status == "failed_retryable",
            ContentItem.published_at >= cutoff_date,
        )
        if pipeline_source_ids:
            query = query.filter(ContentItem.source_id.in_(pipeline_source_ids))

        failed_items = query.order_by(
            ContentItem.published_at.desc()
        ).limit(limit).all()

        if not failed_items:
            logger.info(f"[{self.pipeline}] No failed items to reprocess")
            return {
                "reprocessed_count": 0,
                "processed": 0,
                "total_mentions": 0,
                "errors": 0,
                "pipeline": self.pipeline,
            }

        # Reset failed items so process_batch() picks them up
        item_ids = [item.id for item in failed_items]
        logger.info(f"[{self.pipeline}] Resetting {len(failed_items)} failed items for reprocessing")
        for item in failed_items:
            state = self._load_pipeline_state(item.id)
            if state is None:
                continue
            state.status = "pending"
            state.error_code = None
            state.error_message = None
            state.processed_at = None
        self.db.commit()

        # Delegate to process_batch() with specific IDs — only retries these items
        result = self.process_batch(limit=limit, item_ids=item_ids)
        result["reprocessed_count"] = len(item_ids)
        return result

    def identify_silent_failures(self, max_age_days: int = None) -> dict:
        """
        Identify content items that silently failed extraction in this pipeline.

        These are items with pipeline state marked processed but no mentions for
        the same pipeline. Resets pipeline state back to pending.

        Returns dict with count and list of reset item IDs.
        """
        from datetime import timedelta

        age_days = max_age_days if max_age_days is not None else self.max_age_days
        cutoff_date = datetime.utcnow() - timedelta(days=age_days)
        pipeline_source_ids = self._get_pipeline_source_ids()

        # Subquery: content_item_ids that have at least one theme mention in this pipeline
        mentioned_ids = self.db.query(ThemeMention.content_item_id).filter(
            ThemeMention.pipeline == self.pipeline
        ).distinct().subquery()

        # Find items with pipeline state marked processed but with zero mentions in this pipeline
        query = self.db.query(ContentItem).join(
            ContentItemPipelineState,
            and_(
                ContentItemPipelineState.content_item_id == ContentItem.id,
                ContentItemPipelineState.pipeline == self.pipeline,
            ),
        ).filter(
            ContentItemPipelineState.status == "processed",
            ContentItem.published_at >= cutoff_date,
            ~ContentItem.id.in_(self.db.query(mentioned_ids.c.content_item_id)),
        )
        if pipeline_source_ids:
            query = query.filter(ContentItem.source_id.in_(pipeline_source_ids))

        silent_failures = query.all()

        if not silent_failures:
            logger.info(f"[{self.pipeline}] No silent failures found")
            return {"reset_count": 0, "items": []}

        item_ids = [item.id for item in silent_failures]
        logger.info(f"[{self.pipeline}] Found {len(silent_failures)} silent failures, resetting for reprocessing")

        for item in silent_failures:
            state = self._load_pipeline_state(item.id)
            if state is None:
                continue
            state.status = "pending"
            state.processed_at = None
        self.db.commit()

        return {"reset_count": len(item_ids), "items": item_ids}


class ThemeNormalizationService:
    """
    Service for normalizing and clustering themes

    Uses embeddings to find similar themes and merge them
    """

    def __init__(self, db: Session):
        self.db = db

    def find_similar_themes(self, theme_name: str, threshold: float = 0.8) -> list[ThemeCluster]:
        """
        Find existing theme clusters similar to given theme name

        For now uses simple string matching - can be upgraded to use embeddings
        """
        # Simple approach: check for substring matches and common words
        theme_lower = theme_name.lower()
        theme_words = set(theme_lower.split())

        similar = []
        all_clusters = self.db.query(ThemeCluster).filter(
            ThemeCluster.is_active == True
        ).all()

        for cluster in all_clusters:
            cluster_lower = cluster.name.lower()
            cluster_words = set(cluster_lower.split())

            # Check for exact substring match
            if theme_lower in cluster_lower or cluster_lower in theme_lower:
                similar.append(cluster)
                continue

            # Check word overlap (Jaccard similarity)
            if theme_words and cluster_words:
                intersection = len(theme_words & cluster_words)
                union = len(theme_words | cluster_words)
                jaccard = intersection / union

                if jaccard >= threshold:
                    similar.append(cluster)

        return similar

    def merge_clusters(self, source_id: int, target_id: int):
        """Merge source cluster into target cluster"""
        source = self.db.query(ThemeCluster).filter(ThemeCluster.id == source_id).first()
        target = self.db.query(ThemeCluster).filter(ThemeCluster.id == target_id).first()

        if not source or not target:
            return

        # Move mentions
        self.db.query(ThemeMention).filter(
            ThemeMention.theme_cluster_id == source_id
        ).update({ThemeMention.theme_cluster_id: target_id})

        # Merge constituents
        source_constituents = self.db.query(ThemeConstituent).filter(
            ThemeConstituent.theme_cluster_id == source_id
        ).all()

        for sc in source_constituents:
            target_constituent = self.db.query(ThemeConstituent).filter(
                ThemeConstituent.theme_cluster_id == target_id,
                ThemeConstituent.symbol == sc.symbol
            ).first()

            if target_constituent:
                # Merge counts
                target_constituent.mention_count += sc.mention_count
                target_constituent.first_mentioned_at = min(
                    target_constituent.first_mentioned_at or datetime.utcnow(),
                    sc.first_mentioned_at or datetime.utcnow()
                )
                target_constituent.last_mentioned_at = max(
                    target_constituent.last_mentioned_at or datetime.min,
                    sc.last_mentioned_at or datetime.min
                )
                self.db.delete(sc)
            else:
                sc.theme_cluster_id = target_id

        # Update target aliases
        if target.aliases is None:
            target.aliases = []
        if source.aliases:
            target.aliases = list(set(target.aliases + source.aliases + [source.name]))
        else:
            target.aliases = list(set(target.aliases + [source.name]))

        # Update first seen
        target.first_seen_at = min(
            target.first_seen_at or datetime.utcnow(),
            source.first_seen_at or datetime.utcnow()
        )

        # Deactivate source
        source.is_active = False

        self.db.commit()
        logger.info(f"Merged theme cluster '{source.name}' into '{target.name}'")
