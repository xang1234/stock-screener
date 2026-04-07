"""
Configuration settings for the Stock Scanner application.
Loads environment variables and provides application settings.
"""
import logging
import os
from pathlib import Path
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from typing import List


def _get_project_root() -> Path:
    # settings.py is at backend/app/config/settings.py -> 4 levels up
    return Path(__file__).resolve().parent.parent.parent.parent


_PROJECT_ROOT = _get_project_root()
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    alpha_vantage_api_key: str = ""
    gemini_api_key: str = ""  # For theme extraction via Google Gemini
    google_api_key: str = ""  # Alternative name for Gemini API key
    zai_api_key: str = ""  # For Z.AI GLM models via OpenAI-compatible endpoint
    zai_api_keys: str = ""  # For Z.AI GLM models (multiple keys, comma-separated)
    zai_api_base: str = "https://api.z.ai/api/paas/v4"  # Z.AI OpenAI-compatible base URL
    minimax_api_key: str = ""  # Minimax international API
    minimax_api_base: str = "https://api.minimax.io/v1"  # Minimax OpenAI-compatible base URL
    groq_api_key: str = ""  # For LLM via Groq (single key, backward compatible)
    groq_api_keys: str = ""  # For LLM via Groq (multiple keys, comma-separated)
    deepseek_api_key: str = ""  # For LLM via DeepSeek (cost-effective fallback)
    together_api_key: str = ""  # For LLM via Together AI (wide model selection)
    openrouter_api_key: str = ""  # For LLM via OpenRouter (100+ models, unified billing)
    twitter_bearer_token: str = ""  # Legacy Twitter/X API token (unused for XUI ingestion)
    xui_enabled: bool = True  # Enable xui-reader ingestion for twitter sources
    xui_config_path: str = ""
    xui_profile: str = "default"
    xui_limit_per_source: int = 50
    xui_new_only: bool = True
    xui_checkpoint_mode: str = "auto"
    xui_bridge_enabled: bool = True
    xui_bridge_allowed_origins: str = (
        "http://localhost:80,http://127.0.0.1:80,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )
    xui_bridge_challenge_ttl_seconds: int = 120
    xui_bridge_max_cookies: int = 300
    twitter_request_delay: float = 5.0  # Delay between twitter source fetches (seconds)
    benzinga_api_key: str = ""  # For Benzinga news API (optional)
    tavily_api_key: str = ""  # For web search in chatbot (primary)
    serper_api_key: str = ""  # For web search in chatbot (fallback)

    # LLM Routing Configuration
    llm_default_provider: str = "groq"  # Primary provider: groq, zai, deepseek, together_ai, openrouter
    llm_chatbot_model: str = "groq/qwen-qwen3-32b"  # Model for chatbot (LiteLLM format)
    llm_research_model: str = "groq/qwen-qwen3-32b"  # Model for research agents
    llm_fallback_enabled: bool = True  # Enable automatic fallback to other providers
    llm_fallback_models: str = "groq/llama-3.3-70b-versatile,deepseek/deepseek-chat,together_ai/meta-llama/Llama-3-70b-chat-hf"

    # Runtime profile
    feature_themes: bool = True
    feature_chatbot: bool = True
    feature_tasks: bool = True

    # Database
    database_url: str = ""

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    server_auth_enabled: bool = True
    server_auth_password: str = ""
    server_auth_session_secret: str = ""
    server_auth_cookie_name: str = "stockscanner_session"
    server_auth_session_ttl_hours: int = 24
    server_auth_secure_cookie: bool = False

    # Admin API key (required for config endpoints)
    admin_api_key: str = ""

    # Rate Limiting
    yfinance_rate_limit: int = 1  # requests per second
    alphavantage_rate_limit: int = 25  # requests per day
    finviz_rate_limit_interval: float = 0.5  # seconds between finviz API calls
    yfinance_batch_rate_limit_interval: float = 5.0  # seconds between yfinance batch downloads
    yfinance_per_ticker_delay: float = 0.2  # Deprecated: bulk scheduled jobs should not use per-ticker fetches

    # Scanning
    default_universe: str = "all"
    scan_batch_size: int = 20
    cache_ttl_hours: int = 24
    setup_engine_enabled: bool = True  # Feature flag to toggle Setup Engine scanner

    # Celery / Redis
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_timezone: str = "America/New_York"  # Celery Beat interprets crontab in this tz
    scan_concurrent_workers: int = 4
    scan_rate_limit: float = 1.0  # requests per second (legacy, kept for backward compatibility)

    # Phase 2: Bulk Data Fetching Configuration
    use_bulk_fetching: bool = True  # Use shared yf.download() batch fetching for scheduled jobs

    # Scan use-case configuration
    scan_usecase_chunk_size: int = 25  # Chunk size for use-case path (smaller for cancellation responsiveness)
    feature_snapshot_soft_time_limit_seconds: int = 10800  # 3h budget for full ALL-universe daily snapshot in Docker/Postgres
    feature_snapshot_stale_after_minutes: int = 240  # Running feature runs older than this are treated as stale and failed

    # Cache Configuration
    cache_redis_db: int = 2  # Separate DB for cache data
    cache_ttl_seconds: int = 604800  # 7 days (fundamentals change slowly)
    # NOTE: cache_recent_days is overridden by screener requirements:
    # - Volume Breakthrough Scanner: 5 years (1825 days) for 5-year volume high check
    # - Minervini/CANSLIM: 2 years (730 days) for 200-day MA calculations
    # Actual retention is set in PriceCacheService.RECENT_DAYS = 1825
    cache_recent_days: int = 60  # Default for general use (screeners override)
    cache_warmup_enabled: bool = True  # Enable automatic cache warming
    cache_warmup_symbols: int = 100  # DEPRECATED: All refresh paths now use full universe
    refresh_skip_hours: float = 4.0  # Skip symbols refreshed within this many hours (auto mode)
    cache_incremental_enabled: bool = True  # Enable incremental updates

    # Differential cache TTLs for different data types
    # NOTE: price_cache_ttl is used for Redis TTL. PriceCacheService uses 7 days (604800s)
    # to ensure cached price data survives weekends and market holidays.
    price_cache_ttl: int = 604800  # 7 days for price cache (survives weekends/holidays)
    fundamental_cache_ttl: int = 604800  # 7 days for fundamentals (changes weekly)
    quarterly_cache_ttl: int = 2592000  # 30 days for quarterly data (changes quarterly)

    # Cache Warming Schedule (Celery Beat)
    cache_warm_after_close: bool = True  # Warm cache after market close
    cache_warm_hour: int = 17  # 5 PM ET (after market close)
    cache_warm_minute: int = 30
    cache_weekly_refresh: bool = True  # Full refresh weekly
    cache_weekly_day: int = 0  # Sunday = 0
    cache_weekly_hour: int = 2  # 2 AM ET

    # Fundamental Data Caching
    fundamental_cache_enabled: bool = True  # Enable fundamental data caching
    fundamental_cache_ttl_days: int = 7  # Refresh weekly (7 days)
    fundamental_refresh_enabled: bool = True  # Enable weekly scheduled refresh
    fundamental_refresh_day: int = 6  # Saturday = 6 (avoids collision with Friday 5:30PM cache warmup)
    fundamental_refresh_hour: int = 8  # 8 AM ET (IP has recovered overnight from Friday warmup)

    # Theme Discovery Configuration
    theme_discovery_enabled: bool = True  # Enable theme discovery pipeline
    theme_ingestion_interval_hours: int = 4  # How often to fetch content
    theme_extraction_batch_size: int = 50  # Items per extraction batch
    theme_correlation_threshold: float = 0.5  # Minimum correlation for validation
    theme_velocity_threshold: float = 1.5  # Minimum velocity for "emerging" status

    # IBD Group Rank Gap-Fill Configuration
    group_rank_gapfill_enabled: bool = False  # Disabled - don't auto-run gap-fill on startup
    group_rank_gapfill_max_days: int = 365  # Maximum days to look back for gaps
    group_rank_gapfill_chunk_size: int = 30  # Days per chunk for memory safety
    group_rank_backfill_max_days: int = 365  # API limit for backfill endpoint

    # Breadth Gap-Fill Configuration
    breadth_gapfill_enabled: bool = True  # Enable automatic gap-fill during scheduled task
    breadth_gapfill_max_days: int = 30  # Maximum days to look back for gaps

    # Data Fetch Queue Configuration (prevents API rate limiting)
    data_fetch_queue_name: str = "data_fetch"  # Queue name for serialized data fetching
    data_fetch_lock_timeout: int = 7200  # 2 hours max lock time for long-running tasks
    data_fetch_startup_delay: int = 5  # Seconds to wait before startup task

    # SEC EDGAR Configuration
    sec_user_agent: str = "StockScanner/1.0 (contact@example.com)"
    sec_rate_limit_delay: float = 0.15  # 150ms between requests (SEC allows 10 req/sec)
    sec_cache_ttl_days: int = 30  # Cache SEC filings for 30 days

    # PDF Processing Configuration
    pdf_max_size_mb: int = 50  # Maximum PDF file size in MB
    pdf_request_timeout: int = 60  # Timeout for PDF download in seconds

    # Document Context Window Configuration
    doc_context_window_limit: int = 28000  # Token limit before chunking required
    doc_chunk_target_tokens: int = 2000  # Target tokens per chunk
    doc_chunk_max_tokens: int = 4000  # Maximum tokens per chunk
    doc_chunk_overlap_tokens: int = 200  # Overlap between chunks

    # Redis Bulk Pipeline Configuration (for large multi-symbol fetches)
    redis_bulk_socket_timeout: int = 30  # Timeout for bulk pipeline operations (seconds)
    redis_pipeline_chunk_size: int = 500  # Symbols per Redis pipeline chunk

    # Price Cache Batch Fetching Configuration
    price_cache_yfinance_batch_size: int = 50  # Symbols per yfinance batch in get_many()
    price_cache_yfinance_rate_limit: float = 5.0  # Seconds to wait between batches

    # Snapshot fundamentals / universe lifecycle cutover
    provider_snapshot_ingestion_enabled: bool = False
    provider_snapshot_cutover_enabled: bool = False
    provider_snapshot_on_demand_fallback_enabled: bool = True
    provider_snapshot_min_active_coverage: float = 0.98
    provider_snapshot_max_missing_active_symbols: int = 50

    # Deep Research Configuration
    deep_research_enabled: bool = True  # Enable deep research mode in chatbot
    deep_research_max_concurrent_units: int = 3  # Max parallel research units
    deep_research_max_iterations: int = 5  # Max LLM iterations per unit
    deep_research_max_tool_calls_per_unit: int = 10  # Max tool calls per unit
    deep_research_model: str = "qwen/qwen3-32b"  # Model for research agents
    deep_research_notes_max_tokens: int = 2000  # Max tokens for compression
    deep_research_report_max_tokens: int = 16000  # Max tokens for final report
    read_url_timeout: int = 30  # Timeout for URL fetching
    read_url_max_chars: int = 100000  # Max characters to extract from URLs

    # Hermes / MCP integration
    mcp_server_name: str = "stockscreen-market-copilot"
    mcp_watchlist_writes_enabled: bool = False
    mcp_http_enabled: bool = True

    @field_validator('cache_warm_hour')
    @classmethod
    def validate_cache_warm_hour(cls, v: int) -> int:
        if not 0 <= v <= 23:
            raise ValueError(f"cache_warm_hour must be 0-23, got {v}")
        return v

    @field_validator('cache_warm_minute')
    @classmethod
    def validate_cache_warm_minute(cls, v: int) -> int:
        if not 0 <= v <= 59:
            raise ValueError(f"cache_warm_minute must be 0-59, got {v}")
        return v

    @field_validator('celery_timezone')
    @classmethod
    def validate_celery_timezone(cls, v: str) -> str:
        from zoneinfo import ZoneInfo
        try:
            ZoneInfo(v)
        except (KeyError, Exception):
            raise ValueError(
                f"Invalid celery_timezone: {v!r}. "
                f"Use IANA timezone like 'America/New_York'"
            )
        return v

    @model_validator(mode="after")
    def apply_legacy_twitter_delay_fallback(self) -> "Settings":
        legacy_delay = os.getenv("SOTWE_REQUEST_DELAY")
        explicit_delay = os.getenv("TWITTER_REQUEST_DELAY")
        if not explicit_delay and legacy_delay:
            try:
                self.twitter_request_delay = float(legacy_delay)
                logger.warning(
                    "SOTWE_REQUEST_DELAY is deprecated; use TWITTER_REQUEST_DELAY instead. "
                    "Applying legacy value for compatibility."
                )
            except ValueError:
                logger.warning(
                    "Ignoring invalid SOTWE_REQUEST_DELAY value %r; using TWITTER_REQUEST_DELAY=%s",
                    legacy_delay,
                    self.twitter_request_delay,
                )
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is required. Set it in your .env file, e.g.: "
                "DATABASE_URL=postgresql://user:pass@localhost/stockscanner"
            )
        return self

    @property
    def groq_api_keys_list(self) -> List[str]:
        """Parse comma-separated Groq API keys to list."""
        # First try the multi-key field
        if self.groq_api_keys:
            return [k.strip() for k in self.groq_api_keys.split(",") if k.strip()]
        # Fall back to single key if set
        if self.groq_api_key:
            return [self.groq_api_key]
        return []

    @property
    def zai_api_keys_list(self) -> List[str]:
        """Parse comma-separated Z.AI API keys to list."""
        if self.zai_api_keys:
            return [k.strip() for k in self.zai_api_keys.split(",") if k.strip()]
        if self.zai_api_key:
            return [self.zai_api_key]
        return []

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert CORS origins string to list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def capability_flags(self) -> dict[str, bool]:
        """Expose frontend-relevant feature flags."""
        return {
            "themes": self.feature_themes,
            "chatbot": self.feature_chatbot,
            "tasks": self.feature_tasks,
            "ui_snapshots": True,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()
