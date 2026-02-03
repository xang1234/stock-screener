"""
Configuration settings for the Stock Scanner application.
Loads environment variables and provides application settings.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

# Get project root (StockScreenClaude/)
# settings.py is at backend/app/config/settings.py → 4 levels up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    alpha_vantage_api_key: str = ""
    gemini_api_key: str = ""  # For theme extraction via Google Gemini
    google_api_key: str = ""  # Alternative name for Gemini API key
    groq_api_key: str = ""  # For LLM via Groq (single key, backward compatible)
    groq_api_keys: str = ""  # For LLM via Groq (multiple keys, comma-separated)
    deepseek_api_key: str = ""  # For LLM via DeepSeek (cost-effective fallback)
    together_api_key: str = ""  # For LLM via Together AI (wide model selection)
    openrouter_api_key: str = ""  # For LLM via OpenRouter (100+ models, unified billing)
    twitter_bearer_token: str = ""  # For Twitter/X API (optional)
    sotwe_enabled: bool = True  # Enable Sotwe scraper as Twitter fallback
    sotwe_request_timeout: int = 30  # Timeout for Sotwe HTTP requests (seconds)
    benzinga_api_key: str = ""  # For Benzinga news API (optional)
    tavily_api_key: str = ""  # For web search in chatbot (primary)
    serper_api_key: str = ""  # For web search in chatbot (fallback)

    # LLM Routing Configuration
    llm_default_provider: str = "groq"  # Primary provider: groq, deepseek, together_ai, openrouter
    llm_chatbot_model: str = "groq/qwen-qwen3-32b"  # Model for chatbot (LiteLLM format)
    llm_research_model: str = "groq/qwen-qwen3-32b"  # Model for research agents
    llm_fallback_enabled: bool = True  # Enable automatic fallback to other providers
    llm_fallback_models: str = "groq/llama-3.3-70b-versatile,deepseek/deepseek-chat,together_ai/meta-llama/Llama-3-70b-chat-hf"

    # Database - use absolute path to avoid working directory issues
    database_url: str = f"sqlite:///{_PROJECT_ROOT}/data/stockscanner.db"

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Admin API key (required for config endpoints)
    admin_api_key: str = ""

    # Rate Limiting
    yfinance_rate_limit: int = 1  # requests per second
    alphavantage_rate_limit: int = 25  # requests per day

    # Scanning
    default_universe: str = "all"
    scan_batch_size: int = 20
    cache_ttl_hours: int = 24

    # Celery / Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    scan_concurrent_workers: int = 4
    scan_rate_limit: float = 1.0  # requests per second (legacy, kept for backward compatibility)

    # Phase 2: Bulk Data Fetching Configuration
    use_bulk_fetching: bool = True  # Use yfinance.Tickers() for batch fetching (ENABLED for performance)

    # Phase 3: Parallel Scanning Configuration
    use_parallel_scanning: bool = True  # Feature flag for Phase 3.1 rollout (ENABLED for performance)
    scan_parallel_workers: int = 4 # Conservative: 2 parallel workers per batch
    scan_parallel_batch_size: int = 50  # Stocks per batch
    scan_parallel_rate_limit: float = 0.5  # 500ms = 2 req/sec (conservative)
    scan_rate_limit_max: float = 2.0  # Max delay on errors (adaptive)
    scan_rate_limit_adaptive: bool = True  # Enable adaptive slowdown on errors
    scan_max_concurrent_batches: int = 1  # Max batches running simultaneously

    # Cache Configuration
    cache_redis_db: int = 2  # Separate DB for cache data
    cache_ttl_seconds: int = 604800  # 7 days (fundamentals change slowly)
    # NOTE: cache_recent_days is overridden by screener requirements:
    # - Volume Breakthrough Scanner: 5 years (1825 days) for 5-year volume high check
    # - Minervini/CANSLIM: 2 years (730 days) for 200-day MA calculations
    # Actual retention is set in PriceCacheService.RECENT_DAYS = 1825
    cache_recent_days: int = 60  # Default for general use (screeners override)
    cache_warmup_enabled: bool = True  # Enable automatic cache warming
    cache_warmup_symbols: int = 100  # Number of symbols to pre-warm
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
    fundamental_refresh_day: int = 5  # Friday = 5
    fundamental_refresh_hour: int = 18  # 6 PM ET (after market close at 4 PM)

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
    data_fetch_lock_wait_seconds: int = 7200  # Max wait for lock before failing
    data_fetch_startup_delay: int = 5  # Seconds to wait before startup task

    # One-time cleanup flags
    invalid_universe_cleanup_enabled: bool = False  # Disabled by default to avoid destructive startup

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

    # Price Cache Batch Fetching Configuration
    price_cache_yfinance_batch_size: int = 50  # Symbols per yfinance batch in get_many()
    price_cache_yfinance_rate_limit: float = 1.0  # Seconds to wait between batches

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
    def cors_origins_list(self) -> List[str]:
        """Convert CORS origins string to list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
