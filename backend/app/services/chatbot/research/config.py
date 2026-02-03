"""
Configuration for the Deep Research module.
Extends the main settings with research-specific parameters.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class ResearchSettings(BaseSettings):
    """Settings specific to deep research functionality."""

    # Feature flags
    deep_research_enabled: bool = True

    # Concurrency control
    deep_research_max_concurrent_units: int = 3
    deep_research_max_iterations: int = 5
    deep_research_max_tool_calls_per_unit: int = 10

    # Model configuration (uses Groq qwen3-32b by default)
    deep_research_model: str = "qwen/qwen3-32b"

    # Token budgets
    deep_research_notes_max_tokens: int = 2000
    deep_research_report_max_tokens: int = 16000

    # URL reading configuration
    read_url_timeout: int = 30
    read_url_max_chars: int = 100000
    read_url_max_bytes: int = 5000000
    read_url_user_agent: str = "Mozilla/5.0 (compatible; StockResearchBot/1.0)"

    # Research planner settings
    research_max_sub_questions: int = 5
    research_min_sub_questions: int = 2

    # Follow-up research settings
    follow_up_enabled: bool = True
    follow_up_max_questions: int = 2
    follow_up_max_tool_calls_per_unit: int = 5  # Lighter than initial research

    # Report settings
    report_max_citations: int = 20
    report_include_source_list: bool = True

    # Groq API retry settings
    groq_max_retries: int = 3
    groq_retry_base_delay: float = 1.0
    groq_retry_max_delay: float = 60.0

    class Config:
        env_prefix = "RESEARCH_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global research settings instance
research_settings = ResearchSettings()
