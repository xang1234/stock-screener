"""Database models for Stock Scanner"""
from .stock import StockPrice, StockFundamental, StockTechnical, StockIndustry
from .scan_result import Scan, ScanResult
from .watchlist import Watchlist
from .market import MarketStatus
from .market_breadth import MarketBreadth
from .industry import Industry, IndustryPerformance, SectorRotation, IBDIndustryGroup, IBDGroupPeerCache, IBDGroupRank
from .stock_universe import StockUniverse
from .theme import (
    ContentSource,
    ContentItem,
    ContentItemPipelineState,
    ThemeMention,
    ThemeCluster,
    ThemeConstituent,
    ThemeAlias,
    ThemeMetrics,
    ThemeAlert,
    ThemePipelineRun,
    ThemeEmbedding,
    ThemeMergeSuggestion,
    ThemeMergeHistory,
    ThemeLifecycleTransition,
    ThemeRelationship,
)
from .task_execution import TaskExecutionHistory
from .chatbot import Conversation, Message, AgentExecution
from .chatbot_folder import ChatFolder
from .market_scan import ScanWatchlist
from .user_theme import UserTheme, UserThemeSubgroup, UserThemeStock
from .user_watchlist import UserWatchlist, WatchlistItem
from .ticker_validation import TickerValidationLog
from .filter_preset import FilterPreset
from .prompt_preset import PromptPreset
from .document_cache import DocumentCache, DocumentChunk
from .institutional_ownership import InstitutionalOwnershipHistory
from .app_settings import AppSetting
from app.infra.db.models.feature_store import (
    FeatureRun, FeatureRunUniverseSymbol, StockFeatureDaily, FeatureRunPointer,
)

__all__ = [
    "StockPrice",
    "StockFundamental",
    "StockTechnical",
    "StockIndustry",
    "Scan",
    "ScanResult",
    "Watchlist",
    "MarketStatus",
    "MarketBreadth",
    "Industry",
    "IndustryPerformance",
    "SectorRotation",
    "IBDIndustryGroup",
    "IBDGroupPeerCache",
    "IBDGroupRank",
    "StockUniverse",
    # Theme discovery models
    "ContentSource",
    "ContentItem",
    "ContentItemPipelineState",
    "ThemeMention",
    "ThemeCluster",
    "ThemeConstituent",
    "ThemeAlias",
    "ThemeMetrics",
    "ThemeAlert",
    "ThemePipelineRun",
    "ThemeEmbedding",
    "ThemeMergeSuggestion",
    "ThemeMergeHistory",
    "ThemeLifecycleTransition",
    "ThemeRelationship",
    # Task execution
    "TaskExecutionHistory",
    # Chatbot
    "Conversation",
    "Message",
    "AgentExecution",
    "ChatFolder",
    # Market Scan
    "ScanWatchlist",
    # User Themes
    "UserTheme",
    "UserThemeSubgroup",
    "UserThemeStock",
    # User Watchlists
    "UserWatchlist",
    "WatchlistItem",
    # Ticker Validation
    "TickerValidationLog",
    # Filter Presets
    "FilterPreset",
    # Prompt Presets
    "PromptPreset",
    # Document Cache
    "DocumentCache",
    "DocumentChunk",
    # Institutional Ownership History
    "InstitutionalOwnershipHistory",
    # App Settings
    "AppSetting",
    # Feature Store
    "FeatureRun",
    "FeatureRunUniverseSymbol",
    "StockFeatureDaily",
    "FeatureRunPointer",
]
