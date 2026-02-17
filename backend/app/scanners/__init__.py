"""Stock scanning algorithms"""

# Import base classes and abstractions
from .base_screener import (
    BaseStockScreener,
    DataRequirements,
    ScreenerResult,
    StockData
)
from .screener_registry import ScreenerRegistry, screener_registry, register_screener
from .data_preparation import DataPreparationLayer
from .scan_orchestrator import ScanOrchestrator

# Import screeners (this triggers registration via @register_screener decorator)
from .minervini_scanner_v2 import MinerviniScannerV2
from .canslim_scanner import CANSLIMScanner
from .ipo_scanner import IPOScanner
from .custom_scanner import CustomScanner
from .volume_breakthrough_scanner import VolumeBreakthroughScanner


__all__ = [
    # Base classes
    'BaseStockScreener',
    'DataRequirements',
    'ScreenerResult',
    'StockData',
    # Registry
    'ScreenerRegistry',
    'screener_registry',
    'register_screener',
    # Infrastructure
    'DataPreparationLayer',
    'ScanOrchestrator',
    # Screeners
    'MinerviniScannerV2',
    'CANSLIMScanner',
    'IPOScanner',
    'CustomScanner',
    'VolumeBreakthroughScanner',
]
