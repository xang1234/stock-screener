"""Per-market telemetry package (bead asia.10.1).

Public surface:
- ``PerMarketTelemetry``: emit + read API
- ``get_telemetry()``: process-wide singleton accessor
- ``schema``: versioned payload builders (one function per metric category)
"""

from .per_market_telemetry import PerMarketTelemetry, SHARED_SENTINEL, get_telemetry  # noqa: F401
from . import schema  # noqa: F401
