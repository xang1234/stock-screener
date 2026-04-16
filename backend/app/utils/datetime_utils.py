"""Datetime helpers shared across telemetry and governance code.

Shared here so DB-adapter tz-awareness differences (see
``services/telemetry/weekly_audit._summarize_alerts``) and the launch-gate
``now`` normalization (see ``services/governance/launch_gates.GateContext``)
stay in lockstep on what "make this UTC-aware" actually means.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def as_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Coerce a datetime to UTC-aware. Returns None unchanged.

    - Naive (tzinfo=None): assume UTC and attach. Some adapters and test
      harnesses still return naive values for ``DateTime(timezone=True)``.
    - Already-aware non-UTC: convert to UTC via ``astimezone``.
    - Already-aware UTC: returned unchanged.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
