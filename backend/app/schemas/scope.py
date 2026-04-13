"""Shared Pydantic mixin for response schemas that carry an analytics scope tag.

See :mod:`app.domain.analytics.scope` for the policy that produces these
fields.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScopedResponseMixin(BaseModel):
    """Adds ``market_scope`` + ``scope_reason`` to a response model.

    Use alongside ``app.domain.analytics.scope.us_only_tag`` at the
    endpoint: spread the tag into the constructor via
    ``**scope_response_fields(feature)`` (see the scope module helper).
    """

    market_scope: Optional[str] = Field(
        default=None,
        description="Market scope (e.g. 'US'). See analytics scope policy.",
    )
    scope_reason: Optional[str] = Field(
        default=None,
        description="Why this scope applies (e.g. 'IBD taxonomy is US-specific').",
    )
