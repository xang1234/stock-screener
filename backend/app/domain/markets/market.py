"""Canonical Market value object."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_MARKET_CODES: frozenset[str] = frozenset({"US", "HK", "IN", "JP", "KR", "TW", "CN"})


class UnsupportedMarketError(ValueError):
    """Raised when a Market code is missing, blank, or unsupported."""


@dataclass(frozen=True, slots=True)
class Market:
    """Canonical supported Market code.

    The constructor is intentionally strict. Use ``from_str`` for wire/user
    input that may need whitespace and case normalization.
    """

    code: str

    def __post_init__(self) -> None:
        if self.code not in SUPPORTED_MARKET_CODES:
            supported = ", ".join(sorted(SUPPORTED_MARKET_CODES))
            raise UnsupportedMarketError(
                f"Unsupported market code {self.code!r}. Supported: {supported}"
            )

    @classmethod
    def from_str(cls, raw: object | None) -> "Market":
        if raw is None:
            raise UnsupportedMarketError("Market code is required")
        if not isinstance(raw, str):
            raise UnsupportedMarketError(f"Market code must be a string, got {type(raw).__name__}")
        code = raw.strip().upper()
        if not code:
            raise UnsupportedMarketError("Market code is required")
        return cls(code)

    def __str__(self) -> str:
        return self.code
