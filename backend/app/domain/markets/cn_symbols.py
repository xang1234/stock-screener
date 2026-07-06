"""Mainland China symbol and exchange inference rules."""

from __future__ import annotations

from typing import Final


CN_SYMBOL_SUFFIX_EXCHANGE: Final[dict[str, str]] = {
    ".SS": "SSE",
    ".SZ": "SZSE",
    ".BJ": "BJSE",
}


def normalize_cn_local_code(symbol_or_local_code: str | None) -> str:
    token = str(symbol_or_local_code or "").strip().upper()
    for suffix in CN_SYMBOL_SUFFIX_EXCHANGE:
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token.zfill(6) if token.isdigit() else token


def cn_exchange_from_symbol_suffix(symbol_or_local_code: str | None) -> str | None:
    token = str(symbol_or_local_code or "").strip().upper()
    for suffix, exchange in CN_SYMBOL_SUFFIX_EXCHANGE.items():
        if token.endswith(suffix):
            return exchange
    return None


def cn_price_symbol_for_native_provider(
    requested_symbol: str | None,
    *,
    local_code: str | None,
    canonical_symbol: str | None,
) -> str | None:
    local = normalize_cn_local_code(local_code)
    if not local.isdigit():
        return None
    if cn_exchange_from_symbol_suffix(requested_symbol) is not None:
        canonical = str(canonical_symbol or "").strip().upper()
        return canonical or None
    return local


def infer_cn_a_share_exchange_from_local_code(local_code: str | None) -> str | None:
    token = normalize_cn_local_code(local_code)
    if token.startswith(("600", "601", "603", "605", "688")):
        return "SSE"
    if token.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZSE"
    if token.startswith(("4", "8", "9")):
        return "BJSE"
    return None


def has_cn_a_share_exchange_conflict(symbol_or_local_code: str | None) -> bool:
    explicit_exchange = cn_exchange_from_symbol_suffix(symbol_or_local_code)
    if explicit_exchange is None:
        return False
    inferred_exchange = infer_cn_a_share_exchange_from_local_code(symbol_or_local_code)
    return inferred_exchange != explicit_exchange


def cn_a_share_exchange_for_symbol(symbol_or_local_code: str | None) -> str | None:
    inferred_exchange = infer_cn_a_share_exchange_from_local_code(symbol_or_local_code)
    explicit_exchange = cn_exchange_from_symbol_suffix(symbol_or_local_code)
    if explicit_exchange is None:
        return inferred_exchange
    if inferred_exchange == explicit_exchange:
        return explicit_exchange
    return None


def is_cn_a_share_symbol(symbol_or_local_code: str | None) -> bool:
    return cn_a_share_exchange_for_symbol(symbol_or_local_code) is not None
