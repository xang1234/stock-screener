from __future__ import annotations

import pytest

from app.domain.markets.cn_symbols import (
    cn_a_share_exchange_for_symbol,
    is_cn_a_share_symbol,
    normalize_cn_local_code,
)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("600519.SS", "600519"),
        ("000001.SZ", "000001"),
        ("920118.BJ", "920118"),
        ("1", "000001"),
    ],
)
def test_normalize_cn_local_code_strips_suffix_and_pads_digits(symbol, expected):
    assert normalize_cn_local_code(symbol) == expected


@pytest.mark.parametrize(
    ("symbol", "expected_exchange"),
    [
        ("600519.SS", "SSE"),
        ("000001.SZ", "SZSE"),
        ("920118.BJ", "BJSE"),
        ("000001", "SZSE"),
    ],
)
def test_cn_a_share_exchange_accepts_matching_suffix_or_bare_code(symbol, expected_exchange):
    assert cn_a_share_exchange_for_symbol(symbol) == expected_exchange
    assert is_cn_a_share_symbol(symbol)


@pytest.mark.parametrize("symbol", ["000001.SS", "000300.SS"])
def test_cn_a_share_exchange_rejects_suffix_conflicting_index_symbols(symbol):
    assert cn_a_share_exchange_for_symbol(symbol) is None
    assert not is_cn_a_share_symbol(symbol)


@pytest.mark.parametrize("symbol", ["ABC.SS", "123456.SS", "777777.SZ"])
def test_cn_a_share_exchange_rejects_suffix_only_symbols_without_a_share_code(symbol):
    assert cn_a_share_exchange_for_symbol(symbol) is None
    assert not is_cn_a_share_symbol(symbol)
