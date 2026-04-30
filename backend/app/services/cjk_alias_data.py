"""Seed alias corpus for the deterministic CJK alias resolver.

This is a small *representative* dictionary, not an exhaustive universe.
It covers the formats called out in ``StockScreenClaude-asia.7.4``:
common HK/JP/TW company names (both English and CJK), with Simplified
and Traditional Han variants listed explicitly because NFKC does **not**
unify those scripts.

Versioning
----------
Any edit to this file must bump
:data:`app.services.cjk_alias_resolver_service.POLICY_VERSION`. The
resolver's :func:`describe_policy` snapshot is how operators (and the
golden-corpus test) detect silent drift.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


class _AliasEntry(TypedDict):
    market: str
    aliases: List[str]


# Canonical symbol -> {market, aliases}. Aliases are the raw forms a
# scraper, headline, or LLM extraction might produce. They are matched
# after NFKC + casefold + punctuation strip (see resolver module).
#
# Representative, not exhaustive. Expand via follow-on beads once the
# pipeline is proven — not by inflating this seed.
ASIA_ALIAS_CORPUS: Dict[str, _AliasEntry] = {
    # --- Hong Kong (HKEX) -----------------------------------------
    "0700.HK": {
        "market": "HK",
        "aliases": [
            "Tencent",
            "Tencent Holdings",
            "騰訊",          # Traditional (HK default)
            "騰訊控股",
            "腾讯",          # Simplified (often in mainland feeds)
            "腾讯控股",
        ],
    },
    "0005.HK": {
        "market": "HK",
        "aliases": [
            "HSBC",
            "HSBC Holdings",
            "滙豐",          # Traditional
            "滙豐控股",
            "汇丰",          # Simplified
            "汇丰控股",
        ],
    },
    "9988.HK": {
        "market": "HK",
        "aliases": [
            "Alibaba",
            "Alibaba Group",
            "阿里巴巴",
            "阿里",
        ],
    },

    # --- Japan (TSE) ----------------------------------------------
    "6758.T": {
        "market": "JP",
        "aliases": [
            "Sony",
            "Sony Group",
            "ソニー",         # Katakana
            "ソニーグループ",
        ],
    },
    "9984.T": {
        "market": "JP",
        "aliases": [
            "SoftBank",
            "SoftBank Group",
            "ソフトバンク",
            "ソフトバンクグループ",
        ],
    },
    "7203.T": {
        "market": "JP",
        "aliases": [
            "Toyota",
            "Toyota Motor",
            "トヨタ",
            "トヨタ自動車",
            "豊田自動車",     # kanji variant (historical / some feeds)
        ],
    },

    # --- Taiwan (TWSE) --------------------------------------------
    "2330.TW": {
        "market": "TW",
        "aliases": [
            "TSMC",
            "Taiwan Semiconductor",
            "Taiwan Semiconductor Manufacturing",
            "台積電",         # Traditional (TW default)
            "臺積電",         # Alternate traditional (Unicode variant)
            "台灣積體電路",
        ],
    },
    "2317.TW": {
        "market": "TW",
        "aliases": [
            "Hon Hai",
            "Hon Hai Precision",
            "Foxconn",
            "鴻海",
            "鴻海精密",
        ],
    },

    # --- Mainland China A-shares ------------------------------------
    "600519.SS": {
        "market": "CN",
        "aliases": [
            "Kweichow Moutai",
            "Moutai",
            "贵州茅台",
            "貴州茅台",
            "茅台",
        ],
    },
    "300750.SZ": {
        "market": "CN",
        "aliases": [
            "CATL",
            "Contemporary Amperex Technology",
            "宁德时代",
            "寧德時代",
        ],
    },
    "601318.SS": {
        "market": "CN",
        "aliases": [
            "Ping An",
            "Ping An Insurance",
            "中国平安",
            "中國平安",
        ],
    },
}


__all__ = ["ASIA_ALIAS_CORPUS"]
