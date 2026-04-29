"""CN market universe ingestion adapter with deterministic canonicalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .security_master_service import security_master_resolver

_CN_EXCHANGE_ALIASES: dict[str, str] = {
    "SSE": "SSE",
    "SHSE": "SSE",
    "XSHG": "SSE",
    "SH": "SSE",
    "SS": "SSE",
    "SZSE": "SZSE",
    "XSHE": "SZSE",
    "SZ": "SZSE",
    "BSE": "BSE",
    "BJSE": "BSE",
    "XBSE": "BSE",
    "XBEI": "BSE",
    "BJ": "BSE",
}

_APPROVED_CN_SOURCES: frozenset[str] = frozenset(
    {
        "cn_akshare_eastmoney",
        "cn_baostock",
        "cn_manual_csv",
        "cn_reference_bundle",
    }
)

_CN_LOCAL_CODE_RE = re.compile(r"^[0-9]{6}$")
_EXCLUDED_PRODUCT_TOKENS = (
    "ETF",
    "FUND",
    "LOF",
    "REIT",
    "BOND",
    "CONVERTIBLE",
    "WARRANT",
    "INDEX",
    "B SHARE",
    "B-SHARE",
    "NEEQ",
)

_CN_SECTOR_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Financials", ("银行", "保险", "证券", "金融", "BANK", "INSURANCE", "SECURITIES", "FINANCE")),
    ("Information Technology", ("半导体", "计算机", "软件", "电子", "芯片", "SEMICONDUCTOR", "SOFTWARE", "ELECTRONIC", "TECHNOLOGY")),
    ("Communication Services", ("通信服务", "互联网", "游戏", "广告", "COMMUNICATION", "INTERNET", "GAME", "MEDIA")),
    ("Health Care", ("医药", "生物", "医疗", "制药", "HEALTH", "PHARMA", "BIOTECH", "MEDICAL")),
    ("Consumer Staples", ("食品", "饮料", "白酒", "酿酒", "农业", "农林", "牧", "渔", "FOOD", "BEVERAGE", "AGRICULTURE")),
    ("Consumer Discretionary", ("汽车", "家电", "传媒", "旅游", "酒店", "纺织", "服饰", "零售", "AUTO", "RETAIL", "APPAREL", "TEXTILE", "LEISURE")),
    ("Energy", ("石油", "煤炭", "天然气", "能源", "OIL", "COAL", "GAS", "ENERGY")),
    ("Utilities", ("电力", "公用事业", "水务", "燃气", "UTILITY", "POWER", "WATER")),
    ("Real Estate", ("房地产", "物业", "REAL ESTATE", "PROPERTY")),
    ("Materials", ("化工", "有色", "钢铁", "建材", "材料", "采掘", "CHEMICAL", "STEEL", "MATERIAL", "MINING")),
    ("Industrials", ("机械", "电气设备", "设备", "航空", "航运", "运输", "物流", "建筑", "工业", "MACHINERY", "INDUSTRIAL", "TRANSPORT", "CONSTRUCTION")),
)


def infer_cn_sector(*labels: Any, board: str = "") -> str:
    """Infer a broad English sector from Eastmoney/AKShare industry labels."""
    explicit = str((labels[0] if labels else "") or "").strip()
    if explicit:
        return explicit

    haystack = " ".join(str(label or "").strip().upper() for label in labels[1:] if label)
    for sector, keywords in _CN_SECTOR_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return sector
    if str(board or "").strip():
        return "Other"
    return ""


@dataclass(frozen=True)
class CNCanonicalUniverseRow:
    """Canonical CN row emitted by the ingestion adapter."""

    symbol: str
    name: str
    market: str
    exchange: str
    board: str
    currency: str
    timezone: str
    local_code: str
    sector: str
    industry_group: str
    industry: str
    sub_industry: str
    market_cap: float | None
    source_name: str
    source_symbol: str
    source_row_number: int
    snapshot_id: str
    snapshot_as_of: str | None
    source_metadata: dict[str, Any]
    lineage_hash: str
    row_hash: str


@dataclass(frozen=True)
class CNRejectedUniverseRow:
    """Rejected CN row with reason."""

    source_row_number: int
    source_symbol: str
    reason: str


@dataclass(frozen=True)
class CNCanonicalizationResult:
    """Canonicalization output, split into accepted and rejected rows."""

    canonical_rows: tuple[CNCanonicalUniverseRow, ...]
    rejected_rows: tuple[CNRejectedUniverseRow, ...]


class CNUniverseIngestionAdapter:
    """Normalize and validate mainland China A-share universe rows."""

    @staticmethod
    def normalize_source_name(source_name: str) -> str:
        normalized = (source_name or "").strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("source_name must be provided")
        return normalized

    @classmethod
    def is_approved_source(cls, source_name: str) -> bool:
        normalized = cls.normalize_source_name(source_name)
        return normalized in _APPROVED_CN_SOURCES

    @staticmethod
    def _normalize_source_symbol(raw_symbol: Any) -> str:
        symbol = str(raw_symbol or "").strip().upper().replace(" ", "")
        if symbol.startswith("$"):
            symbol = symbol[1:]
        return symbol

    @staticmethod
    def _infer_exchange_from_symbol(source_symbol: str) -> str | None:
        token = str(source_symbol or "").strip().upper()
        if token.startswith(("SSE:", "SHSE:", "XSHG:", "SH:")) or token.endswith(".SS"):
            return "SSE"
        if token.startswith(("SZSE:", "XSHE:", "SZ:")) or token.endswith(".SZ"):
            return "SZSE"
        if token.startswith(("BSE:", "BJSE:", "XBSE:", "XBEI:", "BJ:")) or token.endswith(".BJ"):
            return "BSE"
        code = CNUniverseIngestionAdapter._strip_symbol_decoration(token)
        if _CN_LOCAL_CODE_RE.fullmatch(code):
            return CNUniverseIngestionAdapter._infer_exchange_from_code(code)
        return None

    @classmethod
    def _normalize_exchange(cls, raw_exchange: Any, *, source_symbol: str = "") -> str:
        exchange = str(raw_exchange or "").strip().upper()
        if not exchange:
            inferred = cls._infer_exchange_from_symbol(source_symbol)
            if inferred is not None:
                return inferred
            raise ValueError(
                "CN exchange is required for undecorated six-digit tickers. "
                "Provide SSE/SZSE/BSE or use a .SS/.SZ/.BJ suffix."
            )
        normalized = _CN_EXCHANGE_ALIASES.get(exchange)
        if normalized is None:
            raise ValueError(
                f"Unsupported CN exchange '{exchange}'. Expected SSE, SZSE, or BSE."
            )
        return normalized

    @staticmethod
    def _strip_symbol_decoration(source_symbol: str) -> str:
        token = source_symbol
        for prefix in ("SSE:", "SHSE:", "XSHG:", "SH:", "SZSE:", "XSHE:", "SZ:", "BSE:", "BJSE:", "XBSE:", "XBEI:", "BJ:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        for suffix in (".SS", ".SZ", ".BJ"):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        return token

    @classmethod
    def _normalize_cn_local_code(cls, source_symbol: str) -> str:
        token = cls._strip_symbol_decoration(source_symbol)
        if not _CN_LOCAL_CODE_RE.fullmatch(token):
            raise ValueError(
                f"Invalid CN symbol '{source_symbol}'. "
                "Expected a six-digit mainland A-share ticker with optional .SS/.SZ/.BJ suffix."
            )
        return token

    @staticmethod
    def _infer_exchange_from_code(local_code: str) -> str | None:
        if local_code.startswith(("600", "601", "603", "605", "688")):
            return "SSE"
        if local_code.startswith(("000", "001", "002", "003", "300", "301")):
            return "SZSE"
        if local_code.startswith(("4", "8", "9")):
            return "BSE"
        return None

    @staticmethod
    def _infer_board(local_code: str, exchange: str) -> str:
        if exchange == "SSE" and local_code.startswith("688"):
            return "SSE_STAR"
        if exchange == "SSE":
            return "SSE_MAIN"
        if exchange == "SZSE" and local_code.startswith(("300", "301")):
            return "SZSE_CHINEXT"
        if exchange == "SZSE":
            return "SZSE_MAIN"
        return "BSE"

    @staticmethod
    def _is_excluded_product(raw_row: Mapping[str, Any], *, local_code: str) -> bool:
        if local_code.startswith(("900", "200")):
            return True
        if local_code.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "159", "113", "123", "127", "128")):
            return True
        tokens = " ".join(
            str(raw_row.get(key) or "").strip().upper()
            for key in (
                "security_type",
                "product_type",
                "instrument_type",
                "market_category",
                "category",
                "name",
            )
        )
        return any(token in tokens for token in _EXCLUDED_PRODUCT_TOKENS)

    @staticmethod
    def _parse_market_cap(raw_value: Any) -> float | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float)):
            return float(raw_value)

        raw = str(raw_value).strip().upper().replace(",", "")
        if not raw or raw == "-":
            return None

        multiplier = 1.0
        if raw.endswith("T"):
            multiplier = 1e12
            raw = raw[:-1]
        elif raw.endswith("B"):
            multiplier = 1e9
            raw = raw[:-1]
        elif raw.endswith("M"):
            multiplier = 1e6
            raw = raw[:-1]
        elif raw.endswith("K"):
            multiplier = 1e3
            raw = raw[:-1]

        try:
            return float(raw) * multiplier
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _hash_payload(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _selection_key(row: CNCanonicalUniverseRow) -> tuple[str, int]:
        return (row.source_symbol, row.source_row_number)

    @staticmethod
    def _prefer_text(primary: str, fallback: str) -> str:
        if primary.strip().upper() == "OTHER" and fallback.strip():
            return fallback
        return primary if primary.strip() else fallback

    def _canonical_payload(
        self,
        row: CNCanonicalUniverseRow,
        *,
        name: str,
        sector: str,
        industry_group: str,
        industry: str,
        sub_industry: str,
        market_cap: float | None,
    ) -> dict[str, Any]:
        return {
            "symbol": row.symbol,
            "market": row.market,
            "exchange": row.exchange,
            "board": row.board,
            "local_code": row.local_code,
            "name": name,
            "sector": sector,
            "industry_group": industry_group,
            "industry": industry,
            "sub_industry": sub_industry,
            "market_cap": market_cap,
            "source_name": row.source_name,
            "snapshot_id": row.snapshot_id,
        }

    def _merge_duplicate_rows(
        self,
        first: CNCanonicalUniverseRow,
        second: CNCanonicalUniverseRow,
    ) -> CNCanonicalUniverseRow:
        if self._selection_key(first) <= self._selection_key(second):
            primary = first
            secondary = second
        else:
            primary = second
            secondary = first

        merged_name = self._prefer_text(primary.name, secondary.name)
        merged_sector = self._prefer_text(primary.sector, secondary.sector)
        merged_group = self._prefer_text(primary.industry_group, secondary.industry_group)
        merged_industry = self._prefer_text(primary.industry, secondary.industry)
        merged_sub_industry = self._prefer_text(primary.sub_industry, secondary.sub_industry)
        merged_market_cap = primary.market_cap if primary.market_cap is not None else secondary.market_cap
        merged_source_metadata = (
            primary.source_metadata if primary.source_metadata else secondary.source_metadata
        )
        merged_row_hash = self._hash_payload(
            self._canonical_payload(
                primary,
                name=merged_name,
                sector=merged_sector,
                industry_group=merged_group,
                industry=merged_industry,
                sub_industry=merged_sub_industry,
                market_cap=merged_market_cap,
            )
        )
        return replace(
            primary,
            name=merged_name,
            sector=merged_sector,
            industry_group=merged_group,
            industry=merged_industry,
            sub_industry=merged_sub_industry,
            market_cap=merged_market_cap,
            source_metadata=merged_source_metadata,
            row_hash=merged_row_hash,
        )

    def canonicalize_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> CNCanonicalizationResult:
        normalized_source_name = self.normalize_source_name(source_name)
        if not self.is_approved_source(normalized_source_name):
            raise ValueError(
                f"Unapproved CN source '{source_name}'. "
                "Use an approved CN source identifier."
            )

        normalized_snapshot_id = (snapshot_id or "").strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id must be provided")

        metadata = dict(source_metadata or {})
        canonical_by_symbol: dict[str, CNCanonicalUniverseRow] = {}
        rejected_rows: list[CNRejectedUniverseRow] = []

        for index, raw_row in enumerate(rows, start=1):
            source_symbol = self._normalize_source_symbol(
                raw_row.get("symbol")
                or raw_row.get("local_code")
                or raw_row.get("ticker")
                or raw_row.get("code")
            )
            if not source_symbol:
                rejected_rows.append(
                    CNRejectedUniverseRow(
                        source_row_number=index,
                        source_symbol="",
                        reason="Missing symbol/local_code/ticker/code",
                    )
                )
                continue

            try:
                exchange = self._normalize_exchange(
                    raw_row.get("exchange") or raw_row.get("market"),
                    source_symbol=source_symbol,
                )
                local_code = self._normalize_cn_local_code(source_symbol)
                inferred_exchange = self._infer_exchange_from_code(local_code)
                if inferred_exchange is None or inferred_exchange != exchange:
                    raise ValueError(
                        "Excluded CN non-A-share or non-operating product"
                    )
                if self._is_excluded_product(raw_row, local_code=local_code):
                    raise ValueError("Excluded CN non-A-share or non-operating product")

                board = self._infer_board(local_code, exchange)
                identity = security_master_resolver.resolve_identity(
                    symbol=source_symbol,
                    market="CN",
                    exchange=exchange,
                    local_code=local_code,
                )
                row_name = str(raw_row.get("name") or raw_row.get("company") or "").strip()
                row_group = str(raw_row.get("industry_group") or "").strip()
                row_industry = str(raw_row.get("industry") or raw_row.get("sub_industry") or "").strip()
                row_sub_industry = str(raw_row.get("sub_industry") or raw_row.get("industry") or "").strip()
                row_sector = infer_cn_sector(
                    raw_row.get("sector"),
                    row_group,
                    row_industry,
                    row_sub_industry,
                    board=board,
                )
                row_market_cap = self._parse_market_cap(
                    raw_row.get("market_cap")
                    or raw_row.get("marketcap")
                    or raw_row.get("total_market_cap")
                )

                lineage_payload = {
                    "source_name": normalized_source_name,
                    "snapshot_id": normalized_snapshot_id,
                    "source_row_number": index,
                    "source_symbol": source_symbol,
                    "canonical_symbol": identity.canonical_symbol,
                }
                canonical_payload = {
                    "symbol": identity.canonical_symbol,
                    "market": identity.market,
                    "exchange": exchange,
                    "board": board,
                    "local_code": identity.local_code,
                    "name": row_name,
                    "sector": row_sector,
                    "industry_group": row_group,
                    "industry": row_industry,
                    "sub_industry": row_sub_industry,
                    "market_cap": row_market_cap,
                    "source_name": normalized_source_name,
                    "snapshot_id": normalized_snapshot_id,
                }
                canonical_row = CNCanonicalUniverseRow(
                    symbol=identity.canonical_symbol,
                    name=row_name,
                    market=identity.market,
                    exchange=exchange,
                    board=board,
                    currency=identity.currency,
                    timezone=identity.timezone,
                    local_code=identity.local_code,
                    sector=row_sector,
                    industry_group=row_group,
                    industry=row_industry,
                    sub_industry=row_sub_industry,
                    market_cap=row_market_cap,
                    source_name=normalized_source_name,
                    source_symbol=source_symbol,
                    source_row_number=index,
                    snapshot_id=normalized_snapshot_id,
                    snapshot_as_of=snapshot_as_of,
                    source_metadata=metadata,
                    lineage_hash=self._hash_payload(lineage_payload),
                    row_hash=self._hash_payload(canonical_payload),
                )

                existing = canonical_by_symbol.get(canonical_row.symbol)
                if existing is None:
                    canonical_by_symbol[canonical_row.symbol] = canonical_row
                else:
                    canonical_by_symbol[canonical_row.symbol] = self._merge_duplicate_rows(
                        existing,
                        canonical_row,
                    )
            except ValueError as exc:
                rejected_rows.append(
                    CNRejectedUniverseRow(
                        source_row_number=index,
                        source_symbol=source_symbol,
                        reason=str(exc),
                    )
                )

        canonical_rows = tuple(sorted(canonical_by_symbol.values(), key=lambda row: row.symbol))
        rejected_rows_tuple = tuple(sorted(rejected_rows, key=lambda row: row.source_row_number))
        return CNCanonicalizationResult(
            canonical_rows=canonical_rows,
            rejected_rows=rejected_rows_tuple,
        )


cn_universe_ingestion_adapter = CNUniverseIngestionAdapter()
