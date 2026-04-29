"""KR market universe ingestion adapter with deterministic canonicalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .security_master_service import security_master_resolver

_KR_EXCHANGE_ALIASES: dict[str, str] = {
    "KOSPI": "KOSPI",
    "KRX": "KOSPI",
    "XKRX": "KOSPI",
    "STK": "KOSPI",
    "KOSDAQ": "KOSDAQ",
    "KSQ": "KOSDAQ",
}

_APPROVED_KR_SOURCES: frozenset[str] = frozenset(
    {
        "krx_official",
        "kr_manual_csv",
        "kr_reference_bundle",
    }
)

_KR_LOCAL_CODE_RE = re.compile(r"^[0-9]{6}$")
_EXCLUDED_PRODUCT_TOKENS = (
    "ETF",
    "ETN",
    "ELW",
    "FUND",
    "REIT",
    "SPAC",
    "WARRANT",
)


@dataclass(frozen=True)
class KRCanonicalUniverseRow:
    """Canonical KR row emitted by the ingestion adapter."""

    symbol: str
    name: str
    market: str
    exchange: str
    currency: str
    timezone: str
    local_code: str
    sector: str
    industry: str
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
class KRRejectedUniverseRow:
    """Rejected KR row with reason."""

    source_row_number: int
    source_symbol: str
    reason: str


@dataclass(frozen=True)
class KRCanonicalizationResult:
    """Canonicalization output, split into accepted and rejected rows."""

    canonical_rows: tuple[KRCanonicalUniverseRow, ...]
    rejected_rows: tuple[KRRejectedUniverseRow, ...]


class KRUniverseIngestionAdapter:
    """Normalize and validate KR universe rows for deterministic snapshots."""

    @staticmethod
    def normalize_source_name(source_name: str) -> str:
        normalized = (source_name or "").strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("source_name must be provided")
        return normalized

    @classmethod
    def is_approved_source(cls, source_name: str) -> bool:
        normalized = cls.normalize_source_name(source_name)
        if normalized in _APPROVED_KR_SOURCES:
            return True
        return normalized.startswith("krx_")

    @staticmethod
    def _normalize_source_symbol(raw_symbol: Any) -> str:
        symbol = str(raw_symbol or "").strip().upper().replace(" ", "")
        if symbol.startswith("$"):
            symbol = symbol[1:]
        return symbol

    @staticmethod
    def _infer_exchange_from_symbol(source_symbol: str) -> str | None:
        token = str(source_symbol or "").strip().upper()
        if token.startswith(("KOSDAQ:", "KSQ:")) or token.endswith(".KQ"):
            return "KOSDAQ"
        if token.startswith(("KOSPI:", "KRX:", "XKRX:", "STK:")) or token.endswith(".KS"):
            return "KOSPI"
        return None

    @classmethod
    def _normalize_exchange(cls, raw_exchange: Any, *, source_symbol: str = "") -> str:
        exchange = str(raw_exchange or "").strip().upper()
        if not exchange:
            inferred = cls._infer_exchange_from_symbol(source_symbol)
            return inferred or "KOSPI"
        normalized = _KR_EXCHANGE_ALIASES.get(exchange)
        if normalized is None:
            raise ValueError(
                f"Unsupported KR exchange '{exchange}'. Expected KOSPI or KOSDAQ."
            )
        return normalized

    @staticmethod
    def _normalize_kr_local_code(source_symbol: str) -> str:
        token = source_symbol
        for prefix in ("KOSPI:", "KOSDAQ:", "KRX:", "XKRX:", "STK:", "KSQ:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break

        if token.endswith(".KS") or token.endswith(".KQ"):
            token = token[:-3]

        if not _KR_LOCAL_CODE_RE.fullmatch(token):
            raise ValueError(
                f"Invalid KR symbol '{source_symbol}'. "
                "Expected a six-digit KRX ticker with optional .KS/.KQ suffix."
            )
        return token

    @staticmethod
    def _is_excluded_product(raw_row: Mapping[str, Any]) -> bool:
        tokens = " ".join(
            str(raw_row.get(key) or "").strip().upper()
            for key in (
                "security_type",
                "product_type",
                "instrument_type",
                "market_category",
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

        return float(raw) * multiplier

    @staticmethod
    def _hash_payload(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _selection_key(row: KRCanonicalUniverseRow) -> tuple[str, int]:
        return (row.source_symbol, row.source_row_number)

    @staticmethod
    def _prefer_text(primary: str, fallback: str) -> str:
        return primary if primary.strip() else fallback

    def _canonical_payload(
        self,
        row: KRCanonicalUniverseRow,
        *,
        name: str,
        sector: str,
        industry: str,
        market_cap: float | None,
    ) -> dict[str, Any]:
        return {
            "symbol": row.symbol,
            "market": row.market,
            "exchange": row.exchange,
            "local_code": row.local_code,
            "name": name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "source_name": row.source_name,
            "snapshot_id": row.snapshot_id,
        }

    def _merge_duplicate_rows(
        self,
        first: KRCanonicalUniverseRow,
        second: KRCanonicalUniverseRow,
    ) -> KRCanonicalUniverseRow:
        if self._selection_key(first) <= self._selection_key(second):
            primary = first
            secondary = second
        else:
            primary = second
            secondary = first

        merged_name = self._prefer_text(primary.name, secondary.name)
        merged_sector = self._prefer_text(primary.sector, secondary.sector)
        merged_industry = self._prefer_text(primary.industry, secondary.industry)
        merged_market_cap = primary.market_cap if primary.market_cap is not None else secondary.market_cap
        merged_source_metadata = (
            primary.source_metadata if primary.source_metadata else secondary.source_metadata
        )
        merged_row_hash = self._hash_payload(
            self._canonical_payload(
                primary,
                name=merged_name,
                sector=merged_sector,
                industry=merged_industry,
                market_cap=merged_market_cap,
            )
        )
        return replace(
            primary,
            name=merged_name,
            sector=merged_sector,
            industry=merged_industry,
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
    ) -> KRCanonicalizationResult:
        normalized_source_name = self.normalize_source_name(source_name)
        if not self.is_approved_source(normalized_source_name):
            raise ValueError(
                f"Unapproved KR source '{source_name}'. "
                "Use an approved KR source identifier."
            )

        normalized_snapshot_id = (snapshot_id or "").strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id must be provided")

        metadata = dict(source_metadata or {})
        canonical_by_symbol: dict[str, KRCanonicalUniverseRow] = {}
        rejected_rows: list[KRRejectedUniverseRow] = []

        for index, raw_row in enumerate(rows, start=1):
            source_symbol = self._normalize_source_symbol(
                raw_row.get("symbol")
                or raw_row.get("local_code")
                or raw_row.get("ticker")
            )
            if not source_symbol:
                rejected_rows.append(
                    KRRejectedUniverseRow(
                        source_row_number=index,
                        source_symbol="",
                        reason="Missing symbol/local_code/ticker",
                    )
                )
                continue

            try:
                exchange = self._normalize_exchange(
                    raw_row.get("exchange") or raw_row.get("market"),
                    source_symbol=source_symbol,
                )
                if self._is_excluded_product(raw_row):
                    raise ValueError("Excluded KR non-operating product")

                local_code = self._normalize_kr_local_code(source_symbol)
                identity = security_master_resolver.resolve_identity(
                    symbol=f"{local_code}.KS",
                    market="KR",
                    exchange=exchange,
                    local_code=local_code,
                )
                row_name = str(raw_row.get("name") or raw_row.get("company") or "").strip()
                row_sector = str(raw_row.get("sector") or raw_row.get("industry_group") or "").strip()
                row_industry = str(raw_row.get("industry") or raw_row.get("sub_industry") or "").strip()
                row_market_cap = self._parse_market_cap(
                    raw_row.get("market_cap") or raw_row.get("marketcap")
                )

                lineage_payload = {
                    "source_name": normalized_source_name,
                    "snapshot_id": normalized_snapshot_id,
                    "source_row_number": index,
                    "source_symbol": source_symbol,
                    "canonical_symbol": identity.canonical_symbol,
                }
                canonical_exchange = identity.exchange or exchange
                canonical_payload = {
                    "symbol": identity.canonical_symbol,
                    "market": identity.market,
                    "exchange": canonical_exchange,
                    "local_code": identity.local_code,
                    "name": row_name,
                    "sector": row_sector,
                    "industry": row_industry,
                    "market_cap": row_market_cap,
                    "source_name": normalized_source_name,
                    "snapshot_id": normalized_snapshot_id,
                }
                canonical_row = KRCanonicalUniverseRow(
                    symbol=identity.canonical_symbol,
                    name=row_name,
                    market=identity.market,
                    exchange=canonical_exchange,
                    currency=identity.currency,
                    timezone=identity.timezone,
                    local_code=identity.local_code,
                    sector=row_sector,
                    industry=row_industry,
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
                    KRRejectedUniverseRow(
                        source_row_number=index,
                        source_symbol=source_symbol,
                        reason=str(exc),
                    )
                )

        canonical_rows = tuple(sorted(canonical_by_symbol.values(), key=lambda row: row.symbol))
        rejected_rows_tuple = tuple(sorted(rejected_rows, key=lambda row: row.source_row_number))
        return KRCanonicalizationResult(
            canonical_rows=canonical_rows,
            rejected_rows=rejected_rows_tuple,
        )


kr_universe_ingestion_adapter = KRUniverseIngestionAdapter()
