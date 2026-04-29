"""Shared market-taxonomy loader for US/HK/IN/JP/KR/TW group classifications."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .security_master_service import security_master_resolver

_HK_LOCAL_CODE_RE = re.compile(r"^[0-9]{1,8}$")
_IN_LOCAL_CODE_RE = re.compile(r"^[0-9A-Z-]{1,16}$")
_JP_LOCAL_CODE_RE = re.compile(r"^[0-9]{3,5}[A-Z]?$")
_KR_LOCAL_CODE_RE = re.compile(r"^[0-9]{6}$")
_TW_LOCAL_CODE_RE = re.compile(r"^[0-9]{3,6}[A-Z]?$")

_TW_EXCHANGE_ALIASES = {
    "TWSE": "TWSE",
    "XTAI": "TWSE",
    "TPEX": "TPEX",
    "TWO": "TPEX",
    "TPEX MARKET": "TPEX",
    "TPEX STOCK": "TPEX",
    "TPEX STOCKS": "TPEX",
    "TPEx": "TPEX",
}

_EMPTY_TOKENS = {"", "-", "N/A", "NA", "NAN", "NONE", "NULL"}
_LOAD_EXCEPTIONS = (OSError, csv.Error, UnicodeError)


class TaxonomyLoadError(RuntimeError):
    """Raised when committed taxonomy source files are missing or malformed."""


@dataclass(frozen=True)
class MarketTaxonomyEntry:
    """Normalized market taxonomy for one symbol."""

    market: str
    symbol: str
    industry_group: str | None = None
    sector: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    themes: tuple[str, ...] = ()

    def themes_list(self) -> list[str]:
        return list(self.themes)


class MarketTaxonomyService:
    """Load market-aware group/theme mappings from committed CSV files."""

    def __init__(self, *, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or self._default_data_dir()
        self._loaded = False
        self._entries: dict[str, dict[str, MarketTaxonomyEntry]] = {
            "US": {},
            "HK": {},
            "IN": {},
            "JP": {},
            "KR": {},
            "TW": {},
        }

    @staticmethod
    def _default_data_dir(*, service_path: Path | None = None) -> Path:
        resolved_path = (service_path or Path(__file__)).resolve()
        candidates: list[Path] = []
        for parent_index in (3, 2):
            try:
                candidates.append(resolved_path.parents[parent_index] / "data")
            except IndexError:
                continue
        candidates.append(Path.cwd() / "data")

        required_files = (
            "IBD_industry_group.csv",
            "hk-deep.csv",
            "india-deep.csv",
            "kabutan_themes_en.csv",
            "korea-deep.csv",
            "taiwan-deep.csv",
        )
        unique_candidates = list(dict.fromkeys(candidates))
        best_candidate = max(
            unique_candidates,
            key=lambda candidate: sum((candidate / filename).exists() for filename in required_files),
        )
        return best_candidate

    def refresh(self) -> None:
        self._entries = {"US": {}, "HK": {}, "IN": {}, "JP": {}, "KR": {}, "TW": {}}
        try:
            self._load_us()
            self._load_hk()
            self._load_in()
            self._load_jp()
            self._load_kr()
            self._load_tw()
        except TaxonomyLoadError:
            self._loaded = False
            raise
        except _LOAD_EXCEPTIONS as exc:
            self._loaded = False
            raise TaxonomyLoadError(
                f"Unable to load market taxonomy from {self._data_dir}: {exc}"
            ) from exc
        self._loaded = True

    def get(
        self,
        symbol: str | None,
        *,
        market: str | None = None,
        exchange: str | None = None,
    ) -> MarketTaxonomyEntry | None:
        self._ensure_loaded()
        normalized_market = security_master_resolver.normalize_market(market)
        if normalized_market is None:
            normalized_market = security_master_resolver.infer_market(symbol or "", exchange)

        candidates = self._candidate_symbols(symbol, market=normalized_market, exchange=exchange)
        market_entries = self._entries.get(normalized_market, {})
        for candidate in candidates:
            entry = market_entries.get(candidate)
            if entry is not None:
                return entry
        return None

    def groups_for_market(self, market: str) -> list[str]:
        """Return sorted distinct industry_group values for one market.

        Feeds IBDGroupRankService for non-US markets — US-side continues to
        query its persisted `ibd_industry_groups` table.
        """
        self._ensure_loaded()
        normalized = security_master_resolver.normalize_market(market) or market.upper()
        entries = self._entries.get(normalized, {})
        groups = {
            entry.industry_group
            for entry in entries.values()
            if entry.industry_group is not None
        }
        return sorted(groups)

    def symbols_for_group(self, market: str, group: str) -> list[str]:
        """Return sorted symbols that belong to a given industry group in a market."""
        self._ensure_loaded()
        normalized = security_master_resolver.normalize_market(market) or market.upper()
        entries = self._entries.get(normalized, {})
        return sorted(
            entry.symbol
            for entry in entries.values()
            if entry.industry_group == group
        )

    def entry_count_for_market(self, market: str) -> int:
        """Return the committed taxonomy row count loaded for one market."""
        self._ensure_loaded()
        normalized = security_master_resolver.normalize_market(market) or market.upper()
        return len(self._entries.get(normalized, {}))

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.refresh()

    @staticmethod
    def _normalize_text(value: object) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text.upper() in _EMPTY_TOKENS:
            return None
        return text

    @staticmethod
    def _require_columns(path: Path, reader: csv.DictReader, required: tuple[str, ...]) -> None:
        fieldnames = set(reader.fieldnames or ())
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise TaxonomyLoadError(
                f"Taxonomy CSV {path} is missing required columns: {', '.join(missing)}"
            )

    def _merge_entry(
        self,
        *,
        market: str,
        symbol: str,
        industry_group: str | None,
        sector: str | None,
        industry: str | None,
        sub_industry: str | None = None,
        themes: Iterable[str],
    ) -> None:
        bucket = self._entries[market]
        normalized_themes = []
        for theme in themes:
            normalized = self._normalize_text(theme)
            if normalized is not None:
                normalized_themes.append(normalized)

        current = bucket.get(symbol)
        if current is None:
            bucket[symbol] = MarketTaxonomyEntry(
                market=market,
                symbol=symbol,
                industry_group=industry_group,
                sector=sector,
                industry=industry,
                sub_industry=sub_industry,
                themes=tuple(dict.fromkeys(normalized_themes)),
            )
            return

        merged_themes = tuple(dict.fromkeys([*current.themes, *normalized_themes]))
        bucket[symbol] = replace(
            current,
            industry_group=current.industry_group or industry_group,
            sector=current.sector or sector,
            industry=current.industry or industry,
            sub_industry=current.sub_industry or sub_industry,
            themes=merged_themes,
        )

    def _load_us(self) -> None:
        path = self._data_dir / "IBD_industry_group.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                symbol = security_master_resolver.normalize_symbol(row[0])
                industry_group = self._normalize_text(row[1])
                if not symbol or industry_group is None:
                    continue
                self._merge_entry(
                    market="US",
                    symbol=symbol,
                    industry_group=industry_group,
                    sector=None,
                    industry=None,
                    themes=(),
                )

    def _load_hk(self) -> None:
        path = self._data_dir / "hk-deep.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._require_columns(path, reader, ("Symbol", "EM Industry (EN)", "Theme (EN)"))
            for row in reader:
                symbol = self._canonicalize_hk_symbol(row.get("Symbol"))
                if symbol is None:
                    continue
                self._merge_entry(
                    market="HK",
                    symbol=symbol,
                    industry_group=self._normalize_text(row.get("EM Industry (EN)")),
                    sector=None,
                    industry=None,
                    themes=[row.get("Theme (EN)") or ""],
                )

    def _load_in(self) -> None:
        path = self._data_dir / "india-deep.csv"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._require_columns(
                path,
                reader,
                ("Symbol", "Exchange", "Industry (Sector)", "Subgroup (Theme)", "Sub-industry"),
            )
            for row in reader:
                symbol = self._canonicalize_in_symbol(
                    row.get("Symbol"),
                    exchange=row.get("Exchange"),
                )
                if symbol is None:
                    continue
                self._merge_entry(
                    market="IN",
                    symbol=symbol,
                    industry_group=self._normalize_text(row.get("Subgroup (Theme)")),
                    sector=self._normalize_text(row.get("Industry (Sector)")),
                    industry=self._normalize_text(row.get("Sub-industry")),
                    themes=(),
                )

    def _load_jp(self) -> None:
        path = self._data_dir / "kabutan_themes_en.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._require_columns(
                path,
                reader,
                ("Symbol", "TSE 33-Sector", "TSE 17-Sector", "Theme (EN)"),
            )
            for row in reader:
                symbol = self._canonicalize_jp_symbol(row.get("Symbol"))
                if symbol is None:
                    continue
                self._merge_entry(
                    market="JP",
                    symbol=symbol,
                    industry_group=self._normalize_text(row.get("TSE 33-Sector")),
                    sector=self._normalize_text(row.get("TSE 17-Sector")),
                    industry=None,
                    themes=[row.get("Theme (EN)") or ""],
                )

    def _load_tw(self) -> None:
        path = self._data_dir / "taiwan-deep.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._require_columns(path, reader, ("Symbol", "Market", "Industry (EN)"))
            for row in reader:
                symbol = self._canonicalize_tw_symbol(
                    row.get("Symbol"),
                    exchange=row.get("Market"),
                )
                if symbol is None:
                    continue
                self._merge_entry(
                    market="TW",
                    symbol=symbol,
                    industry_group=self._normalize_text(row.get("Industry (EN)")),
                    sector=None,
                    industry=None,
                    themes=(),
                )

    def _load_kr(self) -> None:
        path = self._data_dir / "korea-deep.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._require_columns(
                path,
                reader,
                ("Symbol", "Market", "Sector", "Industry Group", "Industry", "Sub-Industry"),
            )
            for row in reader:
                symbol = self._canonicalize_kr_symbol(
                    row.get("Symbol"),
                    exchange=row.get("Market"),
                )
                if symbol is None:
                    continue
                self._merge_entry(
                    market="KR",
                    symbol=symbol,
                    industry_group=self._normalize_text(row.get("Industry Group")),
                    sector=self._normalize_text(row.get("Sector")),
                    industry=self._normalize_text(row.get("Industry")),
                    sub_industry=self._normalize_text(row.get("Sub-Industry")),
                    themes=(),
                )

    def _candidate_symbols(
        self,
        symbol: str | None,
        *,
        market: str,
        exchange: str | None = None,
    ) -> tuple[str, ...]:
        normalized = security_master_resolver.normalize_symbol(symbol)
        if not normalized:
            return ()
        if market == "US":
            return (normalized,)
        if market == "HK":
            canonical = self._canonicalize_hk_symbol(normalized)
            return (canonical,) if canonical else (normalized,)
        if market == "IN":
            candidates: list[str] = []
            canonical = self._canonicalize_in_symbol(normalized, exchange=exchange)
            if canonical:
                candidates.append(canonical)
            if "." not in normalized:
                normalized_exchange = security_master_resolver.normalize_exchange(exchange)
                if normalized_exchange not in {"BSE", "XBOM"}:
                    nse_variant = self._canonicalize_in_symbol(normalized, exchange="XNSE")
                    if nse_variant:
                        candidates.append(nse_variant)
                if normalized_exchange not in {"NSE", "XNSE"}:
                    bse_variant = self._canonicalize_in_symbol(normalized, exchange="XBOM")
                    if bse_variant:
                        candidates.append(bse_variant)
            candidates.append(normalized)
            return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))
        if market == "JP":
            canonical = self._canonicalize_jp_symbol(normalized)
            return (canonical,) if canonical else (normalized,)
        if market == "KR":
            candidates: list[str] = []
            canonical = self._canonicalize_kr_symbol(normalized, exchange=exchange)
            if canonical:
                candidates.append(canonical)
            if "." not in normalized:
                raw_exchange = str(exchange or "").strip().upper()
                if raw_exchange in {"KRX", "XKRX", "STK"}:
                    raw_exchange = ""
                if raw_exchange not in {"KOSDAQ", "KSQ"}:
                    kospi_variant = self._canonicalize_kr_symbol(normalized, exchange="KOSPI")
                    if kospi_variant:
                        candidates.append(kospi_variant)
                if raw_exchange not in {"KOSPI", "KRX", "XKRX", "STK"}:
                    kosdaq_variant = self._canonicalize_kr_symbol(normalized, exchange="KOSDAQ")
                    if kosdaq_variant:
                        candidates.append(kosdaq_variant)
            elif normalized.endswith(".KS"):
                kosdaq_variant = self._canonicalize_kr_symbol(normalized[:-3], exchange="KOSDAQ")
                if kosdaq_variant:
                    candidates.append(kosdaq_variant)
            elif normalized.endswith(".KQ"):
                kospi_variant = self._canonicalize_kr_symbol(normalized[:-3], exchange="KOSPI")
                if kospi_variant:
                    candidates.append(kospi_variant)
            candidates.append(normalized)
            return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))
        if market == "TW":
            candidates: list[str] = []
            canonical = self._canonicalize_tw_symbol(normalized, exchange=exchange)
            if canonical:
                candidates.append(canonical)
            if "." not in normalized:
                raw_exchange = str(exchange or "").strip().upper()
                if raw_exchange not in {"TPEX", "TWO"}:
                    tpex_variant = self._canonicalize_tw_symbol(normalized, exchange="TPEX")
                    if tpex_variant:
                        candidates.append(tpex_variant)
                twse_variant = self._canonicalize_tw_symbol(normalized, exchange="TWSE")
                if twse_variant:
                    candidates.append(twse_variant)
            elif normalized.endswith(".TW"):
                tpex_variant = self._canonicalize_tw_symbol(normalized[:-3], exchange="TPEX")
                if tpex_variant:
                    candidates.append(tpex_variant)
            elif normalized.endswith(".TWO"):
                twse_variant = self._canonicalize_tw_symbol(normalized[:-4], exchange="TWSE")
                if twse_variant:
                    candidates.append(twse_variant)
            candidates.append(normalized)
            return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))
        return (normalized,)

    @staticmethod
    def _canonicalize_hk_symbol(raw_symbol: object) -> str | None:
        token = security_master_resolver.normalize_symbol(str(raw_symbol or ""))
        if not token:
            return None
        for prefix in ("HKEX:", "SEHK:", "XHKG:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        if token.endswith(".HK"):
            token = token[:-3]
        if not _HK_LOCAL_CODE_RE.fullmatch(token):
            return None
        significant = token.lstrip("0")
        if not significant:
            return None
        if len(significant) <= 4:
            local_code = significant.zfill(4)
        elif len(significant) == 5:
            local_code = significant
        else:
            return None
        identity = security_master_resolver.resolve_identity(
            symbol=f"{local_code}.HK",
            market="HK",
            exchange="XHKG",
            local_code=local_code,
        )
        return identity.canonical_symbol

    @staticmethod
    def _canonicalize_in_symbol(
        raw_symbol: object,
        *,
        exchange: object | None = None,
    ) -> str | None:
        token = security_master_resolver.normalize_symbol(str(raw_symbol or ""))
        if not token:
            return None
        for prefix in ("NSE:", "XNSE:", "BSE:", "XBOM:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        if token.endswith(".NS"):
            token = token[:-3]
            exchange = "XNSE"
        elif token.endswith(".BO"):
            token = token[:-3]
            exchange = "XBOM"
        if not _IN_LOCAL_CODE_RE.fullmatch(token):
            return None
        identity = security_master_resolver.resolve_identity(
            symbol=token,
            market="IN",
            exchange=exchange,
            local_code=token,
        )
        return identity.canonical_symbol

    @staticmethod
    def _canonicalize_jp_symbol(raw_symbol: object) -> str | None:
        token = security_master_resolver.normalize_symbol(str(raw_symbol or ""))
        if not token:
            return None
        for prefix in ("TSE:", "JPX:", "XTKS:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        if token.endswith(".JP"):
            token = token[:-3]
        elif token.endswith(".T"):
            token = token[:-2]
        if not _JP_LOCAL_CODE_RE.fullmatch(token):
            return None
        identity = security_master_resolver.resolve_identity(
            symbol=f"{token}.T",
            market="JP",
            exchange="XTKS",
            local_code=token,
        )
        return identity.canonical_symbol

    @staticmethod
    def _canonicalize_kr_symbol(
        raw_symbol: object,
        *,
        exchange: object | None = None,
    ) -> str | None:
        token = security_master_resolver.normalize_symbol(str(raw_symbol or ""))
        if not token:
            return None
        for prefix in ("KOSPI:", "KOSDAQ:", "KRX:", "XKRX:", "STK:", "KSQ:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        normalized_exchange = str(exchange or "").strip().upper()
        if token.endswith(".KS"):
            token = token[:-3]
            normalized_exchange = "KOSPI"
        elif token.endswith(".KQ"):
            token = token[:-3]
            normalized_exchange = "KOSDAQ"
        if normalized_exchange in {"", "KRX", "XKRX", "STK"}:
            normalized_exchange = "KOSPI"
        elif normalized_exchange == "KSQ":
            normalized_exchange = "KOSDAQ"
        if normalized_exchange not in {"KOSPI", "KOSDAQ"}:
            return None
        if not _KR_LOCAL_CODE_RE.fullmatch(token):
            return None
        identity = security_master_resolver.resolve_identity(
            symbol=f"{token}.KS",
            market="KR",
            exchange=normalized_exchange,
            local_code=token,
        )
        return identity.canonical_symbol

    @classmethod
    def _canonicalize_tw_symbol(
        cls,
        raw_symbol: object,
        *,
        exchange: object | None = None,
    ) -> str | None:
        token = security_master_resolver.normalize_symbol(str(raw_symbol or ""))
        if not token:
            return None
        for prefix in ("TWSE:", "XTAI:", "TPEX:", "TWO:"):
            if token.startswith(prefix):
                token = token[len(prefix):]
                break
        normalized_exchange = cls._normalize_tw_exchange(exchange, source_symbol=token)
        if token.endswith(".TWO"):
            token = token[:-4]
            normalized_exchange = "TPEX"
        elif token.endswith(".TW"):
            token = token[:-3]
            normalized_exchange = "TWSE"
        if not _TW_LOCAL_CODE_RE.fullmatch(token):
            return None
        identity = security_master_resolver.resolve_identity(
            symbol=f"{token}.TW",
            market="TW",
            exchange=normalized_exchange,
            local_code=token,
        )
        return identity.canonical_symbol

    @classmethod
    def _normalize_tw_exchange(
        cls,
        raw_exchange: object | None,
        *,
        source_symbol: str,
    ) -> str:
        exchange = str(raw_exchange or "").strip()
        if exchange:
            normalized = _TW_EXCHANGE_ALIASES.get(exchange)
            if normalized is not None:
                return normalized
            normalized = _TW_EXCHANGE_ALIASES.get(exchange.upper())
            if normalized is not None:
                return normalized
        normalized_symbol = source_symbol.upper()
        if normalized_symbol.endswith(".TWO") or normalized_symbol.startswith(("TPEX:", "TWO:")):
            return "TPEX"
        return "TWSE"


_market_taxonomy_service: MarketTaxonomyService | None = None


def get_market_taxonomy_service() -> MarketTaxonomyService:
    global _market_taxonomy_service
    if _market_taxonomy_service is None:
        _market_taxonomy_service = MarketTaxonomyService()
    return _market_taxonomy_service


__all__ = [
    "MarketTaxonomyEntry",
    "MarketTaxonomyService",
    "TaxonomyLoadError",
    "get_market_taxonomy_service",
]
