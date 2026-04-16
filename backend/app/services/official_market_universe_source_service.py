"""Fetch and parse official exchange universe snapshots for HK, JP, and TW."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
import io
import logging
import re
from typing import Any, Iterable

from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib3

from ..config import settings

logger = logging.getLogger(__name__)


_HK_SOURCE_NAME = "hkex_official"
_JP_SOURCE_NAME = "jpx_official"
_TW_SOURCE_NAME = "tw_reference_bundle"

_HK_EQUITY_CATEGORY = "equity"
_HK_EQUITY_SUBCATEGORY_TOKEN = "equity securities"
_JP_ALLOWED_MARKET_SECTIONS = frozenset(
    {
        "プライム（内国株式）",
        "スタンダード（内国株式）",
        "グロース（内国株式）",
    }
)
_TW_UPDATED_AT_RE = re.compile(r"Date\s+Stock\s+Updated:\s*(\d{4}/\d{2}/\d{2})", re.IGNORECASE)
_TW_CODE_NAME_RE = re.compile(r"^([0-9A-Z]{3,6}[A-Z]?)\s+(.+?)$")


@dataclass(frozen=True)
class OfficialMarketUniverseSnapshot:
    """Canonical upstream snapshot ready for stock_universe ingest."""

    market: str
    source_name: str
    snapshot_id: str
    snapshot_as_of: str
    source_metadata: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _FetchedSource:
    url: str
    content: bytes
    fetched_at: str
    last_modified: str | None
    tls_verification_disabled: bool


class OfficialMarketUniverseSourceService:
    """Fetch official exchange listings and normalize them into ingest rows."""

    def __init__(
        self,
        *,
        timeout_seconds: int | None = None,
        user_agent: str | None = None,
    ) -> None:
        self._timeout_seconds = int(
            timeout_seconds or settings.universe_source_timeout_seconds
        )
        self._user_agent = user_agent or settings.universe_source_user_agent

    def fetch_market_snapshot(self, market: str) -> OfficialMarketUniverseSnapshot:
        normalized_market = str(market or "").strip().upper()
        if normalized_market == "HK":
            return self.fetch_hk_snapshot()
        if normalized_market == "JP":
            return self.fetch_jp_snapshot()
        if normalized_market == "TW":
            return self.fetch_tw_snapshot()
        raise ValueError(f"Official universe refresh is unsupported for market {market!r}")

    def fetch_hk_snapshot(self) -> OfficialMarketUniverseSnapshot:
        fetched = self._http_get(settings.hk_universe_source_url)
        fallback_as_of = self._date_from_http_header(fetched.last_modified) or self._utc_today()
        rows = self.parse_hk_rows(fetched.content)
        snapshot_as_of = fallback_as_of.isoformat()
        source_metadata = {
            "source_urls": [settings.hk_universe_source_url],
            "fetched_at": fetched.fetched_at,
            "http_last_modified": fetched.last_modified,
            "tls_verification_disabled": fetched.tls_verification_disabled,
            "filters": {
                "category_equals": "Equity",
                "sub_category_contains": "Equity Securities",
            },
            "sheet_name": "ListOfSecurities",
        }
        return OfficialMarketUniverseSnapshot(
            market="HK",
            source_name=_HK_SOURCE_NAME,
            snapshot_id=f"hkex-listofsecurities-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple(rows),
        )

    def fetch_jp_snapshot(self) -> OfficialMarketUniverseSnapshot:
        fetched = self._http_get(settings.jp_universe_source_url)
        frame = self._read_excel_bytes(fetched.content, engine="xlrd")
        rows, snapshot_date = self._parse_jp_frame(frame)
        snapshot_as_of = snapshot_date.isoformat()
        source_metadata = {
            "source_urls": [settings.jp_universe_source_url],
            "fetched_at": fetched.fetched_at,
            "http_last_modified": fetched.last_modified,
            "tls_verification_disabled": fetched.tls_verification_disabled,
            "filters": {
                "market_product_category_in": sorted(_JP_ALLOWED_MARKET_SECTIONS),
            },
        }
        return OfficialMarketUniverseSnapshot(
            market="JP",
            source_name=_JP_SOURCE_NAME,
            snapshot_id=f"jpx-data-j-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple(rows),
        )

    def fetch_tw_snapshot(self) -> OfficialMarketUniverseSnapshot:
        twse_fetched = self._http_get(
            settings.tw_universe_source_twse_url,
            allow_insecure_fallback=settings.tw_universe_allow_insecure_fallback,
        )
        tpex_fetched = self._http_get(
            settings.tw_universe_source_tpex_url,
            allow_insecure_fallback=settings.tw_universe_allow_insecure_fallback,
        )

        twse_html = twse_fetched.content.decode("cp950", errors="replace")
        tpex_html = tpex_fetched.content.decode("cp950", errors="replace")
        twse_date = self._parse_tw_updated_at(twse_html)
        tpex_date = self._parse_tw_updated_at(tpex_html)
        if twse_date != tpex_date:
            raise ValueError(
                "TW reference bundle date mismatch between TWSE and TPEx "
                f"({twse_date.isoformat()} vs {tpex_date.isoformat()})"
            )

        twse_rows = self.parse_tw_rows(twse_html, exchange="TWSE")
        tpex_rows = self.parse_tw_rows(tpex_html, exchange="TPEX")
        snapshot_as_of = twse_date.isoformat()
        source_metadata = {
            "source_urls": [
                settings.tw_universe_source_twse_url,
                settings.tw_universe_source_tpex_url,
            ],
            "fetched_at": max(twse_fetched.fetched_at, tpex_fetched.fetched_at),
            "http_last_modified": {
                "twse": twse_fetched.last_modified,
                "tpex": tpex_fetched.last_modified,
            },
            "tls_verification_disabled": {
                "twse": twse_fetched.tls_verification_disabled,
                "tpex": tpex_fetched.tls_verification_disabled,
            },
            "filters": {
                "sections": ["Stocks"],
                "exchanges": ["TWSE", "TPEX"],
            },
            "row_counts": {
                "twse": len(twse_rows),
                "tpex": len(tpex_rows),
            },
        }
        return OfficialMarketUniverseSnapshot(
            market="TW",
            source_name=_TW_SOURCE_NAME,
            snapshot_id=f"tw-reference-bundle-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple([*twse_rows, *tpex_rows]),
        )

    def parse_hk_rows(self, content: bytes) -> list[dict[str, Any]]:
        frame = pd.read_excel(
            io.BytesIO(content),
            sheet_name="ListOfSecurities",
            header=2,
        )
        normalized = self._normalize_columns(frame)
        self._require_columns(
            normalized.columns,
            required=("stock_code", "name_of_securities", "category"),
            market="HK",
        )
        category = normalized["category"].astype(str).str.strip().str.lower()
        filtered = normalized[category == _HK_EQUITY_CATEGORY].copy()
        if "sub_category" in filtered.columns:
            sub_category = filtered["sub_category"].astype(str).str.strip().str.lower()
            filtered = filtered[
                sub_category.eq("")
                | sub_category.eq("nan")
                | sub_category.str.contains(_HK_EQUITY_SUBCATEGORY_TOKEN, na=False)
            ].copy()

        rows: list[dict[str, Any]] = []
        for row in filtered.to_dict("records"):
            local_code = self._normalize_digits(row.get("stock_code"), pad_to=4, max_digits=5)
            name = str(row.get("name_of_securities") or "").strip()
            if not local_code or not name:
                continue
            rows.append(
                {
                    "symbol": f"{local_code}.HK",
                    "name": name,
                    "exchange": "XHKG",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                }
            )

        if not rows:
            raise ValueError("HK official universe parse returned no equity rows")
        return rows

    def parse_jp_rows(self, content: bytes) -> OfficialMarketUniverseSnapshot:
        frame = self._read_excel_bytes(content, engine="xlrd")
        rows, snapshot_date = self._parse_jp_frame(frame)
        return OfficialMarketUniverseSnapshot(
            market="JP",
            source_name=_JP_SOURCE_NAME,
            snapshot_id=f"jpx-data-j-{snapshot_date.isoformat()}",
            snapshot_as_of=snapshot_date.isoformat(),
            source_metadata={},
            rows=tuple(rows),
        )

    def parse_tw_rows(self, html: str, *, exchange: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        current_section = ""
        rows: list[dict[str, Any]] = []

        for tr in soup.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue

            texts = [self._collapse_whitespace(td.get_text(" ", strip=True)) for td in cells]
            if len(cells) == 1 and texts:
                current_section = texts[0]
                continue

            if not self._is_tw_stock_section(current_section):
                continue

            first_cell = texts[0] if texts else ""
            if not first_cell or first_cell.lower().startswith("code"):
                continue

            parsed_identity = self._parse_tw_code_name(first_cell)
            if parsed_identity is None:
                continue
            code, name = parsed_identity

            industry = ""
            if len(texts) >= 5:
                industry = texts[4]
            elif len(texts) >= 4:
                industry = texts[3]

            rows.append(
                {
                    "symbol": f"{code}.TW" if exchange == "TWSE" else f"{code}.TWO",
                    "name": name,
                    "exchange": exchange,
                    "sector": industry,
                    "industry": industry,
                    "market_cap": None,
                }
            )

        if not rows:
            raise ValueError(f"TW official universe parse returned no stock rows for {exchange}")
        return rows

    def _parse_jp_frame(self, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], date]:
        normalized = self._normalize_columns(frame)
        self._require_columns(
            normalized.columns,
            required=("日付", "コード", "銘柄名", "市場・商品区分"),
            market="JP",
        )
        category = normalized["市場・商品区分"].astype(str).str.strip()
        filtered = normalized[category.isin(_JP_ALLOWED_MARKET_SECTIONS)].copy()

        dates = {
            self._coerce_date(value)
            for value in filtered["日付"].tolist()
            if self._coerce_date(value) is not None
        }
        if not dates:
            raise ValueError("JP official universe parse could not determine snapshot date")
        if len(dates) != 1:
            raise ValueError(f"JP official universe parse saw multiple snapshot dates: {sorted(d.isoformat() for d in dates)}")
        snapshot_date = next(iter(dates))

        rows: list[dict[str, Any]] = []
        for row in filtered.to_dict("records"):
            code = self._normalize_digits(row.get("コード"), pad_to=4, max_digits=5)
            name = str(row.get("銘柄名") or "").strip()
            if not code or not name:
                continue
            industry = str(row.get("33業種区分") or "").strip()
            sector = str(row.get("17業種区分") or "").strip()
            rows.append(
                {
                    "symbol": f"{code}.T",
                    "name": name,
                    "exchange": "XTKS",
                    "sector": sector,
                    "industry": industry,
                    "market_cap": None,
                }
            )

        if not rows:
            raise ValueError("JP official universe parse returned no domestic equity rows")
        return rows, snapshot_date

    def _http_get(
        self,
        url: str,
        *,
        allow_insecure_fallback: bool = False,
    ) -> _FetchedSource:
        headers = {"User-Agent": self._user_agent}
        fetched_at = datetime.now(UTC).isoformat()
        try:
            response = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            response.raise_for_status()
            return _FetchedSource(
                url=response.url or url,
                content=response.content,
                fetched_at=fetched_at,
                last_modified=response.headers.get("Last-Modified"),
                tls_verification_disabled=False,
            )
        except requests.exceptions.SSLError:
            if not allow_insecure_fallback:
                raise
            logger.warning(
                "Retrying official universe fetch with TLS verification disabled for %s",
                url,
            )
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(
                url,
                headers=headers,
                timeout=self._timeout_seconds,
                verify=False,
            )
            response.raise_for_status()
            return _FetchedSource(
                url=response.url or url,
                content=response.content,
                fetched_at=fetched_at,
                last_modified=response.headers.get("Last-Modified"),
                tls_verification_disabled=True,
            )

    @staticmethod
    def _read_excel_bytes(content: bytes, *, engine: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, Any] = {}
        if engine:
            kwargs["engine"] = engine
        return pd.read_excel(io.BytesIO(content), **kwargs)

    @staticmethod
    def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        renamed = frame.copy()
        renamed.columns = [OfficialMarketUniverseSourceService._normalize_column_name(col) for col in renamed.columns]
        return renamed

    @staticmethod
    def _normalize_column_name(value: Any) -> str:
        text = OfficialMarketUniverseSourceService._collapse_whitespace(str(value or ""))
        text = text.replace("/", "_").replace("-", "_")
        text = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text.lower()

    @staticmethod
    def _require_columns(columns: Iterable[str], *, required: tuple[str, ...], market: str) -> None:
        present = set(columns)
        missing = [column for column in required if column not in present]
        if missing:
            raise ValueError(
                f"{market} official universe source missing required columns: {', '.join(missing)}"
            )

    @staticmethod
    def _normalize_digits(value: Any, *, pad_to: int, max_digits: int) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        match = re.search(r"\d+", raw)
        if match is None:
            return ""
        digits = match.group(0).lstrip("0")
        if not digits:
            return ""
        if len(digits) > max_digits:
            raise ValueError(f"Unexpected local code length for {raw!r}")
        if len(digits) < pad_to:
            return digits.zfill(pad_to)
        return digits

    @staticmethod
    def _coerce_date(value: Any) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()

    @staticmethod
    def _utc_today() -> date:
        return datetime.now(UTC).date()

    @staticmethod
    def _date_from_http_header(header_value: str | None) -> date | None:
        if not header_value:
            return None
        try:
            return parsedate_to_datetime(header_value).astimezone(UTC).date()
        except Exception:
            return None

    @staticmethod
    def _collapse_whitespace(value: str) -> str:
        return " ".join(
            str(value or "").replace("\u3000", " ").replace("\xa0", " ").split()
        )

    @staticmethod
    def _parse_tw_updated_at(html: str) -> date:
        match = _TW_UPDATED_AT_RE.search(html)
        if match is None:
            raise ValueError("TW official universe source missing 'Date Stock Updated' banner")
        return datetime.strptime(match.group(1), "%Y/%m/%d").date()

    @staticmethod
    def _is_tw_stock_section(section_label: str) -> bool:
        normalized = OfficialMarketUniverseSourceService._collapse_whitespace(section_label).lower()
        return normalized in {"stocks", "stock"} or normalized.startswith("stocks ")

    @staticmethod
    def _parse_tw_code_name(cell_text: str) -> tuple[str, str] | None:
        match = _TW_CODE_NAME_RE.match(
            OfficialMarketUniverseSourceService._collapse_whitespace(cell_text)
        )
        if match is None:
            return None
        return match.group(1), match.group(2)
