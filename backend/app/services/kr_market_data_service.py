"""Korea Exchange data access helpers backed by pykrx."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import io
import logging
from typing import Any, Iterable
import zipfile

from defusedxml import ElementTree
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class KrxDependencyError(RuntimeError):
    """Raised when pykrx is not installed but KR market data is requested."""


def _load_pykrx_stock():
    try:
        from pykrx import stock  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise KrxDependencyError("pykrx is required for Korea market data") from exc
    return stock


def _as_yyyymmdd(value: date | str | None) -> str:
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, str) and value.strip():
        token = value.strip().replace("-", "")
        if len(token) == 8 and token.isdigit():
            return token
    return date.today().strftime("%Y%m%d")


def _as_date(value: date | str | None) -> date:
    if isinstance(value, date):
        return value
    token = str(value or "").strip().replace("-", "")
    if len(token) == 8 and token.isdigit():
        return date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
    return date.today()


def _normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _period_start_date(period: str, *, today: date | None = None) -> date:
    """Translate common Yahoo-style periods into a KRX start date."""
    current = today or date.today()
    token = str(period or "2y").strip().lower()
    if token == "max":
        return current - timedelta(days=3650)
    if token.endswith("d") and token[:-1].isdigit():
        return current - timedelta(days=int(token[:-1]))
    if token.endswith("mo") and token[:-2].isdigit():
        return current - timedelta(days=int(token[:-2]) * 31)
    if token.endswith("y") and token[:-1].isdigit():
        return current - timedelta(days=int(token[:-1]) * 365)
    return current - timedelta(days=730)


@dataclass(frozen=True)
class KrxDailyPriceRow:
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    value: float | None = None


class KrxMarketDataService:
    """Fetch KOSPI/KOSDAQ listings, prices, and core valuation fields."""

    def __init__(self, *, stock_module: Any | None = None) -> None:
        self._stock_module = stock_module
        self._market_cap_frames: dict[tuple[str, str], pd.DataFrame | None] = {}
        self._market_fundamental_frames: dict[tuple[str, str], pd.DataFrame | None] = {}

    @property
    def _stock(self):
        if self._stock_module is None:
            self._stock_module = _load_pykrx_stock()
        return self._stock_module

    def listing_rows(
        self,
        *,
        boards: Iterable[str] = ("KOSPI", "KOSDAQ"),
        as_of: date | str | None = None,
    ) -> list[dict[str, Any]]:
        """Return KRX equity listing rows for the requested boards."""
        stock = self._stock
        as_of_token = _as_yyyymmdd(as_of)
        rows: list[dict[str, Any]] = []

        for board in boards:
            normalized_board = str(board or "").strip().upper()
            if normalized_board not in {"KOSPI", "KOSDAQ"}:
                continue
            tickers = stock.get_market_ticker_list(as_of_token, market=normalized_board)
            market_cap_frame = self._market_cap_frame(as_of_token, normalized_board)
            fundamental_frame = self._market_fundamental_frame(as_of_token, normalized_board)

            for ticker in tickers:
                name = stock.get_market_ticker_name(ticker)
                row: dict[str, Any] = {
                    "symbol": f"{ticker}.KS" if normalized_board == "KOSPI" else f"{ticker}.KQ",
                    "local_code": ticker,
                    "name": name,
                    "exchange": normalized_board,
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "source_board": normalized_board,
                }
                if market_cap_frame is not None and ticker in market_cap_frame.index:
                    cap_row = market_cap_frame.loc[ticker]
                    row["market_cap"] = _normalize_float(
                        cap_row.get("시가총액") if hasattr(cap_row, "get") else None
                    )
                    row["shares_outstanding"] = _normalize_float(
                        cap_row.get("상장주식수") if hasattr(cap_row, "get") else None
                    )
                if fundamental_frame is not None and ticker in fundamental_frame.index:
                    f_row = fundamental_frame.loc[ticker]
                    row["pe_ratio"] = _normalize_float(
                        f_row.get("PER") if hasattr(f_row, "get") else None
                    )
                    row["price_to_book"] = _normalize_float(
                        f_row.get("PBR") if hasattr(f_row, "get") else None
                    )
                    row["eps_current"] = _normalize_float(
                        f_row.get("EPS") if hasattr(f_row, "get") else None
                    )
                    row["bps"] = _normalize_float(
                        f_row.get("BPS") if hasattr(f_row, "get") else None
                    )
                    row["dividend_yield"] = _normalize_float(
                        f_row.get("DIV") if hasattr(f_row, "get") else None
                    )
                rows.append(row)

        return rows

    def daily_ohlcv(
        self,
        local_code: str,
        *,
        start: date | str,
        end: date | str | None = None,
    ) -> list[KrxDailyPriceRow]:
        """Return daily OHLCV rows for a KRX local ticker."""
        stock = self._stock
        start_token = _as_yyyymmdd(start)
        end_token = _as_yyyymmdd(end or date.today())
        frame = stock.get_market_ohlcv_by_date(start_token, end_token, local_code)
        if frame is None or frame.empty:
            return []

        result: list[KrxDailyPriceRow] = []
        for index, row in frame.iterrows():
            day = index.date().isoformat() if hasattr(index, "date") else str(index)
            result.append(
                KrxDailyPriceRow(
                    date=day,
                    open=_normalize_float(row.get("시가")),
                    high=_normalize_float(row.get("고가")),
                    low=_normalize_float(row.get("저가")),
                    close=_normalize_float(row.get("종가")),
                    volume=_normalize_float(row.get("거래량")),
                    value=_normalize_float(row.get("거래대금")),
                )
            )
        return result

    def daily_ohlcv_dataframe(
        self,
        local_code: str,
        *,
        period: str = "2y",
        end: date | str | None = None,
    ) -> pd.DataFrame | None:
        """Return KRX OHLCV as the canonical price-cache DataFrame shape."""
        end_date = _as_date(end)
        rows = self.daily_ohlcv(
            local_code,
            start=_period_start_date(period, today=end_date),
            end=end_date,
        )
        if not rows:
            return None

        frame = pd.DataFrame(
            [
                {
                    "Date": row.date,
                    "Open": row.open,
                    "High": row.high,
                    "Low": row.low,
                    "Close": row.close,
                    "Volume": row.volume,
                    "Adj Close": row.close,
                }
                for row in rows
            ]
        )
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date").sort_index()
        return frame

    def core_fundamentals(
        self,
        local_code: str,
        *,
        as_of: date | str | None = None,
    ) -> dict[str, Any]:
        """Return latest KRX valuation fields for one local ticker."""
        as_of_token = _as_yyyymmdd(as_of)
        board = "ALL"
        fields: dict[str, Any] = {}

        cap_frame = self._market_cap_frame(as_of_token, board)
        if cap_frame is not None and local_code in cap_frame.index:
            cap_row = cap_frame.loc[local_code]
            fields["market_cap"] = _normalize_float(cap_row.get("시가총액"))
            fields["shares_outstanding"] = _normalize_float(cap_row.get("상장주식수"))

        fundamental_frame = self._market_fundamental_frame(as_of_token, board)
        if fundamental_frame is not None and local_code in fundamental_frame.index:
            f_row = fundamental_frame.loc[local_code]
            fields.update(
                {
                    "pe_ratio": _normalize_float(f_row.get("PER")),
                    "price_to_book": _normalize_float(f_row.get("PBR")),
                    "eps_current": _normalize_float(f_row.get("EPS")),
                    "bps": _normalize_float(f_row.get("BPS")),
                    "dividend_yield": _normalize_float(f_row.get("DIV")),
                }
            )

        return {key: value for key, value in fields.items() if value is not None}

    def _market_cap_frame(self, as_of_token: str, market: str) -> pd.DataFrame | None:
        key = (as_of_token, market)
        if key not in self._market_cap_frames:
            self._market_cap_frames[key] = self._safe_frame(
                lambda: self._stock.get_market_cap(as_of_token, market=market)
            )
        return self._market_cap_frames[key]

    def _market_fundamental_frame(self, as_of_token: str, market: str) -> pd.DataFrame | None:
        key = (as_of_token, market)
        if key not in self._market_fundamental_frames:
            self._market_fundamental_frames[key] = self._safe_frame(
                lambda: self._stock.get_market_fundamental(as_of_token, market=market)
            )
        return self._market_fundamental_frames[key]

    @staticmethod
    def _safe_frame(fetcher) -> pd.DataFrame | None:
        try:
            frame = fetcher()
        except Exception as exc:  # pragma: no cover - pykrx/network variability
            logger.warning("KRX frame fetch failed: %s", exc)
            return None
        if frame is None or getattr(frame, "empty", True):
            return None
        return frame


class KrxPriceService(KrxMarketDataService):
    """Compatibility alias for price-oriented callers."""


class KrxFundamentalsService(KrxMarketDataService):
    """Compatibility alias for fundamental-oriented callers."""


class OpenDartFundamentalsService:
    """OpenDART enrichment hook for statement-derived KR fields."""

    _CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
    _STATEMENT_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: int = 15,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._corp_code_by_stock_code: dict[str, str] | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_statement_fundamentals(
        self,
        local_code: str,
        *,
        business_year: int | None = None,
        report_code: str = "11011",
    ) -> dict[str, Any]:
        """Return statement-derived fields when OpenDART is configured.

        ``11011`` is OpenDART's annual-report code. The default business year
        is the last completed year because annual statements are not guaranteed
        to exist for the current calendar year.
        """
        if not self.is_configured:
            return {}

        stock_code = str(local_code or "").strip().zfill(6)
        corp_code = self._corp_code_for_stock(stock_code)
        if not corp_code:
            logger.warning("OpenDART corp_code not found for KRX ticker %s", stock_code)
            return {}

        year = int(business_year or (date.today().year - 1))
        response = self._session.get(
            self._STATEMENT_URL,
            params={
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": report_code,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status") or "")
        if status == "013":
            return {}
        if status and status != "000":
            message = payload.get("message") or "OpenDART statement fetch failed"
            raise RuntimeError(f"OpenDART error {status}: {message}")

        fields = self._statement_fields(payload.get("list") or [])
        if fields:
            fields["recent_quarter_date"] = f"{year}-FY"
        return fields

    def _corp_code_for_stock(self, stock_code: str) -> str | None:
        if self._corp_code_by_stock_code is None:
            self._corp_code_by_stock_code = self._fetch_corp_code_map()
        return self._corp_code_by_stock_code.get(stock_code)

    def _fetch_corp_code_map(self) -> dict[str, str]:
        response = self._session.get(
            self._CORP_CODE_URL,
            params={"crtfc_key": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            if not names:
                return {}
            xml_bytes = archive.read(names[0])

        root = ElementTree.fromstring(xml_bytes)
        mapping: dict[str, str] = {}
        for item in root.findall("list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            corp_code = (item.findtext("corp_code") or "").strip()
            if stock_code and corp_code:
                mapping[stock_code.zfill(6)] = corp_code
        return mapping

    @classmethod
    def _statement_fields(cls, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        values: dict[str, float] = {}
        for item in items:
            account_name = str(item.get("account_nm") or "").strip()
            amount = cls._normalize_statement_amount(item.get("thstrm_amount"))
            if not account_name or amount is None:
                continue
            values.setdefault(account_name, amount)

        revenue = cls._first_value(
            values,
            "매출액",
            "수익(매출액)",
            "영업수익",
            "Revenue",
            "Sales",
        )
        operating_income = cls._first_value(values, "영업이익", "Operating income")
        net_income = cls._first_value(
            values,
            "당기순이익",
            "연결당기순이익",
            "Profit (loss)",
            "Net income",
        )
        gross_profit = cls._first_value(values, "매출총이익", "Gross profit")
        assets = cls._first_value(values, "자산총계", "Total assets")
        liabilities = cls._first_value(values, "부채총계", "Total liabilities")
        equity = cls._first_value(values, "자본총계", "Total equity")
        current_assets = cls._first_value(values, "유동자산", "Current assets")
        current_liabilities = cls._first_value(values, "유동부채", "Current liabilities")

        fields: dict[str, Any] = {}
        if revenue is not None:
            fields["revenue_current"] = int(revenue)
            if operating_income is not None:
                fields["operating_margin"] = cls._ratio_pct(operating_income, revenue)
            if net_income is not None:
                fields["profit_margin"] = cls._ratio_pct(net_income, revenue)
            if gross_profit is not None:
                fields["gross_margin"] = cls._ratio_pct(gross_profit, revenue)
        if net_income is not None and equity:
            fields["roe"] = cls._ratio_pct(net_income, equity)
        if net_income is not None and assets:
            fields["roa"] = cls._ratio_pct(net_income, assets)
        if liabilities is not None and equity:
            fields["debt_to_equity"] = cls._ratio_pct(liabilities, equity)
        if current_assets is not None and current_liabilities:
            fields["current_ratio"] = current_assets / current_liabilities

        return {key: value for key, value in fields.items() if value is not None}

    @staticmethod
    def _normalize_statement_amount(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if text in {"", "-", "N/A", "nan"}:
            return None
        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("()")
        try:
            amount = float(text)
        except ValueError:
            return None
        return -amount if negative else amount

    @staticmethod
    def _first_value(values: dict[str, float], *names: str) -> float | None:
        for name in names:
            if name in values:
                return values[name]
        return None

    @staticmethod
    def _ratio_pct(numerator: float, denominator: float) -> float | None:
        if denominator == 0:
            return None
        return (numerator / denominator) * 100


def recent_kr_business_day() -> date:
    """Best-effort fallback for KRX calls that need a recent listing date."""
    today = date.today()
    if today.weekday() >= 5:
        return today - timedelta(days=today.weekday() - 4)
    return today
