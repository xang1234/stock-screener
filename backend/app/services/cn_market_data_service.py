"""Mainland China A-share data access helpers backed by no-key providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import importlib
import logging
import signal
import threading
from typing import Any

import pandas as pd
import requests

from ..config import settings
from .cn_universe_ingestion_adapter import infer_cn_sector

logger = logging.getLogger(__name__)


class CnDependencyError(RuntimeError):
    """Raised when a CN market dependency is unavailable."""


def _normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    raw = str(value).strip().replace(",", "")
    if not raw or raw == "-":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _first_numeric(row: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        getter = getattr(row, "get", None)
        value = getter(key) if callable(getter) else None
        numeric = _normalize_float(value)
        if numeric is not None:
            return numeric
    return None


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


def _period_start_date(period: str, *, today: date | None = None) -> date:
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


def _normalize_code(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw.endswith((".SS", ".SZ", ".BJ")):
        raw = raw[:-3]
    return raw.zfill(6) if raw.isdigit() else raw


def _exchange_for_code(local_code: str) -> str | None:
    if local_code.startswith(("600", "601", "603", "605", "688")):
        return "SSE"
    if local_code.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZSE"
    if local_code.startswith(("4", "8", "9")):
        return "BJSE"
    return None


def _board_for_code(local_code: str, exchange: str) -> str:
    if exchange == "SSE" and local_code.startswith("688"):
        return "SSE_STAR"
    if exchange == "SSE":
        return "SSE_MAIN"
    if exchange == "SZSE" and local_code.startswith(("300", "301")):
        return "SZSE_CHINEXT"
    if exchange == "SZSE":
        return "SZSE_MAIN"
    return "BSE"


def _suffix_for_exchange(exchange: str) -> str:
    if exchange == "SZSE":
        return ".SZ"
    if exchange == "BJSE":
        return ".BJ"
    return ".SS"


def _call_with_timeout(fetcher, *, timeout_seconds: int, operation_name: str):
    if timeout_seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return fetcher()
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return fetcher()

    def _raise_timeout(signum, frame):
        del signum, frame
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        return fetcher()
    except TimeoutError as exc:
        raise requests.exceptions.Timeout(str(exc)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class CnDailyPriceRow:
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    value: float | None = None


class CnMarketDataService:
    """Fetch mainland A-share listings, prices, and core valuation fields."""

    def __init__(
        self,
        *,
        akshare_module: Any | None = None,
        baostock_module: Any | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self._akshare_module = akshare_module
        self._baostock_module = baostock_module
        self._timeout_seconds = int(
            timeout_seconds or settings.universe_source_timeout_seconds
        )
        self._listing_rows_cache: list[dict[str, Any]] | None = None

    @property
    def _akshare(self):
        if self._akshare_module is None:
            try:
                self._akshare_module = importlib.import_module("akshare")
            except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
                raise CnDependencyError("akshare is required for China market data") from exc
        return self._akshare_module

    @property
    def _baostock(self):
        if self._baostock_module is None:
            try:
                self._baostock_module = importlib.import_module("baostock")
            except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
                raise CnDependencyError("baostock is required for China fallback market data") from exc
        return self._baostock_module

    def listing_rows(self, *, as_of: date | str | None = None) -> list[dict[str, Any]]:
        """Return A-share listing rows from AKShare-backed no-key sources."""
        del as_of  # CN listing sources return the current source snapshot.
        if self._listing_rows_cache is not None:
            return [dict(row) for row in self._listing_rows_cache]

        frame = self._listing_frame()
        if frame is None or frame.empty:
            return []

        rows = self._rows_from_listing_frame(frame)
        self._listing_rows_cache = rows
        return [dict(row) for row in rows]

    def _listing_frame(self) -> pd.DataFrame | None:
        spot_error: Exception | None = None
        try:
            frame = self._akshare_spot_frame()
            if frame is not None and not frame.empty:
                return frame
        except CnDependencyError:
            raise
        except Exception as exc:
            spot_error = exc
            logger.warning("AKShare CN spot listing fetch failed: %s", exc)

        fallback_fetcher = getattr(self._akshare, "stock_info_a_code_name", None)
        if callable(fallback_fetcher):
            try:
                frame = _call_with_timeout(
                    fallback_fetcher,
                    timeout_seconds=self._timeout_seconds,
                    operation_name="CN A-share code-name fetch",
                )
                if frame is not None and not frame.empty:
                    logger.info("Using AKShare CN code-name listing fallback")
                    return frame
            except Exception as exc:
                logger.warning("AKShare CN code-name listing fallback failed: %s", exc)
                if spot_error is None:
                    raise

        if spot_error is not None:
            raise spot_error
        return None

    def _akshare_spot_frame(self) -> pd.DataFrame | None:
        frame = _call_with_timeout(
            self._akshare.stock_zh_a_spot_em,
            timeout_seconds=self._timeout_seconds,
            operation_name="CN A-share listing fetch",
        )
        if frame is None or frame.empty:
            logger.warning("AKShare CN spot listing fetch returned no rows")
        return frame

    def _rows_from_listing_frame(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in frame.to_dict("records"):
            local_code = _normalize_code(raw.get("代码") or raw.get("code") or raw.get("symbol"))
            exchange = _exchange_for_code(local_code)
            if exchange is None:
                continue
            suffix = _suffix_for_exchange(exchange)
            latest_price = _first_numeric(raw, ("最新价", "price"))
            market_cap = _first_numeric(raw, ("总市值", "market_cap"))
            shares_outstanding = None
            if market_cap is not None and latest_price is not None and latest_price > 0:
                shares_outstanding = market_cap / latest_price
            industry = str(raw.get("所属行业") or raw.get("行业") or raw.get("industry") or "").strip()
            board = _board_for_code(local_code, exchange)
            sector = infer_cn_sector(raw.get("sector"), raw.get("industry_group"), industry, board=board)
            rows.append(
                {
                    "symbol": f"{local_code}{suffix}",
                    "local_code": local_code,
                    "name": str(raw.get("名称") or raw.get("name") or "").strip(),
                    "exchange": exchange,
                    "board": board,
                    "sector": sector,
                    "industry_group": industry,
                    "industry": industry,
                    "sub_industry": industry,
                    "market_cap": market_cap,
                    "float_market_cap": _first_numeric(raw, ("流通市值", "float_market_cap")),
                    "shares_outstanding": shares_outstanding,
                    "pe_ratio": _first_numeric(raw, ("市盈率-动态", "市盈率", "pe_ratio")),
                    "price_to_book": _first_numeric(raw, ("市净率", "price_to_book")),
                    "dividend_yield": _first_numeric(raw, ("股息率", "dividend_yield")),
                    "source_board": _board_for_code(local_code, exchange),
                }
            )
        return rows

    def core_fundamentals(
        self,
        local_code: str,
        *,
        as_of: date | str | None = None,
    ) -> dict[str, Any]:
        """Return latest AKShare/Eastmoney valuation fields for one A-share."""
        code = _normalize_code(local_code)
        fields: dict[str, Any] = {}
        for row in self.listing_rows(as_of=as_of):
            if row.get("local_code") != code:
                continue
            fields.update(
                {
                    "market_cap": row.get("market_cap"),
                    "shares_outstanding": row.get("shares_outstanding"),
                    "pe_ratio": row.get("pe_ratio"),
                    "price_to_book": row.get("price_to_book"),
                    "dividend_yield": row.get("dividend_yield"),
                    "sector": row.get("sector"),
                    "industry": row.get("industry"),
                }
            )
            break
        return {key: value for key, value in fields.items() if value not in (None, "")}

    def statement_fundamentals(
        self,
        local_code: str,
        *,
        as_of: date | str | None = None,
    ) -> dict[str, Any]:
        """Return statement-style fields from AKShare, with BaoStock fallback."""
        code = _normalize_code(local_code)
        try:
            ak_fields = self._statement_fundamentals_from_akshare(code, as_of=as_of)
            if ak_fields:
                return ak_fields
        except CnDependencyError:
            raise
        except Exception as exc:  # pragma: no cover - network/API variability
            logger.warning("AKShare CN fundamentals fetch failed for %s: %s", code, exc)

        try:
            return self._statement_fundamentals_from_baostock(code, as_of=as_of)
        except CnDependencyError:
            raise
        except Exception as exc:  # pragma: no cover - network/API variability
            logger.warning("BaoStock CN fundamentals fetch failed for %s: %s", code, exc)
            return {}

    def _statement_fundamentals_from_akshare(
        self,
        local_code: str,
        *,
        as_of: date | str | None = None,
    ) -> dict[str, Any]:
        year = _as_date(as_of).year if as_of is not None else date.today().year
        fetcher = getattr(self._akshare, "stock_financial_analysis_indicator", None)
        if not callable(fetcher):
            return {}
        frame = fetcher(symbol=local_code, start_year=str(max(1990, year - 2)))
        if frame is None or frame.empty:
            return {}
        row = frame.iloc[-1]
        fields = {
            "roe": _first_numeric(row, ("净资产收益率", "摊薄净资产收益率", "ROE")),
            "roa": _first_numeric(row, ("总资产报酬率", "资产报酬率", "ROA")),
            "profit_margin": _first_numeric(row, ("销售净利率", "净利率", "profit_margin")),
            "gross_margin": _first_numeric(row, ("销售毛利率", "毛利率", "gross_margin")),
            "debt_to_equity": _first_numeric(row, ("产权比率", "资产负债率", "debt_to_equity")),
            "current_ratio": _first_numeric(row, ("流动比率", "current_ratio")),
            "eps_current": _first_numeric(row, ("基本每股收益", "每股收益", "EPS")),
        }
        return {key: value for key, value in fields.items() if value is not None}

    def _statement_fundamentals_from_baostock(
        self,
        local_code: str,
        *,
        as_of: date | str | None = None,
    ) -> dict[str, Any]:
        bs = self._baostock
        exchange = _exchange_for_code(local_code)
        if exchange is None or exchange == "BJSE":
            return {}
        bs_prefix = "sh" if exchange == "SSE" else "sz"
        bs_code = f"{bs_prefix}.{local_code}"
        target_date = _as_date(as_of)
        year = target_date.year
        quarter = max(1, min(4, (target_date.month - 1) // 3 + 1))
        login_result = bs.login()
        if getattr(login_result, "error_code", "0") != "0":
            return {}
        try:
            fields: dict[str, Any] = {}
            profit = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
            if getattr(profit, "error_code", "0") == "0" and profit.next():
                row = dict(zip(getattr(profit, "fields", []) or (), profit.get_row_data()))
                fields.update(
                    {
                        "roe": _first_numeric(row, ("roeAvg", "roe")),
                        "eps_current": _first_numeric(row, ("epsTTM", "eps")),
                        "profit_margin": _first_numeric(row, ("netProfitMargin",)),
                        "gross_margin": _first_numeric(row, ("grossProfitMargin",)),
                    }
                )
            operation = bs.query_operation_data(code=bs_code, year=year, quarter=quarter)
            if getattr(operation, "error_code", "0") == "0" and operation.next():
                row = dict(zip(getattr(operation, "fields", []) or (), operation.get_row_data()))
                fields.update(
                    {
                        "current_ratio": _first_numeric(row, ("currentRatio",)),
                        "quick_ratio": _first_numeric(row, ("quickRatio",)),
                    }
                )
            return {key: value for key, value in fields.items() if value is not None}
        finally:
            try:
                bs.logout()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def daily_ohlcv(
        self,
        local_code: str,
        *,
        start: date | str,
        end: date | str | None = None,
    ) -> list[CnDailyPriceRow]:
        """Return daily OHLCV rows from AKShare, with BaoStock fallback."""
        code = _normalize_code(local_code)
        start_token = _as_yyyymmdd(start)
        end_token = _as_yyyymmdd(end or date.today())
        try:
            frame = self._akshare.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_token,
                end_date=end_token,
                adjust="",
            )
            rows = self._daily_rows_from_akshare_frame(frame)
            if rows:
                return rows
        except CnDependencyError:
            raise
        except Exception as exc:  # pragma: no cover - network variability
            logger.warning("AKShare CN OHLCV fetch failed for %s: %s", code, exc)

        return self._daily_ohlcv_from_baostock(code, start=start_token, end=end_token)

    @staticmethod
    def _daily_rows_from_akshare_frame(frame: pd.DataFrame | None) -> list[CnDailyPriceRow]:
        if frame is None or frame.empty:
            return []
        result: list[CnDailyPriceRow] = []
        for _, row in frame.iterrows():
            result.append(
                CnDailyPriceRow(
                    date=str(row.get("日期") or row.get("date") or "")[:10],
                    open=_first_numeric(row, ("开盘", "open")),
                    high=_first_numeric(row, ("最高", "high")),
                    low=_first_numeric(row, ("最低", "low")),
                    close=_first_numeric(row, ("收盘", "close")),
                    volume=_first_numeric(row, ("成交量", "volume")),
                    value=_first_numeric(row, ("成交额", "amount")),
                )
            )
        return [row for row in result if row.date]

    def _daily_ohlcv_from_baostock(
        self,
        local_code: str,
        *,
        start: str,
        end: str,
    ) -> list[CnDailyPriceRow]:
        bs = self._baostock
        exchange = _exchange_for_code(local_code)
        if exchange is None or exchange == "BJSE":
            return []
        bs_prefix = "sh" if exchange == "SSE" else "sz"
        bs_code = f"{bs_prefix}.{local_code}"
        login_result = bs.login()
        if getattr(login_result, "error_code", "0") != "0":
            return []
        try:
            query = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=f"{start[:4]}-{start[4:6]}-{start[6:8]}",
                end_date=f"{end[:4]}-{end[4:6]}-{end[6:8]}",
                frequency="d",
                adjustflag="3",
            )
            rows: list[CnDailyPriceRow] = []
            while getattr(query, "error_code", "0") == "0" and query.next():
                item = query.get_row_data()
                rows.append(
                    CnDailyPriceRow(
                        date=item[0],
                        open=_normalize_float(item[1]),
                        high=_normalize_float(item[2]),
                        low=_normalize_float(item[3]),
                        close=_normalize_float(item[4]),
                        volume=_normalize_float(item[5]),
                        value=_normalize_float(item[6]),
                    )
                )
            return rows
        finally:
            try:
                bs.logout()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def daily_ohlcv_dataframe(
        self,
        local_code: str,
        *,
        period: str = "2y",
        end: date | str | None = None,
    ) -> pd.DataFrame | None:
        """Return CN OHLCV as the canonical price-cache DataFrame shape."""
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
                    "Adj Close": row.close,
                    "Volume": row.volume,
                }
                for row in rows
            ]
        )
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date").sort_index()
        return frame[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
