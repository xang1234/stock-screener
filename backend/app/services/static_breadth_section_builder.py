"""Static breadth payload assembly extracted from the main exporter."""

from datetime import date, datetime, timedelta
import math
from typing import Any

import pandas as pd

from app.services.breadth_attribution_service import BreadthAttributionService
from app.services.static_market_artifact_contract import STATIC_SITE_SCHEMA_VERSION
from app.services.static_site_errors import StaticSiteSectionUnavailableError

STATIC_BREADTH_HISTORY_LOOKBACK_DAYS = 90
STATIC_BREADTH_ATTRIBUTION_LOOKBACK_DAYS = 10
STATIC_BREADTH_ATTRIBUTION_MARKETS = ("US",)
STATIC_DEFAULT_MARKET = "US"
STATIC_CHART_LOOKUP_BATCH_SIZE = 250


def _coerce_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class StaticBreadthSectionBuilder:
    def __init__(self, *, ui_snapshot_service, price_cache, benchmark_cache) -> None:
        self._ui_snapshot_service = ui_snapshot_service
        self._price_cache = price_cache
        self._benchmark_cache = benchmark_cache

    def build(self, **kwargs):
        return self._build_breadth_payload(**kwargs)

    def _build_breadth_payload(
        self,
        *,
        generated_at: str,
        expected_as_of_date: date,
        market: str = STATIC_DEFAULT_MARKET,
        serialized_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if serialized_rows is None:
            snapshot = self._ui_snapshot_service.publish_breadth_bootstrap(market=market).to_dict()
            payload = snapshot.get("payload", {})
            current_date = ((payload.get("current") or {}).get("date"))
            if current_date != expected_as_of_date.isoformat():
                raise StaticSiteSectionUnavailableError(
                    section="breadth",
                    reason=(
                        "No breadth snapshot is available for static-site export date "
                        f"{expected_as_of_date.isoformat()} (latest snapshot date: {current_date or 'none'})."
                    ),
                )
            return {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "available": True,
                "published_at": _coerce_datetime(snapshot.get("published_at")),
                "source_revision": snapshot.get("source_revision"),
                "payload": payload,
            }

        symbols = [row["symbol"] for row in serialized_rows if row.get("symbol")]
        if not symbols:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=f"No scan rows are available for market {market} on {expected_as_of_date.isoformat()}.",
            )

        benchmark_symbol, benchmark = self._get_market_benchmark_history(market, period="1y")
        if benchmark.empty:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=f"No cached benchmark price history is available for market {market}.",
            )

        canonical_dates = [
            ts.date()
            for ts in pd.to_datetime(benchmark.index)
            if ts.date() <= expected_as_of_date
        ]
        if expected_as_of_date not in canonical_dates:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=(
                    f"No benchmark trading session is available for market {market} "
                    f"on {expected_as_of_date.isoformat()}."
                ),
            )

        canonical_dates = canonical_dates[-max(STATIC_BREADTH_HISTORY_LOOKBACK_DAYS + 15, 120):]
        price_data = self._get_cached_price_histories(symbols, period="1y")
        metrics_by_date = self._compute_breadth_metrics_by_date(canonical_dates, price_data)
        current = metrics_by_date.get(expected_as_of_date)
        if current is None:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=f"No breadth snapshot could be derived for market {market} on {expected_as_of_date.isoformat()}.",
            )

        ordered_dates = sorted(metrics_by_date.keys())
        ordered_history = [
            {**metrics_by_date[item_date], "market": market}
            for item_date in ordered_dates
        ]
        chart_data = ordered_history[-31:]
        current = {**current, "market": market}
        benchmark_overlay = self._serialize_history_bars(
            benchmark,
            period_days=31,
            end_date=expected_as_of_date,
        )
        group_attribution = self._build_group_attribution(
            market=market,
            serialized_rows=serialized_rows,
            price_data=price_data,
            ordered_dates=ordered_dates,
        )
        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "available": True,
            "published_at": _coerce_datetime(datetime.utcnow()),
            "source_revision": f"feature-run:{market}:{expected_as_of_date.isoformat()}",
            "market": market,
            "payload": {
                "current": current,
                "summary": {
                    "market": market,
                    "latest_date": expected_as_of_date.isoformat(),
                    "total_records": len(ordered_history),
                    "date_range_start": ordered_dates[0].isoformat() if ordered_dates else None,
                    "date_range_end": ordered_dates[-1].isoformat() if ordered_dates else None,
                },
                "history_90d": list(reversed(ordered_history[-STATIC_BREADTH_HISTORY_LOOKBACK_DAYS:])),
                "chart_range": "1M",
                "chart_data": list(reversed(chart_data)),
                "benchmark_symbol": benchmark_symbol,
                "benchmark_overlay": benchmark_overlay,
                "spy_overlay": benchmark_overlay,
                "group_attribution": group_attribution,
            },
        }

    def _build_group_attribution(
        self,
        *,
        market: str,
        serialized_rows: list[dict[str, Any]],
        price_data: dict[str, pd.DataFrame | None],
        ordered_dates: list[date],
    ) -> dict[str, Any]:
        """Attribute ±4% movers to IBD industry groups for the most recent sessions.

        Only enabled for markets in ``STATIC_BREADTH_ATTRIBUTION_MARKETS`` — non-US
        taxonomies aren't wired in for the first cut. Returns an
        ``{available: False, reason}`` payload when skipped so the static client
        can hide the feature cleanly.
        """
        if market not in STATIC_BREADTH_ATTRIBUTION_MARKETS:
            return {
                "available": False,
                "reason": f"Group attribution is not yet supported for market {market}.",
            }
        if not ordered_dates:
            return {
                "available": False,
                "reason": "No trading dates were available to attribute.",
            }

        attribution_dates = ordered_dates[-STATIC_BREADTH_ATTRIBUTION_LOOKBACK_DAYS:]
        symbols_meta = [
            {
                "symbol": row.get("symbol"),
                "company_name": row.get("company_name"),
                "ibd_industry_group": row.get("ibd_industry_group"),
            }
            for row in serialized_rows
            if row.get("symbol")
        ]
        service = BreadthAttributionService()
        history = service.compute(
            symbols_meta=symbols_meta,
            price_data=price_data,
            target_dates=attribution_dates,
        )
        has_any_mover = any(
            (day.get("stocks_up_4pct", 0) + day.get("stocks_down_4pct", 0)) > 0
            for day in history
        )
        if not history or not has_any_mover:
            return {
                "available": False,
                "reason": "No 4%+ movers were attributable for the lookback window.",
            }

        latest = history[-1]
        return {
            "available": True,
            "market": market,
            "threshold_pct": 4.0,
            "lookback_days": STATIC_BREADTH_ATTRIBUTION_LOOKBACK_DAYS,
            "latest_date": latest["date"] if latest else None,
            "history": list(reversed(history)),
        }

    def _get_cached_price_histories(
        self,
        symbols: list[str],
        *,
        period: str,
    ) -> dict[str, pd.DataFrame | None]:
        results: dict[str, pd.DataFrame | None] = {}
        for start in range(0, len(symbols), STATIC_CHART_LOOKUP_BATCH_SIZE):
            batch = symbols[start:start + STATIC_CHART_LOOKUP_BATCH_SIZE]
            results.update(self._price_cache.get_many_cached_only(batch, period=period))
        return results

    def _get_market_benchmark_history(self, market: str, *, period: str) -> tuple[str, pd.DataFrame]:
        for candidate in self._benchmark_cache.get_benchmark_candidates(market):
            history = self._get_symbol_price_history(candidate, period=period)
            if history is not None and not history.empty:
                return candidate, history
        return self._benchmark_cache.get_benchmark_symbol(market), pd.DataFrame()

    def _get_symbol_price_history(self, symbol: str, *, period: str) -> pd.DataFrame | None:
        data = self._price_cache.get_cached_only(symbol.upper(), period=period)
        if data is None or data.empty:
            return None
        return data

    def _compute_breadth_metrics_by_date(
        self,
        canonical_dates: list[date],
        price_data: dict[str, pd.DataFrame | None],
    ) -> dict[date, dict[str, Any]]:
        if not canonical_dates:
            return {}

        date_index = pd.Index(canonical_dates)
        def empty() -> list[int]:
            return [0] * len(canonical_dates)

        aggregates = {
            "stocks_up_4pct": empty(),
            "stocks_down_4pct": empty(),
            "stocks_up_25pct_quarter": empty(),
            "stocks_down_25pct_quarter": empty(),
            "stocks_up_25pct_month": empty(),
            "stocks_down_25pct_month": empty(),
            "stocks_up_50pct_month": empty(),
            "stocks_down_50pct_month": empty(),
            "stocks_up_13pct_34days": empty(),
            "stocks_down_13pct_34days": empty(),
            "total_stocks_scanned": empty(),
        }

        for history in price_data.values():
            if history is None or history.empty or "Close" not in history.columns:
                continue
            close_series = pd.Series(
                history["Close"].to_numpy(),
                index=[ts.date() for ts in pd.to_datetime(history.index)],
            )
            close_series = close_series[~close_series.index.duplicated(keep="last")].sort_index()
            pct_1d = ((close_series / close_series.shift(1)) - 1.0) * 100.0
            pct_21d = ((close_series / close_series.shift(21)) - 1.0) * 100.0
            pct_34d = ((close_series / close_series.shift(34)) - 1.0) * 100.0
            pct_63d = ((close_series / close_series.shift(63)) - 1.0) * 100.0

            close_series = close_series.reindex(date_index)
            pct_1d = pct_1d.reindex(date_index)
            pct_21d = pct_21d.reindex(date_index)
            pct_34d = pct_34d.reindex(date_index)
            pct_63d = pct_63d.reindex(date_index)
            valid = close_series.notna().to_numpy()

            for index, is_valid in enumerate(valid):
                if is_valid:
                    aggregates["total_stocks_scanned"][index] += 1

            for key, series in (
                ("stocks_up_4pct", pct_1d >= 4.0),
                ("stocks_down_4pct", pct_1d <= -4.0),
                ("stocks_up_25pct_month", pct_21d >= 25.0),
                ("stocks_down_25pct_month", pct_21d <= -25.0),
                ("stocks_up_50pct_month", pct_21d >= 50.0),
                ("stocks_down_50pct_month", pct_21d <= -50.0),
                ("stocks_up_13pct_34days", pct_34d >= 13.0),
                ("stocks_down_13pct_34days", pct_34d <= -13.0),
                ("stocks_up_25pct_quarter", pct_63d >= 25.0),
                ("stocks_down_25pct_quarter", pct_63d <= -25.0),
            ):
                flags = series.fillna(False).to_numpy()
                for index, flag in enumerate(flags):
                    if flag and valid[index]:
                        aggregates[key][index] += 1

        results: dict[date, dict[str, Any]] = {}
        for index, item_date in enumerate(canonical_dates):
            ratio_5day = None
            ratio_10day = None
            if index >= 5:
                up_5 = sum(aggregates["stocks_up_4pct"][max(index - 5, 0):index])
                down_5 = sum(aggregates["stocks_down_4pct"][max(index - 5, 0):index])
                ratio_5day = round(up_5 / down_5, 2) if down_5 > 0 else None
            if index >= 10:
                up_10 = sum(aggregates["stocks_up_4pct"][index - 10:index])
                down_10 = sum(aggregates["stocks_down_4pct"][index - 10:index])
                ratio_10day = round(up_10 / down_10, 2) if down_10 > 0 else None

            results[item_date] = {
                "date": item_date.isoformat(),
                "stocks_up_4pct": int(aggregates["stocks_up_4pct"][index]),
                "stocks_down_4pct": int(aggregates["stocks_down_4pct"][index]),
                "ratio_5day": ratio_5day,
                "ratio_10day": ratio_10day,
                "stocks_up_25pct_quarter": int(aggregates["stocks_up_25pct_quarter"][index]),
                "stocks_down_25pct_quarter": int(aggregates["stocks_down_25pct_quarter"][index]),
                "stocks_up_25pct_month": int(aggregates["stocks_up_25pct_month"][index]),
                "stocks_down_25pct_month": int(aggregates["stocks_down_25pct_month"][index]),
                "stocks_up_50pct_month": int(aggregates["stocks_up_50pct_month"][index]),
                "stocks_down_50pct_month": int(aggregates["stocks_down_50pct_month"][index]),
                "stocks_up_13pct_34days": int(aggregates["stocks_up_13pct_34days"][index]),
                "stocks_down_13pct_34days": int(aggregates["stocks_down_13pct_34days"][index]),
                "total_stocks_scanned": int(aggregates["total_stocks_scanned"][index]),
            }
        return results

    @staticmethod
    def _serialize_close_history(data: pd.DataFrame | None, *, days: int) -> list[dict[str, Any]]:
        if data is None or data.empty or "Close" not in data.columns:
            return []
        frame = data.tail(days).reset_index()
        date_col = frame.columns[0]
        frame = frame.rename(columns={date_col: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
        return [
            {
                "date": row["Date"],
                "close": round(float(row["Close"]), 2),
            }
            for _, row in frame.iterrows()
            if row["Close"] is not None and not math.isnan(float(row["Close"]))
        ]

    @staticmethod
    def _serialize_history_bars(
        data: pd.DataFrame | None,
        *,
        period_days: int,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        if data is None or data.empty:
            return []
        end_timestamp = pd.Timestamp(end_date or datetime.utcnow())
        cutoff_date = end_timestamp - timedelta(days=period_days)
        if data.index.tz is not None:
            cutoff_date = cutoff_date.tz_localize(data.index.tz)
            end_timestamp = end_timestamp.tz_localize(data.index.tz)
        filtered = data[(data.index >= cutoff_date) & (data.index <= end_timestamp)]
        if filtered.empty:
            return []
        frame = filtered.reset_index()
        date_col = frame.columns[0]
        frame = frame.rename(columns={date_col: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
        return [
            {
                "date": row["Date"],
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
            for _, row in frame.iterrows()
            if all(
                value is not None and not math.isnan(float(value))
                for value in (row["Open"], row["High"], row["Low"], row["Close"], row["Volume"])
            )
        ]
