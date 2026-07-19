from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from app.analysis.patterns.rs_line import blue_dot_series, compute_rs_line
from app.domain.feature_store.run_metadata import feature_run_market
from app.services.preset_screens import (
    PRESET_SCREENS,
    get_preset_chart_symbols,
)


logger = logging.getLogger(__name__)
CHART_BUNDLE_SCHEMA_VERSION = "static-charts-v1"
STATIC_CHART_LIMIT = 200
STATIC_CHART_PERIOD = "6mo"
STATIC_CHART_PERIOD_DAYS = 180
STATIC_CHART_LOOKUP_BATCH_SIZE = 250
STATIC_CHART_PRESET_TOP_N = 200
STATIC_CHART_TOP_N_GROUPS = 50
STATIC_DEFAULT_MARKET = "US"


class StaticChartBundleExporter:
    def __init__(
        self,
        *,
        price_cache,
        fundamentals_cache,
        benchmark_cache,
        json_writer,
        scan_row_serializer,
    ) -> None:
        self._price_cache = price_cache
        self._fundamentals_cache = fundamentals_cache
        self._benchmark_cache = benchmark_cache
        self._write_json = json_writer
        self._serialize_scan_row = scan_row_serializer

    def export(
        self,
        *,
        output_dir: Path,
        generated_at: str,
        run,
        rows: list[Any],
        serialized_rows: list[dict[str, Any]] | None = None,
        path_prefix: Path | None = None,
        groups_payload: dict[str, Any] | None = None,
        preset_screens: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_prefix = Path() if path_prefix is None else Path(path_prefix)
        (output_dir / normalized_prefix / "charts").mkdir(parents=True, exist_ok=True)
        market = feature_run_market(run) or STATIC_DEFAULT_MARKET
        benchmark_symbol, benchmark_df = self._market_benchmark_history(
            market, period="2y"
        )
        entries: list[dict[str, Any]] = []
        skipped_symbols: list[str] = []
        row_by_symbol: dict[str, Any] = {}
        ordered_rows = list(rows)

        def emit(symbol, *, rank, stock_data, price_df, fundamentals_value):
            bars = self._serialize_chart_bars(price_df)
            if not bars:
                skipped_symbols.append(symbol)
                return
            rs_line, blue_dots = self._serialize_rs_line(price_df, benchmark_df)
            rel_path = self._chart_payload_path(
                symbol, path_prefix=normalized_prefix
            )
            self._write_json(
                output_dir / rel_path,
                {
                    "schema_version": CHART_BUNDLE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "as_of_date": run.as_of_date.isoformat(),
                    "symbol": symbol,
                    "rank": rank,
                    "period": STATIC_CHART_PERIOD,
                    "bars": bars,
                    "rs_line": rs_line,
                    "blue_dots": blue_dots,
                    "benchmark_symbol": benchmark_symbol,
                    "stock_data": stock_data,
                    "fundamentals": fundamentals_value,
                },
            )
            entries.append(
                {"symbol": symbol, "rank": rank, "path": rel_path.as_posix()}
            )

        def expand(candidate_symbols, *, log_label):
            extra = sorted(
                candidate_symbols
                - {entry["symbol"] for entry in entries}
                - set(skipped_symbols)
            )
            if not extra:
                return
            before = len(entries)
            serialized_by_symbol = {
                row["symbol"]: row
                for row in (serialized_rows or [])
                if row.get("symbol")
            }
            for row in ordered_rows:
                symbol = getattr(row, "symbol", None)
                if symbol and symbol not in row_by_symbol:
                    row_by_symbol[symbol] = row
            for offset in range(0, len(extra), STATIC_CHART_LOOKUP_BATCH_SIZE):
                batch = extra[offset : offset + STATIC_CHART_LOOKUP_BATCH_SIZE]
                price_data = self._price_cache.get_many_cached_only(batch, period="2y")
                fundamentals = self._fundamentals_cache.get_many_cached_only(batch)
                for symbol in batch:
                    domain_row = row_by_symbol.get(symbol)
                    emit(
                        symbol,
                        rank=None,
                        stock_data=(
                            self._serialize_scan_row(domain_row)
                            if domain_row
                            else serialized_by_symbol.get(symbol)
                        ),
                        price_df=price_data.get(symbol),
                        fundamentals_value=fundamentals.get(symbol),
                    )
            logger.info(
                "%s added %d charts (%d extra symbols attempted)",
                log_label,
                len(entries) - before,
                len(extra),
            )

        if serialized_rows is not None:
            raw_rows = {
                getattr(row, "symbol", None): row
                for row in rows
                if getattr(row, "symbol", None)
            }
            ordered_symbols = [row["symbol"] for row in serialized_rows if row.get("symbol")]
            ordered_rows = [raw_rows[symbol] for symbol in ordered_symbols if symbol in raw_rows]
            seen = {getattr(row, "symbol", None) for row in ordered_rows}
            ordered_rows.extend(
                row for row in rows if getattr(row, "symbol", None) not in seen
            )

        for start in range(0, len(ordered_rows), STATIC_CHART_LOOKUP_BATCH_SIZE):
            if len(entries) >= STATIC_CHART_LIMIT:
                break
            batch_rows = list(
                ordered_rows[start : start + STATIC_CHART_LOOKUP_BATCH_SIZE]
            )
            symbols = [row.symbol for row in batch_rows if getattr(row, "symbol", None)]
            price_data = self._price_cache.get_many_cached_only(symbols, period="2y")
            fundamentals = self._fundamentals_cache.get_many_cached_only(symbols)
            for rank, row in enumerate(batch_rows, start=start + 1):
                if len(entries) >= STATIC_CHART_LIMIT:
                    break
                symbol = getattr(row, "symbol", None)
                if not symbol:
                    continue
                row_by_symbol[symbol] = row
                emit(
                    symbol,
                    rank=rank,
                    stock_data=self._serialize_scan_row(row),
                    price_df=price_data.get(symbol),
                    fundamentals_value=fundamentals.get(symbol),
                )

        if serialized_rows is not None:
            expand(
                get_preset_chart_symbols(
                    serialized_rows,
                    PRESET_SCREENS if preset_screens is None else preset_screens,
                    STATIC_CHART_PRESET_TOP_N,
                ),
                log_label="Preset screen expansion",
            )
        group_symbols = self._top_group_symbols(
            groups_payload=groups_payload,
            top_n=STATIC_CHART_TOP_N_GROUPS,
        )
        if group_symbols:
            expand(
                group_symbols,
                log_label=f"Top-{STATIC_CHART_TOP_N_GROUPS} groups expansion",
            )

        index_path = normalized_prefix / "charts" / "index.json"
        self._write_json(
            output_dir / index_path,
            {
                "schema_version": CHART_BUNDLE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "as_of_date": run.as_of_date.isoformat(),
                "limit": STATIC_CHART_LIMIT,
                "symbols_total": len(entries),
                "skipped_symbols": skipped_symbols,
                "symbols": entries,
            },
        )
        return {
            "path": index_path.as_posix(),
            "limit": STATIC_CHART_LIMIT,
            "symbols_total": len(entries),
            "available": bool(entries),
            "skipped_symbols": skipped_symbols,
        }

    @staticmethod
    def _top_group_symbols(*, groups_payload, top_n: int) -> set[str]:
        if not groups_payload or not groups_payload.get("available"):
            return set()
        details = (groups_payload.get("payload") or {}).get("group_details") or {}
        return {
            stock["symbol"]
            for detail in details.values()
            if isinstance(detail, dict)
            and detail.get("current_rank") is not None
            and detail["current_rank"] <= top_n
            for stock in detail.get("stocks") or []
            if isinstance(stock, dict) and stock.get("symbol")
        }

    def _market_benchmark_history(self, market: str, *, period: str):
        for candidate in self._benchmark_cache.get_benchmark_candidates(market):
            history = self._price_cache.get_cached_only(candidate.upper(), period=period)
            if history is not None and not history.empty:
                return candidate, history
        return self._benchmark_cache.get_benchmark_symbol(market), pd.DataFrame()

    @staticmethod
    def _cutoff(index) -> datetime:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=STATIC_CHART_PERIOD_DAYS)
        index_tz = getattr(index, "tz", None)
        if index_tz is not None:
            return cutoff.tz_convert(index_tz).to_pydatetime()
        return cutoff.tz_localize(None).to_pydatetime()

    def _serialize_rs_line(self, stock_data, benchmark_df):
        if (
            stock_data is None
            or getattr(stock_data, "empty", True)
            or benchmark_df is None
            or benchmark_df.empty
            or "Close" not in benchmark_df.columns
        ):
            return [], []
        line = compute_rs_line(
            stock_data["Close"], benchmark_df["Close"], normalize=True
        )
        blue = blue_dot_series(stock_data["Close"], benchmark_df["Close"])
        cutoff = self._cutoff(line.index)
        line = line[line.index >= cutoff].dropna()
        blue = blue[(blue.index >= cutoff) & blue]
        return (
            [
                {"time": ts.strftime("%Y-%m-%d"), "value": round(float(value), 4)}
                for ts, value in line.items()
            ],
            [ts.strftime("%Y-%m-%d") for ts in blue.index],
        )

    def _serialize_chart_bars(self, data):
        if data is None or getattr(data, "empty", True):
            return []
        filtered = data[data.index >= self._cutoff(data.index)]
        if filtered.empty:
            return []
        frame = filtered.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "Date"})
        frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
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
        ]

    @staticmethod
    def _chart_payload_path(symbol: str, *, path_prefix: Path | None = None) -> Path:
        prefix = Path() if path_prefix is None else Path(path_prefix)
        return prefix / "charts" / f"{quote(symbol, safe='')}.json"
