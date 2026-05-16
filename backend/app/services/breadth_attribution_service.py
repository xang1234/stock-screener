"""Service for attributing daily breadth movers (4% up/down) to IBD industry groups.

Given the symbol universe for a market and their cached price histories, this
service classifies each ±4% daily mover into its IBD industry group (US) so the
breadth UI can show "what is driving the breadth" per session. Symbols with no
IBD group assignment fall into the synthetic "No Group" bucket.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Iterable, Mapping

import pandas as pd

logger = logging.getLogger(__name__)


NO_GROUP_LABEL = "No Group"
MOVER_THRESHOLD_PCT = 4.0


class BreadthAttributionService:
    """Compute per-date IBD-group attribution for ±4% daily movers."""

    def __init__(self, *, threshold_pct: float = MOVER_THRESHOLD_PCT) -> None:
        self._threshold_pct = float(threshold_pct)

    def compute(
        self,
        *,
        symbols_meta: Iterable[Mapping[str, Any]],
        price_data: Mapping[str, pd.DataFrame | None],
        target_dates: Iterable[date],
    ) -> list[dict[str, Any]]:
        """Return attribution rows ordered oldest → newest.

        Args:
            symbols_meta: Iterable of dicts with at minimum ``symbol``; optional
                ``company_name`` and ``ibd_industry_group``.
            price_data: ``{symbol: DataFrame}`` of cached price history (Close
                required, datetime index). Missing/empty frames are skipped.
            target_dates: Trading dates to attribute. Each yields one entry in
                the returned list.

        Each entry shape::

            {
                "date": "YYYY-MM-DD",
                "stocks_up_4pct": int,
                "stocks_down_4pct": int,
                "groups": [
                    {
                        "group": str,
                        "up_count": int,
                        "down_count": int,
                        "net": int,                # up - down
                        "up_stocks":   [{symbol, name, pct_change, close}, ...],
                        "down_stocks": [{symbol, name, pct_change, close}, ...],
                    },
                    ...
                ],
            }
        """
        meta_by_symbol: dict[str, Mapping[str, Any]] = {}
        for entry in symbols_meta:
            if not entry:
                continue
            symbol = entry.get("symbol")
            if not symbol:
                continue
            meta_by_symbol[str(symbol).upper()] = entry

        ordered_dates = sorted({d for d in target_dates if d is not None})
        if not ordered_dates or not meta_by_symbol:
            return []

        date_index = pd.Index(ordered_dates)
        per_date: dict[date, dict[str, dict[str, Any]]] = {d: {} for d in ordered_dates}

        for symbol, meta in meta_by_symbol.items():
            history = price_data.get(symbol)
            if history is None or getattr(history, "empty", True):
                continue
            if "Close" not in history.columns:
                continue

            close_series = pd.Series(
                history["Close"].to_numpy(),
                index=[ts.date() for ts in pd.to_datetime(history.index)],
            )
            close_series = close_series[~close_series.index.duplicated(keep="last")].sort_index()
            if close_series.empty:
                continue

            pct_1d = ((close_series / close_series.shift(1)) - 1.0) * 100.0
            close_aligned = close_series.reindex(date_index)
            pct_aligned = pct_1d.reindex(date_index)

            group = self._resolve_group(meta.get("ibd_industry_group"))
            name = meta.get("company_name") or meta.get("name")

            for idx, d in enumerate(ordered_dates):
                close_val = close_aligned.iloc[idx]
                pct_val = pct_aligned.iloc[idx]
                if pd.isna(close_val) or pd.isna(pct_val):
                    continue
                pct_val = float(pct_val)
                if not math.isfinite(pct_val):
                    continue

                direction = self._classify(pct_val)
                if direction is None:
                    continue

                close_float = float(close_val) if math.isfinite(float(close_val)) else None
                stock_entry = {
                    "symbol": symbol,
                    "name": name,
                    "pct_change": round(pct_val, 2),
                    "close": round(close_float, 2) if close_float is not None else None,
                }

                bucket = per_date[d].setdefault(
                    group,
                    {"up_stocks": [], "down_stocks": []},
                )
                bucket[f"{direction}_stocks"].append(stock_entry)

        results: list[dict[str, Any]] = []
        for d in ordered_dates:
            groups_for_day = per_date[d]
            groups_payload: list[dict[str, Any]] = []
            for group_name, bucket in groups_for_day.items():
                up_stocks = sorted(
                    bucket["up_stocks"],
                    key=lambda row: row["pct_change"],
                    reverse=True,
                )
                down_stocks = sorted(
                    bucket["down_stocks"],
                    key=lambda row: row["pct_change"],
                )
                up_count = len(up_stocks)
                down_count = len(down_stocks)
                if up_count == 0 and down_count == 0:
                    continue
                groups_payload.append(
                    {
                        "group": group_name,
                        "up_count": up_count,
                        "down_count": down_count,
                        "net": up_count - down_count,
                        "up_stocks": up_stocks,
                        "down_stocks": down_stocks,
                    }
                )

            # Sort by total activity descending, then net descending, then name.
            groups_payload.sort(
                key=lambda row: (
                    -(row["up_count"] + row["down_count"]),
                    -row["net"],
                    row["group"],
                )
            )
            total_up = sum(row["up_count"] for row in groups_payload)
            total_down = sum(row["down_count"] for row in groups_payload)
            results.append(
                {
                    "date": d.isoformat(),
                    "stocks_up_4pct": total_up,
                    "stocks_down_4pct": total_down,
                    "groups": groups_payload,
                }
            )

        return results

    def _classify(self, pct_change: float) -> str | None:
        if pct_change >= self._threshold_pct:
            return "up"
        if pct_change <= -self._threshold_pct:
            return "down"
        return None

    @staticmethod
    def _resolve_group(raw_group: Any) -> str:
        if raw_group is None:
            return NO_GROUP_LABEL
        text = str(raw_group).strip()
        if not text:
            return NO_GROUP_LABEL
        return text
