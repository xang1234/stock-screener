"""Export scan results to a downloadable file (CSV, etc.).

Pure use case: takes domain objects in, returns bytes + metadata out.
No HTTP, no ORM — just business logic for formatting scan results.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Any

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.filter_spec import FilterSpec, SortSpec
from app.domain.scanning.models import ExportFormat, ScanResultItemDomain


# ---------------------------------------------------------------------------
# Query / Result DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportScanResultsQuery:
    scan_id: str
    filters: FilterSpec = field(default_factory=FilterSpec)
    sort: SortSpec = field(default_factory=SortSpec)
    export_format: ExportFormat = ExportFormat.CSV
    passes_only: bool = False


@dataclass(frozen=True)
class ExportScanResultsResult:
    content: bytes
    filename: str
    media_type: str


# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

# Each entry is (header_name, extractor_fn) — ordered for the output CSV.
_CSV_COLUMNS: list[tuple[str, Any]] = [
    ("Symbol", lambda item: item.symbol),
    ("Company Name", lambda item: item.extended_fields.get("company_name")),
    ("Composite Score", lambda item: item.composite_score),
    ("Rating", lambda item: item.rating),
    ("Current Price", lambda item: item.current_price),
    # Screener scores
    ("Minervini Score", lambda item: item.extended_fields.get("minervini_score")),
    ("CANSLIM Score", lambda item: item.extended_fields.get("canslim_score")),
    ("IPO Score", lambda item: item.extended_fields.get("ipo_score")),
    ("Custom Score", lambda item: item.extended_fields.get("custom_score")),
    ("Volume Breakthrough Score", lambda item: item.extended_fields.get("volume_breakthrough_score")),
    # RS ratings
    ("RS Rating", lambda item: item.extended_fields.get("rs_rating")),
    ("RS 1M", lambda item: item.extended_fields.get("rs_rating_1m")),
    ("RS 3M", lambda item: item.extended_fields.get("rs_rating_3m")),
    ("RS 12M", lambda item: item.extended_fields.get("rs_rating_12m")),
    # Stage / Template
    ("Stage", lambda item: item.extended_fields.get("stage")),
    ("Stage Name", lambda item: item.extended_fields.get("stage_name")),
    ("Passes Template", lambda item: item.extended_fields.get("passes_template")),
    # VCP
    ("VCP Detected", lambda item: item.extended_fields.get("vcp_detected")),
    ("VCP Score", lambda item: item.extended_fields.get("vcp_score")),
    ("VCP Pivot", lambda item: item.extended_fields.get("vcp_pivot")),
    ("VCP Ready", lambda item: item.extended_fields.get("vcp_ready_for_breakout")),
    ("VCP Contraction", lambda item: item.extended_fields.get("vcp_contraction_ratio")),
    ("VCP ATR Score", lambda item: item.extended_fields.get("vcp_atr_score")),
    # Technicals
    ("MA Alignment", lambda item: item.extended_fields.get("ma_alignment")),
    ("ADR %", lambda item: item.extended_fields.get("adr_percent")),
    # Growth
    ("EPS Growth Q/Q", lambda item: item.extended_fields.get("eps_growth_qq")),
    ("Sales Growth Q/Q", lambda item: item.extended_fields.get("sales_growth_qq")),
    ("EPS Growth Y/Y", lambda item: item.extended_fields.get("eps_growth_yy")),
    ("Sales Growth Y/Y", lambda item: item.extended_fields.get("sales_growth_yy")),
    # Valuation
    ("PEG Ratio", lambda item: item.extended_fields.get("peg_ratio")),
    ("EPS Rating", lambda item: item.extended_fields.get("eps_rating")),
    # Industry
    ("IBD Industry Group", lambda item: item.extended_fields.get("ibd_industry_group")),
    ("IBD Group Rank", lambda item: item.extended_fields.get("ibd_group_rank")),
    ("GICS Sector", lambda item: item.extended_fields.get("gics_sector")),
    ("GICS Industry", lambda item: item.extended_fields.get("gics_industry")),
    # Performance
    ("Price Change 1D", lambda item: item.extended_fields.get("price_change_1d")),
    ("Perf Week", lambda item: item.extended_fields.get("perf_week")),
    ("Perf Month", lambda item: item.extended_fields.get("perf_month")),
    ("Perf 3M", lambda item: item.extended_fields.get("perf_3m")),
    ("Perf 6M", lambda item: item.extended_fields.get("perf_6m")),
    # Momentum
    ("RS Trend", lambda item: item.extended_fields.get("rs_trend")),
    ("Price Trend", lambda item: item.extended_fields.get("price_trend")),
    # EMA distances
    ("EMA 10 Distance", lambda item: item.extended_fields.get("ema_10_distance")),
    ("EMA 20 Distance", lambda item: item.extended_fields.get("ema_20_distance")),
    ("EMA 50 Distance", lambda item: item.extended_fields.get("ema_50_distance")),
    # 52-week
    ("52W High Distance", lambda item: item.extended_fields.get("week_52_high_distance")),
    ("52W Low Distance", lambda item: item.extended_fields.get("week_52_low_distance")),
    # Volume
    ("Volume", lambda item: item.extended_fields.get("volume")),
    ("Market Cap", lambda item: item.extended_fields.get("market_cap")),
    ("Gap %", lambda item: item.extended_fields.get("gap_percent")),
    ("Volume Surge", lambda item: item.extended_fields.get("volume_surge")),
    # Beta
    ("Beta", lambda item: item.extended_fields.get("beta")),
    ("Beta-Adj RS", lambda item: item.extended_fields.get("beta_adj_rs")),
    ("Beta-Adj RS 1M", lambda item: item.extended_fields.get("beta_adj_rs_1m")),
    ("Beta-Adj RS 3M", lambda item: item.extended_fields.get("beta_adj_rs_3m")),
    ("Beta-Adj RS 12M", lambda item: item.extended_fields.get("beta_adj_rs_12m")),
    # Other
    ("IPO Date", lambda item: item.extended_fields.get("ipo_date")),
    ("Screeners Run", lambda item: item.screeners_run),
]


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Format a single cell value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return str(round(value, 2))
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


# ---------------------------------------------------------------------------
# CSV formatter
# ---------------------------------------------------------------------------


def _format_csv(items: tuple[ScanResultItemDomain, ...]) -> bytes:
    """Build CSV bytes from domain items, with UTF-8 BOM for Excel."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header row
    writer.writerow([col_name for col_name, _ in _CSV_COLUMNS])

    # Data rows
    for item in items:
        row = [_format_value(extractor(item)) for _, extractor in _CSV_COLUMNS]
        writer.writerow(row)

    return b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Use case
# ---------------------------------------------------------------------------


class ExportScanResultsUseCase:
    """Export scan results to a downloadable file."""

    def execute(
        self, uow: UnitOfWork, query: ExportScanResultsQuery
    ) -> ExportScanResultsResult:
        with uow:
            scan = uow.scans.get_by_scan_id(query.scan_id)
            if scan is None:
                raise EntityNotFoundError("Scan", query.scan_id)

            # Build effective filters
            filters = query.filters
            if query.passes_only:
                filters.add_categorical(
                    "rating", ("Strong Buy", "Buy")
                )

            items = uow.scan_results.query_all(
                scan_id=query.scan_id,
                filters=filters,
                sort=query.sort,
            )

        # Format output
        if query.export_format == ExportFormat.CSV:
            content = _format_csv(items)
            media_type = "text/csv"
            ext = "csv"
        else:
            raise ValueError(f"Unsupported export format: {query.export_format}")

        # Build filename from scan metadata
        label = getattr(scan, "scan_id", query.scan_id)
        # Truncate long scan IDs for filename readability
        short_label = label[:12] if len(label) > 12 else label

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{short_label}_{timestamp}.{ext}"

        return ExportScanResultsResult(
            content=content,
            filename=filename,
            media_type=media_type,
        )
