"""Export the chunked static scan bundle and its manifest."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.scanning.default_filters import resolve_default_scan_filters
from app.domain.scanning.materialization import (
    config_has_opportunity_state_materialization,
)
from app.infra.serialization import json_safe
from app.schemas.scanning import FilterOptionsResponse, ScanResultItem
from app.services.feature_run_rs_identity import resolve_feature_run_rs_identity
from app.services.preset_screens import (
    PRESET_SCREENS,
    resolve_preset_screens_for_defaults,
)

SCAN_BUNDLE_SCHEMA_VERSION = "static-scan-v2"
SCAN_CHUNK_SIZE = 1000

RowSerializer = Callable[[Any], dict[str, Any]]
JsonWriter = Callable[[Path, dict[str, Any]], None]
RsIdentityResolver = Callable[..., Any]


@dataclass(frozen=True)
class StaticScanBundleRequest:
    output_dir: Path
    generated_at: str
    run: Any
    rows: Sequence[Any]
    filter_options: Any
    path_prefix: Path | None = None
    market: str | None = None
    row_serializer: RowSerializer | None = None
    chunk_size: int = SCAN_CHUNK_SIZE


@dataclass(frozen=True)
class StaticScanBundleResult:
    manifest: dict[str, Any]
    serialized_rows: tuple[dict[str, Any], ...]


class StaticScanBundleExporter:
    """Own scan serialization, ranking, chunking, and manifest construction."""

    def __init__(
        self,
        *,
        json_writer: JsonWriter | None = None,
        rs_identity_resolver: RsIdentityResolver = resolve_feature_run_rs_identity,
    ) -> None:
        self._json_writer = json_writer or self.write_json
        self._rs_identity_resolver = rs_identity_resolver

    def export_scan_bundle(
        self, request: StaticScanBundleRequest
    ) -> StaticScanBundleResult:
        output_dir = Path(request.output_dir)
        prefix = Path() if request.path_prefix is None else Path(request.path_prefix)
        scan_dir = output_dir / prefix / "scan"
        chunk_dir = scan_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        serializer = request.row_serializer or self.serialize_scan_row
        rows = [serializer(row) for row in request.rows]
        rs_metadata = self._rs_metadata(request.run)
        for row in rows:
            row.update(rs_metadata)
        self.annotate_percentile_ranks(rows)
        rows = self.sort_static_scan_rows(rows)

        capable = config_has_opportunity_state_materialization(request.run.config_json)
        defaults = self.resolve_static_default_filters(request.market)
        presets = resolve_preset_screens_for_defaults(
            [
                preset
                for preset in PRESET_SCREENS
                if capable or preset.get("id") != "correction_survivors"
            ],
            defaults,
        )
        default_rows = self.apply_static_default_filters(
            rows,
            default_filters=defaults,
        )
        chunks = self._write_chunks(
            request=request,
            prefix=prefix,
            rows=rows,
            rs_metadata=rs_metadata,
        )
        manifest = {
            "schema_version": SCAN_BUNDLE_SCHEMA_VERSION,
            "features": {"opportunity_state": capable},
            "generated_at": request.generated_at,
            "as_of_date": request.run.as_of_date.isoformat(),
            "run_id": request.run.id,
            **rs_metadata,
            "sort": {"field": "composite_score", "order": "desc"},
            "default_page_size": 50,
            "chunk_size": request.chunk_size,
            "rows_total": len(rows),
            "default_filters": dict(defaults),
            "default_filtered_rows_total": len(default_rows),
            "filter_options": FilterOptionsResponse(
                ibd_industries=list(request.filter_options.ibd_industries),
                gics_sectors=list(request.filter_options.gics_sectors),
                ratings=list(request.filter_options.ratings),
            ).model_dump(mode="json"),
            "preset_screens": presets,
            "chunks": chunks,
            "initial_rows": default_rows[:50],
            "preview_rows": default_rows[:10],
        }
        self._json_writer(scan_dir / "manifest.json", manifest)
        return StaticScanBundleResult(
            manifest=manifest,
            serialized_rows=tuple(rows),
        )

    def _rs_metadata(self, run: Any) -> dict[str, Any]:
        publication = self._rs_identity_resolver(
            run,
            ranking_date=run.as_of_date,
        ).publication
        return {
            "rs_formula_version": publication.snapshot.formula_version,
            "market_rs_run_id": publication.market_rs_run_id,
            "rs_as_of_date": publication.snapshot.as_of_date.isoformat(),
            "rs_universe_size": publication.universe_size,
        }

    def _write_chunks(
        self,
        *,
        request: StaticScanBundleRequest,
        prefix: Path,
        rows: list[dict[str, Any]],
        rs_metadata: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        refs = []
        for index in range(0, len(rows), request.chunk_size):
            chunk_rows = rows[index : index + request.chunk_size]
            chunk_num = (index // request.chunk_size) + 1
            rel_path = prefix / "scan" / "chunks" / f"chunk-{chunk_num:04d}.json"
            self._json_writer(
                Path(request.output_dir) / rel_path,
                {
                    "schema_version": SCAN_BUNDLE_SCHEMA_VERSION,
                    "generated_at": request.generated_at,
                    "as_of_date": request.run.as_of_date.isoformat(),
                    "run_id": request.run.id,
                    "chunk_index": chunk_num,
                    **rs_metadata,
                    "rows": chunk_rows,
                },
            )
            refs.append({"path": rel_path.as_posix(), "count": len(chunk_rows)})
        return refs

    @staticmethod
    def serialize_scan_row(row: Any) -> dict[str, Any]:
        item = ScanResultItem.from_domain(
            row,
            include_setup_payload=False,
        ).model_dump(
            mode="json",
            exclude={"se_explain", "se_candidates"},
        )
        extended = row.extended_fields or {}
        item.update(
            {
                field: extended.get(field)
                for field in (
                    "perf_week",
                    "perf_month",
                    "perf_3m",
                    "perf_6m",
                    "gap_percent",
                    "ema_10_distance",
                    "ema_20_distance",
                    "ema_50_distance",
                    "week_52_high_distance",
                    "week_52_low_distance",
                )
            }
        )
        return item

    @staticmethod
    def annotate_percentile_ranks(rows: list[dict[str, Any]]) -> None:
        for source, destination in (
            ("price_change_1d", "pct_day"),
            ("perf_week", "pct_week"),
            ("perf_month", "pct_month"),
        ):
            StaticScanBundleExporter._annotate_percentile_rank(
                rows,
                source=source,
                destination=destination,
            )

    @staticmethod
    def _annotate_percentile_rank(
        rows: list[dict[str, Any]],
        *,
        source: str,
        destination: str,
    ) -> None:
        ranked = sorted(
            (
                (index, row[source])
                for index, row in enumerate(rows)
                if row.get(source) is not None
            ),
            key=lambda pair: pair[1],
        )
        for row in rows:
            row[destination] = None
        position = 0
        while position < len(ranked):
            end = position
            value = ranked[position][1]
            while end + 1 < len(ranked) and ranked[end + 1][1] == value:
                end += 1
            percentile = round(((end + 1) / len(ranked)) * 100, 2)
            for rank_index in range(position, end + 1):
                row_index, _ = ranked[rank_index]
                rows[row_index][destination] = percentile
            position = end + 1

    @staticmethod
    def resolve_static_default_filters(
        market: str | None,
    ) -> dict[str, int | None]:
        return resolve_default_scan_filters(market)

    @staticmethod
    def apply_static_default_filters(
        rows: list[dict[str, Any]],
        *,
        default_filters: dict[str, int | None] | None = None,
    ) -> list[dict[str, Any]]:
        filters = default_filters or {"minVolume": None}
        min_volume = filters.get("minVolume")
        if min_volume is None:
            return list(rows)
        return [
            row
            for row in rows
            if row.get("volume") is not None and row["volume"] >= min_volume
        ]

    @staticmethod
    def sort_static_scan_rows(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        def score(row: dict[str, Any]) -> float:
            value = row.get("composite_score")
            return float(value) if value is not None else float("-inf")

        return sorted(
            rows,
            key=lambda row: (
                row.get("minervini_score") is None and row.get("canslim_score") is None,
                -score(row),
                row.get("symbol") or "",
            ),
        )

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                json_safe(payload),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
