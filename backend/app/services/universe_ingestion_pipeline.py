"""Shared official-source Universe ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol

from sqlalchemy.orm import Session

from ..domain.markets.mic_aliases import mic_alias_registry
from ..domain.universe.ingestion import (
    CanonicalUniverseIngestionResult,
    CanonicalUniverseRow,
    RejectedUniverseRow,
    UniverseIndustryTaxonomy,
    UniverseIngestionContext,
    UniverseIngestionSideEffects,
    UniverseLifecycleMetadata,
    UniverseReconciliationPolicy,
    UniverseSourceProvenance,
    is_fund_industry,
)
from ..models.stock_universe import (
    StockUniverse,
    StockUniverseStatusEvent,
    UNIVERSE_EVENT_LISTING_TIER_CHANGED,
    UNIVERSE_STATUS_ACTIVE,
)
from .universe_classification import prefer_meaningful


class UniverseCanonicalizer(Protocol):
    def canonicalize_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> CanonicalUniverseIngestionResult:
        """Return accepted/rejected canonical rows for one source snapshot."""


@dataclass(frozen=True, slots=True)
class UniverseBeforeReconciliationContext:
    market: str
    source_name: str
    snapshot_id: str
    result: CanonicalUniverseIngestionResult

    @property
    def canonical_rows(self) -> tuple[CanonicalUniverseRow, ...]:
        return self.result.canonical_rows


class UniverseBeforeReconciliationHook(Protocol):
    def __call__(
        self,
        db: Session,
        context: UniverseBeforeReconciliationContext,
        *,
        now: datetime,
    ) -> Mapping[str, Any]:
        """Persist market-specific side effects before reconciliation is recorded."""


class FlatUniverseCanonicalizerAdapter:
    """Adapt legacy flat canonicalizer rows to shared Universe ingestion models."""

    def __init__(
        self,
        canonicalizer: Any,
    ) -> None:
        self._canonicalizer = canonicalizer

    def canonicalize_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> CanonicalUniverseIngestionResult:
        result = self._canonicalizer.canonicalize_rows(
            rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
        )
        if isinstance(result, CanonicalUniverseIngestionResult):
            return result
        flat_canonical_rows = tuple(result.canonical_rows)
        canonical_rows = tuple(self._canonical_row(row) for row in flat_canonical_rows)
        return CanonicalUniverseIngestionResult(
            canonical_rows=canonical_rows,
            rejected_rows=tuple(
                self._rejected_row(row) for row in result.rejected_rows
            ),
            side_effects=self._side_effects(flat_canonical_rows),
        )

    def _canonical_row(self, row: Any) -> CanonicalUniverseRow:
        return self.canonical_row_from_flat(row)

    @staticmethod
    def canonical_row_from_flat(
        row: Any,
    ) -> CanonicalUniverseRow:
        source_metadata = dict(getattr(row, "source_metadata", {}) or {})
        market = row.market
        source_exchange = str(getattr(row, "exchange", "") or "").strip().upper()
        mic = FlatUniverseCanonicalizerAdapter._canonical_mic(market, source_exchange)
        if source_exchange and source_exchange != mic:
            source_metadata.setdefault("source_exchange", source_exchange)

        return CanonicalUniverseRow(
            symbol=row.symbol,
            name=row.name,
            market=market,
            mic=mic,
            currency=row.currency,
            timezone=row.timezone,
            local_code=row.local_code,
            listing_tier=getattr(row, "listing_tier", None),
            sector=row.sector,
            industry=row.industry,
            market_cap=row.market_cap,
            provenance=UniverseSourceProvenance(
                source_name=row.source_name,
                source_symbol=row.source_symbol,
                source_row_number=row.source_row_number,
                snapshot_id=row.snapshot_id,
                snapshot_as_of=row.snapshot_as_of,
                source_metadata=source_metadata,
                lineage_hash=row.lineage_hash,
                row_hash=row.row_hash,
            ),
        )

    @staticmethod
    def _canonical_mic(market: str, exchange: str) -> str:
        resolved = mic_alias_registry.resolve(market, exchange)
        if resolved is not None:
            return resolved.mic
        return str(exchange or "").strip().upper()

    @staticmethod
    def _side_effects(rows: Iterable[Any]) -> UniverseIngestionSideEffects:
        return UniverseIngestionSideEffects(
            industry_taxonomy_rows=tuple(
                FlatUniverseCanonicalizerAdapter._industry_taxonomy(row)
                for row in rows
                if FlatUniverseCanonicalizerAdapter._has_industry_taxonomy(row)
            )
        )

    @staticmethod
    def _has_industry_taxonomy(row: Any) -> bool:
        if str(getattr(row, "market", "") or "").strip().upper() != "CN":
            return False
        return any(
            str(getattr(row, field_name, "") or "").strip()
            for field_name in ("sector", "industry_group", "industry", "sub_industry")
        )

    @staticmethod
    def _industry_taxonomy(row: Any) -> UniverseIndustryTaxonomy:
        return UniverseIndustryTaxonomy(
            symbol=row.symbol,
            sector=getattr(row, "sector", "") or "",
            industry_group=getattr(row, "industry_group", "") or "",
            industry=getattr(row, "industry", "") or "",
            sub_industry=getattr(row, "sub_industry", "") or "",
        )

    @staticmethod
    def _rejected_row(row: Any) -> RejectedUniverseRow:
        return FlatUniverseCanonicalizerAdapter.rejected_row_from_flat(row)

    @staticmethod
    def rejected_row_from_flat(row: Any) -> RejectedUniverseRow:
        return RejectedUniverseRow(
            source_row_number=row.source_row_number,
            source_symbol=row.source_symbol,
            reason=row.reason,
            source_name=getattr(row, "source_name", None),
            snapshot_id=getattr(row, "snapshot_id", None),
            snapshot_as_of=getattr(row, "snapshot_as_of", None),
        )


class StockUniversePersistenceService(Protocol):
    def _apply_status_transition(
        self,
        db: Session,
        record: StockUniverse,
        *,
        new_status: str,
        trigger_source: str,
        reason: str,
        now: datetime | None = None,
        payload: dict[str, Any] | None = None,
        source: str | None = None,
        clear_failures: bool = False,
        seen_in_source: bool = False,
    ) -> bool:
        """Apply lifecycle state and emit status events when needed."""

    def _apply_market_reconciliation_policy(
        self,
        db: Session,
        *,
        market: str,
        snapshot_id: str,
        trigger_source: str,
        reconciliation_policy: UniverseReconciliationPolicy | None = None,
        reconciliation: Mapping[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        """Apply destructive reconciliation safety policy."""

    def _build_metadata_event_record(
        self,
        *,
        symbol: str,
        event_type: str,
        trigger_source: str,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> StockUniverseStatusEvent:
        """Build a metadata audit event."""

    def _build_status_event_record(
        self,
        *,
        symbol: str,
        old_status: str | None,
        new_status: str,
        trigger_source: str,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> StockUniverseStatusEvent:
        """Build a lifecycle audit event."""

    def _bulk_insert_records(self, db: Session, objects: list[Any]) -> None:
        """Insert ORM records in bulk."""

    def _record_market_reconciliation_run(
        self,
        db: Session,
        *,
        market: str,
        source_name: str,
        snapshot_id: str,
        canonical_rows: Iterable[Any],
    ) -> dict[str, Any]:
        """Persist one reconciliation artifact."""


class UniversePersistence:
    """Persist canonical Universe rows and shared reconciliation side effects."""

    def __init__(self, service: StockUniversePersistenceService) -> None:
        self._service = service

    @classmethod
    def for_stock_universe_service(
        cls,
        service: StockUniversePersistenceService,
    ) -> "UniversePersistence":
        return cls(service)

    def persist(
        self,
        db: Session,
        *,
        market: str,
        source_name: str,
        snapshot_id: str,
        result: CanonicalUniverseIngestionResult,
        ingestion_context: UniverseIngestionContext,
        before_reconciliation: UniverseBeforeReconciliationHook | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.utcnow()
        trigger_source = ingestion_context.trigger_source
        row_source = ingestion_context.row_source
        canonical_rows = result.canonical_rows
        canonical_symbols = [row.symbol for row in canonical_rows]
        existing_rows = (
            {
                row.symbol: row
                for row in db.query(StockUniverse)
                .filter(StockUniverse.symbol.in_(canonical_symbols))
                .all()
            }
            if canonical_symbols
            else {}
        )

        added_count = 0
        updated_count = 0
        new_rows: list[StockUniverse] = []
        new_events: list[StockUniverseStatusEvent] = []

        for row in canonical_rows:
            event_payload = self._event_payload(row)
            reason = self._status_reason(row, market=market)
            existing = existing_rows.get(row.symbol)

            if existing is not None:
                self._update_existing_row(
                    db,
                    existing,
                    row=row,
                    now=now,
                    trigger_source=trigger_source,
                    row_source=row_source,
                    reason=reason,
                    event_payload=event_payload,
                    new_events=new_events,
                )
                updated_count += 1
                continue

            new_rows.append(
                self._new_universe_row(
                    row,
                    now=now,
                    source=row_source,
                    reason=reason,
                )
            )
            new_events.append(
                self._service._build_status_event_record(
                    symbol=row.symbol,
                    old_status=None,
                    new_status=row.lifecycle.status,
                    trigger_source=trigger_source,
                    reason=reason,
                    payload=event_payload,
                )
            )
            added_count += 1

        self._service._bulk_insert_records(db, new_rows)
        self._service._bulk_insert_records(db, new_events)

        extra_summary: dict[str, Any] = {}
        if before_reconciliation is not None:
            extra_summary.update(
                dict(
                    before_reconciliation(
                        db,
                        UniverseBeforeReconciliationContext(
                            market=market,
                            source_name=source_name,
                            snapshot_id=snapshot_id,
                            result=result,
                        ),
                        now=now,
                    )
                )
            )

        reconciliation = self._service._record_market_reconciliation_run(
            db,
            market=market,
            source_name=source_name,
            snapshot_id=snapshot_id,
            canonical_rows=canonical_rows,
        )
        reconciliation = self._service._apply_market_reconciliation_policy(
            db,
            market=market,
            snapshot_id=snapshot_id,
            trigger_source=trigger_source,
            reconciliation_policy=ingestion_context.reconciliation_policy,
            reconciliation=reconciliation,
            now=now,
        )
        db.commit()

        return {
            "added": added_count,
            "updated": updated_count,
            "reconciliation": reconciliation,
            "extra_summary": extra_summary,
        }

    def _update_existing_row(
        self,
        db: Session,
        existing: StockUniverse,
        *,
        row: CanonicalUniverseRow,
        now: datetime,
        trigger_source: str,
        row_source: str,
        reason: str,
        event_payload: dict[str, Any],
        new_events: list[StockUniverseStatusEvent],
    ) -> None:
        existing.name = row.name or existing.name
        existing.market = row.market
        existing.exchange = row.mic
        existing.currency = row.currency
        existing.timezone = row.timezone
        existing.local_code = row.local_code or existing.local_code
        existing.sector = prefer_meaningful(row.sector, existing.sector)
        existing.industry = prefer_meaningful(row.industry, existing.industry)
        # A non-fund keeps its stored flag so an operator's classification survives.
        if is_fund_industry(existing.industry):
            existing.is_common_stock = False
        if row.market_cap is not None:
            existing.market_cap = row.market_cap

        self._apply_listing_tier(
            existing,
            row=row,
            trigger_source=trigger_source,
            event_payload=event_payload,
            new_events=new_events,
        )

        self._service._apply_status_transition(
            db,
            existing,
            new_status=row.lifecycle.status,
            trigger_source=trigger_source,
            reason=reason,
            now=now,
            payload=event_payload,
            source=row_source,
            clear_failures=row.lifecycle.is_active,
            seen_in_source=row.lifecycle.is_active,
        )
        self._apply_lifecycle_metadata(existing, row.lifecycle, now=now)

    def _apply_listing_tier(
        self,
        existing: StockUniverse,
        *,
        row: CanonicalUniverseRow,
        trigger_source: str,
        event_payload: dict[str, Any],
        new_events: list[StockUniverseStatusEvent],
    ) -> None:
        if row.listing_tier is None:
            return

        previous_listing_tier = existing.listing_tier
        existing.listing_tier = row.listing_tier
        if previous_listing_tier != row.listing_tier:
            new_events.append(
                self._service._build_metadata_event_record(
                    symbol=row.symbol,
                    event_type=UNIVERSE_EVENT_LISTING_TIER_CHANGED,
                    trigger_source=trigger_source,
                    reason="listing tier changed",
                    payload={
                        **event_payload,
                        "previous": previous_listing_tier,
                        "current": row.listing_tier,
                    },
                )
            )

    @staticmethod
    def _apply_lifecycle_metadata(
        existing: StockUniverse,
        lifecycle: UniverseLifecycleMetadata,
        *,
        now: datetime,
    ) -> None:
        if existing.first_seen_at is None:
            existing.first_seen_at = lifecycle.first_seen_at or now
        if lifecycle.last_seen_in_source_at is not None:
            existing.last_seen_in_source_at = lifecycle.last_seen_in_source_at
        if not lifecycle.is_active and lifecycle.deactivated_at is not None:
            existing.deactivated_at = lifecycle.deactivated_at
        existing.consecutive_fetch_failures = lifecycle.consecutive_fetch_failures

    @staticmethod
    def _new_universe_row(
        row: CanonicalUniverseRow,
        *,
        now: datetime,
        source: str,
        reason: str,
    ) -> StockUniverse:
        lifecycle = row.lifecycle
        first_seen_at = lifecycle.first_seen_at or now
        last_seen_in_source_at = (
            lifecycle.last_seen_in_source_at
            if lifecycle.last_seen_in_source_at is not None
            else now if lifecycle.is_active else None
        )
        deactivated_at = (
            lifecycle.deactivated_at
            if lifecycle.deactivated_at is not None
            else None if lifecycle.is_active else now
        )
        return StockUniverse(
            symbol=row.symbol,
            name=row.name,
            market=row.market,
            exchange=row.mic,
            listing_tier=row.listing_tier,
            currency=row.currency,
            timezone=row.timezone,
            local_code=row.local_code,
            sector=row.sector,
            industry=row.industry,
            market_cap=row.market_cap,
            is_active=lifecycle.is_active,
            status=lifecycle.status,
            status_reason=reason,
            is_common_stock=not is_fund_industry(row.industry),
            source=source,
            consecutive_fetch_failures=lifecycle.consecutive_fetch_failures,
            added_at=first_seen_at,
            first_seen_at=first_seen_at,
            last_seen_in_source_at=last_seen_in_source_at,
            deactivated_at=deactivated_at,
            updated_at=now,
        )

    @staticmethod
    def _event_payload(row: CanonicalUniverseRow) -> dict[str, Any]:
        provenance = row.provenance
        return {
            "source_name": provenance.source_name,
            "source_symbol": provenance.source_symbol,
            "source_row_number": provenance.source_row_number,
            "snapshot_id": provenance.snapshot_id,
            "snapshot_as_of": provenance.snapshot_as_of,
            "source_metadata": provenance.source_metadata,
            "lineage_hash": provenance.lineage_hash,
            "row_hash": provenance.row_hash,
            "listing_tier": row.listing_tier,
        }

    @staticmethod
    def _status_reason(row: CanonicalUniverseRow, *, market: str) -> str:
        if row.lifecycle.status_reason:
            return row.lifecycle.status_reason
        if row.lifecycle.status == UNIVERSE_STATUS_ACTIVE:
            return f"Present in {market} source snapshot {row.provenance.snapshot_id}"
        return (
            f"{market} source snapshot {row.provenance.snapshot_id} marks "
            f"row {row.lifecycle.status}"
        )


class UniverseIngestionPipeline:
    """Canonicalize, persist, reconcile, and summarize one market snapshot."""

    def __init__(
        self,
        *,
        canonicalizers: Mapping[str, UniverseCanonicalizer],
        persistence: UniversePersistence,
        before_reconciliation_hooks: (
            Mapping[str, UniverseBeforeReconciliationHook] | None
        ) = None,
    ) -> None:
        self._canonicalizers = {
            str(market).strip().upper(): canonicalizer
            for market, canonicalizer in canonicalizers.items()
        }
        self._persistence = persistence
        self._before_reconciliation_hooks = {
            str(market).strip().upper(): hook
            for market, hook in (before_reconciliation_hooks or {}).items()
        }

    def ingest_snapshot_rows(
        self,
        db: Session,
        *,
        market: str,
        rows: Iterable[Mapping[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        strict: bool = True,
        ingestion_context: UniverseIngestionContext | None = None,
    ) -> dict[str, Any]:
        market_code = str(market or "").strip().upper()
        canonicalizer = self._canonicalizers.get(market_code)
        if canonicalizer is None:
            raise ValueError(f"Universe ingestion is unsupported for market {market!r}")

        result = canonicalizer.canonicalize_rows(
            rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
        )
        return self.ingest_canonicalized_result(
            db,
            market=market_code,
            source_name=source_name,
            snapshot_id=snapshot_id,
            result=result,
            strict=strict,
            ingestion_context=ingestion_context,
        )

    def ingest_canonicalized_result(
        self,
        db: Session,
        *,
        market: str,
        source_name: str,
        snapshot_id: str,
        result: CanonicalUniverseIngestionResult,
        strict: bool = True,
        ingestion_context: UniverseIngestionContext | None = None,
    ) -> dict[str, Any]:
        market_code = str(market or "").strip().upper()
        blocking_rejections = tuple(row for row in result.rejected_rows if row.strict)
        if strict and blocking_rejections:
            sample = self._rejected_sample(blocking_rejections)
            raise ValueError(
                f"{market_code} ingestion rejected {len(blocking_rejections)} "
                f"row(s). {sample}"
            )

        context = ingestion_context or UniverseIngestionContext.default_for_market(market_code)
        persisted = self._persistence.persist(
            db,
            market=market_code,
            source_name=source_name,
            snapshot_id=snapshot_id,
            result=result,
            ingestion_context=context,
            before_reconciliation=self._before_reconciliation_hooks.get(market_code),
        )
        return self._summary(
            market=market_code,
            source_name=source_name,
            snapshot_id=snapshot_id,
            result=result,
            persisted=persisted,
        )

    @staticmethod
    def _rejected_sample(rejected_rows: tuple[RejectedUniverseRow, ...]) -> str:
        return "; ".join(
            f"row {row.source_row_number}: {row.reason}" for row in rejected_rows[:3]
        )

    @staticmethod
    def _summary(
        *,
        market: str,
        source_name: str,
        snapshot_id: str,
        result: CanonicalUniverseIngestionResult,
        persisted: Mapping[str, Any],
    ) -> dict[str, Any]:
        details_limit = 25
        canonical_preview = result.canonical_rows[:details_limit]
        rejected_preview = result.rejected_rows[:details_limit]
        summary = {
            "added": persisted["added"],
            "updated": persisted["updated"],
            "total": len(result.canonical_rows),
            "rejected": len(result.rejected_rows),
            "source_name": source_name,
            "snapshot_id": snapshot_id,
            "canonical_rows": [
                {
                    "symbol": row.symbol,
                    "local_code": row.local_code,
                    "exchange": row.mic,
                    "source_symbol": row.provenance.source_symbol,
                    "lineage_hash": row.provenance.lineage_hash,
                    "row_hash": row.provenance.row_hash,
                }
                for row in canonical_preview
            ],
            "rejected_rows": [
                {
                    "source_row_number": row.source_row_number,
                    "source_symbol": row.source_symbol,
                    "reason": row.reason,
                }
                for row in rejected_preview
            ],
            "canonical_rows_truncated": len(result.canonical_rows) > details_limit,
            "rejected_rows_truncated": len(result.rejected_rows) > details_limit,
            "reconciliation": persisted["reconciliation"],
            "pipeline": {
                "market": market,
                "version": "universe-ingestion-pipeline-v1",
            },
        }
        summary.update(dict(persisted.get("extra_summary") or {}))
        return summary
