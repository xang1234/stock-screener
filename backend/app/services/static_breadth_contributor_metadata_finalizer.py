"""Finalize frozen display metadata for static breadth contributor snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session, joinedload

from app.models.breadth_contributor import MarketBreadthContributorSnapshot
from app.services.breadth.contributor_metadata import BreadthContributorMetadataLoader
from app.services.breadth.contributors import NO_GROUP_LABEL
from app.services.breadth.types import BreadthContributorMetadata
from app.services.static_breadth_contributor_metadata_contract import (
    STATIC_BREADTH_CONTRIBUTOR_METADATA_RETENTION_DATES,
    STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION,
    FrozenBreadthContributorMetadata,
    FrozenBreadthContributorSession,
    StaticBreadthContributorMetadataState,
    normalize_static_breadth_contributor_metadata_market,
    read_static_breadth_contributor_metadata,
    write_static_breadth_contributor_metadata,
)


class StaticBreadthContributorMetadataCoverageError(RuntimeError):
    """Raised when a non-empty static contributor bundle lacks useful metadata."""


@dataclass(frozen=True)
class StaticBreadthContributorMetadataFinalizationReport:
    market: str
    retained_dates: int
    contributors: int
    restored_contributors: int
    bootstrapped_contributors: int
    named_contributors: int
    classified_contributors: int
    source_status: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "market": self.market,
            "retained_dates": self.retained_dates,
            "contributors": self.contributors,
            "restored_contributors": self.restored_contributors,
            "bootstrapped_contributors": self.bootstrapped_contributors,
            "named_contributors": self.named_contributors,
            "classified_contributors": self.classified_contributors,
            "source_status": self.source_status,
        }


class StaticBreadthContributorMetadataFinalizer:
    """Freeze restored metadata and bootstrap metadata for newly seen contributors."""

    def __init__(
        self,
        db: Session,
        *,
        metadata_loader: type[BreadthContributorMetadataLoader] = (
            BreadthContributorMetadataLoader
        ),
    ) -> None:
        self.db = db
        self.metadata_loader = metadata_loader

    def finalize(
        self,
        *,
        market: str,
        source_path: Path,
        output_path: Path,
        source_status: str,
        limit: int = STATIC_BREADTH_CONTRIBUTOR_METADATA_RETENTION_DATES,
    ) -> StaticBreadthContributorMetadataFinalizationReport:
        normalized_market = normalize_static_breadth_contributor_metadata_market(market)
        normalized_status = str(source_status).strip().lower()
        retention_limit = min(
            max(int(limit), 0),
            STATIC_BREADTH_CONTRIBUTOR_METADATA_RETENTION_DATES,
        )
        restored_by_date_symbol: dict[tuple[object, str], BreadthContributorMetadata] = {}
        if normalized_status == "restored":
            restored = read_static_breadth_contributor_metadata(
                source_path,
                expected_market=normalized_market,
            )
            restored_by_date_symbol = {
                (session.date, item.symbol): BreadthContributorMetadata(
                    company_name=item.company_name,
                    ibd_industry_group=item.ibd_industry_group,
                )
                for session in restored.sessions
                for item in session.contributors
            }
        elif normalized_status != "missing":
            raise ValueError(
                "Breadth contributor metadata finalization requires a restored or "
                "missing source status."
            )

        snapshots = (
            self.db.query(MarketBreadthContributorSnapshot)
            .options(joinedload(MarketBreadthContributorSnapshot.contributors))
            .filter(MarketBreadthContributorSnapshot.market == normalized_market)
            .order_by(MarketBreadthContributorSnapshot.date.desc())
            .limit(retention_limit)
            .all()
        )
        symbols = tuple(
            sorted(
                {
                    str(item.symbol).strip().upper()
                    for snapshot in snapshots
                    for item in snapshot.contributors
                }
            )
        )
        current = self.metadata_loader.current(
            self.db,
            normalized_market,
            symbols,
        )

        total = 0
        restored_count = 0
        bootstrapped_count = 0
        named_count = 0
        classified_count = 0
        sessions: list[FrozenBreadthContributorSession] = []
        try:
            for snapshot in snapshots:
                frozen_items: list[FrozenBreadthContributorMetadata] = []
                for contributor in snapshot.contributors:
                    symbol = str(contributor.symbol).strip().upper()
                    frozen = restored_by_date_symbol.get((snapshot.date, symbol))
                    if frozen is not None:
                        restored_count += 1
                    else:
                        frozen = current.get(
                            symbol,
                            BreadthContributorMetadata(
                                company_name=None,
                                ibd_industry_group=NO_GROUP_LABEL,
                            ),
                        )
                        bootstrapped_count += 1

                    company_name = (
                        str(frozen.company_name).strip()
                        if frozen.company_name is not None
                        and str(frozen.company_name).strip()
                        else None
                    )
                    group = (
                        str(frozen.ibd_industry_group or "").strip()
                        or NO_GROUP_LABEL
                    )
                    contributor.company_name = company_name
                    contributor.ibd_industry_group = group
                    total += 1
                    named_count += int(company_name is not None)
                    classified_count += int(group != NO_GROUP_LABEL)
                    frozen_items.append(
                        FrozenBreadthContributorMetadata(
                            symbol=symbol,
                            company_name=company_name,
                            ibd_industry_group=group,
                        )
                    )
                sessions.append(
                    FrozenBreadthContributorSession(
                        date=snapshot.date,
                        contributors=tuple(sorted(frozen_items, key=lambda item: item.symbol)),
                    )
                )

            if total and not named_count:
                raise StaticBreadthContributorMetadataCoverageError(
                    f"Static breadth contributors for {normalized_market} have no "
                    "company name metadata."
                )
            if total and not classified_count:
                raise StaticBreadthContributorMetadataCoverageError(
                    f"Static breadth contributors for {normalized_market} have no "
                    "industry group metadata."
                )

            state = StaticBreadthContributorMetadataState(
                schema_version=(
                    STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION
                ),
                market=normalized_market,
                generated_at=datetime.now(timezone.utc),
                sessions=tuple(sessions),
            )
            self.db.flush()
            write_static_breadth_contributor_metadata(output_path, state)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        return StaticBreadthContributorMetadataFinalizationReport(
            market=normalized_market,
            retained_dates=len(sessions),
            contributors=total,
            restored_contributors=restored_count,
            bootstrapped_contributors=bootstrapped_count,
            named_contributors=named_count,
            classified_contributors=classified_count,
            source_status=normalized_status,
        )


__all__ = [
    "StaticBreadthContributorMetadataCoverageError",
    "StaticBreadthContributorMetadataFinalizationReport",
    "StaticBreadthContributorMetadataFinalizer",
]
