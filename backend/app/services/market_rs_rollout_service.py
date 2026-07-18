"""Resumable backfill, validation, and atomic activation for balanced Market RS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.domain.feature_store.models import RunStatus
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.infra.db.repositories.feature_run_repo import SqlFeatureRunRepository
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.industry import IBDGroupRank
from app.models.stock import StockPrice
from app.services.benchmark_registry_service import benchmark_registry
from app.services.canonical_group_ranking_service import CanonicalGroupRankingService
from app.services.ibd_industry_service import IBDIndustryService
from app.services.market_calendar_service import MarketCalendarService
from app.services.market_rs_inputs import MarketRsInputLoader, MarketRsInputUnavailable
from app.services.market_rs_snapshot_service import MarketRsSnapshotService
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGDatabasePayloadSource,
    StaticGroupsRRGUnavailableError,
)
from app.services.static_rrg_history_contract import (
    STATIC_RRG_HISTORY_SCHEMA_VERSION,
)
from app.services.static_site_export_service import STATIC_SITE_SCHEMA_VERSION
from app.tasks.market_queues import normalize_market


@dataclass(frozen=True)
class BackfillDateResult:
    as_of_date: date
    status: str
    market_rs_run_id: int | None
    group_market_rs_run_id: int | None
    eligible_symbol_count: int
    group_row_count: int
    reason_code: str | None = None
    diagnostics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat()
        return payload


@dataclass(frozen=True)
class BackfillReport:
    market: str
    formula_version: str
    requested_start_date: date | None
    through_date: date
    first_valid_date: date | None
    candidate_count: int
    completed_count: int
    failed_count: int
    latest_run_id: int | None
    group_row_count: int
    results: tuple[BackfillDateResult, ...]
    validation_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.failed_count == 0 and not self.validation_errors

    @property
    def failed_dates(self) -> tuple[date, ...]:
        return tuple(item.as_of_date for item in self.results if item.status == "failed")

    def to_dict(self) -> dict[str, object]:
        return {
            "market": self.market,
            "formula_version": self.formula_version,
            "requested_start_date": (
                self.requested_start_date.isoformat()
                if self.requested_start_date is not None
                else None
            ),
            "through_date": self.through_date.isoformat(),
            "first_valid_date": (
                self.first_valid_date.isoformat()
                if self.first_valid_date is not None
                else None
            ),
            "candidate_count": self.candidate_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "latest_run_id": self.latest_run_id,
            "group_row_count": self.group_row_count,
            "failed_dates": [value.isoformat() for value in self.failed_dates],
            "validation_errors": list(self.validation_errors),
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(frozen=True)
class ActivationValidationReport:
    market: str
    formula_version: str
    through_date: date
    first_valid_date: date | None
    candidate_count: int
    latest_market_rs_run_id: int | None
    latest_universe_hash: str | None
    feature_run_id: int | None
    feature_universe_hash: str | None
    static_manifest_sha256: str | None
    errors: tuple[str, ...]
    rrg_status: str | None = None
    rrg_history_schema_version: str = STATIC_RRG_HISTORY_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["through_date"] = self.through_date.isoformat()
        payload["first_valid_date"] = (
            self.first_valid_date.isoformat()
            if self.first_valid_date is not None
            else None
        )
        payload["ok"] = self.ok
        payload["errors"] = list(self.errors)
        return payload


class MarketRsActivationRejected(RuntimeError):
    def __init__(self, errors: tuple[str, ...] | list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors) or "Market RS activation rejected")


FeatureRunRepositoryFactory = Callable[[Session], SqlFeatureRunRepository]


class MarketRsRolloutService:
    """Coordinates shadow history without changing active pointers until validated."""

    def __init__(
        self,
        *,
        calendar_service: MarketCalendarService,
        input_loader: MarketRsInputLoader,
        market_rs_snapshot_service: MarketRsSnapshotService,
        market_rs_repository: MarketRsRunRepository,
        canonical_group_service: CanonicalGroupRankingService,
        feature_run_repository_factory: FeatureRunRepositoryFactory | None = None,
    ) -> None:
        self.calendar_service = calendar_service
        self.input_loader = input_loader
        self.market_rs_snapshot_service = market_rs_snapshot_service
        self.market_rs_repository = market_rs_repository
        self.canonical_group_service = canonical_group_service
        self._feature_run_repository_factory = (
            feature_run_repository_factory or SqlFeatureRunRepository
        )

    @staticmethod
    def _normalize_market(market: str) -> str:
        normalized = normalize_market(market)
        if normalized == "SHARED":
            raise ValueError("Market RS rollout requires an explicit market")
        return normalized

    @staticmethod
    def _reason_code(exc: Exception, *, stage: str) -> str:
        if isinstance(exc, MarketRsInputUnavailable):
            return exc.reason_code
        name = type(exc).__name__
        snake = "".join(
            ("_" + char.lower()) if char.isupper() else char
            for char in name
        ).lstrip("_")
        return f"{stage}_{snake}" if snake else f"{stage}_failed"

    def _earliest_available_price_date(self, db: Session, market: str) -> date | None:
        candidates = benchmark_registry.get_candidate_symbols(market)
        return (
            db.query(func.min(StockPrice.date))
            .filter(StockPrice.symbol.in_(tuple(candidates)))
            .scalar()
        )

    def earliest_backfillable_date(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
    ) -> date | None:
        normalized = self._normalize_market(market)
        available_start = self._earliest_available_price_date(db, normalized)
        if available_start is None or available_start > through_date:
            return None
        sessions = self.calendar_service.trading_days(
            normalized,
            available_start,
            through_date,
        )
        for session_date in sessions:
            try:
                inputs = self.input_loader.load(
                    db,
                    market=normalized,
                    as_of_date=session_date,
                )
            except Exception:
                continue
            if len(inputs.excess_returns_by_symbol) >= 2:
                return session_date
        return None

    def candidate_dates(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        first_valid_date: date | None = None,
    ) -> tuple[date, ...]:
        normalized = self._normalize_market(market)
        boundary = first_valid_date or self.earliest_backfillable_date(
            db,
            market=normalized,
            through_date=through_date,
        )
        if boundary is None:
            return ()
        return tuple(
            session_date
            for session_date in self.calendar_service.trading_days(
                normalized,
                boundary,
                through_date,
            )
            if boundary <= session_date <= through_date
        )

    def backfill(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        start_date: date | None = None,
    ) -> BackfillReport:
        normalized = self._normalize_market(market)
        first_valid = self.earliest_backfillable_date(
            db,
            market=normalized,
            through_date=through_date,
        )
        if first_valid is None:
            return BackfillReport(
                market=normalized,
                formula_version=BALANCED_RS_FORMULA_VERSION,
                requested_start_date=start_date,
                through_date=through_date,
                first_valid_date=None,
                candidate_count=0,
                completed_count=0,
                failed_count=1,
                latest_run_id=None,
                group_row_count=0,
                results=(),
                validation_errors=("No valid balanced Market RS history boundary was found.",),
            )

        candidates = self.candidate_dates(
            db,
            market=normalized,
            through_date=through_date,
            first_valid_date=first_valid,
        )
        results: list[BackfillDateResult] = []
        for calculation_date in candidates:
            run = self.market_rs_repository.get_completed_exact(
                db,
                market=normalized,
                as_of_date=calculation_date,
                formula_version=BALANCED_RS_FORMULA_VERSION,
            )
            if start_date is not None and calculation_date < start_date and run is None:
                results.append(
                    BackfillDateResult(
                        as_of_date=calculation_date,
                        status="failed",
                        market_rs_run_id=None,
                        group_market_rs_run_id=None,
                        eligible_symbol_count=0,
                        group_row_count=0,
                        reason_code="resume_limiter_skipped_incomplete",
                        diagnostics={
                            "start_date": start_date.isoformat(),
                            "error": "Required date is incomplete before the calculation resume limit.",
                        },
                    )
                )
                continue
            try:
                if run is None:
                    run = self.market_rs_snapshot_service.calculate(
                        db,
                        market=normalized,
                        as_of_date=calculation_date,
                        formula_version=BALANCED_RS_FORMULA_VERSION,
                    )
                groups = self.canonical_group_service.calculate_and_store(
                    db,
                    market=normalized,
                    as_of_date=calculation_date,
                    formula_version=BALANCED_RS_FORMULA_VERSION,
                )
                if not groups:
                    raise RuntimeError("No eligible Group rows were produced")
                group_run_ids = {
                    int(row["market_rs_run_id"])
                    for row in groups
                    if row.get("market_rs_run_id") is not None
                }
                group_run_id = next(iter(group_run_ids)) if len(group_run_ids) == 1 else None
                if group_run_id != run.id:
                    raise RuntimeError("Group rows do not reference the exact Market RS run")
                results.append(
                    BackfillDateResult(
                        as_of_date=calculation_date,
                        status="completed",
                        market_rs_run_id=run.id,
                        group_market_rs_run_id=group_run_id,
                        eligible_symbol_count=int(run.eligible_symbol_count),
                        group_row_count=len(groups),
                    )
                )
            except Exception as exc:
                db.rollback()
                stage = (
                    "group_calculation"
                    if run is not None and not isinstance(exc, MarketRsInputUnavailable)
                    else "stock_calculation"
                )
                reason_code = self._reason_code(exc, stage=stage)
                if stage == "group_calculation":
                    reason_code = "group_calculation_failed"
                diagnostics: dict[str, object] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if isinstance(exc, MarketRsInputUnavailable):
                    diagnostics.update(exc.diagnostics)
                results.append(
                    BackfillDateResult(
                        as_of_date=calculation_date,
                        status="failed",
                        market_rs_run_id=getattr(run, "id", None),
                        group_market_rs_run_id=None,
                        eligible_symbol_count=int(
                            getattr(run, "eligible_symbol_count", 0) or 0
                        ),
                        group_row_count=0,
                        reason_code=reason_code,
                        diagnostics=diagnostics,
                    )
                )

        completed = tuple(item for item in results if item.status == "completed")
        failed = tuple(item for item in results if item.status == "failed")
        return BackfillReport(
            market=normalized,
            formula_version=BALANCED_RS_FORMULA_VERSION,
            requested_start_date=start_date or first_valid,
            through_date=through_date,
            first_valid_date=first_valid,
            candidate_count=len(candidates),
            completed_count=len(completed),
            failed_count=len(failed),
            latest_run_id=(completed[-1].market_rs_run_id if completed else None),
            group_row_count=sum(item.group_row_count for item in completed),
            results=tuple(results),
        )

    @staticmethod
    def _json_file(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _static_manifest_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _validate_run_and_groups(
        self,
        db: Session,
        *,
        market: str,
        calculation_date: date,
        errors: list[str],
    ) -> Any | None:
        run = self.market_rs_repository.get_completed_exact(
            db,
            market=market,
            as_of_date=calculation_date,
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )
        if run is None:
            errors.append(f"Missing completed stock RS snapshot for {calculation_date}.")
            return None
        if len(run.rows) != int(run.eligible_symbol_count):
            errors.append(
                f"Stock row count mismatch for {calculation_date}: "
                f"{len(run.rows)} != {run.eligible_symbol_count}."
            )
        for row in run.rows:
            ratings = (
                row.overall_rs,
                row.rs_1m,
                row.rs_3m,
                row.rs_6m,
                row.rs_9m,
                row.rs_12m,
            )
            if any(not isinstance(value, int) or value < 1 or value > 99 for value in ratings):
                errors.append(f"Out-of-range stock RS rating for {row.symbol} on {calculation_date}.")
            if not math.isfinite(float(row.weighted_composite)):
                errors.append(f"Non-finite stock RS composite for {row.symbol} on {calculation_date}.")

        group_rows = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == market,
                IBDGroupRank.date == calculation_date,
                IBDGroupRank.rs_formula_version == BALANCED_RS_FORMULA_VERSION,
            )
            .all()
        )
        eligible_symbols = {row.symbol for row in run.rows}
        expected_groups: set[str] = set()
        try:
            for group_name in IBDIndustryService.get_all_groups(db, market=market):
                symbols = set(
                    IBDIndustryService.get_group_symbols(
                        db,
                        group_name,
                        market=market,
                    )
                )
                if len(symbols & eligible_symbols) >= 3:
                    expected_groups.add(group_name)
        except Exception as exc:
            errors.append(f"Could not reconstruct expected Groups for {calculation_date}: {exc}")
        stored_groups = {row.industry_group for row in group_rows}
        missing_groups = sorted(expected_groups - stored_groups)
        if missing_groups:
            errors.append(
                f"Missing eligible Group rows for {calculation_date}: {', '.join(missing_groups)}."
            )
        if any(
            row.rs_formula_version != BALANCED_RS_FORMULA_VERSION
            or row.market_rs_run_id != run.id
            for row in group_rows
        ):
            errors.append(f"Mixed Group formula/run IDs for {calculation_date}.")
        ordered = sorted(group_rows, key=lambda row: row.rank)
        if [row.rank for row in ordered] != list(range(1, len(ordered) + 1)):
            errors.append(f"Non-contiguous Group ranks for {calculation_date}.")
        deterministic = sorted(
            group_rows,
            key=lambda row: (-float(row.avg_rs_rating), row.industry_group),
        )
        if [row.industry_group for row in ordered] != [
            row.industry_group for row in deterministic
        ]:
            errors.append(f"Non-deterministic Group rank order for {calculation_date}.")
        return run

    def _validate_static_artifacts(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        latest_run: Any,
        feature_run_id: int,
        static_staging_dir: Path,
        errors: list[str],
    ) -> tuple[str | None, str | None]:
        manifest_path = static_staging_dir / "manifest.json"
        if not manifest_path.is_file():
            errors.append("Missing staged static-site-v3 manifest.")
            return None, None
        manifest_hash = self._static_manifest_hash(manifest_path)
        try:
            manifest = self._json_file(manifest_path)
        except Exception as exc:
            errors.append(f"Invalid staged static manifest: {exc}")
            return manifest_hash, None
        if manifest.get("schema_version") != STATIC_SITE_SCHEMA_VERSION:
            errors.append("Staged static manifest is not static-site-v3.")
        entry = (manifest.get("markets") or {}).get(market) or {}
        expected_entry = {
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": latest_run.id,
            "rs_as_of_date": through_date.isoformat(),
            "rs_universe_size": latest_run.eligible_symbol_count,
        }
        for key, expected in expected_entry.items():
            if entry.get(key) != expected:
                errors.append(
                    f"Staged static Market metadata {key}={entry.get(key)!r}; expected {expected!r}."
                )

        market_dir = static_staging_dir / "markets" / market.lower()
        groups_path = market_dir / "groups.json"
        scan_path = market_dir / "scan" / "manifest.json"
        if not groups_path.is_file():
            errors.append("Missing staged static Groups artifact.")
            groups_payload: dict[str, Any] = {}
        else:
            groups_payload = self._json_file(groups_path)
            if (
                groups_payload.get("schema_version") != STATIC_SITE_SCHEMA_VERSION
                or groups_payload.get("rs_formula_version") != BALANCED_RS_FORMULA_VERSION
                or groups_payload.get("market_rs_run_id") != latest_run.id
            ):
                errors.append("Staged Groups artifact has mismatched schema/formula/run metadata.")
        if not scan_path.is_file():
            errors.append("Missing staged static Scan manifest.")
            scan_payload: dict[str, Any] = {}
        else:
            scan_payload = self._json_file(scan_path)
            if scan_payload.get("run_id") != feature_run_id:
                errors.append("Staged Scan manifest names a different Feature run.")

        live_groups = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == market,
                IBDGroupRank.date == through_date,
                IBDGroupRank.rs_formula_version == BALANCED_RS_FORMULA_VERSION,
            )
            .order_by(IBDGroupRank.rank)
            .all()
        )
        static_groups = (
            (((groups_payload.get("payload") or {}).get("rankings") or {}).get("rankings"))
            or []
        )
        static_by_name = {row.get("industry_group"): row for row in static_groups}
        parity_fields = (
            "rank",
            "avg_rs_rating",
            "avg_rs_rating_1m",
            "avg_rs_rating_3m",
            "num_stocks",
            "top_symbol",
            "rs_formula_version",
            "market_rs_run_id",
        )
        for live in live_groups:
            static = static_by_name.get(live.industry_group)
            if static is None:
                errors.append(f"Static Groups artifact omits {live.industry_group}.")
                continue
            for field in parity_fields:
                live_value = getattr(live, field)
                static_value = static.get(field)
                if isinstance(live_value, float):
                    equal = math.isclose(
                        live_value,
                        float(static_value),
                        rel_tol=0,
                        abs_tol=1e-9,
                    ) if static_value is not None else False
                else:
                    equal = live_value == static_value
                if not equal:
                    errors.append(
                        f"Live/static Group mismatch for {live.industry_group}.{field}."
                    )

        scan_rows = list(scan_payload.get("preview_rows") or [])
        if not scan_rows:
            scan_rows = list(scan_payload.get("initial_rows") or [])[:10]
        stock_by_symbol = {row.symbol: row for row in latest_run.rows}
        stock_fields = {
            "rs_rating": "overall_rs",
            "rs_rating_1m": "rs_1m",
            "rs_rating_3m": "rs_3m",
            "rs_rating_12m": "rs_12m",
        }
        compared = 0
        for static_row in sorted(scan_rows, key=lambda row: str(row.get("symbol"))):
            stock = stock_by_symbol.get(static_row.get("symbol"))
            if stock is None:
                continue
            compared += 1
            for static_field, stock_field in stock_fields.items():
                if static_row.get(static_field) != getattr(stock, stock_field):
                    errors.append(
                        f"Live/static stock mismatch for {stock.symbol}.{static_field}."
                    )
            if static_row.get("rs_formula_version") not in (
                None,
                BALANCED_RS_FORMULA_VERSION,
            ):
                errors.append(f"Static stock {stock.symbol} has a mixed RS formula.")
            if static_row.get("market_rs_run_id") not in (None, latest_run.id):
                errors.append(f"Static stock {stock.symbol} has a mixed Market RS run.")
        if latest_run.rows and compared == 0:
            errors.append("No deterministic stock sample overlaps the staged Scan preview.")

        rrg_path = market_dir / "groups_rrg.json"
        rrg_status: str | None = None
        try:
            expected_rrg = StaticGroupsRRGDatabasePayloadSource(
                schema_version=STATIC_SITE_SCHEMA_VERSION,
            ).build(
                db=db,
                generated_at="validation",
                expected_as_of_date=through_date,
                market=market,
                formula_version=BALANCED_RS_FORMULA_VERSION,
            )
        except StaticGroupsRRGUnavailableError as exc:
            reason = str(exc).lower()
            if "too short" in reason or "absent" in reason or "could be computed" in reason:
                rrg_status = "insufficient_balanced_history"
                if rrg_path.is_file():
                    payload = self._json_file(rrg_path)
                    if payload.get("available") or payload.get("payload"):
                        errors.append("Static RRG contains coordinates despite insufficient balanced history.")
            else:
                errors.append(f"Balanced RRG validation failed: {exc}")
        else:
            rrg_status = "available"
            if not rrg_path.is_file():
                errors.append("Missing staged balanced RRG artifact.")
            else:
                actual_rrg = self._json_file(rrg_path)
                if actual_rrg.get("rs_formula_version") != BALANCED_RS_FORMULA_VERSION:
                    errors.append("Staged RRG artifact uses a mixed RS formula.")
                if actual_rrg.get("payload") != expected_rrg.get("payload"):
                    errors.append("Live/static RRG coordinates diverge.")
        return manifest_hash, rrg_status

    def validate_activation(
        self,
        db: Session,
        *,
        market: str,
        through_date: date,
        feature_run_id: int,
        static_staging_dir: Path,
    ) -> ActivationValidationReport:
        normalized = self._normalize_market(market)
        errors: list[str] = []
        first_valid = self.earliest_backfillable_date(
            db,
            market=normalized,
            through_date=through_date,
        )
        candidates = self.candidate_dates(
            db,
            market=normalized,
            through_date=through_date,
            first_valid_date=first_valid,
        ) if first_valid is not None else ()
        if not candidates:
            errors.append("No required balanced Market RS candidate dates were found.")

        latest_run = None
        for calculation_date in candidates:
            run = self._validate_run_and_groups(
                db,
                market=normalized,
                calculation_date=calculation_date,
                errors=errors,
            )
            if calculation_date == through_date:
                latest_run = run
        if candidates and candidates[-1] != through_date:
            errors.append("Candidate trading-date history does not reach the activation date.")

        feature_repo = self._feature_run_repository_factory(db)
        feature = None
        try:
            feature = feature_repo.get_run(feature_run_id)
        except Exception as exc:
            errors.append(f"Feature run {feature_run_id} is unavailable: {exc}")
        if feature is not None:
            config = feature.config or {}
            expected_config = {
                "market": normalized,
                "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
                "market_rs_run_id": getattr(latest_run, "id", None),
                "rs_as_of_date": through_date.isoformat(),
            }
            if feature.status != RunStatus.PUBLISHED or feature.as_of_date != through_date:
                errors.append("Candidate Feature run is not published for the activation date.")
            for key, expected in expected_config.items():
                if config.get(key) != expected:
                    errors.append(
                        f"Candidate Feature run {key}={config.get(key)!r}; expected {expected!r}."
                    )

        manifest_hash = None
        rrg_status = None
        if latest_run is not None:
            try:
                manifest_hash, rrg_status = self._validate_static_artifacts(
                    db,
                    market=normalized,
                    through_date=through_date,
                    latest_run=latest_run,
                    feature_run_id=feature_run_id,
                    static_staging_dir=Path(static_staging_dir),
                    errors=errors,
                )
            except Exception as exc:
                errors.append(f"Staged static artifact validation failed: {exc}")
        elif not (Path(static_staging_dir) / "manifest.json").is_file():
            errors.append("Missing staged static-site-v3 manifest.")

        return ActivationValidationReport(
            market=normalized,
            formula_version=BALANCED_RS_FORMULA_VERSION,
            through_date=through_date,
            first_valid_date=first_valid,
            candidate_count=len(candidates),
            latest_market_rs_run_id=getattr(latest_run, "id", None),
            latest_universe_hash=getattr(latest_run, "universe_hash", None),
            feature_run_id=getattr(feature, "id", feature_run_id),
            feature_universe_hash=getattr(feature, "universe_hash", None),
            static_manifest_sha256=manifest_hash,
            errors=tuple(dict.fromkeys(errors)),
            rrg_status=rrg_status,
        )

    @staticmethod
    def _validate_feature_candidate(
        feature: Any,
        *,
        market: str,
        formula_version: str,
        through_date: date,
        market_rs_run_id: int,
    ) -> None:
        status = getattr(feature.status, "value", feature.status)
        config = feature.config or {}
        if (
            status != RunStatus.PUBLISHED.value
            or feature.as_of_date != through_date
            or config.get("market") != market
            or config.get("rs_formula_version") != formula_version
            or config.get("market_rs_run_id") != market_rs_run_id
            or config.get("rs_as_of_date") != through_date.isoformat()
        ):
            raise MarketRsActivationRejected(
                ("Candidate Feature run changed or no longer matches the validated Market RS run.",)
            )

    def activate(
        self,
        db: Session,
        *,
        market: str,
        formula_version: str,
        feature_run_id: int,
        validation: ActivationValidationReport,
    ) -> None:
        normalized = self._normalize_market(market)
        if not validation.ok:
            raise MarketRsActivationRejected(validation.errors)
        if (
            validation.market != normalized
            or validation.formula_version != formula_version
            or validation.feature_run_id != feature_run_id
            or formula_version != BALANCED_RS_FORMULA_VERSION
            or validation.latest_market_rs_run_id is None
        ):
            raise MarketRsActivationRejected(("Activation request does not match its validation report.",))

        try:
            current_run = self.market_rs_repository.get_completed_exact(
                db,
                market=normalized,
                as_of_date=validation.through_date,
                formula_version=formula_version,
            )
            if (
                current_run is None
                or current_run.id != validation.latest_market_rs_run_id
                or current_run.universe_hash != validation.latest_universe_hash
            ):
                raise MarketRsActivationRejected(
                    ("Validated Market RS run changed before activation.",)
                )
            feature_repo = self._feature_run_repository_factory(db)
            feature = feature_repo.get_run(feature_run_id)
            self._validate_feature_candidate(
                feature,
                market=normalized,
                formula_version=formula_version,
                through_date=validation.through_date,
                market_rs_run_id=current_run.id,
            )
            if feature.universe_hash != validation.feature_universe_hash:
                raise MarketRsActivationRejected(
                    ("Validated Feature universe changed before activation.",)
                )
            self.market_rs_repository.activate_formula(
                db,
                market=normalized,
                formula_version=formula_version,
            )
            feature_repo.repoint_published(
                feature_run_id,
                pointer_key=f"latest_published_market:{normalized}",
            )
            db.commit()
        except Exception:
            db.rollback()
            raise


__all__ = [
    "ActivationValidationReport",
    "BackfillDateResult",
    "BackfillReport",
    "MarketRsActivationRejected",
    "MarketRsRolloutService",
]
