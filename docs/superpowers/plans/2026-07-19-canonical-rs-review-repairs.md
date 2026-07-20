# Canonical RS Review Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair every blocking canonical RS review finding so Scan, live Groups, static Groups, and RRG use one exact Market/date/formula identity, adjusted-close-only inputs, and fail-closed publication paths.

**Architecture:** Add one immutable Group snapshot identity, an exact stored-snapshot reader, and a formula-aware coordinator used by every Group producer. Make scanner hydration a typed provider contract, make static combination and rollout activation verify formula/content integrity, and extract the oversized RS-touched modules behind compatibility facades.

**Tech Stack:** Python 3, SQLAlchemy, PostgreSQL/SQLite, Celery, FastAPI/Pydantic, pandas, React 18, TanStack Query, MUI, pytest, Vitest, ESLint, GitHub Actions.

## Global Constraints

- Formula versions remain exactly `balanced-horizon-percentile-v2` and `legacy-linear-v1`.
- Balanced weights remain exactly 1M 20%, 3M 30%, 6M 20%, 9M 15%, and 12M 15%; percentile mathematics is unchanged.
- Group identity is always `market + as_of_date + formula_version`; balanced rows must also reference the exact completed Market RS run.
- No Group, feature-enrichment, static, or RRG query may omit `rs_formula_version` after its boundary resolves the active formula.
- Feature enrichment and static export only read stored Group rank; they never recompute Group rank from serialized stock rows.
- Canonical Market RS accepts `StockPrice.adj_close` only and records `price_basis: adj_close_only`; calculation never substitutes raw `close`.
- Canonical scanner filtering does not require benchmark bars; legacy scanner calculation still does.
- Static combine validates current and fallback artifacts against the requested formula and never falls back across formulas.
- Public Scan/Group payloads, static schema v3, database schema, legacy rollback, Group ordering, and RRG mathematics remain unchanged.
- Any caught database failure must call `rollback()` before that session is reused.
- No modified production Python module may exceed 1,000 lines; no new extracted production module may exceed 700 lines; `GroupRankingsPage.jsx` must remain below 1,000 lines.
- Preserve unrelated user changes. Use `apply_patch` for edits, red-green-refactor TDD, one focused commit per task, and update `stockscreenclaude-stm` throughout.
- The known hook bug `stockscreenclaude-7jr` exports a duplicate root `issues.jsonl`; commits in this plan use `--no-verify` after explicit staging, and every task verifies that only `.beads/issues.jsonl` is tracked.

---

## File Structure

New domain and Group services:

- `backend/app/domain/relative_strength/group_snapshot.py`: immutable Group identity.
- `backend/app/services/group_rank_snapshot_reader.py`: exact formula/date reads and integrity checks.
- `backend/app/services/group_rank_snapshot_coordinator.py`: balanced/legacy snapshot creation and per-date recovery.
- `backend/app/services/feature_run_rs_identity.py`: deterministic persisted/inferred-legacy feature identity.
- `backend/app/services/feature_run_group_enrichment.py`: exact-snapshot Group metadata enrichment for feature rows.

New static and rollout services:

- `backend/app/services/static_group_section_builder.py`: exact static Group payload assembly.
- `backend/app/services/static_site_errors.py`: shared static export exceptions without import cycles.
- `backend/app/services/static_artifact_combiner.py`: current/fallback selection and formula validation.
- `backend/app/services/market_rs_rollout_models.py`: rollout reports and rejection types.
- `backend/app/services/market_rs_backfill_service.py`: resumable stock/Group history creation.
- `backend/app/services/market_rs_activation_validator.py`: stock/Group/static/RRG validation.
- `backend/app/services/market_rs_activation_service.py`: final hash check and pointer switch.
- `backend/app/domain/relative_strength/run_policy.py`: balanced adjusted-price compatibility marker.

New structural modules:

- `backend/app/wiring/market_rs_services.py`: construction of core Market RS services.
- `backend/app/services/group_rank_query_service.py`: formula-aware Group reads/history/movers.
- `backend/app/services/legacy_group_rank_data.py`: legacy cache prefetch and vectorized RS inputs.
- `backend/app/services/legacy_group_rank_calculator.py`: one-date legacy Group calculation/storage.
- `backend/app/services/legacy_group_rank_backfill.py`: legacy range/gap backfills.
- `backend/app/tasks/group_rank_workflows.py`: daily task workflow without Celery decoration.
- `backend/app/services/static_chart_bundle_exporter.py`: static chart selection and serialization.
- `frontend/src/features/groups/groupRankingFields.js`: shared overall/1M/3M definitions and formatter.
- `frontend/src/features/groups/LiveGroupRankingsTable.jsx`: sortable live Group table.
- `frontend/src/features/groups/GroupDetailModal.jsx`: live Group detail UI.
- `frontend/src/static/components/StaticGroupRankingsTable.jsx`: static Group table.

Compatibility facades retained:

- `backend/app/services/ibd_group_rank_service.py`
- `backend/app/services/market_rs_rollout_service.py`
- `backend/app/services/static_site_export_service.py`
- `backend/app/wiring/bootstrap.py`

---

### Task 1: Exact Group Snapshot Identity and Reader

**Files:**
- Create: `backend/app/domain/relative_strength/group_snapshot.py`
- Modify: `backend/app/domain/relative_strength/__init__.py`
- Create: `backend/app/services/group_rank_snapshot_reader.py`
- Modify: `backend/app/services/group_ranking_payloads.py`
- Create: `backend/tests/unit/services/test_group_rank_snapshot_reader.py`
- Modify: `backend/tests/unit/test_group_ranking_payloads.py`

**Interfaces:**
- Consumes: existing `IBDGroupRank`, `MarketRsRun`, `StockUniverse`, and `rank_record_payload` persistence contracts.
- Produces: `GroupSnapshotIdentity`, `GroupSnapshotIntegrityError`, `GroupRankSnapshotReader.load_exact`, `load_rank_map`, and `available_dates`.

- [ ] **Step 1: Write failing identity and exact-reader tests**

Add these cases to `backend/tests/unit/services/test_group_rank_snapshot_reader.py`:

```python
from datetime import date

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)
from app.infra.db.models.relative_strength import MarketRsRun
from app.models.industry import IBDGroupRank
from app.services.group_rank_snapshot_reader import (
    GroupRankSnapshotReader,
    GroupSnapshotIntegrityError,
)


AS_OF = date(2026, 4, 10)


def _run(db_session, *, run_id=41, as_of_date=AS_OF):
    row = MarketRsRun(
        id=run_id,
        market="US",
        as_of_date=as_of_date,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        status="completed",
        benchmark_symbol="SPY",
        benchmark_as_of_date=as_of_date,
        universe_hash="reader-test",
        expected_symbol_count=3,
        eligible_symbol_count=3,
        excluded_symbol_count=0,
        diagnostics_json={"price_basis": "adj_close_only"},
    )
    db_session.add(row)
    db_session.flush()
    return row


def _rank(db_session, *, formula, rank, run_id=None, group="Software"):
    db_session.add(
        IBDGroupRank(
            market="US",
            industry_group=group,
            date=AS_OF,
            rank=rank,
            avg_rs_rating=88.0,
            num_stocks=3,
            num_stocks_rs_above_80=2,
            top_symbol="AAA",
            top_rs_rating=99.0,
            rs_formula_version=formula,
            market_rs_run_id=run_id,
        )
    )


def test_identity_normalizes_market_and_rejects_blank_formula():
    identity = GroupSnapshotIdentity(" hk ", AS_OF, BALANCED_RS_FORMULA_VERSION)
    assert identity.market == "HK"
    with pytest.raises(ValueError, match="formula_version"):
        GroupSnapshotIdentity("US", AS_OF, " ")


def test_load_exact_never_crosses_formula(db_session):
    run = _run(db_session)
    _rank(db_session, formula=BALANCED_RS_FORMULA_VERSION, rank=1, run_id=run.id)
    _rank(db_session, formula=LEGACY_RS_FORMULA_VERSION, rank=9, group="Legacy")
    db_session.commit()

    rows = GroupRankSnapshotReader().load_exact(
        db_session,
        identity=GroupSnapshotIdentity("US", AS_OF, BALANCED_RS_FORMULA_VERSION),
    )

    assert [row["industry_group"] for row in rows] == ["Software"]
    assert rows[0]["market_rs_run_id"] == run.id


def test_balanced_rows_must_share_the_exact_completed_run(db_session):
    first = _run(db_session, run_id=41)
    _run(db_session, run_id=42, as_of_date=date(2026, 4, 9))
    _rank(db_session, formula=BALANCED_RS_FORMULA_VERSION, rank=1, run_id=first.id)
    _rank(
        db_session,
        formula=BALANCED_RS_FORMULA_VERSION,
        rank=2,
        run_id=42,
        group="Hardware",
    )
    db_session.commit()

    with pytest.raises(GroupSnapshotIntegrityError, match="Market RS run"):
        GroupRankSnapshotReader().load_exact(
            db_session,
            identity=GroupSnapshotIdentity("US", AS_OF, BALANCED_RS_FORMULA_VERSION),
        )


def test_available_dates_is_formula_scoped(db_session):
    _rank(db_session, formula=LEGACY_RS_FORMULA_VERSION, rank=1)
    db_session.commit()
    assert GroupRankSnapshotReader().available_dates(
        db_session,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=AS_OF,
    ) == ()
```

- [ ] **Step 2: Run the new tests and verify the missing imports**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/services/test_group_rank_snapshot_reader.py -q
```

Expected: collection fails because `GroupSnapshotIdentity` and `GroupRankSnapshotReader` do not exist.

- [ ] **Step 3: Implement the immutable identity**

Create `backend/app/domain/relative_strength/group_snapshot.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class GroupSnapshotIdentity:
    market: str
    as_of_date: date
    formula_version: str

    def __post_init__(self) -> None:
        market = str(self.market).strip().upper()
        formula = str(self.formula_version).strip()
        if not market:
            raise ValueError("market is required")
        if not formula:
            raise ValueError("formula_version is required")
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "formula_version", formula)
```

Export it from `backend/app/domain/relative_strength/__init__.py`.

- [ ] **Step 4: Implement the exact reader and integrity checks**

Create `backend/app/services/group_rank_snapshot_reader.py` with this public behavior:

```python
from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)
from app.infra.db.models.relative_strength import MarketRsRun
from app.models.industry import IBDGroupRank
from app.services.group_ranking_payloads import (
    annotate_top_symbol_names,
    rank_record_payload,
)


class GroupSnapshotIntegrityError(RuntimeError):
    pass


class GroupSnapshotUnavailable(LookupError):
    def __init__(self, identity: GroupSnapshotIdentity) -> None:
        self.identity = identity
        super().__init__(
            f"Group snapshot unavailable for {identity.market} on "
            f"{identity.as_of_date.isoformat()} ({identity.formula_version})"
        )


class GroupRankSnapshotReader:
    def load_exact(
        self,
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
        include_top_symbol_names: bool = True,
    ) -> list[dict[str, Any]]:
        records = (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.market == identity.market,
                IBDGroupRank.date == identity.as_of_date,
                IBDGroupRank.rs_formula_version == identity.formula_version,
            )
            .order_by(IBDGroupRank.rank, IBDGroupRank.industry_group)
            .all()
        )
        self._validate(db, identity=identity, records=records)
        payload = [
            rank_record_payload(
                record,
                pct_rs_above_80=(
                    round(
                        100.0
                        * int(record.num_stocks_rs_above_80 or 0)
                        / int(record.num_stocks),
                        1,
                    )
                    if record.num_stocks
                    else None
                ),
            )
            for record in records
        ]
        if include_top_symbol_names:
            annotate_top_symbol_names(db, payload)
        return payload

    def load_rank_map(
        self,
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
    ) -> dict[str, int]:
        return {
            str(row["industry_group"]): int(row["rank"])
            for row in self.load_exact(
                db,
                identity=identity,
                include_top_symbol_names=False,
            )
        }

    def available_dates(
        self,
        db: Session,
        *,
        market: str,
        formula_version: str,
        through_date: date,
    ) -> tuple[date, ...]:
        rows = (
            db.query(IBDGroupRank.date)
            .filter(
                IBDGroupRank.market == str(market).strip().upper(),
                IBDGroupRank.rs_formula_version == str(formula_version).strip(),
                IBDGroupRank.date <= through_date,
            )
            .distinct()
            .order_by(IBDGroupRank.date)
            .all()
        )
        return tuple(row[0] for row in rows)

    @staticmethod
    def _validate(
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
        records: list[IBDGroupRank],
    ) -> None:
        if not records:
            return
        ranks = [int(record.rank) for record in records]
        if ranks != list(range(1, len(records) + 1)):
            raise GroupSnapshotIntegrityError("Group ranks are not contiguous")
        if any(
            record.market != identity.market
            or record.date != identity.as_of_date
            or record.rs_formula_version != identity.formula_version
            for record in records
        ):
            raise GroupSnapshotIntegrityError("Group rows do not match their identity")
        if identity.formula_version != BALANCED_RS_FORMULA_VERSION:
            return
        run_ids = {record.market_rs_run_id for record in records}
        if None in run_ids or len(run_ids) != 1:
            raise GroupSnapshotIntegrityError("Balanced Group rows mix Market RS run IDs")
        run = db.get(MarketRsRun, int(next(iter(run_ids))))
        if (
            run is None
            or run.status != "completed"
            or run.market != identity.market
            or run.as_of_date != identity.as_of_date
            or run.formula_version != identity.formula_version
        ):
            raise GroupSnapshotIntegrityError("Group rows reference the wrong Market RS run")
```

Move the existing batch name-map logic from `IBDGroupRankService` into public
`annotate_top_symbol_names(db, rows)` in `group_ranking_payloads.py`; the helper
queries `StockUniverse` once and mutates only each payload row's
`top_symbol_name`. Add a payload test for one known and one unknown symbol.
Task 10 deletes the old private duplicate when it extracts the facade.

- [ ] **Step 5: Run focused tests and existing Group serialization tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_group_rank_snapshot_reader.py \
  tests/unit/test_canonical_group_ranking_service.py \
  tests/unit/test_group_ranking_payloads.py \
  tests/unit/test_group_rank_service.py::test_current_rankings_reads_only_active_formula -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the identity boundary**

```bash
git add backend/app/domain/relative_strength backend/app/services/group_rank_snapshot_reader.py backend/app/services/group_ranking_payloads.py backend/tests/unit/services/test_group_rank_snapshot_reader.py backend/tests/unit/test_group_ranking_payloads.py
test ! -e issues.jsonl
git commit --no-verify -m "refactor: add exact Group snapshot identity"
```

---

### Task 2: Formula-Aware Group Snapshot Coordinator and Backfill

**Files:**
- Create: `backend/app/services/group_rank_snapshot_coordinator.py`
- Modify: `backend/app/services/group_rank_history_backfill_service.py`
- Create: `backend/tests/unit/services/test_group_rank_snapshot_coordinator.py`
- Modify: `backend/tests/unit/test_group_rank_history_backfill_service.py`

**Interfaces:**
- Consumes: `GroupSnapshotIdentity`, `GroupRankSnapshotReader`, `MarketRsSnapshotService.calculate`, `CanonicalGroupRankingService.calculate_and_store`, and the explicit legacy `calculate_group_rankings` path.
- Produces: `GroupSnapshotStatus`, `GroupSnapshotResult`, `GroupBackfillReport`, `GroupRankSnapshotCoordinator.ensure_snapshot`, and `backfill`.

- [ ] **Step 1: Write failing coordinator tests for dispatch, formula isolation, and rollback**

Add tests with injected fakes:

```python
from datetime import date
from unittest.mock import Mock, call

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)
from app.services.group_rank_snapshot_coordinator import (
    GroupRankSnapshotCoordinator,
    GroupSnapshotStatus,
)


AS_OF = date(2026, 4, 10)


def _coordinator(reader, stock, canonical, legacy):
    return GroupRankSnapshotCoordinator(
        reader=reader,
        market_rs_snapshot_service=stock,
        canonical_group_service=canonical,
        legacy_group_service=legacy,
    )


def test_balanced_snapshot_never_calls_legacy(db_session):
    reader = Mock()
    reader.load_exact.side_effect = [[], [{"market_rs_run_id": 44}]]
    stock = Mock()
    stock.calculate.return_value.id = 44
    canonical = Mock()
    legacy = Mock()
    identity = GroupSnapshotIdentity("US", AS_OF, BALANCED_RS_FORMULA_VERSION)

    result = _coordinator(reader, stock, canonical, legacy).ensure_snapshot(
        db_session, identity=identity
    )

    assert result.status is GroupSnapshotStatus.PROCESSED
    stock.calculate.assert_called_once_with(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    canonical.calculate_and_store.assert_called_once()
    legacy.calculate_group_rankings.assert_not_called()


def test_legacy_snapshot_never_calls_canonical_stock_or_group(db_session):
    reader = Mock()
    reader.load_exact.side_effect = [[], [{"market_rs_run_id": None}]]
    stock = Mock()
    canonical = Mock()
    legacy = Mock()
    identity = GroupSnapshotIdentity("US", AS_OF, LEGACY_RS_FORMULA_VERSION)

    _coordinator(reader, stock, canonical, legacy).ensure_snapshot(
        db_session, identity=identity
    )

    legacy.calculate_group_rankings.assert_called_once_with(
        db_session,
        AS_OF,
        market="US",
        formula_version=LEGACY_RS_FORMULA_VERSION,
    )
    stock.calculate.assert_not_called()
    canonical.calculate_and_store.assert_not_called()


def test_backfill_rolls_back_failed_date_before_processing_next(db_session):
    coordinator = _coordinator(Mock(), Mock(), Mock(), Mock())
    first = GroupSnapshotIdentity("US", date(2026, 4, 9), BALANCED_RS_FORMULA_VERSION)
    second = GroupSnapshotIdentity("US", AS_OF, BALANCED_RS_FORMULA_VERSION)
    coordinator.ensure_snapshot = Mock(
        side_effect=[RuntimeError("database aborted"), Mock(status=GroupSnapshotStatus.PROCESSED, row_count=3, market_rs_run_id=8)]
    )
    db_session.rollback = Mock(wraps=db_session.rollback)

    report = coordinator.backfill(
        db_session,
        identities=(first, second),
        continue_on_error=True,
    )

    assert db_session.rollback.call_count == 1
    assert [item.status for item in report.results] == [
        GroupSnapshotStatus.ERRORED,
        GroupSnapshotStatus.PROCESSED,
    ]
    assert coordinator.ensure_snapshot.call_args_list == [
        call(db_session, identity=first),
        call(db_session, identity=second),
    ]
```

- [ ] **Step 2: Run tests and verify the coordinator import fails**

```bash
cd backend && source venv/bin/activate && pytest tests/unit/services/test_group_rank_snapshot_coordinator.py -q
```

Expected: collection fails because the coordinator module does not exist.

- [ ] **Step 3: Implement coordinator results and dispatch**

Create `backend/app/services/group_rank_snapshot_coordinator.py` with these exact public signatures:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    GroupSnapshotIdentity,
)


class GroupSnapshotStatus(StrEnum):
    EXISTING = "existing"
    PROCESSED = "processed"
    EMPTY = "empty"
    ERRORED = "errored"


@dataclass(frozen=True)
class GroupSnapshotResult:
    identity: GroupSnapshotIdentity
    status: GroupSnapshotStatus
    row_count: int
    market_rs_run_id: int | None
    reason_code: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class GroupBackfillReport:
    results: tuple[GroupSnapshotResult, ...]

    @property
    def processed(self) -> int:
        return sum(item.status is GroupSnapshotStatus.PROCESSED for item in self.results)

    @property
    def existing(self) -> int:
        return sum(item.status is GroupSnapshotStatus.EXISTING for item in self.results)

    @property
    def errors(self) -> int:
        return sum(item.status is GroupSnapshotStatus.ERRORED for item in self.results)


class GroupRankSnapshotCoordinator:
    def __init__(
        self,
        *,
        reader,
        market_rs_snapshot_service,
        canonical_group_service,
        legacy_group_service,
    ) -> None:
        self.reader = reader
        self.market_rs_snapshot_service = market_rs_snapshot_service
        self.canonical_group_service = canonical_group_service
        self.legacy_group_service = legacy_group_service

    def ensure_snapshot(
        self,
        db: Session,
        *,
        identity: GroupSnapshotIdentity,
    ) -> GroupSnapshotResult:
        existing = self.reader.load_exact(
            db,
            identity=identity,
            include_top_symbol_names=False,
        )
        if existing:
            return self._result(identity, GroupSnapshotStatus.EXISTING, existing)

        if identity.formula_version == BALANCED_RS_FORMULA_VERSION:
            run = self.market_rs_snapshot_service.calculate(
                db,
                market=identity.market,
                as_of_date=identity.as_of_date,
                formula_version=identity.formula_version,
            )
            self.canonical_group_service.calculate_and_store(
                db,
                market=identity.market,
                as_of_date=identity.as_of_date,
                formula_version=identity.formula_version,
            )
            rows = self.reader.load_exact(
                db,
                identity=identity,
                include_top_symbol_names=False,
            )
            if rows and {row.get("market_rs_run_id") for row in rows} != {run.id}:
                raise RuntimeError("Group snapshot does not reference the exact Market RS run")
        elif identity.formula_version == LEGACY_RS_FORMULA_VERSION:
            self.legacy_group_service.calculate_group_rankings(
                db,
                identity.as_of_date,
                market=identity.market,
                formula_version=LEGACY_RS_FORMULA_VERSION,
            )
            rows = self.reader.load_exact(
                db,
                identity=identity,
                include_top_symbol_names=False,
            )
        else:
            raise ValueError(f"Unsupported Group RS formula: {identity.formula_version}")

        status = GroupSnapshotStatus.PROCESSED if rows else GroupSnapshotStatus.EMPTY
        return self._result(identity, status, rows)

    def backfill(
        self,
        db: Session,
        *,
        identities: Iterable[GroupSnapshotIdentity],
        continue_on_error: bool,
    ) -> GroupBackfillReport:
        results: list[GroupSnapshotResult] = []
        for identity in sorted(identities, key=lambda item: item.as_of_date):
            try:
                results.append(self.ensure_snapshot(db, identity=identity))
            except Exception as exc:
                db.rollback()
                if not continue_on_error:
                    raise
                results.append(
                    GroupSnapshotResult(
                        identity=identity,
                        status=GroupSnapshotStatus.ERRORED,
                        row_count=0,
                        market_rs_run_id=None,
                        reason_code=f"{type(exc).__name__}",
                        error=str(exc),
                    )
                )
        return GroupBackfillReport(results=tuple(results))

    @staticmethod
    def _result(identity, status, rows) -> GroupSnapshotResult:
        run_ids = {row.get("market_rs_run_id") for row in rows}
        run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
        return GroupSnapshotResult(
            identity=identity,
            status=status,
            row_count=len(rows),
            market_rs_run_id=run_id,
        )
```

- [ ] **Step 4: Make recent-history backfill require formula identity**

In `group_rank_history_backfill_service.py`:

- Add required `formula_version: str` to `backfill`.
- Filter `IBDGroupRank.rs_formula_version == formula_version` in the existing-date query.
- Replace the `GroupRankGapFiller` protocol with a coordinator dependency.
- Build one `GroupSnapshotIdentity` for every missing date and call:

```python
report = self.group_snapshot_coordinator.backfill(
    db,
    identities=tuple(
        GroupSnapshotIdentity(normalized_market, calculation_date, formula_version)
        for calculation_date in missing_dates
    ),
    continue_on_error=True,
)
```

Map `report.processed + report.existing`, `report.errors`, and per-date errors into the existing `GroupRankHistoryBackfillResult` payload so callers keep their current response shape.

- [ ] **Step 5: Run coordinator and history tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_group_rank_snapshot_coordinator.py \
  tests/unit/test_group_rank_history_backfill_service.py -q
```

Expected: all tests pass, including the rollback-before-next-date regression.

- [ ] **Step 6: Commit the coordinator**

```bash
git add backend/app/services/group_rank_snapshot_coordinator.py backend/app/services/group_rank_history_backfill_service.py backend/tests/unit/services/test_group_rank_snapshot_coordinator.py backend/tests/unit/test_group_rank_history_backfill_service.py
test ! -e issues.jsonl
git commit --no-verify -m "fix: coordinate formula-aware Group snapshots"
```

---

### Task 3: Route Live and Static Group Production Through the Coordinator

**Files:**
- Create: `backend/app/wiring/market_rs_services.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Modify: `backend/app/scripts/export_static_site.py`
- Modify: `backend/tests/unit/test_group_rank_tasks.py`
- Modify: `backend/tests/unit/test_export_static_site_script.py`
- Create: `backend/tests/unit/test_market_rs_services.py`

**Interfaces:**
- Consumes: Task 2 coordinator and existing runtime service accessors.
- Produces: `MarketRsServices`, `get_group_rank_snapshot_coordinator`, and one ordered static refresh path.

- [ ] **Step 1: Write failing live/static routing tests**

Add a Group task assertion that a balanced gap fill calls the coordinator with balanced identities and never calls `fill_gaps_optimized`. Update `test_run_daily_refresh_activates_balanced_rs_before_static_consumers` to require this order:

```python
assert events == [
    "prices",
    "market_rs_snapshot",
    "formula_activate",
    "formula_commit",
    "group_history",
    "exposure",
    "feature_snapshot",
]
```

Also assert the static history call receives:

```python
assert history_calls == [
    {
        "market": "US",
        "as_of_date": date(2026, 4, 10),
        "formula_version": BALANCED_RS_FORMULA_VERSION,
    }
]
```

- [ ] **Step 2: Run the routing tests and confirm the current order/default fails**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_export_static_site_script.py::test_run_daily_refresh_activates_balanced_rs_before_static_consumers -q
```

Expected: failure because static Group history still runs after the feature snapshot and omits the formula.

- [ ] **Step 3: Extract core Market RS construction from bootstrap**

Create `backend/app/wiring/market_rs_services.py`:

```python
from dataclasses import dataclass

from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.market_rs_inputs import MarketRsInputLoader
from app.services.market_rs_reader import SqlMarketRsReader
from app.services.market_rs_snapshot_service import MarketRsSnapshotService


@dataclass(frozen=True)
class MarketRsServices:
    repository: MarketRsRunRepository
    input_loader: MarketRsInputLoader
    snapshot_service: MarketRsSnapshotService
    reader: SqlMarketRsReader


def build_market_rs_services(
    *,
    session_factory,
    point_in_time_universe,
    market_calendar,
) -> MarketRsServices:
    repository = MarketRsRunRepository()
    input_loader = MarketRsInputLoader(
        point_in_time_universe=point_in_time_universe,
        market_calendar=market_calendar,
    )
    snapshot_service = MarketRsSnapshotService(
        input_loader=input_loader,
        repository=repository,
    )
    return MarketRsServices(
        repository=repository,
        input_loader=input_loader,
        snapshot_service=snapshot_service,
        reader=SqlMarketRsReader(session_factory, repository=repository),
    )
```

Replace the four separate lazy fields in `RuntimeServices` with one lazy `MarketRsServices` bundle. Keep existing accessor names as one-line delegations. Add lazy `group_rank_snapshot_reader()` and `group_rank_snapshot_coordinator()` methods plus global `get_group_rank_snapshot_coordinator()`.

- [ ] **Step 4: Replace task-local balanced dispatch with the coordinator**

In `group_rank_tasks.py`, remove `_prepare_group_rs_inputs` and `_calculate_balanced_group_dates`. Resolve the active formula once, construct exact identities, and call:

```python
report = get_group_rank_snapshot_coordinator().backfill(
    db,
    identities=tuple(
        GroupSnapshotIdentity(effective_market, item_date, resolved_formula)
        for item_date in missing_dates
    ),
    continue_on_error=True,
)
gap_stats = {
    "total_dates": len(report.results),
    "processed": report.processed,
    "skipped": report.existing,
    "errors": report.errors,
}
```

For the current date call `ensure_snapshot` with the same identity. Keep Celery retry, activity, cache invalidation, and payload behavior unchanged.

- [ ] **Step 5: Reorder static refresh and pass the selected formula**

Change `_ensure_group_rank_history` to:

```python
def _ensure_group_rank_history(
    *,
    as_of_date: date,
    market: str,
    formula_version: str,
) -> GroupRankHistoryBackfillResult:
    return GroupRankHistoryBackfillService(
        session_factory=SessionLocal,
        calendar_service=get_market_calendar_service(),
        group_snapshot_coordinator=get_group_rank_snapshot_coordinator(),
    ).backfill(
        as_of_date=as_of_date,
        market=market,
        formula_version=formula_version,
    )
```

In `_run_daily_refresh`, run the Group history loop immediately after
`results["market_rs"]` and before Market exposure/feature snapshots. Remove its
old `_snapshot_publishable(feature_snapshot)` gate: formula preparation either
returns successfully for the Market or raises before this point, and Group
history does not depend on exposure or a feature run. Record each result in
`results["group_rank_history_backfill"]` exactly as today. Preserve the
post-build enrichment loop for already-published runs; it still requires both a
publishable feature snapshot and a `ready_for_enrichment` Group result, and now
sees the exact stored snapshot created earlier.

- [ ] **Step 6: Run live/static routing and wiring tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_market_rs_services.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_export_static_site_script.py -q
```

Expected: all tests pass and no balanced assertion observes a legacy gap-fill call.

- [ ] **Step 7: Commit production routing**

```bash
git add backend/app/wiring/market_rs_services.py backend/app/wiring/bootstrap.py backend/app/tasks/group_rank_tasks.py backend/app/scripts/export_static_site.py backend/tests/unit/test_market_rs_services.py backend/tests/unit/test_group_rank_tasks.py backend/tests/unit/test_export_static_site_script.py
test ! -e issues.jsonl
git commit --no-verify -m "fix: route Group workflows through canonical identity"
```

---

### Task 4: Formula-Isolated Feature Group Enrichment

**Files:**
- Create: `backend/app/services/feature_run_rs_identity.py`
- Create: `backend/app/services/feature_run_group_enrichment.py`
- Modify: `backend/app/interfaces/tasks/feature_store_tasks.py`
- Modify: `backend/tests/unit/test_feature_store_tasks.py`
- Create: `backend/tests/unit/services/test_feature_run_rs_identity.py`
- Create: `backend/tests/unit/services/test_feature_run_group_enrichment.py`

**Interfaces:**
- Consumes: Task 1 exact reader and persisted `FeatureRun.config_json`.
- Produces: `FeatureRunRsIdentityResolution`, `FeatureRunRsIdentityError`, `FeatureRunGroupEnrichmentService.enrich`, and one stored Group rank-map path for every Market.

- [ ] **Step 1: Write failing identity and enrichment regressions**

Add these behaviors:

```python
def test_feature_identity_infers_only_fully_legacy_metadata():
    run = SimpleNamespace(
        as_of_date=date(2026, 4, 10),
        config_json={"market": "US", "universe": {"market": "US"}},
    )
    resolution = resolve_feature_run_rs_identity(run, ranking_date=date(2026, 4, 10))
    assert resolution.identity.formula_version == LEGACY_RS_FORMULA_VERSION
    assert resolution.identity_source == "inferred_legacy"


def test_partial_canonical_feature_metadata_is_rejected():
    run = SimpleNamespace(
        as_of_date=date(2026, 4, 10),
        config_json={"market": "US", "market_rs_run_id": 7},
    )
    with pytest.raises(FeatureRunRsIdentityError, match="partial canonical"):
        resolve_feature_run_rs_identity(run, ranking_date=date(2026, 4, 10))
```

Update enrichment fixtures to persist explicit formula metadata. Seed both legacy and balanced same-date Group rows for US and assert only the configured formula wins. For HK, seed a stored `Internet Services` Group row whose rank differs from the rank serialized rows would produce and assert the stored rank is used. Add a missing-snapshot case that asserts a previous valid `ibd_group_rank` remains unchanged after `GroupSnapshotUnavailable`.
Put identity-only cases in `test_feature_run_rs_identity.py`, direct
session/batching/rollback cases in `test_feature_run_group_enrichment.py`, and
thin-wrapper/task behavior in `test_feature_store_tasks.py`.

- [ ] **Step 2: Run focused tests and verify cross-formula/non-US failures**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_feature_run_rs_identity.py \
  tests/unit/services/test_feature_run_group_enrichment.py -q
pytest tests/unit/test_feature_store_tasks.py \
  -k "enrich_feature_run_with_ibd_metadata" -q
```

Expected: new tests fail because US omits formula and non-US recomputes from serialized rows.

- [ ] **Step 3: Implement deterministic feature identity resolution**

Create `backend/app/services/feature_run_rs_identity.py`:

```python
from dataclasses import dataclass
from datetime import date

from app.domain.feature_store.run_metadata import feature_run_market
from app.domain.relative_strength import LEGACY_RS_FORMULA_VERSION, GroupSnapshotIdentity


class FeatureRunRsIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureRunRsIdentityResolution:
    identity: GroupSnapshotIdentity
    identity_source: str


def resolve_feature_run_rs_identity(
    feature_run,
    *,
    ranking_date: date,
) -> FeatureRunRsIdentityResolution:
    if feature_run is None:
        raise FeatureRunRsIdentityError("Feature run does not exist")
    config = dict(feature_run.config_json or {})
    market = feature_run_market(feature_run)
    formula = config.get("rs_formula_version")
    canonical_values = (
        config.get("market_rs_run_id"),
        config.get("rs_as_of_date"),
        config.get("rs_universe_size"),
    )
    if formula is not None and str(formula).strip():
        return FeatureRunRsIdentityResolution(
            identity=GroupSnapshotIdentity(market, ranking_date, str(formula)),
            identity_source="persisted",
        )
    if all(value is None for value in canonical_values):
        return FeatureRunRsIdentityResolution(
            identity=GroupSnapshotIdentity(
                market,
                ranking_date,
                LEGACY_RS_FORMULA_VERSION,
            ),
            identity_source="inferred_legacy",
        )
    raise FeatureRunRsIdentityError(
        "Feature run has partial canonical RS metadata without rs_formula_version"
    )
```

- [ ] **Step 4: Extract enrichment and replace Market-specific rank calculation with one reader**

Move the database/session, taxonomy, batching, details mutation, commit, and
rollback body of `_enrich_feature_run_with_ibd_metadata` into
`FeatureRunGroupEnrichmentService`. Its constructor requires keyword-only
`session_factory`, `taxonomy_service`, `snapshot_reader:
GroupRankSnapshotReader`, and `batch_size: int`. Its public method is
`enrich(*, feature_run_id: int, ranking_date: date) -> dict[str, int | str]`.

The task module retains `_enrich_feature_run_with_ibd_metadata` with its current
signature as a thin compatibility wrapper that builds the service and calls
`enrich`. Existing direct task tests therefore remain valid while the touched
task module falls below 1,000 lines.

Inside the extracted service:

- Resolve the feature run before iterating rows; a missing run raises
  `FeatureRunRsIdentityError` rather than being treated as legacy US data.
- Resolve its identity with `resolve_feature_run_rs_identity`.
- Instantiate/inject `GroupRankSnapshotReader`.
- Call `load_exact`; if it returns no rows, raise `GroupSnapshotUnavailable` before any feature row is modified.
- Build `ranks_by_group` directly from those already-loaded exact rows for US and non-US; do not issue a second snapshot query.
- Keep US `IBDIndustryGroup` taxonomy and non-US `MarketTaxonomyService` taxonomy.
- Delete the import/call to `compute_group_rankings_from_serialized_rows` and the serialized-row accumulator.
- Return `rs_formula_version` and `identity_source` in the enrichment statistics.

The exact pre-mutation guard is:

```python
identity_resolution = resolve_feature_run_rs_identity(
    feature_run,
    ranking_date=ranking_date,
)
identity = identity_resolution.identity
snapshot_rows = snapshot_reader.load_exact(
    db,
    identity=identity,
    include_top_symbol_names=False,
)
if not snapshot_rows:
    raise GroupSnapshotUnavailable(identity)
ranks_by_group = {
    str(row["industry_group"]): int(row["rank"])
    for row in snapshot_rows
}
```

- [ ] **Step 5: Run feature identity/enrichment tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_feature_run_rs_identity.py \
  tests/unit/services/test_feature_run_group_enrichment.py -q
pytest tests/unit/test_feature_store_tasks.py \
  -k "enrich_feature_run_with_ibd_metadata" -q
```

Expected: all US/non-US formula-isolation and no-overwrite tests pass.

- [ ] **Step 6: Commit stored-snapshot enrichment**

```bash
git add backend/app/services/feature_run_rs_identity.py backend/app/services/feature_run_group_enrichment.py backend/app/interfaces/tasks/feature_store_tasks.py backend/tests/unit/services/test_feature_run_rs_identity.py backend/tests/unit/services/test_feature_run_group_enrichment.py backend/tests/unit/test_feature_store_tasks.py
test ! -e issues.jsonl
git commit --no-verify -m "fix: enrich feature rows from exact Group snapshots"
```

---

### Task 5: Extract Exact Static Group Section Assembly

**Files:**
- Create: `backend/app/services/static_group_section_builder.py`
- Create: `backend/app/services/static_site_errors.py`
- Modify: `backend/app/services/static_site_export_service.py`
- Create: `backend/tests/unit/services/test_static_group_section_builder.py`
- Modify: `backend/tests/unit/test_static_site_export_service.py`

**Interfaces:**
- Consumes: Task 1 exact reader, existing `build_static_groups_payload`, feature-run metadata, and serialized rows for detail presentation only.
- Produces: `StaticGroupSectionBuilder.build` returning the existing static-site-v3 Groups payload.

- [ ] **Step 1: Write failing builder tests**

In the new module import `date`, `SimpleNamespace`, `Mock`, the balanced formula
constant, `GroupSnapshotIdentity`, the builder, and the static error; define
`AS_OF = date(2026, 4, 10)`. Cover these assertions:

```python
def test_builder_reads_exact_formula_history_and_never_serialized_group_rank(
    monkeypatch,
):
    current = [{
        "industry_group": "Software",
        "date": "2026-04-10",
        "rank": 1,
        "avg_rs_rating": 84.0,
        "avg_rs_rating_1m": 73.0,
        "avg_rs_rating_3m": 79.0,
        "median_rs_rating": 85.0,
        "weighted_avg_rs_rating": 86.0,
        "rs_std_dev": 4.0,
        "num_stocks": 3,
        "num_stocks_rs_above_80": 2,
        "pct_rs_above_80": 66.7,
        "top_symbol": "AAA",
        "top_symbol_name": "AAA Inc.",
        "top_rs_rating": 95.0,
        "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 44,
        "rank_change_1w": None,
        "rank_change_1m": None,
        "rank_change_3m": None,
        "rank_change_6m": None,
    }]
    reader = Mock()
    reader.load_exact.side_effect = [current, current]
    reader.available_dates.return_value = (AS_OF,)
    history_reader = Mock()
    history_reader.get_historical_ranks_batch.return_value = {}
    monkeypatch.setattr(
        "app.services.static_group_section_builder.group_snapshot_metadata",
        lambda *_args, **_kwargs: {
            "rs_as_of_date": AS_OF.isoformat(),
            "rs_universe_size": 500,
        },
    )
    feature_run = SimpleNamespace(
        id=12,
        status="published",
        as_of_date=AS_OF,
        config_json={
            "market": "US",
            "universe": {"market": "US"},
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 44,
            "rs_as_of_date": AS_OF.isoformat(),
            "rs_universe_size": 500,
        },
    )
    builder = StaticGroupSectionBuilder(
        snapshot_reader=reader,
        rank_history_reader=history_reader,
    )

    payload = builder.build(
        db=Mock(),
        generated_at="2026-04-10T23:00:00Z",
        identity=GroupSnapshotIdentity("US", AS_OF, BALANCED_RS_FORMULA_VERSION),
        latest_run=feature_run,
        serialized_rows=[{
            "symbol": "AAA",
            "ibd_industry_group": "Software",
            "rs_rating": 1,
            "market_rs_run_id": 44,
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
        }],
    )

    assert payload["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert payload["payload"]["rankings"]["rankings"][0]["rank"] == 1
    assert payload["payload"]["rankings"]["rankings"][0]["avg_rs_rating"] == 84.0
```

Add a mixed feature formula test that raises `StaticSiteSectionUnavailableError`, and a parity test comparing builder rows with the live exact-reader rows field-by-field for `rank`, overall RS, 1M RS, 3M RS, stock count, top symbol, formula, and run ID.

- [ ] **Step 2: Run builder tests and verify the class is absent**

```bash
cd backend && source venv/bin/activate && pytest tests/unit/services/test_static_group_section_builder.py -q
```

Expected: collection fails because `StaticGroupSectionBuilder` does not exist.

- [ ] **Step 3: Move Group assembly into the focused builder**

Move these exact methods from `StaticSiteExportService` into `StaticGroupSectionBuilder`, preserving their current payload behavior while replacing direct `IBDGroupRank` queries with `GroupRankSnapshotReader`:

```text
_build_groups_payload -> build
_validate_feature_run_group_source
_load_stored_group_history
_build_stored_group_details
_build_group_movers
```

Use this constructor and entry point:

```python
class StaticGroupSectionBuilder:
    def __init__(
        self,
        *,
        snapshot_reader: GroupRankSnapshotReader,
        rank_history_reader,
    ) -> None:
        self._snapshot_reader = snapshot_reader
        self._rank_history_reader = rank_history_reader

    def build(
        self,
        *,
        db: Session,
        generated_at: str,
        identity: GroupSnapshotIdentity,
        latest_run: FeatureRun,
        serialized_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rankings = self._snapshot_reader.load_exact(db, identity=identity)
        if not rankings:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"No exact stored Group snapshot is available for {identity.market} "
                    f"on {identity.as_of_date.isoformat()} with formula "
                    f"{identity.formula_version}."
                ),
            )
        try:
            metadata = group_snapshot_metadata(
                db,
                market=identity.market,
                rankings=rankings,
            )
        except RuntimeError as exc:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=str(exc),
            ) from exc
        self._validate_feature_run_group_source(
            latest_run=latest_run,
            identity=identity,
            market_rs_run_id=rankings[0].get("market_rs_run_id"),
            rs_universe_size=metadata["rs_universe_size"],
            serialized_rows=serialized_rows,
        )
        historical_ranks = self._rank_history_reader.get_historical_ranks_batch(
            db,
            [str(row["industry_group"]) for row in rankings],
            identity.as_of_date,
            GROUP_RANK_CHANGE_CALENDAR_DAYS,
            market=identity.market,
            formula_version=identity.formula_version,
        )
        for row in rankings:
            for period_name in GROUP_RANK_CHANGE_CALENDAR_DAYS:
                historical_rank = historical_ranks.get(
                    (str(row["industry_group"]), period_name)
                )
                row[f"rank_change_{period_name}"] = (
                    int(historical_rank) - int(row["rank"])
                    if historical_rank is not None
                    else None
                )
        historical = self._load_stored_group_history(db, identity=identity)
        details = self._build_stored_group_details(
            rankings=rankings,
            serialized_rows=serialized_rows,
            historical_rankings=historical,
        )
        return build_static_groups_payload(
            StaticGroupsSnapshot(
                date=identity.as_of_date.isoformat(),
                rankings=rankings,
                movers=self._build_group_movers(rankings),
                group_details=details,
                market=identity.market,
                rs_formula_version=identity.formula_version,
                market_rs_run_id=rankings[0].get("market_rs_run_id"),
                rs_as_of_date=metadata["rs_as_of_date"],
                rs_universe_size=metadata["rs_universe_size"],
            ),
            generated_at=generated_at,
            schema_version=STATIC_SITE_SCHEMA_VERSION,
        )
```

`_load_stored_group_history` must call `available_dates`, take the latest
`STATIC_GROUP_HISTORY_RUNS` dates in the existing descending/newest-first
payload order, and call `load_exact` with the same formula for every selected
date using `include_top_symbol_names=False` (history consumes only date/rank/RS
fields). The current snapshot keeps the default `True` so top-stock display
names remain unchanged. Serialized rows remain inputs only to
detail/constituent presentation.

Adapt `_validate_feature_run_group_source` to accept `identity`,
`market_rs_run_id`, and `rs_universe_size` while preserving every existing
published-status, Market, date, formula, run-ID, universe-size, and serialized
row check. Do not drop validation during the extraction.

Move `NoPublishedStaticMarketArtifact` and `StaticSiteSectionUnavailableError`
into `static_site_errors.py`; import and re-export them from
`static_site_export_service.py` so existing imports remain valid and the new
builder/combiner do not create a circular import.

- [ ] **Step 4: Delegate from the static exporter**

Add optional keyword-only `group_section_builder: StaticGroupSectionBuilder |
None = None` to `StaticSiteExportService.__init__`. Store the injected builder
for focused tests; otherwise construct one builder, injecting both
`GroupRankSnapshotReader` and the existing formula-aware Group rank history
query service. Replace the current `_build_groups_payload` lambda with:

```python
build=lambda: self._group_section_builder.build(
    db=db,
    generated_at=generated_at,
    identity=GroupSnapshotIdentity(market, latest_run.as_of_date, formula_version),
    latest_run=latest_run,
    serialized_rows=serialized_rows,
)
```

Delete the five moved methods and their now-unused imports from the exporter.

- [ ] **Step 5: Run static Group and parity tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_static_group_section_builder.py \
  tests/unit/test_static_site_export_service.py -k "group" \
  tests/parity/test_canonical_market_rs_parity.py::test_group_live_and_static_payloads_match_and_main_rank_uses_overall_only -q
```

Expected: all tests pass with the existing static v3 shape.

- [ ] **Step 6: Commit static Group extraction**

```bash
git add backend/app/services/static_group_section_builder.py backend/app/services/static_site_errors.py backend/app/services/static_site_export_service.py backend/tests/unit/services/test_static_group_section_builder.py backend/tests/unit/test_static_site_export_service.py
test ! -e issues.jsonl
git commit --no-verify -m "refactor: build static Groups from exact snapshots"
```

---

### Task 6: Fail-Closed Static Artifact Combination

**Files:**
- Create: `backend/app/services/static_artifact_combiner.py`
- Modify: `backend/app/services/static_site_export_service.py`
- Modify: `backend/app/scripts/export_static_site.py`
- Modify: `.github/workflows/static-site.yml`
- Create: `backend/tests/unit/services/test_static_artifact_combiner.py`
- Modify: `backend/tests/unit/test_static_site_export_service.py`
- Modify: `backend/tests/unit/test_export_static_site_script.py`
- Modify: `backend/tests/unit/test_static_site_workflow.py`

**Interfaces:**
- Consumes: existing per-Market metadata directories and static schema v3.
- Produces: `StaticArtifactCombineResult`, `StaticArtifactCombiner.combine`, and mandatory CLI/workflow formula propagation.

- [ ] **Step 1: Write failing current/fallback formula tests**

Create the test module with `json` and `Path` imports plus the static schema,
supported/default Markets, metadata filename, formula constants, combiner, and
shared static errors. Use this concrete artifact helper, then add the cases
below:

```python
def write_market_artifact(
    root: Path,
    *,
    market: str,
    formula: str,
) -> Path:
    market_dir = root / f"static-market-{market}" / "markets" / market.lower()
    (market_dir / "scan").mkdir(parents=True)
    (market_dir / "scan" / "manifest.json").write_text(
        '{"ok": true}\n',
        encoding="utf-8",
    )
    entry = {
        "market": market,
        "display_name": market,
        "as_of_date": "2026-04-10",
        "rs_formula_version": formula,
        "features": {
            "scan": True,
            "breadth": False,
            "groups": False,
            "charts": False,
        },
        "pages": {
            "scan": {"path": f"markets/{market.lower()}/scan/manifest.json"},
        },
        "assets": {},
    }
    (market_dir / STATIC_MARKET_METADATA_FILENAME).write_text(
        json.dumps({
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": "2026-04-10T22:00:00Z",
            "market": market,
            "entry": entry,
            "warnings": [],
        }),
        encoding="utf-8",
    )
    return root


def combiner() -> StaticArtifactCombiner:
    return StaticArtifactCombiner(
        schema_version=STATIC_SITE_SCHEMA_VERSION,
        supported_markets=STATIC_SUPPORTED_MARKETS,
        default_market=STATIC_DEFAULT_MARKET,
    )


def test_combiner_rejects_wrong_formula_current_without_using_fallback(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=LEGACY_RS_FORMULA_VERSION,
    )
    fallback = write_market_artifact(
        tmp_path / "fallback",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
    )
    with pytest.raises(StaticArtifactFormulaError, match="US current"):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=fallback,
            output_dir=tmp_path / "out",
            required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_combiner_rejects_wrong_formula_fallback(tmp_path):
    fallback = write_market_artifact(
        tmp_path / "fallback",
        market="HK",
        formula=LEGACY_RS_FORMULA_VERSION,
    )
    with pytest.raises(StaticArtifactFormulaError, match="HK fallback"):
        combiner().combine(
            artifacts_dir=tmp_path / "empty-current",
            fallback_artifacts_dir=fallback,
            output_dir=tmp_path / "out",
            required_formula_by_market={"HK": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_combiner_requires_every_market_named_by_formula_map(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
    )
    with pytest.raises(NoPublishedStaticMarketArtifact) as exc_info:
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={
                "US": BALANCED_RS_FORMULA_VERSION,
                "HK": BALANCED_RS_FORMULA_VERSION,
            },
            clean=True,
        )
    assert exc_info.value.markets == ("HK",)
```

Update the CLI spy to assert `rs_formula_version_overrides` is passed in combine mode. Update the workflow test to require `--rs-formula-version "$RS_FORMULA_VERSION"` in the combine job.
Add cases where `entry.features.groups` or `entry.features.rrg` is true but the
corresponding file is absent; both must fail before any output is copied.
Seed an existing output sentinel in the wrong-formula test and assert it still
exists after rejection; validation failure must not erase the last good
combined output.

- [ ] **Step 2: Run static combination tests and verify the guard is not wired**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_static_artifact_combiner.py \
  tests/unit/test_export_static_site_script.py \
  tests/unit/test_static_site_workflow.py -k "combine" -q
```

Expected: failures show the missing combiner class and missing formula argument in CLI/workflow combine mode.

- [ ] **Step 3: Extract and implement artifact selection**

Create `backend/app/services/static_artifact_combiner.py`. Move `_collect_market_artifacts`, combine-time copying, warning creation, and manifest construction out of `StaticSiteExportService`. Import `NoPublishedStaticMarketArtifact` from the neutral `static_site_errors.py` module introduced in Task 5. Use these fail-closed checks before copying a Market directory:

```python
@dataclass(frozen=True)
class StaticArtifactCombineResult:
    output_dir: Path
    generated_at: str
    as_of_date: str
    warnings: tuple[str, ...]
    manifest: dict[str, Any]


class StaticArtifactFormulaError(RuntimeError):
    pass


def _validate_formula(
    *,
    market: str,
    source_label: str,
    metadata: dict,
    market_dir: Path,
    expected_formula: str,
) -> dict:
    entry = metadata.get("entry")
    if not isinstance(entry, dict):
        raise RuntimeError(f"{market} {source_label} metadata has no Market entry")
    observed = {"market entry": entry.get("rs_formula_version")}
    features = entry.get("features") if isinstance(entry.get("features"), dict) else {}
    groups_path = market_dir / "groups.json"
    if features.get("groups") and not groups_path.is_file():
        raise StaticArtifactFormulaError(
            f"{market} {source_label} artifact advertises Groups but groups.json is absent"
        )
    if groups_path.is_file():
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        if groups.get("available", True):
            observed["Groups"] = groups.get("rs_formula_version")
    rrg_path = market_dir / "groups_rrg.json"
    if features.get("rrg") and not rrg_path.is_file():
        raise StaticArtifactFormulaError(
            f"{market} {source_label} artifact advertises RRG but groups_rrg.json is absent"
        )
    if rrg_path.is_file():
        rrg = json.loads(rrg_path.read_text(encoding="utf-8"))
        if rrg.get("available", True):
            observed["RRG"] = rrg.get("rs_formula_version")
    mismatches = {
        source: formula
        for source, formula in observed.items()
        if formula != expected_formula
    }
    if mismatches:
        rendered = ", ".join(
            f"{source}={formula!r}" for source, formula in sorted(mismatches.items())
        )
        raise StaticArtifactFormulaError(
            f"{market} {source_label} artifact uses incompatible RS formula: "
            f"{rendered}; expected {expected_formula!r}"
        )
    return entry
```

`StaticArtifactCombiner.combine` must:

1. Validate current artifacts first and raise immediately on a present incompatible current artifact.
2. Select fallback only for Markets with no current artifact.
3. Apply the same validation to fallback.
4. For keys in `required_formula_by_market` that remain absent, raise:

   ```python
   raise NoPublishedStaticMarketArtifact(
       "No published compatible static artifact is available for required "
       f"Markets: {', '.join(missing_markets)}.",
       markets=tuple(missing_markets),
   )
   ```
5. Preserve subset/warning behavior only when `required_formula_by_market` is empty, keeping direct legacy callers compatible.
6. Complete all discovery and validation in memory before cleaning or writing
   `output_dir`; an incompatible or missing required artifact leaves an
   existing output untouched.
7. Only after validation succeeds, perform the existing clean/copy behavior and
   write the combined manifest last.

- [ ] **Step 4: Delegate the static service facade**

Replace `StaticSiteExportService.combine_market_artifacts` with:

```python
@classmethod
def combine_market_artifacts(
    cls,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    fallback_artifacts_dir: Path | None = None,
    clean: bool = True,
    rs_formula_version_overrides: Mapping[str, str] | None = None,
) -> StaticSiteExportResult:
    combined = StaticArtifactCombiner(
        schema_version=STATIC_SITE_SCHEMA_VERSION,
        supported_markets=STATIC_SUPPORTED_MARKETS,
        default_market=STATIC_DEFAULT_MARKET,
    ).combine(
        artifacts_dir=Path(artifacts_dir),
        fallback_artifacts_dir=(
            Path(fallback_artifacts_dir)
            if fallback_artifacts_dir is not None
            else None
        ),
        output_dir=Path(output_dir),
        required_formula_by_market=rs_formula_version_overrides or {},
        clean=clean,
    )
    return StaticSiteExportResult(
        output_dir=combined.output_dir,
        generated_at=combined.generated_at,
        as_of_date=combined.as_of_date,
        warnings=combined.warnings,
        manifest=combined.manifest,
    )
```

The focused combiner must not import `static_site_export_service.py`; the
facade performs this result adaptation to avoid a circular dependency.

- [ ] **Step 5: Wire CLI and workflow formula propagation**

In combine mode, pass:

```python
rs_formula_version_overrides={
    market: args.rs_formula_version
    for market in STATIC_EXPORT_MARKETS
},
```

In `.github/workflows/static-site.yml`, give the combine step the same defaulted environment value as the Market jobs and add:

```yaml
env:
  RS_FORMULA_VERSION: ${{ github.event.inputs.rs_formula_version || 'balanced-horizon-percentile-v2' }}
run: |
  cd backend
  python -m app.scripts.export_static_site \
    --output-dir ../frontend/public/static-data \
    --combine-artifacts-dir /tmp/static-market-artifacts-current \
    --fallback-artifacts-dir /tmp/static-market-artifacts-fallback \
    --rs-formula-version "$RS_FORMULA_VERSION"
```

- [ ] **Step 6: Run all static combine/workflow tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_static_artifact_combiner.py \
  tests/unit/test_static_site_export_service.py \
  tests/unit/test_export_static_site_script.py \
  -k "combine or collect_market_artifacts" -q
pytest tests/unit/test_static_site_workflow.py -q
```

Expected: all tests pass, including compatible holiday fallback and incompatible current/fallback rejection.

- [ ] **Step 7: Commit static publication integrity**

```bash
git add backend/app/services/static_artifact_combiner.py backend/app/services/static_site_export_service.py backend/app/scripts/export_static_site.py .github/workflows/static-site.yml backend/tests/unit/services/test_static_artifact_combiner.py backend/tests/unit/test_static_site_export_service.py backend/tests/unit/test_export_static_site_script.py backend/tests/unit/test_static_site_workflow.py
test ! -e issues.jsonl
git commit --no-verify -m "fix: enforce RS formula during static combine"
```

---

### Task 7: Typed RRG Failures and Race-Free Rollout Activation

**Files:**
- Modify: `backend/app/services/static_groups_rrg_export.py`
- Modify: `backend/app/services/static_rrg_history_bundle.py`
- Modify: `backend/app/services/rrg_history_provider.py`
- Create: `backend/app/services/market_rs_rollout_models.py`
- Create: `backend/app/services/market_rs_backfill_service.py`
- Create: `backend/app/services/market_rs_activation_validator.py`
- Create: `backend/app/services/market_rs_activation_service.py`
- Modify: `backend/app/services/market_rs_rollout_service.py`
- Modify: `backend/app/scripts/backfill_market_rs.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/tests/unit/test_static_groups_rrg_sources.py`
- Modify: `backend/tests/unit/test_static_rrg_history_bundle.py`
- Modify: `backend/tests/unit/test_rrg_service.py`
- Modify: `backend/tests/unit/test_market_rs_rollout_service.py`
- Modify: `backend/tests/unit/test_backfill_market_rs_script.py`
- Modify: `backend/tests/integration/test_market_rs_activation.py`

**Interfaces:**
- Consumes: Task 2 coordinator, exact Group reader, staged static v3 directory, and existing active-formula repository.
- Produces: `StaticGroupsRRGUnavailableReason`, focused rollout collaborators, and activation that re-hashes staged content.

- [ ] **Step 1: Write failing reason-code and changed-manifest tests**

Add these concrete regressions, extending the existing `_service` helper in
`test_market_rs_rollout_service.py` to assemble the three focused
collaborators after the split:

```python
def test_rrg_unavailable_exposes_insufficient_history_reason_code(monkeypatch):
    rrg_service = Mock()
    rrg_service.get_rrg_scopes.return_value = {
        "groups": {"date": "2026-04-10", "groups": []},
        "sectors": {"date": "2026-04-10", "groups": []},
    }
    monkeypatch.setattr(
        StaticGroupsRRGPayloadBuilder,
        "_preflight_tables",
        lambda *_args, **_kwargs: None,
    )
    builder = StaticGroupsRRGPayloadBuilder(
        schema_version=STATIC_SITE_SCHEMA_VERSION,
        rrg_service=rrg_service,
        rs_formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    with pytest.raises(StaticGroupsRRGUnavailableError) as exc_info:
        builder.build(
            db=Mock(),
            generated_at="validation",
            expected_as_of_date=date(2026, 4, 10),
            market="US",
        )
    assert exc_info.value.reason_code is StaticGroupsRRGUnavailableReason.INSUFFICIENT_HISTORY


def test_activation_rejects_manifest_changed_after_validation(db_session, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version": "static-site-v3"}\n',
        encoding="utf-8",
    )
    validation = ActivationValidationReport(
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        through_date=date(2026, 4, 10),
        first_valid_date=date(2026, 4, 8),
        candidate_count=3,
        latest_market_rs_run_id=42,
        latest_universe_hash="universe-a",
        feature_run_id=71,
        feature_universe_hash="feature-a",
        static_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        errors=(),
    )
    manifest_path.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(MarketRsActivationRejected, match="changed after validation"):
        _service().activate(
            db_session,
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            feature_run_id=71,
            validation=validation,
            static_staging_dir=tmp_path,
        )
```

Add lower-layer history tests for typed `NOT_ENABLED`,
`CURRENT_SNAPSHOT_MISSING`, and `SOURCE_UNAVAILABLE` reasons. Update the
rollout validator test so an `INSUFFICIENT_HISTORY` exception yields
`rrg_status == "insufficient_balanced_history"`, while `FORMULA_MISMATCH` is
appended to errors without inspecting English text. In
`test_market_rs_activation.py`, add a real-validator regression that keeps the
manifest bytes unchanged but changes `groups.json` to the legacy formula after
the first validation; activation must reject the now-invalid staged directory.
Add live and static RRG history regressions proving that a same-formula date
with mixed Market RS run IDs raises a typed integrity/source failure, and that
the exact reader—not a raw range query—is called for every selected date.

- [ ] **Step 2: Run rollout/RRG tests and verify string matching and TOCTOU failure**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_static_groups_rrg_sources.py \
  tests/unit/test_static_rrg_history_bundle.py \
  tests/unit/test_rrg_service.py \
  tests/unit/test_market_rs_rollout_service.py \
  tests/unit/test_backfill_market_rs_script.py \
  tests/integration/test_market_rs_activation.py -q
```

Expected: new assertions fail because the exception has no reason code and activation does not receive/re-hash the staging directory.

- [ ] **Step 3: Add stable RRG reason codes**

In `static_groups_rrg_export.py` add:

```python
class StaticGroupsRRGUnavailableReason(StrEnum):
    NOT_ENABLED = "not_enabled"
    INSUFFICIENT_HISTORY = "insufficient_history"
    FORMULA_MISMATCH = "formula_mismatch"
    DATE_MISMATCH = "date_mismatch"
    SOURCE_UNAVAILABLE = "source_unavailable"


class StaticGroupsRRGUnavailableError(RuntimeError):
    def __init__(
        self,
        *,
        section: str,
        reason_code: StaticGroupsRRGUnavailableReason,
        reason: str,
    ) -> None:
        self.section = section
        self.reason_code = reason_code
        self.reason = reason
        super().__init__(reason)
```

Add a typed lower-layer reason in `static_rrg_history_bundle.py`:

```python
class StaticRRGHistoryUnavailableReason(StrEnum):
    NOT_ENABLED = "not_enabled"
    CURRENT_SNAPSHOT_MISSING = "current_snapshot_missing"
    SOURCE_UNAVAILABLE = "source_unavailable"


class StaticRRGHistoryUnavailableError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        reason_code: StaticRRGHistoryUnavailableReason,
    ) -> None:
        self.reason = reason
        self.reason_code = reason_code
        super().__init__(reason)
```

Add `unavailable_reason: StaticRRGHistoryUnavailableReason | None = None` to
`StaticRRGHistoryPreparation`. `prepare` returns `NOT_ENABLED` when its plan is
disabled and preserves any caught typed reason when it returns `state=None`.
Assign `NOT_ENABLED` at the disabled-Market build raise,
`SOURCE_UNAVAILABLE` at the missing-table raise, and
`CURRENT_SNAPSHOT_MISSING` when no current Group snapshot exists.

Update every raise site in `static_groups_rrg_export.py`: no configured scopes
uses `NOT_ENABLED`; empty/thin computed history uses `INSUFFICIENT_HISTORY`;
rolling-bundle formula mismatch uses `FORMULA_MISMATCH`; output date mismatch
uses `DATE_MISMATCH`; absent tables/schema/source uses `SOURCE_UNAVAILABLE`.
Map lower-layer reason codes with an explicit dictionary:

```python
_HISTORY_REASON_MAP = {
    StaticRRGHistoryUnavailableReason.NOT_ENABLED:
        StaticGroupsRRGUnavailableReason.NOT_ENABLED,
    StaticRRGHistoryUnavailableReason.CURRENT_SNAPSHOT_MISSING:
        StaticGroupsRRGUnavailableReason.INSUFFICIENT_HISTORY,
    StaticRRGHistoryUnavailableReason.SOURCE_UNAVAILABLE:
        StaticGroupsRRGUnavailableReason.SOURCE_UNAVAILABLE,
}
```

Both database and rolling history sources use this map; neither inspects
exception or warning text. A generic `StaticRRGHistoryBundleError` maps to
`SOURCE_UNAVAILABLE`, while the explicit state-formula comparison remains
`FORMULA_MISMATCH`.

Inject `GroupRankSnapshotReader` into `StaticRRGHistoryBundleService`. Replace
its range query over raw `IBDGroupRank` objects with:

```python
dates = tuple(
    item
    for item in self.snapshot_reader.available_dates(
        db,
        market=normalized_market,
        formula_version=normalized_formula,
        through_date=through_date,
    )
    if cutoff <= item <= through_date
)
snapshot_rows = [
    (
        snapshot_date,
        self.snapshot_reader.load_exact(
            db,
            identity=GroupSnapshotIdentity(
                normalized_market,
                snapshot_date,
                normalized_formula,
            ),
            include_top_symbol_names=False,
        ),
    )
    for snapshot_date in dates
]
```

Adapt `_weekly_snapshots` to consume these `(date, payload_rows)` pairs without
changing weekly selection or RRG math. Convert `GroupSnapshotIntegrityError`
to typed lower-layer `SOURCE_UNAVAILABLE`; do not continue with a mixed-run
date. Add a regression containing same-formula Group rows tied to different
Market RS runs and assert RRG history fails closed.

Update `test_static_rrg_history_bundle.py`'s minimal database fixture to create
`MarketRsRun` before `IBDGroupRank`. Whenever `_seed_weeks` creates balanced
rows, create one matching completed Market RS run per date with
`diagnostics_json={"price_basis": "adj_close_only"}` and attach that run ID to
every Group row for the date. Legacy fixtures retain a null run ID. This keeps
the existing formula-switch tests valid under the exact-reader invariant.
Apply the same valid balanced-run fixture update to
`test_rrg_service.py` wherever it builds static balanced history.

Remove the live `StoredGroupRankHistoryProvider` raw `IBDGroupRank` range
query too. Inject `GroupRankSnapshotReader`, resolve the active formula, find
the exact latest/current date, then load every cutoff-bounded date through
`load_exact(..., include_top_symbol_names=False)`. Build `latest_date`, current
metadata, and `_collect_group_series` inputs from those payload rows. Preserve
the public `build_rrg_history_provider` and `USGroupRankHistoryProvider`
constructors as compatibility factories, but make both construct/inject the
exact reader. Add live RRG mixed-run and cross-formula regressions in
`test_rrg_service.py`.

Replace unit tests that currently pass `_StubRankService` solely to synthesize
RRG history with a test-only `_StubRRGHistoryProvider` implementing
`get_all_groups_history`. Do not retain a production raw-query/stub bypass just
to satisfy those tests. Tests intended to cover stored history should seed the
database and use the exact production provider.

Expose `MarketRsActivationValidator.revalidate_static` as a focused wrapper
around the same validation logic, not a second implementation:

```python
def revalidate_static(
    self,
    db: Session,
    *,
    market: str,
    through_date: date,
    feature_run_id: int,
    static_staging_dir: Path,
) -> tuple[str, ...]:
    errors: list[str] = []
    latest_run = self._validate_run_and_groups(
        db,
        market=market,
        calculation_date=through_date,
        errors=errors,
    )
    if latest_run is not None:
        self._validate_static_artifacts(
            db,
            market=market,
            through_date=through_date,
            latest_run=latest_run,
            feature_run_id=feature_run_id,
            static_staging_dir=static_staging_dir,
            errors=errors,
        )
    return tuple(errors)
```

- [ ] **Step 4: Split rollout models, backfill, validation, and activation**

Move the existing dataclasses and rejection type unchanged into `market_rs_rollout_models.py`. Move these exact method groups without changing their calculation semantics:

```text
MarketRsBackfillService:
  _normalize_market
  _reason_code
  _earliest_available_price_date
  earliest_backfillable_date
  candidate_dates
  backfill

MarketRsActivationValidator:
  _json_file
  _static_manifest_hash
  _validate_run_and_groups
  _validate_static_artifacts
  revalidate_static
  validate_activation

MarketRsActivationService:
  _validate_feature_candidate
  activate
```

`MarketRsRolloutService` becomes a compatibility facade with explicit delegation:

```python
class MarketRsRolloutService:
    def __init__(self, *, backfill_service, activation_validator, activation_service) -> None:
        self._backfill_service = backfill_service
        self._activation_validator = activation_validator
        self._activation_service = activation_service

    def backfill(self, db, **kwargs):
        return self._backfill_service.backfill(db, **kwargs)

    def earliest_backfillable_date(self, db, **kwargs):
        return self._backfill_service.earliest_backfillable_date(db, **kwargs)

    def candidate_dates(self, db, **kwargs):
        return self._backfill_service.candidate_dates(db, **kwargs)

    def validate_activation(self, db, **kwargs):
        return self._activation_validator.validate_activation(db, **kwargs)

    def activate(self, db, **kwargs) -> None:
        self._activation_service.activate(db, **kwargs)
```

Construct the three collaborators in bootstrap. Inject the Task 2 Group coordinator into `MarketRsBackfillService` so rollout no longer calls `CanonicalGroupRankingService` directly. Preserve every public rollout method through an explicit facade delegate; do not rely on `__getattr__`.

Inject `GroupRankSnapshotReader` into `MarketRsActivationValidator`.
`_validate_run_and_groups` keeps its stock-row/rating and expected-constituent
checks, but obtains stored Group rows with `load_exact` for
`GroupSnapshotIdentity(market, calculation_date,
BALANCED_RS_FORMULA_VERSION)` and `include_top_symbol_names=False`. Convert
`GroupSnapshotIntegrityError` into a validation error and stop validating that
date's Group payload; remove its duplicate raw Group run-ID/contiguous-rank
checks.

- [ ] **Step 5: Remove error-message parsing in validation**

Replace the substring branch with:

```python
except StaticGroupsRRGUnavailableError as exc:
    if exc.reason_code is StaticGroupsRRGUnavailableReason.INSUFFICIENT_HISTORY:
        rrg_status = "insufficient_balanced_history"
        if rrg_path.is_file():
            payload = self._json_file(rrg_path)
            if payload.get("available") or payload.get("payload"):
                errors.append(
                    "Static RRG contains coordinates despite insufficient balanced history."
                )
    else:
        errors.append(
            f"Balanced RRG validation failed ({exc.reason_code.value}): {exc.reason}"
        )
```

- [ ] **Step 6: Re-hash and revalidate staged content inside activation**

Add required `static_staging_dir: Path` to the facade and activation service.
Inject the activation validator into the activation service through a narrow
`revalidate_static(...) -> tuple[str, ...]` protocol. Immediately before
opening the pointer update transaction:

1. Hash the manifest and compare it with the validation report.
2. Re-run all static Market/Groups/RRG formula, date, run-ID, universe-size,
   feature-run, and schema checks against `static_staging_dir`.
3. Hash the manifest again and require both pre/post hashes to equal the
   originally validated hash.
4. Reject on any revalidation error; never switch either pointer.

The hash checks are:

```python
manifest_path = Path(static_staging_dir) / "manifest.json"
if not manifest_path.is_file():
    raise MarketRsActivationRejected(
        ("Validated static manifest disappeared before activation.",)
    )
current_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if current_hash != validation.static_manifest_sha256:
    raise MarketRsActivationRejected(
        ("Validated static manifest changed after validation.",)
    )

static_errors = self._static_validator.revalidate_static(
    db,
    market=market,
    through_date=validation.through_date,
    feature_run_id=feature_run_id,
    static_staging_dir=Path(static_staging_dir),
)
post_validation_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if static_errors or post_validation_hash != current_hash:
    raise MarketRsActivationRejected(
        tuple(static_errors)
        or ("Validated static manifest changed during activation revalidation.",)
    )
```

Pass `static_staging_dir=staging_dir` from `backfill_market_rs.py` to `service.activate`.

- [ ] **Step 7: Run rollout, RRG, and script tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_static_groups_rrg_sources.py \
  tests/unit/test_static_rrg_history_bundle.py \
  tests/unit/test_rrg_service.py \
  tests/unit/test_market_rs_rollout_service.py \
  tests/unit/test_backfill_market_rs_script.py \
  tests/integration/test_market_rs_activation.py \
  tests/unit/test_static_site_export_service.py -q
```

Expected: all tests pass without exception-message substring checks.

- [ ] **Step 8: Commit rollout integrity and decomposition**

```bash
git add backend/app/services/static_groups_rrg_export.py backend/app/services/static_rrg_history_bundle.py backend/app/services/rrg_history_provider.py backend/app/services/market_rs_rollout_models.py backend/app/services/market_rs_backfill_service.py backend/app/services/market_rs_activation_validator.py backend/app/services/market_rs_activation_service.py backend/app/services/market_rs_rollout_service.py backend/app/scripts/backfill_market_rs.py backend/app/wiring/bootstrap.py backend/tests/unit/test_static_groups_rrg_sources.py backend/tests/unit/test_static_rrg_history_bundle.py backend/tests/unit/test_rrg_service.py backend/tests/unit/test_market_rs_rollout_service.py backend/tests/unit/test_backfill_market_rs_script.py backend/tests/integration/test_market_rs_activation.py
test ! -e issues.jsonl
git commit --no-verify -m "fix: make RS activation validation race-free"
```

---

### Task 8: Typed Scanner Hydration and Benchmark-Independent Canonical Filtering

**Files:**
- Modify: `backend/app/domain/scanning/ports.py`
- Modify: `backend/app/infra/providers/stock_data.py`
- Modify: `backend/app/use_cases/scanning/run_bulk_scan.py`
- Modify: `backend/app/use_cases/feature_store/build_daily_snapshot.py`
- Modify: `backend/app/scanners/scan_orchestrator.py`
- Modify: `backend/app/scanners/criteria/rs_resolution.py`
- Modify: `backend/app/scanners/custom_scanner.py`
- Modify: `backend/tests/unit/use_cases/conftest.py`
- Modify: `backend/tests/unit/infra/test_stock_data_provider.py`
- Modify: `backend/tests/unit/use_cases/test_run_bulk_scan.py`
- Modify: `backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py`
- Modify: `backend/tests/unit/test_scan_orchestrator.py`
- Modify: `backend/tests/unit/test_scan_orchestrator_quality_policy.py`
- Modify: `backend/tests/unit/domain/test_rating_calculator.py`
- Modify: `backend/tests/unit/test_custom_scanner.py`

**Interfaces:**
- Consumes: existing `MarketRsResolution` and `StockData` canonical fields.
- Produces: required `StockDataProvider.apply_market_rs_resolution` and typed canonical/legacy RS unavailability.

- [ ] **Step 1: Write failing provider and Custom Scanner regressions**

Import `pandas as pd` and `StockData` in the test module, then add the direct
reproduction:

```python
def make_stock_data(*, price_days: int = 252) -> StockData:
    index = pd.date_range("2025-04-01", periods=price_days, freq="B")
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=index,
    )
    return StockData(
        symbol="AAA",
        price_data=prices,
        benchmark_data=prices.copy(),
        fundamentals={},
    )


def test_canonical_rs_filter_does_not_require_benchmark_data():
    data = make_stock_data(price_days=252)
    data.benchmark_data = pd.DataFrame()
    data.rs_formula_version = BALANCED_RS_FORMULA_VERSION
    data.canonical_rs_ratings = {
        "rs_rating": 87,
        "rs_rating_1m": 72,
        "rs_rating_3m": 80,
        "rs_rating_12m": 91,
    }

    result = CustomScanner().scan_stock(
        "AAA",
        data,
        {"custom_filters": {"rs_rating_min": 80}, "min_score": 0},
    )

    assert result.passes is True
    assert result.details["filter_results"]["rs_rating"]["passes"] is True
    assert result.details["filter_results"]["rs_rating"]["rs_rating"] == 87.0


def test_legacy_rs_filter_with_empty_benchmark_is_insufficient_data():
    data = make_stock_data(price_days=252)
    data.benchmark_data = pd.DataFrame()
    data.rs_formula_version = LEGACY_RS_FORMULA_VERSION
    result = CustomScanner().scan_stock(
        "AAA",
        data,
        {"custom_filters": {"rs_rating_min": 80}, "min_score": 0},
    )
    assert result.passes is False
    assert result.rating == "Insufficient Data"
    assert "benchmark" in result.details["reason"].lower()
```

Add adapter delegation and bulk/snapshot/orchestrator tests that spy on `apply_market_rs_resolution` and assert one direct call. Add an ABC test proving a provider without the method cannot instantiate.

- [ ] **Step 2: Run focused tests and verify canonical 87 is discarded**

```bash
cd backend && source venv/bin/activate
pytest tests/unit/test_custom_scanner.py \
  -k "canonical_rs_filter or legacy_rs_filter" -q
pytest tests/unit/infra/test_stock_data_provider.py -k "market_rs_resolution" -q
pytest tests/unit/use_cases/test_run_bulk_scan.py -k "market_rs" -q
pytest tests/unit/use_cases/feature_store/test_build_daily_snapshot.py \
  -k "market_rs" -q
```

Expected: the canonical Custom Scanner case fails and the provider contract tests fail.

- [ ] **Step 3: Make hydration part of the provider port**

Move `MarketRsResolution` above `StockDataProvider` in `ports.py` and add:

```python
@abc.abstractmethod
def apply_market_rs_resolution(
    self,
    results: dict[str, object],
    resolution: MarketRsResolution,
) -> None:
    raise NotImplementedError
```

Keep `DataPrepStockDataProvider` delegation and add the method to `FakeStockDataProvider`:

```python
def apply_market_rs_resolution(self, results, resolution) -> None:
    for symbol, stock_data in results.items():
        normalized = str(symbol).strip().upper()
        stock_data.canonical_rs_ratings = resolution.ratings_by_symbol.get(normalized)
        stock_data.rs_formula_version = resolution.formula_version
        stock_data.market_rs_run_id = resolution.run_id
        stock_data.rs_universe_size = resolution.universe_size
```

Run `rg -n "class .*\\(StockDataProvider\\)" backend/tests` and add the same
explicit method to every concrete test provider, including
`LocalFakeProvider` in `test_rating_calculator.py` and `_FakeProvider` in
`test_scan_orchestrator_quality_policy.py`. The ABC-instantiation regression
must use a deliberately incomplete local class so production and ordinary
test fakes remain constructible.

- [ ] **Step 4: Delete dynamic hydration fallbacks**

In `RunBulkScanUseCase` and `BuildDailyFeatureSnapshotUseCase`, replace each `getattr`/`setattr` branch with:

```python
self._data_provider.apply_market_rs_resolution(market_data, resolution)
```

In `ScanOrchestrator`, remove `_attach_market_rs_resolution` and use:

```python
self._data_provider.apply_market_rs_resolution(
    {stock_data.symbol.strip().upper(): stock_data},
    resolution,
)
```

Run `rg -n "apply_market_rs_resolution|getattr.*apply_market_rs|canonical_rs_ratings"` and verify hydration assignments exist only in `DataPreparationLayer`, provider implementations, and explicit test fakes.

- [ ] **Step 5: Resolve Custom Scanner RS before benchmark gating**

Add typed legacy failure:

```python
class StockRsUnavailable(RuntimeError):
    pass


class CanonicalStockRsUnavailable(StockRsUnavailable):
    pass


class LegacyStockRsUnavailable(StockRsUnavailable):
    pass
```

Replace the current benchmark-wrapped Custom Scanner branch with:

```python
if filters.get("rs_rating_min") is not None:
    try:
        def legacy_rs():
            if benchmark_data.empty or "Close" not in benchmark_data.columns:
                raise LegacyStockRsUnavailable(
                    f"{symbol}: legacy RS requires benchmark Close history"
                )
            return self.rs_calc.calculate_rs_rating(
                symbol,
                price_data["Close"],
                benchmark_data["Close"],
                universe_performances=(
                    data.rs_universe_performances.get("weighted")
                    if data.rs_universe_performances
                    else None
                ),
            )

        rs_result = resolve_stock_rs(data, legacy_rs)
        filter_results.append(
            self._check_rs_rating(
                symbol,
                price_data,
                benchmark_data,
                filters,
                precomputed_rs_result=rs_result,
            )
        )
    except StockRsUnavailable as exc:
        return ScreenerResult(
            score=0.0,
            passes=False,
            rating="Insufficient Data",
            breakdown={},
            details={"reason": str(exc)},
            screener_name=self.screener_name,
        )
```

- [ ] **Step 6: Run scanner/provider/use-case tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_custom_scanner.py \
  tests/unit/infra/test_stock_data_provider.py \
  tests/unit/use_cases/test_run_bulk_scan.py \
  tests/unit/use_cases/feature_store/test_build_daily_snapshot.py \
  tests/unit/test_scan_orchestrator.py \
  tests/unit/test_scan_orchestrator_quality_policy.py \
  tests/unit/domain/test_rating_calculator.py -q
```

Expected: all tests pass and the canonical empty-benchmark reproduction passes.

- [ ] **Step 7: Commit the typed scanner boundary**

```bash
git add backend/app/domain/scanning/ports.py backend/app/infra/providers/stock_data.py backend/app/use_cases/scanning/run_bulk_scan.py backend/app/use_cases/feature_store/build_daily_snapshot.py backend/app/scanners/scan_orchestrator.py backend/app/scanners/criteria/rs_resolution.py backend/app/scanners/custom_scanner.py backend/tests/unit/use_cases/conftest.py backend/tests/unit/infra/test_stock_data_provider.py backend/tests/unit/use_cases/test_run_bulk_scan.py backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py backend/tests/unit/test_scan_orchestrator.py backend/tests/unit/test_scan_orchestrator_quality_policy.py backend/tests/unit/domain/test_rating_calculator.py backend/tests/unit/test_custom_scanner.py
test ! -e issues.jsonl
git commit --no-verify -m "fix: make canonical scanner RS hydration explicit"
```

---

### Task 9: Adjusted-Close-Only Runs and Explicit Incompatible Rebuild

**Files:**
- Create: `backend/app/domain/relative_strength/run_policy.py`
- Modify: `backend/app/domain/relative_strength/__init__.py`
- Modify: `backend/app/services/market_rs_inputs.py`
- Modify: `backend/app/infra/db/repositories/market_rs_repo.py`
- Modify: `backend/app/services/market_rs_snapshot_service.py`
- Modify: `backend/app/services/market_rs_reader.py`
- Modify: `backend/app/services/group_rank_snapshot_reader.py`
- Modify: `backend/app/services/market_rs_activation_validator.py`
- Modify: `backend/app/services/feature_run_rs_identity.py`
- Modify: `backend/app/tasks/market_rs_tasks.py`
- Modify: `backend/app/interfaces/tasks/feature_store_tasks.py`
- Modify: `backend/app/scripts/export_static_site.py`
- Modify: `backend/app/services/market_rs_backfill_service.py`
- Modify: `backend/tests/unit/test_market_rs_inputs.py`
- Modify: `backend/tests/unit/test_market_rs_snapshot_service.py`
- Modify: `backend/tests/unit/test_market_rs_reader.py`
- Modify: `backend/tests/unit/test_market_rs_tasks.py`
- Modify: `backend/tests/unit/test_export_static_site_script.py`
- Modify: `backend/tests/unit/test_feature_store_tasks.py`
- Modify: `backend/tests/unit/services/test_feature_run_rs_identity.py`
- Modify: `backend/tests/unit/test_canonical_group_ranking_service.py`
- Modify: `backend/tests/unit/test_groups_api_no_data.py`
- Modify: `backend/tests/unit/test_static_site_export_service.py`
- Modify: `backend/tests/unit/test_ui_snapshot_service.py`
- Modify: `backend/tests/unit/repositories/test_market_rs_repo.py`
- Modify: `backend/tests/unit/test_market_rs_rollout_service.py`
- Modify: `backend/tests/integration/test_market_rs_activation.py`
- Modify: `backend/tests/parity/test_canonical_market_rs_parity.py`

**Interfaces:**
- Consumes: existing unique Market/date/formula run and explicit build/backfill callers.
- Produces: `BALANCED_RS_PRICE_BASIS`, compatibility policy, `MarketRsSnapshotIncompatible`, and `rebuild_incompatible`.

- [ ] **Step 1: Write failing raw-close and incompatible-run tests**

First change the reader fixture helper to
`def _seed_balanced_run(db_session, *, diagnostics: dict | None = None)` and
store the explicit value when supplied, while defaulting ordinary successful
fixtures to the compatible marker:

```python
diagnostics_json=(
    diagnostics
    if diagnostics is not None
    else {"price_basis": BALANCED_RS_PRICE_BASIS}
)
```

Then add:

```python
def test_raw_close_cannot_replace_missing_adjusted_anchor(db_session):
    rows = _complete_rows("SPY", {offset: 100.0 for offset in ANCHORS})
    rows += _complete_rows("AAA", {offset: 100.0 for offset in ANCHORS})
    target = next(row for row in rows if row.symbol == "AAA" and row.date == ANCHORS[63])
    target.adj_close = None
    target.close = 500.0
    db_session.add_all(rows)
    db_session.commit()

    inputs = _loader(("AAA",)).load(db_session, market="US", as_of_date=ANCHORS[0])

    assert "AAA" not in inputs.excess_returns_by_symbol
    assert inputs.exclusions["AAA"] == "missing_adjusted_63_session_anchor"


def test_reader_rejects_completed_run_without_price_basis_marker(db_session):
    _seed_balanced_run(db_session, diagnostics={})
    with pytest.raises(CanonicalMarketRsUnavailable, match="price basis"):
        _reader(db_session).get(
            market="US",
            symbols=("AAA",),
            as_of_date=AS_OF,
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )


def test_explicit_rebuild_replaces_incompatible_completed_run(db_session):
    old = _seed_balanced_run(db_session, diagnostics={})
    old_id = old.id
    service = MarketRsSnapshotService(
        input_loader=_FakeInputLoader(_complete_inputs()),
        repository=MarketRsRunRepository(),
    )
    with pytest.raises(MarketRsSnapshotIncompatible):
        service.calculate(db_session, market="US", as_of_date=AS_OF)

    rebuilt = service.calculate(
        db_session,
        market="US",
        as_of_date=AS_OF,
        rebuild_incompatible=True,
    )
    assert rebuilt.id != old_id
    assert rebuilt.diagnostics_json["price_basis"] == BALANCED_RS_PRICE_BASIS
```

Add a repository/snapshot regression that seeds balanced `IBDGroupRank` rows
pointing at `old.id`. A successful incompatible rebuild must remove those rows
before commit, after which the Group coordinator recreates them from the new
stock ratings. Add a calculation-failure regression proving transaction
rollback restores the old completed stock run and its old Group rows rather
than leaving a half-invalidated identity.

Add task/caller regressions before implementation: the Market RS task forwards
`rebuild_incompatible`; static preparation sets it true and passes the selected
formula into feature snapshot construction; ordinary daily calls leave it
false; and an `already_published` feature run with the old Market RS run ID is
not reused after replacement, while an exact matching provenance run is.

- [ ] **Step 2: Run Market RS input/snapshot/reader tests and verify failures**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_market_rs_inputs.py \
  tests/unit/test_market_rs_snapshot_service.py \
  tests/unit/test_market_rs_reader.py \
  tests/unit/test_market_rs_tasks.py \
  tests/unit/test_export_static_site_script.py \
  tests/unit/test_feature_store_tasks.py \
  tests/unit/services/test_feature_run_rs_identity.py \
  tests/unit/repositories/test_market_rs_repo.py -q
```

Expected: raw close satisfies the anchor today; incompatible runs are returned;
no rebuild API exists; and task/feature-run provenance tests show no explicit
rebuild or rotated-run guard.

- [ ] **Step 3: Add one reusable run policy**

Create `run_policy.py` and export its definitions:

```python
from collections.abc import Mapping


BALANCED_RS_PRICE_BASIS = "adj_close_only"


def balanced_run_has_required_price_basis(run) -> bool:
    diagnostics = getattr(run, "diagnostics_json", None)
    return (
        isinstance(diagnostics, Mapping)
        and diagnostics.get("price_basis") == BALANCED_RS_PRICE_BASIS
    )
```

- [ ] **Step 4: Remove raw-close fallback and stabilize reason codes**

In `MarketRsInputLoader` query only symbol/date/`StockPrice.adj_close`. Replace the price extraction with:

```python
for row in rows:
    if self._valid_price(row.adj_close):
        prices[(row.symbol, row.date)] = float(row.adj_close)
```

Use these exact reason codes:

```text
benchmark_adjusted_anchor_missing
current_adjusted_price_coverage_below_threshold
missing_adjusted_current_session_anchor
missing_adjusted_21_session_anchor
missing_adjusted_63_session_anchor
missing_adjusted_126_session_anchor
missing_adjusted_189_session_anchor
missing_adjusted_252_session_anchor
```

- [ ] **Step 5: Add explicit completed-run rebuild semantics**

Add `rebuild_completed: bool = False` to
`MarketRsRunRepository.start_or_restart`. Return a completed run only when
`rebuild_completed` is false. When true, replace it with a new run ID for the
same unique Market/date/formula identity and delete every `IBDGroupRank` row
whose `market_rs_run_id` references the old run. Rotating the ID is required:
feature-run metadata uses it as the provenance token, so an in-place rewrite
would make old stock/Group derivatives look current.

Perform the replacement under the existing row lock and caller transaction:
temporarily move the old row to a unique internal formula key no longer than 64
characters, flush; create/flush the new exact-identity running row so the
database allocates a distinct ID; then delete the old row (cascading its stock
rows) and its Group rows. No temporary formula is committed. A later
calculation failure and `rollback()` restores the previous completed stock run
and Group rows. Add repository tests for distinct IDs, one surviving exact
identity, no committed temporary key, and rollback restoration.

Add:

```python
class MarketRsSnapshotIncompatible(RuntimeError):
    pass
```

and change `MarketRsSnapshotService.calculate` to:

```python
def calculate(
    self,
    db: Session,
    *,
    market: str,
    as_of_date: date,
    formula_version: str = BALANCED_RS_FORMULA_VERSION,
    rebuild_incompatible: bool = False,
) -> MarketRsRun:
    if formula_version != BALANCED_RS_FORMULA_VERSION:
        raise ValueError(
            f"Unsupported Market RS formula for snapshot calculation: {formula_version}"
        )
    existing = self.repository.get_completed_exact(
        db,
        market=market,
        as_of_date=as_of_date,
        formula_version=formula_version,
    )
    if existing is not None and balanced_run_has_required_price_basis(existing):
        return existing
    if existing is not None and not rebuild_incompatible:
        raise MarketRsSnapshotIncompatible(
            f"Completed Market RS run {existing.id} has an incompatible price basis"
        )
    # Keep the existing MarketRsInputUnavailable try/except around this load.
    inputs = self.input_loader.load(db, market=market, as_of_date=as_of_date)
    run = self.repository.start_or_restart(
        db,
        market=market,
        as_of_date=as_of_date,
        formula_version=formula_version,
        benchmark_symbol=inputs.benchmark_symbol,
        benchmark_as_of_date=inputs.benchmark_as_of_date,
        universe_hash=inputs.universe_hash,
        expected_symbol_count=len(inputs.expected_symbols),
        rebuild_completed=rebuild_incompatible,
    )
```

Keep the existing calculation/failure transaction code after this preamble. Add
`"price_basis": BALANCED_RS_PRICE_BASIS` to successful completion diagnostics,
and assert that marker in the successful-run and explicit-rebuild tests.

Audit every test fixture that creates a successful balanced `MarketRsRun`:

```bash
rg -n "diagnostics_json=\\{\\}|mark_completed\\(" backend/tests
```

Add the marker when that fixture represents a valid canonical run. Keep `{}`
only in tests that intentionally exercise incompatible pre-policy data. This
includes Group reader/API, activation, static exporter, UI snapshot, parity,
repository, and task fixtures—not only `test_market_rs_reader.py`.

- [ ] **Step 6: Make every balanced read and activation validation fail closed**

In `SqlMarketRsReader` and `GroupRankSnapshotReader`, reject a completed balanced run when `balanced_run_has_required_price_basis(run)` is false. Do not mutate it. Use `CanonicalMarketRsUnavailable` for stock readers and `GroupSnapshotIntegrityError` for Group readers. In `MarketRsActivationValidator._validate_run_and_groups`, append a validation error for the same incompatibility so neither initial validation nor activation-time revalidation can approve an old run. Add a rollout test asserting `validation.ok is False` and the price-basis diagnostic is present.

- [ ] **Step 7: Restrict rebuild authority to explicit build paths**

Add `rebuild_incompatible: bool = False` to `calculate_market_rs_snapshot`; pass it to the snapshot service. Call the task with `rebuild_incompatible=True` from `_prepare_balanced_static_rs`. In `MarketRsBackfillService.backfill`, call the snapshot service for every candidate with `rebuild_incompatible=True` instead of skipping any completed run before compatibility is checked. Leave live read and ordinary daily task defaults false.

Before `build_daily_snapshot` returns an `already_published` feature run, resolve
the requested/active formula and exact completed Market RS run. Reuse the
feature run only when its persisted `rs_formula_version`, `market_rs_run_id`,
and `rs_as_of_date` match that exact run. A feature run pointing to the replaced
incompatible run ID must fall through to a fresh build. Pass
`rs_formula_version_override=rs_formula_version` from static refresh so the
check does not depend on an implicit default. Make the Step 1 live/static
provenance regressions pass.

Put the comparison in the focused identity module:

```python
def feature_run_matches_rs_source(
    feature_run,
    *,
    identity: GroupSnapshotIdentity,
    market_rs_run_id: int | None,
) -> bool:
    try:
        resolved = resolve_feature_run_rs_identity(
            feature_run,
            ranking_date=identity.as_of_date,
        )
    except FeatureRunRsIdentityError:
        return False
    if resolved.identity != identity:
        return False
    config = dict(feature_run.config_json or {})
    configured_as_of = config.get("rs_as_of_date")
    if (
        configured_as_of is not None
        and str(configured_as_of) != identity.as_of_date.isoformat()
    ):
        return False
    configured_run_id = config.get("market_rs_run_id")
    if market_rs_run_id is None:
        return configured_run_id is None
    try:
        return int(configured_run_id) == market_rs_run_id
    except (TypeError, ValueError):
        return False
```

The task resolves `formula` from `rs_formula_version_override` or
`MarketRsRunRepository.active_formula`. For balanced mode it obtains the exact
completed run and requires the adjusted-close marker before considering reuse;
for legacy, `market_rs_run_id=None`. Only call the helper after the existing
input/universe/date match is found.

Update the rollout backfill regression so `snapshot.calculate` returns one run
per candidate and every call includes `rebuild_incompatible=True`; a compatible
completed run is still returned unchanged by `MarketRsSnapshotService`, while
an incompatible one is rebuilt. Remove the old assertion that the snapshot
service is called only for dates absent from the repository.

- [ ] **Step 8: Run strict-input, rebuild, rollout, and parity tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_market_rs_inputs.py \
  tests/unit/test_market_rs_snapshot_service.py \
  tests/unit/test_market_rs_reader.py \
  tests/unit/test_market_rs_tasks.py \
  tests/unit/test_export_static_site_script.py \
  tests/unit/test_feature_store_tasks.py \
  tests/unit/services/test_feature_run_rs_identity.py \
  tests/unit/services/test_group_rank_snapshot_reader.py \
  tests/unit/test_canonical_group_ranking_service.py \
  tests/unit/test_groups_api_no_data.py \
  tests/unit/test_static_site_export_service.py \
  tests/unit/test_ui_snapshot_service.py \
  tests/unit/repositories/test_market_rs_repo.py \
  tests/unit/test_market_rs_rollout_service.py \
  tests/integration/test_market_rs_activation.py \
  tests/parity/test_canonical_market_rs_parity.py -q
```

Expected: all tests pass; parity fixtures now seed `price_basis: adj_close_only`.

- [ ] **Step 9: Commit strict Market RS inputs**

```bash
git add backend/app/domain/relative_strength backend/app/services/market_rs_inputs.py backend/app/infra/db/repositories/market_rs_repo.py backend/app/services/market_rs_snapshot_service.py backend/app/services/market_rs_reader.py backend/app/services/group_rank_snapshot_reader.py backend/app/services/market_rs_activation_validator.py backend/app/services/feature_run_rs_identity.py backend/app/tasks/market_rs_tasks.py backend/app/interfaces/tasks/feature_store_tasks.py backend/app/scripts/export_static_site.py backend/app/services/market_rs_backfill_service.py backend/tests/unit/test_market_rs_inputs.py backend/tests/unit/test_market_rs_snapshot_service.py backend/tests/unit/test_market_rs_reader.py backend/tests/unit/test_market_rs_tasks.py backend/tests/unit/test_export_static_site_script.py backend/tests/unit/test_feature_store_tasks.py backend/tests/unit/services/test_feature_run_rs_identity.py backend/tests/unit/services/test_group_rank_snapshot_reader.py backend/tests/unit/test_canonical_group_ranking_service.py backend/tests/unit/test_groups_api_no_data.py backend/tests/unit/test_static_site_export_service.py backend/tests/unit/test_ui_snapshot_service.py backend/tests/unit/repositories/test_market_rs_repo.py backend/tests/unit/test_market_rs_rollout_service.py backend/tests/integration/test_market_rs_activation.py backend/tests/parity/test_canonical_market_rs_parity.py
test ! -e issues.jsonl
git commit --no-verify -m "fix: require adjusted-close-only Market RS runs"
```

---

### Task 10: Complete Backend Structural Decomposition

**Files:**
- Create: `backend/app/services/group_rank_query_service.py`
- Create: `backend/app/services/legacy_group_rank_data.py`
- Create: `backend/app/services/legacy_group_rank_calculator.py`
- Create: `backend/app/services/legacy_group_rank_backfill.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Create: `backend/app/tasks/group_rank_workflows.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Create: `backend/app/services/static_chart_bundle_exporter.py`
- Modify: `backend/app/services/static_site_export_service.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`
- Modify: `backend/tests/unit/test_group_rank_tasks.py`
- Modify: `backend/tests/unit/test_static_site_export_service.py`
- Create: `backend/tests/unit/test_rs_module_boundaries.py`

**Interfaces:**
- Consumes: all behavior stabilized by Tasks 1–9.
- Produces: focused legacy/query/task/chart components and small compatibility facades.

- [ ] **Step 1: Add an architecture guard that fails on current giant files**

Create `backend/tests/unit/test_rs_module_boundaries.py`:

```python
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[2]
LIMITS = {
    "app/services/ibd_group_rank_service.py": 1000,
    "app/services/static_site_export_service.py": 1000,
    "app/services/market_rs_rollout_service.py": 1000,
    "app/tasks/group_rank_tasks.py": 1000,
    "app/interfaces/tasks/feature_store_tasks.py": 1000,
    "app/wiring/bootstrap.py": 1000,
}


@pytest.mark.parametrize("relative_path,limit", LIMITS.items())
def test_rs_touched_production_modules_stay_bounded(relative_path, limit):
    path = BACKEND_ROOT / relative_path
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count <= limit, f"{relative_path} has {line_count} lines; limit is {limit}"


def test_new_extracted_modules_stay_below_seven_hundred_lines():
    paths = [
        "app/services/group_rank_query_service.py",
        "app/services/legacy_group_rank_data.py",
        "app/services/legacy_group_rank_calculator.py",
        "app/services/legacy_group_rank_backfill.py",
        "app/tasks/group_rank_workflows.py",
        "app/services/static_chart_bundle_exporter.py",
        "app/services/feature_run_group_enrichment.py",
    ]
    for relative_path in paths:
        line_count = len(
            (BACKEND_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 700, f"{relative_path} has {line_count} lines"
```

- [ ] **Step 2: Run the guard and record the giant-file failures**

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_rs_module_boundaries.py -q
```

Expected: `ibd_group_rank_service.py`, `static_site_export_service.py`, and any still-oversized task/bootstrap file fail.

- [ ] **Step 3: Split the legacy Group service by cohesive method groups**

Move code with no formula or numerical changes into these owners:

```text
GroupRankQueryService
  get_current_rankings
  get_historical_ranks_batch
  _get_historical_ranks_batch
  get_group_history
  _get_constituent_stocks
  get_rank_movers
  formula/date resolution and historical annotation helpers

LegacyGroupRankData
  GroupRankPrefetchData
  _get_validated_group_symbols
  _get_market_caps_for_symbols
  _coerce_prefetch_data
  _symbols_by_group_for_run
  _prefetch_all_data
  _get_cached_only_benchmark_data
  _calculate_rs_by_symbol_for_dates
  series/return/vector helpers

LegacyGroupRankCalculator
  calculate_group_rankings
  _calculate_group_rs
  _store_rankings
  _ranking_values
  _store_rankings_sqlalchemy_fallback
  _calculate_group_rs_from_cache
  _calculate_group_metrics_from_rs

LegacyGroupRankBackfillService
  _delete_rankings_for_range
  backfill_rankings_optimized
  backfill_rankings
  find_missing_dates
  fill_gaps
  fill_gaps_optimized
  backfill_rankings_chunked
```

`LegacyGroupRankBackfillService.fill_gaps_optimized` gains required/defaulted `formula_version=LEGACY_RS_FORMULA_VERSION` and rejects any other formula before prefetch or storage.

`GroupRankQueryService` receives `GroupRankSnapshotReader` and the Market RS
repository. `get_current_rankings` resolves the requested/active formula,
uses a formula-scoped `max(IBDGroupRank.date)` query only when
`calculation_date` is omitted, constructs `GroupSnapshotIdentity(market,
resolved_date, formula)`, and loads its base rows through `load_exact` before applying the existing limit,
historical-rank annotations, and top-symbol names. It must not retain the old
raw current-row query. Historical batch queries remain formula-scoped. This
makes the live Groups response enforce the same run-integrity checks as feature,
static, and RRG readers.

Delete `_rank_record_payload`, `_annotate_top_symbol_names`, and
`_get_symbol_name_map` from the legacy facade; Task 1's shared payload helper
and exact reader now own that presentation behavior.

Update `test_group_rank_service.py`'s minimal schema/fixtures: create
`MarketRsRun` for tests that select balanced, attach the exact run ID to their
Group rows, and seed the adjusted-close price-basis marker. Tests exercising
legacy rows continue to use null run IDs. Add a live-query regression for mixed
balanced run IDs raising `GroupSnapshotIntegrityError`.

Keep `IBDGroupRankService` as an explicit facade. Its dispatch method is:

```python
def calculate_group_rankings(
    self,
    db,
    calculation_date=None,
    *,
    market=None,
    cache_only=False,
    cache_requirement=GroupRankCacheRequirement.disabled(),
    formula_version=None,
):
    normalized = (market or "US").upper()
    formula = formula_version or self.market_rs_repository.active_formula(
        db, market=normalized
    )
    if formula == BALANCED_RS_FORMULA_VERSION:
        return self.canonical_group_service.calculate_and_store(
            db,
            market=normalized,
            as_of_date=calculation_date or datetime.now().date(),
            formula_version=formula,
        )
    if formula != LEGACY_RS_FORMULA_VERSION:
        raise ValueError(f"Unsupported Group RS formula: {formula}")
    return self.legacy_calculator.calculate_group_rankings(
        db,
        calculation_date,
        market=normalized,
        cache_only=cache_only,
        cache_requirement=cache_requirement,
    )
```

Delegate each public query/backfill method explicitly:
`get_current_rankings`, `get_historical_ranks_batch`, `get_group_history`,
`get_rank_movers`, `backfill_rankings_optimized`, `backfill_rankings`,
`find_missing_dates`, `fill_gaps`, `fill_gaps_optimized`, and
`backfill_rankings_chunked`. Do not use `__getattr__`. Update tests that
intentionally exercise private vector/prefetch helpers to instantiate
`LegacyGroupRankData` or `LegacyGroupRankCalculator` directly; keep facade
contract tests for public behavior.

- [ ] **Step 4: Extract the undecorated Group task workflow**

Move the body of `calculate_daily_group_rankings` and its non-Celery calculation helpers into `GroupRankDailyWorkflow.run` in `group_rank_workflows.py`. Inject calendar, coordinator, legacy facade, activity callbacks, and cache invalidation. Keep validation, reason codes, and result dictionaries unchanged. The Celery task retains decorators/retry and calls:

```python
return build_group_rank_daily_workflow().run(
    task=self,
    calculation_date=calculation_date,
    force_cache_only=force_cache_only,
    market=market,
    activity_lifecycle=activity_lifecycle,
)
```

Move no Celery decorator or task name, preserving queue/API compatibility.

- [ ] **Step 5: Extract static chart export and serialization**

Move these exact methods into `StaticChartBundleExporter`:

```text
_export_chart_bundle
_collect_top_group_constituent_symbols
_get_cached_price_histories
_get_market_benchmark_history
_get_symbol_price_history
_static_chart_cutoff
_serialize_rs_line
_serialize_chart_bars
_chart_payload_path
```

Inject price cache, benchmark cache, and JSON writer. Replace the exporter call with:

```python
chart_manifest = self._chart_exporter.export(
    output_dir=output_dir,
    generated_at=generated_at,
    run=latest_run,
    rows=scan_rows,
    serialized_rows=serialized_rows,
    path_prefix=path_prefix,
    groups_payload=groups_payload,
    preset_screens=scan_manifest.get("preset_screens"),
)
```

- [ ] **Step 6: Run focused behavior tests after each mechanical move**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_static_site_export_service.py \
  tests/unit/test_rs_module_boundaries.py -q
```

Expected: all behavior and line-boundary tests pass.

- [ ] **Step 7: Check imports and line counts directly**

```bash
cd backend && python -m compileall -q app/services app/tasks app/wiring
wc -l app/services/ibd_group_rank_service.py app/services/static_site_export_service.py app/services/market_rs_rollout_service.py app/tasks/group_rank_tasks.py app/interfaces/tasks/feature_store_tasks.py app/wiring/bootstrap.py
```

Expected: compile succeeds and every listed count is at most 1,000.

- [ ] **Step 8: Commit backend decomposition**

```bash
git add backend/app/services/group_rank_query_service.py backend/app/services/legacy_group_rank_data.py backend/app/services/legacy_group_rank_calculator.py backend/app/services/legacy_group_rank_backfill.py backend/app/services/ibd_group_rank_service.py backend/app/tasks/group_rank_workflows.py backend/app/tasks/group_rank_tasks.py backend/app/services/static_chart_bundle_exporter.py backend/app/services/static_site_export_service.py backend/tests/unit/test_group_rank_service.py backend/tests/unit/test_group_rank_tasks.py backend/tests/unit/test_static_site_export_service.py backend/tests/unit/test_rs_module_boundaries.py
test ! -e issues.jsonl
git commit --no-verify -m "refactor: decompose RS orchestration boundaries"
```

---

### Task 11: Shared Live and Static Group RS Fields and UI Extraction

**Files:**
- Create: `frontend/src/features/groups/groupRankingFields.js`
- Create: `frontend/src/features/groups/groupRankingFields.test.js`
- Create: `frontend/src/features/groups/LiveGroupRankingsTable.jsx`
- Create: `frontend/src/features/groups/GroupDetailModal.jsx`
- Create: `frontend/src/static/components/StaticGroupRankingsTable.jsx`
- Modify: `frontend/src/pages/GroupRankingsPage.jsx`
- Modify: `frontend/src/pages/GroupRankingsPage.test.jsx`
- Modify: `frontend/src/static/pages/StaticGroupsPage.jsx`
- Modify: `frontend/src/static/pages/StaticGroupsPage.test.jsx`

**Interfaces:**
- Consumes: unchanged live/static payload fields.
- Produces: shared `GROUP_RS_FIELDS`, `formatGroupRs`, extracted live/static tables, and a smaller orchestration page.

- [ ] **Step 1: Write failing shared-field and surface tests**

Create the feature directory, then create the shared-field test:

```bash
mkdir -p frontend/src/features/groups
```

```javascript
import { describe, expect, it } from 'vitest';
import { GROUP_RS_FIELDS, formatGroupRs } from './groupRankingFields';

describe('groupRankingFields', () => {
  it('keeps live and static overall/1M/3M fields identical', () => {
    expect(GROUP_RS_FIELDS).toEqual([
      { field: 'avg_rs_rating', label: 'RS', staticLabel: 'Avg RS' },
      { field: 'avg_rs_rating_1m', label: '1M RS', staticLabel: '1M RS' },
      { field: 'avg_rs_rating_3m', label: '3M RS', staticLabel: '3M RS' },
    ]);
  });

  it('formats finite ratings and renders missing values safely', () => {
    expect(formatGroupRs(87.25)).toBe('87.3');
    expect(formatGroupRs(null)).toBe('-');
    expect(formatGroupRs(Number.NaN)).toBe('-');
  });
});
```

Keep/add page tests that render both modes and assert the labels `RS`/`Avg RS`, `1M RS`, and `3M RS`, plus values from all three fields. Add live sorting assertions for 1M and 3M.

- [ ] **Step 2: Run frontend tests and verify shared module is absent**

```bash
cd frontend && npm run test:run -- src/features/groups/groupRankingFields.test.js src/pages/GroupRankingsPage.test.jsx src/static/pages/StaticGroupsPage.test.jsx
```

Expected: shared module import fails.

- [ ] **Step 3: Implement shared field definitions**

Create `groupRankingFields.js`:

```javascript
export const GROUP_RS_FIELDS = Object.freeze([
  Object.freeze({ field: 'avg_rs_rating', label: 'RS', staticLabel: 'Avg RS' }),
  Object.freeze({ field: 'avg_rs_rating_1m', label: '1M RS', staticLabel: '1M RS' }),
  Object.freeze({ field: 'avg_rs_rating_3m', label: '3M RS', staticLabel: '3M RS' }),
]);

export const formatGroupRs = (value) => (
  Number.isFinite(value) ? value.toFixed(1) : '-'
);
```

- [ ] **Step 4: Extract the live table and detail modal**

Move `GroupDetailModal` unchanged into its own file with all chart/query imports. Move lines currently responsible for the live `<TableContainer>`/header/body into `LiveGroupRankingsTable` with this explicit function signature (the frontend does not depend on `prop-types`):

```javascript
export default function LiveGroupRankingsTable({
  rankings,
  order,
  orderBy,
  onSort,
  onSelectGroup,
  showHistoricalRanks,
}) {
  // Existing table rendering moves here unchanged except for shared RS fields.
}
```

Generate the first three RS header/cells from `GROUP_RS_FIELDS`, using `field.label`, `formatGroupRs(row[field.field])`, and the existing `onSort(field.field)` behavior. Keep rank, constituent, top-stock, and change columns unchanged.

- [ ] **Step 5: Extract the static table with the same field definitions**

Move `GroupsTableView` to `StaticGroupRankingsTable.jsx`. Generate its three RS columns from `GROUP_RS_FIELDS`, using `staticLabel` and `formatGroupRs`. Keep static rank changes and row selection unchanged.

Update both pages to import the extracted components. `GroupRankingsPage.jsx` retains data loading, view state, sorting state, RRG, calculation controls, and composition only.

- [ ] **Step 6: Run frontend tests, lint, and line count**

```bash
cd frontend && npm run test:run -- src/features/groups/groupRankingFields.test.js src/pages/GroupRankingsPage.test.jsx src/static/pages/StaticGroupsPage.test.jsx
npm run lint
wc -l src/pages/GroupRankingsPage.jsx
```

Expected: tests/lint pass and the page is below 1,000 lines.

- [ ] **Step 7: Commit frontend extraction**

```bash
git add frontend/src/features/groups frontend/src/static/components/StaticGroupRankingsTable.jsx frontend/src/pages/GroupRankingsPage.jsx frontend/src/pages/GroupRankingsPage.test.jsx frontend/src/static/pages/StaticGroupsPage.jsx frontend/src/static/pages/StaticGroupsPage.test.jsx
test ! -e issues.jsonl
git commit --no-verify -m "refactor: share Group RS table fields"
```

---

### Task 12: Cross-Surface Parity, Documentation, and Full Verification

**Files:**
- Modify: `backend/tests/parity/test_canonical_market_rs_parity.py`
- Modify: `README.md`
- Modify: `docs/LIVE_APP_GUIDE.md`
- Modify: `docs/STATIC_SITE.md`
- Modify: `docs/superpowers/specs/2026-07-18-canonical-market-rs-design.md`
- Modify: `.beads/issues.jsonl`
- Modify: `.beads/interactions.jsonl`

**Interfaces:**
- Consumes: all completed repair components.
- Produces: acceptance proof, corrected operator/user documentation, closed issue, and a pushed clean branch.

- [ ] **Step 1: Extend the parity fixture across the repaired readers**

Update `_seed_canonical_snapshot` to complete runs with:

```python
repository.mark_completed(
    run,
    excluded_symbol_count=0,
    diagnostics={"price_basis": BALANCED_RS_PRICE_BASIS},
)
```

Extend the Group parity test to read the stored rows through `GroupRankSnapshotReader` and assert:

```python
identity = GroupSnapshotIdentity("US", AS_OF, BALANCED_RS_FORMULA_VERSION)
stored = GroupRankSnapshotReader().load_exact(db_session, identity=identity)
assert [row["rank"] for row in stored] == [row["rank"] for row in static_rows]
for stored_row, static_row in zip(stored, static_rows, strict=True):
    for field in CANONICAL_GROUP_RS_FIELDS:
        if isinstance(stored_row[field], float):
            assert static_row[field] == pytest.approx(stored_row[field])
        else:
            assert static_row[field] == stored_row[field]
    assert stored_row["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert stored_row["market_rs_run_id"] == run.id
```

Also retain the stock subset/watchlist invariance, extreme-return bounding, and live/static RRG transform assertions.

- [ ] **Step 2: Run the complete parity suite**

```bash
cd backend && source venv/bin/activate && pytest tests/parity/test_canonical_market_rs_parity.py -q
```

Expected: all parity tests pass.

- [ ] **Step 3: Correct documentation for strict inputs and one Group source**

Make these statements explicit in all relevant documents:

```text
- Market RS requires the database adj_close value at all six session anchors.
- The calculator does not substitute raw close when adj_close is missing.
- Overall RS weights five per-horizon Market percentiles; raw 1,000% returns are bounded.
- Group overall/1M/3M values average the matching canonical stock ratings.
- Feature rows, live Groups, and static Groups read one stored Market/date/formula Group snapshot.
- Static current/fallback artifacts must match the requested formula.
- Rebuilding an incompatible completed Market RS snapshot rotates its run ID,
  invalidates derived Group rows, and prevents reuse of a feature run tied to
  the replaced provenance ID.
```

In the parent design, replace the sentence allowing calculation-time close fallback with the strict adjusted-close policy and link to the repair design.

- [ ] **Step 4: Run targeted backend regression suites**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/services/test_group_rank_snapshot_reader.py \
  tests/unit/services/test_group_rank_snapshot_coordinator.py \
  tests/unit/test_group_rank_history_backfill_service.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_feature_store_tasks.py \
  tests/unit/services/test_static_group_section_builder.py \
  tests/unit/services/test_static_artifact_combiner.py \
  tests/unit/test_static_site_export_service.py \
  tests/unit/test_export_static_site_script.py \
  tests/unit/test_static_site_workflow.py \
  tests/unit/test_static_groups_rrg_sources.py \
  tests/unit/test_market_rs_rollout_service.py \
  tests/unit/test_backfill_market_rs_script.py \
  tests/unit/test_custom_scanner.py \
  tests/unit/infra/test_stock_data_provider.py \
  tests/unit/use_cases/test_run_bulk_scan.py \
  tests/unit/use_cases/feature_store/test_build_daily_snapshot.py \
  tests/unit/test_market_rs_inputs.py \
  tests/unit/test_market_rs_snapshot_service.py \
  tests/unit/test_market_rs_reader.py \
  tests/unit/test_rs_module_boundaries.py \
  tests/parity/test_canonical_market_rs_parity.py -q
```

Expected: all targeted regressions pass.

- [ ] **Step 5: Run full backend and frontend quality gates**

```bash
cd backend && source venv/bin/activate && pytest
cd frontend && npm run test:run
cd frontend && npm run lint
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 6: Re-run static/build and architectural guards**

```bash
cd backend && python -m compileall -q app
wc -l app/services/ibd_group_rank_service.py app/services/static_site_export_service.py app/services/market_rs_rollout_service.py app/tasks/group_rank_tasks.py app/interfaces/tasks/feature_store_tasks.py app/wiring/bootstrap.py
cd ../frontend && wc -l src/pages/GroupRankingsPage.jsx
rg -n "compute_group_rankings_from_serialized_rows" ../backend/app/interfaces/tasks/feature_store_tasks.py ../backend/app/services/static_site_export_service.py
rg -n "getattr\([^\n]*apply_market_rs_resolution|setattr\([^\n]*canonical_rs_ratings" ../backend/app
```

Expected: compile succeeds; all Python files are at most 1,000 lines; the page is below 1,000; both `rg` commands return no matches.

- [ ] **Step 7: Commit parity and documentation**

```bash
git add backend/tests/parity/test_canonical_market_rs_parity.py README.md docs/LIVE_APP_GUIDE.md docs/STATIC_SITE.md docs/superpowers/specs/2026-07-18-canonical-market-rs-design.md
test ! -e issues.jsonl
git commit --no-verify -m "docs: finalize canonical RS repair contract"
```

- [ ] **Step 8: Close the issue and land the verified branch**

```bash
bd close stockscreenclaude-stm --reason "Resolved canonical RS review blockers with exact Group identity, strict adjusted inputs, typed scanner hydration, guarded static/rollout publication, parity tests, and bounded modules."
git add .beads/issues.jsonl .beads/interactions.jsonl
test ! -e issues.jsonl
git commit --no-verify -m "chore: close canonical RS repair issue"
git pull --rebase
bd vc status
test ! -e issues.jsonl
git push
git status --short --branch
```

Expected: the known Beads hook bug has not created an untracked repository-root
`issues.jsonl`; push succeeds; and status shows the branch synchronized with
origin with no working-tree changes. The installed Beads version exposes
`bd vc status` rather than the obsolete `bd sync` command.
