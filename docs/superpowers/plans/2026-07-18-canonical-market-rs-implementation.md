# Canonical Market RS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace divergent raw-return RS calculations with one versioned, Market-wide, per-horizon-percentile RS snapshot used by Scan, live Groups, static Groups, and RRG, while adding Group 1M RS and 3M RS columns.

**Architecture:** A pure domain calculator produces bounded 1–99 component and overall ratings from one point-in-time Market universe. A transactional snapshot service persists stock ratings once per Market/date/formula, Group rankings aggregate those rows, and every reader resolves the same active formula pointer. The balanced formula is backfilled beside legacy history and activated per Market only after coverage, parity, and RRG validation succeed.

**Tech Stack:** Python 3, SQLAlchemy, Alembic, PostgreSQL, pandas, Celery, FastAPI/Pydantic, React 18, TanStack Query, MUI, pytest, Vitest.

## Global Constraints

- Formula version is exactly `balanced-horizon-percentile-v2`; legacy rows are labeled `legacy-linear-v1`.
- Horizons and weights are exactly 1M/21 sessions at 20%, 3M/63 at 30%, 6M/126 at 20%, 9M/189 at 15%, and 12M/252 at 15%.
- Rank each horizon independently across all active, eligible stocks in the same Market, then rank the weighted component composite again.
- Stock component and overall ratings are integers from 1 through 99; Group ratings are decimal arithmetic means.
- Ties use average ascending rank and half-up mapping; an all-tied distribution maps to 50.
- All component ratings use the same 12M-eligible stock set and split-adjusted prices at exact Market-session anchors.
- Group RS, Group 1M RS, and Group 3M RS use the same eligible constituent set; main Group Rank uses only unrounded Group RS.
- Live and static Scan/Group/RRG consumers must never create their own percentile universe after balanced-formula activation.
- RRG transformation mathematics remains unchanged; formula versions may not mix in one history series or rolling bundle.
- Existing legacy rows and unrelated working-tree changes must remain recoverable throughout rollout.
- Before Task 1, run `bd ready`, claim/create the implementation issue with `bd update <id> --status in_progress`, and keep its status current; if the repository's `bd` binary is unavailable, record that constraint without modifying unrelated `.beads` state.
- Use red-green-refactor TDD and make one focused commit at the end of every task.

---

## File Structure

New bounded-context files:

- `backend/app/domain/relative_strength/__init__.py`: public formula constants and value objects.
- `backend/app/domain/relative_strength/calculator.py`: pure percentile and balanced-composite calculation.
- `backend/app/infra/db/models/relative_strength.py`: Market run, stock snapshot, and active-formula ORM models.
- `backend/app/infra/db/repositories/market_rs_repo.py`: transactional run lifecycle, lookup, and activation operations.
- `backend/app/services/point_in_time_universe_service.py`: active-symbol reconstruction for historical dates.
- `backend/app/services/market_rs_inputs.py`: session-anchor, adjusted-price, benchmark, and coverage loading.
- `backend/app/services/market_rs_snapshot_service.py`: canonical snapshot orchestration and atomic publication.
- `backend/app/services/market_rs_reader.py`: read-only adapter used by scan and feature-snapshot paths.
- `backend/app/services/canonical_group_ranking_service.py`: Group aggregation from canonical stock rows.
- `backend/app/services/market_rs_rollout_service.py`: backfill validation and atomic formula activation.
- `backend/app/tasks/market_rs_tasks.py`: daily and historical Celery entry points.
- `backend/app/scripts/backfill_market_rs.py`: resumable operator CLI for shadow history and activation.
- `backend/alembic/versions/20260718_0025_add_canonical_market_rs.py`: versioned persistence and Group schema migration.

Primary existing files modified:

- `backend/app/models/industry.py`, `backend/app/models/__init__.py`: versioned Group fields and model registration.
- `backend/app/services/market_calendar_service.py`: exact session-anchor resolution.
- `backend/app/scanners/base_screener.py`, `backend/app/scanners/scan_orchestrator.py`, and RS-using screeners: canonical rating injection and guarded legacy fallback.
- `backend/app/use_cases/feature_store/build_daily_snapshot.py`, `backend/app/use_cases/scanning/run_bulk_scan.py`, `backend/app/wiring/bootstrap.py`: canonical reader wiring.
- `backend/app/services/ibd_group_rank_service.py`, `backend/app/services/market_group_ranking_service.py`, `backend/app/infra/db/repositories/scan_result_repo.py`: version-aware Group reads and enrichment.
- `backend/app/schemas/groups.py`, `backend/app/api/v1/groups.py`, `backend/app/services/ui_snapshot_service.py`: live payload fields and metadata.
- `backend/app/services/rrg_history_provider.py`, `backend/app/services/static_rrg_history_contract.py`, `backend/app/services/static_rrg_history_bundle.py`: formula-isolated RRG history.
- `backend/app/services/static_site_export_service.py`, `backend/app/services/static_groups_payload_builder.py`: exact stored Group snapshot export and static schema v3.
- `backend/app/tasks/daily_market_pipeline_tasks.py`, `backend/app/tasks/group_rank_tasks.py`: ordered daily and manual execution.
- `frontend/src/pages/GroupRankingsPage.jsx`, `frontend/src/static/pages/StaticGroupsPage.jsx`: 1M/3M RS columns and live date rollover.
- `README.md`, `docs/LIVE_APP_GUIDE.md`, `docs/STATIC_SITE.md`: formula, fields, parity, and rollout documentation.

---

### Task 1: Pure Balanced-Horizon Percentile Calculator

**Files:**
- Create: `backend/app/domain/relative_strength/__init__.py`
- Create: `backend/app/domain/relative_strength/calculator.py`
- Create: `backend/tests/unit/domain/test_relative_strength_calculator.py`

**Interfaces:**
- Consumes: `Mapping[str, Mapping[str, float]]` keyed by symbol and horizon.
- Produces: `calculate_balanced_rs(excess_returns_by_symbol) -> dict[str, StockRsScore]` and formula constants used by every later task.

- [ ] **Step 1: Write failing tests for percentile mapping, ties, weighting, and raw-return bounding**

```python
from dataclasses import replace

import pytest

from app.domain.relative_strength.calculator import (
    BALANCED_RS_FORMULA_VERSION,
    HORIZON_WEIGHTS,
    StockRsScore,
    calculate_balanced_rs,
    percentile_ratings,
)


def test_percentile_ratings_use_average_ties_and_half_up_mapping():
    assert percentile_ratings({"A": 0.0, "B": 10.0, "C": 10.0, "D": 20.0}) == {
        "A": 1,
        "B": 50,
        "C": 50,
        "D": 99,
    }


def test_percentile_ratings_map_an_all_tied_distribution_to_50():
    assert percentile_ratings({"A": 7.0, "B": 7.0, "C": 7.0}) == {
        "A": 50,
        "B": 50,
        "C": 50,
    }


def test_balanced_weights_are_exactly_one():
    assert sum(HORIZON_WEIGHTS.values()) == pytest.approx(1.0)
    assert HORIZON_WEIGHTS == {
        "1m": 0.20,
        "3m": 0.30,
        "6m": 0.20,
        "9m": 0.15,
        "12m": 0.15,
    }


def test_raw_magnitude_cannot_change_a_symbol_after_horizon_rank_is_fixed():
    returns = {
        "A": {"1m": -0.40, "3m": -0.20, "6m": 0.10, "9m": 0.40, "12m": 10.0},
        "B": {"1m": 0.10, "3m": 0.15, "6m": 0.20, "9m": 0.30, "12m": 1.0},
        "C": {"1m": 0.20, "3m": 0.25, "6m": 0.30, "9m": 0.20, "12m": 0.5},
    }
    baseline = calculate_balanced_rs(returns)
    extreme = calculate_balanced_rs(
        {**returns, "A": {**returns["A"], "12m": 10_000.0}}
    )

    assert baseline["A"].rs_12m == 99
    assert extreme["A"].rs_12m == 99
    assert extreme["A"].weighted_composite == baseline["A"].weighted_composite
    assert extreme["A"].overall_rs == baseline["A"].overall_rs


def test_recent_relative_weakness_controls_half_of_the_composite():
    scores = calculate_balanced_rs(
        {
            "FORMER": {"1m": -0.50, "3m": -0.35, "6m": 0.10, "9m": 1.0, "12m": 10.0},
            "STEADY": {"1m": 0.12, "3m": 0.18, "6m": 0.20, "9m": 0.25, "12m": 0.30},
            "MIDDLE": {"1m": 0.02, "3m": 0.04, "6m": 0.08, "9m": 0.12, "12m": 0.15},
        }
    )

    assert scores["FORMER"].rs_1m == 1
    assert scores["FORMER"].rs_3m == 1
    assert scores["FORMER"].weighted_composite < scores["STEADY"].weighted_composite


def test_calculator_requires_one_common_complete_eligible_set():
    with pytest.raises(ValueError, match="missing horizons"):
        calculate_balanced_rs(
            {
                "A": {"1m": 0.1, "3m": 0.2, "6m": 0.3, "9m": 0.4, "12m": 0.5},
                "B": {"1m": 0.1, "3m": 0.2},
            }
        )


def test_formula_version_is_stable():
    assert BALANCED_RS_FORMULA_VERSION == "balanced-horizon-percentile-v2"
```

- [ ] **Step 2: Run the calculator tests and verify the import failure**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/domain/test_relative_strength_calculator.py -q
```

Expected: fail with `ModuleNotFoundError: app.domain.relative_strength`.

- [ ] **Step 3: Implement the pure calculator**

Create `backend/app/domain/relative_strength/calculator.py` with these public definitions and no database/pandas imports:

```python
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

BALANCED_RS_FORMULA_VERSION = "balanced-horizon-percentile-v2"
LEGACY_RS_FORMULA_VERSION = "legacy-linear-v1"
HORIZON_SESSIONS = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252}
HORIZON_WEIGHTS = {"1m": 0.20, "3m": 0.30, "6m": 0.20, "9m": 0.15, "12m": 0.15}
HORIZONS = tuple(HORIZON_SESSIONS)


@dataclass(frozen=True)
class StockRsScore:
    symbol: str
    overall_rs: int
    rs_1m: int
    rs_3m: int
    rs_6m: int
    rs_9m: int
    rs_12m: int
    weighted_composite: float
    excess_return_1m: float
    excess_return_3m: float
    excess_return_6m: float
    excess_return_9m: float
    excess_return_12m: float

    def as_scanner_fields(self) -> dict[str, int]:
        return {
            "rs_rating": self.overall_rs,
            "rs_rating_1m": self.rs_1m,
            "rs_rating_3m": self.rs_3m,
            "rs_rating_12m": self.rs_12m,
        }


def percentile_ratings(values: Mapping[str, float]) -> dict[str, int]:
    if len(values) < 2:
        raise ValueError("percentile ratings require at least two observations")
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("percentile inputs must be finite")

    ordered = sorted((float(value), symbol) for symbol, value in values.items())
    result: dict[str, int] = {}
    index = 0
    count = len(ordered)
    while index < count:
        end = index
        while end + 1 < count and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + (end + 1)) / 2.0
        rating = 1 + math.floor(98.0 * (average_rank - 1.0) / (count - 1) + 0.5)
        for position in range(index, end + 1):
            result[ordered[position][1]] = int(rating)
        index = end + 1
    return result


def calculate_balanced_rs(
    excess_returns_by_symbol: Mapping[str, Mapping[str, float]],
) -> dict[str, StockRsScore]:
    if len(excess_returns_by_symbol) < 2:
        raise ValueError("balanced RS requires at least two eligible stocks")
    expected = set(HORIZONS)
    normalized: dict[str, dict[str, float]] = {}
    for symbol, values in excess_returns_by_symbol.items():
        missing = expected - set(values)
        if missing:
            raise ValueError(f"{symbol} missing horizons: {', '.join(sorted(missing))}")
        normalized[symbol] = {horizon: float(values[horizon]) for horizon in HORIZONS}

    components = {
        horizon: percentile_ratings(
            {symbol: values[horizon] for symbol, values in normalized.items()}
        )
        for horizon in HORIZONS
    }
    composites = {
        symbol: sum(
            HORIZON_WEIGHTS[horizon] * components[horizon][symbol]
            for horizon in HORIZONS
        )
        for symbol in normalized
    }
    overall = percentile_ratings(composites)

    return {
        symbol: StockRsScore(
            symbol=symbol,
            overall_rs=overall[symbol],
            rs_1m=components["1m"][symbol],
            rs_3m=components["3m"][symbol],
            rs_6m=components["6m"][symbol],
            rs_9m=components["9m"][symbol],
            rs_12m=components["12m"][symbol],
            weighted_composite=composites[symbol],
            excess_return_1m=normalized[symbol]["1m"],
            excess_return_3m=normalized[symbol]["3m"],
            excess_return_6m=normalized[symbol]["6m"],
            excess_return_9m=normalized[symbol]["9m"],
            excess_return_12m=normalized[symbol]["12m"],
        )
        for symbol in normalized
    }
```

Export the constants, `StockRsScore`, `calculate_balanced_rs`, and `percentile_ratings` from `backend/app/domain/relative_strength/__init__.py`.

- [ ] **Step 4: Run the calculator tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/domain/test_relative_strength_calculator.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the domain calculator**

```bash
git add backend/app/domain/relative_strength backend/tests/unit/domain/test_relative_strength_calculator.py
git commit -m "feat: add balanced percentile RS calculator"
```

---

### Task 2: Versioned RS Persistence and Group Schema

**Files:**
- Create: `backend/app/infra/db/models/relative_strength.py`
- Create: `backend/alembic/versions/20260718_0025_add_canonical_market_rs.py`
- Modify: `backend/app/models/industry.py`
- Modify: `backend/app/models/__init__.py`
- Create: `backend/tests/unit/test_market_rs_schema.py`

**Interfaces:**
- Consumes: formula constants from Task 1.
- Produces: `MarketRsRun`, `StockRsSnapshot`, `MarketRsFormulaPointer`, and versioned `IBDGroupRank` columns used by repositories and readers.

- [ ] **Step 1: Write failing schema tests**

```python
from app.infra.db.models.relative_strength import (
    MarketRsFormulaPointer,
    MarketRsRun,
    StockRsSnapshot,
)
from app.models.industry import IBDGroupRank


def test_market_rs_models_expose_versioned_snapshot_contract():
    assert MarketRsRun.__table__.name == "market_rs_runs"
    assert StockRsSnapshot.__table__.primary_key.columns.keys() == ["run_id", "symbol"]
    assert MarketRsFormulaPointer.__table__.primary_key.columns.keys() == ["market"]
    assert {
        "avg_rs_rating_1m",
        "avg_rs_rating_3m",
        "rs_formula_version",
        "market_rs_run_id",
    }.issubset(IBDGroupRank.__table__.columns.keys())


def test_group_unique_constraint_includes_formula_version():
    names = {
        constraint.name
        for constraint in IBDGroupRank.__table__.constraints
        if constraint.name
    }
    assert "uix_ibd_group_rank_market_date_formula" in names
```

Add a migration-chain assertion that revision `20260718_0025` has `down_revision == "20260701_0024"`.

- [ ] **Step 2: Run the schema test and verify it fails**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_schema.py -q
```

Expected: fail because the relative-strength ORM module and Group columns do not exist.

- [ ] **Step 3: Add the ORM models**

Create `backend/app/infra/db/models/relative_strength.py` with:

```python
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class MarketRsRun(Base):
    __tablename__ = "market_rs_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(8), nullable=False)
    as_of_date = Column(Date, nullable=False)
    formula_version = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False)
    benchmark_symbol = Column(String(32), nullable=False)
    benchmark_as_of_date = Column(Date, nullable=False)
    universe_hash = Column(String(64), nullable=False)
    expected_symbol_count = Column(Integer, nullable=False)
    eligible_symbol_count = Column(Integer, nullable=False, default=0)
    excluded_symbol_count = Column(Integer, nullable=False, default=0)
    diagnostics_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    rows = relationship(
        "StockRsSnapshot",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("market", "as_of_date", "formula_version", name="uq_market_rs_run"),
        CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_market_rs_run_status"),
        Index("ix_market_rs_run_lookup", "market", "formula_version", "as_of_date", "status"),
    )


class StockRsSnapshot(Base):
    __tablename__ = "stock_rs_snapshots"

    run_id = Column(Integer, ForeignKey("market_rs_runs.id", ondelete="CASCADE"), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    overall_rs = Column(SmallInteger, nullable=False)
    rs_1m = Column(SmallInteger, nullable=False)
    rs_3m = Column(SmallInteger, nullable=False)
    rs_6m = Column(SmallInteger, nullable=False)
    rs_9m = Column(SmallInteger, nullable=False)
    rs_12m = Column(SmallInteger, nullable=False)
    weighted_composite = Column(Float, nullable=False)
    excess_return_1m = Column(Float, nullable=False)
    excess_return_3m = Column(Float, nullable=False)
    excess_return_6m = Column(Float, nullable=False)
    excess_return_9m = Column(Float, nullable=False)
    excess_return_12m = Column(Float, nullable=False)

    run = relationship("MarketRsRun", back_populates="rows")

    __table_args__ = (
        CheckConstraint(
            "overall_rs BETWEEN 1 AND 99 AND rs_1m BETWEEN 1 AND 99 "
            "AND rs_3m BETWEEN 1 AND 99 AND rs_6m BETWEEN 1 AND 99 "
            "AND rs_9m BETWEEN 1 AND 99 AND rs_12m BETWEEN 1 AND 99",
            name="ck_stock_rs_rating_range",
        ),
        Index("ix_stock_rs_symbol_run", "symbol", "run_id"),
    )


class MarketRsFormulaPointer(Base):
    __tablename__ = "market_rs_formula_pointers"

    market = Column(String(8), primary_key=True)
    formula_version = Column(String(64), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
```

Register these models in `backend/app/models/__init__.py` so Alembic and `Base.metadata.create_all()` see them.

- [ ] **Step 4: Add version fields to `IBDGroupRank` and create the migration**

Add these model columns:

```python
avg_rs_rating_1m = Column(Float)
avg_rs_rating_3m = Column(Float)
rs_formula_version = Column(String(64), nullable=False, default="legacy-linear-v1")
market_rs_run_id = Column(Integer, ForeignKey("market_rs_runs.id", ondelete="SET NULL"))
```

Replace `uix_ibd_group_rank_market_date` with:

```python
UniqueConstraint(
    "industry_group",
    "date",
    "market",
    "rs_formula_version",
    name="uix_ibd_group_rank_market_date_formula",
)
```

The migration must perform this ordered upgrade:

1. Create `market_rs_runs`, `stock_rs_snapshots`, and `market_rs_formula_pointers` with the model constraints and indexes.
2. Add the four Group columns as nullable.
3. Set every existing Group row to `legacy-linear-v1`.
4. Make `rs_formula_version` non-null.
5. Replace the old Group uniqueness constraint with the versioned constraint.
6. Seed one legacy pointer for every distinct Market found in `stock_universe` or `ibd_group_ranks` and for every catalog Market supported at this revision: `US`, `HK`, `IN`, `JP`, `KR`, `TW`, `CN`, `CA`, `DE`, `SG`, `AU`, and `MY`.

The downgrade must first delete non-legacy Group rows, restore the old uniqueness constraint, drop the four Group columns, and then drop the three new tables in reverse dependency order.

- [ ] **Step 5: Run migration/model tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_schema.py tests/unit/test_alembic_baseline.py -q
```

Expected: all tests pass and Alembic reports one head at `20260718_0025`.

- [ ] **Step 6: Commit the persistence schema**

```bash
git add backend/app/infra/db/models/relative_strength.py backend/app/models/industry.py backend/app/models/__init__.py backend/alembic/versions/20260718_0025_add_canonical_market_rs.py backend/tests/unit/test_market_rs_schema.py
git commit -m "feat: add versioned market RS persistence"
```

---

### Task 3: Point-in-Time Universe and Exact Session Anchors

**Files:**
- Create: `backend/app/services/point_in_time_universe_service.py`
- Modify: `backend/app/services/market_calendar_service.py`
- Create: `backend/tests/unit/test_point_in_time_universe_service.py`
- Modify: `backend/tests/unit/test_market_calendar_service.py`

**Interfaces:**
- Consumes: `StockUniverse`, `StockUniverseStatusEvent`, Market timezone/calendar facts.
- Produces: `PointInTimeUniverse(symbols, universe_hash)` and `MarketCalendarService.session_anchors(market, as_of_date, offsets=offsets)` for Task 4.

- [ ] **Step 1: Add failing session-anchor tests**

```python
def test_session_anchors_return_exact_market_session_offsets():
    sessions = pd.date_range("2025-01-01", periods=260, freq="B")
    service = MarketCalendarService(calendar_provider=lambda _name: FakeCalendar(sessions))
    as_of = sessions[-1].date()

    anchors = service.session_anchors("US", as_of, offsets=(21, 63, 126, 189, 252))

    assert anchors[0] == as_of
    assert anchors[21] == sessions[-22].date()
    assert anchors[252] == sessions[-253].date()
```

Also assert that a non-session `as_of_date` and fewer than 253 sessions raise descriptive `ValueError`s.

- [ ] **Step 2: Add failing point-in-time universe tests**

Seed one always-active symbol, one symbol deactivated before the requested date, and one symbol activated after it. Assert:

```python
snapshot = service.resolve(db_session, market="US", as_of_date=date(2026, 4, 10))
assert snapshot.symbols == ("ACTIVE",)
assert snapshot.universe_hash == hashlib.sha256(b"ACTIVE\n").hexdigest()
```

Add a historical case with a missing status event and assert `PointInTimeUniverseUnavailable` rather than falling back to the current active flag.

- [ ] **Step 3: Run the tests and verify missing interfaces**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_calendar_service.py tests/unit/test_point_in_time_universe_service.py -q
```

Expected: fail because `session_anchors` and the point-in-time service do not exist.

- [ ] **Step 4: Implement exact session anchors**

Add this public method to `MarketCalendarService`:

```python
def session_anchors(
    self,
    market: str,
    as_of_date: date,
    *,
    offsets: tuple[int, ...],
    mic: str | None = None,
) -> dict[int, date]:
    normalized = self.normalize_market(market)
    if not offsets or min(offsets) < 1:
        raise ValueError("session offsets must be positive")
    if not self.is_trading_day(normalized, as_of_date, mic=mic):
        raise ValueError(f"{as_of_date.isoformat()} is not a {normalized} trading session")
    maximum = max(offsets)
    start = as_of_date - timedelta(days=maximum * 2 + 30)
    sessions = self.trading_days(normalized, start, as_of_date, mic=mic)
    if len(sessions) <= maximum:
        raise ValueError(
            f"{normalized} calendar has {len(sessions)} sessions; {maximum + 1} required"
        )
    return {0: sessions[-1], **{offset: sessions[-1 - offset] for offset in offsets}}
```

- [ ] **Step 5: Implement historical universe reconstruction**

Create immutable `PointInTimeUniverse` and `PointInTimeUniverseUnavailable`. `resolve()` must:

1. Normalize the Market and compute the UTC cutoff for midnight after `as_of_date` in that Market's display timezone.
2. Use current `StockUniverse.active_filter()` only when `as_of_date` equals `last_completed_trading_day(market)`.
3. For older dates, load Market rows whose `first_seen_at` predates the cutoff and replay each symbol's latest `status_changed` event before the cutoff.
4. Raise if any historical candidate lacks a lifecycle event.
5. Return sorted active symbols and `sha256("".join(f"{symbol}\n" for symbol in symbols).encode("utf-8"))`.

Use this result type:

```python
@dataclass(frozen=True)
class PointInTimeUniverse:
    market: str
    as_of_date: date
    symbols: tuple[str, ...]
    universe_hash: str
```

- [ ] **Step 6: Run the focused tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_calendar_service.py tests/unit/test_point_in_time_universe_service.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit point-in-time input primitives**

```bash
git add backend/app/services/market_calendar_service.py backend/app/services/point_in_time_universe_service.py backend/tests/unit/test_market_calendar_service.py backend/tests/unit/test_point_in_time_universe_service.py
git commit -m "feat: resolve RS point-in-time inputs"
```

---

### Task 4: Atomic Canonical Market RS Snapshot Service

**Files:**
- Create: `backend/app/services/market_rs_inputs.py`
- Create: `backend/app/infra/db/repositories/market_rs_repo.py`
- Create: `backend/app/services/market_rs_snapshot_service.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Create: `backend/tests/unit/test_market_rs_inputs.py`
- Create: `backend/tests/unit/repositories/test_market_rs_repo.py`
- Create: `backend/tests/unit/test_market_rs_snapshot_service.py`
- Modify: `backend/tests/unit/test_runtime_services.py`

**Interfaces:**
- Consumes: `PointInTimeUniverseService.resolve`, `MarketCalendarService.session_anchors`, formula calculator, `StockPrice`, and benchmark registry candidates.
- Produces: `MarketRsSnapshotService.calculate(db, market, as_of_date, formula_version) -> MarketRsRun` and exact completed-run repository reads.

- [ ] **Step 1: Write failing input-loader tests**

Create fixtures with three active symbols and one benchmark at the six required dates. Use `adj_close` values that make the expected returns obvious. Assert:

```python
inputs = loader.load(db_session, market="US", as_of_date=date(2026, 4, 10))

assert inputs.benchmark_symbol == "SPY"
assert inputs.expected_symbols == ("AAA", "BBB", "YOUNG")
assert set(inputs.excess_returns_by_symbol) == {"AAA", "BBB"}
assert inputs.exclusions == {"YOUNG": "missing_252_session_anchor"}
assert inputs.current_price_coverage == pytest.approx(1.0)
assert inputs.excess_returns_by_symbol["AAA"]["1m"] == pytest.approx(
    (120.0 / 100.0 - 1.0) - (110.0 / 100.0 - 1.0)
)
```

Add cases proving `adj_close` is preferred over `close`, a complete fallback benchmark candidate is selected deterministically, stale/missing benchmark anchors fail the run, and current-price coverage below 90% raises `MarketRsInputUnavailable`.

- [ ] **Step 2: Write failing repository lifecycle tests**

```python
def test_completed_run_is_invisible_until_rows_and_status_commit(db_session):
    repo = MarketRsRunRepository()
    run = repo.start_or_restart(
        db_session,
        market="US",
        as_of_date=date(2026, 4, 10),
        formula_version=BALANCED_RS_FORMULA_VERSION,
        benchmark_symbol="SPY",
        benchmark_as_of_date=date(2026, 4, 10),
        universe_hash="a" * 64,
        expected_symbol_count=2,
    )
    db_session.commit()

    assert repo.get_completed_exact(
        db_session,
        market="US",
        as_of_date=date(2026, 4, 10),
        formula_version=BALANCED_RS_FORMULA_VERSION,
    ) is None


def test_failed_restart_clears_partial_rows_but_completed_run_is_idempotent(db_session):
    repo = MarketRsRunRepository()
    run = seed_failed_market_rs_run(db_session)
    restarted = repo.start_or_restart(
        db_session,
        market=run.market,
        as_of_date=run.as_of_date,
        formula_version=run.formula_version,
        benchmark_symbol=run.benchmark_symbol,
        benchmark_as_of_date=run.benchmark_as_of_date,
        universe_hash=run.universe_hash,
        expected_symbol_count=2,
    )
    assert restarted.id == run.id
    assert restarted.status == "running"
    assert restarted.rows == []
```

Also test that `activate_formula()` changes only the requested Market pointer, rejects balanced mode with no completed run, allows `legacy-linear-v1` for rollback without a canonical run, and rejects unknown formula strings. Assert `active_formula()` rejects an unconfigured Market.

Add a two-session contention test for the same `(market, as_of_date, formula_version)`: only one run row may exist, a completed winner may not be reset by the loser, and readers must never observe a partially written row set.

- [ ] **Step 3: Write failing service tests for publication and failure isolation**

```python
def test_snapshot_service_publishes_all_rows_and_run_atomically(db_session):
    service = MarketRsSnapshotService(
        input_loader=FakeInputLoader(complete_market_inputs()),
        repository=MarketRsRunRepository(),
    )

    run = service.calculate(
        db_session,
        market="US",
        as_of_date=date(2026, 4, 10),
    )

    assert run.status == "completed"
    assert run.eligible_symbol_count == 3
    assert len(run.rows) == 3
    assert all(1 <= row.overall_rs <= 99 for row in run.rows)


def test_snapshot_failure_keeps_previous_completed_date_readable(db_session):
    previous = seed_completed_market_rs_run(db_session, as_of_date=date(2026, 4, 9))
    service = MarketRsSnapshotService(
        input_loader=FakeInputLoader(error=MarketRsInputUnavailable("benchmark missing")),
        repository=MarketRsRunRepository(),
    )

    with pytest.raises(MarketRsInputUnavailable):
        service.calculate(db_session, market="US", as_of_date=date(2026, 4, 10))

    assert service.repository.get_latest_completed(
        db_session,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
).id == previous.id
```

Also assert the service rejects `legacy-linear-v1` before loading inputs or writing a canonical run.

Extend `test_runtime_services.py` with a failing singleton/reset assertion for `get_market_rs_snapshot_service()` and its point-in-time/input/repository dependencies.

- [ ] **Step 4: Run all new service tests and verify failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_inputs.py tests/unit/repositories/test_market_rs_repo.py tests/unit/test_market_rs_snapshot_service.py -q
```

Expected: fail because the loader, repository, and service are not defined.

- [ ] **Step 5: Implement the adjusted-price input loader**

Define:

```python
@dataclass(frozen=True)
class MarketRsInputs:
    market: str
    as_of_date: date
    benchmark_symbol: str
    benchmark_as_of_date: date
    universe_hash: str
    expected_symbols: tuple[str, ...]
    excess_returns_by_symbol: dict[str, dict[str, float]]
    exclusions: dict[str, str]
    current_price_coverage: float
```

Define `MarketRsInputUnavailable` with required `reason_code` and `diagnostics`, plus context fields `benchmark_symbol`, `universe_hash`, and `expected_symbol_count`. Default unavailable context is the registry's primary benchmark, `sha256(b"").hexdigest()`, and zero expected symbols; the loader replaces those defaults as soon as universe/benchmark facts are known. This lets failed runs remain schema-valid without pretending missing inputs were completed.

The loader must resolve anchors once and load all prices in one bounded query:

```python
anchor_dates = set(anchors.values())
rows = (
    db.query(
        StockPrice.symbol,
        StockPrice.date,
        StockPrice.adj_close,
        StockPrice.close,
    )
    .filter(
        StockPrice.symbol.in_([*universe.symbols, *benchmark_candidates]),
        StockPrice.date.in_(anchor_dates),
    )
    .all()
)
prices = {
    (row.symbol, row.date): float(
        row.adj_close if row.adj_close is not None else row.close
    )
    for row in rows
    if (row.adj_close is not None or row.close is not None)
}
```

Choose the first benchmark candidate with every anchor. For each stock, require every anchor and calculate:

```python
stock_return = current_stock / past_stock - 1.0
benchmark_return = current_benchmark / past_benchmark - 1.0
excess_returns[horizon] = stock_return - benchmark_return
```

Use `HORIZON_SESSIONS` to map horizon names to dates. Count current-session price availability separately and raise below `0.90`; insufficient long history is an exclusion, not a refresh-coverage failure.

- [ ] **Step 6: Implement the repository lifecycle**

`MarketRsRunRepository` must provide these exact signatures:

- `start_or_restart(self, db: Session, *, market: str, as_of_date: date, formula_version: str, benchmark_symbol: str, benchmark_as_of_date: date, universe_hash: str, expected_symbol_count: int) -> MarketRsRun`
- `replace_rows(self, db: Session, run: MarketRsRun, scores: Mapping[str, StockRsScore]) -> None`
- `mark_completed(self, run: MarketRsRun, *, excluded_symbol_count: int, diagnostics: dict[str, object]) -> MarketRsRun`
- `mark_failed(self, run: MarketRsRun, *, diagnostics: dict[str, object]) -> MarketRsRun`
- `get_completed_exact(self, db: Session, *, market: str, as_of_date: date, formula_version: str) -> MarketRsRun | None`
- `get_latest_completed(self, db: Session, *, market: str, formula_version: str, through_date: date | None = None) -> MarketRsRun | None`
- `active_formula(self, db: Session, *, market: str) -> str`
- `activate_formula(self, db: Session, *, market: str, formula_version: str) -> None`

`start_or_restart()` locks an existing key with `SELECT ... FOR UPDATE`; an insert race catches the unique-key conflict in a savepoint and reloads the winner. It returns an existing completed run unchanged, but resets a failed/running run, refreshes its input metadata, and deletes its partial rows. `replace_rows()` sets `eligible_symbol_count = len(scores)`, writes all rows, and flushes without committing. `mark_completed()` validates `len(run.rows) == eligible_symbol_count`, `expected_symbol_count == eligible_symbol_count + excluded_symbol_count`, finite composites, and 1–99 component/overall ranges before setting the timestamp and status. `active_formula()` raises `MarketRsFormulaNotConfigured` for an unseeded Market instead of silently selecting a mode. `activate_formula()` locks its pointer row. Repository mutators flush but never commit so the snapshot and rollout services own transaction boundaries.

- [ ] **Step 7: Implement the snapshot service transaction boundaries**

At entry, `MarketRsSnapshotService.calculate` must reject any formula other than `balanced-horizon-percentile-v2`; legacy remains an isolated calculation/read path and does not write canonical stock rows. Use this lifecycle:

```python
run = self.repository.start_or_restart(
    db,
    market=market,
    as_of_date=as_of_date,
    formula_version=formula_version,
    benchmark_symbol=inputs.benchmark_symbol,
    benchmark_as_of_date=inputs.benchmark_as_of_date,
    universe_hash=inputs.universe_hash,
    expected_symbol_count=len(inputs.expected_symbols),
)
if run.status == "completed":
    return run

try:
    scores = calculate_balanced_rs(inputs.excess_returns_by_symbol)
    self.repository.replace_rows(db, run, scores)
    self.repository.mark_completed(
        run,
        excluded_symbol_count=len(inputs.exclusions),
        diagnostics={
            "current_price_coverage": inputs.current_price_coverage,
            "exclusions": inputs.exclusions,
        },
    )
    db.commit()
    db.refresh(run)
    return run
except Exception as exc:
    db.rollback()
    failed = self.repository.start_or_restart(
        db,
        market=market,
        as_of_date=as_of_date,
        formula_version=formula_version,
        benchmark_symbol=inputs.benchmark_symbol,
        benchmark_as_of_date=inputs.benchmark_as_of_date,
        universe_hash=inputs.universe_hash,
        expected_symbol_count=len(inputs.expected_symbols),
    )
    self.repository.mark_failed(
        failed,
        diagnostics={"error_type": type(exc).__name__, "error": str(exc)},
    )
    db.commit()
    raise
```

Keep the `running` insert/lock, row replacement, and transition to `completed` in one database transaction; do not commit `running` separately. Resolve inputs before `start_or_restart`. Catch `MarketRsInputUnavailable` separately, call `start_or_restart()` with the exception's context and `benchmark_as_of_date=as_of_date`, mark that run failed with `{"reason_code": exc.reason_code, **exc.diagnostics}`, commit, and re-raise. Do not catch programming errors as input failures. The broader calculation `except` shown above applies only after a running row exists; it rolls back the attempted publication, creates/resets the failed run, records `error_type`/`error`, commits `failed`, and re-raises.

- [ ] **Step 8: Run focused tests and the existing price/calendar suite**

Before running, add lazy point-in-time universe, input-loader, repository, and snapshot-service properties to `RuntimeServices`; expose `get_market_rs_snapshot_service()` and clear all new cached objects in `reset_for_tests()`.

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_inputs.py tests/unit/repositories/test_market_rs_repo.py tests/unit/test_market_rs_snapshot_service.py tests/unit/test_price_row_normalization.py tests/unit/test_market_calendar_service.py tests/unit/test_runtime_services.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit canonical snapshot publication**

```bash
git add backend/app/services/market_rs_inputs.py backend/app/infra/db/repositories/market_rs_repo.py backend/app/services/market_rs_snapshot_service.py backend/app/wiring/bootstrap.py backend/tests/unit/test_market_rs_inputs.py backend/tests/unit/repositories/test_market_rs_repo.py backend/tests/unit/test_market_rs_snapshot_service.py backend/tests/unit/test_runtime_services.py
git commit -m "feat: publish canonical market RS snapshots"
```

---

### Task 5: Canonical RS Reader and Scan-Path Hydration

**Files:**
- Create: `backend/app/services/market_rs_reader.py`
- Modify: `backend/app/domain/scanning/ports.py`
- Modify: `backend/app/scanners/base_screener.py`
- Modify: `backend/app/use_cases/feature_store/build_daily_snapshot.py`
- Modify: `backend/app/use_cases/scanning/run_bulk_scan.py`
- Modify: `backend/app/interfaces/tasks/feature_store_tasks.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Create: `backend/tests/unit/test_market_rs_reader.py`
- Modify: `backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py`
- Modify: `backend/tests/unit/use_cases/test_run_bulk_scan.py`
- Modify: `backend/tests/unit/test_feature_store_tasks.py`
- Modify: `backend/tests/unit/test_runtime_services.py`

**Interfaces:**
- Consumes: completed active-formula stock snapshots from Task 4.
- Produces: `MarketRsReader.get(market=market, symbols=symbols, as_of_date=as_of_date, formula_version=formula_version) -> MarketRsResolution` and hydrated `StockData.canonical_rs_ratings` plus audit metadata.

- [ ] **Step 1: Write failing exact-date and universe-independence reader tests**

```python
resolution = reader.get(
    market="US",
    symbols=("AAA", "CCC"),
    as_of_date=date(2026, 4, 10),
)

assert resolution.formula_version == BALANCED_RS_FORMULA_VERSION
assert resolution.run_id == completed_run.id
assert resolution.universe_size == 3
assert resolution.ratings_by_symbol["AAA"] == {
    "rs_rating": 99,
    "rs_rating_1m": 50,
    "rs_rating_3m": 75,
    "rs_rating_12m": 99,
}
assert "BBB" not in resolution.ratings_by_symbol
```

Call again with a one-symbol watchlist and assert `AAA` receives the same values. Add a balanced-active/missing-exact-run case that raises `CanonicalMarketRsUnavailable`, and a legacy-active case that returns `mode == "legacy"` without querying stock snapshot rows.

- [ ] **Step 2: Add failing use-case hydration tests**

For both feature snapshot and bulk scan, provide two prefetched `StockData` rows in different Markets and a fake reader. Assert the reader is called once per Market and the scanner receives:

```python
assert stock_data.canonical_rs_ratings["rs_rating"] == 87
assert stock_data.rs_formula_version == BALANCED_RS_FORMULA_VERSION
assert stock_data.market_rs_run_id == 42
assert stock_data.rs_universe_size == 5000
```

For `BuildDailySnapshotCommand`, assert `as_of_date` is passed exactly. For a live bulk scan, assert `as_of_date=None` resolves the latest active completed run.

Add a rollout case with `rs_formula_version_override=BALANCED_RS_FORMULA_VERSION` while the Market pointer is legacy. Assert the exact balanced run is used and the Feature run's `config_json` records both `rs_formula_version` and `market_rs_run_id`.

- [ ] **Step 3: Run focused tests and verify missing reader fields**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_reader.py tests/unit/use_cases/feature_store/test_build_daily_snapshot.py tests/unit/use_cases/test_run_bulk_scan.py -q
```

Expected: fail because the reader and `StockData` audit fields are absent.

- [ ] **Step 4: Define the read contract**

Add to `backend/app/domain/scanning/ports.py`:

```python
@dataclass(frozen=True)
class MarketRsResolution:
    market: str
    as_of_date: date | None
    formula_version: str
    mode: str
    run_id: int | None
    universe_size: int | None
    ratings_by_symbol: dict[str, dict[str, int]]


class MarketRsReader(Protocol):
    def get(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        as_of_date: date | None,
        formula_version: str | None = None,
    ) -> MarketRsResolution:
        raise NotImplementedError
```

Extend `StockData` with:

```python
canonical_rs_ratings: Optional[Dict[str, int]] = None
rs_formula_version: Optional[str] = None
market_rs_run_id: Optional[int] = None
rs_universe_size: Optional[int] = None
```

- [ ] **Step 5: Implement the SQL reader**

`SqlMarketRsReader(session_factory)` opens one short-lived session per `get()`. It uses the explicit `formula_version` when supplied; otherwise it reads the Market pointer. For legacy mode it returns no rows. For balanced mode it resolves an exact completed run when a date is supplied, or latest completed when no date is supplied, then fetches only requested symbols:

```python
rows = (
    db.query(StockRsSnapshot)
    .filter(
        StockRsSnapshot.run_id == run.id,
        StockRsSnapshot.symbol.in_(normalized_symbols),
    )
    .all()
)
ratings = {
    row.symbol: {
        "rs_rating": int(row.overall_rs),
        "rs_rating_1m": int(row.rs_1m),
        "rs_rating_3m": int(row.rs_3m),
        "rs_rating_12m": int(row.rs_12m),
    }
    for row in rows
}
```

- [ ] **Step 6: Hydrate prefetched stock data in both use cases**

Add `rs_formula_version_override: str | None = None` to `BuildDailySnapshotCommand` and the `build_daily_snapshot` task interface. After resolving the Market-scoped symbol list but before `feature_runs.start_run()`, call the reader once with that list/date/override; write the resolved formula, run ID, RS date, and universe size into the new run's `config_json`. Pass that immutable resolution into `_run()` and attach its values to each prefetched `StockData` before `scan_stock_multi()`.

For bulk scans, group prefetched `StockData` by normalized Market, call the reader once per Market with no override, and attach the same four fields. Do not ask `DataPreparationLayer` to construct a percentile universe. This gives the static rollout an auditable exact feature run before pointer activation.

Inject the reader through each use-case constructor and wire one process-scoped `SqlMarketRsReader` in `RuntimeServices`. Reset it in `reset_for_tests()`.

- [ ] **Step 7: Run hydration and runtime wiring tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_reader.py tests/unit/use_cases/feature_store/test_build_daily_snapshot.py tests/unit/use_cases/test_run_bulk_scan.py tests/unit/test_feature_store_tasks.py tests/unit/test_runtime_services.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit the canonical reader integration**

```bash
git add backend/app/services/market_rs_reader.py backend/app/domain/scanning/ports.py backend/app/scanners/base_screener.py backend/app/use_cases/feature_store/build_daily_snapshot.py backend/app/use_cases/scanning/run_bulk_scan.py backend/app/interfaces/tasks/feature_store_tasks.py backend/app/wiring/bootstrap.py backend/tests/unit/test_market_rs_reader.py backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py backend/tests/unit/use_cases/test_run_bulk_scan.py backend/tests/unit/test_feature_store_tasks.py backend/tests/unit/test_runtime_services.py
git commit -m "feat: hydrate scans from canonical RS snapshots"
```

---

### Task 6: Make Every Scanner Consume Canonical Ratings

**Files:**
- Create: `backend/app/scanners/criteria/rs_resolution.py`
- Modify: `backend/app/scanners/scan_orchestrator.py`
- Modify: `backend/app/scanners/minervini_scanner.py`
- Modify: `backend/app/scanners/canslim_scanner.py`
- Modify: `backend/app/scanners/custom_scanner.py`
- Modify: `backend/app/scanners/setup_engine_screener.py`
- Modify: `backend/app/scanners/partial_history_metrics.py`
- Modify: `backend/app/scanners/data_preparation.py`
- Modify: `backend/app/scanners/criteria/relative_strength.py`
- Modify: `backend/app/api/v1/technical.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Create: `backend/tests/unit/test_canonical_rs_scanner_resolution.py`
- Create: `backend/tests/unit/test_technical_rs_api.py`
- Modify: `backend/tests/unit/test_precomputed_scan_context.py`
- Modify: `backend/tests/unit/test_custom_scanner.py`
- Modify: `backend/tests/unit/test_runtime_services.py`
- Modify: `backend/tests/parity/test_market_parity_e6.py`

**Interfaces:**
- Consumes: hydrated `StockData` fields from Task 5.
- Produces: one guarded `resolve_stock_rs(stock_data, legacy_factory=legacy_factory)` policy used by Minervini, CANSLIM, custom RS filters, setup engine, and partial-history rows.

- [ ] **Step 1: Write failing consumer-policy tests**

Cover these three states:

```python
def test_balanced_mode_returns_only_canonical_values():
    data = stock_data(
        canonical_rs_ratings={
            "rs_rating": 88,
            "rs_rating_1m": 12,
            "rs_rating_3m": 25,
            "rs_rating_12m": 99,
        },
        rs_formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    assert resolve_stock_rs(data, legacy_factory=unexpected_call) == data.canonical_rs_ratings


def test_balanced_ineligible_stock_does_not_fall_back_to_linear_scaling():
    data = stock_data(
        canonical_rs_ratings=None,
        rs_formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    with pytest.raises(CanonicalStockRsUnavailable):
        resolve_stock_rs(data, legacy_factory=unexpected_call)


def test_legacy_mode_keeps_existing_calculator_until_activation():
    expected = {"rs_rating": 72, "rs_rating_1m": 65, "rs_rating_3m": 70, "rs_rating_12m": 80}
    data = stock_data(rs_formula_version=LEGACY_RS_FORMULA_VERSION)
    assert resolve_stock_rs(data, legacy_factory=lambda: expected) == expected
```

Add scanner assertions that the top-level result and Minervini/CANSLIM details contain the identical canonical values, and that a custom `rs_rating_min` filter evaluates the canonical overall rating. Call the single-symbol Minervini route without prehydrated `StockData` and assert the orchestrator resolves the latest canonical row.

For `/technical/{symbol}/rs-rating`, activate balanced mode and assert it returns canonical overall/1M/3M/12M plus formula/run/date/universe metadata without calling yfinance or `RelativeStrengthCalculator`. Assert an ineligible symbol gets the existing not-enough-history response rather than a linear rating.

- [ ] **Step 2: Run scanner tests and verify they fail**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_canonical_rs_scanner_resolution.py tests/unit/test_precomputed_scan_context.py tests/unit/test_custom_scanner.py tests/unit/test_technical_rs_api.py -q
```

Expected: fail because scanner paths still call `RelativeStrengthCalculator` directly.

- [ ] **Step 3: Add the one resolution policy**

Create:

```python
class CanonicalStockRsUnavailable(RuntimeError):
    pass


def resolve_stock_rs(
    stock_data: StockData,
    *,
    legacy_factory: Callable[[], dict[str, float]],
) -> dict[str, float]:
    if stock_data.canonical_rs_ratings is not None:
        return dict(stock_data.canonical_rs_ratings)
    if stock_data.rs_formula_version == BALANCED_RS_FORMULA_VERSION:
        raise CanonicalStockRsUnavailable(
            f"{stock_data.symbol} is not eligible in canonical Market RS run "
            f"{stock_data.market_rs_run_id}"
        )
    return legacy_factory()
```

- [ ] **Step 4: Route all scanner consumers through the policy**

In `build_precomputed_scan_context()`, set `rs_ratings` with `resolve_stock_rs`; preserve the current `RelativeStrengthCalculator` call only inside `legacy_factory`.

In Minervini, CANSLIM, setup engine, custom scanner, and partial-history metrics:

- read `precomputed.rs_ratings` or call the same policy;
- remove balanced-mode calls to `calculate_rs_rating`, `calculate_period_rs_rating`, and `calculate_all_rs_ratings`;
- convert `CanonicalStockRsUnavailable` into the existing insufficient-history/result-status path;
- promote `rs_formula_version`, `market_rs_run_id`, and `rs_universe_size` into the top-level scan result and feature-row details.

Inject `MarketRsReader` into `ScanOrchestrator`. If `scan_stock_multi()` prepared a single stock directly and it has no RS resolution, resolve its normalized Market/symbol with `as_of_date=None`, attach canonical fields, and only then build the precomputed context. Bulk and feature paths keep their one-call-per-Market hydration and therefore do not add per-symbol queries.

Change `/technical/{symbol}/rs-rating` to resolve the symbol's Market from `StockUniverse`, then call `MarketRsReader`. Balanced mode returns the stored display fields and audit metadata; it never fetches SPY or calculates a local rating. Keep the existing live-fetch calculation only when that Market's active formula is `legacy-linear-v1`, and label the helper/docstring legacy-only.

Keep `_compute_market_rs_universe_performances()`, `_attach_market_rs_universe_performances()`, the legacy field, and the legacy calculator for rollback. Mark them legacy-only and call them only when the pre-resolved `MarketRsResolution.mode == "legacy"`; balanced preparation must bypass them entirely. Add a regression test proving legacy mode retains its current behavior while balanced mode never builds a scan-local percentile universe.

- [ ] **Step 5: Run scanner, parity, and feature-row tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_canonical_rs_scanner_resolution.py tests/unit/test_precomputed_scan_context.py tests/unit/test_custom_scanner.py tests/unit/test_minervini_scanner.py tests/unit/test_canslim_scanner.py tests/unit/test_technical_rs_api.py tests/unit/test_feature_store_tasks.py tests/unit/test_runtime_services.py tests/parity/test_market_parity_e6.py -q
```

Expected: all tests pass; no balanced test invokes the legacy calculator.

- [ ] **Step 6: Commit scanner standardization**

```bash
git add backend/app/scanners/criteria/rs_resolution.py backend/app/scanners/scan_orchestrator.py backend/app/scanners/minervini_scanner.py backend/app/scanners/canslim_scanner.py backend/app/scanners/custom_scanner.py backend/app/scanners/setup_engine_screener.py backend/app/scanners/partial_history_metrics.py backend/app/scanners/data_preparation.py backend/app/scanners/criteria/relative_strength.py backend/app/api/v1/technical.py backend/app/wiring/bootstrap.py backend/tests/unit/test_canonical_rs_scanner_resolution.py backend/tests/unit/test_precomputed_scan_context.py backend/tests/unit/test_custom_scanner.py backend/tests/unit/test_technical_rs_api.py backend/tests/unit/test_runtime_services.py backend/tests/parity/test_market_parity_e6.py
git commit -m "refactor: consume canonical RS in every scanner"
```

---

### Task 7: Canonical Group Aggregation and Versioned Group Reads

**Files:**
- Create: `backend/app/services/canonical_group_ranking_service.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/app/services/group_ranking_payloads.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`
- Create: `backend/tests/unit/test_canonical_group_ranking_service.py`
- Modify: `backend/tests/unit/test_group_ranking_payloads.py`
- Modify: `backend/tests/unit/test_runtime_services.py`

**Interfaces:**
- Consumes: one completed `MarketRsRun` and its `StockRsSnapshot` rows.
- Produces: versioned `IBDGroupRank` rows whose overall/1M/3M metrics share one constituent set.

- [ ] **Step 1: Write failing Group aggregation tests**

Seed one completed Market run with four stock rows and two mapped Groups. Assert:

```python
rankings = service.calculate_and_store(
    db_session,
    market="US",
    as_of_date=date(2026, 4, 10),
    formula_version=BALANCED_RS_FORMULA_VERSION,
)

leaders = next(row for row in rankings if row["industry_group"] == "Leaders")
assert leaders["avg_rs_rating"] == pytest.approx((99 + 80 + 50) / 3)
assert leaders["avg_rs_rating_1m"] == pytest.approx((10 + 20 + 30) / 3)
assert leaders["avg_rs_rating_3m"] == pytest.approx((20 + 40 + 60) / 3)
assert leaders["num_stocks"] == 3
assert leaders["market_rs_run_id"] == market_run.id
assert leaders["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
```

Add tests that:

- a mapped symbol absent from the canonical run is excluded from all three means;
- Groups with fewer than three eligible constituents are omitted;
- main ordering uses unrounded average overall RS and alphabetical exact-tie order;
- market-cap-weighted RS does not alter main rank;
- top-stock ties use overall RS, 1M RS, market cap, then symbol;
- legacy and balanced rows coexist for the same Market/date/Group.

- [ ] **Step 2: Write failing active-formula read tests**

```python
seed_group_row(db_session, formula_version=LEGACY_RS_FORMULA_VERSION, avg_rs=95.0)
seed_group_row(db_session, formula_version=BALANCED_RS_FORMULA_VERSION, avg_rs=72.0)
set_active_formula(db_session, "US", BALANCED_RS_FORMULA_VERSION)

rows = ibd_service.get_current_rankings(db_session, market="US", limit=197)

assert len(rows) == 1
assert rows[0]["avg_rs_rating"] == 72.0
assert rows[0]["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
```

Exercise rank changes and Group history with both versions present and assert historical lookups never cross the active version.

Add a failing runtime test that the wired `IBDGroupRankService` receives the one process-scoped `CanonicalGroupRankingService` instance and that reset clears it.

- [ ] **Step 3: Run Group tests and verify failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_canonical_group_ranking_service.py tests/unit/test_group_rank_service.py tests/unit/test_group_ranking_payloads.py -q
```

Expected: fail because Group rows lack canonical component aggregation and reads do not filter formula.

- [ ] **Step 4: Implement canonical Group metrics**

`CanonicalGroupRankingService.calculate_and_store()` must resolve one exact completed run and query all its stock rows once. For every taxonomy Group, form one eligible row list and compute:

```python
overall = [float(row.overall_rs) for row in rows]
one_month = [float(row.rs_1m) for row in rows]
three_month = [float(row.rs_3m) for row in rows]
caps = [float(market_caps.get(row.symbol) or 0.0) for row in rows]

avg_rs = statistics.fmean(overall)
avg_rs_1m = statistics.fmean(one_month)
avg_rs_3m = statistics.fmean(three_month)
median_rs = statistics.median(overall)
std_dev = statistics.pstdev(overall)
positive_cap_total = sum(cap for cap in caps if cap > 0)
weighted_avg = (
    sum(rs * cap for rs, cap in zip(overall, caps) if cap > 0) / positive_cap_total
    if positive_cap_total > 0
    else None
)
```

Sort Group dictionaries with:

```python
group_metrics.sort(key=lambda row: (-row["_unrounded_avg_rs"], row["industry_group"]))
for rank, row in enumerate(group_metrics, start=1):
    row["rank"] = rank
```

Compute `num_stocks_rs_above_80` from overall ratings and select the top stock with `(-overall_rs, -rs_1m, -market_cap, symbol)`. Persist the full-precision floating means so the stored main rank remains auditable; round only at API/UI display time. Remove `_unrounded_avg_rs` from the public payload. Upsert on `(industry_group, date, market, rs_formula_version)` and include `avg_rs_rating_1m`, `avg_rs_rating_3m`, and `market_rs_run_id` in both PostgreSQL and SQLite paths.

- [ ] **Step 5: Dispatch calculation by explicit or active formula**

Extend `IBDGroupRankService.__init__` with required `canonical_group_service`. Add a lazy canonical Group service to `RuntimeServices` and inject that exact instance into the wired `IBDGroupRankService`. In `calculate_group_rankings()`:

```python
requested_formula = formula_version or self.market_rs_repository.active_formula(
    db,
    market=normalized_market,
)
if requested_formula == BALANCED_RS_FORMULA_VERSION:
    return self.canonical_group_service.calculate_and_store(
        db,
        market=normalized_market,
        as_of_date=calculation_date,
        formula_version=requested_formula,
    )
```

Keep the existing raw-return implementation only for `legacy-linear-v1` rollback operation. Every legacy write must set `rs_formula_version=legacy-linear-v1`, `market_rs_run_id=None`, and component Group averages to `None`.

- [ ] **Step 6: Filter every Group read by one formula version**

Add keyword-only `formula_version: str | None = None` to `get_current_rankings`, `get_group_history`, and `get_rank_movers`; propagate the resolved value into `_get_historical_ranks_batch` and every summary/history query. Use an explicit version for rollout/static reads or resolve the active version once when omitted, then include:

```python
IBDGroupRank.rs_formula_version == formula_version
```

Move the ORM-to-dict serializer into public `group_ranking_payloads.rank_record_payload()` and have `IBDGroupRankService` delegate to it, so live and static readers cannot drift. Add these payload fields there:

```python
"avg_rs_rating_1m": ranking.avg_rs_rating_1m,
"avg_rs_rating_3m": ranking.avg_rs_rating_3m,
"rs_formula_version": ranking.rs_formula_version,
"market_rs_run_id": ranking.market_rs_run_id,
```

- [ ] **Step 7: Run Group tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_canonical_group_ranking_service.py tests/unit/test_group_rank_service.py tests/unit/test_group_ranking_payloads.py tests/unit/test_group_ranking_history.py tests/unit/test_runtime_services.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit canonical Group aggregation**

```bash
git add backend/app/services/canonical_group_ranking_service.py backend/app/services/ibd_group_rank_service.py backend/app/services/group_ranking_payloads.py backend/app/wiring/bootstrap.py backend/tests/unit/test_canonical_group_ranking_service.py backend/tests/unit/test_group_rank_service.py backend/tests/unit/test_group_ranking_payloads.py backend/tests/unit/test_group_ranking_history.py backend/tests/unit/test_runtime_services.py
git commit -m "feat: aggregate groups from canonical RS"
```

---

### Task 8: Live Group API, Bootstrap, and Scan Enrichment Metadata

**Files:**
- Modify: `backend/app/schemas/groups.py`
- Modify: `backend/app/api/v1/groups.py`
- Modify: `backend/app/services/ui_snapshot_service.py`
- Modify: `backend/app/services/market_group_ranking_service.py`
- Modify: `backend/app/infra/db/repositories/scan_result_repo.py`
- Modify: `backend/tests/unit/test_groups_api_no_data.py`
- Modify: `backend/tests/unit/test_ui_snapshot_service.py`
- Modify: `backend/tests/unit/test_market_group_ranking_service.py`
- Modify: `backend/tests/integration/test_scan_result_repo_enrichment.py`

**Interfaces:**
- Consumes: version-filtered Group payloads from Task 7 and Market run metadata from Task 4.
- Produces: live API/bootstrap envelopes exposing Group 1M/3M and canonical formula/date/universe metadata.

- [ ] **Step 1: Add failing schema and endpoint assertions**

Extend existing API fixtures and assert:

```python
payload = response.json()
assert payload["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
assert payload["rs_as_of_date"] == "2026-04-10"
assert payload["rs_universe_size"] == 5000
assert payload["rankings"][0]["avg_rs_rating_1m"] == 41.5
assert payload["rankings"][0]["avg_rs_rating_3m"] == 63.2
```

Add the same assertions to the Groups bootstrap snapshot. For non-US Group detail and scan enrichment, seed conflicting feature-run-derived values and assert the stored active-formula `IBDGroupRank` wins.

- [ ] **Step 2: Run live payload tests and verify failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_groups_api_no_data.py tests/unit/test_ui_snapshot_service.py tests/unit/test_market_group_ranking_service.py tests/integration/test_scan_result_repo_enrichment.py -q
```

Expected: fail because the response models and non-US read paths do not expose canonical metadata.

- [ ] **Step 3: Extend Pydantic response contracts**

Add to `GroupRankResponse`:

```python
avg_rs_rating_1m: Optional[float] = Field(None, description="Average constituent 1-month Market RS")
avg_rs_rating_3m: Optional[float] = Field(None, description="Average constituent 3-month Market RS")
rs_formula_version: str = Field(description="Canonical RS formula version")
market_rs_run_id: Optional[int] = Field(None, description="Canonical Market RS run identifier")
```

Add to `GroupRankingsResponse`:

```python
rs_formula_version: str
rs_as_of_date: str
rs_universe_size: Optional[int] = None
```

Add `current_avg_rs_1m` and `current_avg_rs_3m` to `GroupDetailResponse`, and carry component fields in historical points only when available.

- [ ] **Step 4: Build metadata from the exact canonical run**

Add a service helper that validates every returned Group row has the same formula/run and loads the referenced run:

```python
def group_snapshot_metadata(
    db: Session,
    *,
    market: str,
    rankings: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rankings:
        raise RuntimeError("no Group rankings are available")
    formula_versions = {row["rs_formula_version"] for row in rankings}
    run_ids = {row.get("market_rs_run_id") for row in rankings if row.get("market_rs_run_id")}
    if len(formula_versions) != 1 or len(run_ids) > 1:
        raise RuntimeError("group snapshot mixes canonical RS sources")
    formula_version = next(iter(formula_versions))
    if formula_version == BALANCED_RS_FORMULA_VERSION and len(run_ids) != 1:
        raise RuntimeError("balanced Group snapshot has no single Market RS run")
    run = db.get(MarketRsRun, next(iter(run_ids))) if run_ids else None
    if run is not None and (
        run.market != market
        or run.formula_version != formula_version
        or run.as_of_date.isoformat() != rankings[0]["date"]
        or run.status != "completed"
    ):
        raise RuntimeError("Group snapshot metadata does not match its Market RS run")
    return {
        "rs_formula_version": formula_version,
        "rs_as_of_date": rankings[0]["date"],
        "rs_universe_size": run.eligible_symbol_count if run is not None else None,
    }
```

Use it in `/rankings/current`, `/bootstrap`, `UI SnapshotService._build_groups_payload`, movers, and home/top-Group payloads. Preserve each caller's existing empty/no-data response before invoking the helper so a Market with no snapshot does not become a 500 response.

- [ ] **Step 5: Remove feature-row Group recomputation from live non-US reads**

Change the non-US detail path to call `IBDGroupRankService.get_group_history(db, industry_group, days, market=normalized_market)`. Convert `MarketGroupRankingService` into a compatibility adapter that delegates current/history/mover calls to the version-aware stored Group service; it must not call `compute_group_rankings_from_serialized_rows`.

In `SqlScanResultRepository._load_symbol_enrichment`, resolve the active formula once per Market and query `IBDGroupRank` with Market/date/formula. Remove `MarketGroupRankingService.get_current_rank_snapshot()` from this persistence path.

- [ ] **Step 6: Run live API and enrichment tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_groups_api_no_data.py tests/unit/test_ui_snapshot_service.py tests/unit/test_market_group_ranking_service.py tests/integration/test_scan_result_repo_enrichment.py -q
```

Expected: all tests pass for US and non-US Markets.

- [ ] **Step 7: Commit live payload standardization**

```bash
git add backend/app/schemas/groups.py backend/app/api/v1/groups.py backend/app/services/ui_snapshot_service.py backend/app/services/market_group_ranking_service.py backend/app/infra/db/repositories/scan_result_repo.py backend/tests/unit/test_groups_api_no_data.py backend/tests/unit/test_ui_snapshot_service.py backend/tests/unit/test_market_group_ranking_service.py backend/tests/integration/test_scan_result_repo_enrichment.py
git commit -m "feat: expose canonical Group RS metadata"
```

---

### Task 9: Formula-Isolated Live and Static RRG History

**Files:**
- Modify: `backend/app/services/rrg_history_provider.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/app/services/static_rrg_history_contract.py`
- Modify: `backend/app/services/static_rrg_history_bundle.py`
- Modify: `backend/app/services/static_groups_rrg_export.py`
- Modify: `backend/tests/unit/test_rrg_service.py`
- Modify: `backend/tests/unit/test_static_rrg_history_bundle.py`
- Modify: `backend/tests/unit/test_static_groups_rrg_sources.py`
- Modify: `backend/tests/unit/golden/test_golden_rrg.py`

**Interfaces:**
- Consumes: versioned `IBDGroupRank` history.
- Produces: one formula-filtered history provider for every Market and `static-rrg-history-v4` bundles carrying `rs_formula_version`.

- [ ] **Step 1: Add failing mixed-history tests**

Seed legacy and balanced Group rows for identical dates with deliberately different averages. Activate balanced and assert the live provider series contains only balanced values. Activate legacy and assert only legacy values.

For static history, assert:

```python
state = service.build(
    db_session,
    market="US",
    through_date=date(2026, 4, 10),
    formula_version=BALANCED_RS_FORMULA_VERSION,
)
assert state.schema_version == "static-rrg-history-v4"
assert state.rs_formula_version == BALANCED_RS_FORMULA_VERSION
```

Pass a legacy previous bundle to the low-level balanced merge validator and assert `StaticRRGHistoryBundleError` before any merge occurs. Through the normal `prepare()` workflow, assert that mismatch is treated as an invalid prior state and triggers a full balanced database rebuild rather than aborting export.

- [ ] **Step 2: Run RRG tests and verify mixed-version failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_rrg_service.py tests/unit/test_static_rrg_history_bundle.py tests/unit/test_static_groups_rrg_sources.py tests/unit/golden/test_golden_rrg.py -q
```

Expected: fail because providers query by Market/date only and rolling state has no formula field.

- [ ] **Step 3: Use one stored Group history provider for every Market**

Replace US/non-US dispatch with `StoredGroupRankHistoryProvider`. It resolves the active formula once, calls version-aware `get_current_rankings`, and filters its history query by:

```python
IBDGroupRank.market == market,
IBDGroupRank.rs_formula_version == formula_version,
IBDGroupRank.date >= cutoff,
IBDGroupRank.date <= latest_day,
```

Wire this provider into `RRGService` for all supported Markets. Do not change `RRGService`, `analysis/rrg_weekly.py`, EMA windows, z-scores, momentum windows, weekly bucketing, or tail construction.

- [ ] **Step 4: Version the rolling static RRG contract**

Change:

```python
StaticRRGHistorySchemaVersion = Literal["static-rrg-history-v4"]
STATIC_RRG_HISTORY_SCHEMA_VERSION = "static-rrg-history-v4"
```

Add required `rs_formula_version: str` to `StaticRRGHistoryState`. Pass an explicit formula through `prepare()`, `build()`, `StaticGroupsRRGRollingHistoryExportSession`, and `_build_payload_from_state`. Reject `previous.rs_formula_version != requested_formula` in the merge validator; the preparation layer logs that invalid state and rebuilds all retained weeks from version-filtered database rows.

- [ ] **Step 5: Prove the RRG mathematics did not change**

Update only fixture envelopes/formula metadata in the golden tests. Keep all existing expected x/y coordinates byte-for-byte. Add:

```python
assert balanced_output["payload"]["groups"]["groups"] == expected_coordinates
assert live_output["payload"]["groups"] == static_output["payload"]["groups"]
```

- [ ] **Step 6: Run live/static RRG tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_rrg_service.py tests/unit/test_static_rrg_history_bundle.py tests/unit/test_static_groups_rrg_sources.py tests/unit/golden/test_golden_rrg.py -q
```

Expected: all tests pass and existing coordinate fixtures are unchanged.

- [ ] **Step 7: Commit RRG history isolation**

```bash
git add backend/app/services/rrg_history_provider.py backend/app/wiring/bootstrap.py backend/app/services/static_rrg_history_contract.py backend/app/services/static_rrg_history_bundle.py backend/app/services/static_groups_rrg_export.py backend/tests/unit/test_rrg_service.py backend/tests/unit/test_static_rrg_history_bundle.py backend/tests/unit/test_static_groups_rrg_sources.py backend/tests/unit/golden/test_golden_rrg.py
git commit -m "fix: isolate RRG history by RS formula"
```

---

### Task 10: Export the Exact Stored Group Snapshot in Static Schema v3

**Files:**
- Modify: `backend/app/services/static_groups_payload_builder.py`
- Modify: `backend/app/services/static_site_export_service.py`
- Modify: `backend/tests/unit/test_static_groups_payload_builder.py`
- Modify: `backend/tests/unit/test_static_site_export_service.py`
- Modify: `backend/tests/unit/test_static_groups_rrg_sources.py`

**Interfaces:**
- Consumes: an explicit rollout formula/run override or the Market's active formula/latest feature pointer, plus exact-date `IBDGroupRank` rows from Task 7.
- Produces: `static-site-v3` Groups artifacts whose rankings, history, movers, and RS metadata all come from one stored formula/run.

- [ ] **Step 1: Add failing tests that reject independent static recomputation**

Seed feature rows with Group RS values that conflict with stored balanced Group rows. Build the static Groups page and assert:

```python
groups = artifact["payload"]["rankings"]
assert artifact["schema_version"] == "static-site-v3"
assert groups["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
assert groups["rs_as_of_date"] == "2026-04-10"
assert groups["rs_universe_size"] == 5000
assert groups["rankings"][0]["avg_rs_rating"] == 81.25
assert groups["rankings"][0]["avg_rs_rating_1m"] == 34.5
assert groups["rankings"][0]["avg_rs_rating_3m"] == 57.75
```

Assert the feature-row values are absent. Add cases where no exact-date Group snapshot exists, rows mix formula/run IDs, or their run date differs; each must raise `StaticSiteSectionUnavailableError` with `section="groups"` and a reason naming the mismatch instead of recomputing.

- [ ] **Step 2: Add a failing live/static parity test**

For a fixed database fixture, call the live `IBDGroupRankService.get_current_rankings()` and static `_build_groups_payload()`. Compare these fields for every Group after sorting by name:

```python
PARITY_FIELDS = (
    "industry_group",
    "rank",
    "avg_rs_rating",
    "avg_rs_rating_1m",
    "avg_rs_rating_3m",
    "num_stocks",
    "top_symbol",
    "rs_formula_version",
    "market_rs_run_id",
)
```

Also assert static RRG receives the same explicit formula as the Groups artifact.

- [ ] **Step 3: Run static payload tests and verify failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_groups_payload_builder.py tests/unit/test_static_site_export_service.py tests/unit/test_static_groups_rrg_sources.py -q
```

Expected: fail because `_build_groups_payload()` still calls `compute_group_rankings_from_serialized_rows()` and the static schema is v2.

- [ ] **Step 4: Extend the normalized static snapshot contract**

Extend `StaticGroupsSnapshot` with required fields:

```python
rs_formula_version: str
market_rs_run_id: int | None
rs_universe_size: int | None
```

Construct `GroupRankingsResponse` with `rs_formula_version`, `rs_as_of_date`, and `rs_universe_size`. Add the same three values to the page envelope so the manifest builder can audit them without opening nested rankings.

- [ ] **Step 5: Replace feature-row ranking construction with an exact stored query**

Set:

```python
STATIC_SITE_SCHEMA_VERSION = "static-site-v3"
```

Add `rs_formula_version_overrides: Mapping[str, str] | None = None` and `feature_run_ids_by_market: Mapping[str, int] | None = None` to `StaticSiteExportService.export()` and its per-Market builder. Normal exports resolve the active formula and latest published feature pointer. Rollout exports can request balanced rows plus an exact balanced Feature run before activation.

In `_build_groups_payload()`:

1. Resolve exactly one formula and Feature run for the Market. An explicit Feature run must be published, Market-scoped, and record the requested formula/run in `config_json`.
2. Call version-aware `IBDGroupRankService.get_current_rankings()` at `market`, `expected_as_of_date`, and that explicit formula; its query targets stored `IBDGroupRank` rows and already supplies rank changes/top-symbol names.
3. Require non-empty rows, a single non-null `market_rs_run_id` for balanced mode, and a completed `MarketRsRun` whose Market/date/formula match both Group rows and Feature-run config.
4. Serialize current values with shared `group_ranking_payloads.rank_record_payload()`; do not import or call `compute_group_rankings_from_serialized_rows()`.
5. Load prior ranks/history from version-filtered `IBDGroupRank` rows, not historical feature rows.
6. Use current serialized stock rows only to populate stock-level detail cards; their `rs_formula_version` and `market_rs_run_id` must match the selected Group snapshot.
7. Build movers from the stored current rankings and stored rank-change fields.

Pass the selected formula into the static RRG export from Task 9. Put `rs_formula_version`, `rs_as_of_date`, and `rs_universe_size` in the Market manifest entry and reject fallback artifacts whose schema or formula differs from the requested Market artifact.

- [ ] **Step 6: Run static export tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_groups_payload_builder.py tests/unit/test_static_site_export_service.py tests/unit/test_static_groups_rrg_sources.py -q
```

Expected: all tests pass; source inspection in the test confirms `StaticSiteExportService` no longer references `compute_group_rankings_from_serialized_rows`.

- [ ] **Step 7: Commit static canonicalization**

```bash
git add backend/app/services/static_groups_payload_builder.py backend/app/services/static_site_export_service.py backend/tests/unit/test_static_groups_payload_builder.py backend/tests/unit/test_static_site_export_service.py backend/tests/unit/test_static_groups_rrg_sources.py
git commit -m "fix: export canonical Group RS snapshots"
```

---

### Task 11: Add Live/Static Group 1M and 3M Columns and Fix Date Rollover

**Files:**
- Modify: `frontend/src/pages/GroupRankingsPage.jsx`
- Modify: `frontend/src/pages/GroupRankingsPage.test.jsx`
- Modify: `frontend/src/static/pages/StaticGroupsPage.jsx`
- Modify: `frontend/src/static/pages/StaticGroupsPage.test.jsx`

**Interfaces:**
- Consumes: the additive Group fields and snapshot metadata from Tasks 8 and 10.
- Produces: visible 1M/3M RS columns in both applications and a live page that moves from an old pinned ranking date to a newly published date.

- [ ] **Step 1: Add failing live-table tests**

Extend the ranking fixture with `avg_rs_rating_1m: 38.25` and `avg_rs_rating_3m: 61.75`. Assert the table header order begins:

```text
Rank | Industry Group | RS | 1M RS | 3M RS | Med RS | Wtd RS
```

Assert `38.3` and `61.8` render, clicking `1M RS` sorts by `avg_rs_rating_1m`, and clicking `3M RS` sorts by `avg_rs_rating_3m`. Keep existing rank-change headers distinguishable as `1M Δ` and `3M Δ` so component RS is not confused with historical rank change.

- [ ] **Step 2: Add a failing bootstrap rollover test**

Use fake timers and return a bootstrap dated `2026-04-09`, followed by one dated `2026-04-10`. Advance 60 seconds and assert:

```javascript
expect(getGroupsBootstrap).toHaveBeenCalledTimes(2);
expect(await screen.findByText(/2026-04-10/)).toBeInTheDocument();
```

Switch to RRG and assert the API call is pinned to `2026-04-10`, not `2026-04-09`. In a separate test, complete a manual calculation task and assert `groupsBootstrap` is invalidated/refetched before any date-pinned ranking refetch.

- [ ] **Step 3: Add failing static-table tests**

Assert the static header order is:

```text
Rank | Group | Avg RS | 1M RS | 3M RS | Stocks | 1W | 1M | 3M | 6M | Top Stock
```

Assert missing component values render `-`, while zero renders `0.0`.

- [ ] **Step 4: Run frontend tests and verify failures**

Run:

```bash
cd frontend && npm run test:run -- GroupRankingsPage.test.jsx StaticGroupsPage.test.jsx
```

Expected: fail because neither table has component columns and bootstrap has no polling interval.

- [ ] **Step 5: Implement the two live sortable columns**

Insert `avg_rs_rating_1m` and `avg_rs_rating_3m` immediately after `avg_rs_rating` in both the header and row. Use the existing `TableSortLabel`, `handleSort`, and null-ordering behavior. Format only finite values:

```javascript
const formatRs = (value) => Number.isFinite(value) ? value.toFixed(1) : '-';
```

Use labels `1M RS` and `3M RS`; relabel only the existing rank-change headers to `1M Δ` and `3M Δ` without changing their values or sort keys.

- [ ] **Step 6: Make the bootstrap date advance while the page remains open**

Add `refetchInterval: 60_000` to `groupsBootstrapQuery`. Its existing fetch closure must continue seeding cache keys using the date from that exact response. When calculation status becomes completed, call an async helper from the effect:

```javascript
await queryClient.invalidateQueries({
  queryKey: ['groupsBootstrap', selectedMarket],
  refetchType: 'none',
});
await queryClient.refetchQueries({
  queryKey: ['groupsBootstrap', selectedMarket],
  type: 'active',
  exact: true,
});
await Promise.all([
  queryClient.invalidateQueries({ queryKey: ['groupRankings', selectedMarket] }),
  queryClient.invalidateQueries({ queryKey: ['groupMovers'] }),
  queryClient.invalidateQueries({ queryKey: ['groupRRGBundle', selectedMarket] }),
]);
```

Invoke the helper with `void refreshPublishedGroups()` and handle rejection by setting the existing calculation error. Only after the bootstrap refetch resolves may it invalidate `groupRankings`, `groupMovers`, and `groupRRGBundle`. Do not call the captured `refetchRankings()` first, because it still belongs to the old date key.

- [ ] **Step 7: Implement the two static display columns**

Insert `1M RS` and `3M RS` immediately after `Avg RS` and render them with the same `formatRs` helper. Do not add a new client-side sort model to the static table.

- [ ] **Step 8: Run focused frontend tests and lint the changed files**

Run:

```bash
cd frontend && npm run test:run -- GroupRankingsPage.test.jsx StaticGroupsPage.test.jsx
cd frontend && npm run lint -- src/pages/GroupRankingsPage.jsx src/static/pages/StaticGroupsPage.jsx
```

Expected: tests and lint pass.

- [ ] **Step 9: Commit both application surfaces**

```bash
git add frontend/src/pages/GroupRankingsPage.jsx frontend/src/pages/GroupRankingsPage.test.jsx frontend/src/static/pages/StaticGroupsPage.jsx frontend/src/static/pages/StaticGroupsPage.test.jsx
git commit -m "feat: show Group 1M and 3M RS"
```

---

### Task 12: Put Canonical RS in the Daily and Manual Execution Paths

**Files:**
- Create: `backend/app/tasks/market_rs_tasks.py`
- Modify: `backend/app/tasks/daily_market_pipeline_tasks.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Modify: `backend/app/celery_app.py`
- Create: `backend/tests/unit/test_market_rs_tasks.py`
- Modify: `backend/tests/unit/test_daily_market_pipeline_tasks.py`
- Modify: `backend/tests/unit/test_group_rank_tasks.py`
- Modify: `backend/tests/unit/test_celery_config.py`

**Interfaces:**
- Consumes: the snapshot and Group services from Tasks 4 and 7.
- Produces: an ordered, guarded daily chain and a manual Group path that both require the exact Market/date canonical input when balanced mode is active.

- [ ] **Step 1: Add failing task lifecycle tests**

For `calculate_market_rs_snapshot`, assert a successful result has this stable shape:

```python
{
    "status": "completed",
    "market": "US",
    "as_of_date": "2026-04-10",
    "formula_version": BALANCED_RS_FORMULA_VERSION,
    "market_rs_run_id": 42,
    "eligible_symbol_count": 5000,
}
```

Assert `MarketRsInputUnavailable` produces `status="failed"` with diagnostics. With balanced active, the daily guard raises and leaves later task signatures unexecuted; with legacy active, it records `stage="market_rs_shadow"` and allows the existing legacy Group/feature stages to continue.

- [ ] **Step 2: Add a failing daily-order test**

Update the expected signature names to this exact sequence:

```python
[
    "app.tasks.cache_tasks.smart_refresh_cache",
    "app.tasks.daily_market_pipeline_tasks.guard_price_refresh",
    "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
    "app.tasks.daily_market_pipeline_tasks.guard_market_rs_result",
    "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
    "app.tasks.daily_market_pipeline_tasks.guard_breadth_result",
    "app.tasks.breadth_tasks.calculate_market_exposure",
    "app.tasks.daily_market_pipeline_tasks.guard_exposure_result",
    "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
    "app.tasks.daily_market_pipeline_tasks.guard_group_result",
    "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
    "app.tasks.daily_market_pipeline_tasks.guard_snapshot_result",
]
```

Assert the RS, Group, and feature tasks all receive the identical ISO trading date and normalized Market.

- [ ] **Step 3: Add failing manual Group-path tests**

When the active formula is balanced, invoke `calculate_daily_group_rankings` directly and assert it first calls:

```python
market_rs_service.calculate(
    db,
    market="US",
    as_of_date=date(2026, 4, 10),
    formula_version=BALANCED_RS_FORMULA_VERSION,
)
```

Then assert Group calculation receives that formula/date. If canonical RS fails, Group calculation must not run. Under legacy active mode, assert the new snapshot service is not invoked.

- [ ] **Step 4: Run task tests and verify failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_tasks.py tests/unit/test_daily_market_pipeline_tasks.py tests/unit/test_group_rank_tasks.py -q
```

Expected: fail because the task, guard, and chain stage do not exist.

- [ ] **Step 5: Implement the idempotent Market RS Celery task**

Create `calculate_market_rs_snapshot(market: str, calculation_date: str, formula_version: str = BALANCED_RS_FORMULA_VERSION)`. It must:

1. Normalize and require an explicit non-shared Market.
2. Parse and validate the trading date with `MarketCalendarService`.
3. Open one `SessionLocal`, call the snapshot service, and close the session in `finally`.
4. Return completed-run metadata without activating the formula.
5. Return a structured failed result for canonical input/calculation errors; let transient database/connection errors use the task's bounded retry policy.

Register `app.tasks.market_rs_tasks` in Celery `include` and route `calculate_market_rs_snapshot` with `_MARKET_JOB_TASKS`.

- [ ] **Step 6: Insert and guard the daily stage**

Add `guard_market_rs_result`; when balanced is active it accepts only `status == "completed"`, the requested Market/date, `balanced-horizon-percentile-v2`, and a non-null run ID. When legacy is active, a completed shadow run is recorded normally and a failed shadow run logs diagnostics but does not block the legacy pipeline. Insert the immutable `.si()` snapshot task immediately after the price guard and the mutable `.s()` result guard immediately after it.

This task always computes balanced v2 shadow data. It does not change the active pointer, so legacy Markets continue using legacy Group/Scan behavior until Task 13 activates them.

- [ ] **Step 7: Protect manual and gapfill Group refreshes**

Before `IBDGroupRankService.calculate_group_rankings()` runs, resolve the explicit/active formula. For balanced mode, call `MarketRsSnapshotService.calculate()` for the exact calculation date and pass `formula_version` into Group calculation. Apply the same helper from daily, gapfill, and historical Group entry points; do not duplicate service-resolution logic.

- [ ] **Step 8: Run task and routing tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_tasks.py tests/unit/test_daily_market_pipeline_tasks.py tests/unit/test_group_rank_tasks.py tests/unit/test_celery_config.py -q
```

Expected: all tests pass and the active legacy pointer remains unchanged during shadow runs.

- [ ] **Step 9: Commit execution-path ordering**

```bash
git add backend/app/tasks/market_rs_tasks.py backend/app/tasks/daily_market_pipeline_tasks.py backend/app/tasks/group_rank_tasks.py backend/app/celery_app.py backend/tests/unit/test_market_rs_tasks.py backend/tests/unit/test_daily_market_pipeline_tasks.py backend/tests/unit/test_group_rank_tasks.py backend/tests/unit/test_celery_config.py
git commit -m "feat: schedule canonical Market RS before groups"
```

---

### Task 13: Backfill, Validate, Stage Static v3, and Activate Per Market

**Files:**
- Create: `backend/app/services/market_rs_rollout_service.py`
- Create: `backend/app/scripts/backfill_market_rs.py`
- Modify: `backend/app/domain/feature_store/ports.py`
- Modify: `backend/app/infra/db/repositories/feature_run_repo.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Create: `backend/tests/unit/test_market_rs_rollout_service.py`
- Create: `backend/tests/unit/test_backfill_market_rs_script.py`
- Modify: `backend/tests/unit/repositories/test_feature_run_repo.py`
- Modify: `backend/tests/unit/test_runtime_services.py`
- Create: `backend/tests/integration/test_market_rs_activation.py`

**Interfaces:**
- Consumes: explicit balanced stock/Group builders, feature snapshot override, static v3 exporter, and formula-isolated RRG builder.
- Produces: resumable full-history reports and one validated transaction that switches a Market's RS and feature pointers together.

- [ ] **Step 1: Add failing candidate-date and resumability tests**

Seed Market sessions with invalid early history, followed by the first date having 253 benchmark sessions, a reconstructable point-in-time universe, at least 90% current-price coverage, and at least two fully eligible stocks. Assert `candidate_dates()` starts on that first valid date and contains every Market session through the requested date.

Seed one completed balanced date, one failed date, and one absent date. Assert `backfill()` leaves the completed run unchanged, retries the failed run, calculates the absent run, and returns an ordered report with per-date stock and Group run IDs.

After the first valid date, missing point-in-time lifecycle, benchmark, price-coverage, stock snapshot, or expected Group output must be a recorded failure that blocks activation; it must not silently remove the date from the candidate set.

- [ ] **Step 2: Add failing activation-gate tests**

Build one test for each rejection:

- a candidate trading-date gap;
- stock rows not matching `eligible_symbol_count`;
- any rating outside 1–99 or a non-finite composite;
- Group rows absent for a mapped Group with at least three eligible stocks;
- mixed Group formula/run IDs or non-deterministic ranks;
- live/static Group field mismatch;
- RRG mixed-formula history, coordinate divergence, or an unexplained build failure;
- missing `static-site-v3` staged artifact;
- staged Feature run config that names a different canonical run/formula/date.

For every case, assert both `MarketRsFormulaPointer` and `FeatureRunPointer(key=f"latest_published_market:{market}")` retain their old values.

In `test_feature_run_repo.py`, first assert `repoint_published()` moves a named pointer to an already published candidate without changing its status/timestamps, and rejects a running/completed/quarantined candidate.

- [ ] **Step 3: Add a failing successful-activation integration test**

Record activation events and require this order:

```python
assert events == [
    "backfill_stock_history",
    "backfill_group_history",
    "validate_ranges_and_coverage",
    "build_balanced_feature_run",
    "build_balanced_rrg",
    "stage_static_site_v3",
    "validate_live_static_parity",
    "commit_market_and_feature_pointers",
    "publish_live_groups_bootstrap",
]
```

After activation, resolve the same symbol from full-Market, index, and watchlist feature consumers and assert identical overall/1M/3M/12M ratings. Assert the staged static Groups and RRG artifacts carry balanced v2 and the new static schema versions.

- [ ] **Step 4: Run rollout tests and verify missing service failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_rollout_service.py tests/unit/test_backfill_market_rs_script.py tests/integration/test_market_rs_activation.py -q
```

Expected: fail because the rollout service and operator script do not exist.

- [ ] **Step 5: Implement explicit rollout reports and the valid-history boundary**

Define immutable `BackfillDateResult`, `BackfillReport`, and `ActivationValidationReport`. Reports must include Market, formula, requested range, first valid date, candidate/completed/failed counts, failed dates with reason codes, latest run ID, Group-row counts, and validation errors.

`earliest_backfillable_date()` scans Market sessions in ascending order and selects the first date for which input probing proves all of these facts without writing a run:

1. point-in-time universe reconstruction succeeds;
2. exact 21/63/126/189/252 session anchors exist;
3. one benchmark candidate has every anchor;
4. current-price coverage is at least 90%; and
5. at least two stocks have every anchor.

Once that boundary is found, every session through `through_date` is a required candidate. A later input failure remains in the report and blocks activation.

Wire a lazy `market_rs_rollout_service` property and getter into `RuntimeServices`, with reset coverage, after the new service exists.

- [ ] **Step 6: Implement the resumable stock and Group backfill**

Process each candidate date as an isolated resumable unit:

```python
run = market_rs_snapshot_service.calculate(
    db,
    market=market,
    as_of_date=calculation_date,
    formula_version=BALANCED_RS_FORMULA_VERSION,
)
groups = canonical_group_ranking_service.calculate_and_store(
    db,
    market=market,
    as_of_date=calculation_date,
    formula_version=BALANCED_RS_FORMULA_VERSION,
)
```

Completed stock and Group rows are idempotent. If Group aggregation fails after the stock run has committed, retain that completed stock run so the next attempt resumes at Group aggregation. Roll back uncommitted work for the failing stage, record the date/reason, and continue so one operator run identifies the full repair set. Never activate from `backfill()` itself.

- [ ] **Step 7: Implement the activation validator**

`validate_activation()` must query the database again rather than trusting the backfill report. It must verify every gate from Step 2 across the full required range, then:

1. Build balanced Group live payloads from stored rows.
2. Build a balanced-only `static-rrg-history-v4` state and RRG payload. If all available valid history is shorter than the unchanged RRG warm-up windows, require both live and static outputs to report `insufficient_balanced_history` with no legacy coordinates; this explicit unavailable state is valid until enough balanced weeks accumulate.
3. Require a published Feature run for the through date whose config identifies the same balanced Market RS run.
4. Inspect the staged `static-site-v3` manifest, Groups artifact, Scan manifest/preview rows, and RRG artifact.
5. Compare live and static Group parity fields and a deterministic sample of stock overall/1M/3M/12M fields.

Return `ok=False` with every error found; do not stop at the first mismatch.

- [ ] **Step 8: Implement atomic pointer activation**

Expose:

```python
def activate(
    self,
    db: Session,
    *,
    market: str,
    formula_version: str,
    feature_run_id: int,
    validation: ActivationValidationReport,
) -> None:
    if not validation.ok:
        raise MarketRsActivationRejected(validation.errors)
    self.repository.activate_formula(
        db,
        market=market,
        formula_version=formula_version,
    )
    self.feature_run_repository.repoint_published(
        pointer_key=f"latest_published_market:{market}",
        run_id=feature_run_id,
    )
    db.commit()
```

Add `repoint_published(run_id: int, pointer_key: str) -> FeatureRunDomain` to the Feature-run port/repository. It must require an already `PUBLISHED` run and update only the pointer; unlike `publish_atomically`, it does not attempt a second status transition. Before mutation, re-read and revalidate the latest run IDs/hashes captured in the report to prevent time-of-check/time-of-use drift. Any exception rolls back both pointer changes. Keep legacy rows and the prior Feature run for rollback.

- [ ] **Step 9: Implement the operator CLI with a mandatory staged artifact**

Support:

```bash
cd backend && source venv/bin/activate
python -m app.scripts.backfill_market_rs \
  --market US \
  --through-date 2026-04-10 \
  --static-staging-dir /absolute/path/to/rs-v2-us \
  --activate
```

`--start-date` is an optional resume limiter for calculation only; activation validation still checks from the computed first valid date. `--activate` requires an empty/non-serving `--static-staging-dir` and performs this sequence:

1. backfill stock and Group history;
2. stop if any required date failed;
3. build a through-date Feature snapshot with `rs_formula_version_override=balanced-horizon-percentile-v2` and pointer key `rollout_rs:balanced-horizon-percentile-v2:{market}`;
4. export static v3 into the staging directory using both `feature_run_ids_by_market` and `rs_formula_version_overrides`;
5. run the complete validator;
6. atomically activate the Market and Feature pointers;
7. publish the live Groups bootstrap and invalidate Group caches.

Without `--activate`, print the JSON report and leave every active pointer unchanged. The CLI exits nonzero on any failed date or validation error and prints exact repair dates/reasons.

- [ ] **Step 10: Run rollout and regression tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_market_rs_rollout_service.py tests/unit/test_backfill_market_rs_script.py tests/unit/repositories/test_feature_run_repo.py tests/unit/test_runtime_services.py tests/integration/test_market_rs_activation.py tests/unit/test_static_site_export_service.py tests/unit/test_static_rrg_history_bundle.py tests/unit/test_ui_snapshot_service.py -q
```

Expected: all tests pass; a rejected activation changes neither pointer.

- [ ] **Step 11: Commit the guarded rollout tooling**

```bash
git add backend/app/services/market_rs_rollout_service.py backend/app/scripts/backfill_market_rs.py backend/app/domain/feature_store/ports.py backend/app/infra/db/repositories/feature_run_repo.py backend/app/wiring/bootstrap.py backend/tests/unit/test_market_rs_rollout_service.py backend/tests/unit/test_backfill_market_rs_script.py backend/tests/unit/repositories/test_feature_run_repo.py backend/tests/unit/test_runtime_services.py backend/tests/integration/test_market_rs_activation.py
git commit -m "feat: backfill and activate balanced Market RS"
```

---

### Task 14: Prove End-to-End Market, Scan, Group, Static, and RRG Parity

**Files:**
- Create: `backend/tests/parity/test_canonical_market_rs_parity.py`
- Create: `backend/tests/integration/test_canonical_market_rs_migration.py`
- Modify: `backend/tests/parity/golden_fixtures.py`
- Modify: `backend/tests/unit/golden/test_golden_rrg.py`

**Interfaces:**
- Consumes: the completed implementation and a deterministic Market/date fixture.
- Produces: one acceptance-level proof covering the design's cross-surface invariants and migration safety.

- [ ] **Step 1: Add the failing cross-surface acceptance fixture**

Seed at least two Groups with at least three eligible stocks each, one benchmark, all six anchor prices, active universe lifecycle events, classification mappings, and sufficient versioned Group history for RRG. Include a former winner with extreme 12M performance and severe 1M/3M weakness.

Calculate balanced RS and obtain the same symbols through:

- full-Market feature scan;
- index-subset scan;
- watchlist-subset scan;
- live Group rankings/detail;
- static Scan and Groups artifact; and
- live/static RRG payloads.

- [ ] **Step 2: Assert exact stock and Group parity**

For every overlapping stock, compare these exact fields without tolerance because stock ratings are integers:

```python
STOCK_RS_FIELDS = (
    "rs_rating",
    "rs_rating_1m",
    "rs_rating_3m",
    "rs_rating_12m",
    "rs_formula_version",
    "market_rs_run_id",
    "rs_universe_size",
)
```

For every Group, compare formula/run/count exactly and decimal averages with `pytest.approx`:

```python
GROUP_RS_FIELDS = (
    "rank",
    "avg_rs_rating",
    "avg_rs_rating_1m",
    "avg_rs_rating_3m",
    "num_stocks",
    "top_symbol",
    "rs_formula_version",
    "market_rs_run_id",
)
```

Assert Group rank order equals sorting by unrounded `avg_rs_rating` descending then `industry_group` ascending. Assert changing weighted/median/1M/3M fields while holding overall means fixed cannot change main Group ranks.

- [ ] **Step 3: Assert bounded-outlier and RRG invariants**

Assert the extreme stock's 12M magnitude can change from 1,000% to 10,000% without changing its P12, weighted composite, or overall rating when horizon ordering is unchanged. Assert recent weak P1/P3 jointly contribute exactly 50% of the composite weights.

Compare all live/static RRG coordinates and tails for identical versioned history. Preserve the existing golden coordinate numbers; only envelope/schema/formula fields may change.

- [ ] **Step 4: Add migration upgrade/downgrade coverage with legacy data**

Start at revision `20260701_0024`, insert representative legacy Group rows, upgrade to `20260718_0025`, and assert:

- legacy rows are labeled `legacy-linear-v1`;
- one pointer exists per represented Market;
- new Group component fields remain null for legacy rows;
- balanced and legacy rows can coexist for one Market/date;
- uniqueness rejects a duplicate within one formula; and
- downgrade removes balanced rows first and restores the prior schema without losing legacy rows.

- [ ] **Step 5: Run acceptance and migration tests and verify initial failures**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/parity/test_canonical_market_rs_parity.py tests/integration/test_canonical_market_rs_migration.py tests/unit/golden/test_golden_rrg.py -q
```

Expected before completing the fixture adapters: fail at the first missing parity or migration assertion.

- [ ] **Step 6: Complete only fixture/adaptor changes needed by the acceptance proof**

Do not introduce a new production formula or fallback in this task. If a parity failure exposes production divergence, return to the owning earlier task, add a focused regression test there, fix it, rerun that task's suite, and then rerun this acceptance suite.

- [ ] **Step 7: Run the complete parity and migration suites**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/parity tests/unit/test_scan_path_parity.py tests/unit/test_alembic_baseline.py tests/unit/test_main_migrations.py tests/integration/test_canonical_market_rs_migration.py -q
```

Expected: all parity, golden, and migration tests pass.

- [ ] **Step 8: Commit acceptance coverage**

```bash
git add backend/tests/parity/test_canonical_market_rs_parity.py backend/tests/integration/test_canonical_market_rs_migration.py backend/tests/parity/golden_fixtures.py backend/tests/unit/golden/test_golden_rrg.py
git commit -m "test: prove canonical RS cross-surface parity"
```

---

### Task 15: Update Methodology, Operations, and Run All Quality Gates

**Files:**
- Modify: `README.md`
- Modify: `docs/LIVE_APP_GUIDE.md`
- Modify: `docs/STATIC_SITE.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/release-checklist.md`

**Interfaces:**
- Consumes: the final formula, schemas, and rollout commands.
- Produces: user/operator documentation and the final verified delivery.

- [ ] **Step 1: Add a documentation contract check**

Before editing, run:

```bash
rg -n "3mo 40%|6/9/12mo 20%|scaled 0–100|static-site-v2|static-rrg-history-v3" README.md docs/LIVE_APP_GUIDE.md docs/STATIC_SITE.md docs/OPERATIONS.md docs/release-checklist.md
```

Expected: at least the old 3M/6M/9M/12M description is found and must be replaced, not left as a second active definition.

- [ ] **Step 2: Update the README and live application guide**

Document, in this order:

1. Market-wide same-set eligibility and adjusted-close session anchors.
2. Five excess-return horizons and weights: 1M 20%, 3M 30%, 6M 20%, 9M 15%, 12M 15%.
3. Independent per-horizon percentile ratings P1/P3/P6/P9/P12, followed by the final overall composite percentile.
4. Why a 1,000% raw return cannot dominate after percentile normalization.
5. Scan display fields: overall, 1M, 3M, and 12M.
6. Group overall/1M/3M as equal constituent averages over the same eligible set, with Group Rank based only on overall.
7. The live/static 1M RS and 3M RS columns.
8. The explicit statement that this is IBD/CANSLIM-inspired but not IBD's undisclosed proprietary calculation.
9. Formula/date/universe metadata and the unchanged RRG transformation.
10. The rollout expectation that stock and Group ranks will change, especially for former long-term winners with weak recent performance, and that rank-change/RRG comparisons must stay within one formula version.

- [ ] **Step 3: Update static and operator documentation**

In `docs/STATIC_SITE.md`, document `static-site-v3`, `static-rrg-history-v4`, exact stored Group export, formula isolation, new fields, and fallback-artifact rejection.

In `docs/OPERATIONS.md` and `docs/release-checklist.md`, document:

- shadow backfill and report inspection;
- the mandatory non-serving static staging directory;
- validation gates and nonzero failure behavior;
- per-Market activation command;
- live bootstrap verification;
- static artifact promotion/deployment using the project's existing release procedure; and
- rollback by restoring both the legacy formula pointer and its retained legacy Feature-run pointer, followed by bootstrap/static regeneration.

- [ ] **Step 4: Verify documentation contains one canonical definition**

Run:

```bash
rg -n "balanced-horizon-percentile-v2|1M 20%|3M 30%|Group 1M RS|static-site-v3|static-rrg-history-v4" README.md docs/LIVE_APP_GUIDE.md docs/STATIC_SITE.md docs/OPERATIONS.md docs/release-checklist.md
rg -n "3mo 40%|6/9/12mo 20%|scaled 0–100" README.md docs/LIVE_APP_GUIDE.md docs/STATIC_SITE.md docs/OPERATIONS.md docs/release-checklist.md
```

Expected: the first command finds the new contract; the second returns no active-methodology matches.

- [ ] **Step 5: Run all backend quality gates**

Run:

```bash
cd backend && source venv/bin/activate && pytest
```

Expected: full backend suite passes.

- [ ] **Step 6: Run all frontend quality gates**

Run:

```bash
cd frontend && npm run test:run
cd frontend && npm run lint
cd frontend && npm run build
```

Expected: tests, ESLint, and production build pass.

- [ ] **Step 7: Check migration head, formatting, and unintended changes**

Run:

```bash
cd backend && source venv/bin/activate && alembic heads
cd .. && git diff --check
git status --short
```

Expected: one Alembic head at `20260718_0025`, no whitespace errors, and only intentional task files plus pre-existing user changes.

- [ ] **Step 8: Commit documentation**

```bash
git add README.md docs/LIVE_APP_GUIDE.md docs/STATIC_SITE.md docs/OPERATIONS.md docs/release-checklist.md
git commit -m "docs: define canonical balanced Market RS"
```

- [ ] **Step 9: Land the completed branch per repository policy**

Run `bd close` for the claimed implementation issue, then:

```bash
git pull --rebase
bd sync
git push
git status
```

Expected: the implementation branch is clean and reports up to date with its remote. If `bd` is unavailable in the environment, record that tool failure in the handoff, but still commit and push all implementation changes.
