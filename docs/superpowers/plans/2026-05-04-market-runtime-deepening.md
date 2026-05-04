# Market Runtime Deepening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the Market runtime architecture by introducing a backend-owned Market Catalog, a Bootstrap Plan module, and a Bootstrap Readiness module.

**Architecture:** Market Catalog owns stable Market facts only. Bootstrap Plan owns the ordered per-Market Bootstrap workflow and exposes a small interface that Celery adapts into task signatures. Bootstrap Readiness owns the rule for when enabled Markets have enough Universe, price, fundamentals, and auto Scan state for the runtime to become ready. Runtime Preferences remains focused on mutable primary/enabled Market choices and persisted Bootstrap state; Market Workload remains focused on activity/progress state.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, SQLAlchemy, Celery, pytest, React 18, TanStack Query, Material UI, Vitest, React Testing Library.

---

## Scope

This plan implements the Market runtime track:

- Add a deep Market Catalog module for stable Market facts.
- Expose Market Catalog through app capabilities while preserving `supported_markets`.
- Move Bootstrap task construction into a pure Bootstrap Plan module.
- Move Bootstrap readiness checks out of Runtime Preferences and task code.
- Keep existing public HTTP payloads backward-compatible.
- Keep queue naming and live progress in Market Workload modules.

This plan does not migrate Scan admission, Scan Session, static export parity, provider routing, or cache policy. Those are separate tracks.

## File Structure

- Create `backend/app/domain/markets/__init__.py`: exports Market Catalog interface.
- Create `backend/app/domain/markets/catalog.py`: stable Market facts and frontend-ready payload serialization.
- Create `backend/app/domain/bootstrap/__init__.py`: exports Bootstrap Plan and Bootstrap Readiness interfaces.
- Create `backend/app/domain/bootstrap/plan.py`: pure Bootstrap stage plan for a primary Market and enabled Markets.
- Create `backend/app/services/bootstrap_readiness_service.py`: SQL-backed adapter that evaluates Bootstrap readiness for enabled Markets.
- Modify `backend/app/tasks/runtime_bootstrap_tasks.py`: adapt Bootstrap Plan stages into Celery signatures; ask Bootstrap Readiness for completion checks.
- Modify `backend/app/services/runtime_preferences_service.py`: keep Runtime Preferences persistence; delegate readiness evaluation.
- Modify `backend/app/services/market_activity_service.py`: read Bootstrap state from Runtime Preferences plus Bootstrap Readiness result without owning readiness rules.
- Modify `backend/app/schemas/app_runtime.py`: add Market Catalog response schemas.
- Modify `backend/app/api/v1/app_runtime.py`: include Market Catalog in app capabilities and derive compatibility `supported_markets` from it.
- Modify `frontend/src/contexts/RuntimeContext.jsx`: normalize `market_catalog`, expose `marketCatalog`, and keep legacy `supportedMarkets`.
- Modify `frontend/src/components/App/BootstrapSetupScreen.jsx`: render Market labels from Market Catalog and keep codes as submitted values.
- Test `backend/tests/unit/domain/test_market_catalog.py`: Market Catalog shape and lookup.
- Test `backend/tests/unit/domain/test_bootstrap_plan.py`: Bootstrap stage order and per-Market variations.
- Test `backend/tests/unit/test_bootstrap_readiness_service.py`: readiness rules through SQL-backed adapter.
- Test `backend/tests/unit/test_runtime_preferences_service.py`: Runtime Preferences delegates readiness.
- Test `backend/tests/unit/test_runtime_bootstrap_tasks.py`: Celery task adapter uses Bootstrap Plan.
- Test `backend/tests/unit/test_app_runtime_endpoints.py`: app capabilities payload includes Market Catalog and compatibility field.
- Test `frontend/src/contexts/RuntimeContext.test.jsx`: Market Catalog normalization.
- Test `frontend/src/components/App/BootstrapSetupScreen.test.jsx`: Market labels render while codes are submitted.

## Domain Rules

- **Market Catalog** owns code, label, currency, timezone, calendar ID, exchanges, indexes, and coarse capabilities.
- **Runtime Preferences** owns primary Market, enabled Markets, and persisted Bootstrap state.
- **Bootstrap** owns the first-run hydration workflow.
- **Bootstrap Plan** owns stage order, stage keys, queue kind, US-only stages, universe names, publish pointer keys, and cache-only snapshot policy.
- **Bootstrap Readiness** owns the ready/required decision for enabled Markets.
- **Market Workload** owns activity progress, stage labels, active holders, and queue/lease state.
- `SHARED` remains a Market Workload sentinel, not a Market Catalog entry.

## Task 1: Add Market Catalog

**Files:**
- Create: `backend/app/domain/markets/__init__.py`
- Create: `backend/app/domain/markets/catalog.py`
- Test: `backend/tests/unit/domain/test_market_catalog.py`

- [ ] **Step 1: Write failing Market Catalog tests**

Create `backend/tests/unit/domain/test_market_catalog.py`:

```python
from __future__ import annotations

import pytest

from app.domain.markets.catalog import MarketCatalogError, get_market_catalog


def test_market_catalog_lists_supported_markets_in_runtime_order() -> None:
    catalog = get_market_catalog()

    assert catalog.supported_market_codes() == ["US", "HK", "IN", "JP", "KR", "TW", "CN"]


def test_market_catalog_entry_contains_stable_market_facts() -> None:
    catalog = get_market_catalog()

    hk = catalog.get("hk")

    assert hk.code == "HK"
    assert hk.label == "Hong Kong"
    assert hk.currency == "HKD"
    assert hk.timezone == "Asia/Hong_Kong"
    assert hk.calendar_id == "XHKG"
    assert hk.exchanges == ("HKEX", "SEHK", "XHKG")
    assert hk.indexes == ("HSI",)
    assert hk.capabilities.official_universe is True
    assert hk.capabilities.finviz_screening is False


def test_market_catalog_rejects_unknown_market() -> None:
    catalog = get_market_catalog()

    with pytest.raises(MarketCatalogError, match="Unsupported market 'EU'"):
        catalog.get("EU")


def test_market_catalog_runtime_payload_is_frontend_ready() -> None:
    payload = get_market_catalog().as_runtime_payload()

    assert payload["version"] == "2026-05-04.v1"
    assert [market["code"] for market in payload["markets"]] == [
        "US",
        "HK",
        "IN",
        "JP",
        "KR",
        "TW",
        "CN",
    ]
    assert payload["markets"][0] == {
        "code": "US",
        "label": "United States",
        "currency": "USD",
        "timezone": "America/New_York",
        "calendar_id": "XNYS",
        "exchanges": ["NYSE", "NASDAQ", "AMEX"],
        "indexes": ["SP500"],
        "capabilities": {
            "benchmark": True,
            "breadth": True,
            "fundamentals": True,
            "group_rankings": True,
            "feature_snapshot": True,
            "official_universe": False,
            "finviz_screening": True,
        },
    }
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/domain/test_market_catalog.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.domain.markets'`.

- [ ] **Step 3: Implement Market Catalog**

Create `backend/app/domain/markets/__init__.py`:

```python
"""Market domain modules."""

from .catalog import (
    MARKET_CATALOG,
    MarketCapabilities,
    MarketCatalog,
    MarketCatalogEntry,
    MarketCatalogError,
    get_market_catalog,
)

__all__ = [
    "MARKET_CATALOG",
    "MarketCapabilities",
    "MarketCatalog",
    "MarketCatalogEntry",
    "MarketCatalogError",
    "get_market_catalog",
]
```

Create `backend/app/domain/markets/catalog.py`:

```python
"""Stable Market Catalog facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

CATALOG_VERSION = "2026-05-04.v1"


class MarketCatalogError(ValueError):
    """Raised when a caller asks for an unsupported Market."""


@dataclass(frozen=True)
class MarketCapabilities:
    benchmark: bool
    breadth: bool
    fundamentals: bool
    group_rankings: bool
    feature_snapshot: bool
    official_universe: bool
    finviz_screening: bool


@dataclass(frozen=True)
class MarketCatalogEntry:
    code: str
    label: str
    currency: str
    timezone: str
    calendar_id: str
    exchanges: tuple[str, ...]
    indexes: tuple[str, ...]
    capabilities: MarketCapabilities

    def as_runtime_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exchanges"] = list(self.exchanges)
        payload["indexes"] = list(self.indexes)
        return payload


class MarketCatalog:
    """Stable Market facts; Runtime Preferences and Market Workload live elsewhere."""

    def __init__(self, entries: Iterable[MarketCatalogEntry]) -> None:
        self._entries = tuple(entries)
        self._by_code = {entry.code: entry for entry in self._entries}

    def supported_market_codes(self) -> list[str]:
        return [entry.code for entry in self._entries]

    def get(self, market: str | None) -> MarketCatalogEntry:
        code = (market or "").strip().upper()
        try:
            return self._by_code[code]
        except KeyError as exc:
            supported = ", ".join(self.supported_market_codes())
            raise MarketCatalogError(
                f"Unsupported market {market!r}. Supported: {supported}"
            ) from exc

    def as_runtime_payload(self) -> dict[str, object]:
        return {
            "version": CATALOG_VERSION,
            "markets": [entry.as_runtime_payload() for entry in self._entries],
        }


FULL_CAPABILITIES = MarketCapabilities(
    benchmark=True,
    breadth=True,
    fundamentals=True,
    group_rankings=True,
    feature_snapshot=True,
    official_universe=True,
    finviz_screening=False,
)

MARKET_CATALOG = MarketCatalog(
    [
        MarketCatalogEntry(
            code="US",
            label="United States",
            currency="USD",
            timezone="America/New_York",
            calendar_id="XNYS",
            exchanges=("NYSE", "NASDAQ", "AMEX"),
            indexes=("SP500",),
            capabilities=MarketCapabilities(
                benchmark=True,
                breadth=True,
                fundamentals=True,
                group_rankings=True,
                feature_snapshot=True,
                official_universe=False,
                finviz_screening=True,
            ),
        ),
        MarketCatalogEntry("HK", "Hong Kong", "HKD", "Asia/Hong_Kong", "XHKG", ("HKEX", "SEHK", "XHKG"), ("HSI",), FULL_CAPABILITIES),
        MarketCatalogEntry("IN", "India", "INR", "Asia/Kolkata", "XNSE", ("NSE", "XNSE", "BSE", "XBOM"), (), FULL_CAPABILITIES),
        MarketCatalogEntry("JP", "Japan", "JPY", "Asia/Tokyo", "XTKS", ("TSE", "JPX", "XTKS"), ("NIKKEI225",), FULL_CAPABILITIES),
        MarketCatalogEntry("KR", "South Korea", "KRW", "Asia/Seoul", "XKRX", ("KOSPI", "KOSDAQ", "KRX", "XKRX"), (), FULL_CAPABILITIES),
        MarketCatalogEntry("TW", "Taiwan", "TWD", "Asia/Taipei", "XTAI", ("TWSE", "TPEX", "XTAI"), ("TAIEX",), FULL_CAPABILITIES),
        MarketCatalogEntry("CN", "China A-shares", "CNY", "Asia/Shanghai", "XSHG", ("SSE", "SZSE", "BJSE", "XSHG", "XSHE", "XBSE"), (), FULL_CAPABILITIES),
    ]
)


def get_market_catalog() -> MarketCatalog:
    return MARKET_CATALOG
```

- [ ] **Step 4: Verify Market Catalog tests pass**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/domain/test_market_catalog.py -v
```

Expected: PASS.

## Task 2: Expose Market Catalog Through Runtime Capabilities

**Files:**
- Modify: `backend/app/schemas/app_runtime.py`
- Modify: `backend/app/api/v1/app_runtime.py`
- Test: `backend/tests/unit/test_app_runtime_endpoints.py`

- [ ] **Step 1: Add failing app capabilities assertion**

In `backend/tests/unit/test_app_runtime_endpoints.py`, extend the existing app capabilities test to assert:

```python
assert data["supported_markets"] == ["US", "HK", "IN", "JP", "KR", "TW", "CN"]
assert data["market_catalog"]["version"] == "2026-05-04.v1"
assert data["market_catalog"]["markets"][0]["code"] == "US"
assert data["market_catalog"]["markets"][0]["label"] == "United States"
assert data["market_catalog"]["markets"][1]["code"] == "HK"
assert data["market_catalog"]["markets"][1]["capabilities"]["finviz_screening"] is False
```

- [ ] **Step 2: Run endpoint test and verify failure**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/test_app_runtime_endpoints.py -v
```

Expected: FAIL with `KeyError: 'market_catalog'`.

- [ ] **Step 3: Add schema models**

In `backend/app/schemas/app_runtime.py`, add:

```python
class MarketCapabilitiesResponse(BaseModel):
    benchmark: bool = False
    breadth: bool = False
    fundamentals: bool = False
    group_rankings: bool = False
    feature_snapshot: bool = False
    official_universe: bool = False
    finviz_screening: bool = False


class MarketCatalogEntryResponse(BaseModel):
    code: str
    label: str
    currency: str
    timezone: str
    calendar_id: str
    exchanges: list[str] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=list)
    capabilities: MarketCapabilitiesResponse


class MarketCatalogResponse(BaseModel):
    version: str
    markets: list[MarketCatalogEntryResponse] = Field(default_factory=list)
```

Add this field to `AppCapabilitiesResponse`:

```python
market_catalog: MarketCatalogResponse
```

- [ ] **Step 4: Populate app capabilities from Market Catalog**

In `backend/app/api/v1/app_runtime.py`, import `get_market_catalog` and update `get_app_capabilities`:

```python
from ...domain.markets.catalog import get_market_catalog
```

Inside the endpoint:

```python
market_catalog = get_market_catalog()
```

Set response fields:

```python
supported_markets=market_catalog.supported_market_codes(),
market_catalog=market_catalog.as_runtime_payload(),
```

- [ ] **Step 5: Verify endpoint tests pass**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/test_app_runtime_endpoints.py tests/unit/domain/test_market_catalog.py -v
```

Expected: PASS.

## Task 3: Add Bootstrap Plan Module

**Files:**
- Create: `backend/app/domain/bootstrap/__init__.py`
- Create: `backend/app/domain/bootstrap/plan.py`
- Test: `backend/tests/unit/domain/test_bootstrap_plan.py`

- [ ] **Step 1: Write failing Bootstrap Plan tests**

Create `backend/tests/unit/domain/test_bootstrap_plan.py`:

```python
from __future__ import annotations

from app.domain.bootstrap.plan import BootstrapQueueKind, build_bootstrap_plan


def test_us_bootstrap_plan_includes_us_only_industry_group_seed() -> None:
    plan = build_bootstrap_plan(primary_market="US", enabled_markets=["US"])

    assert [stage.key for stage in plan.market_plans[0].stages] == [
        "universe",
        "industry_groups",
        "prices",
        "fundamentals",
        "breadth",
        "groups",
        "snapshot",
    ]
    assert plan.market_plans[0].stages[1].queue_kind == BootstrapQueueKind.MARKET_JOBS


def test_non_us_bootstrap_plan_uses_official_universe_without_industry_seed() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["HK", "US"])
    hk_plan = plan.market_plans[0]

    assert hk_plan.market == "HK"
    assert [stage.task_name for stage in hk_plan.stages] == [
        "refresh_official_market_universe",
        "smart_refresh_cache",
        "refresh_all_fundamentals",
        "calculate_daily_breadth_with_gapfill",
        "calculate_daily_group_rankings_with_gapfill",
        "build_daily_snapshot",
    ]
    assert hk_plan.stages[-1].kwargs == {
        "market": "HK",
        "universe_name": "market:HK",
        "publish_pointer_key": "latest_published_market:HK",
        "activity_lifecycle": "bootstrap",
        "bootstrap_cache_only_if_covered": True,
    }


def test_bootstrap_plan_deduplicates_primary_and_enabled_markets_in_order() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["US", "HK", "US"])

    assert [market_plan.market for market_plan in plan.market_plans] == ["HK", "US"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/domain/test_bootstrap_plan.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.domain.bootstrap'`.

- [ ] **Step 3: Implement Bootstrap Plan dataclasses**

Create `backend/app/domain/bootstrap/__init__.py`:

```python
"""Bootstrap domain modules."""

from .plan import (
    BootstrapPlan,
    BootstrapQueueKind,
    BootstrapStage,
    MarketBootstrapPlan,
    build_bootstrap_plan,
)

__all__ = [
    "BootstrapPlan",
    "BootstrapQueueKind",
    "BootstrapStage",
    "MarketBootstrapPlan",
    "build_bootstrap_plan",
]
```

Create `backend/app/domain/bootstrap/plan.py`:

```python
"""Pure Bootstrap workflow plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from app.domain.markets.catalog import get_market_catalog


class BootstrapQueueKind(str, Enum):
    DATA_FETCH = "data_fetch"
    MARKET_JOBS = "market_jobs"


@dataclass(frozen=True)
class BootstrapStage:
    key: str
    task_name: str
    queue_kind: BootstrapQueueKind
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class MarketBootstrapPlan:
    market: str
    stages: tuple[BootstrapStage, ...]


@dataclass(frozen=True)
class BootstrapPlan:
    primary_market: str
    enabled_markets: tuple[str, ...]
    market_plans: tuple[MarketBootstrapPlan, ...]


def _normalize_markets(primary_market: str, enabled_markets: Iterable[str]) -> tuple[str, ...]:
    catalog = get_market_catalog()
    primary = catalog.get(primary_market).code
    ordered: list[str] = []
    for raw_market in [primary, *list(enabled_markets)]:
        code = catalog.get(raw_market).code
        if code not in ordered:
            ordered.append(code)
    return tuple(ordered)


def _stage(
    *,
    key: str,
    task_name: str,
    queue_kind: BootstrapQueueKind,
    market: str,
    **kwargs: Any,
) -> BootstrapStage:
    payload = {"market": market, "activity_lifecycle": "bootstrap", **kwargs}
    return BootstrapStage(key=key, task_name=task_name, queue_kind=queue_kind, kwargs=payload)


def _build_market_plan(market: str) -> MarketBootstrapPlan:
    stages: list[BootstrapStage] = []
    stages.append(
        _stage(
            key="universe",
            task_name="refresh_stock_universe" if market == "US" else "refresh_official_market_universe",
            queue_kind=BootstrapQueueKind.DATA_FETCH,
            market=market,
        )
    )
    if market == "US":
        stages.append(
            _stage(
                key="industry_groups",
                task_name="load_tracked_ibd_industry_groups",
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            )
        )
    stages.extend(
        [
            _stage(key="prices", task_name="smart_refresh_cache", queue_kind=BootstrapQueueKind.DATA_FETCH, market=market, mode="full"),
            _stage(key="fundamentals", task_name="refresh_all_fundamentals", queue_kind=BootstrapQueueKind.DATA_FETCH, market=market),
            _stage(key="breadth", task_name="calculate_daily_breadth_with_gapfill", queue_kind=BootstrapQueueKind.MARKET_JOBS, market=market),
            _stage(key="groups", task_name="calculate_daily_group_rankings_with_gapfill", queue_kind=BootstrapQueueKind.MARKET_JOBS, market=market),
            _stage(
                key="snapshot",
                task_name="build_daily_snapshot",
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
                universe_name=f"market:{market}",
                publish_pointer_key=f"latest_published_market:{market}",
                bootstrap_cache_only_if_covered=True,
            ),
        ]
    )
    return MarketBootstrapPlan(market=market, stages=tuple(stages))


def build_bootstrap_plan(*, primary_market: str, enabled_markets: Iterable[str]) -> BootstrapPlan:
    markets = _normalize_markets(primary_market, enabled_markets)
    return BootstrapPlan(
        primary_market=markets[0],
        enabled_markets=markets,
        market_plans=tuple(_build_market_plan(market) for market in markets),
    )
```

- [ ] **Step 4: Verify Bootstrap Plan tests pass**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/domain/test_bootstrap_plan.py tests/unit/domain/test_market_catalog.py -v
```

Expected: PASS.

## Task 4: Adapt Runtime Bootstrap Tasks to Bootstrap Plan

**Files:**
- Modify: `backend/app/tasks/runtime_bootstrap_tasks.py`
- Test: `backend/tests/unit/test_runtime_bootstrap_tasks.py`

- [ ] **Step 1: Add failing task adapter assertion**

In `backend/tests/unit/test_runtime_bootstrap_tasks.py`, update the existing signature-building tests to assert task names and queue kinds come from `build_bootstrap_plan`. Add a focused test:

```python
def test_runtime_bootstrap_signatures_follow_bootstrap_plan(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    signatures = module._build_market_bootstrap_signatures("HK")

    assert [signature.task for signature in signatures] == [
        "app.tasks.universe_tasks.refresh_official_market_universe",
        "app.tasks.cache_tasks.smart_refresh_cache",
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
    ]
    assert signatures[-1].kwargs["publish_pointer_key"] == "latest_published_market:HK"
```

- [ ] **Step 2: Run runtime bootstrap task tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/test_runtime_bootstrap_tasks.py -v
```

Expected: tests should pass before refactor; keep this as a characterization gate.

- [ ] **Step 3: Replace hardcoded stage construction with adapter**

In `backend/app/tasks/runtime_bootstrap_tasks.py`, import:

```python
from ..domain.bootstrap.plan import BootstrapQueueKind, build_bootstrap_plan
```

Add:

```python
def _queue_for_stage(stage) -> str:
    if stage.queue_kind == BootstrapQueueKind.DATA_FETCH:
        return data_fetch_queue_for_market(stage.kwargs["market"])
    if stage.queue_kind == BootstrapQueueKind.MARKET_JOBS:
        return market_jobs_queue_for_market(stage.kwargs["market"])
    raise ValueError(f"Unsupported bootstrap queue kind: {stage.queue_kind}")
```

Replace `_build_market_bootstrap_signatures` implementation so it builds a one-market plan and maps stage `task_name` to the existing Celery task objects:

```python
task_by_name = {
    "refresh_stock_universe": refresh_stock_universe,
    "refresh_official_market_universe": refresh_official_market_universe,
    "load_tracked_ibd_industry_groups": load_tracked_ibd_industry_groups,
    "smart_refresh_cache": smart_refresh_cache,
    "refresh_all_fundamentals": refresh_all_fundamentals,
    "calculate_daily_breadth_with_gapfill": calculate_daily_breadth_with_gapfill,
    "calculate_daily_group_rankings_with_gapfill": calculate_daily_group_rankings_with_gapfill,
    "build_daily_snapshot": build_daily_snapshot,
}
```

For each stage:

```python
task_by_name[stage.task_name].si(**stage.kwargs).set(queue=_queue_for_stage(stage))
```

Update `queue_local_runtime_bootstrap` to call:

```python
plan = build_bootstrap_plan(primary_market=primary_market, enabled_markets=enabled_markets)
```

Use `plan.primary_market` and `list(plan.enabled_markets)` for completion payloads and logging.

- [ ] **Step 4: Verify adapter tests pass**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/test_runtime_bootstrap_tasks.py tests/unit/domain/test_bootstrap_plan.py -v
```

Expected: PASS.

## Task 5: Add Bootstrap Readiness Service

**Files:**
- Create: `backend/app/services/bootstrap_readiness_service.py`
- Test: `backend/tests/unit/test_bootstrap_readiness_service.py`

- [ ] **Step 1: Write failing readiness tests**

Create `backend/tests/unit/test_bootstrap_readiness_service.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from app.services.bootstrap_readiness_service import BootstrapReadinessService


class FakeBootstrapReadinessService(BootstrapReadinessService):
    def __init__(self, *, core_ready: dict[str, bool], scan_ready: dict[str, bool], empty: bool = False):
        self.core_ready = core_ready
        self.scan_ready = scan_ready
        self.empty = empty

    def is_empty_system(self, db) -> bool:
        return self.empty

    def has_core_market_data(self, db, market: str) -> bool:
        return self.core_ready.get(market, False)

    def has_completed_auto_scan(self, db, market: str) -> bool:
        return self.scan_ready.get(market, False)


def test_readiness_requires_core_data_and_auto_scan_for_every_enabled_market() -> None:
    service = FakeBootstrapReadinessService(
        core_ready={"US": True, "HK": True},
        scan_ready={"US": True, "HK": False},
    )

    result = service.evaluate(object(), enabled_markets=["US", "HK"])

    assert result.ready is False
    assert result.missing_markets == ["HK"]
    assert result.market_results["HK"].core_ready is True
    assert result.market_results["HK"].scan_ready is False


def test_readiness_is_ready_when_every_enabled_market_is_complete() -> None:
    service = FakeBootstrapReadinessService(
        core_ready={"US": True, "HK": True},
        scan_ready={"US": True, "HK": True},
    )

    result = service.evaluate(object(), enabled_markets=["US", "HK"])

    assert result.ready is True
    assert result.missing_markets == []


def test_empty_system_is_reported_independently_from_market_readiness() -> None:
    service = FakeBootstrapReadinessService(core_ready={}, scan_ready={}, empty=True)

    result = service.evaluate(object(), enabled_markets=["US"])

    assert result.empty_system is True
    assert result.ready is False
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/test_bootstrap_readiness_service.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.bootstrap_readiness_service'`.

- [ ] **Step 3: Implement readiness result and service**

Create `backend/app/services/bootstrap_readiness_service.py`:

```python
"""Bootstrap readiness evaluation for enabled Markets."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..infra.db.models.feature_store import FeatureRun
from ..models.scan_result import SCAN_TRIGGER_SOURCE_AUTO, Scan
from ..models.stock import StockFundamental, StockPrice
from ..models.stock_universe import StockUniverse
from ..domain.markets.catalog import get_market_catalog


@dataclass(frozen=True)
class MarketBootstrapReadiness:
    market: str
    core_ready: bool
    scan_ready: bool

    @property
    def ready(self) -> bool:
        return self.core_ready and self.scan_ready


@dataclass(frozen=True)
class BootstrapReadiness:
    empty_system: bool
    market_results: dict[str, MarketBootstrapReadiness]

    @property
    def ready(self) -> bool:
        return all(result.ready for result in self.market_results.values())

    @property
    def missing_markets(self) -> list[str]:
        return [market for market, result in self.market_results.items() if not result.ready]


class BootstrapReadinessService:
    def normalize_market(self, market: str) -> str:
        return get_market_catalog().get(market).code

    def is_empty_system(self, db: Session) -> bool:
        return not (
            self._has_active_universe_rows(db)
            or self._has_price_rows(db)
            or self._has_fundamental_rows(db)
        )

    def has_core_market_data(self, db: Session, market: str) -> bool:
        return (
            self._has_active_universe_rows(db, market)
            and self._has_price_rows(db, market)
            and self._has_fundamental_rows(db, market)
        )

    def has_completed_auto_scan(self, db: Session, market: str) -> bool:
        normalized_market = self.normalize_market(market)
        return (
            db.query(Scan.id)
            .join(FeatureRun, FeatureRun.id == Scan.feature_run_id)
            .filter(
                Scan.universe_market == normalized_market,
                Scan.status == "completed",
                Scan.trigger_source == SCAN_TRIGGER_SOURCE_AUTO,
                FeatureRun.status == "published",
            )
            .limit(1)
            .first()
            is not None
        )

    def evaluate(self, db: Session, *, enabled_markets: list[str]) -> BootstrapReadiness:
        normalized_markets = [self.normalize_market(market) for market in enabled_markets]
        return BootstrapReadiness(
            empty_system=self.is_empty_system(db),
            market_results={
                market: MarketBootstrapReadiness(
                    market=market,
                    core_ready=self.has_core_market_data(db, market),
                    scan_ready=self.has_completed_auto_scan(db, market),
                )
                for market in normalized_markets
            },
        )

    def _has_active_universe_rows(self, db: Session, market: str | None = None) -> bool:
        query = db.query(StockUniverse.id).filter(StockUniverse.is_active.is_(True))
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None

    def _has_price_rows(self, db: Session, market: str | None = None) -> bool:
        query = (
            db.query(StockPrice.id)
            .join(StockUniverse, StockUniverse.symbol == StockPrice.symbol)
            .filter(StockUniverse.is_active.is_(True))
        )
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None

    def _has_fundamental_rows(self, db: Session, market: str | None = None) -> bool:
        query = (
            db.query(StockFundamental.id)
            .join(StockUniverse, StockUniverse.symbol == StockFundamental.symbol)
            .filter(StockUniverse.is_active.is_(True))
        )
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None
```

- [ ] **Step 4: Verify readiness tests pass**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/test_bootstrap_readiness_service.py -v
```

Expected: PASS.

## Task 6: Delegate Runtime Preferences Readiness

**Files:**
- Modify: `backend/app/services/runtime_preferences_service.py`
- Modify: `backend/app/tasks/runtime_bootstrap_tasks.py`
- Test: `backend/tests/unit/test_runtime_preferences_service.py`
- Test: `backend/tests/unit/test_runtime_bootstrap_tasks.py`

- [ ] **Step 1: Add failing delegation test**

In `backend/tests/unit/test_runtime_preferences_service.py`, add:

```python
def test_bootstrap_status_uses_readiness_service(monkeypatch):
    from app.services import runtime_preferences_service as module

    prefs = module.RuntimePreferences(
        primary_market="US",
        enabled_markets=["US", "HK"],
        bootstrap_state="running",
    )
    readiness = module.BootstrapReadiness(
        empty_system=False,
        market_results={
            "US": module.MarketBootstrapReadiness("US", core_ready=True, scan_ready=True),
            "HK": module.MarketBootstrapReadiness("HK", core_ready=True, scan_ready=False),
        },
    )

    monkeypatch.setattr(module, "get_runtime_preferences", lambda _db: prefs)
    monkeypatch.setattr(module, "get_bootstrap_readiness_service", lambda: type("Fake", (), {"evaluate": lambda self, db, enabled_markets: readiness})())

    status = module.get_runtime_bootstrap_status(object())

    assert status.empty_system is False
    assert status.bootstrap_required is True
    assert status.bootstrap_state == "running"
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest tests/unit/test_runtime_preferences_service.py::test_bootstrap_status_uses_readiness_service -v
```

Expected: FAIL because `BootstrapReadiness` and service getter are not imported/delegated.

- [ ] **Step 3: Add service getter and simplify Runtime Preferences**

In `backend/app/services/runtime_preferences_service.py`, import:

```python
from .bootstrap_readiness_service import (
    BootstrapReadiness,
    BootstrapReadinessService,
    MarketBootstrapReadiness,
)
from ..domain.markets.catalog import get_market_catalog
```

Add:

```python
def get_bootstrap_readiness_service() -> BootstrapReadinessService:
    return BootstrapReadinessService()
```

Replace `_normalize_supported_market` to use Market Catalog:

```python
def _normalize_supported_market(value: str | None) -> str:
    return get_market_catalog().get(value).code
```

Replace `supported_markets=list(SUPPORTED_MARKETS)` with:

```python
supported_markets=get_market_catalog().supported_market_codes()
```

Replace readiness logic inside `get_runtime_bootstrap_status` with:

```python
readiness = get_bootstrap_readiness_service().evaluate(
    db,
    enabled_markets=enabled_markets,
)
empty_system = readiness.empty_system
all_markets_ready = readiness.ready if should_check_readiness else False
```

Remove direct imports of `StockUniverse`, `StockPrice`, `StockFundamental`, `FeatureRun`, and `Scan` from Runtime Preferences after tests pass.

- [ ] **Step 4: Update runtime bootstrap completion checks**

In `backend/app/tasks/runtime_bootstrap_tasks.py`, replace private `_has_completed_auto_scan` import with:

```python
from ..services.bootstrap_readiness_service import BootstrapReadinessService
```

Inside `complete_local_runtime_bootstrap`, replace missing market calculation with:

```python
readiness = BootstrapReadinessService().evaluate(db, enabled_markets=enabled_markets)
missing_markets = readiness.missing_markets
```

- [ ] **Step 5: Verify backend runtime tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest \
  tests/unit/test_runtime_preferences_service.py \
  tests/unit/test_runtime_bootstrap_tasks.py \
  tests/unit/test_bootstrap_readiness_service.py \
  tests/unit/domain/test_bootstrap_plan.py \
  tests/unit/domain/test_market_catalog.py \
  -v
```

Expected: PASS.

## Task 7: Add Frontend Market Catalog Projection

**Files:**
- Modify: `frontend/src/contexts/RuntimeContext.jsx`
- Modify: `frontend/src/contexts/RuntimeContext.test.jsx`
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/components/App/BootstrapSetupScreen.jsx`
- Modify: `frontend/src/components/App/BootstrapSetupScreen.test.jsx`

- [ ] **Step 1: Add failing RuntimeContext test**

In `frontend/src/contexts/RuntimeContext.test.jsx`, update `RuntimeProbe` to read `marketCatalog` and render:

```jsx
<div data-testid="market-catalog-labels">
  {marketCatalog.markets.map((market) => market.label).join(',')}
</div>
```

Add `market_catalog` to the mocked app capabilities response:

```js
market_catalog: {
  version: '2026-05-04.v1',
  markets: [
    { code: 'US', label: 'United States', currency: 'USD', timezone: 'America/New_York', calendar_id: 'XNYS', exchanges: ['NYSE'], indexes: ['SP500'], capabilities: {} },
    { code: 'HK', label: 'Hong Kong', currency: 'HKD', timezone: 'Asia/Hong_Kong', calendar_id: 'XHKG', exchanges: ['HKEX'], indexes: ['HSI'], capabilities: {} },
  ],
},
```

Assert:

```js
expect(screen.getByTestId('market-catalog-labels')).toHaveTextContent('United States,Hong Kong');
```

- [ ] **Step 2: Run frontend context test and verify failure**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/frontend"
npm run test:run -- src/contexts/RuntimeContext.test.jsx
```

Expected: FAIL because `marketCatalog` is not exposed.

- [ ] **Step 3: Implement RuntimeContext projection**

In `frontend/src/contexts/RuntimeContext.jsx`, add:

```js
export const DEFAULT_MARKET_CATALOG = {
  version: 'fallback.v1',
  markets: [
    { code: 'US', label: 'United States', currency: 'USD', timezone: 'America/New_York', calendar_id: 'XNYS', exchanges: ['NYSE', 'NASDAQ', 'AMEX'], indexes: ['SP500'], capabilities: {} },
    { code: 'HK', label: 'Hong Kong', currency: 'HKD', timezone: 'Asia/Hong_Kong', calendar_id: 'XHKG', exchanges: ['HKEX'], indexes: ['HSI'], capabilities: {} },
    { code: 'IN', label: 'India', currency: 'INR', timezone: 'Asia/Kolkata', calendar_id: 'XNSE', exchanges: ['NSE', 'BSE'], indexes: [], capabilities: {} },
    { code: 'JP', label: 'Japan', currency: 'JPY', timezone: 'Asia/Tokyo', calendar_id: 'XTKS', exchanges: ['TSE', 'JPX'], indexes: ['NIKKEI225'], capabilities: {} },
    { code: 'KR', label: 'South Korea', currency: 'KRW', timezone: 'Asia/Seoul', calendar_id: 'XKRX', exchanges: ['KOSPI', 'KOSDAQ'], indexes: [], capabilities: {} },
    { code: 'TW', label: 'Taiwan', currency: 'TWD', timezone: 'Asia/Taipei', calendar_id: 'XTAI', exchanges: ['TWSE', 'TPEX'], indexes: ['TAIEX'], capabilities: {} },
    { code: 'CN', label: 'China A-shares', currency: 'CNY', timezone: 'Asia/Shanghai', calendar_id: 'XSHG', exchanges: ['SSE', 'SZSE', 'BJSE'], indexes: [], capabilities: {} },
  ],
};
```

Add to `DEFAULT_CAPABILITIES`:

```js
market_catalog: DEFAULT_MARKET_CATALOG,
```

Preserve in `mergeBootstrapCapabilities`:

```js
market_catalog: data.market_catalog
  ?? previous?.market_catalog
  ?? DEFAULT_CAPABILITIES.market_catalog,
```

Expose in provider value:

```js
marketCatalog: capabilities.market_catalog ?? DEFAULT_CAPABILITIES.market_catalog,
supportedMarkets: capabilities.supported_markets
  ?? (capabilities.market_catalog?.markets ?? DEFAULT_MARKET_CATALOG.markets).map((market) => market.code),
```

- [ ] **Step 4: Wire BootstrapSetupScreen labels**

In `frontend/src/App.jsx`, read `marketCatalog` from `useRuntime()` and pass:

```jsx
marketCatalog={marketCatalog}
```

In `frontend/src/components/App/BootstrapSetupScreen.jsx`, accept `marketCatalog` and derive options:

```js
const marketOptions = useMemo(() => {
  const catalogMarkets = marketCatalog?.markets ?? [];
  if (catalogMarkets.length > 0) {
    return catalogMarkets.map((market) => ({
      code: market.code,
      label: market.label || market.code,
    }));
  }
  return supportedMarkets.map((market) => ({ code: market, label: market }));
}, [marketCatalog?.markets, supportedMarkets]);
```

Replace `supportedMarkets.map((market) => ...` with `marketOptions.map((market) => ...`, using `market.code` for values and `market.label` for display.

- [ ] **Step 5: Verify frontend tests**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/frontend"
npm run test:run -- src/contexts/RuntimeContext.test.jsx src/components/App/BootstrapSetupScreen.test.jsx
```

Expected: PASS.

## Task 8: Final Verification

**Files:**
- All files from prior tasks.

- [ ] **Step 1: Run focused backend suite**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest \
  tests/unit/domain/test_market_catalog.py \
  tests/unit/domain/test_bootstrap_plan.py \
  tests/unit/test_bootstrap_readiness_service.py \
  tests/unit/test_runtime_preferences_service.py \
  tests/unit/test_runtime_bootstrap_tasks.py \
  tests/unit/test_app_runtime_endpoints.py \
  tests/unit/test_market_activity_service.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend suite**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/frontend"
npm run test:run -- src/contexts/RuntimeContext.test.jsx src/components/App/BootstrapSetupScreen.test.jsx src/components/Layout/Layout.test.jsx
```

Expected: PASS.

- [ ] **Step 3: Run quality gates**

Run:

```bash
cd "$(git rev-parse --show-toplevel)/backend"
source venv/bin/activate
pytest
```

Expected: PASS.

Run:

```bash
cd "$(git rev-parse --show-toplevel)/frontend"
npm run test:run
npm run lint
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
cd "$(git rev-parse --show-toplevel)"
git status --short
git add \
  backend/app/domain/markets \
  backend/app/domain/bootstrap \
  backend/app/services/bootstrap_readiness_service.py \
  backend/app/tasks/runtime_bootstrap_tasks.py \
  backend/app/services/runtime_preferences_service.py \
  backend/app/services/market_activity_service.py \
  backend/app/schemas/app_runtime.py \
  backend/app/api/v1/app_runtime.py \
  backend/tests/unit/domain/test_market_catalog.py \
  backend/tests/unit/domain/test_bootstrap_plan.py \
  backend/tests/unit/test_bootstrap_readiness_service.py \
  backend/tests/unit/test_runtime_preferences_service.py \
  backend/tests/unit/test_runtime_bootstrap_tasks.py \
  backend/tests/unit/test_app_runtime_endpoints.py \
  backend/tests/unit/test_market_activity_service.py \
  frontend/src/contexts/RuntimeContext.jsx \
  frontend/src/contexts/RuntimeContext.test.jsx \
  frontend/src/App.jsx \
  frontend/src/components/App/BootstrapSetupScreen.jsx \
  frontend/src/components/App/BootstrapSetupScreen.test.jsx
git commit -m "feat: deepen market runtime modules"
```

Expected: commit succeeds and includes only this Market runtime track.

## Self-Review

- Spec coverage: Market Catalog, Bootstrap Plan, and Bootstrap Readiness each have a dedicated module, tests, and integration steps.
- Locality: stable Market facts move behind Market Catalog; Bootstrap workflow moves behind Bootstrap Plan; readiness moves behind Bootstrap Readiness.
- Compatibility: `supported_markets` remains in backend and frontend payloads.
- Test surface: pure module tests cover catalog and plan; service tests cover readiness; endpoint/provider tests cover integration.
- Deferred work: Scan Admission, Scan Session, static Scan data parity, cache policy, and provider routing remain out of scope.
