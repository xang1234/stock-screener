# Market Module, Activity Gate, and Cache Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Market` a deep backend module, move scan-blocking policy into a `MarketActivityGate`, then use that market module to introduce market-scoped cache policy for price, benchmark, and fundamentals caches.

**Architecture:** Phase A implements a domain `Market` value object plus `MarketRegistry` as the single source of stable Market facts. Scan gating is moved out of `api/v1/scans.py` into a `MarketActivityGate` that asks Market Workload for activity state and returns a typed decision. Phase B introduces a `MarketAwareCachePolicy` that owns cache key construction, TTL lookup, and market-aware freshness helpers; cache services adopt it behind backward-compatible method signatures.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, Celery, Redis, pytest, dataclasses, existing `bd` issue workflow.

---

## Scope

This plan intentionally does two related projects in sequence:

1. **Market module + scan gate**: candidates 1 and 7 together.
2. **Market-scoped cache policy**: candidate 2 after Market exists.

This plan does not migrate every `market: str` call site in the repo in one pass. Existing public function signatures stay compatible where broad migration would create unnecessary risk. New modules accept `Market`; legacy adapters normalize strings at their boundaries and delegate to the new module.

## Domain Rules

- **Market Catalog** owns stable Market facts: code, label, currency, timezone, calendar ID, exchanges, indexes, benchmark mapping, and scan guard mappings.
- **Runtime Preferences** owns enabled/primary Market choices.
- **Market Workload** owns operational state and queue derivation. Queue names stay derived from Market codes; they are not stable Market Catalog facts.
- `SHARED` remains a queue/workload sentinel, not a `Market`.
- Scan-blocking policy remains part of the **Scan** path, but the route no longer owns the rule.

## File Map

### Create

- `backend/app/domain/markets/__init__.py`: exports the Market module API.
- `backend/app/domain/markets/market.py`: `Market` value object and validation errors.
- `backend/app/domain/markets/registry.py`: `MarketProfile` rows and `MarketRegistry` lookup helpers.
- `backend/app/services/market_activity_gate.py`: scan-start gate over Market Workload activity state.
- `backend/app/services/cache/market_cache_policy.py`: cache key, TTL, and freshness policy helpers.
- `backend/tests/unit/domain/markets/test_market.py`: value object tests.
- `backend/tests/unit/domain/markets/test_market_registry.py`: registry coverage and lookup tests.
- `backend/tests/unit/test_market_activity_gate.py`: scan gating policy tests.
- `backend/tests/unit/test_market_cache_policy.py`: key/TTL/freshness policy tests.

### Modify

- `backend/app/tasks/market_queues.py`: keep public API, delegate supported-market validation to `Market`.
- `backend/app/services/market_calendar_service.py`: use `MarketRegistry` for calendar/timezone lookup.
- `backend/app/domain/common/benchmarks.py`: use `MarketRegistry` for primary benchmark symbols.
- `backend/app/services/benchmark_registry_service.py`: use `MarketRegistry` for primary symbols and supported-market coverage.
- `backend/app/services/security_master_service.py`: align supported markets, defaults, and exchange mappings with `MarketRegistry`.
- `backend/app/api/v1/scans.py`: remove route-local scan guard maps and call `MarketActivityGate`.
- `backend/app/services/benchmark_cache_service.py`: use cache policy for benchmark data/lock keys and market-aware invalidation.
- `backend/app/services/fundamentals_cache_service.py`: use cache policy for fundamentals keys; add market-aware bulk cache reads.
- `backend/app/services/price_cache_service.py`: use cache policy for price keys and metadata; add market-aware bulk cache reads.
- `backend/app/scanners/data_preparation.py`: pass symbol-to-market maps into cache bulk reads.

## Phase A: Market Module + Scan Gate

### Task 1: Add `Market`

**Files:**
- Create: `backend/app/domain/markets/__init__.py`
- Create: `backend/app/domain/markets/market.py`
- Test: `backend/tests/unit/domain/markets/test_market.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove:

- `Market.from_str(" hk ") == Market("HK")`
- `Market("HK")` is immutable and hashable
- `Market("hk")`, `Market.from_str(None)`, and `Market.from_str("EU")` raise `UnsupportedMarketError`
- supported codes are exactly `US, HK, IN, JP, KR, TW, CN`

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/domain/markets/test_market.py -v
```

Expected: fails because `app.domain.markets` does not exist.

- [ ] **Step 2: Implement minimal value object**

Implement:

```python
SUPPORTED_MARKET_CODES = frozenset({"US", "HK", "IN", "JP", "KR", "TW", "CN"})

class UnsupportedMarketError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class Market:
    code: str

    def __post_init__(self) -> None:
        if self.code not in SUPPORTED_MARKET_CODES:
            supported = ", ".join(sorted(SUPPORTED_MARKET_CODES))
            raise UnsupportedMarketError(
                f"Unsupported market code {self.code!r}. Supported: {supported}"
            )

    @classmethod
    def from_str(cls, raw: str | None) -> "Market":
        if raw is None:
            raise UnsupportedMarketError("Market code is required")
        code = raw.strip().upper()
        if not code:
            raise UnsupportedMarketError("Market code is required")
        return cls(code)
```

- [ ] **Step 3: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/domain/markets/test_market.py -v
```

Expected: pass.

### Task 2: Add `MarketRegistry`

**Files:**
- Create: `backend/app/domain/markets/registry.py`
- Modify: `backend/app/domain/markets/__init__.py`
- Test: `backend/tests/unit/domain/markets/test_market_registry.py`

- [ ] **Step 1: Write failing registry tests**

Add tests that assert every supported Market has:

- label
- currency
- timezone
- calendar ID
- exchanges
- indexes
- primary benchmark symbol
- scan guard exchange/index mappings

Also assert representative lookups:

- `registry.profile("hk").calendar_id == "XHKG"`
- `registry.profile(Market("IN")).timezone_name == "Asia/Kolkata"`
- `registry.market_for_index("HSI") == Market("HK")`
- `registry.market_for_exchange("XBOM") == Market("IN")`

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/domain/markets/test_market_registry.py -v
```

Expected: fails because `registry.py` does not exist.

- [ ] **Step 2: Implement profile rows**

Create `MarketProfile` as a frozen dataclass with:

```python
market: Market
label: str
currency: str
timezone_name: str
calendar_id: str
provider_calendar_id: str | None
exchanges: tuple[str, ...]
indexes: tuple[str, ...]
primary_benchmark_symbol: str
benchmark_fallback_symbol: str | None
benchmark_primary_kind: str
benchmark_fallback_kind: str | None
```

Populate rows for:

- US: `XNYS`, `America/New_York`, `SPY`, exchanges `NYSE/NASDAQ/AMEX`, index `SP500`
- HK: `XHKG`, `Asia/Hong_Kong`, `^HSI`, exchange `HKEX`, index `HSI`
- IN: `XNSE`, provider calendar `NSE`, `Asia/Kolkata`, `^NSEI`, exchanges `NSE/XNSE/BSE/XBOM`
- JP: `XTKS`, `Asia/Tokyo`, `^N225`, exchange `TSE/JPX/XTKS`, index `NIKKEI225`
- KR: `XKRX`, `Asia/Seoul`, `^KS11`, exchanges `KOSPI/KOSDAQ/KRX/XKRX`
- TW: `XTAI`, `Asia/Taipei`, `^TWII`, exchanges `TWSE/TPEX/XTAI`, index `TAIEX`
- CN: `XSHG`, `Asia/Shanghai`, `000300.SS`, exchanges `SSE/SZSE/BJSE/XSHG/XSHE/XBSE`

- [ ] **Step 3: Add registry helpers**

Add:

```python
def profile(self, market: Market | str) -> MarketProfile
def supported_markets(self) -> tuple[Market, ...]
def supported_market_codes(self) -> tuple[str, ...]
def market_for_exchange(self, exchange: str | None) -> Market | None
def market_for_index(self, index: str | None) -> Market | None
```

- [ ] **Step 4: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/domain/markets/test_market_registry.py -v
```

Expected: pass.

### Task 3: Delegate existing market helpers to `MarketRegistry`

**Files:**
- Modify: `backend/app/tasks/market_queues.py`
- Modify: `backend/app/services/market_calendar_service.py`
- Modify: `backend/app/domain/common/benchmarks.py`
- Modify: `backend/app/services/benchmark_registry_service.py`
- Modify: `backend/app/services/security_master_service.py`
- Tests: existing market queue/calendar/benchmark/security-master tests

- [ ] **Step 1: Add compatibility tests before changing implementation**

Extend existing tests to assert:

- `tasks.market_queues.SUPPORTED_MARKETS == registry.supported_market_codes()`
- `MarketCalendarService.calendar_id(market)` equals registry calendar ID for every supported Market
- `supported_benchmark_markets()` equals registry supported codes
- `SecurityMasterResolver.normalize_market(code)` accepts every registry supported code

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/test_market_queues.py \
       tests/unit/test_market_calendar_service.py \
       tests/unit/test_benchmark_registry_service.py \
       tests/unit/test_security_master_service.py -v
```

Expected: new tests may fail where constants drift.

- [ ] **Step 2: Replace duplicated supported-market lists**

Keep public constants for compatibility, but derive them from the registry:

```python
SUPPORTED_MARKETS: tuple[str, ...] = market_registry.supported_market_codes()
```

In calendar and benchmark modules, keep public methods accepting `str | None`, normalize at the boundary, and use `registry.profile(Market.from_str(market))`.

- [ ] **Step 3: Verify no behavior drift**

Run the same pytest command from Step 1.

Expected: pass.

### Task 4: Move scan guard market resolution into `MarketRegistry`

**Files:**
- Modify: `backend/app/domain/markets/registry.py`
- Modify: `backend/app/api/v1/scans.py`
- Test: `backend/tests/unit/test_scan_create_endpoint.py`

- [ ] **Step 1: Add tests for index/exchange scan guard coverage**

Extend tests to assert:

- market universe resolves to its explicit market
- exchange universe uses `market_registry.market_for_exchange`
- index universe uses `market_registry.market_for_index`
- `HSI -> HK`, `NIKKEI225 -> JP`, `TAIEX -> TW`
- `KOSPI/KOSDAQ -> KR`, `XBOM -> IN`, `SSE/SZSE/BJSE -> CN`

- [ ] **Step 2: Replace route-local maps**

Remove `SCAN_GUARD_MARKET_BY_INDEX` and `SCAN_GUARD_MARKET_BY_EXCHANGE` from `api/v1/scans.py`.

Implement `_resolve_scan_guard_market` as:

```python
def _resolve_scan_guard_market(universe_def: Any) -> str | None:
    if getattr(universe_def, "market", None):
        return universe_def.market.value
    if getattr(universe_def, "exchange", None):
        market = market_registry.market_for_exchange(universe_def.exchange.value)
        return market.code if market else None
    if getattr(universe_def, "index", None):
        market = market_registry.market_for_index(universe_def.index.value)
        return market.code if market else None
    return None
```

- [ ] **Step 3: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/test_scan_create_endpoint.py -v
```

Expected: pass.

### Task 5: Add `MarketActivityGate`

**Files:**
- Create: `backend/app/services/market_activity_gate.py`
- Modify: `backend/app/api/v1/scans.py`
- Test: `backend/tests/unit/test_market_activity_gate.py`
- Test: `backend/tests/unit/test_scan_create_endpoint.py`

- [ ] **Step 1: Write gate tests**

Test these cases with fake runtime activity payloads:

- no market returns allowed
- active `prices/running` for same Market returns conflict
- active `fundamentals/queued` for same Market returns conflict
- active `breadth/running` for same Market is allowed
- active refresh in a different Market is allowed
- multiple active blocking stages return sorted `active_stages`

- [ ] **Step 2: Implement gate types**

Add:

```python
@dataclass(frozen=True)
class MarketGateAllowed:
    market: Market | None = None

@dataclass(frozen=True)
class MarketGateConflict:
    market: Market
    active_stages: tuple[str, ...]
    lifecycle: str | None
    message: str

    def to_http_detail(self) -> dict[str, object]:
        return {
            "code": "market_refresh_active",
            "message": self.message,
            "market": self.market.code,
            "active_stages": list(self.active_stages),
            "lifecycle": self.lifecycle,
        }
```

Add `MarketActivityGate.check(market: Market | str | None) -> MarketGateAllowed | MarketGateConflict`.

Constructor accepts:

```python
get_runtime_activity_status: Callable[[Session], dict[str, Any]]
session_factory: Callable[[], Session]
blocking_stages: frozenset[str] = frozenset({"prices", "fundamentals"})
blocking_statuses: frozenset[str] = frozenset({"queued", "running"})
```

- [ ] **Step 3: Wire route to the gate**

In `api/v1/scans.py`, replace `_get_market_refresh_conflict_detail` internals with:

```python
decision = get_market_activity_gate().check(guard_market)
if isinstance(decision, MarketGateConflict):
    raise HTTPException(status_code=409, detail=decision.to_http_detail())
```

Add a dependency helper in wiring if current project style prefers it; otherwise use a module-level factory consistent with nearby services.

- [ ] **Step 4: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/test_market_activity_gate.py tests/unit/test_scan_create_endpoint.py -v
```

Expected: pass.

### Task 6: Add consistency guard tests

**Files:**
- Create or extend: `backend/tests/unit/domain/markets/test_market_registry.py`
- Existing modules covered by assertions

- [ ] **Step 1: Add registry consistency tests**

Add one test that asserts:

- market queue supported codes match registry codes
- benchmark supported codes match registry codes
- calendar service supports every registry code
- every registry code has a primary benchmark symbol
- every scan guard index/exchange maps to a supported Market

- [ ] **Step 2: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/domain/markets/test_market_registry.py -v
```

Expected: pass.

## Phase B: Market-Scoped Cache Policy

### Task 7: Add `MarketAwareCachePolicy`

**Files:**
- Create: `backend/app/services/cache/market_cache_policy.py`
- Test: `backend/tests/unit/test_market_cache_policy.py`

- [ ] **Step 1: Write policy tests**

Assert key shape:

- price recent: `price:hk:0700.HK:recent`
- price last update: `price:hk:0700.HK:last_update`
- price metadata: `price:hk:0700.HK:fetch_meta`
- benchmark data: `benchmark:hk:^HSI:2y`
- benchmark lock: `benchmark:hk:^HSI:2y:lock`
- fundamentals: `fundamentals:hk:0700.HK`

Assert default TTLs:

- price: 604800
- fundamentals: 604800
- benchmark: 86400

Assert legacy key helpers are available for fallback reads:

- `legacy_price_recent_key("AAPL") == "price:AAPL:recent"`
- `legacy_fundamentals_key("AAPL") == "fundamentals:AAPL"`
- `legacy_benchmark_key("SPY", "2y") == "benchmark:SPY:2y"`

- [ ] **Step 2: Implement policy**

Create a small module with:

```python
class CacheNamespace(str, Enum):
    PRICE = "price"
    FUNDAMENTALS = "fundamentals"
    BENCHMARK = "benchmark"

@dataclass(frozen=True)
class MarketAwareCachePolicy:
    price_ttl_seconds: int = 604800
    fundamentals_ttl_seconds: int = 604800
    benchmark_ttl_seconds: int = 86400

    def suffix(self, market: Market | str) -> str:
        resolved = market if isinstance(market, Market) else Market.from_str(market)
        return resolved.code.lower()

    def price_recent_key(self, symbol: str, market: Market | str) -> str:
        return f"price:{self.suffix(market)}:{symbol}:recent"

    def price_last_update_key(self, symbol: str, market: Market | str) -> str:
        return f"price:{self.suffix(market)}:{symbol}:last_update"

    def price_fetch_meta_key(self, symbol: str, market: Market | str) -> str:
        return f"price:{self.suffix(market)}:{symbol}:fetch_meta"

    def benchmark_data_key(self, symbol: str, period: str, market: Market | str) -> str:
        return f"benchmark:{self.suffix(market)}:{symbol}:{period}"

    def benchmark_lock_key(self, symbol: str, period: str, market: Market | str) -> str:
        return f"{self.benchmark_data_key(symbol, period, market)}:lock"

    def fundamentals_key(self, symbol: str, market: Market | str) -> str:
        return f"fundamentals:{self.suffix(market)}:{symbol}"
```

Use `Market.from_str()` for validation; do not accept `SHARED` in this policy.

- [ ] **Step 3: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/test_market_cache_policy.py -v
```

Expected: pass.

### Task 8: Adopt policy in `BenchmarkCacheService`

**Files:**
- Modify: `backend/app/services/benchmark_cache_service.py`
- Test: `backend/tests/unit/test_benchmark_cache_service.py`
- Test: `backend/tests/unit/test_warm_spy_cache_market_scoping.py`

- [ ] **Step 1: Add failing service tests**

Assert:

- `get_benchmark_data(market="HK")` reads `benchmark:hk:^HSI:2y`
- lock key is `benchmark:hk:^HSI:2y:lock`
- legacy key fallback still reads `benchmark:^HSI:2y` when the new key misses
- when legacy fallback succeeds, service writes the new market-scoped key

- [ ] **Step 2: Inject policy**

Add optional constructor argument:

```python
cache_policy: MarketAwareCachePolicy | None = None
```

Replace `_redis_data_key(benchmark_symbol, period)` with `_redis_data_key(benchmark_symbol, period, market)`.

Read order:

1. new market-scoped key
2. legacy key
3. database/network

Write target: new market-scoped key only.

- [ ] **Step 3: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/test_benchmark_cache_service.py \
       tests/unit/test_warm_spy_cache_market_scoping.py -v
```

Expected: pass.

### Task 9: Adopt policy in `FundamentalsCacheService`

**Files:**
- Modify: `backend/app/services/fundamentals_cache_service.py`
- Modify: `backend/app/scanners/data_preparation.py`
- Test: `backend/tests/unit/test_fundamentals_cache_service.py`
- Test: `backend/tests/unit/infra/test_stock_data_provider.py`

- [ ] **Step 1: Add failing tests**

Assert:

- `get_fundamentals("0700.HK", market="HK")` reads/writes `fundamentals:hk:0700.HK`
- legacy fallback reads `fundamentals:0700.HK` and rewrites `fundamentals:hk:0700.HK`
- `get_many(["AAPL", "0700.HK"], market_by_symbol={"AAPL": "US", "0700.HK": "HK"})` pipelines market-scoped keys
- `DataPreparationLayer.prepare_data_bulk()` passes market map from resolved identities

- [ ] **Step 2: Update service API compatibly**

Keep existing signature working:

```python
def get_many(
    self,
    symbols: list[str],
    *,
    market_by_symbol: dict[str, str] | None = None,
) -> dict[str, dict | None]:
```

Default missing market to `US` for legacy callers.

- [ ] **Step 3: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/test_fundamentals_cache_service.py \
       tests/unit/infra/test_stock_data_provider.py -v
```

Expected: pass.

### Task 10: Adopt policy in `PriceCacheService`

**Files:**
- Modify: `backend/app/services/price_cache_service.py`
- Modify: `backend/app/services/cache/price_cache_freshness.py`
- Modify: `backend/app/scanners/data_preparation.py`
- Test: `backend/tests/unit/test_price_cache_service.py`
- Test: `backend/tests/unit/infra/test_stock_data_provider.py`

- [ ] **Step 1: Add failing tests**

Assert:

- `get_historical_data("0700.HK", market="HK")` reads/writes `price:hk:0700.HK:recent`
- last update key is `price:hk:0700.HK:last_update`
- fetch metadata key is `price:hk:0700.HK:fetch_meta`
- `get_many(["0700.HK"], market_by_symbol={"0700.HK": "HK"})` batches market-scoped keys
- legacy fallback reads old `price:{symbol}:recent` and rewrites new scoped keys

- [ ] **Step 2: Update service API compatibly**

Add optional market args:

```python
def get_historical_data(
    self,
    symbol: str,
    period: str = "2y",
    force_refresh: bool = False,
    market: str | None = None,
) -> Optional[pd.DataFrame]

def get_many(
    self,
    symbols: list[str],
    period: str = "2y",
    market_by_symbol: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame | None]
```

Default missing market to `US` for legacy callers.

Update internal helpers to require market where keys are constructed:

- `_store_recent_in_redis(symbol, data, market)`
- `_store_fetch_metadata(symbol, market)`
- `_get_fetch_metadata(symbol, market)`
- `_clear_fetch_metadata(symbol, market)`

- [ ] **Step 3: Keep legacy scans safe**

For key scans:

- new all-symbol scan pattern becomes `price:*:*:recent`
- legacy fallback scan pattern remains `price:*:recent`
- parser must distinguish 4-part scoped keys from 3-part legacy keys

- [ ] **Step 4: Verify**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/test_price_cache_service.py \
       tests/unit/infra/test_stock_data_provider.py -v
```

Expected: pass.

### Task 11: End-to-end cache policy regression tests

**Files:**
- Test: existing scan/data-prep/cache tests

- [ ] **Step 1: Add cross-service invariant test**

Create a test that imports `MarketAwareCachePolicy` and confirms no cache service constructs a Redis key using old hardcoded formats for market-aware methods. Keep this scoped to public helper behavior; do not brittle-match every internal string.

- [ ] **Step 2: Run focused backend suite**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest tests/unit/domain/markets \
       tests/unit/test_market_activity_gate.py \
       tests/unit/test_scan_create_endpoint.py \
       tests/unit/test_market_cache_policy.py \
       tests/unit/test_benchmark_cache_service.py \
       tests/unit/test_fundamentals_cache_service.py \
       tests/unit/test_price_cache_service.py \
       tests/unit/infra/test_stock_data_provider.py -v
```

Expected: pass.

### Task 12: Quality gates and issue updates

**Files:**
- No code files unless failures require fixes.

- [ ] **Step 1: Run backend quality gate**

Run:

```bash
cd /Users/admin/StockScreenClaude/backend
source venv/bin/activate
pytest
```

Expected: pass.

- [ ] **Step 2: Run frontend only if runtime payload or UI files changed**

Run:

```bash
cd /Users/admin/StockScreenClaude/frontend
npm run test:run
npm run lint
```

Expected: pass.

- [ ] **Step 3: Close or create beads**

If this plan is executed under a bead, close it after tests pass:

```bash
bd close <issue-id> --reason "Completed market module, scan gate, and cache policy refactor"
bd sync
```

If execution uncovers follow-up work, create focused beads before closing:

```bash
bd create --title="Migrate remaining legacy market string call sites to Market value object" --type=task --priority=3
bd create --title="Remove legacy unscoped Redis cache-key fallback after rollout window" --type=task --priority=3
bd sync
```

## Commit Plan

Use small commits:

```bash
git commit -m "feat(domain): add Market value object and registry"
git commit -m "refactor(markets): delegate calendar benchmark and queue helpers to registry"
git commit -m "feat(scans): move market refresh gating into service"
git commit -m "feat(cache): add market-aware cache policy"
git commit -m "refactor(cache): scope benchmark cache keys by market"
git commit -m "refactor(cache): scope fundamentals cache keys by market"
git commit -m "refactor(cache): scope price cache keys by market"
```

## Rollout Notes

- Cache key changes should be dual-read/new-write, not destructive migration.
- Legacy cache keys can be removed in a later bead after one TTL window and operator confirmation.
- Unknown Market handling should fail closed in new code, except existing compatibility functions that intentionally default `None` to US.
- Avoid moving queue names into the Market Catalog; derive them from `Market.code` inside Market Workload helpers.

## Self-Review

- Spec coverage: candidate 1 is covered by Tasks 1-4 and 6; candidate 7 is covered by Task 5; candidate 2 is covered by Tasks 7-11.
- Placeholder scan: no implementation step relies on open-ended "add tests" without naming behaviors.
- Type consistency: new domain APIs use `Market`; legacy service APIs accept strings and normalize at boundaries.
