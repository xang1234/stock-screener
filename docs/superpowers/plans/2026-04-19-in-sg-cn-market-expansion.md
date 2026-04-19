# India, Singapore, and China Market Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing US/HK/JP/TW multi-market scanner to support India (`IN`), Singapore (`SG`), and mainland China (`CN`) for universe ingestion, price/fundamentals refresh, benchmark-relative scanning, bootstrap, and UI market selection.

**Architecture:** Reuse the existing ASIA v2 abstractions instead of adding market-specific branches. Add the new markets to typed contracts first, then extend `SecurityMaster`, `BenchmarkRegistry`, `MarketCalendarService`, provider routing, and official-universe adapters in that order. Ship in phases: India first, Singapore second, China third. Keep breadth, IBD group ranks, and theme discovery explicitly US-only for the new markets until separate parity work exists.

**Tech Stack:** Python backend, Celery, yfinance, requests/pandas/BeautifulSoup, `exchange_calendars`, `pandas_market_calendars`, Pydantic schemas, React frontend, beads issue tracking.

---

## Investigation Snapshot

Validated on 2026-04-19 in `backend/venv`:

- `yfinance` worked for representative India symbols: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`
- `yfinance` worked for representative Singapore symbols: `D05.SI`, `C6L.SI`, `Z74.SI`
- `yfinance` worked for representative China symbols: `600519.SS`, `601318.SS`, `000858.SZ`
- Benchmark/index symbols also resolved in `yfinance`:
  - India: `^NSEI`, `^BSESN`, `NIFTYBEES.NS`
  - Singapore: `^STI`, `ES3.SI`
  - China: `000300.SS`, `000001.SS`, `399001.SZ`, `510300.SS`
- Current repo blocker is not price/fundamentals fetch. The blocker is market hard-coding across:
  - `backend/app/schemas/universe.py`
  - `backend/app/services/security_master_service.py`
  - `backend/app/services/benchmark_registry_service.py`
  - `backend/app/services/market_calendar_service.py`
  - `backend/app/services/provider_routing_policy.py`
  - `backend/app/tasks/market_queues.py`
  - `backend/app/tasks/universe_tasks.py`
  - `frontend/src/features/scan/constants.js`
  - `frontend/src/contexts/RuntimeContext.jsx`
- Current product-scope constraint: breadth, group rankings, and theme discovery are intentionally US-only. New market support must not claim parity for those features.

### Source Readiness By Market

| Market | Price/Fundamentals | Official Universe Source | Confidence | Decision |
| --- | --- | --- | --- | --- |
| `IN` | Good via `yfinance` | NSE equity CSV is publicly reachable: `https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv` | High | Implement first |
| `SG` | Good via `yfinance` | SGX public pages are reachable, but the stock screener is JS-driven and the stable machine-readable universe endpoint was not conclusively identified | Medium | Timebox discovery before enabling |
| `CN` | Good via `yfinance` for `.SS` and `.SZ` | SZSE English data-services page explicitly advertises end-of-day Excel/CSV/TXT/Database data; SSE quotation pages are reachable but a clean bulk listing export still needs targeted discovery | Medium | Implement after source hardening |

### Calendar Findings

- `exchange_calendars` already supports:
  - `XSES` for Singapore
  - `XSHG` for Shanghai
  - `BSE` / `XBSE` for India
- `exchange_calendars` does **not** expose `XNSE`
- `pandas_market_calendars` does expose `XNSE`
- Implementation implication:
  - `SG` can stay on `exchange_calendars`
  - `CN` can use `XSHG` as the canonical mainland calendar, with the assumption that Shanghai/Shenzhen trading sessions remain aligned
  - `IN` needs an explicit fallback path to `pandas_market_calendars`

### Rollout Rules

1. Do not enable `SG` until a stable, headless, license-safe official universe source is validated.
2. Do not enable `CN` as a whole-market scan unless both SSE and SZSE source coverage are implemented or the UI is explicitly narrowed to the supported exchange subset.
3. Do not remove the current US-only guards for breadth, group rankings, or themes during this effort.
4. Do not add BSE equity support in v1. Scope India to NSE-first unless a separate issue expands the market definition.

## File Map

### Existing files that will change

- `backend/app/config/settings.py`
- `backend/app/schemas/universe.py`
- `backend/app/services/security_master_service.py`
- `backend/app/services/benchmark_registry_service.py`
- `backend/app/services/market_calendar_service.py`
- `backend/app/services/provider_routing_policy.py`
- `backend/app/services/growth_cadence_service.py`
- `backend/app/services/official_market_universe_source_service.py`
- `backend/app/services/stock_universe_service.py`
- `backend/app/tasks/market_queues.py`
- `backend/app/tasks/universe_tasks.py`
- `backend/app/tasks/runtime_bootstrap_tasks.py`
- `backend/start_celery.sh`
- `backend/requirements-runtime.txt`
- `backend/requirements-server.txt`
- `frontend/src/features/scan/constants.js`
- `frontend/src/contexts/RuntimeContext.jsx`
- `backend/app/domain/analytics/scope.py`
- `backend/app/tasks/breadth_tasks.py`
- `backend/app/tasks/group_rank_tasks.py`
- `docs/asia/README.md`
- `docs/asia/benchmark_registry_table_v1.md`

### New files to create

- `backend/app/services/in_universe_ingestion_adapter.py`
- `backend/app/services/sg_universe_ingestion_adapter.py`
- `backend/app/services/cn_universe_ingestion_adapter.py`
- `backend/tests/unit/fixtures/universe_sources/nse_equity_l_fixture.csv`
- `backend/tests/unit/fixtures/universe_sources/sgx_equities_fixture.json`
- `backend/tests/unit/fixtures/universe_sources/sse_equities_fixture.csv`
- `backend/tests/unit/fixtures/universe_sources/szse_equities_fixture.csv`
- `docs/asia/asia_v3_in_sg_cn_source_matrix.md`

### Primary test files

- `backend/tests/unit/test_security_master_service.py`
- `backend/tests/unit/test_market_queues.py`
- `backend/tests/unit/test_benchmark_registry_service.py`
- `backend/tests/unit/test_market_calendar_service.py`
- `backend/tests/unit/test_market_calendar_service_engine.py`
- `backend/tests/unit/test_provider_routing_policy.py`
- `backend/tests/unit/test_growth_cadence_service.py`
- `backend/tests/unit/test_data_source_service_policy.py`
- `backend/tests/unit/test_official_market_universe_source_service.py`
- `backend/tests/unit/test_stock_universe_service.py`
- `backend/tests/unit/test_universe_tasks.py`
- `backend/tests/unit/test_runtime_bootstrap_tasks.py`
- `frontend/src/contexts/RuntimeContext.test.jsx`

## Recommended Execution Order

1. Task 1 and Task 2 together
2. Task 3
3. Task 4
4. Task 5
5. Task 6
6. Task 7
7. Task 8
8. Task 9

### Task 1: Lock Scope, Sources, and Non-Goals

**Files:**
- Create: `docs/asia/asia_v3_in_sg_cn_source_matrix.md`
- Modify: `docs/asia/README.md`
- Modify: `docs/asia/benchmark_registry_table_v1.md`

- [ ] **Step 1: Write the source matrix document before code changes**

```markdown
# ASIA v3 Source Matrix: IN / SG / CN

| Market | Canonical market code | Price/fundamentals provider | Official universe source | Launch status |
| --- | --- | --- | --- | --- |
| India | `IN` | `yfinance` (`.NS`) | NSE equity list CSV (`EQUITY_L.csv`) | Ready for implementation |
| Singapore | `SG` | `yfinance` (`.SI`) | SGX endpoint discovery required | Blocked on source validation |
| China | `CN` | `yfinance` (`.SS`, `.SZ`) | SSE + SZSE source bundle | Blocked on SSE source validation |

Non-goals for this track:
- breadth parity
- IBD group rank parity
- non-NSE India exchanges
- SG/CN enablement without official-universe proof
```

- [ ] **Step 2: Verify the repo docs still describe the old HK/JP/TW-only scope**

Run: `rg -n "HK/JP/TW|US/HK/JP/TW|multi-market" docs/asia docs/README*`
Expected: existing docs mention the old four-market scope and give you exact lines to update.

- [ ] **Step 3: Update the docs to state the new phased rollout**

```markdown
Implementation order:
1. `IN` (NSE-first)
2. `SG` (after official-source validation)
3. `CN` (after SSE/SZSE source validation)

The scanner may support additional markets before analytics parity exists.
Breadth, group rankings, and themes remain US-only until a separate program expands them.
```

- [ ] **Step 4: Re-run the doc grep and confirm the new plan is discoverable**

Run: `rg -n "IN|SG|CN|NSE|SGX|SZSE|SSE" docs/asia`
Expected: matches include the new source matrix and updated README references.

- [ ] **Step 5: Commit**

```bash
git add docs/asia/README.md docs/asia/benchmark_registry_table_v1.md docs/asia/asia_v3_in_sg_cn_source_matrix.md
git commit -m "docs: record in sg cn expansion scope"
```

### Task 2: Expand Typed Market Contracts and Frontend Market Lists

**Files:**
- Modify: `backend/app/schemas/universe.py`
- Modify: `frontend/src/features/scan/constants.js`
- Modify: `frontend/src/contexts/RuntimeContext.jsx`
- Test: `frontend/src/contexts/RuntimeContext.test.jsx`

- [ ] **Step 1: Write failing contract tests for the new markets**

```python
def test_market_enum_supports_in_sg_cn():
    assert Market("IN") is Market.IN
    assert Market("SG") is Market.SG
    assert Market("CN") is Market.CN


def test_market_universe_labels_cover_new_markets():
    assert UniverseDefinition(type=UniverseType.MARKET, market=Market.IN).label() == "India Market"
    assert UniverseDefinition(type=UniverseType.MARKET, market=Market.SG).label() == "Singapore Market"
    assert UniverseDefinition(type=UniverseType.MARKET, market=Market.CN).label() == "China Market"
```

```javascript
it('includes IN SG and CN in supported runtime markets', () => {
  expect(DEFAULT_CAPABILITIES.supported_markets).toEqual(['US', 'HK', 'JP', 'TW', 'IN', 'SG', 'CN']);
});
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_universe_resolver_asia_indices.py tests/unit/test_security_master_service.py -q`
Expected: failures or missing enum references for `IN`, `SG`, and `CN`.

Run: `cd frontend && npm run test:run -- RuntimeContext.test.jsx`
Expected: failure because the frontend defaults still expose only `US/HK/JP/TW`.

- [ ] **Step 3: Implement the minimal contract and UI changes**

```python
class Market(str, Enum):
    US = "US"
    HK = "HK"
    JP = "JP"
    TW = "TW"
    IN = "IN"
    SG = "SG"
    CN = "CN"
```

```python
market_labels = {
    Market.US: "US Market",
    Market.HK: "Hong Kong Market",
    Market.JP: "Japan Market",
    Market.TW: "Taiwan Market",
    Market.IN: "India Market",
    Market.SG: "Singapore Market",
    Market.CN: "China Market",
}
```

```javascript
export const UNIVERSE_GEOGRAPHIC_MARKETS = ['US', 'HK', 'JP', 'TW', 'IN', 'SG', 'CN'];

export const UNIVERSE_MARKETS = [
  { value: 'US', label: 'United States' },
  { value: 'HK', label: 'Hong Kong' },
  { value: 'JP', label: 'Japan' },
  { value: 'TW', label: 'Taiwan' },
  { value: 'IN', label: 'India' },
  { value: 'SG', label: 'Singapore' },
  { value: 'CN', label: 'China' },
  { value: 'TEST', label: 'Test Mode' },
];
```

- [ ] **Step 4: Re-run the tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_security_master_service.py -q`
Expected: the new enum/label checks pass.

Run: `cd frontend && npm run test:run -- RuntimeContext.test.jsx`
Expected: passing runtime capability defaults with the expanded market list.

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas/universe.py frontend/src/features/scan/constants.js frontend/src/contexts/RuntimeContext.jsx frontend/src/contexts/RuntimeContext.test.jsx
git commit -m "feat: extend market contracts for in sg cn"
```

### Task 3: Extend SecurityMaster, Queue Topology, and Runtime Worker Support

**Files:**
- Modify: `backend/app/services/security_master_service.py`
- Modify: `backend/app/tasks/market_queues.py`
- Modify: `backend/start_celery.sh`
- Test: `backend/tests/unit/test_security_master_service.py`
- Test: `backend/tests/unit/test_market_queues.py`

- [ ] **Step 1: Add failing tests for symbol resolution and queue support**

```python
def test_resolve_identity_supports_india_and_singapore_suffixes():
    resolver = SecurityMasterResolver()
    assert resolver.resolve_identity(symbol="RELIANCE.NS").market == "IN"
    assert resolver.resolve_identity(symbol="D05.SI").market == "SG"


def test_resolve_identity_preserves_cn_exchange_specific_suffixes():
    resolver = SecurityMasterResolver()
    sh = resolver.resolve_identity(symbol="600519", exchange="XSHG")
    sz = resolver.resolve_identity(symbol="000858", exchange="XSHE")
    assert sh.canonical_symbol == "600519.SS"
    assert sz.canonical_symbol == "000858.SZ"
    assert sh.market == sz.market == "CN"
```

```python
def test_supported_markets_include_in_sg_cn():
    assert SUPPORTED_MARKETS == ("US", "HK", "JP", "TW", "IN", "SG", "CN")
```

- [ ] **Step 2: Run the focused tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_security_master_service.py tests/unit/test_market_queues.py -q`
Expected: failures because `IN`, `SG`, and `CN` are unknown.

- [ ] **Step 3: Implement the identity and queue changes**

```python
_SUPPORTED_MARKETS = {"US", "HK", "JP", "TW", "IN", "SG", "CN"}

_MARKET_DEFAULTS.update({
    "IN": ("INR", "Asia/Kolkata"),
    "SG": ("SGD", "Asia/Singapore"),
    "CN": ("CNY", "Asia/Shanghai"),
})

_MARKET_BY_EXCHANGE.update({
    "NSE": "IN",
    "XNSE": "IN",
    "SGX": "SG",
    "SES": "SG",
    "XSES": "SG",
    "SSE": "CN",
    "XSHG": "CN",
    "SZSE": "CN",
    "XSHE": "CN",
})

_MARKET_BY_SUFFIX = (
    (".HK", "HK"),
    (".NS", "IN"),
    (".SI", "SG"),
    (".SS", "CN"),
    (".SZ", "CN"),
    (".TWO", "TW"),
    (".TW", "TW"),
    (".T", "JP"),
)
```

```python
_SUFFIX_BY_EXCHANGE.update({
    "XNSE": ".NS",
    "NSE": ".NS",
    "XSES": ".SI",
    "SGX": ".SI",
    "XSHG": ".SS",
    "SSE": ".SS",
    "XSHE": ".SZ",
    "SZSE": ".SZ",
})
```

```python
SUPPORTED_MARKETS: tuple[str, ...] = ("US", "HK", "JP", "TW", "IN", "SG", "CN")
```

```bash
-Q data_fetch_shared,data_fetch_us,data_fetch_hk,data_fetch_jp,data_fetch_tw,data_fetch_in,data_fetch_sg,data_fetch_cn
```

- [ ] **Step 4: Verify tests and shell syntax**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_security_master_service.py tests/unit/test_market_queues.py -q`
Expected: pass.

Run: `bash -n backend/start_celery.sh`
Expected: no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/security_master_service.py backend/app/tasks/market_queues.py backend/start_celery.sh backend/tests/unit/test_security_master_service.py backend/tests/unit/test_market_queues.py
git commit -m "feat: extend identity and queues for in sg cn"
```

### Task 4: Add Benchmark and Calendar Support

**Files:**
- Modify: `backend/app/services/benchmark_registry_service.py`
- Modify: `backend/app/services/market_calendar_service.py`
- Modify: `backend/requirements-runtime.txt`
- Modify: `backend/requirements-server.txt`
- Test: `backend/tests/unit/test_benchmark_registry_service.py`
- Test: `backend/tests/unit/test_market_calendar_service.py`
- Test: `backend/tests/unit/test_market_calendar_service_engine.py`

- [ ] **Step 1: Write failing tests for benchmark candidates and calendar IDs**

```python
def test_registry_mapping_exists_for_new_markets():
    table = benchmark_registry.mapping_table()
    assert set(table.keys()) == {"US", "HK", "JP", "TW", "IN", "SG", "CN"}
    assert benchmark_registry.get_candidate_symbols("IN") == ["^NSEI", "NIFTYBEES.NS"]
    assert benchmark_registry.get_candidate_symbols("SG") == ["^STI", "ES3.SI"]
    assert benchmark_registry.get_candidate_symbols("CN") == ["000300.SS", "510300.SS"]
```

```python
def test_market_calendar_service_uses_new_market_ids():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    assert service.calendar_id("IN") == "XNSE"
    assert service.calendar_id("SG") == "XSES"
    assert service.calendar_id("CN") == "XSHG"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_benchmark_registry_service.py tests/unit/test_market_calendar_service.py -q`
Expected: failures for unsupported markets.

- [ ] **Step 3: Implement the registry and calendar changes**

```python
_TABLE["IN"] = BenchmarkRegistryEntry(
    market="IN",
    primary_symbol="^NSEI",
    primary_kind="index",
    fallback_symbol="NIFTYBEES.NS",
    fallback_kind="etf",
    notes="Nifty 50 primary with NIFTYBEES ETF fallback.",
)
_TABLE["SG"] = BenchmarkRegistryEntry(
    market="SG",
    primary_symbol="^STI",
    primary_kind="index",
    fallback_symbol="ES3.SI",
    fallback_kind="etf",
    notes="STI primary with SPDR STI ETF fallback.",
)
_TABLE["CN"] = BenchmarkRegistryEntry(
    market="CN",
    primary_symbol="000300.SS",
    primary_kind="index",
    fallback_symbol="510300.SS",
    fallback_kind="etf",
    notes="CSI 300 primary with CSI 300 ETF fallback.",
)
```

```python
CALENDAR_ID_BY_MARKET.update({
    "IN": "XNSE",
    "SG": "XSES",
    "CN": "XSHG",
})
TIMEZONE_BY_MARKET.update({
    "IN": "Asia/Kolkata",
    "SG": "Asia/Singapore",
    "CN": "Asia/Shanghai",
})
```

```python
try:
    import pandas_market_calendars as pmc
except ModuleNotFoundError:  # pragma: no cover
    pmc = None
```

```python
if calendar_id == "XNSE" and self._calendar_provider is None and pmc is not None:
    self._calendar_cache[calendar_id] = pmc.get_calendar("XNSE")
```

- [ ] **Step 4: Pin the missing calendar dependency and rerun tests**

```text
pandas_market_calendars==5.3.0
```

Run: `cd backend && ./venv/bin/pytest tests/unit/test_benchmark_registry_service.py tests/unit/test_market_calendar_service.py tests/unit/test_market_calendar_service_engine.py -q`
Expected: pass, including India fallback-path tests.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/benchmark_registry_service.py backend/app/services/market_calendar_service.py backend/requirements-runtime.txt backend/requirements-server.txt backend/tests/unit/test_benchmark_registry_service.py backend/tests/unit/test_market_calendar_service.py backend/tests/unit/test_market_calendar_service_engine.py
git commit -m "feat: add benchmark and calendar support for in sg cn"
```

### Task 5: Extend Provider Routing and Growth Cadence Policy

**Files:**
- Modify: `backend/app/services/provider_routing_policy.py`
- Modify: `backend/app/services/growth_cadence_service.py`
- Test: `backend/tests/unit/test_provider_routing_policy.py`
- Test: `backend/tests/unit/test_growth_cadence_service.py`
- Test: `backend/tests/unit/test_data_source_service_policy.py`

- [ ] **Step 1: Add failing tests for routing and cadence**

```python
@pytest.mark.parametrize("market", ["IN", "SG", "CN"])
def test_new_markets_are_yfinance_only(market):
    assert providers_for(market) == (PROVIDER_YFINANCE,)
```

```python
def test_in_semiannual_cadence_uses_comparable_period_yoy():
    cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-06-30"), pd.Timestamp("2024-12-31")]
    df = _income_frame(cols, eps=[1.2, 1.0, 0.8], revenue=[120, 110, 100])
    result = compute_cadence_aware_growth(df, market="IN")
    assert result["growth_reporting_cadence"] == CADENCE_SEMIANNUAL
    assert result["growth_metric_basis"] == BASIS_COMPARABLE_YOY
    assert result["eps_growth_qq"] == 50.0
```

- [ ] **Step 2: Run the tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_provider_routing_policy.py tests/unit/test_growth_cadence_service.py tests/unit/test_data_source_service_policy.py -q`
Expected: failures because the new markets are unknown and India/Singapore cadence still falls back to US semantics.

- [ ] **Step 3: Implement the routing and cadence changes**

```python
MARKET_IN = "IN"
MARKET_SG = "SG"
MARKET_CN = "CN"

KNOWN_MARKETS = frozenset({MARKET_US, MARKET_HK, MARKET_JP, MARKET_TW, MARKET_IN, MARKET_SG, MARKET_CN})

_POLICY_MATRIX = {
    MARKET_US: (PROVIDER_FINVIZ, PROVIDER_YFINANCE, PROVIDER_ALPHAVANTAGE),
    MARKET_HK: (PROVIDER_YFINANCE,),
    MARKET_JP: (PROVIDER_YFINANCE,),
    MARKET_TW: (PROVIDER_YFINANCE,),
    MARKET_IN: (PROVIDER_YFINANCE,),
    MARKET_SG: (PROVIDER_YFINANCE,),
    MARKET_CN: (PROVIDER_YFINANCE,),
}
```

```python
_MARKETS_COMPARABLE_PERIOD_PRIMARY = frozenset(
    {
        routing_policy.MARKET_HK,
        routing_policy.MARKET_JP,
        routing_policy.MARKET_IN,
        routing_policy.MARKET_SG,
    }
)
```

- [ ] **Step 4: Re-run the focused tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_provider_routing_policy.py tests/unit/test_growth_cadence_service.py tests/unit/test_data_source_service_policy.py -q`
Expected: pass, with explicit yfinance-only routing and correct semiannual fallback handling for `IN` and `SG`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/provider_routing_policy.py backend/app/services/growth_cadence_service.py backend/tests/unit/test_provider_routing_policy.py backend/tests/unit/test_growth_cadence_service.py backend/tests/unit/test_data_source_service_policy.py
git commit -m "feat: extend routing and cadence policy for in sg cn"
```

### Task 6: Implement India Official Universe Ingestion (NSE-First)

**Files:**
- Modify: `backend/app/config/settings.py`
- Modify: `backend/app/services/official_market_universe_source_service.py`
- Create: `backend/app/services/in_universe_ingestion_adapter.py`
- Modify: `backend/app/services/stock_universe_service.py`
- Modify: `backend/app/tasks/universe_tasks.py`
- Create: `backend/tests/unit/fixtures/universe_sources/nse_equity_l_fixture.csv`
- Test: `backend/tests/unit/test_official_market_universe_source_service.py`
- Test: `backend/tests/unit/test_stock_universe_service.py`
- Test: `backend/tests/unit/test_universe_tasks.py`

- [ ] **Step 1: Add fixture-driven failing tests for NSE parsing and ingestion**

```python
def test_fetch_in_snapshot_parses_nse_equity_rows(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    fetched = _FetchedSource(
        url="https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        content=_fixture_bytes("nse_equity_l_fixture.csv"),
        fetched_at="2026-04-19T00:00:00+00:00",
        last_modified="Sat, 18 Apr 2026 00:00:00 GMT",
        tls_verification_disabled=False,
    )
    monkeypatch.setattr(service, "_http_get", lambda *args, **kwargs: fetched)
    snapshot = service.fetch_market_snapshot("IN")
    assert snapshot.source_name == "nse_official"
    assert {"RELIANCE.NS", "TCS.NS"} <= {row["symbol"] for row in snapshot.rows}
```

- [ ] **Step 2: Run the official-source and ingestion tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_official_market_universe_source_service.py tests/unit/test_stock_universe_service.py tests/unit/test_universe_tasks.py -q`
Expected: failures because `IN` official refresh is unsupported.

- [ ] **Step 3: Implement the NSE source and adapter**

```python
nse_universe_source_url: str = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
```

```python
if normalized_market == "IN":
    return self.fetch_in_snapshot()
```

```python
def fetch_in_snapshot(self) -> OfficialMarketUniverseSnapshot:
    fetched = self._http_get(settings.nse_universe_source_url)
    frame = pd.read_csv(io.BytesIO(fetched.content))
    rows = self.parse_in_rows(frame)
    snapshot_as_of = (self._date_from_http_header(fetched.last_modified) or self._utc_today()).isoformat()
    return OfficialMarketUniverseSnapshot(
        market="IN",
        source_name="nse_official",
        snapshot_id=f"nse-equity-l-{snapshot_as_of}",
        snapshot_as_of=snapshot_as_of,
        source_metadata={"source_urls": [settings.nse_universe_source_url]},
        rows=tuple(rows),
    )
```

```python
_OFFICIAL_SOURCE_MARKETS = {"HK", "JP", "TW", "IN"}
```

- [ ] **Step 4: Re-run the tests and one targeted live smoke**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_official_market_universe_source_service.py tests/unit/test_stock_universe_service.py tests/unit/test_universe_tasks.py -q`
Expected: pass.

Run: `cd backend && ./venv/bin/python - <<'PY'\nfrom app.services.official_market_universe_source_service import OfficialMarketUniverseSourceService\nsnap = OfficialMarketUniverseSourceService().fetch_market_snapshot('IN')\nprint(snap.market, snap.source_name, len(snap.rows))\nprint(snap.rows[0]['symbol'])\nPY`
Expected: `IN nse_official` followed by a positive row count and an `.NS` symbol.

- [ ] **Step 5: Commit**

```bash
git add backend/app/config/settings.py backend/app/services/official_market_universe_source_service.py backend/app/services/in_universe_ingestion_adapter.py backend/app/services/stock_universe_service.py backend/app/tasks/universe_tasks.py backend/tests/unit/fixtures/universe_sources/nse_equity_l_fixture.csv backend/tests/unit/test_official_market_universe_source_service.py backend/tests/unit/test_stock_universe_service.py backend/tests/unit/test_universe_tasks.py
git commit -m "feat: add nse official universe ingestion"
```

### Task 7: Timebox Singapore Source Discovery, Then Implement the SG Adapter

**Files:**
- Modify: `backend/app/config/settings.py`
- Modify: `backend/app/services/official_market_universe_source_service.py`
- Create: `backend/app/services/sg_universe_ingestion_adapter.py`
- Modify: `backend/app/services/stock_universe_service.py`
- Create: `backend/tests/unit/fixtures/universe_sources/sgx_equities_fixture.json`
- Test: `backend/tests/unit/test_official_market_universe_source_service.py`
- Modify: `docs/asia/asia_v3_in_sg_cn_source_matrix.md`

- [ ] **Step 1: Prove the SGX source is machine-readable before touching runtime enablement**

Run: `curl -I https://www.sgx.com/stock-screener`
Expected: reachable HTML shell only.

Run: `rg -n "Screener.svc|SECURITIES_API_URL|getMetaDataByStockCode" /tmp/sgx-bundle.js`
Expected: evidence of the client-side data API from the downloaded bundle.

Acceptance for this step:
- identify a stable unauthenticated endpoint or downloadable file
- confirm it returns listed-equity rows headlessly
- record the exact URL and response shape in `docs/asia/asia_v3_in_sg_cn_source_matrix.md`

- [ ] **Step 2: Add a failing fixture-driven parsing test**

```python
def test_fetch_sg_snapshot_parses_equity_rows(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    fetched = _FetchedSource(
        url="https://www.sgx.com/stock-screener",
        content=_fixture_bytes("sgx_equities_fixture.json"),
        fetched_at="2026-04-19T00:00:00+00:00",
        last_modified=None,
        tls_verification_disabled=False,
    )
    monkeypatch.setattr(service, "_http_get", lambda *args, **kwargs: fetched)
    snapshot = service.fetch_market_snapshot("SG")
    assert snapshot.source_name == "sgx_official"
    assert {"D05.SI", "C6L.SI"} <= {row["symbol"] for row in snapshot.rows}
```

- [ ] **Step 3: Implement the SG parser only after the endpoint is proven**

```python
sg_universe_source_url: str = ""
```

```python
if normalized_market == "SG":
    return self.fetch_sg_snapshot()
```

```python
def fetch_sg_snapshot(self) -> OfficialMarketUniverseSnapshot:
    if not settings.sg_universe_source_url:
        raise ValueError("SG official universe source is not configured")
    fetched = self._http_get(settings.sg_universe_source_url)
    rows = self.parse_sg_rows(fetched.content)
    snapshot_as_of = self._utc_today().isoformat()
    return OfficialMarketUniverseSnapshot(
        market="SG",
        source_name="sgx_official",
        snapshot_id=f"sgx-equities-{snapshot_as_of}",
        snapshot_as_of=snapshot_as_of,
        source_metadata={"source_urls": [settings.sg_universe_source_url]},
        rows=tuple(rows),
    )
```

- [ ] **Step 4: Run tests and stop if the source cannot be validated**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_official_market_universe_source_service.py -k "sg" -q`
Expected: pass **only** if the source is validated and fixtures match the real response shape.

If source proof fails:
- do **not** merge `SG` runtime enablement
- update `docs/asia/asia_v3_in_sg_cn_source_matrix.md` with the blocker
- create a blocking bead issue for SGX source licensing/discovery

- [ ] **Step 5: Commit**

```bash
git add backend/app/config/settings.py backend/app/services/official_market_universe_source_service.py backend/app/services/sg_universe_ingestion_adapter.py backend/app/services/stock_universe_service.py backend/tests/unit/fixtures/universe_sources/sgx_equities_fixture.json backend/tests/unit/test_official_market_universe_source_service.py docs/asia/asia_v3_in_sg_cn_source_matrix.md
git commit -m "feat: add sg official universe ingestion"
```

### Task 8: Implement China Source Aggregation and CN Adapter

**Files:**
- Modify: `backend/app/config/settings.py`
- Modify: `backend/app/services/official_market_universe_source_service.py`
- Create: `backend/app/services/cn_universe_ingestion_adapter.py`
- Modify: `backend/app/services/stock_universe_service.py`
- Create: `backend/tests/unit/fixtures/universe_sources/sse_equities_fixture.csv`
- Create: `backend/tests/unit/fixtures/universe_sources/szse_equities_fixture.csv`
- Test: `backend/tests/unit/test_official_market_universe_source_service.py`
- Test: `backend/tests/unit/test_stock_universe_service.py`

- [ ] **Step 1: Validate both SSE and SZSE source shapes**

Run: `curl -I https://www.szse.cn/English/services/dataServices/index.html`
Expected: reachable official SZSE data-services page.

Run: `curl -I https://english.sse.com.cn/markets/equities/list/quotation/`
Expected: reachable SSE quotation page.

Acceptance for this step:
- locate the exact machine-readable listing source for SSE
- keep SZSE and SSE raw fixtures separate
- do not collapse both exchanges into one parser before the raw shapes are known

- [ ] **Step 2: Add failing tests for combined CN snapshot ingestion**

```python
def test_fetch_cn_snapshot_merges_sse_and_szse_rows(monkeypatch):
    service = OfficialMarketUniverseSourceService()
    # monkeypatch the per-exchange fetch helpers to return frozen CSV fixtures
    snapshot = service.fetch_market_snapshot("CN")
    symbols = {row["symbol"] for row in snapshot.rows}
    assert "600519.SS" in symbols
    assert "000858.SZ" in symbols
    assert snapshot.source_name == "cn_reference_bundle"
```

- [ ] **Step 3: Implement the CN bundle and canonicalization**

```python
sse_universe_source_url: str = ""
szse_universe_source_url: str = ""
```

```python
def fetch_cn_snapshot(self) -> OfficialMarketUniverseSnapshot:
    if not settings.sse_universe_source_url or not settings.szse_universe_source_url:
        raise ValueError("CN official universe sources are not fully configured")
    sse_rows = self.fetch_sse_rows()
    szse_rows = self.fetch_szse_rows()
    snapshot_as_of = self._utc_today().isoformat()
    return OfficialMarketUniverseSnapshot(
        market="CN",
        source_name="cn_reference_bundle",
        snapshot_id=f"cn-reference-bundle-{snapshot_as_of}",
        snapshot_as_of=snapshot_as_of,
        source_metadata={"sources": ["sse_official", "szse_official"]},
        rows=tuple([*sse_rows, *szse_rows]),
    )
```

```python
MARKET_EXCHANGE_FALLBACKS["CN"] = ("SSE", "XSHG", "SZSE", "XSHE")
```

- [ ] **Step 4: Re-run tests and a live parser smoke**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_official_market_universe_source_service.py tests/unit/test_stock_universe_service.py -k "cn or sse or szse" -q`
Expected: pass with both `.SS` and `.SZ` preserved.

Run: `cd backend && ./venv/bin/python - <<'PY'\nfrom app.services.security_master_service import security_master_resolver\nprint(security_master_resolver.resolve_identity(symbol='600519', exchange='XSHG').canonical_symbol)\nprint(security_master_resolver.resolve_identity(symbol='000858', exchange='XSHE').canonical_symbol)\nPY`
Expected:
```text
600519.SS
000858.SZ
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/config/settings.py backend/app/services/official_market_universe_source_service.py backend/app/services/cn_universe_ingestion_adapter.py backend/app/services/stock_universe_service.py backend/tests/unit/fixtures/universe_sources/sse_equities_fixture.csv backend/tests/unit/fixtures/universe_sources/szse_equities_fixture.csv backend/tests/unit/test_official_market_universe_source_service.py backend/tests/unit/test_stock_universe_service.py
git commit -m "feat: add cn official universe ingestion"
```

### Task 9: Wire Bootstrap, Runtime Preferences, and Product Guardrails

**Files:**
- Modify: `backend/app/tasks/runtime_bootstrap_tasks.py`
- Modify: `backend/app/tasks/universe_tasks.py`
- Modify: `backend/app/domain/analytics/scope.py`
- Modify: `backend/app/tasks/breadth_tasks.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Modify: `frontend/src/features/scan/constants.js`
- Modify: `frontend/src/contexts/RuntimeContext.jsx`
- Test: `backend/tests/unit/test_runtime_bootstrap_tasks.py`
- Test: `backend/tests/unit/test_breadth_tasks.py`
- Test: `backend/tests/unit/test_group_rank_tasks.py`

- [ ] **Step 1: Add failing tests for bootstrap and US-only guard behavior**

```python
def test_bootstrap_queues_official_refresh_for_in_sg_cn(monkeypatch):
    signatures = _build_market_bootstrap_signatures("IN")
    names = [sig.task for sig in signatures]
    assert "app.tasks.universe_tasks.refresh_official_market_universe" in names
```

```python
@pytest.mark.parametrize("market", ["IN", "SG", "CN"])
def test_breadth_and_group_rankings_stay_explicitly_us_only(market):
    assert calculate_daily_breadth.run(market=market)["reason"] == "breadth_calculation_is_us_only"
    assert calculate_daily_group_rankings.run(market=market)["reason"] == "group_rankings_are_us_only"
```

- [ ] **Step 2: Run the targeted tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_runtime_bootstrap_tasks.py tests/unit/test_breadth_tasks.py tests/unit/test_group_rank_tasks.py -q`
Expected: failures where the new markets are not yet wired or the guard behavior is not fully covered.

- [ ] **Step 3: Implement the bootstrap wiring and explicit guardrails**

```python
if market == "US":
    ...
else:
    refresh_official_market_universe.si(market=market, activity_lifecycle="bootstrap")
```

```python
_US_ONLY_FEATURES = {
    AnalyticsFeature.THEME_DISCOVERY: "...",
    AnalyticsFeature.IBD_GROUP_RANK: "...",
    AnalyticsFeature.BREADTH_SNAPSHOT: "...",
}
```

```javascript
SG: [{ value: 'market', label: 'All Singapore' }],
IN: [{ value: 'market', label: 'All India' }],
CN: [{ value: 'market', label: 'All China' }],
```

- [ ] **Step 4: Re-run tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_runtime_bootstrap_tasks.py tests/unit/test_breadth_tasks.py tests/unit/test_group_rank_tasks.py -q`
Expected: pass, with new markets bootstrapping correctly and analytics still blocked by policy.

- [ ] **Step 5: Commit**

```bash
git add backend/app/tasks/runtime_bootstrap_tasks.py backend/app/tasks/universe_tasks.py backend/app/domain/analytics/scope.py backend/app/tasks/breadth_tasks.py backend/app/tasks/group_rank_tasks.py frontend/src/features/scan/constants.js frontend/src/contexts/RuntimeContext.jsx backend/tests/unit/test_runtime_bootstrap_tasks.py backend/tests/unit/test_breadth_tasks.py backend/tests/unit/test_group_rank_tasks.py
git commit -m "feat: wire bootstrap and guardrails for in sg cn"
```

## Final Verification Checklist

- [ ] Run backend focused suite:

```bash
cd backend && ./venv/bin/pytest \
  tests/unit/test_security_master_service.py \
  tests/unit/test_market_queues.py \
  tests/unit/test_benchmark_registry_service.py \
  tests/unit/test_market_calendar_service.py \
  tests/unit/test_market_calendar_service_engine.py \
  tests/unit/test_provider_routing_policy.py \
  tests/unit/test_growth_cadence_service.py \
  tests/unit/test_data_source_service_policy.py \
  tests/unit/test_official_market_universe_source_service.py \
  tests/unit/test_stock_universe_service.py \
  tests/unit/test_universe_tasks.py \
  tests/unit/test_runtime_bootstrap_tasks.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_group_rank_tasks.py -q
```

Expected: all focused tests pass.

- [ ] Run frontend tests:

```bash
cd frontend && npm run test:run -- RuntimeContext.test.jsx
```

Expected: pass.

- [ ] Run live smoke checks for each market after implementation:

```bash
cd backend && ./venv/bin/python - <<'PY'
from app.services.data_source_service import DataSourceService
svc = DataSourceService()
for symbol, market in [("RELIANCE.NS", "IN"), ("D05.SI", "SG"), ("600519.SS", "CN")]:
    fundamentals = svc.get_fundamentals(symbol, market=market)
    growth = svc.get_quarterly_growth(symbol, market=market)
    print(market, symbol, bool(fundamentals), growth.get("growth_metric_basis"))
PY
```

Expected: non-empty fundamentals for all three markets, and a stable growth basis (`quarterly_qoq` or `comparable_period_yoy`) without unknown-market warnings.

- [ ] Run session landing workflow:

```bash
git status
git pull --rebase
bd sync
git push
git status
```

Expected: branch is up to date with origin.

## Open Questions To Resolve During Implementation

1. Should `CN` be a single market backed by `XSHG` calendar semantics, or should the repo grow `CN-SSE` / `CN-SZSE` sub-markets later? This plan keeps one `CN` market for minimal disruption.
2. Does SGX expose a stable official listed-equities endpoint without licensing restrictions? If not, `SG` must remain disabled.
3. Is NSE-first India enough for launch, or does product want BSE parity before user-facing enablement? This plan assumes NSE-first.

## Success Definition

- `IN`, `SG`, and `CN` are accepted by the backend and frontend market contracts.
- `SecurityMaster`, benchmarks, calendars, provider routing, and bootstrap are market-correct.
- India is fully launchable once tests pass.
- Singapore and China are only enabled after their official-universe sources are proven.
- US-only analytics remain explicitly fenced and observable.
