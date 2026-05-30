# Australia Market Universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Australia (`AU`) as a harmonized market whose universe is fetched live from ASX's public listed-companies CSV before AU prices, fundamentals, breadth, groups, or feature snapshots run, with a committed CSV fallback for degraded/offline runs.

**Architecture:** AU follows the existing Market Catalog, MIC alias, symbol suffix, provider data plan, official source service, market-specific universe adapter, and universe task dispatch patterns. ASX's public CSV is the source of truth for v1; the live fetch emits `source_name == "asx_official_public_csv"` and falls back to `data/au_asx_listed_companies.csv` only when the live URL is blank, unreachable, empty, or below the configured size floor.

**Tech Stack:** Python 3, SQLAlchemy, Celery, Pydantic settings, yfinance provider plans, ASX CSV over HTTP, pytest.

---

## Source Decision

Use the ASX public CSV:

```text
https://www.asx.com.au/asx/research/ASXListedCompanies.csv
```

The CSV currently includes a metadata preamble line, then a header row:

```csv
Company name,ASX code,GICS industry group
```

The AU parser must locate that header row dynamically rather than assuming fixed line numbers. Each accepted row maps:

```python
{
    "symbol": f"{asx_code}.AX",
    "local_code": asx_code,
    "name": company_name,
    "exchange": "XASX",
    "industry": gics_industry_group,
    "gics_industry_group": gics_industry_group,
}
```

Keep ASX ReferencePoint / Daily Official List as the future enterprise-grade option, not the v1 dependency.

## File Structure

Create:

- `backend/app/services/au_universe_ingestion_adapter.py` - AU-specific canonicalizer for ASX/Yahoo `.AX` symbols.
- `backend/tests/unit/test_au_universe_ingestion_adapter.py` - adapter unit tests.
- `backend/tests/unit/fixtures/universe_sources/asx_listed_companies_fixture.csv` - small parser fixture with ASX preamble/header shape.
- `data/au_asx_listed_companies.csv` - broad fallback CSV copied from the public ASX CSV format.

Modify:

- `backend/app/config/settings.py` - AU source URL, fallback path, live min-size floor, rate-limit knobs, cache warm defaults.
- `backend/app/domain/markets/catalog.py` - AU Market Catalog entry with MIC/currency/timezone/capabilities.
- `backend/app/domain/markets/mic_aliases.py` - `ASX -> XASX` scoped alias.
- `backend/app/domain/markets/symbol_suffixes.py` - `.AX` suffix mapping for AU.
- `backend/app/domain/markets/registry.py` - AU benchmark facts.
- `backend/app/domain/universe/indexes.py` - ASX 200 index definition.
- `backend/app/domain/providers/data_plan.py` - yfinance-only AU fundamentals and prices plan; bump `PLAN_VERSION`.
- `backend/app/services/provider_routing_policy.py` - compatibility constant `MARKET_AU`.
- `backend/app/services/rate_budget_policy.py` - AU default batch sizes, workers, and backoff.
- `backend/app/services/official_market_universe_source_service.py` - live-first ASX CSV fetch and fallback parser.
- `backend/app/services/stock_universe_service.py` - wire AU adapter into the shared ingestion pipeline.
- `backend/app/tasks/universe_tasks.py` - dispatch AU official universe snapshots to `ingest_au_snapshot_rows`.
- Existing tests under `backend/tests/unit/domain/`, `backend/tests/unit/domain/markets/`, `backend/tests/unit/test_official_market_universe_source_service.py`, `backend/tests/unit/test_stock_universe_service.py`, and `backend/tests/unit/test_universe_tasks.py`.

---

### Task 1: Add AU Market Facts

**Files:**
- Modify: `backend/app/domain/markets/catalog.py`
- Modify: `backend/app/domain/markets/mic_aliases.py`
- Modify: `backend/app/domain/markets/symbol_suffixes.py`
- Modify: `backend/app/domain/markets/registry.py`
- Modify: `backend/app/domain/universe/indexes.py`
- Modify: `backend/tests/unit/domain/test_market_catalog.py`
- Modify: `backend/tests/unit/domain/markets/test_mic_aliases.py`
- Modify: `backend/tests/unit/domain/markets/test_symbol_suffixes.py`
- Modify: `backend/tests/unit/domain/markets/test_market_registry.py`
- Modify: `backend/tests/unit/domain/universe/test_indexes.py`

- [ ] **Step 1: Write failing tests for AU market identity**

Add assertions that AU is a supported market, uses `XASX`, resolves `ASX` aliases, uses `.AX`, and maps ASX 200 to AU:

```python
def test_au_market_catalog_entry_is_harmonized():
    entry = get_market_catalog().get("AU")
    assert entry.label == "Australia"
    assert entry.primary_mic == "XASX"
    assert entry.mics == ("XASX",)
    assert entry.supported_currencies == ("AUD",)
    assert entry.default_currency == "AUD"
    assert entry.timezone == "Australia/Sydney"
    assert "ASX" in entry.exchanges
    assert entry.capabilities.official_universe is True
    assert entry.capabilities.fundamentals is True
    assert entry.capabilities.feature_snapshot is True
```

```python
def test_au_mic_aliases_and_symbol_suffixes():
    assert mic_alias_registry.resolve("AU", "ASX").mic == "XASX"
    assert mic_alias_registry.resolve("AU", "XASX").mic == "XASX"
    assert market_symbol_suffix_registry.suffix_for("AU", "ASX") == ".AX"
    assert market_symbol_suffix_registry.suffix_for("AU", None) == ".AX"
    assert market_symbol_suffix_registry.market_for_symbol("BHP.AX") == "AU"
    assert market_symbol_suffix_registry.mic_for_symbol("BHP.AX") == "XASX"
```

```python
def test_au_registry_and_index_mapping():
    assert market_registry.market_for_exchange("ASX") == Market("AU")
    assert market_registry.mic_for_exchange("AU", "ASX") == "XASX"
    assert market_registry.market_for_index("ASX200") == Market("AU")
    assert market_registry.benchmark_for("AU").symbol == "^AXJO"
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/domain/test_market_catalog.py \
  tests/unit/domain/markets/test_mic_aliases.py \
  tests/unit/domain/markets/test_symbol_suffixes.py \
  tests/unit/domain/markets/test_market_registry.py \
  tests/unit/domain/universe/test_indexes.py -q
```

Expected: FAIL because `AU`, `XASX`, `.AX`, and ASX 200 are not registered yet.

- [ ] **Step 3: Add AU catalog and registry entries**

Insert the market entry after MY or beside the other same-region markets:

```python
_market_entry(
    code="AU",
    label="Australia",
    primary_mic="XASX",
    mic_facts=(
        _mic_facts(
            "XASX",
            timezone="Australia/Sydney",
            default_currency="AUD",
        ),
    ),
    exchanges=("ASX", "XASX"),
    capabilities=MarketCapabilities(
        benchmark=True,
        breadth=False,
        fundamentals=True,
        group_rankings=False,
        feature_snapshot=True,
        official_universe=True,
        finviz_screening=False,
    ),
),
```

Add the MIC alias:

```python
MicAliasDefinition("AU", "XASX", ("ASX",)),
```

Add the symbol suffix:

```python
MarketSymbolSuffixDefinition(
    "AU", ".AX", ("ASX", "XASX"), mic="XASX", is_default=True
),
```

Add benchmark facts:

```python
"AU": BenchmarkFacts("^AXJO", "IOZ.AX", "index", "etf"),
```

Add ASX 200 to `backend/app/domain/universe/indexes.py`:

```python
IndexDefinition(
    key="ASX200",
    market="AU",
    display_name="S&P/ASX 200",
    aliases=("ASX 200", "S&P ASX 200", "XJO", "AXJO"),
),
```

- [ ] **Step 4: Run tests and confirm they pass**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/domain/markets/catalog.py \
  backend/app/domain/markets/mic_aliases.py \
  backend/app/domain/markets/symbol_suffixes.py \
  backend/app/domain/markets/registry.py \
  backend/app/domain/universe/indexes.py \
  backend/tests/unit/domain/test_market_catalog.py \
  backend/tests/unit/domain/markets/test_mic_aliases.py \
  backend/tests/unit/domain/markets/test_symbol_suffixes.py \
  backend/tests/unit/domain/markets/test_market_registry.py \
  backend/tests/unit/domain/universe/test_indexes.py
git commit -m "feat: register australia market facts"
```

### Task 2: Add AU Provider and Rate Plans

**Files:**
- Modify: `backend/app/domain/providers/data_plan.py`
- Modify: `backend/app/services/provider_routing_policy.py`
- Modify: `backend/app/services/rate_budget_policy.py`
- Modify: `backend/app/config/settings.py`
- Modify: `backend/tests/unit/domain/test_provider_data_plan.py`
- Modify: `backend/tests/unit/test_rate_budget_policy.py`

- [ ] **Step 1: Write failing provider-plan tests**

Add AU assertions:

```python
def test_au_provider_plan_uses_yfinance_only():
    fundamentals = provider_data_plan_registry.plan_for("AU", DATASET_FUNDAMENTALS)
    prices = provider_data_plan_registry.plan_for("AU", DATASET_PRICES)

    assert fundamentals.providers == (PROVIDER_YFINANCE,)
    assert fundamentals.step_for(PROVIDER_YFINANCE).batch_size == 50
    assert prices.providers == (PROVIDER_YFINANCE,)
    assert prices.step_for(PROVIDER_YFINANCE).batch_size == 50
```

Add rate-budget assertions:

```python
def test_au_rate_budget_defaults_match_yfinance_peer_markets():
    policy = RateBudgetPolicy(redis_client_factory=lambda: None)
    assert policy.batch_size("yfinance", "AU") == 50
    assert policy.get_provider_workers("yfinance", "AU") == 1
    assert policy.get_provider_workers("finviz", "AU") == 2
    assert policy.backoff("yfinance", "AU")["max_s"] == 600
    assert policy.backoff("finviz", "AU")["max_s"] == 480
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/domain/test_provider_data_plan.py \
  tests/unit/test_rate_budget_policy.py -q
```

Expected: FAIL because AU has no provider/rate defaults yet.

- [ ] **Step 3: Add AU provider/rate settings**

In `backend/app/domain/providers/data_plan.py`, bump:

```python
PLAN_VERSION = "2026.05.30.1"
```

Add:

```python
("AU", DATASET_FUNDAMENTALS): (_yf(),),
("AU", DATASET_PRICES): (_yf(batch_size=50),),
```

In `backend/app/services/provider_routing_policy.py`, add:

```python
MARKET_AU = "AU"
```

In `backend/app/config/settings.py`, add AU knobs next to same-market settings:

```python
yfinance_rate_limit_au: float | None = None
finviz_rate_limit_au: float | None = None
yfinance_batch_size_au: int | None = None
yfinance_batch_rate_limit_au: float | None = None
yfinance_backoff_max_s_au: int | None = None
finviz_workers_au: int | None = None
cache_warm_hour_au: int = 7
cache_warm_minute_au: int = 0
```

In `backend/app/services/rate_budget_policy.py`, add AU to every built-in dict:

```python
"yfinance": {"AU": 50, ...}
"finviz": {"AU": 50, ...}
"yfinance": {"AU": 1, ...}
"finviz": {"AU": 2, ...}
"AU": {"base_s": 60, "max_s": 600, "factor": 2.0}
"AU": {"base_s": 60, "max_s": 480, "factor": 2.0}
```

- [ ] **Step 4: Run tests and confirm they pass**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/domain/providers/data_plan.py \
  backend/app/services/provider_routing_policy.py \
  backend/app/services/rate_budget_policy.py \
  backend/app/config/settings.py \
  backend/tests/unit/domain/test_provider_data_plan.py \
  backend/tests/unit/test_rate_budget_policy.py
git commit -m "feat: add australia provider data plan"
```

### Task 3: Add AU Universe Ingestion Adapter

**Files:**
- Create: `backend/app/services/au_universe_ingestion_adapter.py`
- Create: `backend/tests/unit/test_au_universe_ingestion_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

Use tests parallel to SG because AU can use the shared `CanonicalUniverseRow` model:

```python
from __future__ import annotations

import pytest

from app.services.au_universe_ingestion_adapter import au_universe_ingestion_adapter


def test_au_adapter_canonicalizes_asx_rows_with_metadata():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {
                "symbol": "BHP",
                "name": "BHP Group Limited",
                "exchange": "ASX",
                "industry": "Materials",
            },
            {
                "symbol": "CBA.AX",
                "name": "Commonwealth Bank of Australia",
                "exchange": "XASX",
                "industry": "Banks",
            },
        ],
        source_name="asx_official_public_csv",
        snapshot_id="asx-listed-2026-05-30",
        snapshot_as_of="2026-05-30",
        source_metadata={"row_counts": {"xasx": 2}},
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert set(by_symbol) == {"BHP.AX", "CBA.AX"}
    assert by_symbol["BHP.AX"].market == "AU"
    assert by_symbol["BHP.AX"].mic == "XASX"
    assert by_symbol["BHP.AX"].currency == "AUD"
    assert by_symbol["BHP.AX"].timezone == "Australia/Sydney"
    assert by_symbol["BHP.AX"].local_code == "BHP"
    assert by_symbol["BHP.AX"].industry == "Materials"
    assert by_symbol["BHP.AX"].provenance.source_metadata["row_counts"] == {"xasx": 2}


def test_au_adapter_rejects_invalid_exchange_and_symbol():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "BHP", "name": "Bad Exchange", "exchange": "NYSE"},
            {"symbol": "BAD!@#", "name": "Bad Symbol", "exchange": "XASX"},
            {"symbol": "", "name": "Missing Symbol", "exchange": "XASX"},
            {"symbol": "TOOLONGSYMBOL", "name": "Too long", "exchange": "XASX"},
        ],
        source_name="asx_official_public_csv",
        snapshot_id="asx-listed-2026-05-30",
    )

    assert result.canonical_rows == ()
    reasons = [row.reason for row in result.rejected_rows]
    assert any("Unsupported AU exchange" in reason for reason in reasons)
    assert any("Invalid AU symbol" in reason for reason in reasons)
    assert any("Missing symbol" in reason for reason in reasons)


def test_au_adapter_deduplicates_deterministically():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "BHP", "name": "", "exchange": "XASX"},
            {"symbol": "BHP.AX", "name": "BHP Group Limited", "exchange": "XASX"},
        ],
        source_name="asx_official_public_csv",
        snapshot_id="asx-listed-2026-05-30",
    )

    assert result.rejected_rows == ()
    assert len(result.canonical_rows) == 1
    assert result.canonical_rows[0].symbol == "BHP.AX"
    assert result.canonical_rows[0].name == "BHP Group Limited"


def test_au_adapter_rejects_unapproved_source():
    with pytest.raises(ValueError, match="Unapproved AU source"):
        au_universe_ingestion_adapter.canonicalize_rows(
            [{"symbol": "BHP.AX", "name": "BHP"}],
            source_name="random_third_party",
            snapshot_id="asx-listed-2026-05-30",
        )
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_au_universe_ingestion_adapter.py -q
```

Expected: FAIL because the AU adapter module does not exist.

- [ ] **Step 3: Implement the AU adapter**

Create `backend/app/services/au_universe_ingestion_adapter.py` by following the SG adapter shape. The AU-specific constants and normalization must be:

```python
_AU_EXCHANGE_ALIASES: dict[str, str] = {
    "ASX": "XASX",
    "XASX": "XASX",
}

_APPROVED_AU_SOURCES: frozenset[str] = frozenset(
    {
        "asx_official_public_csv",
        "au_manual_csv",
        "au_reference_bundle",
        "asx_official",
    }
)

_AU_LOCAL_CODE_RE = re.compile(r"^[A-Z0-9]{2,6}$")
```

Use these AU-specific methods inside the same `canonicalize_rows` loop pattern as SG:

```python
@staticmethod
def _normalize_exchange(raw_exchange: Any) -> str:
    exchange = str(raw_exchange or "").strip().upper() or "XASX"
    normalized = _AU_EXCHANGE_ALIASES.get(exchange)
    if normalized is None:
        raise ValueError(
            f"Unsupported AU exchange '{exchange}'. Expected one of: ASX, XASX"
        )
    return normalized

@staticmethod
def _normalize_au_local_code(source_symbol: str) -> str:
    token = source_symbol
    for prefix in ("ASX:", "XASX:"):
        if token.startswith(prefix):
            token = token[len(prefix):]
            break
    if token.endswith(".AX"):
        token = token[:-3]

    if not _AU_LOCAL_CODE_RE.fullmatch(token):
        raise ValueError(
            f"Invalid AU symbol '{source_symbol}'. "
            "Expected 2-6 alphanumeric ASX local code with optional .AX suffix."
        )
    return token
```

Resolve identity with:

```python
identity = security_master_resolver.resolve_identity(
    symbol=f"{local_code}.AX",
    market="AU",
    exchange=exchange,
    local_code=local_code,
)
```

Export:

```python
au_universe_ingestion_adapter = AUUniverseIngestionAdapter()
```

- [ ] **Step 4: Run tests and confirm they pass**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/au_universe_ingestion_adapter.py \
  backend/tests/unit/test_au_universe_ingestion_adapter.py
git commit -m "feat: add australia universe adapter"
```

### Task 4: Add Live ASX Source Fetch With CSV Fallback

**Files:**
- Modify: `backend/app/config/settings.py`
- Modify: `backend/app/services/official_market_universe_source_service.py`
- Create: `backend/tests/unit/fixtures/universe_sources/asx_listed_companies_fixture.csv`
- Create: `data/au_asx_listed_companies.csv`
- Modify: `backend/tests/unit/test_official_market_universe_source_service.py`

- [ ] **Step 1: Write failing ASX source tests**

Add a fixture:

```csv
ASX listed companies as at Sat May 30 00:00:00 AEST 2026

Company name,ASX code,GICS industry group
"BHP GROUP LIMITED","BHP","Materials"
"COMMONWEALTH BANK OF AUSTRALIA.","CBA","Banks"
"1414 DEGREES LIMITED","14D","Capital Goods"
"NOT VALID LIMITED","TOOLONGSYMBOL","Capital Goods"
```

Add tests:

```python
def test_fetch_au_snapshot_parses_public_asx_csv(monkeypatch):
    content = _fixture_bytes("asx_listed_companies_fixture.csv")
    monkeypatch.setattr(app_settings, "au_universe_source_url", _ASX_CSV_URL)
    monkeypatch.setattr(app_settings, "au_live_min_universe_size", 0)
    service = OfficialMarketUniverseSourceService()
    monkeypatch.setattr(
        service,
        "_http_get",
        lambda url, **kwargs: _fetched_source(url=_ASX_CSV_URL, content=content),
    )

    snapshot = service.fetch_market_snapshot("AU")

    assert snapshot.market == "AU"
    assert snapshot.source_name == "asx_official_public_csv"
    assert snapshot.source_metadata["fetch_mode"] == "live_http"
    assert snapshot.snapshot_id.startswith("asx-listed-companies-")
    assert [row["symbol"] for row in snapshot.rows] == ["14D.AX", "BHP.AX", "CBA.AX"]
    assert snapshot.rows[0]["exchange"] == "XASX"
    assert snapshot.rows[0]["industry"] == "Capital Goods"
```

```python
def test_fetch_au_snapshot_falls_back_to_csv_when_live_fails(monkeypatch, tmp_path):
    fallback = tmp_path / "au_asx_listed_companies.csv"
    fallback.write_text(
        "Company name,ASX code,GICS industry group\n"
        "\"BHP GROUP LIMITED\",\"BHP\",\"Materials\"\n"
        "\"COMMONWEALTH BANK OF AUSTRALIA.\",\"CBA\",\"Banks\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "au_universe_source_url", _ASX_CSV_URL)
    monkeypatch.setattr(app_settings, "au_universe_fallback_csv_path", str(fallback))
    monkeypatch.setattr(app_settings, "au_live_min_universe_size", 0)
    service = OfficialMarketUniverseSourceService()

    def fail_live(url, **kwargs):
        raise requests.exceptions.ConnectionError("synthetic ASX outage")

    monkeypatch.setattr(service, "_http_get", fail_live)

    snapshot = service.fetch_market_snapshot("AU")

    assert snapshot.market == "AU"
    assert snapshot.source_name == "au_manual_csv"
    assert snapshot.source_metadata["fetch_mode"] == "csv_fallback"
    assert "synthetic ASX outage" in snapshot.source_metadata["fetch_errors"]["live_http"]
    assert {row["symbol"] for row in snapshot.rows} == {"BHP.AX", "CBA.AX"}
```

```python
def test_fetch_au_snapshot_rejects_tiny_fallback_when_floor_enabled(monkeypatch, tmp_path):
    fallback = tmp_path / "au_asx_listed_companies.csv"
    fallback.write_text(
        "Company name,ASX code,GICS industry group\n"
        "\"BHP GROUP LIMITED\",\"BHP\",\"Materials\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "au_universe_source_url", "")
    monkeypatch.setattr(app_settings, "au_universe_fallback_csv_path", str(fallback))
    monkeypatch.setattr(app_settings, "au_live_min_universe_size", 1500)

    with pytest.raises(ValueError, match="below 1500 threshold"):
        OfficialMarketUniverseSourceService().fetch_market_snapshot("AU")
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_official_market_universe_source_service.py -q
```

Expected: FAIL because AU settings and source fetch methods do not exist yet.

- [ ] **Step 3: Add AU settings**

Add next to SG/MY universe settings:

```python
# ASX public listed-company CSV. Live fetch is enabled by default because ASX
# publishes a stable CSV link from its official site; the repo CSV is fallback.
au_universe_source_url: str = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"
au_universe_fallback_csv_path: str = str(
    _PROJECT_ROOT / "data" / "au_asx_listed_companies.csv"
)
au_live_min_universe_size: int = 1500
```

- [ ] **Step 4: Implement AU source service methods**

In `official_market_universe_source_service.py`, add constants:

```python
_AU_SOURCE_NAME = "asx_official_public_csv"
_AU_FALLBACK_SOURCE_NAME = "au_manual_csv"
_AU_LIVE_TICKER_RE = re.compile(r"^[A-Z0-9]{2,6}$")
```

Add dispatch:

```python
if normalized_market == "AU":
    return self.fetch_au_snapshot()
```

Add `fetch_au_snapshot`:

```python
def fetch_au_snapshot(self) -> OfficialMarketUniverseSnapshot:
    rows: list[dict[str, Any]] = []
    fetch_mode = "live_http"
    fetch_errors: dict[str, str] = {}
    fetched_at: str | None = None
    last_modified: str | None = None
    tls_disabled = False

    if settings.au_universe_source_url:
        try:
            fetched = self._http_get(settings.au_universe_source_url)
            rows = self._parse_au_asx_csv(fetched.content)
            fetched_at = fetched.fetched_at
            last_modified = fetched.last_modified
            tls_disabled = fetched.tls_verification_disabled
        except Exception as exc:
            fetch_errors["live_http"] = str(exc)
            rows = self._load_au_csv_fallback()
            fetch_mode = "csv_fallback"
    else:
        rows = self._load_au_csv_fallback()
        fetch_mode = "csv_fallback"

    min_size = int(settings.au_live_min_universe_size or 0)
    if fetch_mode == "live_http" and min_size and len(rows) < min_size:
        fetch_errors["live_http"] = f"live ASX CSV returned {len(rows)} rows, below {min_size} threshold"
        rows = self._load_au_csv_fallback()
        fetch_mode = "csv_fallback"

    if not rows:
        raise ValueError("AU official universe fetch returned no rows (live + fallback both empty)")
    if fetch_mode == "csv_fallback" and min_size and len(rows) < min_size:
        raise ValueError(
            f"AU official universe fallback returned {len(rows)} rows, below {min_size} threshold"
        )

    snapshot_as_of = self._utc_today().isoformat()
    source_metadata: dict[str, Any] = {
        "source_urls": [settings.au_universe_source_url] if settings.au_universe_source_url else [],
        "fetch_mode": fetch_mode,
        "fetched_at": fetched_at,
        "http_last_modified": last_modified,
        "tls_verification_disabled": tls_disabled,
        "row_count": len(rows),
        "filters": {
            "source": "ASX listed companies public CSV",
            "symbol_regex": _AU_LIVE_TICKER_RE.pattern,
        },
    }
    if fetch_errors:
        source_metadata["fetch_errors"] = fetch_errors
    if fetch_mode == "csv_fallback":
        source_metadata["fallback_csv_path"] = settings.au_universe_fallback_csv_path

    prefix = "asx-listed-companies" if fetch_mode == "live_http" else "au-csv-fallback"
    return OfficialMarketUniverseSnapshot(
        market="AU",
        source_name=_AU_SOURCE_NAME if fetch_mode == "live_http" else _AU_FALLBACK_SOURCE_NAME,
        snapshot_id=f"{prefix}-{snapshot_as_of}",
        snapshot_as_of=snapshot_as_of,
        source_metadata=source_metadata,
        rows=tuple(sorted(rows, key=lambda row: row["symbol"])),
    )
```

Add the parser:

```python
@classmethod
def _parse_au_asx_csv(cls, content: bytes) -> list[dict[str, Any]]:
    text = content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().lower().startswith("company name,asx code,gics industry group")
        ),
        None,
    )
    if header_index is None:
        raise ValueError("ASX CSV header not found")

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        name = str(raw.get("Company name") or "").strip()
        local_code = str(raw.get("ASX code") or "").strip().upper()
        gics = str(raw.get("GICS industry group") or "").strip()
        if not name or not local_code:
            continue
        if not _AU_LIVE_TICKER_RE.fullmatch(local_code):
            continue
        rows.append(
            {
                "symbol": f"{local_code}.AX",
                "local_code": local_code,
                "name": name,
                "exchange": "XASX",
                "sector": "",
                "industry": gics,
                "gics_industry_group": gics,
                "market_cap": None,
            }
        )
    return rows
```

Add fallback loader:

```python
@classmethod
def _load_au_csv_fallback(cls) -> list[dict[str, Any]]:
    csv_path = Path(settings.au_universe_fallback_csv_path)
    if not csv_path.exists():
        return []
    return cls._parse_au_asx_csv(csv_path.read_bytes())
```

- [ ] **Step 5: Add broad fallback CSV**

Create `data/au_asx_listed_companies.csv` by downloading the ASX CSV:

```bash
curl -fsSL "https://www.asx.com.au/asx/research/ASXListedCompanies.csv" \
  -o data/au_asx_listed_companies.csv
```

Validate the fallback size:

```bash
python - <<'PY'
from pathlib import Path
import csv, io
path = Path("data/au_asx_listed_companies.csv")
lines = path.read_text(encoding="utf-8-sig").splitlines()
header = next(i for i, line in enumerate(lines) if line.startswith("Company name,ASX code,"))
rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
print(len(rows))
assert len(rows) >= 1500
PY
```

Expected: prints at least `1500`.

- [ ] **Step 6: Run tests and confirm they pass**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/config/settings.py \
  backend/app/services/official_market_universe_source_service.py \
  backend/tests/unit/fixtures/universe_sources/asx_listed_companies_fixture.csv \
  backend/tests/unit/test_official_market_universe_source_service.py \
  data/au_asx_listed_companies.csv
git commit -m "feat: fetch australia universe from asx csv"
```

### Task 5: Wire AU Universe Ingestion and Bootstrap Ordering

**Files:**
- Modify: `backend/app/services/stock_universe_service.py`
- Modify: `backend/app/tasks/universe_tasks.py`
- Modify: `backend/tests/unit/test_stock_universe_service.py`
- Modify: `backend/tests/unit/test_universe_tasks.py`
- Modify: `backend/tests/unit/domain/test_bootstrap_plan.py`

- [ ] **Step 1: Write failing integration tests**

Add AU to the official universe dispatch test:

```python
def test_official_universe_markets_include_au():
    from app.tasks import universe_tasks as module

    assert module._OFFICIAL_UNIVERSE_INGEST_METHODS["AU"] == "ingest_au_snapshot_rows"
    assert "AU" in module._OFFICIAL_SOURCE_MARKETS
```

Add service wiring assertion:

```python
def test_stock_universe_service_wires_au_pipeline():
    service = StockUniverseService()
    assert "AU" in service._universe_ingestion_pipeline._canonicalizers
```

Add bootstrap ordering assertion:

```python
def test_au_bootstrap_fetches_universe_before_prices_and_fundamentals():
    plan = build_bootstrap_plan(primary_market="AU", enabled_markets=["AU"])
    au_plan = plan.market_plans[0]

    assert [stage.key for stage in au_plan.stages[:3]] == [
        "universe",
        "prices",
        "fundamentals",
    ]
    assert au_plan.stages[0].task_name == "refresh_official_market_universe"
    assert au_plan.stages[0].kwargs["market"] == "AU"
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_stock_universe_service.py \
  tests/unit/test_universe_tasks.py \
  tests/unit/domain/test_bootstrap_plan.py -q
```

Expected: FAIL because AU dispatch and service wiring are missing.

- [ ] **Step 3: Wire AU adapter into service and task dispatch**

In `stock_universe_service.py`, import:

```python
from .au_universe_ingestion_adapter import au_universe_ingestion_adapter
```

Initialize:

```python
self._au_ingestion = au_universe_ingestion_adapter
```

Add canonicalizer:

```python
"AU": self._au_ingestion,
```

Add method:

```python
def ingest_au_snapshot_rows(
    self,
    db: Session,
    *,
    rows: Iterable[dict[str, Any]],
    source_name: str,
    snapshot_id: str,
    snapshot_as_of: str | None = None,
    source_metadata: Optional[dict[str, Any]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Ingest AU rows with deterministic canonicalization and lineage metadata."""
    return self._ingest_snapshot_rows_via_pipeline(
        db,
        market="AU",
        rows=rows,
        source_name=source_name,
        snapshot_id=snapshot_id,
        snapshot_as_of=snapshot_as_of,
        source_metadata=source_metadata,
        strict=strict,
    )
```

In `universe_tasks.py`, add:

```python
"AU": "ingest_au_snapshot_rows",
```

The existing bootstrap planner already runs `refresh_official_market_universe` before `smart_refresh_cache` and `refresh_all_fundamentals` for non-US markets. The AU bootstrap test from Step 1 makes that live-before-price ordering explicit.

- [ ] **Step 4: Run tests and confirm they pass**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/stock_universe_service.py \
  backend/app/tasks/universe_tasks.py \
  backend/tests/unit/test_stock_universe_service.py \
  backend/tests/unit/test_universe_tasks.py \
  backend/tests/unit/domain/test_bootstrap_plan.py
git commit -m "feat: wire australia official universe ingestion"
```

### Task 6: End-to-End Verification and Beads Handoff

**Files:**
- Modify/create beads issue records as needed under `.beads/`

- [ ] **Step 1: Run targeted quality gates**

Run:

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/domain/test_market_catalog.py \
  tests/unit/domain/markets/test_market_registry.py \
  tests/unit/domain/markets/test_mic_aliases.py \
  tests/unit/domain/markets/test_symbol_suffixes.py \
  tests/unit/domain/universe/test_indexes.py \
  tests/unit/domain/test_provider_data_plan.py \
  tests/unit/test_rate_budget_policy.py \
  tests/unit/test_security_master_service.py \
  tests/unit/test_fx_service.py \
  tests/unit/test_au_universe_ingestion_adapter.py \
  tests/unit/test_official_market_universe_source_service.py \
  tests/unit/test_stock_universe_service.py \
  tests/unit/test_universe_tasks.py \
  tests/unit/domain/test_bootstrap_plan.py -q
```

Expected: PASS.

- [ ] **Step 2: Run import/compile check**

Run:

```bash
cd backend && source venv/bin/activate && python -m py_compile \
  app/services/au_universe_ingestion_adapter.py \
  app/services/official_market_universe_source_service.py \
  app/services/stock_universe_service.py \
  app/tasks/universe_tasks.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Optional live ASX smoke**

Run only when network access is acceptable for the session:

```bash
cd backend && source venv/bin/activate && python - <<'PY'
from app.services.official_market_universe_source_service import OfficialMarketUniverseSourceService

snapshot = OfficialMarketUniverseSourceService().fetch_market_snapshot("AU")
print(snapshot.market, snapshot.source_name, snapshot.source_metadata["fetch_mode"], len(snapshot.rows))
assert snapshot.market == "AU"
assert snapshot.source_name == "asx_official_public_csv"
assert snapshot.source_metadata["fetch_mode"] == "live_http"
assert len(snapshot.rows) >= 1500
PY
```

Expected: prints `AU asx_official_public_csv live_http <row_count>` with `<row_count> >= 1500`.

- [ ] **Step 4: Update beads**

Create or update a bead for this work:

```bash
bd create --title="Add Australia market universe from ASX public CSV" --type=feature --priority=2
bd update <created-id> --status in_progress
```

After all code and tests pass:

```bash
bd close <created-id> --reason="Implemented AU market facts, live ASX universe fetch, fallback CSV, provider plan, and bootstrap ordering"
```

If this `bd` installation lacks `bd sync`, export bead state explicitly:

```bash
bd export -o .beads/issues.jsonl
```

- [ ] **Step 5: Final commit and push**

Run:

```bash
git status --short
git pull --rebase
git push -u origin feat/au-market-universe
git status --short --branch
```

Expected: branch is pushed and status shows it is up to date with origin.

---

## Self-Review

Spec coverage:

- AU source method is ASX public CSV.
- Live fetch is the default path.
- Fallback CSV is committed under `data/`.
- AU symbols canonicalize to `<ASX code>.AX`.
- AU market facts use `AU`, `XASX`, `AUD`, `Australia/Sydney`, and yfinance.
- Runtime bootstrap ordering verifies universe before price and fundamentals.

Placeholder scan:

- No forbidden placeholder markers remain.
- Every task has concrete files, test commands, code snippets, and expected outcomes.

Type consistency:

- The adapter uses the shared `CanonicalUniverseRow` result shape like SG.
- The official source emits plain row dictionaries accepted by `ingest_au_snapshot_rows`.
- Task dispatch names match the `StockUniverseService` method names.
