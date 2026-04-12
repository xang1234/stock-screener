# Expand Stock Screening to Hong Kong, Japan, and Taiwan



## Context



The platform currently screens **US-only stocks** sourced from Finviz (NYSE, NASDAQ, AMEX). Every layer — universe population, data fetching, fundamental services, theme extraction, and the frontend — assumes US tickers and US data providers. The goal is to extend coverage to **HKEX (Hong Kong), TSE/JPX (Japan), and TWSE (Taiwan)** while preserving the existing US screening pipeline unchanged.



---



## Architecture Overview



```

┌────────────────────────────────────────────────────────────────────────┐

│                        CURRENT (US-ONLY)                               │

│                                                                        │

│  Finviz ──► stock_universe ──► DataPreparationLayer ──► Screeners     │

│                                    │                                   │

│               yfinance ◄───────────┘                                   │

│               Alpha Vantage ◄──────┘                                   │

│               Finviz fundamentals ◄┘                                   │

│                                                                        │

│  Theme Extraction: "US-listed stock tickers only"                      │

│  Benchmark: SPY (hardcoded)                                            │

│  Exchanges: NYSE | NASDAQ | AMEX (enum)                                │

└────────────────────────────────────────────────────────────────────────┘



                              ▼  PROPOSED  ▼



┌────────────────────────────────────────────────────────────────────────┐

│                     MULTI-MARKET ARCHITECTURE                          │

│                                                                        │

│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │

│  │ US       │  │ HK       │  │ JP       │  │ TW       │              │

│  │ Finviz   │  │ CSV/     │  │ CSV/     │  │ CSV/     │              │

│  │          │  │ yfinance │  │ yfinance │  │ yfinance │              │

│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘              │

│       │              │              │              │                    │

│       └──────────────┴──────────────┴──────────────┘                   │

│                              │                                         │

│                     stock_universe (+ market column)                    │

│                              │                                         │

│              ┌───────────────┼───────────────┐                         │

│              ▼               ▼               ▼                         │

│     DataPreparationLayer  Benchmark     Fundamentals                   │

│     (yfinance for all)    per-market:   yfinance (all markets)         │

│                           US→SPY        Finviz (US-only)               │

│                           HK→^HSI       AV (US-only)                   │

│                           JP→^N225                                     │

│                           TW→^TWII                                     │

│              │                                                         │

│              ▼                                                         │

│         Screeners (unchanged logic, market-aware benchmark)            │

│              │                                                         │

│              ▼                                                         │

│         Theme Extraction (multi-market ticker recognition)             │

│         Translation Layer (JP/TW/HK article → English)                 │

└────────────────────────────────────────────────────────────────────────┘

```



---



## Data Sources Analysis



### Price Data (OHLCV)



| Market | yfinance ticker format | Example | Coverage | Notes |

|--------|----------------------|---------|----------|-------|

| **HK** | `{code}.HK` | `0700.HK` (Tencent) | Full daily OHLCV | Zero-padded 4-digit codes |

| **JP** | `{code}.T` | `6758.T` (Sony), `7203.T` (Toyota) | Full daily OHLCV | TSE main market |

| **TW** | `{code}.TW` | `2330.TW` (TSMC), `2317.TW` (Hon Hai) | Full daily OHLCV | TWSE listed |

| **TW OTC** | `{code}.TWO` | `6547.TWO` | Full daily OHLCV | TPEx (OTC) market |



**yfinance works for all three markets** — no new price data provider needed. The existing `PriceCacheService`, `BenchmarkCacheService`, and `DataPreparationLayer` can handle these tickers with suffix-aware changes.



### Fundamental Data



| Source | US | HK | JP | TW | Fields |

|--------|----|----|----|----|--------|

| **yfinance `.info`** | Yes | Partial | Partial | Partial | market_cap, PE, EPS, margins, sector/industry |

| **yfinance `.income_stmt`** | Yes | Yes | Yes | Yes | Revenue, net income, EPS (for growth calc) |

| **yfinance `.quarterly_income_stmt`** | Yes | Yes | Yes | Yes | Quarterly revenue/EPS for Q/Q and Y/Y growth |

| **Finviz** | Yes | **No** | **No** | **No** | Short interest, insider txns, forward estimates |

| **Alpha Vantage** | Yes | Limited | Limited | **No** | Company overview, earnings |



**Strategy**: yfinance is the **universal fundamental provider** across all four markets. Finviz and Alpha Vantage remain US-only supplemental sources. The `HybridFundamentalsService` already has a tiered fallback — Asian market stocks simply skip the Finviz tier and rely on yfinance + technical calculator.



**Known gaps for Asian markets** (yfinance limitations):

- Short interest data: Not available

- Insider/institutional transaction data: Spotty

- Forward EPS estimates: May be missing for smaller caps

- Sector/industry: Available but uses Yahoo's classification (not GICS)



### Universe Population



Unlike US (Finviz scraping), there is no free equivalent for HK/JP/TW. Options:



1. **Static CSV seeding** — Maintain curated CSVs of major listed stocks per market (the `import_from_csv` path already exists)

2. **Wikipedia scraping** — Lists of constituents for Hang Seng, Nikkei 225, TAIEX exist on Wikipedia (similar to existing `fetch_sp500_symbols`)

3. **Exchange website scraping** — HKEX, JPX, and TWSE publish listed company CSVs

4. **yfinance `.info` validation** — Validate tickers by attempting to fetch info



**Recommended approach**: CSV seeding + Wikipedia index scraping for major indices (HSI, Nikkei 225, TWSE 50). Users can also add individual tickers via the existing manual-add API.



---



## Ticker Format & Normalization



### Internal Format Convention



Store the **yfinance-compatible ticker** as the canonical symbol in `stock_universe`:



| Market | Internal symbol | Display | yfinance query |

|--------|----------------|---------|----------------|

| US | `AAPL` | `AAPL` | `AAPL` |

| HK | `0700.HK` | `0700.HK` | `0700.HK` |

| JP | `6758.T` | `6758.T` | `6758.T` |

| TW | `2330.TW` | `2330.TW` | `2330.TW` |



### Ticker Suffix → Market Mapping



```python

MARKET_SUFFIX_MAP = {

    ".HK": "HK",

    ".T": "JP",

    ".TW": "TW",

    ".TWO": "TW",  # Taiwan OTC → same market

}



def detect_market(symbol: str) -> str:

    """Return market code from ticker suffix. Default 'US'."""

    for suffix, market in MARKET_SUFFIX_MAP.items():

        if symbol.upper().endswith(suffix):

            return market

    return "US"

```



### Handling Tickers in News/Tweets



Articles from Asian financial media mention tickers in local formats:

- **HK**: `騰訊(0700)` or `700 HK` or `0700.HK`

- **JP**: `ソニーグループ(6758)` or `6758` (bare code with exchange context)

- **TW**: `台積電(2330)` or `2330`



**Strategy**: The theme extraction LLM prompt will be extended to recognize multi-market tickers and normalize them to yfinance format. The prompt already converts company names → tickers for US; the same pattern extends to Asian markets with market-context hints.



---



## Translation Architecture



### When Translation Is Needed



Content from Asian sources (Substack RSS in Chinese/Japanese, Twitter posts) needs English translation for:

1. Theme extraction LLM prompts (the LLM works best in English)

2. UI display of extracted themes and excerpts

3. Chatbot context



### Translation Approach



**LLM-based translation integrated into the theme extraction pipeline**:



Rather than a separate translation step, modify the extraction prompt to handle multilingual input natively. Modern LLMs (Groq/Llama, Minimax) handle CJK input well. The prompt instructs:



1. Accept content in any language

2. Extract themes with English canonical names

3. Translate excerpts to English

4. Recognize local company names and map to yfinance tickers



This is a **zero-infrastructure** approach — no Google Translate API, no DeepL. The LLM that already does extraction also handles translation in one pass.



**For content display**, store both original language and translated text in `ContentItem`:

- `content` field: original language (for dedup and source fidelity)

- New `content_translated` field: English translation (populated during extraction)

- New `title_translated` field: English title



### Prompt Changes



The `EXTRACTION_SYSTEM_PROMPT` currently says:

> "Only include actual tradeable US stock tickers (NYSE, NASDAQ)"



This must be updated to:

> "Include tradeable stock tickers from: US (NYSE, NASDAQ), Hong Kong (HKEX, suffix .HK), Japan (TSE, suffix .T), Taiwan (TWSE, suffix .TW). Normalize all tickers to yfinance format."



---



## Implementation Plan



### Phase 1: Data Model & Market Infrastructure (Backend Core)



**1.1 Add `market` column to `stock_universe`**



File: `backend/app/models/stock_universe.py`

- Add `market = Column(String(5), nullable=False, default="US", index=True)` 

- Add composite index `idx_universe_market_active` on `(market, is_active)`

- Widen `symbol` column from `String(10)` to `String(15)` to accommodate `0700.HK` format

- Widen the `exchange` column to support `HKEX`, `TSE`, `TWSE`, `TPEX`



File: `backend/app/models/stock.py`

- Widen `symbol` from `String(10)` to `String(15)` in `StockPrice`, `StockFundamental`, `StockTechnical`, `StockIndustry`



File: `backend/alembic/versions/` — new migration:

- `ALTER TABLE stock_universe ADD COLUMN market VARCHAR(5) NOT NULL DEFAULT 'US'`

- Alter symbol column widths

- Add index



**1.2 Extend Exchange enum and Universe schema**



File: `backend/app/schemas/universe.py`

- Add to `Exchange` enum: `HKEX`, `TSE`, `TWSE`, `TPEX`

- Add `Market` enum: `US`, `HK`, `JP`, `TW`

- Add `UniverseType.MARKET` for scanning an entire market

- Add `market` field to `UniverseDefinition`



File: `backend/app/services/universe_resolver.py`

- Handle `UniverseType.MARKET` → filter by `stock_universe.market`



**1.3 Market-aware ticker utilities**



New file: `backend/app/utils/market_utils.py`

```python

MARKET_SUFFIX_MAP = {".HK": "HK", ".T": "JP", ".TW": "TW", ".TWO": "TW"}

MARKET_BENCHMARK = {"US": "SPY", "HK": "^HSI", "JP": "^N225", "TW": "^TWII"}

MARKET_CURRENCY = {"US": "USD", "HK": "HKD", "JP": "JPY", "TW": "TWD"}



def detect_market(symbol: str) -> str: ...

def get_benchmark_symbol(market: str) -> str: ...

def normalize_ticker(raw: str, market_hint: str = None) -> str: ...

```



**1.4 Market-aware benchmark cache**



File: `backend/app/services/benchmark_cache_service.py`

- Generalize from hardcoded `SPY` to support per-market benchmarks

- Redis keys: `benchmark:{symbol}:{period}` (already uses this pattern, just need to support new symbols)

- The `DataPreparationLayer` must select the correct benchmark for each stock's market



File: `backend/app/scanners/data_preparation.py`

- `prepare_data()` takes stock's market → fetches the appropriate benchmark

- `StockData.benchmark_data` remains a DataFrame but contains the correct market benchmark



File: `backend/app/scanners/criteria/relative_strength.py`

- The RS calculator already accepts a `benchmark` parameter — it will receive market-appropriate benchmark data via `StockData`

- RS percentile ranking should be **per-market** (HK stocks ranked against HK universe, etc.)



### Phase 2: Universe Population for Asian Markets



**2.1 CSV-based seeding for HK/JP/TW**



File: `backend/app/services/stock_universe_service.py`

- Extend `populate_from_csv` to accept a `market` parameter that sets the `market` column

- The CSV import path already exists; just needs market-awareness



**2.2 Wikipedia/exchange index scraping** (similar to `fetch_sp500_symbols`)



File: `backend/app/services/stock_universe_service.py`

- `fetch_hsi_symbols()` — Hang Seng Index constituents from Wikipedia

- `fetch_nikkei225_symbols()` — Nikkei 225 from Wikipedia

- `fetch_twse50_symbols()` — FTSE TWSE Taiwan 50 from Wikipedia

- Each returns symbols in yfinance format (e.g., `0700.HK`)



**2.3 Market-filtered universe management**



File: `backend/app/services/stock_universe_service.py`

- `get_active_symbols()` — add `market` filter parameter

- `populate_universe()` — only deactivate within same market during sync

- Safety thresholds per market (HK ~2500, JP ~3800, TW ~1700)



### Phase 3: Fundamentals & Data Fetching Adaptations



**3.1 HybridFundamentalsService market-awareness**



File: `backend/app/services/hybrid_fundamentals_service.py`

- Skip Finviz tier for non-US markets (Finviz only covers US)

- Skip Alpha Vantage for non-US markets

- yfinance + technical calculator remain universal



**3.2 yfinance service — no changes needed**



yfinance already handles `.HK`, `.T`, `.TW` suffixed tickers transparently. The `get_fundamentals()`, `get_historical_data()`, `get_stock_info()` methods all work with these symbols as-is.



**3.3 Finviz service — guard against non-US tickers**



File: `backend/app/services/finviz_service.py`

- Add early return for non-US tickers (Finviz will error on `0700.HK`)



### Phase 4: Theme Extraction & Translation



**4.1 Multi-market extraction prompt**



File: `backend/app/services/theme_extraction_service.py`

- Update `EXTRACTION_SYSTEM_PROMPT`: accept tickers from HK, JP, TW markets

- Update `EXTRACTION_USER_PROMPT`: add `market_context` hint field

- Instruct LLM to normalize tickers to yfinance format

- Instruct LLM to translate non-English excerpts to English



**4.2 Content model translation fields**



File: `backend/app/models/theme.py` (ContentItem)

- Add `title_translated = Column(String)` 

- Add `content_translated = Column(Text)`

- Add `source_language = Column(String(5))` (en, zh, ja, zh-TW)



Migration to add these columns.



**4.3 Translation in extraction pipeline**



File: `backend/app/services/theme_extraction_service.py`

- During extraction, if source language is detected as non-English, request translated title + excerpt in the extraction response

- Store translated content alongside original



### Phase 5: Frontend Changes



**5.1 Universe selector**



File: `frontend/src/features/scan/components/ScanControlBar.jsx`

- Add market-level grouping in Universe dropdown:

  - US: All US / NYSE / NASDAQ / AMEX / S&P 500

  - HK: All HK / Hang Seng

  - JP: All JP / Nikkei 225

  - TW: All TW / TWSE 50

- Update `stockCountLabel()` for new markets



File: `frontend/src/api/scans.js`

- Support new universe types in `createScan()`



**5.2 Display ticker market badges**



- Show market origin badge (US/HK/JP/TW) next to ticker in results table

- Link to appropriate exchange page (not finviz for non-US)



**5.3 Theme display**



- Show translated content when available

- Language indicator on content items



### Phase 6: Celery Tasks & Scheduling



File: `backend/app/tasks/`

- Market-aware cache warmup (stagger by market to spread API load)

- Separate scheduled tasks per market for universe sync

- Benchmark cache warmup for all four benchmarks



---



## Scalability Considerations



1. **Rate limiting**: yfinance is the bottleneck. Each market adds ~2000-4000 tickers. The existing `data_fetch` queue serialization prevents rate limit violations. Consider market-specific rate limit budgets.



2. **RS percentile ranking**: Must be computed **within market**, not globally. A HK stock with RS 90 means it's in the top 10% of HK stocks, not mixed with US stocks.



3. **Benchmark cache**: Four benchmarks instead of one. Memory impact is negligible (4 DataFrames ~2y each).



4. **Database growth**: ~10,000 additional universe records. Price history at ~250 rows/year/stock = ~2.5M additional price rows/year. PostgreSQL handles this fine with existing indices.



5. **Scan parallelism**: Each market can be scanned independently. Consider market-level scan isolation to prevent a slow JP scan from blocking a US scan.



---



## Key Files to Modify



| File | Change |

|------|--------|

| `backend/app/models/stock_universe.py` | Add `market` column, widen `symbol` |

| `backend/app/models/stock.py` | Widen `symbol` columns |

| `backend/app/schemas/universe.py` | Add Market enum, extend Exchange enum |

| `backend/app/services/stock_universe_service.py` | Market-aware population, Asian index scrapers |

| `backend/app/services/universe_resolver.py` | Handle MARKET universe type |

| `backend/app/services/hybrid_fundamentals_service.py` | Skip Finviz/AV for non-US |

| `backend/app/services/finviz_service.py` | Guard non-US tickers |

| `backend/app/services/benchmark_cache_service.py` | Multi-benchmark support |

| `backend/app/scanners/data_preparation.py` | Market-aware benchmark selection |

| `backend/app/services/theme_extraction_service.py` | Multi-market prompt, translation |

| `backend/app/models/theme.py` | Translation fields on ContentItem |

| `backend/app/utils/market_utils.py` | **New** — ticker/market utilities |

| `frontend/src/features/scan/components/ScanControlBar.jsx` | Market universe selector |

| `backend/alembic/versions/` | **New** migration for schema changes |



---



## Verification Plan



1. **Unit tests**: 

   - `test_market_utils.py` — ticker normalization, market detection, benchmark mapping

   - `test_stock_universe_service.py` — CSV import with market param, Asian index scraping

   - `test_universe_resolver.py` — MARKET universe type resolution



2. **Integration tests**:

   - Fetch `0700.HK`, `6758.T`, `2330.TW` via yfinance — verify price data returns

   - Fetch fundamentals for same tickers — verify market_cap, PE, EPS fields populate

   - Run Minervini scanner on a HK stock with `^HSI` benchmark — verify RS calculation

   - Theme extraction with Chinese-language content — verify ticker normalization + translation



3. **Manual verification**:

   - Import a small CSV of HK stocks → verify they appear in universe

   - Run a scan with `market: HK` → verify results include HK tickers with correct benchmark

   - Check that US scans are completely unaffected (no regression)



4. **Frontend verification**:

   - Universe dropdown shows market groupings

   - Scan results display `.HK`, `.T`, `.TW` tickers correctly

   - Theme page shows translated content for Asian-source articles

