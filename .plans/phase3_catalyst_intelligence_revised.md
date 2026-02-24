# Phase 3 (Revised): Unstructured Data → Catalyst Intelligence + Narrative/Setup Fusion

> **Scope:** This is an architecture + feature plan to evolve the repo’s existing **Theme Discovery** into a robust **Catalyst Intelligence** layer that is (1) reliable, (2) performant, and (3) directly useful to a swing/position workflow—especially for **pre-breakout discovery**.
>
> This is **analysis tooling**, not financial advice.

---

## 0) What exists today (important so we build on it, not around it)

From the current repository architecture:

### Theme Discovery (already strong)
- `ContentSource` with source type, priority, fetch interval, and pipeline assignment (`pipelines` JSON).
- `ContentItem` storing raw text content + dedup via `source_type + external_id`, plus processing flags.
- `ThemeMention` extracted by an LLM (theme text, tickers[], sentiment, confidence, excerpt) + pipeline label.
- `ThemeCluster`, `ThemeConstituent`, `ThemeMetrics`, `ThemeAlert`.
- Correlation validation/discovery service (`ThemeCorrelationService`) that clusters by price action and validates cohesiveness.
- Async orchestration (`ThemePipelineRun`, Celery tasks in `theme_discovery_tasks.py`) and API endpoints (`/api/v1/themes/*`).

### Document cache (partial RAG foundation)
- `DocumentCache` + `DocumentChunk` with embeddings for SEC/IR PDFs; includes `document_hash` for change detection.

### Chatbot (multi-provider + tools)
- Tool-based “research mode” that can query DB + yfinance + web search.
- It does not yet have a unified “internal citations / RAG index” across all ingested content.

---

## 1) Why revise Phase 3

Your original Phase 3 direction is correct (timeline + narrative momentum + internal RAG), but it can be **much more compelling** if:

1) **Catalysts are modeled as first-class events** (typed, deduped, confidence-scored) rather than just “mentions”.
2) **Entity resolution** becomes explicit (tickers, company names, CIKs, aliases) to reduce noisy extraction.
3) **Performance is engineered** via incremental pipelines + indexing, not “query everything at runtime”.
4) The system outputs **workflow-ready artifacts**:
   - a per-ticker **Catalyst Timeline**,
   - narrative momentum **time series**,
   - and RAG that answers *from your own data with citations*.

---

## 2) Revised North-Star Deliverable

### A. “Ticker Intelligence Panel” (one place a trader lives)
For any ticker, show:
- **Setup context** (from Setup Engine): pattern, pivot, distance, readiness.
- **Catalyst timeline** (typed events + confidence + sources).
- **Narrative momentum** (velocity, source diversity, novelty, sentiment dispersion).
- **Price reaction overlays** to catalysts (1d/3d/5d return & volume response).
- “What changed?” diffs for filings / transcripts.
- A “Chat with citations” mode grounded in internal docs, not the public web.

### B. “Pre-breakout Funnel” filter
Enable filtering like:
> `setup_ready AND bb_squeeze AND narrative_accelerating AND catalyst_recent`

This is how Phase 3 directly becomes “find stocks before they breakout” rather than a separate theme page.

---

## 3) Architectural Revision: From “Theme pipeline” to “Unstructured Intelligence Platform”

### 3.1 Three-layer model: Raw → Extracted → Derived

**Layer 1: Raw**
- Content is ingested and stored with complete provenance.
- We preserve “what was seen” and when.

**Layer 2: Extracted**
- LLM + deterministic extractors produce structured objects:
  - themes, tickers, sentiment,
  - **events/catalysts** (new),
  - entity mentions (people/orgs/products) (optional but high value).

**Layer 3: Derived**
- Time-series features: narrative momentum, novelty, confidence, event impact.
- Aggregates: per-theme and per-ticker dashboards.
- Indices: vector + keyword for retrieval.

This separation gives reliability and makes reprocessing safe when prompts change.

### 3.2 Event-driven pipeline shape (Celery-friendly)

Instead of a monolithic “extract everything”, use small idempotent tasks:

```
[ingest_content] → [normalize_dedup] → [extract_themes] → [extract_catalysts]
                                          ↓
                                   [chunk_and_embed]
                                          ↓
                             [update_metrics + alerts]
```

Each task:
- accepts a batch of IDs,
- is idempotent,
- stores progress and errors,
- can be re-run safely.

You already have `ThemePipelineRun`; extend it or create a sister model for catalyst runs.

---

## 4) Data Model Upgrades

This is the biggest improvement for reliability and compelling UX.

### 4.1 Add canonical entity resolution (reduces LLM noise dramatically)

Add:

#### `TickerEntity` (new)
Stores canonical mapping for:
- symbol, company name, common aliases, and optionally CIK.
- historical symbol changes (optional).

Fields:
- `symbol` (PK)
- `company_name`
- `aliases_json` (list of strings)
- `cik`
- `exchange`
- `is_active`
- `updated_at`

Why:
- LLM extraction becomes “map detected company mention → TickerEntity” instead of guessing.
- Dedup / merges become possible (“Google” → GOOG/GOOGL logic).

Populate from your existing `StockUniverse` plus SEC CIK mapping when available.

#### `EntityMention` (optional but recommended)
A generic table for entity extraction from content (orgs, products, regulators, people).
This unlocks:
- “contract with X”, “FDA decision”, “DoD award”, “Apple supplier”, etc.

### 4.2 Turn catalysts into first-class typed events

Add:

#### `CatalystEvent` (new)
A normalized “real-world thing happened” event.

Key columns:
- `id` (PK)
- `event_type` (enum-ish string)
  - `earnings`, `guidance`, `sec_filing`, `insider_trade`, `contract_award`,
    `product_launch`, `fda_decision`, `m_and_a`, `analyst_action`,
    `partnership`, `macro_policy`, `litigation`, `rumor`, etc.
- `symbol` (indexed)
- `event_time` (datetime, indexed)
- `title` (short)
- `summary` (short, LLM or rules)
- `confidence` (0–1)
- `verification_level` (string): `primary`, `multi_source`, `single_source`, `unverified`
- `novelty_score` (0–1)
- `impact_score` (0–1) *(see §6)*

#### `CatalystEvidence` (new)
Links events to their sources (so we can cite them + dedup them).

- `event_id` (FK)
- `source_kind`: `content_item`, `sec_doc`, `transcript`, `press_release`, etc.
- `source_id`: points to `ContentItem.id` or `DocumentCache.id`
- `url`
- `excerpt`
- `author/source_name`
- `published_at`
- `stance/sentiment` (optional)
- `evidence_confidence` (0–1)

This evidence model makes timelines credible and debuggable.

### 4.3 Unify chunking + embeddings across all document types (major RAG upgrade)

Right now you have `DocumentChunk` for SEC/IR docs only.
To make Phase 3 compelling, you need RAG across:

- ingested articles/posts/tweets (`ContentItem`)
- filings and PDFs (`DocumentCache`)
- transcripts (if you add them)

**Best architecture (recommended): create a unified chunk table**:

#### `UnstructuredDocument` (new)
A generic document registry that points to a raw source:
- `doc_type`: `content_item`, `sec_filing`, `ir_pdf`, `transcript`, etc.
- `source_pk`: integer reference to the raw row (`ContentItem.id` or `DocumentCache.id`)
- `symbol` (nullable; may be multi-symbol docs later)
- `canonical_url` + `document_hash`
- `published_at`
- `title`
- `text` (optional; or store pointer to raw)

#### `UnstructuredChunk` (new)
- `document_id` (FK to UnstructuredDocument)
- `chunk_index`
- `chunk_text`
- `chunk_tokens`
- `embedding`
- `embedding_model`
- `metadata_json` (section name, offsets, etc.)

**Why unify:** Your chatbot and UI can query a single retrieval surface.

**Migration path:**
- Keep existing `DocumentCache`/`DocumentChunk` intact for now.
- Add a background job that registers them into `UnstructuredDocument` and copies chunks.
- Later, deprecate direct use of `DocumentChunk` in favor of unified chunks.

---

## 5) Extraction Pipeline (Catalyst + Narrative) — Make it Reliable

### 5.1 Two-stage extraction: cheap gate → LLM enrichment

To reduce cost and improve throughput:

**Stage A: Heuristic gating**
- Basic NLP/regex to detect:
  - tickers (`$AAPL`, `AAPL`), known company aliases, CIK patterns,
  - event keywords (e.g., “8-K”, “guidance”, “awarded”, “FDA”, “acquired”),
  - and confidence that text is “event-like”.

Only send event-like items to the LLM.

**Stage B: LLM structured extraction**
Prompt outputs strict JSON for:
- `event_type`
- `symbols[]` + confidence per symbol
- `event_time` (if present)
- `summary` (short)
- `key_entities[]` (optional)
- `stance/sentiment`
- `quoted_evidence` (short excerpt)

Store:
- `prompt_version`, `model_id`, `provider`, `latency_ms`, and `extraction_version`.
This is critical for reproducibility.

### 5.2 Dedup + event fusion (prevents timeline spam)

Create an “event fingerprint” for dedup:
- hash of: `(symbol, event_type, date_bucket, normalized_title, key_entity_hash)`
- allow fuzzy merges using:
  - embedding similarity on summaries,
  - overlapping source URLs,
  - same SEC accession number.

Rules:
- Prefer **primary sources** (SEC filing / IR press release) over commentary.
- Merge low-confidence duplicates into the event as additional evidence.

### 5.3 Confidence model (don’t let LLM guesses dominate)

Define confidence as:
- base confidence from extractor
- boosted by:
  - primary source evidence
  - multi-source confirmation
  - ticker match via `TickerEntity`
- penalized by:
  - ambiguity (multiple possible tickers)
  - low-quality sources
  - rumor language (“might”, “could”, “reportedly”)

Expose `verification_level` so the UI can label events clearly.

---

## 6) Narrative Momentum (Revised) — from “counts” to “momentum + quality”

You already compute theme-level mention velocity.
Phase 3 should add **ticker-level** narrative metrics and make them **source-aware** and **novelty-aware**.

### 6.1 New per-ticker daily table: `TickerNarrativeMetrics` (new)

Key fields (daily snapshots):
- `symbol`, `date`
- **velocity**
  - `mentions_1d`, `mentions_7d`, `mentions_30d`
  - `velocity_7d_over_30d`
  - `acceleration` = change in velocity vs prior week
- **source diversity**
  - `unique_sources_7d`
  - `unique_authors_7d`
  - `source_entropy` (higher = broader dissemination)
  - `new_source_ratio` (fraction of sources not seen in prior 30d)
- **sentiment quality**
  - `sentiment_mean`
  - `sentiment_dispersion` (controversy proxy)
  - `bull_bear_ratio`
- **novelty**
  - `novelty_mean` (embedding distance vs prior 30d content)
  - `repeat_rate` (how much is rehash)
- **catalyst proximity**
  - `recent_catalyst_count_7d`
  - `last_primary_catalyst_days_ago`
- **integrations**
  - `setup_ready_count` (how many scans show “setup_ready”)
  - `avg_setup_score` (optional)

### 6.2 Weighted sources (use what you already have)
Leverage `ContentSource.priority`:
- Weight mentions by priority so “high signal” sources matter more.
- Allow a per-source “trust score” that updates based on extraction quality / user feedback.

### 6.3 “Narrative momentum meets price” (the compelling part)
Add derived features that answer:
- **Is narrative leading price?**
  - correlation of narrative velocity to future returns
  - lag analysis (7d narrative change vs next 5d/10d return)
- **Did catalysts historically produce follow-through?**
  - event_type → avg 5d/20d reaction for the ticker or theme
  - later you can tie this into the Learning Loop.

---

## 7) Catalyst Timeline UX (Revised)

### 7.1 Timeline is event-first, not mention-first
Show `CatalystEvent` rows with:
- event icon (type)
- “verified” badge based on `verification_level`
- confidence score
- top evidence sources (clickable)
- short summary (2–3 bullet points)
- price reaction overlay (see below)

### 7.2 Price reaction overlay (lightweight, high value)
For each event, compute:
- `ret_1d`, `ret_3d`, `ret_5d` vs SPY
- `volume_spike_1d` vs 50d average
Store in `CatalystEventImpact` (new) or inline into `CatalystEvent` as JSON.

This makes the tool feel “trader-grade” immediately.

### 7.3 “What changed” diff view for filings/transcripts
For SEC filings:
- store section-level chunks
- compute “changed sections” by comparing chunk hashes/embeddings vs prior filing
- summarize changes per section (Risk Factors, MD&A, Guidance, etc.)

This is both:
- useful for traders,
- and reduces hallucinations because you ground summaries in diffs.

---

## 8) RAG Store (Revised) — credible internal chat with citations

### 8.1 Retrieval should be hybrid
Pure vector search is not enough for finance text (tickers, numbers, acronyms).
Implement **hybrid retrieval**:

1) Keyword search (SQLite FTS5 if staying on SQLite)
2) Vector search
3) Merge + rerank (simple linear weighting to start)

### 8.2 Citations must be first-class
For each chunk returned, store:
- document URL
- published_at
- source name
- chunk text excerpt

Chat responses should cite:
- which sources/chunks they used
- and link back to the timeline/event where applicable.

### 8.3 Cache “ticker digests”
To keep chat snappy:
- precompute a daily “ticker digest” summary from last N docs/events
- store it in DB
- let the chatbot retrieve the digest as a first step, then drill into documents only if needed.

---

## 9) API + Backend Surface Area (what to build)

### 9.1 New endpoints (v1)

- `GET /api/v1/catalysts/{symbol}/timeline?days=90`
- `GET /api/v1/catalysts/{symbol}/metrics?days=180`
- `GET /api/v1/catalysts/theme/{theme_id}/timeline`
- `POST /api/v1/catalysts/pipeline/run` (async)
- `GET /api/v1/catalysts/pipeline/{run_id}/status`
- `POST /api/v1/rag/search` (hybrid retrieval for UI & chatbot)

### 9.2 Extend chatbot tools
Add to `DatabaseTools`:
- `get_catalyst_timeline(symbol, days=90)`
- `get_narrative_metrics(symbol, days=180)`
- `search_internal_documents(query, symbol=None, days=None)`

This makes “research mode” meaningfully better without web-search.

---

## 10) Performance + Reliability Engineering (don’t skip this)

### 10.1 Idempotency + reprocessing
For any extraction output:
- store `extraction_version` and `prompt_version`
- allow re-run when:
  - prompt version increases
  - entity mapping improves
  - you add a new event type

### 10.2 Incremental indexing
- chunk+embed only new/changed docs (via `content_hash` or `document_hash`)
- compute metrics nightly; compute “hot” metrics hourly if needed

### 10.3 Indexing choices (pragmatic)
Base: SQLite + FTS5 + brute-force vector for small corpora  
Scale-up path (optional):
- Postgres + `pgvector` for vector search
- background index build (FAISS) if you want to stay file-based

Make this a config option, not a rewrite.

### 10.4 Observability
Add:
- counts of ingested items
- extraction success rate
- average latency per provider/model
- number of events created/deduped
- index size and retrieval latency

Expose summary in pipeline run objects and logs.

---

## 11) Implementation Roadmap (tight, incremental, shippable)

### Phase 3.0 — Hardening the existing theme pipeline (1–2 PRs)
- Add `content_hash` to `ContentItem` for change detection + dedup improvements
- Ensure schema consistency between models and chatbot DB tools
- Add prompt/model version fields to `ThemeMention` extraction metadata

### Phase 3.1 — CatalystEvent foundation (core value)
- Add `TickerEntity`, `CatalystEvent`, `CatalystEvidence`
- Build `CatalystExtractionService` (two-stage gating + LLM)
- Add Celery tasks + run tracking
- Expose `/api/v1/catalysts/{symbol}/timeline`

### Phase 3.2 — Narrative momentum metrics (ticker-level)
- Add `TickerNarrativeMetrics`
- Nightly job to compute metrics from content + events
- Expose API + integrate with watchlist UI

### Phase 3.3 — Unified RAG (compelling chatbot upgrade)
- Add `UnstructuredDocument` + `UnstructuredChunk`
- Migrate SEC docs + content items into unified chunk table
- Implement hybrid retrieval endpoint
- Update chatbot to cite internal sources by default

### Phase 3.4 — “Price reaction overlays” + “What changed”
- Add event impact computation (returns + volume)
- Implement filing diff summaries and section-level change flags
- UI overlay on timeline + chart

---

## 12) What makes this version “better” than the original Phase 3

- **Credibility:** events have evidence + verification levels, not just LLM mentions.
- **Reliability:** entity resolution reduces false tickers and makes extraction deterministic.
- **Performance:** incremental pipelines + indexing avoids runtime heavy queries.
- **Compelling UX:** timeline + price reaction + diffs + citations is extremely sticky for traders.
- **Integrates with Setup Engine:** narrative + catalysts become filterable signals for “pre-breakout discovery”.

