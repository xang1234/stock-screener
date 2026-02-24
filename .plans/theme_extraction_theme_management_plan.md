# Stock Screener Theme Extraction: Practical Plan to Reduce Duplicate Themes While Tracking Old + Detecting New

This plan is tailored to the current `xang1234/stock-screener` backend implementation and focuses on **stopping theme explosion** (too many near-duplicate clusters) while still enabling:
- **new theme detection** (novelty),
- **old theme tracking** (continuity / reactivation),
- and **safe merges** with auditability.

---

## 0) Current behavior recap (why you’re getting ~1.8k themes)

### Where themes are created today
**Theme clusters are created from the LLM string output**:

- Theme mentions are extracted by the LLM and stored as `ThemeMention` rows (with `raw_theme`, `canonical_theme`, `pipeline`, etc.).
- For each mention, the service calls `_get_or_create_cluster()` which:
  1) normalizes the theme via `_normalize_theme()`
  2) queries `ThemeCluster` by exact `name == canonical_theme` **and** `pipeline == self.pipeline`
  3) if not found, **creates a new `ThemeCluster`** and adds the raw theme string to `aliases`.

References:
- `backend/app/services/theme_extraction_service.py` (`_normalize_theme`, `_get_or_create_cluster`)
  - `_normalize_theme()` currently does `theme.strip().title()` plus a small hardcoded mapping.
  - `_get_or_create_cluster()` is exact-match-or-create.  
  - Raw file: https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/services/theme_extraction_service.py

### Why it explodes
1) **LLM string variance**: even if two themes are conceptually identical, the LLM may phrase them differently.
2) **Normalization is too limited**: only a small mapping + `.title()` means most synonyms remain distinct strings.
3) **Exact-match clustering**: if canonical strings differ, a new cluster is created.
4) **Both pipelines run** by default (`technical`, `fundamental`), increasing parallel theme namespaces.
   - `extract_themes()` runs both when `pipeline=None`.
   - Raw file: https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/tasks/theme_discovery_tasks.py

---

## 1) Goals + constraints

### Goals
- **Reduce duplicates**: cluster synonymous strings into stable theme IDs.
- **Detect new themes**: create clusters when a concept truly appears for the first time.
- **Track old themes**: keep stable IDs; mark dormant vs active; reactivate when seen again.
- **Keep it maintainable**: align with your existing style:
  - SQLAlchemy + SQLite
  - idempotent startup migrations (no Alembic)
  - Celery tasks already exist

### Non-goals
- Perfect ontology / taxonomy from day 1 (you’ll iterate as data grows).

---

## 2) Fix the 2 gotchas first

### Gotcha #1 — `ThemeCluster.name` is globally unique but code treats it as pipeline-scoped
In the model, `ThemeCluster.name` is `unique=True`, but `_get_or_create_cluster()` matches by `(name, pipeline)`.  
This is inconsistent and will cause constraint conflicts or forced cross-pipeline coupling.

Reference:
- `ThemeCluster.name = Column(... unique=True ...)` in `backend/app/models/theme.py`  
  Raw file: https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/models/theme.py

#### Fix: make uniqueness composite on `(pipeline, name)`
**Recommended schema behavior**:
- allow `("technical", "AI Infrastructure")` and `("fundamental", "AI Infrastructure")` to coexist

**Implementation (SQLite-safe, idempotent)**:
- Add a new startup migration in `backend/app/db_migrations/theme_clusters_migration.py`
- In `backend/app/main.py` `lifespan()`, call `run_theme_clusters_migration()` similarly to the existing universe migration pattern.

Reference for startup migrations style:
- `run_universe_migration()` in `backend/app/main.py`  
  Raw file: https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/main.py
- example migration implementation style: `backend/app/db_migrations/universe_migration.py`  
  Raw file: https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/db_migrations/universe_migration.py

**SQLite rebuild approach** (recommended):
1) Create a new table `theme_clusters_new` with:
   - `name` NOT unique
   - `UNIQUE(pipeline, name)` constraint
2) Copy existing rows
3) Drop old table
4) Rename new table

**Acceptance criteria**
- schema allows same `name` in different pipelines
- extraction does not raise unique constraint violations

---

### Gotcha #2 — `.title()` normalization creates false differences (Ai vs AI)
`_normalize_theme()` currently does:

- `theme = theme.strip().title()`

This turns `"AI"` into `"Ai"` unless it is later corrected by a mapping entry.  
That increases fragmentation and makes exact-match clustering worse.

Reference:
- `_normalize_theme()` in `backend/app/services/theme_extraction_service.py`  
  Raw file: https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/services/theme_extraction_service.py

#### Fix: use acronym-safe normalization (low risk)
Replace `.title()` with a “safe cleanup” approach:
- trim whitespace
- normalize punctuation/hyphens
- preserve:
  - all-caps tokens (`AI`, `EV`, `GPU`)
  - tokens with digits/hyphens (`GLP-1`, `H100`)
- keep the synonym mapping as *supplemental*, not the primary dedupe method

**Acceptance criteria**
- `"AI infrastructure"` canonicalizes to `"AI Infrastructure"` (not `"Ai Infrastructure"`)
- acronyms + digit terms don’t change case unpredictably

---

## 3) Add “match-before-create” (the main fix)

Right now `_get_or_create_cluster()` is: exact match → else create.

Change it to a multi-stage matcher:

### 3.1 Matching stages (in order)
**Stage A — exact match** (current behavior)
- fastest, keep it

**Stage B — alias exact match**
- if the raw theme string matches an alias of any cluster in the same pipeline, attach to that cluster

**Stage C — fuzzy string match**
- use RapidFuzz ratio against:
  - cluster `name`
  - cluster `aliases`
- thresholds:
  - >= 92: auto-attach
  - 85–92: attach, but also create a merge suggestion record (optional)

**Stage D — embedding similarity match**
- compute embedding for candidate theme string
- search nearest clusters in `theme_embeddings`
- if cosine >= threshold (e.g., 0.88–0.92), attach and add candidate as alias

### 3.2 Where to implement
- In `ThemeExtractionService._get_or_create_cluster()`:
  - call `_match_cluster(...)` first
  - only create a new `ThemeCluster` if `_match_cluster` returns None

### 3.3 Why embeddings are easy here
You already have:
- a `ThemeEmbedding` model/table
- `sentence-transformers` in requirements

References:
- `ThemeEmbedding` model in `backend/app/models/theme.py`
  - https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/models/theme.py
- `sentence-transformers` dependency in `backend/requirements.txt`
  - https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/requirements.txt

---

## 4) Build the embedding pipeline (so matching works)

### 4.1 Add `ThemeEmbeddingService`
Create `backend/app/services/theme_embedding_service.py` with:
- `compute_embedding_text(cluster)`:
  - `cluster.name + " | " + top aliases (dedup, max N)`
- `upsert_theme_embedding(theme_cluster_id, embedding, embedding_text, model_name)`
- `get_or_create_theme_embedding(theme_cluster_id)`
- `nearest_clusters_by_embedding(candidate_embedding, pipeline, k=10)`

### 4.2 Backfill embeddings (one-time task)
Add a Celery task:
- `backfill_theme_embeddings(pipeline=None, limit=...)`

Hook it into your operational cadence:
- run once after deployment
- then nightly

Reference for Celery task style:
- `backend/app/tasks/theme_discovery_tasks.py`
  - https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/tasks/theme_discovery_tasks.py

### 4.3 Keep embeddings fresh
When aliases change (you add a new alias during matching):
- mark embedding stale and recompute
  - either inline (simple but slower extraction)
  - or queue a recompute task (recommended)

---

## 5) Detect new themes *without flooding the UI* (promotion gating)

Even with strong matching, truly-new phrases will appear.  
You want to create clusters (so you can track them), but avoid surfacing noisy one-offs.

### 5.1 Create candidate themes, then promote
When creating a brand new cluster:
- set `is_emerging=True`
- set `is_active=True`
- track `first_seen_at`, `last_seen_at`

(You already do `is_emerging=True` on creation.)

Reference:
- cluster creation in `_get_or_create_cluster()`  
  https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/services/theme_extraction_service.py

### 5.2 Promotion rules (practical defaults)
Promote from “candidate” → “active theme” when:
- `mentions_7d >= 3` AND
- distinct sources in 30d >= 2 AND
- average confidence >= 0.7

You already store mention metrics in `ThemeMetrics` and have fields for lifecycle:
- `is_emerging`, `is_active`, `first_seen_at`, `last_seen_at`  
  https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/models/theme.py

### 5.3 Dormancy + reactivation (tracking old themes)
- Mark theme as dormant if `last_seen_at < now - 60 days` (keep record)
- If matched again, reactivate and update `last_seen_at`

---

## 6) Merge duplicates safely (use suggestions + audit trail)

You already have:
- `theme_merge_suggestions`
- `theme_merge_history`
…but you need a task that populates suggestions and a merge workflow.

Reference:
- Merge suggestion/history models in `backend/app/models/theme.py`  
  https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/models/theme.py

### 6.1 Add daily “merge suggestion” job
For each active cluster:
- find top-k nearest neighbors by embedding similarity
- if similarity > threshold (e.g., 0.93), insert `ThemeMergeSuggestion`

### 6.2 Auto-merge obvious cases (optional)
Auto-merge if:
- cosine similarity >= 0.95 AND
- ticker overlap >= 0.6 (based on `ThemeConstituent`)

### 6.3 Record merges to `ThemeMergeHistory`
When merging:
- always write an audit row with:
  - source/target ids + names
  - similarity
  - merged counts (mentions, constituents)
  - merge type (auto/manual)

---

## 7) One-time cleanup to reduce your existing 1.8k clusters

After match-before-create is deployed (so you stop creating new near-duplicates):

### 7.1 Backfill embeddings for all existing clusters
- run `backfill_theme_embeddings`

### 7.2 Cluster themes in batch
Options:
- Agglomerative clustering (cosine threshold)
- HDBSCAN (if you bring it in)

### 7.3 Merge clusters group-by-group
For each embedding cluster group:
- select canonical cluster (max mentions, most recent, or oldest)
- merge others into it

---

## 8) Optional: where keyphrase extraction can help (TextRank / RAKE / YAKE)

Keyphrase extraction is **not sufficient** by itself, but it can reduce LLM variance and cost.

### 8.1 Best use: candidate generation + constraints
- extract top N keyphrases from each article
- embed keyphrases / article
- match to existing theme cluster by embedding similarity
- only use the LLM to:
  - assign a canonical label to a cluster (once)
  - map constituents (tickers) when needed

### 8.2 Best use: improve LLM consistency
Provide the LLM:
- top keyphrases
- top-10 candidate existing clusters (from embedding search)
…and ask it to pick best match or create a new theme.

This reduces the chance the LLM invents a new phrasing for an existing concept.

---

## 9) Rollout order (low risk → high impact)

1) **Add UI/API filtering**: show only active + top candidates  
2) **Fix Gotcha #2**: normalization improvements (acronym-safe)  
3) **Add embeddings + match-before-create** (largest reduction in duplicates)  
4) **Backfill embeddings** and run once  
5) **Add merge suggestions** task + manual review UI  
6) **Fix Gotcha #1**: schema uniqueness to `(pipeline, name)` (SQLite table rebuild)  
7) **Run one-time consolidation** of existing clusters

---

## 10) Success metrics (what to measure)

Track per day / per extraction run:
- clusters created per 100 articles (should drop sharply)
- % mentions matched to existing clusters (should rise)
- average mentions per active cluster (should rise)
- number of themes shown in UI (should become manageable)

---

## Appendix A — Files to touch / add

### Modify
- `backend/app/services/theme_extraction_service.py`
  - replace `.title()` normalization
  - implement `_match_cluster()` + match-before-create
- `backend/app/models/theme.py`
  - change uniqueness constraint for `ThemeCluster` (see migration)
- `backend/app/main.py`
  - add `run_theme_clusters_migration()` to startup lifecycle

### Add
- `backend/app/services/theme_embedding_service.py`
- `backend/app/tasks/theme_embedding_tasks.py` (or add tasks to existing theme tasks)
- `backend/app/db_migrations/theme_clusters_migration.py`

### Reference: current db + migration style
- `backend/app/database.py` (init + WAL + create_all)
  - https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/database.py
- `backend/app/db_migrations/universe_migration.py` (idempotent migration pattern)
  - https://raw.githubusercontent.com/xang1234/stock-screener/main/backend/app/db_migrations/universe_migration.py

