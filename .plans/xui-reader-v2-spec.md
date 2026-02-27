# XUI Reader v2 Specification
*A small Python package for read-only reading of X Lists + specific accounts via the normal web UI (Playwright + saved logged-in session state).*

**Version:** 2.0 (spec)  
**Date:** 2026-02-27  
**Status:** Design ready for implementation

---

## Table of contents
- [1. Goals and non-goals](#1-goals-and-non-goals)
- [2. Product overview](#2-product-overview)
- [3. CLI user experience](#3-cli-user-experience)
- [4. Configuration](#4-configuration)
- [5. Data model](#5-data-model)
- [6. Package architecture](#6-package-architecture)
- [7. Browser automation design](#7-browser-automation-design)
- [8. Collectors](#8-collectors)
- [9. Extraction and normalization](#9-extraction-and-normalization)
- [10. Storage design](#10-storage-design)
- [11. Scheduling and watch mode](#11-scheduling-and-watch-mode)
- [12. Performance strategy](#12-performance-strategy)
- [13. Reliability strategy](#13-reliability-strategy)
- [14. Diagnostics and observability](#14-diagnostics-and-observability)
- [15. Security and privacy](#15-security-and-privacy)
- [16. Testing strategy](#16-testing-strategy)
- [17. Packaging and distribution](#17-packaging-and-distribution)
- [18. Milestones](#18-milestones)
- [19. Future ideas](#19-future-ideas)
- [20. Implementation elaboration and delivery blueprint](#20-implementation-elaboration-and-delivery-blueprint)

---

## 1. Goals and non-goals

### Goals
1) **Read-only CLI**: read posts from:
   - **X Lists** (by list ID or list URL)
   - **Specific accounts** (by handle or profile URL)

2) **Normal web UI method**:
   - Uses Playwright to load `x.com` pages
   - Auth via Playwright **storage state** (cookies/localStorage) saved after a manual login.

3) **“What’s new” workflows**:
   - `--new` for each source (list/user)
   - Per-source **checkpoints** so users can run it repeatedly and only see new items.

4) **Robustness-first**:
   - Natural brakes (stop scrolling when no new IDs appear)
   - Conservative pacing + jitter
   - Fail-closed if not logged in / interstitial blocks.

5) **Useful outputs**:
   - `pretty` terminal output (human)
   - `plain` (tab-separated) for piping
   - `json` / `jsonl` for tooling and automation

### Non-goals (explicit)
- **No posting / liking / following / bookmarking / DMs**.
- **No private GraphQL client** (no query-id scraping/refresh, no “browser header mimicry playbook”).
- **No “stealth” automation tuning** (fingerprint spoofing, challenge bypassing).  
  If you see a login wall or challenge, the tool stops and asks for manual intervention.

---

## 2. Product overview

### Core user journeys
1) **Login once** (headful browser):
   - `xui auth login --profile default`
   - Saves `storage_state.json` for that profile.

2) **Read a list**
   - `xui list read 84839422 --limit 50 --new`

3) **Read a user**
   - `xui user read @somehandle --limit 30 --new`

4) **Read many sources and merge**
   - `xui read` (reads sources from config; merges and sorts by time)

5) **Run hourly**
   - `xui watch` with:
     - hourly cadence
     - daily shutdown window (8–10 hours)
     - budgets to avoid “always-on” behavior

---

## 3. CLI user experience

### Command summary
```
xui
├── auth
│   ├── login         # open browser, manual login, save storage_state
│   ├── status        # verify the session is still logged in
│   └── logout        # delete storage_state for a profile
├── profiles
│   ├── list
│   ├── create
│   ├── delete
│   └── switch
├── list
│   ├── read          # read timeline for a list ID / URL
│   └── parse-id      # extract list ID from a URL
├── user
│   ├── read          # read timeline for a handle / URL
│   └── parse-handle  # extract handle from a URL
├── read              # read configured sources (lists + users), merge output
├── watch             # daemon-style polling with shutdown window and budgets
├── doctor            # diagnostics and debug artifacts
└── config
    ├── init          # write a default config file
    └── show          # print resolved config and active paths
```

### Global options (apply to most commands)
- `--profile NAME` (default: config `app.default_profile`)
- `--format pretty|plain|json|jsonl`
- `--headful/--headless` (defaults vary by command)
- `--debug` (saves artifacts on failures; increases logging)
- `--timeout-ms` (navigation + selector waits)

### `list read` command
**Purpose:** read posts from an X List timeline via UI.

**Signature**
```
xui list read <list_id_or_url>
  [--limit N]
  [--max-scrolls N]
  [--scroll-delay-ms 1250]
  [--scroll-jitter-ms 250]
  [--new]
  [--checkpoint-mode id|time]
  [--include-reposts/--exclude-reposts]
  [--include-pinned/--exclude-pinned]
  [--expand-truncated]
  [--format ...]
  [--output FILE]
```

### `user read` command
**Purpose:** read posts from a specific account’s timeline via UI.

**Signature**
```
xui user read <handle_or_url>
  [--tab posts|replies|media]
  [--limit N]
  [--max-scrolls N]
  [--new]
  [... same output + pacing flags ...]
```

### `read` command (multi-source)
Reads all enabled sources from config and merges items by `created_at` then `id`.

**Signature**
```
xui read
  [--sources "list:84839422,user:somehandle"]
  [--limit-per-source N]
  [--new]
  [--format ...]
```

### `watch` command (daemon)
Designed for long-running usage and cron-like operation:
- Poll hourly with jitter
- Respect shutdown window
- Enforce daily budgets (page loads, scroll loops)

**Signature**
```
xui watch
  [--interval-sec 3600]
  [--jitter-pct 0.07]
  [--shutdown "01:00-10:00"]
  [--max-runs-per-day 24]
  [--daily-budget-page-loads 200]
  [--daily-budget-scrolls 500]
  [--once]
  [--new]
```

---

## 4. Configuration

### Files and directories (per profile)
Use `platformdirs` to resolve the base config dir (platform-specific).

**Example (Linux):**
```
~/.config/xui-reader/
  config.toml
  selectors/
    default.json
    override.json
  profiles/
    default/
      storage_state.json
      state.json
      tweets.sqlite3
      artifacts/
```

### Config format
**TOML** (Python 3.11+ has `tomllib` built in).

### Example `config.toml`
```toml
[app]
default_profile = "default"
timezone = "Asia/Singapore"
default_format = "pretty"

[browser]
engine = "chromium"           # chromium|firefox|webkit
headless = true
navigation_timeout_ms = 30000
action_timeout_ms = 10000
block_resources = true        # block images/video/fonts by default
viewport_width = 1280
viewport_height = 720
locale = "en-US"

[collection]
limit = 50
max_scrolls = 10
scroll_delay_ms = 1250
scroll_jitter_ms = 250
stagnation_rounds = 2         # stop after N scrolls with no new tweet IDs
expand_truncated = false
include_reposts = true
include_pinned = false

[checkpoints]
mode = "id"                   # id|time
stop_early_on_old = true
old_streak_to_stop = 12       # stop after N old tweets in a row once new found

[scheduler]
interval_sec = 3600
jitter_pct = 0.07
shutdown_local = "01:00-10:00"
daily_budget_page_loads = 200
daily_budget_scrolls = 500

[storage]
db_filename = "tweets.sqlite3"
keep_days = 30
store_raw_html = false
store_raw_json = true

[selectors]
pack = "default"
override_filename = "selectors/override.json"

[[sources]]
id = "list:84839422"
kind = "list"
list_id = "84839422"
label = "Tech List"
enabled = true

[[sources]]
id = "user:somehandle"
kind = "user"
handle = "somehandle"
tab = "posts"
label = "@somehandle"
enabled = true
```

### Selector overrides (`selectors/override.json`)
User-editable file that can patch DOM selectors without code changes.

```json
{
  "tweet_article": "article[data-testid='tweet']",
  "tweet_text": "div[data-testid='tweetText']",
  "user_name_block": "div[data-testid='User-Name']",
  "time": "time",
  "status_link": "a[href*='/status/']"
}
```

---

## 5. Data model

### Source
```python
@dataclass(frozen=True)
class Source:
    id: str                 # e.g. "list:84839422", "user:somehandle"
    kind: Literal["list","user"]
    label: str
    enabled: bool
    params: dict            # list_id/handle/tab/etc
```

### TweetItem (normalized record)
```python
@dataclass
class TweetItem:
    id: str
    url: str
    source_id: str

    author_handle: str | None
    author_name: str | None

    created_at: datetime | None
    text: str

    is_reply: bool | None
    is_repost: bool | None
    is_pinned: bool | None

    quoted_tweet_id: str | None
    quoted_url: str | None

    fetched_at: datetime
    raw: dict | None
```

### Checkpoint
```python
@dataclass
class Checkpoint:
    source_id: str
    last_seen_id: str | None
    last_seen_time: datetime | None
    updated_at: datetime
```

---

## 6. Package architecture

### Package name and CLI
- **Project (PyPI):** `xui-reader`
- **Python module:** `xui_reader`
- **CLI entrypoint:** `xui`

### Recommended folder structure (src layout)
```
xui-reader/
  pyproject.toml
  README.md
  src/
    xui_reader/
      __init__.py
      cli.py

      paths.py
      config.py
      models.py
      errors.py
      redact.py
      logging.py

      browser/
        session.py
        routes.py
        overlays.py
        waits.py

      selectors/
        pack.py
        default.json

      collectors/
        base.py
        list_timeline.py
        user_timeline.py

      extract/
        tweet_extractor.py
        normalize.py

      store/
        db.py
        schema.sql
        migrations/
          001_init.sql
        checkpoints.py
        retention.py

      render/
        pretty.py
        plain.py
        jsonout.py

      scheduler/
        watch.py
        shutdown.py
        budget.py

      diagnostics/
        doctor.py
        artifacts.py

  tests/
    test_shutdown.py
    test_budget.py
    test_checkpoint.py
    test_extractor_snapshots.py
```

### Key interfaces
- `BrowserSessionManager`: create/reuse context; load storage_state; apply routing; provide pages.
- `Collector`: `collect(page, source, cfg) -> list[TweetItem], CheckpointCandidate`
- `Store`: save tweets, update checkpoints, query “new since last”.
- `Renderer`: output formatting.

---

## 7. Browser automation design

### Authentication method (storage state)
- `auth login` opens a **headful** browser, user logs in manually, then:
  - `context.storage_state(path=storage_state.json)`

**Rules**
- Never attempt to scrape OS cookie stores.
- Never print storage_state contents.
- Treat storage_state as password-class secret.

### Browser session lifecycle
- Interactive read commands:
  - Launch browser/context once per command.
  - Reuse one context for multiple sources (multi-source `read`) to avoid repeated cold starts.
- Watch mode:
  - Launch browser once, reuse across cycles (with periodic restart every N cycles to avoid memory leaks), OR
  - Launch per cycle for simplicity and crash containment (configurable).

**Default**: reuse per cycle (cheaper), restart every 6 cycles (configurable).

### Navigation strategy
- Canonical URLs:
  - List: `https://x.com/i/lists/<list_id>`
  - User: `https://x.com/<handle>` then click tab if needed.
- Bounded timeouts; on timeout, capture artifacts and fail the source (not the whole run).

### Overlay handling (non-evasive)
Allowed “quality-of-life” dismissal only:
- “Turn on notifications”
- “Use app” prompts
- Cookie consent banners (if present)

Not allowed: bypassing login walls, challenges, CAPTCHAs.

---

## 8. Collectors

Collectors are responsible for:
1) Navigating to the correct page/tab
2) Running the timeline scroll loop
3) Returning extracted `TweetItem`s + a candidate checkpoint

### 8.1 ListCollector
**Input**: `Source(kind="list", params={list_id})`  
**Page**: `https://x.com/i/lists/<list_id>`

### 8.2 UserCollector
**Input**: `Source(kind="user", params={handle, tab})`

**Page**: `https://x.com/<handle>` then:
- if tab != posts: click the tab link by robust selector strategy:
  - match by href endings (preferred)
  - fallback: visible text in nav tabs container

### 8.3 MultiSourceReader
Reads multiple sources and merges:
- For each source:
  - collect items
  - apply `--new` filtering (checkpoint)
  - store
- merge by:
  - `created_at` desc (if present), else `id` numeric desc

---

## 9. Extraction and normalization

### Selector pack strategy
- Built-in `selectors/default.json` shipped with package
- Optional user override file `selectors/override.json`
- Resolver merges (override wins)

### Extraction algorithm (primary + fallback)
**Primary extraction**
- Locate each tweet article (e.g., `article[data-testid="tweet"]`)
- From each article:
  - status link `/status/<id>`
  - time element `datetime`
  - handle from user block link
  - tweet text block

**Fallback extraction**
- Find status links first, walk up to nearest `article`, then attempt the same fields.

### Truncated text
Config: `expand_truncated` (default false)
- If enabled, click “Show more” within the tweet article before reading text.
- Strictly bounded (max clicks per run/source).

### Deduplication
- In-run: `seen_ids` set
- Cross-run: SQLite primary key `tweet_id` + source_id indexing

### Natural brakes (stop conditions)
Stop when:
- Collected >= limit
- scrolls >= max_scrolls
- stagnant_rounds >= stagnation_rounds (no new IDs discovered)
- `--new` + stop-early logic indicates only old tweets remain

---

## 10. Storage design

### Storage types
1) **State JSON** (lightweight)
- per profile: `state.json`
- stores:
  - checkpoints per source
  - daily budgets counters
  - last success timestamps

2) **SQLite DB** (primary v2 store)
- checkpoints, dedupe, caching, retention

### SQLite schema (v2)
`schema.sql` (initial migration):

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  label TEXT,
  params_json TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
  source_id TEXT PRIMARY KEY,
  last_seen_id TEXT,
  last_seen_time INTEGER,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS tweets (
  tweet_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  author_handle TEXT,
  author_name TEXT,
  created_at INTEGER,
  text TEXT,
  url TEXT NOT NULL,
  is_reply INTEGER,
  is_repost INTEGER,
  is_pinned INTEGER,
  quoted_tweet_id TEXT,
  quoted_url TEXT,
  fetched_at INTEGER NOT NULL,
  raw_json TEXT,
  raw_html TEXT,
  FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE INDEX IF NOT EXISTS idx_tweets_source_created
ON tweets(source_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tweets_source_id
ON tweets(source_id, tweet_id DESC);

CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  success INTEGER,
  error_code TEXT,
  error_message TEXT,
  counters_json TEXT
);
```

### Retention
- Config `storage.keep_days` default 30
- Retention job deletes old tweets by `fetched_at` (or `created_at`).

---

## 11. Scheduling and watch mode

### Watch loop behavior
1) Load config + sources
2) If in shutdown window → sleep until end
3) Apply jitter to next run time
4) Execute one `read` cycle:
   - per source: navigate, collect, store, render delta
5) Persist:
   - updated checkpoints
   - budgets and metrics
6) Repeat

### Shutdown window parsing
- Format: `"HH:MM-HH:MM"` local time
- Supports windows that cross midnight (e.g., `23:00-07:00`)

### Budget model
Budgets are safety valves:
- `daily_budget_page_loads`
- `daily_budget_scrolls`

If exceeded:
- stop further collection
- record counters
- exit non-zero (optional)

---

## 12. Performance strategy

### Resource blocking
If `browser.block_resources = true`, abort request types:
- image
- media (video/audio)
- font
- (optional) stylesheet — use caution

Keep:
- document
- script
- xhr/fetch (required for dynamic timeline loading)

### Session reuse
- Reuse one browser context for multiple sources per run.
- Watch mode: reuse across cycles, restart periodically.

### Concurrency (optional)
Default concurrency:
- Interactive `read`: up to 2 sources concurrently (two pages) max
- Watch mode: 1 (sequential) for stability

---

## 13. Reliability strategy

### Timeouts
- Navigation timeout: 30s default
- Selector wait timeout: 30s for first tweet article
- Action timeout: 10s

### Retries
- Per source:
  - 1 retry on transient navigation timeouts
  - 0 retries on login wall / challenge detection (fail closed)

### Fail-closed conditions
If detected:
- stop collection for that source
- record an actionable error
- optionally exit non-zero in watch mode

Triggers:
- Redirect to `/login` or obvious login wall
- Challenge page indicators (generic “verify” flows)
- No timeline + no error after multiple scrolls (likely blocked or selector mismatch)

### “New only” checkpoint logic
Because pinned/older resurfaced items exist:
- Default checkpoint: **max tweet ID printed** per source.
- Stop early once you hit old streak:
  - once at least one new tweet is found, stop after `old_streak_to_stop` consecutive “old” tweets.
- Exclude pinned items from old-streak counting by default.

### Natural brakes (anti-infinite loop)
Stop scrolling after `stagnation_rounds` with no new IDs.

---

## 14. Diagnostics and observability

### `doctor` command
Checks:
1) Config parse + resolved paths
2) Storage state exists and loads
3) `auth status` check
4) Smoke test:
   - load 1 list source and extract 1 tweet
   - load 1 user source and extract 1 tweet

Artifacts on failure:
- screenshot (PNG)
- HTML snapshot of primary column
- “selector report” (match counts)

### Structured local logs (optional)
- Default: minimal
- `--debug`: JSON logs to `profiles/<profile>/artifacts/logs/*.jsonl`

Counters:
- `page_loads`, `scrolls`, `tweets_seen`, `tweets_emitted`, `sources_ok`, `sources_failed`

Redaction rules:
- Never log cookies/storage state
- Never log full page HTML unless explicitly enabled

---

## 15. Security and privacy

### Treat cookies as password-class secrets
- storage_state contains session credentials
- protect file permissions (POSIX 0600)
- no telemetry
- strict redaction

### Optional protections
- At-rest encryption for storage_state (extra dependency), with key via OS keychain or passphrase.

---

## 16. Testing strategy

### Unit tests
- Shutdown window parsing
- Budget counter reset
- Checkpoint update rules
- Natural brakes logic

### Snapshot tests
- Sanitized HTML fragments (tweet article blocks) to test:
  - tweet ID parsing
  - time extraction
  - handle extraction
  - text extraction

### Integration tests (gated)
- Optional live smoke tests behind env flag.

---

## 17. Packaging and distribution

### Python version
- Python 3.11+

### Dependencies
Core:
- playwright
- typer
- rich (recommended)
- platformdirs
- pydantic (recommended)

### `pyproject.toml` (sketch)
```toml
[project]
name = "xui-reader"
version = "2.0.0"
requires-python = ">=3.11"
dependencies = [
  "playwright>=1.40",
  "typer>=0.12",
  "rich>=13.0",
  "platformdirs>=4.0",
  "pydantic>=2.0"
]

[project.scripts]
xui = "xui_reader.cli:app"
```

Install note:
- `python -m playwright install chromium`

---

## 18. Milestones

### v2.0 (ship)
- Package skeleton + CLI entrypoint
- Profiles + storage_state login/status/logout
- ListCollector + UserCollector
- Extractor with selector packs + overrides
- Checkpoints + `--new`
- Outputs: pretty/plain/json/jsonl
- Artifacts + doctor

### v2.1 (hardening)
- SQLite cache + retention
- `read` multi-source merge
- Watch mode with shutdown window + budgets
- Snapshot tests and gated live tests

### v2.2 (quality-of-life)
- Export to markdown / RSS
- Hooks: `on_new` command execution with JSON payload
- Search (SQLite FTS5)

---

## 19. Future ideas
- TUI mode (curses/rich-live) with keyboard navigation
- Smart filtering rules (include/exclude words)
- Notification plugins (user-provided hooks only; no telemetry)

---

## Appendix: Design principles from your research notes
This spec intentionally adopts:
- Read-only default posture
- Natural brakes (stop when progress stalls)
- Conservative cadence (hourly) and daily shutdown window
- Local-only state/caching; strict secret handling and redaction
- Runtime overrides for “fast fixes” when the upstream UI changes

---

## 20. Implementation elaboration and delivery blueprint

This section translates the design into an execution-ready build plan with explicit sequencing, invariants, and “why this order” rationale. It is intended to be the implementation contract for tickets, reviews, and release decisions.

### 20.1 Workstream decomposition (what gets built and why)

#### WS0: Foundations and scaffolding
**Scope**
- repo skeleton (`src/` layout, pyproject, entrypoint wiring)
- shared models/errors/logging/redaction
- config loading + path resolution

**Reasoning**
- Every downstream workstream consumes these contracts.
- Stabilizing type boundaries early reduces refactor churn once collector logic lands.

**Output**
- Runnable `xui --help`
- validated config schema and default file generation
- central error taxonomy used by CLI exit codes and diagnostics

#### WS1: Auth and profile lifecycle
**Scope**
- `auth login/status/logout`
- profile create/list/delete/switch
- storage_state persistence with secret-grade handling

**Reasoning**
- No collector can run safely without deterministic auth state and profile resolution.
- This is the first user-facing trust boundary; correctness beats feature count.

**Output**
- profile directory bootstrap
- manual login capture path
- fail-closed status checks for expired/invalid sessions

#### WS2: Browser session and collection engine
**Scope**
- browser/context factory
- routing/resource blocking policy
- overlay dismissal (allowed-only)
- list/user collectors + scroll loop + natural brakes

**Reasoning**
- This is the highest breakage surface (external DOM volatility + network timing).
- Must be isolated behind stable interfaces before extraction/storage tuning.

**Output**
- `list read` and `user read` collect loops that terminate safely under all stop conditions
- deterministic per-source counters for diagnostics

#### WS3: Extraction and normalization robustness
**Scope**
- selector pack loading/merging
- primary + fallback extraction flows
- truncated expansion bounds
- in-run dedupe and normalization

**Reasoning**
- Data quality is the core product value.
- Extraction is versioned logic that needs test fixtures and clear fallback semantics.

**Output**
- normalized `TweetItem` with consistent nullability/typing
- selector override path for fast prod fixes without code release

#### WS4: Storage, checkpoints, and new-only semantics
**Scope**
- SQLite init/migrations
- sources/checkpoints/tweets/runs persistence
- checkpoint advancement and old-streak early stop
- retention job and bounded raw payload storage

**Reasoning**
- `--new` is a first-class product promise.
- durability and dedupe semantics are where silent regressions are most expensive.

**Output**
- idempotent upsert model
- per-source checkpoint integrity and replay-safe behavior

#### WS5: Multi-source read and scheduler/watch
**Scope**
- `read` merged execution over configured sources
- watch loop cadence, jitter, shutdown window, and daily budgets
- restart policy and graceful interruption

**Reasoning**
- Operational behavior (long-running automation) amplifies all reliability flaws.
- Must ship with guardrails preventing “infinite collector” posture.

**Output**
- bounded, budget-aware daemon behavior
- clear run accounting for page loads/scrolls/success-failure counts

#### WS6: Diagnostics, observability, and operator UX
**Scope**
- `doctor` command checks
- debug artifact capture and redaction gates
- structured logging (opt-in debug mode)

**Reasoning**
- External UI automation fails in unpredictable ways; diagnosis speed determines MTTR.
- Artifacts must be useful without leaking sensitive state.

**Output**
- reproducible triage bundle (screenshot + constrained HTML + selector report)
- actionable error guidance and next-step messaging

#### WS7: Test matrix, packaging, and release hardening
**Scope**
- unit/snapshot/integration gates
- CLI smoke tests
- package metadata and install docs

**Reasoning**
- Without regression gates, selector and parser changes will drift silently.
- Distribution quality determines real-world adoption cost.

**Output**
- repeatable CI commands
- versioned release checklist for v2.0/v2.1/v2.2 increments

### 20.2 Build phases and critical path

#### Phase A: Contract-first bootstrap
Includes WS0 + minimal WS1.

**Exit gate**
- config initializes
- profile exists
- `auth login` saves state
- `auth status` reports deterministic pass/fail

#### Phase B: Core reading path (single source)
Includes WS2 + WS3 + initial WS4.

**Exit gate**
- `list read` and `user read` return normalized outputs
- stop conditions proven in tests (limit, max_scrolls, stagnation)
- checkpoint write/read works for one source

#### Phase C: Multi-source + persistence hardening
Completes WS4 and introduces WS5 `read`.

**Exit gate**
- multi-source merge ordering deterministic (`created_at`, tie-break by id)
- in-run and cross-run dedupe validated
- retention and migrations verified on non-empty DB

#### Phase D: Watch mode and operability
Completes WS5 + WS6.

**Exit gate**
- watch loop honors shutdown windows and budgets
- daily counter reset semantics validated
- `doctor` produces actionable artifacts for forced failures

#### Phase E: Verification and release
Completes WS7.

**Exit gate**
- test suite green
- packaging/install smoke succeeds
- release notes include known limits and operator runbook

### 20.3 Command-level Definition of Done (DoD)

#### `auth login`
- launches headful browser only
- saves storage state with secure permissions
- does not print credential-bearing payload

#### `auth status`
- fast deterministic probe
- exits non-zero on invalid/expired state
- surfaces recovery action text

#### `list read` / `user read`
- bounded completion via natural brakes
- emits selected format exactly
- when `--new` is active, outputs only unseen records for that source

#### `read`
- respects enabled sources
- per-source failures isolated and reported
- merged output globally ordered and deterministic

#### `watch`
- respects interval + jitter + shutdown + budget limits
- persists per-run counters
- terminates safely on interrupt with state flush

#### `doctor`
- verifies config/auth/storage preconditions
- executes minimal live smoke (if configured source exists)
- writes artifacts with redaction policy

### 20.4 Non-negotiable invariants (guardrails)

1) **Read-only posture**: no mutating X actions are introduced.
2) **Fail-closed auth**: suspected login wall/challenge halts collection.
3) **Bounded loops**: all scroll and retry behavior has strict hard caps.
4) **Secret hygiene**: storage_state treated as password-class secret and never logged.
5) **Deterministic `--new`**: checkpoint semantics are source-local and replay-safe.
6) **Operator clarity**: every failure path emits actionable next steps.

### 20.5 Risk register with mitigation strategy

#### R1: Selector drift from X UI changes
- **Impact**: extraction misses fields or returns empty output.
- **Mitigation**:
  - selector overrides shipped as runtime config
  - snapshot fixtures versioned and required in PRs touching extraction
  - `doctor` selector report highlights zero-match selectors quickly

#### R2: Session invalidation and interstitial challenges
- **Impact**: runs silently degrade or loop.
- **Mitigation**:
  - explicit login/challenge detection checks
  - no retry on fail-closed conditions
  - clear manual remediation guidance in CLI errors

#### R3: Watch-mode resource creep
- **Impact**: memory growth, excessive page activity, platform suspicion risk.
- **Mitigation**:
  - cycle restart policy
  - daily budget ceilings
  - shutdown window to force off-hours idle

#### R4: Checkpoint correctness regressions
- **Impact**: duplicate floods or missed new posts.
- **Mitigation**:
  - unit tests around old-streak and pinned handling
  - checkpoint updates only after successful parse/store path
  - run metadata to audit source-level counts

#### R5: Data leakage in diagnostics
- **Impact**: credential exposure in logs/artifacts.
- **Mitigation**:
  - redaction utility centralized
  - artifact capture excludes storage state by design
  - raw HTML capture off by default, explicit opt-in only

### 20.6 Implementation notes for future maintainers

1) Prefer additive schema migrations and explicit migration IDs over in-place mutation.
2) Keep collector logic free of output-format concerns; rendering belongs in `render/*`.
3) Preserve “safety before completeness”: stop early on ambiguous blocked states rather than guessing.
4) Any selector update must include a fixture/snapshot update in the same change.
5) New command flags must be reflected in:
   - CLI help text
   - config defaults (if applicable)
   - doctor output or run metadata when operationally relevant

### 20.7 Mapping to issue planning

The issue hierarchy should mirror WS0-WS7 with:
- one root epic for the full v2 initiative
- one epic per workstream
- tasks/subtasks aligned to phase gates
- explicit cross-workstream dependencies only where contract-level blocking exists

This prevents hidden coupling and keeps execution order auditable in the dependency graph.
