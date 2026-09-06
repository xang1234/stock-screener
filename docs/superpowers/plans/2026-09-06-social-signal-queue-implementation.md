/Users/admin/.zshenv:.:1: no such file or directory: /Users/admin/.cargo/env
# Social Signal Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-only, shared Social Signal Queue that combines administrator-managed X lists, resolves US/HK/CN/JP/TW securities, ranks them with deterministic Blended and Pure Social modes, and works with either the official X API or the private `xui-reader` CLI without making the private package a public-repository dependency.

**Architecture:** Keep all domain models, source administration, persistence, scoring, APIs, UI, fixtures, and the official provider in the public repository. Put provider reads behind a provider-neutral port and run them on a dedicated `social_ingestion` Celery queue. The private image extends the public backend image, installs `xui-reader` through a credential-safe trusted build, and mounts a dedicated authenticated profile only into that worker. Publish immutable all-enabled-source snapshots through an atomic pointer; live reads never assemble mixed or partial data.

**Tech Stack:** Python 3, FastAPI, Pydantic, SQLAlchemy, Alembic, PostgreSQL/SQLite, Celery, Redis, httpx, React 18, TanStack Query, MUI, Vitest, Playwright, Docker BuildKit, Docker Compose, GitHub Actions, private GHCR.

**Spec:** `docs/superpowers/specs/2026-09-06-social-signal-queue-design.md` (original approval commit `8a57ccec`, revised on `feat/social-signal-queue` for administrator-managed sources).

## Global Constraints

- Begin implementation on a dedicated `feat/social-signal-queue` worktree whose history contains design commit `8a57ccec`; do not implement on the current release branch.
- `SOCIAL_INGEST_PROVIDER` is exactly `disabled|official|xui`, defaults to `disabled`, and never falls back automatically.
- Public dependency files, public images, public CI, and fork PRs must not fetch, import, cache, or require `xui-reader`.
- The private adapter invokes the exact auth/read commands documented in Task 6 as argument arrays with `shell=False`; it never imports private Python modules or invokes interactive login.
- Seed source IDs `1522014550211457024` and `1986290701492232693` as enabled, named system sources. Administrators may add more installation-wide lists without redeployment.
- At least two sources must remain enabled. Every source pinned as enabled at run start must succeed completely before the run may publish.
- New sources require a non-blank local display name, start pending, and cannot be enabled until an explicit test reads at most five posts successfully through the currently selected provider.
- List IDs are immutable; display names are editable; removal archives rather than deletes; every source mutation and test produces a redacted immutable audit event.
- Scheduled reads run every six hours; manual refresh is admin-only and limited to one accepted dispatch per hour.
- First-run coverage is fourteen days with a hard 1,000-post cap per list. A cap-truncated source is incomplete and cannot publish.
- Social windows are rolling UTC `1D|7D|14D`; supported Markets are exactly `US|HK|CN|JP|TW`.
- Formula version `social-signal-v1` uses `queue_score = 0.60 * social_score + 0.40 * confirmation_score`. Pure Social changes ordering only.
- Social weights are acceleration 20, authors 15, engagement 15, recency 5, cross-list 5. Confirmation weights are setup 20, RS 10, group 5, theme 5. Missing inputs are omitted from the available-weight denominator and retained as missing in explanation payloads.
- Percentiles use deterministic mid-rank: `100 * (less_count + 0.5 * equal_count) / cohort_count`; a singleton cohort yields 50. Use per-Market cohorts when at least 20 candidates exist, otherwise one global supported-Market cohort.
- Engagement uses the approved log formula and a deterministic linear-interpolation 95th-percentile winsorization threshold. Likes, reposts, and replies must all be observed; optional quote/bookmark/view fields contribute only when observed.
- A post contributes once per ticker; reposts, empty-thesis quotes, and repeated canonical URLs do not add independent weight; one author contributes at most three posts per ticker per rolling 24 hours.
- Existing Theme rankings must exclude sources with `contributes_to_theme_rankings=false`. Social Pulse is read from the published Social run and cannot affect Theme momentum or ordering.
- Broad Market ETFs are `context`; thematic ETFs stay eligible with an ETF badge. Unresolved text never receives fabricated Market or technical fields.
- `actionable` requires an active supported security, fresh feature and Market inputs, the existing Market liquidity gate, `se_setup_ready=true`, and Market exposure at least 50.
- Failed or partial runs remain auditable but cannot replace the last complete pointer. A published run older than seven hours is shown as stale but remains readable.
- Signed-in users may read; `require_admin` protects health details, source creation/rename/test/state changes, run history, and manual refresh. Responses, audit events, and logs never expose secrets, storage-state paths, post bodies from tests, cookies, raw headers, or debug artifacts.
- Static exporters and `frontend/src/static/**` receive no social feature, API request, data, or route.
- Preserve unrelated workspace changes. Use `apply_patch`, red-green-refactor TDD, focused commits, and verification before every completion claim.

## File Structure

New backend bounded context:

- `backend/app/domain/social_signals/records.py`: provider-neutral posts, source outcomes, security resolutions, and score input/output records.
- `backend/app/domain/social_signals/scoring.py`: deduplication, caps, percentiles, component calculations, and stable ordering.
- `backend/app/domain/social_signals/states.py`: Signal State and ETF classification policy.
- `backend/app/use_cases/social_signals/ports.py`: provider, writer, confirmation reader, published reader, lease, and dispatcher protocols.
- `backend/app/use_cases/social_signals/refresh.py`: one immutable refresh orchestration.
- `backend/app/use_cases/social_signals/validate_source.py`: explicit five-post-or-fewer provider test without evidence persistence or publication.
- `backend/app/use_cases/social_signals/queries.py`: published queue, evidence, pulse, health, and run-history reads.
- `backend/app/infra/providers/official_x_social_provider.py`: official X API adapter.
- `backend/app/infra/providers/xui_cli_social_provider.py`: private CLI adapter without a private import.
- `backend/app/infra/db/models/social_signals.py`: social persistence tables.
- `backend/app/infra/db/repositories/social_signal_writer.py`: transactional observation/run/snapshot writes and atomic publication.
- `backend/app/infra/db/repositories/published_social_signal_reader.py`: pointer-scoped live reads.
- `backend/app/services/social_ticker_resolver.py`: deterministic multi-Market resolution and ETF classification.
- `backend/app/services/social_source_admin_service.py`: list parsing, seed, pending/tested lifecycle, two-enabled invariant, archive behavior, optimistic edits, and audit events.
- `backend/app/services/social_confirmation_reader.py`: coherent feature, exposure, liquidity, group, and non-social theme inputs.
- `backend/app/interfaces/tasks/social_signal_tasks.py`: six-hour/manual worker entry point.
- `backend/app/api/v1/social_signals.py` and `backend/app/schemas/social_signals.py`: authenticated live API.

New frontend feature:

- `frontend/src/api/socialSignals.js`: queue, evidence, pulse, health, and refresh client.
- `frontend/src/features/socialSignals/SocialSignalsTab.jsx`: complete queue surface.
- `frontend/src/features/socialSignals/SocialSignalsTable.jsx`: dense sortable/filterable table.
- `frontend/src/features/socialSignals/SocialEvidenceDrawer.jsx`: explanation and top-three evidence.
- `frontend/src/features/socialSignals/DailySocialSignalsCard.jsx`: top-five Daily Snapshot card.
- `frontend/src/features/socialSignals/SocialSignalHealthPanel.jsx`: Operations/admin panel.
- `frontend/src/features/socialSignals/socialSignalPresentation.js`: state, score, freshness, and route helpers.

Deployment additions:

- `docker-compose.social.yml`: dedicated public/official social worker.
- `docker-compose.social-xui.yml`: private-image/profile overlay.
- `.github/workflows/private-social-worker.yml`: trusted multi-architecture private image build.
- `docs/deployment/social-signal-worker.md`: provider setup, local login, GHCR and local-build workflows, and recovery.

---

### Task 1: Domain Vocabulary, Settings, and Runtime Capability

**Files:**
- Modify: `CONTEXT.md`
- Modify: `backend/app/config/settings.py`
- Modify: `backend/app/schemas/app_runtime.py`
- Modify: `frontend/src/contexts/RuntimeContext.jsx`
- Modify: `.env.docker.example`
- Create: `backend/tests/unit/domain/social_signals/test_settings.py`
- Modify: `backend/tests/unit/test_app_runtime_endpoints.py`
- Modify: `frontend/src/contexts/RuntimeContext.test.jsx`

**Interfaces:**
- Produces settings `social_signals_enabled`, `social_ingest_provider`, schedule hours, freshness hours, caps, xui config/profile, and official API limits. Source membership is database-managed rather than environment-configurable.
- Produces runtime flag `features.social_signals`; it is true only when `SOCIAL_SIGNALS_ENABLED=true` and provider is not disabled.

- [ ] **Step 1: Write failing settings tests**

```python
import pytest
from app.config.settings import Settings

def test_social_ingest_defaults_fail_closed():
    settings = Settings(_env_file=None)
    assert settings.social_ingest_provider == "disabled"
    assert settings.capability_flags()["social_signals"] is False

@pytest.mark.parametrize("value", ["disabled", "official", "xui"])
def test_social_provider_accepts_only_explicit_modes(value):
    assert Settings(_env_file=None, social_ingest_provider=value).social_ingest_provider == value

def test_enabled_flag_does_not_override_disabled_provider():
    settings = Settings(
        _env_file=None,
        social_signals_enabled=True,
        social_ingest_provider="disabled",
    )
    assert settings.capability_flags()["social_signals"] is False
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `cd backend && ./venv/bin/pytest tests/unit/domain/social_signals/test_settings.py tests/unit/test_app_runtime_endpoints.py -q`

Expected: failures for missing social settings and capability.

- [ ] **Step 3: Add exact settings and validators**

Add defaults:

```python
social_signals_enabled: bool = False
social_ingest_provider: str = "disabled"
social_refresh_hours: int = 6
social_manual_refresh_cooldown_seconds: int = 3600
social_stale_after_hours: int = 7
social_initial_backfill_days: int = 14
social_initial_backfill_limit_per_source: int = 1000
social_incremental_limit_per_source: int = 200
social_xui_config_path: str = "/app/data/xui-reader/config.toml"
social_xui_profile: str = "automation"
social_official_daily_post_limit: int = 2000
```

Keep legacy `x_ingest_provider` for the existing generic theme ingestion path; do not silently repurpose it. Validate the new provider independently. Task 4 owns the administrator-managed source registry.

- [ ] **Step 4: Add the capability fallback and vocabulary**

Add `social_signals: false` to `DEFAULT_CAPABILITIES`. Add the approved definitions for Social Source, Social Post, Social Signal Run, Published Social Signal Run, Social Score, Confirmation Score, Queue Score, and Signal State to `CONTEXT.md`.

- [ ] **Step 5: Document all environment variables and run GREEN**

Run: `cd backend && ./venv/bin/pytest tests/unit/domain/social_signals/test_settings.py tests/unit/test_app_runtime_endpoints.py -q`

Run: `cd frontend && npm run test:run -- src/contexts/RuntimeContext.test.jsx`

- [ ] **Step 6: Commit**

```bash
git add CONTEXT.md backend/app/config/settings.py backend/app/schemas/app_runtime.py backend/tests/unit/domain/social_signals/test_settings.py backend/tests/unit/test_app_runtime_endpoints.py frontend/src/contexts/RuntimeContext.jsx frontend/src/contexts/RuntimeContext.test.jsx .env.docker.example
git commit -m "feat: define social signal runtime capability"
```

---

### Task 2: Provider-Neutral Records and Contract Tests

**Files:**
- Create: `backend/app/domain/social_signals/__init__.py`
- Create: `backend/app/domain/social_signals/records.py`
- Create: `backend/app/use_cases/social_signals/__init__.py`
- Create: `backend/app/use_cases/social_signals/ports.py`
- Create: `backend/tests/unit/domain/social_signals/test_records.py`
- Create: `backend/tests/unit/use_cases/social_signals/test_ports.py`
- Create: `backend/tests/fixtures/social/xui_list_read.json`
- Create: `backend/tests/fixtures/social/official_list_read.json`

**Interfaces:**
- `SocialPostRecord` distinguishes `None` from observed zero for every metric.
- `SocialSourceOutcome` records `complete`, `truncated`, checkpoint, counts, and stable error code.
- `SourceTestOutcome` records only provider, status, sample count `0..5`, tested time, and a stable redacted reason code; it contains no posts or raw provider payload.
- `SocialSourceView` is the provider-neutral administrator projection containing source ID, name, canonical URL/list ID, lifecycle/provenance, test summary, collection freshness, optimistic version, and timestamps. `SocialSourceAuditView` contains action, actor, timestamp, and redacted before/after lifecycle metadata.
- `SocialProvider.read_source(request) -> SocialSourceBatch` is the sole ingestion dependency.

- [ ] **Step 1: Write failing immutable-record tests**

```python
from datetime import datetime, timezone
import pytest
from app.domain.social_signals.records import SocialPostRecord

def test_missing_metric_is_not_observed_zero():
    row = SocialPostRecord(
        provider="xui", provider_post_id="101", source_id="1522014550211457024",
        text="$NVDA base", url="https://x.com/a/status/101", author_handle="a",
        created_at=datetime(2026, 9, 5, tzinfo=timezone.utc), observed_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        likes=0, reposts=0, replies=0, quotes=None, bookmarks=None, views=None,
    )
    assert row.likes == 0
    assert row.views is None

def test_future_post_is_rejected():
    with pytest.raises(ValueError, match="future_timestamp"):
        SocialPostRecord.from_untrusted({"tweet_id": "101", "created_at": "2099-01-01T00:00:00Z"}, provider="xui", source_id="1522014550211457024", observed_at=datetime(2026, 9, 6, tzinfo=timezone.utc))
```

- [ ] **Step 2: Define frozen dataclasses and protocols**

Define `SocialReadRequest`, `SocialPostRecord`, `SocialSourceBatch`, `SocialSourceOutcome`, `SourceTestOutcome`, `SocialSourceView`, `SocialSourceAuditView`, `TickerResolution`, `SocialEvidenceInput`, `ConfirmationInput`, `SocialScoreResult`, and `SocialSnapshotRecord`. Reject blank IDs/text/URLs, naive timestamps, future timestamps beyond five minutes, negative metrics, sample counts outside `0..5`, raw provider payload fields, and unsupported providers with stable codes.

- [ ] **Step 3: Add sanitized equivalent fixtures**

Fixtures must contain synthetic handles/text/IDs only and demonstrate missing versus zero metrics, a duplicated cross-list post, a repost, a quote with original thesis text, and CJK company text. Add no cookie, bearer token, storage-state, or live response headers.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/domain/social_signals/test_records.py tests/unit/use_cases/social_signals/test_ports.py -q`

```bash
git add backend/app/domain/social_signals backend/app/use_cases/social_signals backend/tests/unit/domain/social_signals backend/tests/unit/use_cases/social_signals backend/tests/fixtures/social
git commit -m "feat: add provider-neutral social records"
```

---

### Task 3: Social Scoring and Signal State Policy

**Files:**
- Create: `backend/app/domain/social_signals/scoring.py`
- Create: `backend/app/domain/social_signals/states.py`
- Create: `backend/tests/unit/domain/social_signals/test_scoring.py`
- Create: `backend/tests/unit/domain/social_signals/test_states.py`

**Interfaces:**
- `score_social_candidates(evidence, window_days, now) -> tuple[SocialScoreResult, ...]`
- `score_confirmation(input) -> ComponentScore`
- `rank_snapshots(rows, mode) -> tuple[SocialSnapshotRecord, ...]`
- `classify_signal_state(input) -> SignalStateDecision`

- [ ] **Step 1: Write RED tests for deduplication and author caps**

Assert that the same provider post in multiple lists counts once but retains every membership; distinct posts from any two enabled lists receive full binary cross-list credit; evidence from third and later lists does not increase that component; a fourth post from one author in 24 hours has zero mention weight; empty reposts, empty-thesis quotes, and repeated canonical URLs do not inflate counts.

- [ ] **Step 2: Write RED tests for exact formulas**

```python
from math import log1p
from app.domain.social_signals.scoring import engagement_value, queue_score, recency_score

def test_engagement_formula():
    assert engagement_value(likes=10, reposts=2, replies=4, quotes=2, bookmarks=1, views=1000) == pytest.approx(log1p(26.0))

def test_recency_uses_48_hour_half_life():
    assert recency_score(age_hours=48) == pytest.approx(50.0)

def test_blended_queue_score_is_exact():
    assert queue_score(social=80, confirmation=50) == pytest.approx(68.0)
```

Also test mid-rank percentiles, 95th-percentile winsorization, per-Market/global fallback, missing-value renormalization, exact RS blend, exact group formula, highest-theme selection with canonical-key tie-break, and byte-equivalent replay serialization.

- [ ] **Step 3: Write RED state tests**

Cover all five states and assert that changing Market exposure from 65 to 30 changes `actionable` to `risk_off` without changing Social, Confirmation, or Queue scores. Cover inactive Universe rows, stale features, liquidity rejection, setup not ready, broad ETFs, and thematic ETFs.

- [ ] **Step 4: Implement pure scoring and states**

Use `Decimal` internally for weighted aggregation, round stored component and total scores to four decimal places, and keep input coverage and exclusion reasons alongside every component. Pure Social ordering is `(social_score desc, latest_mention desc, canonical_symbol asc)`; Blended adds queue score first.

- [ ] **Step 5: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/domain/social_signals/test_scoring.py tests/unit/domain/social_signals/test_states.py -q`

```bash
git add backend/app/domain/social_signals backend/tests/unit/domain/social_signals
git commit -m "feat: implement deterministic social scoring"
```

---

### Task 4: Persistence Schema and Theme-Ranking Isolation

**Files:**
- Create: `backend/alembic/versions/20260906_0035_add_social_signal_queue.py`
- Modify: `backend/app/models/theme.py`
- Create: `backend/app/infra/db/models/social_signals.py`
- Modify: `backend/app/infra/db/models/__init__.py`
- Modify: `backend/app/models/__init__.py`
- Create: `backend/tests/integration/test_social_signal_migration.py`
- Create: `backend/tests/unit/test_theme_social_source_exclusion.py`
- Modify: `backend/app/services/theme_discovery_service.py`
- Modify: `backend/app/services/theme_pipeline_state_service.py`
- Modify: `backend/app/services/theme_taxonomy_service.py`
- Create: `backend/app/services/social_source_admin_service.py`
- Create: `backend/tests/unit/services/test_social_source_admin_service.py`

**Interfaces:**
- Adds `ContentSource.contributes_to_theme_rankings`, default/server-default true and non-null.
- Produces ORM models `SocialSourceConfiguration` and `SocialSourceAuditEvent` for database-managed list lifecycle and immutable administrator history.
- Adds `social_source_configurations`, `social_source_audit_events`, `social_post_sources`, `social_content_metrics`, `social_post_tickers`, `social_signal_runs`, `social_signal_snapshots`, and `social_signal_run_pointers`.

- [ ] **Step 1: Write RED migration shape tests**

Assert revision `20260906_0035` revises `20260904_0034`; all eight tables, foreign keys, unique constraints, pointer key, score/state/lifecycle checks, and the source opt-out column exist after upgrade and disappear after downgrade.

- [ ] **Step 2: Write RED Theme isolation regression**

Seed two identical mentions where only one source opts out. Assert every Theme aggregate/ranking/count path returns the same result before and after inserting the opted-out mention. Exercise discovery, pipeline-state metrics, taxonomy counts, and Theme API output.

- [ ] **Step 3: Write RED source lifecycle and audit tests**

```python
from datetime import datetime, timezone
import pytest

from app.domain.social_signals.records import SourceTestOutcome
from app.services.social_source_admin_service import (
    SocialSourceAdminService,
    SocialSourceStateError,
)

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)

def test_admin_source_is_pending_until_explicit_current_provider_test(db_session):
    service = SocialSourceAdminService(db_session, current_provider=lambda: "official")
    source = service.create_source("Japan momentum", "https://x.com/i/lists/3001", "admin")
    assert source.lifecycle_state == "pending"
    assert source.x_list_id == "3001"
    assert source.version == 1
    with pytest.raises(SocialSourceStateError, match="current_provider_test_required"):
        service.transition_source(source.id, "enabled", source.version, "admin")

    tested = service.record_test_result(
        source.id,
        "official",
        SourceTestOutcome(provider="official", status="passed", sample_count=5, tested_at=NOW),
        "admin",
    )
    enabled = service.transition_source(tested.id, "enabled", tested.version, "admin")
    assert enabled.lifecycle_state == "enabled"

def test_cannot_drop_below_two_enabled_and_archive_keeps_history(db_session):
    service = SocialSourceAdminService(db_session, current_provider=lambda: "official")
    first, second = service.ensure_seed_sources()
    with pytest.raises(SocialSourceStateError, match="minimum_two_enabled"):
        service.transition_source(first.id, "archived", first.version, "admin")
    assert service.audit_events(first.id)[-1].action == "created"
```

- [ ] **Step 4: Implement schema**

Use these stable keys:

```text
social_post_sources: UNIQUE(content_item_id, content_source_id)
social_content_metrics: UNIQUE(content_item_id)
social_post_tickers: UNIQUE(content_item_id, candidate_key)
social_signal_snapshots: UNIQUE(run_id, window_days, candidate_key)
social_signal_run_pointers: PRIMARY KEY(key), key='latest_published'
social_source_configurations: UNIQUE(content_source_id), UNIQUE(x_list_id)
social_source_audit_events: INDEX(content_source_id, created_at), append-only through repository policy
```

Persist source lifecycle fields exactly as follows:

```text
social_source_configurations:
  content_source_id PK/FK -> content_sources.id ON DELETE RESTRICT
  x_list_id TEXT NOT NULL UNIQUE
  lifecycle_state TEXT NOT NULL CHECK pending|enabled|disabled|archived
  provenance TEXT NOT NULL CHECK system_seed|admin
  tested_provider TEXT NULL CHECK official|xui
  test_status TEXT NULL CHECK queued|running|passed|failed|rate_limited|reauthentication_required|provider_error
  test_sample_count INTEGER NULL CHECK >= 0 AND <= 5
  tested_at TIMESTAMPTZ NULL
  last_successful_collection_at TIMESTAMPTZ NULL
  archived_at TIMESTAMPTZ NULL
  version INTEGER NOT NULL DEFAULT 1
  created_at, updated_at TIMESTAMPTZ NOT NULL

social_source_audit_events:
  id BIGINT PK
  content_source_id FK -> content_sources.id ON DELETE RESTRICT
  action TEXT NOT NULL CHECK created|renamed|test_requested|test_completed|enabled|disabled|archived
  actor TEXT NOT NULL
  before_json JSON NULL
  after_json JSON NOT NULL
  created_at TIMESTAMPTZ NOT NULL
```

`candidate_key` is `MARKET:canonical_symbol` for resolved rows and `"unresolved:" + sha256(f"{raw_token}|{content_item_id}").hexdigest()` otherwise. Store nullable metric columns, pinned JSON explanations, `stock_universe_id`, canonical symbol/Market/MIC/local code, resolution policy version, run source outcomes/checkpoints, feature run IDs per Market, exposure dates, and stable tie-break fields.

- [ ] **Step 5: Implement the administrator-managed source lifecycle**

`SocialSourceAdminService.ensure_seed_sources()` creates these named, enabled system sources idempotently:

```python
SEED_SOCIAL_SOURCES = (
    ("1522014550211457024", "Minervini Research List", "https://x.com/i/lists/1522014550211457024"),
    ("1986290701492232693", "Asia-Pacific Growth List", "https://x.com/i/lists/1986290701492232693"),
)
```

Set `source_type="twitter"`, `pipelines=["technical"]`, `fetch_interval_minutes=360`, `contributes_to_theme_rankings=false`, and lifecycle provenance `system_seed`. Preserve administrator edits on subsequent calls.

Add methods with exact contracts:

```python
create_source(name: str, list_ref: str, actor: str) -> SocialSourceView
rename_source(source_id: int, name: str, expected_version: int, actor: str) -> SocialSourceView
record_test_result(source_id: int, provider: str, outcome: SourceTestOutcome, actor: str) -> SocialSourceView
transition_source(source_id: int, target: Literal["enabled", "disabled", "archived"], expected_version: int, actor: str) -> SocialSourceView
list_sources(include_archived: bool = False) -> tuple[SocialSourceView, ...]
audit_events(source_id: int) -> tuple[SocialSourceAuditView, ...]
```

Accept only a 1-32-digit numeric ID or `https://x.com/i/lists/{id}`. Creation requires a trimmed 1-100-character name, rejects duplicate list IDs, stores the canonical URL, starts `pending`, and sets `ContentSource.is_active=false`. Except for the two `system_seed` rows, enabling requires a passed test matching the current provider. State transitions keep `ContentSource.is_active` true only for `enabled`. Disabling or archiving is rejected when it would leave fewer than two enabled sources. Renaming changes only `ContentSource.name`; the list ID and canonical URL are immutable. Archival is terminal in v1 and never deletes evidence. Every successful mutation, test request, and test completion appends a redacted `SocialSourceAuditEvent`; version mismatches raise a typed optimistic-concurrency error.

- [ ] **Step 6: Apply source opt-out to every Theme metric query**

Join `ThemeMention -> ContentItem -> ContentSource` and require `coalesce(ContentSource.contributes_to_theme_rankings, true)`. Do not exclude social mentions from Theme detail evidence; exclude them only from ranking/momentum/count inputs.

- [ ] **Step 7: Run migration and regressions GREEN**

Run: `cd backend && ./venv/bin/pytest tests/integration/test_social_signal_migration.py tests/unit/test_theme_social_source_exclusion.py tests/unit/services/test_social_source_admin_service.py -q`

- [ ] **Step 8: Commit**

```bash
git add backend/alembic/versions/20260906_0035_add_social_signal_queue.py backend/app/models/theme.py backend/app/infra/db/models/social_signals.py backend/app/infra/db/models/__init__.py backend/app/models/__init__.py backend/app/services/theme_discovery_service.py backend/app/services/theme_pipeline_state_service.py backend/app/services/theme_taxonomy_service.py backend/app/services/social_source_admin_service.py backend/tests/integration/test_social_signal_migration.py backend/tests/unit/test_theme_social_source_exclusion.py backend/tests/unit/services/test_social_source_admin_service.py
git commit -m "feat: persist social runs without changing theme ranks"
```

---

### Task 5: Official X API Provider

**Files:**
- Create: `backend/app/infra/providers/official_x_social_provider.py`
- Create: `backend/app/infra/providers/__init__.py`
- Create: `backend/tests/unit/infra/test_official_x_social_provider.py`
- Modify: `backend/tests/fixtures/social/official_list_read.json`

**Interfaces:**
- Calls `GET /2/lists/{id}/tweets` with `tweet.fields=created_at,public_metrics,author_id,referenced_tweets,entities`, `expansions=author_id`, and `user.fields=username`.
- Maps rate-limit/auth/schema/network conditions to stable provider codes without fallback.

- [ ] **Step 1: Write RED request and normalization tests**

Use `httpx.MockTransport`. Assert pagination checkpoints, author expansion, UTC timestamps, nullable unavailable fields, canonical URLs, source membership, per-run and daily caps, and no request when bearer token is blank.

- [ ] **Step 2: Write RED failure-policy tests**

Assert 429 returns `rate_limited` plus reset time; 401/403 returns `reauthentication_required`; one transient network failure gets one delayed retry; invalid JSON returns `invalid_provider_json`; no case instantiates the xui adapter.

- [ ] **Step 3: Implement the adapter and run GREEN**

Run: `cd backend && ./venv/bin/pytest tests/unit/infra/test_official_x_social_provider.py -q`

- [ ] **Step 4: Commit**

```bash
git add backend/app/infra/providers backend/tests/unit/infra/test_official_x_social_provider.py backend/tests/fixtures/social/official_list_read.json
git commit -m "feat: add official X social provider"
```

---

### Task 6: Private xui CLI Adapter Boundary

**Files:**
- Create: `backend/app/infra/providers/xui_cli_social_provider.py`
- Create: `backend/tests/unit/infra/test_xui_cli_social_provider.py`
- Modify: `backend/tests/fixtures/social/xui_list_read.json`
- Modify: `backend/app/services/twitter_ingestion_providers.py`
- Modify: `backend/tests/unit/test_twitter_ingestion_providers.py`

**Interfaces:**
- Executes exact commands:

```python
["xui", "auth", "status", "--path", config_path, "--profile", profile, "--json"]
["xui", "read", "--path", config_path, "--profile", profile, "--limit", str(limit), "--json", "--new", "--checkpoint-mode", "auto", "--sources", f"list:{source_id}"]
```

- Consumes xui JSON keys `succeeded_sources`, `failed_sources`, `outcomes`, and `items`; maps item fields including `retweets -> reposts`.

- [ ] **Step 1: Write RED subprocess boundary tests**

Inject a runner and assert `shell` is false/absent, timeout is bounded, stdout only is parsed, stderr/path text is redacted, malformed/unknown top-level shapes fail closed, and missing executable returns `provider_unavailable`.

- [ ] **Step 2: Write RED health and challenge tests**

Assert auth status failure prevents `read`; challenge/login-wall/selector-drift output maps to `reauthentication_required` or `provider_error` and starts cooldown; one failed list cannot be marked complete.

- [ ] **Step 3: Replace the incompatible legacy private assumption**

Remove `import xui`/`xui.read_source` from `PrivateXUIFetcher`. Either delegate the legacy fetcher to `XuiCliSocialProvider` for supported list reads or mark the legacy private theme-source path unsupported with a clear configuration error. No public test imports `xui_reader`.

- [ ] **Step 4: Run GREEN and prove package independence**

Run: `cd backend && ./venv/bin/pytest tests/unit/infra/test_xui_cli_social_provider.py tests/unit/test_twitter_ingestion_providers.py -q`

Run: `cd backend && ./venv/bin/python -c "import app.main; print('public import ok')"`

- [ ] **Step 5: Commit**

```bash
git add backend/app/infra/providers/xui_cli_social_provider.py backend/app/services/twitter_ingestion_providers.py backend/tests/unit/infra/test_xui_cli_social_provider.py backend/tests/unit/test_twitter_ingestion_providers.py backend/tests/fixtures/social/xui_list_read.json
git commit -m "feat: integrate private reader through its CLI"
```

---

### Task 7: Ticker Resolution and Targeted Theme Extraction

**Files:**
- Create: `backend/app/services/social_ticker_resolver.py`
- Create: `backend/tests/unit/services/test_social_ticker_resolver.py`
- Modify: `backend/app/services/theme_extraction_service.py`
- Create: `backend/tests/unit/services/test_social_theme_extraction.py`

**Interfaces:**
- Consumes `MultiMarketTickerValidator`, `CJKAliasResolverService`, `SecurityMasterResolver`, and active `StockUniverse` rows.
- Adds `ThemeExtractionService.process_content_ids(content_ids, pipeline="technical")` so a Social run processes only its own new items.

- [ ] **Step 1: Write RED resolution matrix tests**

Cover explicit canonical/cashtag examples for US, HK, CN, JP, and TW; company-name home-primary selection; ADR/alternate related listings; two explicit listings producing two rows; ambiguity/unresolved retention; inactive security rejection; SPY/QQQ context; SMH/REMX thematic ETF eligibility.

- [ ] **Step 2: Implement deterministic precedence**

Extract explicit tokens first, then deterministic aliases. Never ask an LLM to select a listing. Pin `resolution_method`, `resolution_policy_version="social-resolution-v1"`, `reason`, and related listings.

- [ ] **Step 3: Add targeted extraction tests and method**

Assert only supplied ContentItem IDs are claimed, technical pipeline state is seeded idempotently, existing ThemeCluster identity is reused, and social-source ThemeMentions remain available as evidence while excluded from rankings.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/services/test_social_ticker_resolver.py tests/unit/services/test_social_theme_extraction.py -q`

```bash
git add backend/app/services/social_ticker_resolver.py backend/app/services/theme_extraction_service.py backend/tests/unit/services/test_social_ticker_resolver.py backend/tests/unit/services/test_social_theme_extraction.py
git commit -m "feat: resolve social mentions across supported markets"
```

---

### Task 8: Observation Writer and Atomic Publication Repository

**Files:**
- Create: `backend/app/infra/db/repositories/social_signal_writer.py`
- Create: `backend/app/infra/db/repositories/published_social_signal_reader.py`
- Create: `backend/tests/unit/repositories/test_social_signal_writer.py`
- Create: `backend/tests/integration/test_social_signal_publication.py`

**Interfaces:**
- Upserts canonical `ContentItem`, many-to-many memberships, newer metrics, and ticker mappings transactionally per source.
- Creates immutable runs/snapshots and swaps `latest_published` under one transaction only after quality validation.

- [ ] **Step 1: Write RED observation tests**

Assert cross-list dedup retains both memberships; newer observations update only observed metrics; missing new metrics do not erase old values; older observations do not overwrite; duplicate delivery is idempotent.

- [ ] **Step 2: Write RED publication tests**

Assert a run with at least two pinned enabled sources advances the pointer only when every pinned source is complete. A pending/disabled/archived source is absent from the run. Any enabled source that is truncated/partial/failed blocks publication; rollback preserves the previous pointer; reader returns rows from exactly one run; no pointer returns typed unavailable.

- [ ] **Step 3: Implement repositories and run GREEN**

Run: `cd backend && ./venv/bin/pytest tests/unit/repositories/test_social_signal_writer.py tests/integration/test_social_signal_publication.py -q`

- [ ] **Step 4: Commit**

```bash
git add backend/app/infra/db/repositories/social_signal_writer.py backend/app/infra/db/repositories/published_social_signal_reader.py backend/tests/unit/repositories/test_social_signal_writer.py backend/tests/integration/test_social_signal_publication.py
git commit -m "feat: publish immutable social signal snapshots"
```

---

### Task 9: Confirmation Reader and Refresh Orchestration

**Files:**
- Create: `backend/app/services/social_confirmation_reader.py`
- Create: `backend/tests/unit/services/test_social_confirmation_reader.py`
- Create: `backend/app/use_cases/social_signals/refresh.py`
- Create: `backend/tests/unit/use_cases/social_signals/test_refresh.py`
- Modify: `backend/app/wiring/use_case_factories.py`

**Interfaces:**
- Reads one published `FeatureRun` per Market, matching `StockFeatureDaily`, latest non-future `MarketExposure`, Market-specific liquidity eligibility, exact group rank cohort, and non-social linked-theme values.
- `RefreshSocialSignals.execute(origin, now)` owns the twelve-step run flow and returns a redacted `SocialRunResult`.

- [ ] **Step 1: Write RED coherent-confirmation tests**

Assert all rows for a Market use one feature run ID; stale feature/exposure dates are marked missing; setup score/readiness and `rs_rating_1m/3m` come from the feature row; group conversion uses the same Market/date cohort; theme confirmation ignores mention velocity/momentum and sources opted out of Theme rankings.

- [ ] **Step 2: Write RED refresh workflow tests**

Use in-memory fakes to assert order: load and pin every enabled source, require at least two, create run, read each pinned source, validate, write observations, extract/resolve, aggregate, load confirmation, score three windows, persist, quality-check, publish. Assert provider disabled does no I/O; one failed enabled source keeps the old pointer; retry uses the same run identity; source changes during a run do not alter its pinned set; identical pinned inputs serialize identically.

Assert every source without a completed checkpoint—including a newly enabled source added after launch—requests fourteen days and at most 1,000 posts; a later read for that source supplies its saved checkpoint, retains overlap for metric refresh, and caps at 200 posts. A source that reaches 1,000 before proving fourteen-day coverage is marked truncated and blocks publication.

- [ ] **Step 3: Implement confirmation reader and orchestrator**

Persist `feature_run_ids_by_market`, exposure dates, scoring formula, source checkpoints, normalization scope, all component inputs, and state reasons in the immutable run/snapshot records. Never query mutable latest data from the API path.

- [ ] **Step 4: Wire provider selection explicitly**

Factory logic must be a three-way match. `disabled` returns a disabled use case; `official` constructs only the official adapter; `xui` constructs only the CLI adapter. Unknown modes cannot reach runtime because settings validation rejects them.

- [ ] **Step 5: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/services/test_social_confirmation_reader.py tests/unit/use_cases/social_signals/test_refresh.py -q`

```bash
git add backend/app/services/social_confirmation_reader.py backend/app/use_cases/social_signals/refresh.py backend/app/wiring/use_case_factories.py backend/tests/unit/services/test_social_confirmation_reader.py backend/tests/unit/use_cases/social_signals/test_refresh.py
git commit -m "feat: build publishable social signal runs"
```

---

### Task 10: Celery Scheduling, Lease, and Operations Visibility

**Files:**
- Create: `backend/app/interfaces/tasks/social_signal_tasks.py`
- Create: `backend/app/use_cases/social_signals/validate_source.py`
- Create: `backend/tests/unit/test_social_signal_tasks.py`
- Create: `backend/tests/unit/use_cases/social_signals/test_validate_source.py`
- Modify: `backend/app/celery_app.py`
- Modify: `backend/app/services/task_registry_service.py`
- Modify: `backend/app/api/v1/operations.py`
- Modify: `backend/tests/unit/test_operations_endpoints.py`

**Interfaces:**
- Task name `app.interfaces.tasks.social_signal_tasks.refresh_social_signals` routes only to `social_ingestion`.
- Task name `app.interfaces.tasks.social_signal_tasks.validate_social_source` routes only to `social_ingestion` and calls `ValidateSocialSource.execute(source_id: int, actor: str) -> SourceTestOutcome`. Dispatch returns task ID with HTTP 202; the admin projection exposes `queued|running|completed` progress.
- Beat entries run at minute 17, hours `0,6,12,18` in configured Celery timezone only when capability is enabled.
- Redis provider-read lease key `social-signals:provider-read:lease` serializes refresh and source tests; manual cooldown key `social-signals:manual-refresh:cooldown` applies only to refresh dispatch.

- [ ] **Step 1: Write RED schedule and route tests**

Assert disabled mode has no beat entry; enabled mode has one six-hour entry; both task types route to the dedicated queue; stale refresh deliveries collapse; one transient network error retries once; auth/challenge/schema failures do not retry.

- [ ] **Step 2: Write RED lease and cooldown tests**

Assert concurrent deliveries create one run; a source test cannot overlap a refresh or another source test; lease ownership token prevents another worker from releasing it; manual dispatch inside one hour returns HTTP 429 with retry-after; scheduled runs are not blocked by manual cooldown but still honor the singleton provider-read lease.

- [ ] **Step 3: Write RED explicit source-test tests**

Assert validation reads only the selected pending/disabled source, requests at most five posts, stores no `ContentItem`, `SocialPostSource`, metrics, run, snapshot, or pointer, records queued/running/completed state plus only redacted counts/status/provider/time, audits request and completion, rejects pending/disabled enablement when the pass does not match the current provider, and does not invoke any provider automatically when a source is created. Already-enabled sources are validated by the next complete refresh after a deployment changes provider.

- [ ] **Step 4: Implement tasks and Operations projection**

Expose only run ID, provider label, health state, source coverage/counts, timestamps, formula version, cooldown time, and stable error codes. Redact provider error text through an allowlist; never return xui paths or raw stderr.

- [ ] **Step 5: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_social_signal_tasks.py tests/unit/use_cases/social_signals/test_validate_source.py tests/unit/test_operations_endpoints.py -q`

```bash
git add backend/app/interfaces/tasks/social_signal_tasks.py backend/app/use_cases/social_signals/validate_source.py backend/app/celery_app.py backend/app/services/task_registry_service.py backend/app/api/v1/operations.py backend/tests/unit/test_social_signal_tasks.py backend/tests/unit/use_cases/social_signals/test_validate_source.py backend/tests/unit/test_operations_endpoints.py
git commit -m "feat: schedule isolated social ingestion"
```

---

### Task 11: Authenticated Social API

**Files:**
- Create: `backend/app/schemas/social_signals.py`
- Create: `backend/app/use_cases/social_signals/queries.py`
- Create: `backend/app/api/v1/social_signals.py`
- Modify: `backend/app/api/v1/router.py`
- Create: `backend/tests/unit/test_social_signals_api.py`

**Interfaces:**
- `GET /v1/social-signals/summary?market=US`
- `GET /v1/social-signals/queue?market=US&window=7d&view=actionable&rank_mode=blended&page=1&page_size=50`
- `GET /v1/social-signals/candidates/{candidate_key}/evidence?window=7d`
- `GET /v1/social-signals/theme-pulse?market=US`
- Admin: `GET /v1/social-signals/admin/health`, `GET /admin/runs`, `GET /admin/sources`, `POST /admin/sources`, `PATCH /admin/sources/{source_id}`, `POST /admin/sources/{source_id}/test`, `POST /admin/sources/{source_id}/transition`, and `POST /admin/refresh`.

- [ ] **Step 1: Write RED schema/query tests**

Assert supported enum values, 1-100 page size, stable ordering, Market scoping, Actionable filtering, independent rank mode, top-three evidence, short excerpt truncation, canonical X URLs, related listings, unresolved rows, and published-run-only reads.

- [ ] **Step 2: Write RED authorization and redaction tests**

Assert read routes require server session; admin routes additionally require `X-Admin-Key`; disabled returns typed `supported=false`; no published run returns HTTP 200 typed unavailable; payload JSON never contains `token`, `cookie`, `storage_state`, `config_path`, raw stderr, or private package details.

Source administration tests assert a required trimmed name; numeric-ID and canonical-URL parsing; duplicate rejection; pending creation without provider traffic; editable name with immutable list ID; asynchronous test dispatch; enablement only after a current-provider pass; HTTP 409 on stale `expected_version`; HTTP 422 when a disable/archive would leave fewer than two enabled; terminal archival; and an immutable redacted audit record for every accepted action and test result.

- [ ] **Step 3: Implement queries, schemas, and router**

Include formula version, generated/published timestamps, stale/degraded state, enabled-source coverage, supported controls, component explanations, state reasons, and Market-aware canonical symbol. Evidence excerpts are plain text and at most 280 characters.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_social_signals_api.py -q`

```bash
git add backend/app/schemas/social_signals.py backend/app/use_cases/social_signals/queries.py backend/app/api/v1/social_signals.py backend/app/api/v1/router.py backend/tests/unit/test_social_signals_api.py
git commit -m "feat: expose published social signal API"
```

---

### Task 12: Social Queue and Evidence Drawer UI

**Files:**
- Create: `frontend/src/api/socialSignals.js`
- Create: `frontend/src/api/socialSignals.test.js`
- Create: `frontend/src/features/socialSignals/socialSignalPresentation.js`
- Create: `frontend/src/features/socialSignals/SocialSignalsTable.jsx`
- Create: `frontend/src/features/socialSignals/SocialEvidenceDrawer.jsx`
- Create: `frontend/src/features/socialSignals/SocialSignalsTab.jsx`
- Create: `frontend/src/features/socialSignals/SocialSignalsTab.test.jsx`
- Modify: `frontend/src/pages/MarketScanPage.jsx`
- Modify: `frontend/src/pages/MarketScanPage.test.jsx`

**Interfaces:**
- Adds left-nav item `Social Signals` under Daily only when `features.social_signals` is true.
- Reuses `ChartViewerModal`, `AddToWatchlistMenu`, current Market context, existing Theme detail navigation, and canonical symbols.

- [ ] **Step 1: Write RED API/client and presentation tests**

Assert exact query parameters, cache keys include Market/window/view/rank mode/filters, no admin key is sent on read calls, freshness/state labels are deterministic, and Pure Social sort requests do not alter displayed confirmation fields.

- [ ] **Step 2: Write RED queue interaction tests**

Cover disabled omission, loading, unavailable, empty, healthy, partial, stale, and reauthentication states. Exercise Actionable/All Signals, Blended/Pure Social, 1D/7D/14D, source/theme/instrument/state/ticker filters, pagination, and Market changes.

- [ ] **Step 3: Implement dense table and drawer**

Columns: Symbol, Market, Queue, Social, mentions, authors, Setup/readiness, RS, group rank, linked theme, state, freshness. Row click opens the evidence drawer before any chart. Drawer renders component math, state reasons, related listing information, and at most three plain-text posts with author/time/engagement/list badges/Open on X.

- [ ] **Step 4: Reuse application actions**

Add drawer buttons that open the existing chart/setup view, submit the currently visible resolved symbols to Scan using a URL-safe Market-aware symbol query, and open `AddToWatchlistMenu`. Do not duplicate those workflows.

- [ ] **Step 5: Run GREEN and commit**

Run: `cd frontend && npm run test:run -- src/api/socialSignals.test.js src/features/socialSignals/SocialSignalsTab.test.jsx src/pages/MarketScanPage.test.jsx`

Run: `cd frontend && npm run lint`

```bash
git add frontend/src/api/socialSignals.js frontend/src/api/socialSignals.test.js frontend/src/features/socialSignals frontend/src/pages/MarketScanPage.jsx frontend/src/pages/MarketScanPage.test.jsx
git commit -m "feat: add live social signal queue"
```

---

### Task 13: Daily Card, Theme Social Pulse, and Admin Health UI

**Files:**
- Create: `frontend/src/features/socialSignals/DailySocialSignalsCard.jsx`
- Create: `frontend/src/features/socialSignals/DailySocialSignalsCard.test.jsx`
- Create: `frontend/src/features/socialSignals/SocialSignalHealthPanel.jsx`
- Create: `frontend/src/features/socialSignals/SocialSignalHealthPanel.test.jsx`
- Modify: `frontend/src/components/MarketScan/DailyMarketSnapshotTab.jsx`
- Modify: `frontend/src/components/MarketScan/DailyMarketSnapshotTab.test.jsx`
- Modify: `frontend/src/features/themes/pages/ThemesPageContainer.jsx`
- Create: `frontend/src/features/themes/pages/ThemesPageContainer.test.jsx`
- Modify: `frontend/src/pages/OperationsPage.jsx`
- Modify: `frontend/src/pages/OperationsPage.test.jsx`

**Interfaces:**
- Daily card reads the top five for the selected Market and links to the Social tab.
- Theme Social Pulse is display-only and derives from the published Social run.
- Operations health uses the existing admin-key interaction pattern and owns manual refresh plus full source administration.

- [ ] **Step 1: Write RED Daily and Theme tests**

Assert top five, dominant themes, exposure posture, enabled-source coverage such as `3/3`, last success, stale/degraded badge, and navigation. Assert adding Social Pulse does not change Theme ordering, and it disappears when the capability is off.

- [ ] **Step 2: Write RED admin tests**

Assert ordinary Operations inventory remains visible and health prompts for the admin key. Exercise adding a required name plus numeric ID/URL, pending state without network traffic, an explicit Test List warning and asynchronous five-post-or-fewer test, enable after pass, rename without changing list ID, stale-edit conflict, dynamic source coverage, rejection when disable/archive would leave fewer than two enabled, archive confirmation/history, manual refresh 202/429, and the exact non-secret reauthentication action.

The panel displays name, canonical list ID/URL, pending/enabled/disabled/archived state, tested provider/time/outcome, last successful collection, and audit history. Hide archived sources by default with a `Show archived` switch. Do not offer arbitrary source management to ordinary signed-in users.

- [ ] **Step 3: Implement the three surfaces and run GREEN**

Run: `cd frontend && npm run test:run -- src/features/socialSignals/DailySocialSignalsCard.test.jsx src/features/socialSignals/SocialSignalHealthPanel.test.jsx src/components/MarketScan/DailyMarketSnapshotTab.test.jsx src/features/themes/pages/ThemesPageContainer.test.jsx src/pages/OperationsPage.test.jsx`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/features/socialSignals frontend/src/components/MarketScan/DailyMarketSnapshotTab.jsx frontend/src/components/MarketScan/DailyMarketSnapshotTab.test.jsx frontend/src/features/themes/pages/ThemesPageContainer.jsx frontend/src/features/themes/pages/ThemesPageContainer.test.jsx frontend/src/pages/OperationsPage.jsx frontend/src/pages/OperationsPage.test.jsx
git commit -m "feat: surface social pulse and provider health"
```

---

### Task 14: Prove Live-Only Static Isolation

**Files:**
- Modify: `backend/tests/unit/test_static_site_export_service.py`
- Modify: `backend/tests/unit/test_export_static_site_script.py`
- Modify: `frontend/src/static/pages/StaticHomePage.test.jsx`
- Modify: `frontend/src/static/pages/StaticThemesPage.test.jsx`
- Create: `frontend/src/static/socialIsolation.test.jsx`

**Interfaces:**
- Static manifests, files, routes, imports, and requests contain no Social Signal data or feature key.

- [ ] **Step 1: Add failing negative-contract tests**

Recursively inspect exported manifest/data keys and assert none match `social`, `x_post`, `tweet`, `source_metrics`, or `social_signal`. Mock the HTTP client in static app tests and assert no `/social-signals` request. Assert no static navigation label or Social Pulse column.

- [ ] **Step 2: Remove any accidental shared imports/exports**

Keep all new components under the live app import graph. Do not add a social section to `StaticSiteExportService`, static contracts, static schemas, or `frontend/src/static/App.jsx`.

- [ ] **Step 3: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py -q`

Run: `cd frontend && npm run test:run -- src/static/socialIsolation.test.jsx src/static/pages/StaticHomePage.test.jsx src/static/pages/StaticThemesPage.test.jsx`

```bash
git add backend/tests/unit/test_static_site_export_service.py backend/tests/unit/test_export_static_site_script.py frontend/src/static
git commit -m "test: enforce live-only social signal isolation"
```

---

### Task 15: Local Docker and Private Worker Image

**Files:**
- Modify: `backend/Dockerfile`
- Create: `docker-compose.social.yml`
- Create: `docker-compose.social-xui.yml`
- Modify: `docker-compose.prod.yml`
- Modify: `.dockerignore`
- Modify: `.env.docker.example`
- Create: `backend/tests/unit/test_social_worker_compose_contract.py`
- Create: `docs/deployment/social-signal-worker.md`

**Interfaces:**
- Public/official worker uses the normal backend runtime and `social_ingestion` queue.
- Private `social-xui` image target installs `xui-reader` from `git+ssh://git@github.com/xang1234/xui.git` with BuildKit SSH forwarding; the key is never an ARG/ENV/layer.
- Only the private worker mounts `${XUI_PROFILE_DIR}` at `/app/data/xui-reader`.

- [ ] **Step 1: Write RED Compose security tests**

Parse rendered Compose and Dockerfile text. Assert provider defaults disabled; only one worker consumes `social_ingestion`; xui profile is absent from backend/general/data workers; mount is writable only on the private worker; worker user is non-root; no token/key/session path is copied; public target has no xui install.

- [ ] **Step 2: Add public and private Docker targets**

Keep the existing public runtime as the default final stage. Add a named `social-xui` stage containing:

```dockerfile
ARG XUI_READER_REF
RUN test -n "$XUI_READER_REF"
RUN --mount=type=ssh \
    pip install --no-cache-dir \
    "xui-reader[cli] @ git+ssh://git@github.com/xang1234/xui.git@${XUI_READER_REF}"
```

Use the `docker/dockerfile:1.7` syntax line. Do not echo the ref, environment, pip config, or SSH diagnostics into artifacts.

- [ ] **Step 3: Add exact local operator flows**

Document:

```bash
xui config init --path "$XUI_PROFILE_DIR/config.toml"
xui profiles create automation --path "$XUI_PROFILE_DIR/config.toml"
xui auth login --path "$XUI_PROFILE_DIR/config.toml" --profile automation
xui auth status --path "$XUI_PROFILE_DIR/config.toml" --profile automation --json
docker login ghcr.io
docker compose -f docker-compose.yml -f docker-compose.social.yml -f docker-compose.social-xui.yml --profile social up -d
```

Also document GHCR-free development with `DOCKER_BUILDKIT=1 docker build --ssh default --target social-xui --build-arg XUI_READER_REF="$XUI_READER_REF" -t stock-screener-social-xui:dev -f backend/Dockerfile .`, after setting `XUI_READER_REF` to an exact private-repository commit SHA; explain accepted X account risk, read-only behavior, six-hour cadence, profile permissions, re-login, provider switching, and rollback to disabled.

- [ ] **Step 4: Render and test Compose**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_social_worker_compose_contract.py -q`

Run: `SERVER_AUTH_PASSWORD=test XUI_PROFILE_DIR=/tmp/xui-profile SOCIAL_WORKER_IMAGE=stock-screener-social-xui:dev docker compose -f docker-compose.yml -f docker-compose.social.yml -f docker-compose.social-xui.yml config --quiet`

- [ ] **Step 5: Commit**

```bash
git add backend/Dockerfile docker-compose.social.yml docker-compose.social-xui.yml docker-compose.prod.yml .dockerignore .env.docker.example backend/tests/unit/test_social_worker_compose_contract.py docs/deployment/social-signal-worker.md
git commit -m "feat: isolate social ingestion in Docker"
```

---

### Task 16: Trusted Private GHCR Workflow and Public CI Guardrails

**Files:**
- Create: `.github/workflows/private-social-worker.yml`
- Modify: `.github/workflows/ci.yml`
- Create: `scripts/verify_social_private_boundary.py`
- Create: `backend/tests/unit/test_social_private_boundary.py`
- Modify: `docs/deployment/social-signal-worker.md`

**Interfaces:**
- Private build runs only on `workflow_dispatch`, protected release tags, or pushes to the repository's protected main branch; never on `pull_request`.
- Repository secret `XUI_READER_DEPLOY_KEY` is a read-only deploy key for `xang1234/xui`; repository variable `XUI_READER_REF` is a pinned commit; GHCR push uses `GITHUB_TOKEN` with `packages: write`.
- Publishes `ghcr.io/xang1234/stock-screener-social-xui:sha-${GITHUB_SHA}` and the pushed release tag, both `linux/amd64,linux/arm64`; no sole `latest` deployment tag.

- [ ] **Step 1: Write RED workflow boundary tests**

Assert the private workflow has no PR trigger, uses least permissions, loads the deploy key only after trusted-event checks, passes SSH via BuildKit, never uses a token build arg, scans history/artifacts for forbidden names, and marks the package private in operator steps.

- [ ] **Step 2: Add the trusted workflow**

Use `webfactory/ssh-agent@v0.9.0`, `docker/setup-qemu-action@v3`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, and `docker/build-push-action@v6`. Build the `social-xui` target with `ssh: default`, run the worker's unit/import smoke test before push, and attach provenance/SBOM without source or secret artifacts.

- [ ] **Step 3: Add public CI guardrail**

Run `scripts/verify_social_private_boundary.py` in normal CI before backend tests. It must fail if public requirement/lock files, default Docker stages, public workflow steps, or application imports reference the private Git URL, distribution, or module.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_social_private_boundary.py -q`

Run: `python3 scripts/verify_social_private_boundary.py`

```bash
git add .github/workflows/private-social-worker.yml .github/workflows/ci.yml scripts/verify_social_private_boundary.py backend/tests/unit/test_social_private_boundary.py docs/deployment/social-signal-worker.md
git commit -m "ci: publish private social worker safely"
```

---

### Task 17: Fixture-Only End-to-End Flow and Full Verification

**Files:**
- Create: `backend/tests/integration/test_social_signal_end_to_end.py`
- Create: `frontend/tests/smoke/social-signals.spec.js`
- Create: `frontend/tests/smoke/socialSignalFixtures.js`
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/superpowers/specs/2026-09-06-social-signal-queue-design.md`

**Interfaces:**
- One fixture-backed flow ingests both source shapes, publishes, reads the API, and renders the signed-in UI without X access.
- Playwright covers user queue and admin health/manual-refresh behavior with intercepted fixtures only.

- [ ] **Step 1: Write the end-to-end backend test**

Seed the two system lists, create a named pending third list, prove creation performs no provider read, validate it with a five-post fixture, enable it, and seed active securities across all five Markets plus existing feature/exposure/group/theme inputs. Normalize official and xui fixture records through the same contract, publish only after all three enabled sources complete, and assert stable Blended/Pure Social ordering, binary cross-list attribution, state decisions, Theme isolation, and no secret-like fields.

- [ ] **Step 2: Write Playwright tests**

Cover Market selector, Actionable/All Signals, Pure Social, evidence drawer, top-three limit, Open on X, chart/setup action, visible-symbol Scan action, watchlist action, Daily top-five link, Theme Social Pulse, source add/name/test/enable/rename/disable/archive flows, two-enabled rejection, audit history, and admin refresh 202/429 states.

- [ ] **Step 3: Run focused integration and browser tests**

Run: `cd backend && ./venv/bin/pytest tests/integration/test_social_signal_end_to_end.py -q`

Run: `npm --prefix frontend run test:smoke -- --grep "Social Signals"`

- [ ] **Step 4: Run full verification**

Run: `cd backend && ./venv/bin/pytest`

Run: `cd frontend && npm run test:run`

Run: `cd frontend && npm run lint`

Run: `cd frontend && npm run build`

Run: `npm --prefix frontend run test:smoke`

Run: `python3 scripts/verify_social_private_boundary.py`

- [ ] **Step 5: Perform clean-room deployment checks**

Build the public backend without SSH/GHCR/X credentials and verify it starts with `SOCIAL_INGEST_PROVIDER=disabled`. Render official Compose without xui inputs. On a trusted machine only, build/pull the private arm64 image, mount the dedicated automation profile, execute one dry run with publication disabled, and compare counts/mappings/scores against the approved report and SVG.

- [ ] **Step 6: Record rollout evidence and acceptance**

Append a short implementation-evidence section to the design spec with commands, fixture hashes, image digest, all-enabled-source coverage, source-lifecycle audit checks, unresolved mappings, and confirmation that existing Theme rankings are unchanged. Do not include post bodies, credentials, session paths, or private package internals.

- [ ] **Step 7: Commit**

```bash
git add backend/tests/integration/test_social_signal_end_to_end.py frontend/tests/smoke/social-signals.spec.js frontend/tests/smoke/socialSignalFixtures.js .github/workflows/ci.yml docs/superpowers/specs/2026-09-06-social-signal-queue-design.md
git commit -m "test: verify social signal queue end to end"
```

---

## Final Review Checklist

- [ ] Compare every implemented behavior with the approved design and both supplied research artifacts.
- [ ] Confirm no Social Signal task is scheduled when disabled and no provider fallback exists.
- [ ] Confirm both seed source IDs, administrator-added sources, and all five Markets are covered by tests.
- [ ] Confirm required names, pending creation, explicit five-post testing, current-provider validation, two-enabled minimum, optimistic edits, terminal archival, and redacted immutable audits.
- [ ] Confirm every enabled source is pinned at run start and must succeed, while source changes during a run cannot alter that run.
- [ ] Confirm formula weights, percentile tie behavior, recency half-life, winsorization, author cap, and stable tie-breaks are pinned by tests.
- [ ] Confirm partial/truncated/failed runs cannot advance the pointer and stale last-known-good data remains readable.
- [ ] Confirm existing Theme rank fixtures are byte-equivalent before and after social-source insertion.
- [ ] Confirm API authorization/redaction and admin cooldown behavior.
- [ ] Confirm static exports and static UI contain no social data or request.
- [ ] Confirm public install/build/test works with no private dependency or credential.
- [ ] Confirm private image history/config/SBOM contain no SSH key, token, storage state, or checkout artifact.
- [ ] Confirm local Apple Silicon and amd64 image manifests and the dedicated non-root profile mount.
- [ ] Use superpowers:requesting-code-review, resolve findings, then use superpowers:verification-before-completion before claiming completion.

## Rollout Sequence

1. Merge Tasks 1-11 with the capability disabled.
2. Deploy schema and public application; verify existing Themes and static artifacts are unchanged.
3. Merge live UI Tasks 12-14, still disabled.
4. Publish the private worker from Tasks 15-16 and validate its image digest and profile isolation.
5. Run Task 17's fourteen-day dry run with publication disabled; inspect all-enabled-source completeness, source administration/audits, unresolved names, cross-list deduplication, and rankings.
6. Enable publication, then `SOCIAL_SIGNALS_ENABLED=true` for authenticated users.
7. Roll back by setting `SOCIAL_SIGNALS_ENABLED=false` and `SOCIAL_INGEST_PROVIDER=disabled`; retain data and the last pointer for audit.
