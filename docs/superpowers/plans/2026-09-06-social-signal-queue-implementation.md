# Social Signal Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-only, shared Social Signal Queue that combines administrator-managed X lists, resolves US/HK/CN/JP/TW securities, ranks them with deterministic Blended and Pure Social modes, and works with either the official X API or the private `xui-reader` CLI without making the private package a public-repository dependency.

**Architecture:** Keep all domain models, source administration, persistence, scoring, APIs, UI, fixtures, and the official provider in the public repository. Put provider reads behind a provider-neutral port and run them on a dedicated `social_ingestion` Celery queue. The private image extends the public backend image, installs `xui-reader` through a credential-safe trusted build, and mounts a dedicated authenticated profile only into that worker. Publish immutable all-enabled-source snapshots through an atomic pointer; live reads never assemble mixed or partial data.

**Tech Stack:** Python 3, FastAPI, Pydantic, SQLAlchemy, Alembic, PostgreSQL/SQLite, Celery, Redis, httpx, React 18, TanStack Query, MUI, Vitest, Playwright, Docker BuildKit, Docker Compose, GitHub Actions, private GHCR.

**Spec:** `docs/superpowers/specs/2026-09-06-social-signal-queue-design.md` (original approval commit `8a57ccec`; consolidated review decisions dated 2026-09-07).

## Global Constraints

**Review status:** Consolidated against the nine approved review resolutions on 2026-09-07. This is a plan, not implementation evidence. Execute Tasks 7A (budget/backlog), 7B (shared Theme projection), and 7C (Market confirmation) between Tasks 7 and 8.

- Begin implementation on a dedicated `feat/social-signal-queue` worktree whose history contains design commit `8a57ccec`; do not implement on the current release branch.
- Social LLM allowance is US$2/day shared installation-wide, resetting at midnight `Asia/Singapore` (configurable IANA timezone). No rollover. Atomic estimated-cost reservations cover every actual request/retry; reconcile actual usage in its original dispatch-day bucket even across midnight. Host timezone/restarts do not affect allowance. This is an application guardrail, not a guaranteed provider billing cap.
- `SOCIAL_SIGNALS_MODE=off|validation|live`, default off. Validation reads/analyzes under the same budgets but only stages proposals and admin output. Live publication rechecks mode/version before visible Theme/basket/pointer changes. Only live enables ordinary Social routes. Explicit Test List is a separately authorized diagnostic when its provider is configured.
- `SOCIAL_INGEST_PROVIDER` is exactly `disabled|official|xui`, defaults to `disabled`, and never falls back automatically.
- All Signals separates ranked Market-scoped stocks/thematic ETFs, unranked context, and unranked Needs resolution. Market-unknown items remain explicitly global and accessible from every Market view; exclude them from candidate cohorts/top-five/ranked totals.
- Daily freshness uses the latest MIC session whose close plus a configurable two-hour grace has passed. Reuse exchange holidays/early-close/DST calendars and existing indicator validity/history rules. Unknown freshness or missing required inputs means non-Actionable. Social staleness is separately seven hours.
- Missing confirmation produces null confirmation/blended scores, not zero; Blended sorts scored rows first and null rows by Social Score. Partial weights renormalize only inside their 60/40 portions. Missing required checks preclude Actionable in every rank mode.
- Theme market measurements require at least three distinct accepted companies and 70% coverage within the selected Market, counting alternate listings once. Expose measured/accepted counts and Market benchmark. Missing per-indicator history yields a missing component, not synthetic zero.
- Social-led candidate promotion requires three accepted companies and qualifying discussion on three distinct UTC dates in fourteen days. Association acceptance requires two qualifying posts from distinct authors in fourteen days, each supporting the same resolved company's business connection. Copied claims/reposts/list duplicates are not corroboration. Admin accept/reject requires a reason and wins over automatic reassessment. No price, likes, or non-social-coverage threshold.
- Process saved extraction oldest-publication-first within fourteen days. Older work remains stored as outside_window and can be admin-requested under the same budget. Budget deferral resumes after reset without re-reading X. Skips never count as successful extraction, and old posts do not re-enter current windows.
- Public dependency files, public images, public CI, and fork PRs must not fetch, import, cache, or require `xui-reader`.
- The private adapter invokes the exact auth/read commands documented in Task 6 as argument arrays with `shell=False`; it never imports private Python modules or invokes interactive login.
- Seed source IDs `1522014550211457024` and `1986290701492232693` as enabled, named system sources. Administrators may add more installation-wide lists without redeployment.
- At least two sources must remain enabled. Every pinned source must return a valid successful bounded read and all required run inputs must finish processing; limited history is labelled separately. Serialized DB mutations protect the enabled count.
- New sources require a non-blank local display name, start pending, and cannot be enabled until an explicit test reads at most five posts successfully through the currently selected provider.
- List IDs are immutable; display names are editable; removal archives rather than deletes; every source mutation and test produces a redacted immutable audit event.
- Scheduled reads run every six hours; manual refresh is admin-only and limited to one accepted dispatch per hour.
- Attempt fourteen-day initial history with a hard 1,000-post cap per list. A successful capped read may publish warming-up/limited-history results once every enabled source participates successfully and collected run inputs finish analysis. Failed reads and unfinished processing still block publication. Never infer exhaustive history from response length, oldest timestamp, or fourteen elapsed days; retain per-source/window coverage reasons and known gaps. Unsupported acceleration remains missing, not zero.
- Social windows are rolling UTC `1D|7D|14D`; supported Markets are exactly `US|HK|CN|JP|TW`.
- Formula version `social-signal-v1` uses `queue_score = 0.60 * social_score + 0.40 * confirmation_score`. Pure Social changes ordering only.
- Social weights are acceleration 20, authors 15, engagement 15, recency 5, cross-list 5. Confirmation weights are setup 20, RS 10, group 5, theme 5. Missing inputs are omitted from the available-weight denominator and retained as missing in explanation payloads.
- Percentiles use deterministic mid-rank: `100 * (less_count + 0.5 * equal_count) / cohort_count`; a singleton cohort yields 50. Use per-Market cohorts when at least 20 candidates exist, otherwise one global supported-Market cohort.
- Engagement uses the approved log formula and a deterministic linear-interpolation 95th-percentile winsorization threshold. Likes, reposts, and replies must all be observed; optional quote/bookmark/view fields contribute only when observed.
- A post contributes once per ticker; reposts, empty-thesis quotes, and repeated canonical URLs do not add independent weight; one author contributes at most three posts per ticker per rolling 24 hours.
- Social can create shared theme candidates and propose company associations from an empty database. Accepted associations can change measured baskets and rankings; proposed ones cannot. Social Pulse is read from the published Social run and does not feed legacy attention scores or technical confirmation. Keep social strength, market strength, and legacy scoring distinguishable.
- Broad Market ETFs are `context`; thematic ETFs stay eligible with an ETF badge. Unresolved text never receives fabricated Market or technical fields.
- `actionable` requires an active supported security, fresh feature and Market inputs, the existing Market liquidity gate, `se_setup_ready=true`, and Market exposure at least 50.
- Failed-source, missing-participation or unfinished-processing runs remain auditable but cannot replace the pointer. Successful limited-history runs may publish with coverage labels. A published run older than seven hours is shown as stale but remains readable.
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
- `backend/app/services/social_confirmation_reader.py`: coherent feature, exposure, liquidity, group and Market-scoped Theme inputs.
- `backend/app/services/social_extraction_service.py`: structured source-grounded extraction without Theme side effects.
- `backend/app/services/social_llm_budget_service.py`: per-day reservations and actual-usage reconciliation.
- `backend/app/use_cases/social_signals/process_backlog.py`: durable, bounded extraction resumption.
- `backend/app/services/social_theme_projection_service.py`: shared identities, association decisions and social-led lifecycle.
- `backend/app/services/theme_evidence_eligibility_service.py`: content/pipeline/channel provenance for legacy attention and static projection.
- `backend/app/services/social_theme_market_service.py`: accepted Market baskets, benchmark/coverage and price measurements.
- `backend/app/infra/db/models/social_analysis.py`: extraction work, run/work membership, budget/attempt ledger and association/audit records.
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
- Produces settings `social_signals_mode`, `social_ingest_provider`, schedule hours, freshness hours, caps, xui config/profile, and official API limits. Source membership is database-managed rather than environment-configurable.
- Produces runtime flag `features.social_signals`; it is true only when `SOCIAL_SIGNALS_MODE=live` and provider is not disabled. Validation remains administrator-only; off schedules no collection/analysis.

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

def test_live_mode_does_not_override_disabled_provider():
    settings = Settings(
        _env_file=None,
        social_signals_mode="live",
        social_ingest_provider="disabled",
    )
    assert settings.capability_flags()["social_signals"] is False
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `cd backend && ./venv/bin/pytest tests/unit/domain/social_signals/test_settings.py tests/unit/test_app_runtime_endpoints.py -q`

Expected: failures for missing social settings and capability.

- [ ] **Step 3: Add exact settings and validators**

Add defaults (import `Decimal` from `decimal`; validate timezone with `zoneinfo.ZoneInfo`):

```python
social_signals_mode: str = "off"  # validated enum: off|validation|live
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
social_llm_daily_budget_usd: Decimal = Decimal("2.00")
social_llm_budget_timezone: str = "Asia/Singapore"
social_market_close_grace_minutes: int = 120
```

Keep legacy `x_ingest_provider` for the existing generic theme ingestion path; do not silently repurpose it. Validate the new provider independently. Task 4 owns the administrator-managed source registry.

Validate `social_signals_mode` independently and test all mode/provider combinations. Off has no automatic collection/analysis; validation and live schedule six-hour runs only with a configured provider. Validation stores reusable observations/extraction and admin-only results, never advancing the public pointer or mutating the live theme catalog, constituents, taxonomy, lifecycle, or rankings. Stage proposals instead of calling legacy extraction side effects unguarded. A new live run may reuse saved analysis but must re-evaluate current source set, windows, evidence/coverage, and market data before publication. Pin mode/version and recheck it before user-visible writes to handle in-flight mode changes safely. Explicit admin Test List is a separate diagnostic with a configured provider; provider disabled prevents every read. Add integration tests proving ordinary routes cannot retrieve validation data and validation leaves live Theme fixtures/pointer unchanged. Include actual read/LLM cost warnings and the shared $2/day allowance in admin validation UI.

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
- `SocialSourceOutcome` separately records read success/failure, processing completion, historical coverage/limit reasons, observed time bounds/known gaps, application-owned committed progress, counts, and stable error code. A history-limit flag is not a read failure or evidence of exhaustive coverage.
- `SourceTestOutcome` records only provider, status, sample count `0..5`, tested time, and a stable redacted reason code; it contains no posts or raw provider payload.
- `SocialSourceView` is the provider-neutral administrator projection containing source ID, name, canonical URL/list ID, lifecycle/provenance, test summary, collection freshness, optimistic version, and timestamps. `SocialSourceAuditView` contains action, actor, timestamp, and redacted before/after lifecycle metadata.
- `SocialProvider.read_source(request: SocialReadRequest) -> SocialSourceBatch` is the sole ingestion dependency.
- Define `ComponentScore(value: Decimal | None, available_weight: Decimal, total_weight: Decimal, reasons: tuple[str, ...])` and `SignalStateDecision(state: str, reasons: tuple[str, ...])` in records.py.
- `SocialReadRequest` pins source/list ID, initial/incremental/test intent, UTC observed time, limit, target lower publication bound, and application progress. `SocialSourceBatch` contains request identity, tuple of SocialPostRecord, and SocialSourceOutcome.
- `SocialSourceOutcome` uses `read_status=success|failed`, `processing_status=pending|complete|failed`, `history_status=warming_up|limited|observed_window`, tuple coverage reason codes/known-gap intervals, observed oldest/newest timestamps, received count, progress and stable nullable error code. Completion/coverage are independent.
- `SocialRunResult` carries run_id, mode, processing status, published boolean, coverage summary and stable reason codes. `SocialSnapshotRecord` has nullable social/confirmation/queue scores, candidate identity/state, pinned inputs, coverage, and deterministic tie-break fields. If all components in any portion are missing, that portion is null and the blended score is null; null Social Scores sort after valid Social Scores.
- Ports in ports.py must define `SocialProvider.read_source`, `SocialWriter.persist_observations(batch)`, `SocialWriter.publish(run_id, expected_mode_version)`, `ConfirmationReader.read(market, symbols, now)`, `PublishedReader.queue(market, window_days, view, rank_mode, page, page_size)`, `ProviderReadLease.acquire(owner, ttl_seconds)`/`release(owner)`, and `SocialDispatcher.refresh(origin)`/`test_source(source_id, actor)`. Return records above or typed page/result dataclasses; do not pass ORM objects across ports.

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

Define `SocialReadRequest`, `SocialPostRecord`, `SocialSourceBatch`, `SocialSourceOutcome`, `SourceTestOutcome`, `SocialSourceView`, `SocialSourceAuditView`, `TickerResolution`, `SocialEvidenceInput`, `ConfirmationInput`, `SocialScoreResult`, and `SocialSnapshotRecord`. Reject blank IDs/text/URLs, naive timestamps, future timestamps beyond five minutes, negative metrics, sample counts outside `0..5`, unexpected raw provider payload fields, and unsupported providers with stable codes.

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

Use `Decimal` internally for weighted aggregation, round stored component and total scores to four decimal places, and keep input coverage and exclusion reasons alongside every component. Pure Social ordering is `(social_score nulls last desc, latest_mention desc, canonical_symbol asc, candidate_key asc)`; Blended adds `(queue_score nulls last desc)` first. All-missing portions return null without dividing by zero. Context/unresolved evidence is outside these cohorts. Import pytest in the formula test module; the engagement example equals log1p(26).

- [ ] **Step 5: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/domain/social_signals/test_scoring.py tests/unit/domain/social_signals/test_states.py -q`

```bash
git add backend/app/domain/social_signals backend/tests/unit/domain/social_signals
git commit -m "feat: implement deterministic social scoring"
```

---

### Task 4: Source Registry, Provenance, and Collection Ownership

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
- Modify: `backend/app/services/theme_extraction_service.py`
- Create: `backend/app/services/theme_evidence_eligibility_service.py`
- Modify: `backend/app/services/content_ingestion_service.py`
- Modify: `backend/app/tasks/theme_discovery_tasks.py`
- Create: `backend/tests/unit/test_social_collection_ownership.py`
- Create: `backend/app/services/social_source_admin_service.py`
- Create: `backend/tests/unit/services/test_social_source_admin_service.py`

**Interfaces:**
- Uses SocialSourceConfiguration existence as exclusive collection ownership.
- Adds `content_pipeline_eligibility` keyed by `(content_item_id, pipeline, channel)`, channel `legacy|social`, with originating source and observation timestamp. Backfill existing legacy items; Social inserts grant only social eligibility. Independently observed legacy content may grant legacy eligibility once, regardless of ContentItem.source_id insertion order.
- Adds singleton `social_source_registry` row `id=1`, version and timestamps for transaction locking.
- Produces ORM models `SocialSourceConfiguration` and `SocialSourceAuditEvent` for database-managed list lifecycle and immutable administrator history.
- Adds `social_source_registry`, `content_pipeline_eligibility`, `social_source_configurations`, `social_source_audit_events`, `social_post_sources`, `social_content_metrics`, `social_post_tickers`, `social_signal_runs`, `social_signal_snapshots`, and `social_signal_run_pointers`. Analysis-specific tables are added in Tasks 7A/7B.

- [ ] **Step 1: Write RED migration shape tests**

Assert revision `20260906_0035` revises `20260904_0034`; all ten named tables, foreign keys, unique constraints, pointer key, nullable-score/state/lifecycle checks, and provenance backfill exist after upgrade and disappear after downgrade.

- [ ] **Step 2: Write RED provenance regression**

In `test_theme_social_source_exclusion.py`, create the same canonical post via Social then legacy ingestion, and in reverse order. Assert channel eligibility is identical, legacy attention counts it once only after independent legacy ingestion, Social metrics retain memberships, and Social-only ingestion never increases legacy active-ingestion-day counts. Do not assert all live Theme ranks stay unchanged after accepted Social discovery; Task 7B verifies legitimate basket changes and validation isolation.

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
social_source_registry: PRIMARY KEY(id), CHECK(id=1), version, mode, provider, updated_at
content_pipeline_eligibility: PRIMARY KEY(content_item_id, pipeline, channel), originating_source_id FK, observed_at
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
  scope TEXT NOT NULL CHECK source|runtime
  registry_id FK -> social_source_registry.id ON DELETE RESTRICT
  content_source_id NULL FK -> content_sources.id ON DELETE RESTRICT
  CHECK source scope requires content_source_id; runtime scope requires it NULL
  action TEXT NOT NULL CHECK created|renamed|test_requested|test_completed|enabled|disabled|archived|runtime_changed
  actor TEXT NOT NULL
  before_json JSON NULL
  after_json JSON NOT NULL
  created_at TIMESTAMPTZ NOT NULL
```

`candidate_key` is `MARKET:canonical_symbol` for resolved rows and `"unresolved:" + sha256(f"{raw_token}|{content_item_id}").hexdigest()` otherwise. Store nullable metric columns, pinned JSON explanations, `stock_universe_id`, canonical symbol/Market/MIC/local code, resolution policy version, run source outcomes/application progress, feature run IDs per Market, exposure dates, and stable tie-break fields.

- [ ] **Step 5: Implement the administrator-managed source lifecycle**

`SocialSourceAdminService.ensure_seed_sources()` creates these named, enabled system sources idempotently:

```python
SEED_SOCIAL_SOURCES = (
    ("1522014550211457024", "Minervini Research List", "https://x.com/i/lists/1522014550211457024"),
    ("1986290701492232693", "Asia-Pacific Growth List", "https://x.com/i/lists/1986290701492232693"),
)
```

Set `source_type="twitter"`, `pipelines=["technical"]`, `fetch_interval_minutes=360`, lifecycle provenance `system_seed`, and social-only content eligibility. Preserve administrator edits on subsequent calls.

Add methods with exact contracts:

All source-registry mutations must acquire one shared database transaction lock before validating versions, counting enabled sources, applying changes, and appending audits. Use a migration-seeded singleton registry row for PostgreSQL row locking; the table is included in the inventory above. For supported SQLite operation, use an equivalent serialized write transaction rather than relying on ignored row-lock syntax. Keep these transactions short and never perform provider/LLM calls while holding the lock. Runtime mode/provider are installation-wide: initialize from validated deployment settings through an explicit administrative apply step, increment this row's version on changes, and require all workers to read it before dispatch/publication. A worker startup must not silently overwrite newer shared settings with stale environment values. Per-source optimistic versions remain necessary but are insufficient alone.

Add a real concurrent-transaction regression with three enabled lists and simultaneous disables of two different lists: exactly one succeeds, the other fails the minimum-two validation, two lists remain enabled, and only accepted mutations append successful-change audit events. Also test disable/archive races, rollback releasing the lock, and stale-version checks under serialization. Mocks or sequential calls alone do not prove this invariant.

```python
create_source(name: str, list_ref: str, actor: str) -> SocialSourceView
rename_source(source_id: int, name: str, expected_version: int, actor: str) -> SocialSourceView
record_test_result(source_id: int, provider: str, outcome: SourceTestOutcome, actor: str) -> SocialSourceView
transition_source(source_id: int, target: Literal["enabled", "disabled", "archived"], expected_version: int, actor: str) -> SocialSourceView
list_sources(include_archived: bool = False) -> tuple[SocialSourceView, ...]
audit_events(source_id: int) -> tuple[SocialSourceAuditView, ...]
```

Accept only a 1-32-digit numeric ID or `https://x.com/i/lists/{id}`. Creation requires a trimmed 1-100-character name, rejects duplicate list IDs, stores the canonical URL, starts `pending`, and sets `ContentSource.is_active=false`. Except for the two `system_seed` rows, enabling requires a passed test matching the current provider. State transitions keep `ContentSource.is_active` true only for `enabled`. Disabling or archiving is rejected when it would leave fewer than two enabled sources. Renaming changes only `ContentSource.name`; the list ID and canonical URL are immutable. Archival is terminal in v1 and never deletes evidence. Every successful mutation, test request, and test completion appends a redacted `SocialSourceAuditEvent`; version mismatches raise a typed optimistic-concurrency error.

- [ ] **Step 6: Enforce collection ownership and legacy evidence eligibility**

Implement collection ownership using the existence of `SocialSourceConfiguration`, not `is_active` or the current provider. Exclude social-owned sources from every legacy Theme polling/bulk/manual selection, including `poll_due_sources` and `fetch_all_active_sources`. Guard `fetch_source` and `fetch_source_by_id` before any provider I/O so explicit IDs and already-queued legacy tasks cannot bypass the exclusion. Only the dedicated social worker may perform refresh/test reads through the selected Social provider. Preserve generic non-social Theme ingestion and reuse extraction only after social collection.

Add regression tests in `test_social_collection_ownership.py` covering legacy scheduled polling, manual ingest-all, direct source reads, and already-queued tasks for every social lifecycle state. Assert zero legacy provider calls for social-owned sources, including when Social is disabled while the legacy provider is official. Assert ordinary Theme sources still collect, dedicated Social refresh/test reads remain supported, and Social disabled makes no provider calls.

Preserve test request identity/version/provider when recording asynchronous source-test results. A late result for a renamed/disabled/archived or superseded-test version cannot enable or resurrect a source. Test duplicate/out-of-order completions under the same registry transaction lock.

Use an EXISTS check on `content_pipeline_eligibility(channel='legacy')` in legacy attention, source-diversity, active-ingestion-day, and static evidence queries. Social-only items cannot be selected by legacy extraction/reprocessing jobs: add selection and per-item guards to `theme_extraction_service.py`. Do not use ContentItem.source_id as channel authority. Keep Social Theme evidence accessible in live detail projections. Live accepted-basket price changes are governed by Tasks 7B/7C, not this legacy attention filter.

- [ ] **Step 7: Run migration and regressions GREEN**

Run: `cd backend && ./venv/bin/pytest tests/integration/test_social_signal_migration.py tests/unit/test_theme_social_source_exclusion.py tests/unit/test_social_collection_ownership.py tests/unit/services/test_social_source_admin_service.py -q`

- [ ] **Step 8: Commit**

```bash
git add backend/alembic/versions/20260906_0035_add_social_signal_queue.py backend/app/models/theme.py backend/app/infra/db/models/social_signals.py backend/app/infra/db/models/__init__.py backend/app/models/__init__.py backend/app/services/theme_discovery_service.py backend/app/services/theme_pipeline_state_service.py backend/app/services/theme_taxonomy_service.py backend/app/services/social_source_admin_service.py backend/tests/integration/test_social_signal_migration.py backend/tests/unit/test_theme_social_source_exclusion.py backend/tests/unit/services/test_social_source_admin_service.py
git add backend/app/services/content_ingestion_service.py backend/app/tasks/theme_discovery_tasks.py backend/app/services/theme_extraction_service.py backend/app/services/theme_evidence_eligibility_service.py backend/tests/unit/test_social_collection_ownership.py
git commit -m "feat: persist social sources with collection and evidence ownership"
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

Use `httpx.MockTransport`. Assert resumable pagination progress committed with observations, author expansion, UTC timestamps, nullable unavailable fields, canonical URLs, source membership, per-run and daily caps, and no request when bearer token is blank.

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
["xui", "read", "--path", config_path, "--profile", profile, "--limit", str(limit), "--json", "--sources", f"list:{source_id}"]
```

- Consumes xui JSON keys `succeeded_sources`, `failed_sources`, `outcomes`, and `items`; maps item fields including `retweets -> reposts`.
- Never use `--new` or reader-managed automatic checkpoints. Use bounded overlapping reads; application collection progress advances only with durable observation commits. Retried items update engagement without duplicate evidence, and unchanged text reuses extraction. Test List must not advance production progress; reader-local storage is not the application's source of truth. Keep the six-hour cadence and bounded read limits.

- [ ] **Step 1: Write RED subprocess boundary tests**

Inject a runner and assert `shell` is false/absent, timeout is bounded, stdout only is parsed, stderr/path text is redacted, malformed/unknown top-level shapes fail closed, and missing executable returns `provider_unavailable`.

Assert the command omits new-only/checkpoint flags. Add application-boundary regressions for crash before commit followed by identical retry, repeated posts with updated engagement and unchanged extraction count, atomic progress/observation commits, and a five-post Test List followed by production collection retaining those posts. Do not claim gap-free history merely because a bounded retry succeeds; coverage assessment is a separate requirement.

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

### Task 7: Source-Grounded Extraction and Deterministic Resolution

**Files:**

- Create: `backend/app/services/social_ticker_resolver.py`
- Create: `backend/app/services/social_extraction_service.py`
- Modify: `backend/app/services/theme_extraction_service.py`
- Modify: `backend/app/domain/social_signals/records.py`
- Create: `backend/tests/unit/services/test_social_ticker_resolver.py`
- Create: `backend/tests/unit/services/test_social_theme_extraction.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ExtractionClaim:
    post_id: str
    theme_key: str
    raw_theme: str
    company_token: str
    relationship: str
    excerpt: str
    support: Literal["supported", "uncertain", "unsupported"]
    duplicate_of_post_ids: tuple[str, ...]

@dataclass(frozen=True)
class ExtractionResult:
    input_hash: str
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    claims: tuple[ExtractionClaim, ...]
    usage_input_tokens: int | None
    usage_output_tokens: int | None

class SocialExtractionService:
    async def extract(self, posts: tuple[SocialPostRecord, ...]) -> ExtractionResult: ...

class SocialTickerResolver:
    def resolve(self, raw_token: str, explicit_market: str | None) -> TickerResolution: ...
```

The signatures above specify contracts, not permission to bypass Task 7A's budget wrapper. Extract typed content only; no cluster/alias/constituent/lifecycle writes. Reuse the sanctioned LLMService and configured extraction model. Extract a read-only theme-match helper from the existing service so exact/alias matching can be reused without its current reactivation, alias updates, constituent additions, or taxonomy side effects.

- [ ] **Step 1: Write RED resolution matrix tests**

Cover explicit US/HK/CN/JP/TW cashtags and canonical symbols, CJK/home-primary aliases, ADR-related listings, two explicit listings, unresolved/inactive securities, SPY/QQQ context, and SMH/REMX eligibility. Assert the LLM never chooses an ambiguous listing. A verified company identity groups ADRs/alternate listings; absent a deterministic company linkage, do not invent one or overcount it toward theme gates.

- [ ] **Step 2: Write RED structured-output tests**

```python
def test_relationship_needs_source_support(extraction_parser):
    post = {"post_id": "101", "text": "$AAA rose alongside cooling stocks"}
    result = extraction_parser.parse(
        post, {"theme_key": "data_center_cooling", "company_token": "AAA",
               "relationship": "supplies cooling equipment",
               "excerpt": "supplies cooling equipment", "support": "supported"}
    )
    assert result.error_code == "excerpt_not_in_source"
```

Define the `extraction_parser` fixture as the real parser built by this task, not a stub. Test a genuine business-connection excerpt, bare co-occurrence, unsupported facts, malformed JSON, successful empty result, prompt injection, copied claims, and uncertain paraphrases. Assert zero live Theme mutations for both extraction success and failure.

- [ ] **Step 3: Implement structured extraction and resolution**

Version prompt/schema as `social-extraction-v1`. Require excerpts to be substrings of the supplied post; support flags are semantic model judgments, not verified facts. Batch only bounded supplied posts with per-post attribution; invalid/missing outputs cannot make an entire batch successful. Store original-language evidence. Freeze actual model/provider metadata; a changed prompt/model produces a new result, not an overwrite of a published one. Score code reads persisted judgments only.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/services/test_social_ticker_resolver.py tests/unit/services/test_social_theme_extraction.py -q`

Commit only the Files listed above with message `feat: extract source-grounded social relationships without live mutations`.

---

### Task 7A: Durable Extraction Backlog and Daily Budget

**Files:**

- Create: `backend/alembic/versions/20260907_0036_add_social_analysis_work.py` (revises Task 4's `20260906_0035`)
- Create: `backend/app/infra/db/models/social_analysis.py`
- Modify: `backend/app/infra/db/models/__init__.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/domain/social_signals/records.py`
- Create: `backend/app/services/social_llm_budget_service.py`
- Create: `backend/app/use_cases/social_signals/process_backlog.py`
- Modify: `backend/app/services/llm/llm_service.py` only if required to ensure one metered dispatch
- Create: `backend/tests/unit/services/test_social_llm_budget.py`
- Create: `backend/tests/integration/test_social_analysis_backlog.py`

**Persistence contract:**

```text
social_extraction_work:
  id PK; content_item_id FK; input_hash; prompt_version; schema_version;
  selected_model; input_snapshot_json; actual_provider/model; result_json; error_code;
  state CHECK pending|running|waiting_budget|succeeded|failed_retryable|failed_terminal|outside_window;
  claim_token; claim_expires_at; requested_by_admin; created_at; updated_at
  UNIQUE(content_item_id, input_hash, prompt_version, schema_version, selected_model)
social_run_work:
  run_id FK; work_id FK; input_hash; included_at; UNIQUE(run_id, work_id)
social_llm_budget_days:
  id PK; budget_date; timezone; period_start_utc; period_end_utc;
  limit_usd NUMERIC; reserved_usd NUMERIC; actual_usd NUMERIC; version
  UNIQUE(period_start_utc, period_end_utc)
social_llm_attempts:
  id PK; idempotency_key UNIQUE; budget_day_id FK; work_ids JSON;
  estimated_usd NUMERIC; actual_usd NUMERIC NULL; pricing_version;
  input/output_token_limit; actual_input/output_tokens NULL;
  state CHECK reserved|dispatched|reconciled|uncertain|released;
  provider_request_id NULL; created_at; completed_at NULL
```

Money uses Decimal, never binary floating point. Source content lives in application data, not logs/audit messages. Reconcile late completions to the dispatch-day bucket. A timezone change carries forward overlapping period usage/reservations rather than minting a new allowance.

**Interfaces:**

```python
def social_budget_date(now: datetime, timezone_name: str) -> date:
    return now.astimezone(ZoneInfo(timezone_name)).date()

class SocialLLMBudgetService:
    def reserve(self, attempt_key: str, work_ids: tuple[int, ...],
                maximum_usd: Decimal, now: datetime) -> int | None: ...
    def reconcile(self, attempt_id: int, actual_usd: Decimal | None,
                  provider_request_id: str | None) -> None: ...

@dataclass(frozen=True)
class BacklogResult:
    succeeded: int
    deferred: int
    failed: int
    outside_window: int
    next_reset_at: datetime

class ProcessSocialBacklog:
    async def execute(self, now: datetime, limit: int,
                      admin_work_ids: tuple[int, ...] = ()) -> BacklogResult: ...
```

- [ ] **Step 1: Write RED reset and reservation tests**

```python
def test_singapore_budget_day_changes_at_1600_utc():
    assert social_budget_date(datetime(2026, 9, 7, 15, 59, tzinfo=timezone.utc),
                              "Asia/Singapore") == date(2026, 9, 7)
    assert social_budget_date(datetime(2026, 9, 7, 16, 0, tzinfo=timezone.utc),
                              "Asia/Singapore") == date(2026, 9, 8)
```

Use real concurrent transactions: two $0.20 reservations with only $0.25 left admit one request. Test actual cost below/above reservation, cross-midnight completion, missing usage/pricing, worker restart, timezone changes, repeated reconciliation, and cancellation before dispatch. Missing/ambiguous billing retains conservative reservation; it does not automatically refund or retry.

- [ ] **Step 2: Write RED backlog tests**

With a fake metered LLM, exhaust $2, restart the worker, and resume saved work after reset with no X call. Preserve successful extraction after engagement-only changes. Cover oldest-publication-first ordering, exact fourteen-day boundary, retained outside_window records, admin-requested old analysis under budget, expired work claims, and duplicate task delivery. Do not count outside_window or failed results as completed analysis for a pinned historical run.

- [ ] **Step 3: Implement one metered call per reservation**

Use `LLMService.completion(allow_fallbacks=False, num_retries=0, ...)` plus verify that the underlying LiteLLM client also has retries disabled for this path. Inspect actual HTTP attempt counts in a fake transport test. Set an output token cap and reserve known input plus maximum billable output at versioned configured pricing; unknown pricing pauses dispatch. Retry scheduling, if warranted, obtains a new reservation and must not erase an uncertain prior charge. Never log raw model responses.

Use short DB claims/reservations; do not hold DB locks during LLM requests. A completed bounded batch records per-post results idempotently. At budget reset, dispatch backlog processing, not another X collection cycle. Re-evaluate a publishable run against current source/window/mode/market inputs after resumption; do not pretend a days-old frozen run is fresh.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/services/test_social_llm_budget.py tests/integration/test_social_analysis_backlog.py -q`

Commit this task's Files with message `feat: defer social analysis within a shared daily budget`.

---

### Task 7B: Shared Theme Discovery, Association Decisions, and Validation Isolation

**Files:**

- Create: `backend/alembic/versions/20260907_0037_add_social_theme_associations.py` (revises `20260907_0036`)
- Modify: `backend/app/infra/db/models/social_analysis.py`
- Modify: `backend/app/domain/social_signals/records.py`
- Create: `backend/app/services/social_theme_projection_service.py`
- Modify: `backend/app/services/theme_extraction_service.py`
- Modify: `backend/app/services/theme_discovery_service.py`
- Modify: `backend/app/services/theme_lifecycle_service.py`
- Create: `backend/tests/integration/test_social_theme_projection.py`

**Persistence contract:**

```text
social_theme_associations:
  id PK; theme_cluster_id FK; company_key; market; canonical_symbol;
  state CHECK proposed|accepted|rejected; origin CHECK social|legacy;
  decision_owner CHECK system|admin; evidence_work_ids JSON;
  policy_version; version; first_seen_at; accepted_at NULL; updated_at
  UNIQUE(theme_cluster_id, market, canonical_symbol)
social_theme_decisions:
  id PK; association_id FK; run_id NULL FK; actor; reason;
  before_state; after_state; policy_version; evidence_work_ids JSON; created_at
```

Existing constituents are preserved as legacy-accepted provenance. Social rejection cannot remove independently supported legacy membership. Keep admin decisions immutable in history and authoritative until another admin decision. Do not expire accepted business membership just because its supporting posts leave a scoring window. Proposed validation discoveries remain in extraction results; they do not need live ThemeCluster IDs.

**Interfaces:**

```python
@dataclass(frozen=True)
class ThemeProjection:
    run_id: int
    policy_version: str
    proposals: tuple[ExtractionClaim, ...]

class SocialThemeProjectionService:
    def prepare(self, run_id: int, now: datetime) -> ThemeProjection: ...
    def apply_live(self, projection: ThemeProjection, expected_mode_version: int) -> None: ...
    def decide(self, association_id: int, target: Literal["accepted", "rejected"],
               reason: str, actor: str, expected_version: int) -> None: ...
```

`apply_live` participates in Task 8's caller-owned transaction; it never commits independently. `decide` requires admin authorization and live mode; validation has no live association mutation endpoint.

- [ ] **Step 1: Write RED empty-database and validation tests**

```python
def test_validation_does_not_mutate_live_catalog(social_fixture):
    before = social_fixture.live_theme_state()
    run = social_fixture.process(mode="validation", posts="three_company_theme")
    assert run.validation_candidates
    assert social_fixture.live_theme_state() == before
    assert social_fixture.published_run_id() is None
```

Implement `social_fixture` with real database repositories and synthetic saved LLM outputs. Then switch to live and re-evaluate: assert one shared theme identity, source-grounded mentions, proposed associations after one author, accepted after two qualifying authors within fourteen days, and active after three companies plus three UTC discussion dates. All tests run without live LLM/X.

- [ ] **Step 2: Write RED provenance and override tests**

Test identical social/legacy post insertion in both orders, same company/alternate listing counts, copied claims, ambiguous identities, weak share prices, no non-social coverage, manual accept/reject with required reason, stale versions, and repeat publication idempotence. Verify a proposed association changes no measured basket; an accepted one may. Social Pulse changes alone do not change legacy attention. Social-only ingestion cannot alter the legacy active-ingestion-day denominator. Verify legacy lifecycle passes cannot immediately demote a valid social-led candidate/active theme using invisible evidence.

- [ ] **Step 3: Implement stage-to-live projection**

Prepare read-only identity matches; re-resolve under the existing pipeline/canonical-key uniqueness constraint when applying. Reuse naming, matching and taxonomy utilities without calling the legacy all-in-one extraction side-effect loop. Record source/work/run provenance for materialized mentions. Use existing lifecycle audit infrastructure with `social-theme-v1` policy. Live market baskets use accepted membership; legacy attention/static projections use legacy eligibility. Gate aliases, reactivation, classification and lifecycle changes together with constituents and the Social pointer; off/validation mode changes cancel pending visible writes.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/integration/test_social_theme_projection.py -q`

Commit this task's Files with message `feat: share social-led theme discovery with explicit association provenance`.

---

### Task 7C: Market-Scoped Theme Measurements and Freshness

**Files:**

- Create: `backend/app/services/social_theme_market_service.py`
- Modify: `backend/app/domain/social_signals/records.py`
- Modify: `backend/app/services/social_confirmation_reader.py` (create here; Task 9 completes orchestration)
- Modify: `backend/app/services/theme_discovery_service.py`
- Create: `backend/tests/unit/services/test_social_theme_market_service.py`
- Create: `backend/tests/unit/services/test_social_confirmation_reader.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ThemeMarketEvidence:
    theme_key: str
    market: str
    session_date: date
    benchmark_symbol: str
    basket_version: str
    accepted_company_count: int
    components: dict[str, float | None]
    measured_company_counts: dict[str, int]
    reasons: dict[str, str]

class SocialThemeMarketService:
    def measure(self, theme_key: str, market: str, as_of: datetime,
                accepted_symbols: tuple[str, ...]) -> ThemeMarketEvidence: ...
```

Define this record in `records.py`. Reuse Market Benchmark Registry selection and price/RS calculations, not a global ThemeMetrics/SPY shortcut. Freeze basket membership and values into run explanations; proposed relations never enter measurement.

- [ ] **Step 1: Write RED coverage and freshness tests**

Test two valid companies failing the minimum, 3/4 passing coverage, 3/5 failing, 7/10 passing exactly, duplicate listings not increasing numerator/denominator, mixed Markets isolated, missing benchmarks, and no valid history yielding missing rather than zero. For every component apply the minimum-three/70% gate to its usable company cohort. Reuse existing indicator validity rules (e.g. a 50-session measure needs the engine's valid 50-session history); do not fabricate estimates from shorter series.

- [ ] **Step 2: Implement session eligibility and component projection**

Use the existing calendar service for MIC session closes plus 120-minute configurable grace. Add fixed-clock tests for intraday, weekends, holidays, early closes, DST, exact grace boundary and missing calendar. Show actual session dates. Feature runs and Market exposure must be coherent and non-future. Price/RS/breadth components stay independent of Social engagement even when Social discovered the basket.

- [ ] **Step 3: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/services/test_social_theme_market_service.py tests/unit/services/test_social_confirmation_reader.py -q`

Commit this task's Files with message `feat: measure theme confirmation by market and coverage`.

---

### Task 8: Observation Writer and Atomic Publication Repository

**Files:**
- Create: `backend/app/infra/db/repositories/social_signal_writer.py`
- Create: `backend/app/infra/db/repositories/published_social_signal_reader.py`
- Create: `backend/tests/unit/repositories/test_social_signal_writer.py`
- Create: `backend/tests/integration/test_social_signal_publication.py`

**Interfaces:**
- Upserts canonical `ContentItem`, many-to-many memberships, newer metrics, and ticker mappings transactionally per source.
- Creates immutable runs/snapshots and swaps `latest_published` together with eligible live Theme projection changes under one short transaction after quality validation and mode/version recheck. Staged validation results never enter this transaction.

- [ ] **Step 1: Write RED observation tests**

Assert cross-list dedup retains both memberships; newer observations update only observed metrics; missing new metrics do not erase old values; older observations do not overwrite; duplicate delivery is idempotent.

- [ ] **Step 2: Write RED publication tests**

Assert a run with at least two pinned enabled sources advances the pointer only when every pinned source has a valid successful bounded read and its collected run inputs finish processing. A pending/disabled/archived source is absent from the run. Failed reads, missing participation, or unfinished analysis block publication; successful capped/limited-history reads may publish warming-up results with source/window coverage labels and unsupported acceleration omitted. Rollback preserves the previous pointer; reader returns rows from exactly one run; no pointer returns typed unavailable. Test startup and adding a busy list, unknown/short outcomes not proving exhaustion, persistent gaps despite fourteen elapsed days, and no missing-history-as-zero calculations.

- [ ] **Step 3: Implement repositories and run GREEN**

Prepare slow work outside the publication transaction. Lock the shared runtime-policy row and pointer, verify expected mode/version is still live, apply the prepared Theme projection idempotently, and swap the pointer. Rollback must undo both projection and pointer. Validation stores an admin result only. Record run input/work IDs and copied metric observations so later engagement updates do not mutate published explanations. Budget-paused runs keep durable work without holding the provider-read lease; subsequent collection proceeds.

Reject a delayed run that would overwrite a newer published generation. When backlog finishes, create/re-evaluate a current generation with the current enabled-source set, windows, coverage and market inputs; do not change timestamps on an old immutable run or mark aged-out work successful.

Run: `cd backend && ./venv/bin/pytest tests/unit/repositories/test_social_signal_writer.py tests/integration/test_social_signal_publication.py -q`

- [ ] **Step 4: Commit**

```bash
git add backend/app/infra/db/repositories/social_signal_writer.py backend/app/infra/db/repositories/published_social_signal_reader.py backend/tests/unit/repositories/test_social_signal_writer.py backend/tests/integration/test_social_signal_publication.py
git commit -m "feat: publish immutable social signal snapshots"
```

---

### Task 9: Confirmation Reader and Refresh Orchestration

**Files:**
- Modify: `backend/app/services/social_confirmation_reader.py`
- Modify: `backend/tests/unit/services/test_social_confirmation_reader.py`
- Create: `backend/app/use_cases/social_signals/refresh.py`
- Create: `backend/tests/unit/use_cases/social_signals/test_refresh.py`
- Modify: `backend/app/wiring/use_case_factories.py`

**Interfaces:**
- Reads one published `FeatureRun` per Market, matching `StockFeatureDaily`, latest non-future `MarketExposure`, Market-specific liquidity eligibility, exact group rank cohort, and Task 7C's Market-scoped accepted-basket Theme values. Reuse the public Opportunity State evidence adapter in `backend/app/services/opportunity_state_service.py` for Market liquidity availability/pass status; do not invent a second liquidity threshold or call its private policy helpers.
- `RefreshSocialSignals.execute(origin, now)` owns the twelve-step run flow and returns a redacted `SocialRunResult`.

- [ ] **Step 1: Write RED coherent-confirmation tests**

Assert all rows for a Market use one feature run ID; stale feature/exposure dates are marked missing; setup score/readiness and `rs_rating_1m/3m` come from the feature row; group conversion uses the same Market/date cohort; theme confirmation ignores mention velocity/momentum and uses Task 7C's three-company/70% gates, Market benchmark and fresh accepted-basket components. Social discovery provenance does not disqualify independent price evidence.

- [ ] **Step 2: Write RED refresh workflow tests**

Use in-memory fakes to assert order: load mode/version and pin every enabled source, require at least two, create run, read each source, validate, write observations, reuse/enqueue metered extraction, resolve, prepare Theme changes, aggregate, load confirmation, score three windows, persist, quality-check, and publish only in live mode. Assert provider disabled/off does no automatic I/O, validation changes no live tables, budget exhaustion retains work and releases the provider lease, and one failed enabled source keeps the old pointer; retry uses the same run identity; source changes during a run do not alter its pinned set; identical pinned inputs serialize identically.

Assert every source without an initial successful read—including a newly enabled source added after launch—attempts fourteen-day history with at most 1,000 posts; later reads retain overlap for metric refresh and cap at 200 posts, using application-owned progress without xui new-only filtering. A successful read reaching its limit records limited history rather than blocking publication by itself. Source/read failure and incomplete LLM processing still block. Preserve the first successful bounded read so a permanently busy source does not repeat an initial backfill forever.

- [ ] **Step 3: Implement confirmation reader and orchestrator**

Persist `feature_run_ids_by_market`, exposure dates, scoring formula, committed source progress, extraction work/result versions, operating mode/version, accepted-basket/benchmark IDs, historical coverage, normalization scope, all component inputs, and state reasons in the immutable run/snapshot records. Never query mutable latest data from the API path.

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
- Beat collection entries run at minute 17, hours `0,6,12,18` in configured Celery timezone only in validation/live with a configured provider. The ordinary UI capability is not the scheduler gate.
- Add `resume_social_analysis` on `social_ingestion`, using ProcessSocialBacklog, scheduled for the next database budget reset (midnight Asia/Singapore by default). This resumes analysis and re-evaluation without another X read. Recheck mode/provider on delivery; off/disabled leaves work saved.
- Redis provider-read lease key `social-signals:provider-read:lease` serializes refresh and source tests; manual cooldown key `social-signals:manual-refresh:cooldown` applies only to refresh dispatch.

- [ ] **Step 1: Write RED schedule and route tests**

Assert off/disabled schedules no collection/analysis; validation/live schedule one six-hour collection entry plus budget-reset resumption; all social task types route to the dedicated queue; stale refresh deliveries collapse; one transient network error retries once; auth/challenge/schema failures do not retry.

- [ ] **Step 2: Write RED lease and cooldown tests**

Assert concurrent deliveries create one run; a source test cannot overlap a refresh or another source test; lease ownership token prevents another worker from releasing it; manual dispatch inside one hour returns HTTP 429 with retry-after; scheduled runs are not blocked by manual cooldown but still honor the singleton provider-read lease.

- [ ] **Step 3: Write RED explicit source-test tests**

Assert validation reads only the selected pending/disabled source, requests at most five posts, stores no `ContentItem`, `SocialPostSource`, metrics, run, snapshot, or pointer, records queued/running/completed state plus only redacted counts/status/provider/time, audits request and completion, rejects pending/disabled enablement when the pass does not match the current provider, and does not invoke any provider automatically when a source is created. Already-enabled sources are validated by the next complete refresh after a deployment changes provider.

- [ ] **Step 4: Implement tasks and Operations projection**

Expose run ID, operating mode, collection/processing/history/freshness dimensions, provider/model labels, source counts/coverage, budget spent/reserved/remaining and reset time, oldest backlog age/waiting/skipped counts, timestamps, formula version, cooldown and stable codes. Validation candidate/evidence projections are separate admin-authorized responses, not logs. Redact provider error text through an allowlist; never return xui paths or raw stderr.

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
- Modify: `backend/app/api/v1/themes_content_sources.py`
- Create: `backend/tests/unit/test_theme_social_source_admin_boundary.py`

**Interfaces:**
- `GET /v1/social-signals/summary?market=US`
- `GET /v1/social-signals/queue?market=US&window=7d&view=actionable&rank_mode=blended&page=1&page_size=50`
- `GET /v1/social-signals/candidates/{candidate_key}/evidence?window=7d`
- `GET /v1/social-signals/theme-pulse?market=US`
- `GET /v1/social-signals/context?market=US&window=7d&page=1&page_size=50`
- `GET /v1/social-signals/unresolved?scope=market&market=US&window=7d&page=1&page_size=50`; `scope=unknown` is explicitly global.
- Admin: `GET/PATCH /v1/social-signals/admin/runtime` for shared mode/provider and expected_version (credentials remain deployment secrets); environment defaults initialize only an empty registry, later changes use this explicit audited admin operation.
- Admin: `GET /v1/social-signals/admin/validation/{run_id}`, `GET /admin/analysis`, `POST /admin/analysis/{work_id}/retry`, `GET /admin/associations`, `POST /admin/associations/{id}/decision` (accepted/rejected, required reason and expected_version).
- Admin: `GET /v1/social-signals/admin/health`, `GET /admin/runs`, `GET /admin/sources`, `POST /admin/sources`, `PATCH /admin/sources/{source_id}`, `POST /admin/sources/{source_id}/test`, `POST /admin/sources/{source_id}/transition`, and `POST /admin/refresh`.

- [ ] **Step 1: Write RED schema/query tests**

Assert supported enum values, 1-100 page size, stable ordering, Market scoping, Actionable filtering, independent rank mode, top-three evidence, short excerpt truncation, canonical X URLs, related listings, unresolved rows, and published-run-only reads.

- [ ] **Step 2: Write RED authorization and redaction tests**

Assert read routes require server session; admin routes additionally require `X-Admin-Key`; disabled returns typed `supported=false`; no published run returns HTTP 200 typed unavailable; payload JSON never contains `token`, `cookie`, `storage_state`, `config_path`, raw stderr, or private package details.

Test validation output inaccessible to ordinary users even by direct run ID; null scores serialize as null and render as em dash; context/unresolved pages have separate counts and never contaminate rank cohorts. Association decisions require live mode/admin authorization, nonblank reason, optimistic version and immutable audit; backlog retries always use the shared budget. Admin-only routes remain available when normal Social UI is hidden.

Source administration tests assert a required trimmed name; numeric-ID and canonical-URL parsing; duplicate rejection; pending creation without provider traffic; editable name with immutable list ID; asynchronous test dispatch; enablement only after a current-provider pass; HTTP 409 on stale `expected_version`; HTTP 422 when a disable/archive would leave fewer than two enabled; terminal archival; and an immutable redacted audit record for every accepted action and test result.

- [ ] **Step 3: Implement queries, schemas, and router**

Reject legacy Theme source mutations targeting any source with a `SocialSourceConfiguration`, in every lifecycle state and regardless of whether the caller is an administrator. Return HTTP 409 with stable code `social_source_managed_elsewhere` and guidance to Operations → Social Sources, before changing any fields, reconciling pipelines, or dispatching work. Social source mutations must use the admin-protected Social service so testing, minimum-enabled-source validation, immutable identity, and auditing cannot be bypassed. Prevent legacy create/update from duplicating a social-owned canonical list identity under an alternate accepted URL representation. Ordinary Theme source behavior remains unchanged.

In `test_theme_social_source_admin_boundary.py`, exercise legacy rename, URL/type/pipeline changes, activation, deactivation, deletion, and duplicate creation for social-owned sources. Assert rejection for ordinary and admin callers, unchanged database state, and no provider calls. Verify equivalent operations on ordinary Theme sources retain existing behavior and legitimate Social admin actions still apply lifecycle checks and audit records.

Include formula version, generated/published timestamps, stale/degraded state, enabled-source coverage, supported controls, component explanations, state reasons, and Market-aware canonical symbol. Evidence excerpts are plain text and at most 280 characters.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_social_signals_api.py tests/unit/test_theme_social_source_admin_boundary.py -q`

```bash
git add backend/app/schemas/social_signals.py backend/app/use_cases/social_signals/queries.py backend/app/api/v1/social_signals.py backend/app/api/v1/router.py backend/tests/unit/test_social_signals_api.py
git add backend/app/api/v1/themes_content_sources.py backend/tests/unit/test_theme_social_source_admin_boundary.py
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

Cover disabled omission, loading, unavailable, empty, healthy, partial, stale, reauthentication, and published warming-up/limited-history states. Show source/window coverage and unavailable acceleration explicitly. Exercise Actionable/All Signals, Blended/Pure Social, 1D/7D/14D, source/theme/instrument/state/ticker filters, pagination, and Market changes.

- [ ] **Step 3: Implement dense table and drawer**

Columns: Symbol, Market, Queue, Social, mentions, authors, Setup/readiness, RS, group rank, linked theme, state, freshness. Add separate paginated Context and Needs resolution sections in All Signals; Market unknown remains global across tab changes. Render null scores as em dash, reduced-coverage details, and warming-up badges; do not sort unranked evidence into the table. Row click opens the evidence drawer before any chart. Drawer renders component math, state reasons, related listing information, and at most three plain-text posts with author/time/engagement/list badges/Open on X.

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
- Theme Social Pulse is display-only and derives from the published Social run. Add Discovering for candidate shared themes and distinct social/market-strength labels, including company coverage and insufficient-market-data state.
- Operations health uses the existing admin-key interaction pattern and owns manual refresh plus full source administration.

- [ ] **Step 1: Write RED Daily and Theme tests**

Assert top five, dominant themes, exposure posture, enabled-source coverage such as `3/3`, last success, stale/degraded badge, and navigation. Assert adding Social Pulse does not change Theme ordering, and it disappears when the capability is off.

- [ ] **Step 2: Write RED admin tests**

Assert ordinary Operations inventory remains visible and health prompts for the admin key. Exercise adding a required name plus numeric ID/URL, pending state without network traffic, an explicit Test List warning and asynchronous five-post-or-fewer test, enable after pass, rename without changing list ID, stale-edit conflict, dynamic source coverage, rejection when disable/archive would leave fewer than two enabled, archive confirmation/history, manual refresh 202/429, and the exact non-secret reauthentication action. Cover mode/validation cost warnings, admin-only validation preview, $2 budget/day/reset display, saved backlog and outside-window admin retry, and reasoned association accept/reject with stale-version conflict.

The panel displays name, canonical list ID/URL, pending/enabled/disabled/archived state, tested provider/time/outcome, last successful collection, and audit history. Hide archived sources by default with a `Show archived` switch. Do not offer arbitrary source management to ordinary signed-in users.

Update existing Theme source controls to mark social-owned rows as managed in Operations → Social Sources and remove their legacy mutation actions. Show the navigation action only to administrators; enforce the restriction on the server regardless of UI state. Add UI coverage for both user roles and unchanged ordinary Theme source controls.

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
- Modify: `backend/app/services/static_site_export_service.py`
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

Keep new UI components under the live graph. Static Theme export must read legacy-eligible evidence and legacy-accepted membership, not social-only identities or live social-expanded baskets. Add provenance-aware query filtering/projection without adding social keys/routes. Test mixed-source posts and accepted Social additions after publication, not just absence of Social UI imports.

- [ ] **Step 3: Run GREEN and commit**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py -q`

Run: `cd frontend && npm run test:run -- src/static/socialIsolation.test.jsx src/static/pages/StaticHomePage.test.jsx src/static/pages/StaticThemesPage.test.jsx`

```bash
git add backend/tests/unit/test_static_site_export_service.py backend/tests/unit/test_export_static_site_script.py frontend/src/static
git add backend/app/services/static_site_export_service.py
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

Keep the existing public runtime as the default final stage. Rename the existing second stage `runtime-base`. Derive a private builder from `builder`, then a private runtime from `runtime-base`, and keep a final `public` alias as the default. Extend the Dockerfile with:

```dockerfile
FROM builder AS social-xui-builder
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
ARG XUI_READER_REF
RUN test -n "$XUI_READER_REF"
RUN --mount=type=ssh,required=true \
    --mount=type=secret,id=github_known_hosts,target=/root/.ssh/known_hosts,required=true \
    pip install --no-cache-dir \
    "xui-reader[cli] @ git+ssh://git@github.com/xang1234/xui.git@${XUI_READER_REF}"

FROM runtime-base AS social-xui
USER root
COPY --from=social-xui-builder /opt/venv /opt/venv
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/stockscanner/playwright
RUN python -m playwright install --with-deps chromium \
    && chmod -R a+rX /opt/stockscanner/playwright \
    && rm -rf /var/lib/apt/lists/*
USER stockscanner

FROM runtime-base AS public
```

Use the `docker/dockerfile:1.7` syntax line. Do not echo the ref, environment, pip config, or SSH diagnostics into artifacts.

The private stage requires a verified GitHub known-host file supplied through BuildKit secret `github_known_hosts`, in addition to SSH forwarding. The operator verifies host-key fingerprints against GitHub's official published keys; do not trust an unverified keyscan. Require an exact private commit SHA. Install into the intended virtual environment with build-time root permissions, not the current non-root runtime user. Install the Playwright-matched Chromium binary and browser OS dependencies in the final private runtime, set a stable browser path outside mounted data/session volumes, and restore `USER stockscanner`. Preserve a public/default build target with no private dependency acquisition; appending a private target must not make it the unintended default public build. Exclude private checkout/session material from the public context. Private images and caches contain proprietary installed code and must remain private; anyone with pull access is trusted with that code. Build credentials must never persist in layers, and X session material is runtime-mounted only.

Supplement text/Compose checks with trusted builds and real non-root browser-launch smoke tests against local content on both arm64 and amd64. These tests require no X account and make no X requests. Validate executable/browser paths, shared-library availability, permissions, absence of runtime build credentials, and absence of private dependencies from the public image. Do not publish private caches or verbose private install logs as public CI artifacts. Test the local GHCR-free build and the equivalent Compose image override as well as the registry-pull path.

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

Also document GHCR-free development with `DOCKER_BUILDKIT=1 docker build --ssh default --secret id=github_known_hosts,src="$GITHUB_KNOWN_HOSTS_FILE" --target social-xui --build-arg XUI_READER_REF="$XUI_READER_REF" -t stock-screener-social-xui:dev -f backend/Dockerfile .`, after setting `XUI_READER_REF` to an exact private-repository commit SHA and `GITHUB_KNOWN_HOSTS_FILE` to the verified host-key file; explain accepted X account risk, read-only behavior, six-hour cadence, profile permissions, re-login, provider switching, and rollback to disabled.

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

Use `webfactory/ssh-agent@v0.9.0`, `docker/setup-qemu-action@v3`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, and `docker/build-push-action@v6`. Build the `social-xui` target with `ssh: default` and the verified `github_known_hosts` BuildKit secret, run real non-root browser launch against local content on each architecture before push, and keep provenance/SBOM in the private package boundary without checkout or credential artifacts. Private dependency names may appear in private SBOMs; never publish those as unrestricted public workflow artifacts.

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

Seed the two system lists, create a named pending third list, prove creation performs no provider read, validate it with a five-post fixture, enable it, and seed active securities across all five Markets plus existing feature/exposure/group/theme inputs. Normalize official and xui fixture records through the same contract, publish only after all three enabled sources complete, and assert stable Blended/Pure Social ordering, binary cross-list attribution, state decisions, validation isolation plus legitimate shared Theme changes, and no secret-like fields. Include an empty Theme database case, warming-up history, midnight budget resumption, and two-transaction admin contention.

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

Build the public backend without SSH/GHCR/X credentials and verify it starts with `SOCIAL_INGEST_PROVIDER=disabled`. Render official Compose without xui inputs. On a trusted machine only, build/pull the private arm64 image, mount the dedicated automation profile, execute one `SOCIAL_SIGNALS_MODE=validation` run under the approved real-read/LLM budgets, and compare counts/mappings/scores against the approved report and SVG.

- [ ] **Step 6: Record rollout evidence and acceptance**

Append a short implementation-evidence section to the design spec with commands, fixture hashes, image digest, all-enabled-source coverage, source-lifecycle audit checks, unresolved mappings, and evidence of shared discovery, accepted basket changes, and score separation. Do not include post bodies, credentials, session paths, or private package internals.

- [ ] **Step 7: Commit**

```bash
git add backend/tests/integration/test_social_signal_end_to_end.py frontend/tests/smoke/social-signals.spec.js frontend/tests/smoke/socialSignalFixtures.js .github/workflows/ci.yml docs/superpowers/specs/2026-09-06-social-signal-queue-design.md
git commit -m "test: verify social signal queue end to end"
```

---

## Review Resolution Traceability

| Review finding | Implementation coverage |
| --- | --- |
| 1. Duplicate collection through legacy jobs | Task 4 ownership selection/direct-call guards; Task 6 repeatable reads; Task 10 worker routing |
| 2. Legacy admin bypass | Task 4 shared service; Task 11 legacy endpoint rejection; Task 13 managed-elsewhere controls |
| 3. Shared discovery without score leakage | Tasks 7/7B/7C; Task 8 atomic live projection; Tasks 13/14 live/static views |
| 4. New-only reads lose updates/retries | Tasks 6/8 durable repeatable reads and observation/progress transactions |
| 5. Full-history gate prevents startup | Tasks 2/8/9 history/processing separation; Tasks 11/12 coverage labels |
| 6. Incomplete private Docker recipe | Tasks 15/16 explicit private stages, verified host keys, browser runtime and per-architecture launch tests |
| 7. Concurrent minimum-two violation | Task 4 registry lock plus real concurrent-transaction regression |
| 8. Missing scores, freshness, unknown markets | Tasks 3/7C/9 nullable scoring/calendar/coverage; Tasks 11/12 separated sections |
| 9. Missing validation mode | Tasks 1/7B/8/10 mode gates; Task 13 admin preview; rollout uses explicit validation/live |
| Follow-up: LLM dependence and deferred $2 budget | Tasks 7/7A/10/11/13; midnight Asia/Singapore, cached results, aging/admin retry |

## Final Review Checklist

- [ ] Compare every implemented behavior with the approved design and both supplied research artifacts.
- [ ] Confirm off/disabled schedules no collection or analysis, validation publishes nothing user-visible, and no provider/model fallback bypasses explicit policy or budgets.
- [ ] Confirm both seed source IDs, administrator-added sources, and all five Markets are covered by tests.
- [ ] Confirm required names, pending creation, explicit five-post testing, current-provider validation, two-enabled minimum, optimistic edits, terminal archival, and redacted immutable audits.
- [ ] Confirm every enabled source is pinned at run start and must succeed, while source changes during a run cannot alter that run.
- [ ] Confirm formula weights, percentile tie behavior, recency half-life, winsorization, author cap, and stable tie-breaks are pinned by tests.
- [ ] Confirm failed-source/missing-participation/unfinished-processing runs cannot advance the pointer, successful limited-history runs publish only with explicit labels, and stale last-known-good data remains readable.
- [ ] Confirm validation and Social Pulse alone leave live Theme fixtures unchanged; live accepted discovery may change Market baskets while legacy attention remains isolated. Static legacy projections exclude social-only additions.
- [ ] Confirm API authorization/redaction and admin cooldown behavior.
- [ ] Confirm static exports and static UI contain no social data or request.
- [ ] Confirm public install/build/test works with no private dependency or credential.
- [ ] Confirm private image history/config/SBOM contain no SSH key, token, storage state, or checkout artifact.
- [ ] Confirm local Apple Silicon and amd64 image manifests, real non-root local-content browser launches, and the dedicated non-root profile mount.
- [ ] Confirm $2/day reservations, midnight Asia/Singapore reset across worker timezones/restarts, saved backlog, aging, and admin overrides.
- [ ] Confirm shared empty-database discovery, two-author association acceptance, three-company/three-date activation, and per-Market three-company/70% measurements.
- [ ] Confirm MIC-session freshness, two-hour grace, null scores, Market unknown grouping, and warming-up disclosure.
- [ ] Use superpowers:requesting-code-review, resolve findings, then use superpowers:verification-before-completion before claiming completion.

## Rollout Sequence

1. Merge Tasks 1-11 with the capability disabled.
2. Deploy schema and public application in off mode; verify existing Theme behavior and static artifacts remain unchanged before live Social projection.
3. Merge live UI Tasks 12-14, still disabled.
4. Publish the private worker from Tasks 15-16 and validate its image digest and profile isolation.
5. Set the shared runtime to validation with the selected provider through the admin operation (deployment default `SOCIAL_SIGNALS_MODE=validation` initializes a new installation). Attempt the bounded fourteen-day backfill and inspect admin-only results, explicit warming-up/coverage limits, source administration/audits, unresolved names, cross-list deduplication, and rankings. Confirm no live Theme or Social publication side effects. Validation performs real reads/analysis under the same budgets.
6. Explicitly set shared runtime mode to live through the admin operation and align deployment defaults (`SOCIAL_SIGNALS_MODE=live`). Reuse saved observations/extraction in a newly evaluated live run; publish only after current checks pass. Never auto-publish an old validation snapshot.
7. Roll back by applying shared mode off/provider disabled and matching deployment defaults `SOCIAL_SIGNALS_MODE=off`, `SOCIAL_INGEST_PROVIDER=disabled`; retain data and the last pointer for audit.
