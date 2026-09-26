# Company Exposure Map — operator runbook (S1: US verify-only shadow)

Related: [design](../superpowers/specs/2026-09-25-company-exposure-map-design.md) ·
[plan](../superpowers/plans/2026-09-25-company-exposure-map.md) ·
[ADR 0006](../adr/0006-company-exposure-assessments.md) ·
[baseline and recorded deviations](../implementation/company-exposure-baseline.md)

This build installs one stage: **`shadow_verify_us`**. An administrator can
ask whether one US listing is exposed to one Economic Theme; the system
resolves the issuer, reads SEC primary filings, verifies claims with the
OpenCode Go subscription route, and shows a **shadow preview**. Nothing in
this slice changes theme membership, live baskets, classifier grounding or
any published generation. Search, discovery and paid search are not
installed.

| Stage | Status in this build |
|---|---|
| `shadow_verify_us` | installed; off until configured |
| `shadow_verify_all_markets`, `generation_reads`, `automatic_admission`, `bounded_discovery`, `paid_search`, `classifier_grounding` | `not_installed` — no setting enables them |

`EXPOSURE_RESEARCH_MODE=live` is refused (`live_mode_not_installed`); use `shadow`.

## 1. Before first start

1. Apply migrations (additive; `20260925_0058`–`20260926_0060`):
   `cd backend && alembic upgrade head`.
2. Create the evidence store and give it to the backend's non-root user
   (uid/gid 1000, see `CLAUDE.md`):

   ```bash
   mkdir -p ./data/exposure-evidence && sudo chown -R 1000:1000 ./data/exposure-evidence
   ```

   `status` reports a `storage_not_writable` reason for `shadow_verify_us`
   when the worker cannot write there, instead of failing inside a job.
3. Build and start the dedicated worker (Docker):

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml \
     -f docker-compose.exposure.yml --profile exposure-research up -d --build
   ```

   Locally: `EXPOSURE_WORKER_ENABLED=true ./start_celery.sh`. The worker
   consumes only the `exposure_research` queue; no other worker consumes it.
   There is no Redis credential or browser profile to provision in S1.

## 2. Configure the shadow slice

Set in `.env.docker` (or `backend/.env`). Starting the worker or setting a
key does **not** enable research; each item below is explicit.

| Variable | S1 value | Why |
|---|---|---|
| `EXPOSURE_RESEARCH_MODE` | `shadow` | default `disabled` |
| `EXPOSURE_LLM_TEXT_ROUTE_ENABLED` | `true` | subscription text route (`opencode-go/kimi-k2.6`) |
| `OPENCODE_GO_API_KEY` | set | presence only is shown, never the value |
| `EXPOSURE_LLM_DAILY_REQUEST_LIMIT` / `..._TOKEN_LIMIT` | your local allocation | blank = `allocation_not_configured` pause |
| `EXPOSURE_ALLOCATION_TIMEZONE` | e.g. `UTC` | daily period boundary |
| `EXPOSURE_SEC_USER_AGENT` | `Org Name contact@example.com` | SEC requires a contact User-Agent |
| `EXPOSURE_PAID_SEARCH_ENABLED` | `false` | paid search is not installed |
| `ADMIN_API_KEY`, `ADMIN_PRINCIPAL_ID` | set | requests are admin-only and audited as this principal |

Provider remaining balance is reported as `unknown`; the local allocation is
what the system enforces. It never falls back to a metered model.

## 3. Check status

```bash
cd backend
./venv/bin/python scripts/company_exposure.py status
```

The output lists configuration (no secrets), each stage with `allowed` and
its blocking reasons, job counts by state, work items by status, and the
database migration head. `shadow_verify_us` must show `allowed: true`
before requesting research.

## 4. Request research and read the preview

In the app: **Operations → Company exposure research (shadow)** → unlock with
the admin key → US symbol + Economic Theme ID → *Request research*. The
panel polls the job and shows the shadow preview when the dossier revision
is sealed.

API (admin key required; identity comes only from the credential):

```text
POST /api/v1/company-exposures/research-requests
     {"kind": "verify"|"refresh", "symbol": "EXMP" | "security_id": 42,
      "economic_theme_id": "<uuid>", "idempotency_key": "<key>",
      "supplied_cik": "1234567" (optional → reviewable proposal only)}
GET  /api/v1/company-exposures/research-jobs/{job_id}           # operational progress
GET  /api/v1/company-exposures/research-jobs/{job_id}/preview   # view_kind=shadow_preview
```

Repeating a request with the same idempotency key returns the same job. If
the broker was unavailable (`dispatch: not_dispatched`), run
`scripts/company_exposure.py process` to advance queued stages inline.

Stages: `resolve_issuer` → `acquire` → `verify`. Final states: `partial`
(some coverage gap, e.g. an older filing not found) or
`ready_for_publication` (complete for the requested scope — still nothing is
published in S1).

## 5. Paused jobs and how to resolve them

| State / condition | Meaning | Action |
|---|---|---|
| `review_required` / `multiple_ciks`, `ticker_not_in_submissions`, `cik_linked_to_other_issuer`, `security_already_linked_elsewhere`, `cross_listed_issuer`, `ticker_changed_since_prior_link`, `inactive_listing` | The official SEC registry match was not safe to accept automatically | Confirm the CIK from the filing cover page, then resolve (below) |
| `review_required` / `issuer_link_required` | No accepted issuer link and no registry route | Resolve (below) |
| `unavailable_capability` / `sec_user_agent_not_configured`, `route_not_approved`, `subscription_credentials_missing`, `theme_definition_unavailable`, `research_disabled` | Configuration or capability missing | Fix configuration, then resume |
| `paused_allowance` / `allocation_not_configured`, `capacity_exhausted` | Local allocation missing or used up for the period | Wait for the next period or raise the limit; resume |
| `paused_storage` | Evidence store full or below the free-space floor | Free space / raise the cap; resume |
| `retryable_failure` | Transient fetch/provider error; retried with backoff (max 4 attempts) | None; `terminal_failure` after the last attempt |

**Registry-resolved vs administrator-reviewed links.** A single unambiguous
SEC match (ticker → one CIK, confirmed by that CIK's submissions record, one
listing) is accepted automatically under policy
`official_registry_single_listing`. Anything else stays a proposal until an
administrator applies it. To resolve a `review_required` job:

```bash
./venv/bin/python scripts/company_exposure.py resolve-issuer --security-id 42 --cik 1234567           # dry run
./venv/bin/python scripts/company_exposure.py resolve-issuer --security-id 42 --cik 1234567 --apply   # needs ADMIN_PRINCIPAL_ID
./venv/bin/python scripts/company_exposure.py resume <job_id> --apply
./venv/bin/python scripts/company_exposure.py process
```

`resume` re-runs the paused stage; it does not reset the job's cumulative
budgets.

## 6. Holds, freshness and uncertain spend

- **Freshness** follows the claim's own evidence date, never the download
  time: 450 days for roles/products, 180 days for customer relationships and
  commercial status; undated support is held. Materiality is valid for its
  stated period.
- **Provider-free holds**: `refresh-holds` records `stale`/`undated` holds for
  expired selected claims without any network or model call:

  ```bash
  ./venv/bin/python scripts/company_exposure.py refresh-holds          # dry run
  ./venv/bin/python scripts/company_exposure.py refresh-holds --apply
  ./venv/bin/python scripts/company_exposure.py inspect-holds
  ```

  Holds block new automated use; they never delete or rewrite a sealed
  revision. Lifting a hold requires a reason and new cited support.
- **Uncertain reservations**: a model request that may have executed (read
  timeout, mid-stream failure) stays charged as `uncertain`. At period close
  it becomes `expired_uncertain`; it is never refunded. Cancelling a job
  releases only never-dispatched reservations.
- **Evidence retention**: originals are content-addressed under the store
  (5 GiB cap, 1 GiB free-space floor). Unreferenced blobs are collected after
  30 days (temporary files after 24 hours) and leave a tombstone; reads of a
  tombstoned original fail with a typed reason rather than returning data.

## 7. Stop conditions

Stop and investigate — do not force progress — on any of: admin
authorization failure; no local allocation; issuer/scope conflict; invalid
original locator; an unacceptable provider result; a stale safety token;
a skipped or failing required PostgreSQL gate test
(`scripts/run_required_company_exposure_postgres.py --slice S1`).

## 8. Disable and roll back

- **Disable**: set `EXPOSURE_RESEARCH_MODE=disabled` (and optionally stop the
  `celery-exposure-research` service). New requests return
  `409 research_disabled`; queued stages pause with `research_disabled`.
  Evidence, dossiers, holds and job history are retained. Source and Social
  behaviour are unaffected — S1 never writes membership.
- **Rollback**: the S1 tables are additive. Because nothing in this slice is
  published or joined into live membership, an older binary can run beside
  the retained tables. Do not drop them to "clean up"; they are the audit
  history.

## 9. Release evidence

Record for each activation: source SHA, migration head from `status`, the S1
gate artifact (`run_required_company_exposure_postgres.py --artifact …`),
frontend test/build results, and the operator principal. A live SEC probe is
recorded separately: this slice's CI and unit runs use saved fixtures and
mocked transports, which are not a live probe.
