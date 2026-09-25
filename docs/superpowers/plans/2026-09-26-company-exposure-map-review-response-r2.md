# Company Exposure Map — Review Disposition (R2)

**Date:** 2026-09-26  
**Status:** R2 applies the second implementer review. The user accepted these fixes, choosing "no Redis connection for egress" for the browser broker, along with the R1 proposed defaults.  
**Repository inspection:** `xang1234/stock-screener` at `28c220e4e4ca5afcb5a380678bb2b80dd7f388b7`.

## Paired artifacts

| Artifact | SHA-256 |
|---|---|
| `2026-09-26-company-exposure-map-design-r2.md` | `84784bdea4d789c2f7bb0f1e5529054259ab2fa8c66c504f30678efcc8fae2df` |
| `2026-09-26-company-exposure-map-implementation-plan-r2.md` | `ebbf8349f6480af16ebe19451d682a4911191d67efe35e69b9e7805c10f077ce` |
| R1 design (superseded) | `a5f407209777c6e48b9a58311b090aafebec418ad370184455cdd2182ed526d9` |
| Original approved design | `a182065bf6b1e5e07026045b50e6ae9124501893b383e3a940efb5f9f13f8249` |

The plan pins the R2 design hash in its header.

## Disposition

| Finding | Repository evidence | R2 change |
|---|---|---|
| 1. No CIK source for US research | No model, service or test in the repo references a CIK. | Spec §4.2/§9.2/§18. The CIK is resolved from SEC's ticker-to-CIK file, and the CIK's submissions record must list the same ticker. Both documents are retained as captured evidence under `sec_edgar` pacing. A single unambiguous, non-conflicting match is auto-accepted as a registry-resolved **single-listing** link (`official_registry_single_listing`). Multiple candidates, an unconfirmed ticker, conflicting links, cross-listings and ticker changes all go to administrator review. Implemented in Task 05 (`accept_registry_match`), Task 09 (`USIssuerResolver.resolve_cik`) and Task 17A (resolve first; pause as `review_required` otherwise). |
| 2. Three route schemes | Task 21A used `/research`, spec §16 used `/research-requests` and `/research-jobs`, and a test used `/securities/42`. | Spec §16 is the only scheme: `POST /research-requests`, `GET /research-jobs/{job_id}`, and the new `GET /research-jobs/{job_id}/preview`. Tasks 21A/21/22 and tests are updated. A test asserts that no `/research` or `/securities` route is registered. |
| 3. Connect failures burned allowance | `kimi_client.complete_json` maps every `httpx.TimeoutException` to `model_timeout` and every other `httpx.HTTPError` to `model_connection_failed`. | Spec §10.2 and Task 04. `PreparationFailure` gains an optional `dispatch_phase` set from the real exception class, with the classification table in Task 04. Connect errors, connect timeouts and pool timeouts are pre-dispatch and release the reservation. Write/read timeouts and mid-stream errors are uncertain. Uncertain reservations are charged to their dispatch period and close as `expired_uncertain` (Task 04 `close_period`, run by Task 23 maintenance). Existing `code` values are unchanged. |
| 4. Redis ACL could not isolate egress | `docker-compose.yml` runs `redis-server --appendonly yes` with no authentication. | Spec §9.8/§10.3 and Appendix F.3. The broker is attached only to `exposure_public_egress` and holds no Redis, database or app credential. The research worker serves a grant-scoped `RenderPacingSession` on `/run/exposure-pacing/pacing.sock`. For each HTTP attempt it charges the root budget and acquires the shared rate key; if the worker is unreachable, the broker refuses. Removed from R1: the Redis ACL user, the `exposure_rate_control` network, `redis-acl.sh` and the Redis password secret. New F.4 probes cover application-service reachability, Docker host gateway/published-port reachability (requires a host `DOCKER-USER` rule, or rendering stays off) and pacing-RPC refusal. |
| Minor: backend read-only mount | F.1 described the mount but the Compose snippet omitted it. | F.1 adds the `backend` volume `./data/exposure-evidence:/app/exposure-data:ro` and `EXPOSURE_DOCUMENT_STORE`. |
| Minor: uid 1000 ownership | `CLAUDE.md` says the backend runs as uid 1000. | The Task 27 runbook adds `mkdir` + `chown 1000:1000`. `status` reports `storage_not_writable`. |

## Unchanged

- The 41 acceptance-case descriptions are byte-identical to R1 in both documents (checked by script).
- All 28 task identifiers and their dependency structure are unchanged.
- The evidence, materiality, membership, publication and grounding standards are unchanged.
- The R1 defaults stand as accepted: 5 GiB store, 1 GiB free floor, 30-day/24-hour GC, 1 CPU / 2 GiB worker, browser off.

## Verification performed

- All 47 Python code blocks in the plan parse. All 3 YAML blocks load.
- The plan's pinned design hash matches the design file.
- No references to the removed Redis ACL user, rate-control network or parallel research routes remain, apart from the change records that describe their removal.
- **Not performed:** application tests, migrations, Docker builds, isolation probes, live SEC/model calls. Two SEC details still need confirming during Task 09: the exact ticker-file schema, and whether the exchange-qualified variant (`company_tickers_exchange.json`) is available. Task 09 records both in `routes/us.json`.
