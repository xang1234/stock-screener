# Company exposure map: remaining work (S2 to S5)

Recorded 2026-09-28 at branch `claude/brave-carson-9i90ws`, head `3955fac`. This file hands off everything after the S1 slice (US verify-only shadow research).

It summarises three things:

- the design, from the R2 spec `docs/superpowers/specs/2026-09-25-company-exposure-map-design.md`;
- the implementation plan, from the R2 plan `docs/superpowers/plans/2026-09-25-company-exposure-map.md`;
- the handoff state.

The spec and the plan are still the authority. Where this file and they disagree, they win. This file does not change any approved default.

| Slice | Scope | Handoff file |
|---|---|---|
| S2 | HK, JP and TW acquisition; language and vision derivatives; the isolated browser renderer; the adjudicated corpus | [company-exposure-s2-markets-corpus.md](company-exposure-s2-markets-corpus.md) |
| S3 | Generation-bound publication, reviewed membership, research-only compatibility, and dossier reads and UI | [company-exposure-s3-publication-membership.md](company-exposure-s3-publication-membership.md) |
| S4 | Enabled-theme candidate discovery and optional Tavily search | [company-exposure-s4-discovery-search.md](company-exposure-s4-discovery-search.md) |
| S5 | Classifier grounding with pinned exposure claims | [company-exposure-s5-grounding.md](company-exposure-s5-grounding.md) |

## 1. Where S1 left the system

S1 delivered these plan tasks:

- 00–07 and 09;
- 14–16;
- 17A;
- 21A and 22A;
- 23A;
- 26A and 27A.

`docs/implementation/company-exposure-baseline.md` is the delivery record.

**Installed**

- **Research and workers**
  - One research worker on the `exposure_research` queue: `docker-compose.exposure.yml`, or `EXPOSURE_WORKER_ENABLED=true` in `start_celery.sh`.
  - Beat entries `company-exposure-work`, `-holds` and `-evidence-gc`.
- **Evidence and assessment**
  - Subscription dispatch through `OpenCodeGoKimi`, with one reservation per dispatch and `expired_uncertain` at period close.
  - Safe public acquisition, and a private content-addressed store with 5 GiB and 1 GiB-free limits plus garbage collection.
  - HTML and text-PDF passage preparation.
  - Claim verification: deterministic and downgrade-only.
  - Materiality and multi-document assessments.
  - Provider-free freshness holds.
- **US identity**
  - A US SEC adapter (`markets/us.py`) with CIK resolution through `USIssuerResolver`.
  - Reviewed links: `resolve-issuer`, `reject-link`, and the `issuer_link_review_pending` pause.
- **API and UI**
  - `POST /research-requests`, `GET /research-jobs/{id}` and `GET /research-jobs/{id}/preview`.
  - The preview is labelled `shadow_preview` and `authoritative_membership=false`.
  - The research panel and workspace on the Operations page, shown only when the themes feature is on.

**Schema.** The chain is `20260925_0058` → `20260925_0059` → `20260926_0060`, with parent `20260926_0058` from main. It is recorded in `company-exposure-migrations.json`. Membership, selection and grounding tables do not exist yet.

**Refused or reported as not installed**

- `EXPOSURE_RESEARCH_MODE=live` is refused with `live_mode_not_installed`.
- Non-US securities are refused with `market_not_installed`.
- Discovery requests are refused with `discovery_not_installed`.
- In-job gaps are reported as `market_adapter_not_installed` and `supplied_link_route_not_installed`.
- `INSTALLED_MARKETS = frozenset({"US"})` is in `services/company_exposure/research_requests.py`.
- `activation.py` knows only the stage `shadow_verify_us`.

**Gate manifest.** `backend/tests/required_company_exposure_cases.json` has one slice, `S1`. It lists 28 required cases and 14 named PostgreSQL invariants. Its `excluded_future_scope` field names the cases later slices must add:

| Case | Owning slice |
|---|---|
| E11 | S2 |
| E07, E08, E09, I04 | S3, Task 18 |
| R06–R11 | S3, Tasks 19 and 20 |
| I07, I08 | S5 |

I09 appears in S1 only at the unit layer. Its integration layer belongs to S5.

**CLI.** `backend/scripts/company_exposure.py` implements these commands:

- `status` and `job`;
- `resolve-issuer` and `reject-link`;
- `resume` and `process`;
- `inspect-holds` and `refresh-holds`.

### Open S1 items at handoff (not work for S2 to S5)

- **Postgres gate not rerun.** The local S1 PostgreSQL gate last ran before `1b9bbfa`. Commits `1b9bbfa`, `8564fc0`, `95c8046` and `3955fac` came from Codex review fixes. Their unit suites passed, but nobody reran the gate afterwards. Rerun it before merging:

  ```bash
  cd backend && DATABASE_URL=postgresql://ci:ci@localhost:5432/ci STOCKSCANNER_TEST_ALLOW_POSTGRES=1 \
    ./venv/bin/python scripts/run_required_company_exposure_postgres.py --slice S1 --artifact <path>
  ```

- **Stale test counts.** The baseline record says 430 passed, and the PR body shows an older count. Refresh both from that gate run.
- **No live runs.** No live SEC probe and no real model call have run. The implementation sandbox blocked SEC egress, and no credentials were available. The US live route is unverified until an operator runs `probe-market --market US` (S2 adds that command).
- **Reviewer not assigned.** The independent second corpus reviewer is still unassigned. This blocks the automatic-admission gate (Task 25B), not shadow use.

## 2. Dependency graph for the remaining work

Task numbers are stable references, not a serial order (plan Appendix C).

```text
S2  10 HK ─┐
    11 JP ─┼─ 25A corpus collection (during each market) ──► 25B admission gate ──┐
    12 TW ─┘                                                                      │
    08A language/vision adapters (04,06,07)                                       │
    08B isolated renderer (optional; off until F.4 probes pass)                   │
                                                                                  ▼
S3  18 membership ──► 19B membership serving ──► 20 compatibility ──► 21C/22B ──► automatic additions
    19A assessment serving ──► 21B reads ──► 22B dossier UI                        (needs 25B + 26B)
    23 publication triggers join 19
S4  13 Tavily (independent of markets) ─┐
    17B discovery (17A + an installed adapter) ─► 23B weekly schedule ─► activation step 5
S5  24 grounding (14,16,19B,20,21B,23A) ─► grounding validation ─► activation step 6
```

Parallelism the plan allows:

- S2's three market adapters and Task 13 can run in parallel with each other and with S3's Tasks 18 and 19A.
- Task 19A (assessment-only serving) does not wait for Task 18.
- Automatic membership additions need all of the following:
  - S2's four-market admission gate (25B);
  - S3's Task 20 compatibility;
  - Task 26B.

  Passing US alone never enables US automatic admission.

## 3. Rules that bind every remaining slice

These come from the plan's Global Constraints and Appendix A. They are restated here because every new task must satisfy them.

- **Defaults stay off.**
  - `EXPOSURE_RESEARCH_MODE=disabled` and `EXPOSURE_PAID_SEARCH_ENABLED=false`.
  - A credential alone enables nothing. No migration enables a provider, mode or stage.
  - The only LLM route is `OpenCodeGoKimi` / `kimi-k2.6` on the subscription route, with no metered or LiteLLM fallback.
- **Every authoritative write uses the same lock order (A7).** The order is:
  1. the taxonomy shared fence;
  2. the authority;
  3. the issuer or Social registry;
  4. the dossier, work or domain row;
  5. the outbox.

  This applies to research, hold, membership, identity and grounding writes. Provider, network and browser work never happens inside these locks. Quota transactions end before the fence. The existing `EconomicTaxonomyPublicationCoordinator` is the only pointer writer.
- **History is immutable.**
  - Revisions and sealed selections are append-only, with RESTRICT references and raw-SQL immutability tests.
  - Old generations and hashes never change. Generations without the exposure extension return `exposure_research_unavailable_for_generation`.
- **Never fabricate evidence.**
  - Research never creates `ThemeMention`, `ClaimAssignment`, `ThemeObservation` or development events.
  - Search snippets, translations, vision output, model output and classifier output are never primary leaves.
- **Migrations are allocated just in time, per owner.**
  - Owners: Task 18 `exposure_membership_decisions`, Task 19 `exposure_generation_extension`, Task 20 `exposure_origin_compatibility`, Task 24 `exposure_grounding_uses`.
  - Before each migration commit, fetch main, run `alembic heads` and the migration graph test, and reparent if main moved.
  - Record the filename, revision, parent, sha256 and task in `company-exposure-migrations.json`.
  - Never renumber a migration that has been released.
- **Tests carry markers.**
  - Each test carries `@pytest.mark.case("ID")` and `@pytest.mark.exposure_layer(...)`.
  - Each slice adds its own entry to `tests/required_company_exposure_cases.json`, with required case/layer pairs and named PostgreSQL invariants.
  - The gate fails on a missing, skipped, xfailed or zero-collected required node.
  - SQLite results never substitute for PostgreSQL.
- **Mocked retrieval is not a live probe.** Record the actual execution mode in each task. Live probes are opt-in (`--allow-network`, `--allow-subscription-calls`) and never run in CI.

## 4. S1 deferrals that return with their first caller

An over-engineering pass removed code that S1 never called (see the baseline record). Each piece comes back with the slice that first calls it.

| Deferred item | Returns in | First caller |
|---|---|---|
| Social attestation import (`import-identities`) | S3 | Task 20 issuer facade and activation step 1 |
| Authorized original and excerpt reads | S3 | Task 21B/21C evidence endpoints |
| Blocking-only safety check for automatic use | S3, and S5 again | Tasks 18/19 (additions), Task 24 (grounding) |
| Job pause and cancel | S3/S4 | Task 21C admin operations, Task 17B child jobs |
| Evidence-DAG (acyclic support) check | S5 | Task 24 `test_no_circular_support.py` |
| Paid-search and dispatch predicates | S4 | Task 13 |
| Non-overlapping segment sum | First adapter that sums disclosed segments (likely a JP or TW table in S2) | Task 15 extension |
| CLI `probe-market` | S2 | Task 27 four-market probes |
| CLI `set-mode`, `set-stage`, `prepare-preview`, `disable-acquisition`, `reconcile-compatibility` | S3 | Tasks 19–21, 27B |
| CLI `discover` | S4 | Task 17B |

## 5. Staged activation (Task 27B; spec §18)

Each step needs:

- administrator authorization;
- the gate hashes for that step;
- a disposable rehearsal first.

User approval of the plan is not the operator's activation command.

1. **Rehearse and import.** Apply the additive migrations in a rehearsal. Import issuer attestations, and import existing source constituents as unverified research leads. Compare grouping and membership before switching the issuer facade. (S3, Task 20)
2. **Shadow across four markets.** Run research `shadow` in all four markets, with an explicit subscription allowance and no paid search. (S2)
3. **Publish assessment reads.** Publish generation-bound assessment reads through the existing coordinator. Role policies stay unapproved. (S3, Tasks 19A/21B)
4. **Enable eligible additions.** Allow membership additions only for approved role policies and validated scopes. (S3, and needs S2's 25B gate)
5. **Enable bounded discovery.** Enable discovery for selected themes. Paid search stays off unless the operator enables it separately. (S4)
6. **Enable grounding.** Turn on targeted grounding last. (S5)

Each step records these artifacts in the handoff:

- source SHA, spec hash and final migration head;
- capability, test and evaluation hashes;
- model route policy;
- market probe reports;
- generation and checkpoint IDs;
- the operator principal.

## 6. Handoff checklist for any agent taking a slice

1. **Read the sources.** Read the spec, the plan task sections named in the slice file, and plan Appendices A, B, D and F.
2. **Branch from main.**
   - Work on a fresh branch from `origin/main`, after S1 merges.
   - Don't build on an unrelated feature branch.
   - Don't reset anyone's checkout.
3. **Test first.** For each task:
   - write the failing test first;
   - implement the change;
   - run the focused tests, the full `tests/unit/company_exposure` suite and the existing regressions, including `tests/unit/theme_evaluation` and the economic taxonomy gate when publication code changes;
   - from `backend/`, run `ruff check` and `ruff format` on new files only (files that exist on main are edited by hand);
   - run `npm run test:run` and `npm run build` for frontend changes.
4. **Add the slice to the gate manifest.** Add the slice's entry to `required_company_exposure_cases.json` and run `run_required_company_exposure_postgres.py --slice <S>` on PostgreSQL 16.
5. **Update the records.** Update:
   - `company-exposure-baseline.md` (or a per-slice delivery record) with the actual commands, counts, execution mode and unavailable capabilities;
   - the runbook `docs/runbooks/company-exposure-map.md`;
   - `CLAUDE.md` when queue or worker ownership changes.
6. **Get review.** Have an independent reviewer check the high-cost failures:
   - false admission;
   - circular support;
   - lock order;
   - old-generation mutation;
   - spend without enablement.
7. **Don'ts.**
   - Don't enable providers, paid search or stages.
   - Don't approve role policies.
   - Don't invent corpus labels.
   - Don't mark a plan checkbox done until its tests actually ran.

### Human and credential dependencies (none of these can be done by an agent)

| Dependency | Blocks |
|---|---|
| Second corpus reviewer; language-capable reviewers for JA and ZH | Task 25B automatic admission (S3 step 4) |
| Owner sign-off on role-policy templates (memory, cybersecurity, refining, tankers, copper) | Automatic additions per theme |
| OpenCode Go key and local allowance | Any real model call |
| SEC identifying User-Agent | US live probe |
| EDINET API key (optional) | JP EDINET route only; the keyless issuer route must still work |
| HKEXnews automation and retention permission review | HK title-search route; otherwise a truthful `permission_unavailable` gap |
| Host firewall rule (`DOCKER-USER`) and a seccomp-capable host | Browser rendering (08B) |
| Tavily key, cap, currency and max-charge costing policy | Paid search (S4), which is optional |
