# Economic Taxonomy V1 cutover and recovery

This runbook is the production control for moving theme authority through
`legacy -> shadow -> dual -> economic`. It implements the catch-up/barrier
protocol in the [architecture](../superpowers/specs/2026-09-20-economic-taxonomy-design.md),
the [implementation plan](../superpowers/plans/2026-09-21-economic-taxonomy.md),
and [ADR-0005](../adr/0005-economic-taxonomy-snapshots-and-interpretations.md).
The completed disposable-database exercise is recorded in the
[2026-09-21 rehearsal artifact](artifacts/economic-taxonomy-rehearsal-2026-09-21.md).
Deployment readiness gaps, legacy retirement criteria and the legacy
dependency inventory are in the
[2026-10-01 readiness audit](artifacts/economic-taxonomy-readiness-audit-2026-10-01.md).

## Safety model

Catch-up happens outside the publication transaction. The final operation is a
short final compare-and-set barrier under the exclusive writer fence. It checks
the sealed generation-input manifest, expected parent generation,
semantic-invalidation revision, snapshot hashes, projection staging, and reader
capability before changing any reader pointer. No provider request, worker wait,
or compatibility drain occurs while that fence is held.

The cutoff is a committed revision manifest, not a maximum source ID. Let `C`
mean the exact tuple set in the sealed generation-input manifest. The rule is
exact: ordinary revisions after C are backlog, not a stop condition; structural changes after C
invalidate the candidate. The same coordinator and barrier are used for shadow,
dual, economic, routine data-only publication, and rollback.

Use a distinct database backup and deployment identifier in every command:

```bash
cd backend
export TARGET_DATABASE_URL='postgresql://operator:REDACTED@db.example/stockscanner'
export RELEASE_TEST_DATABASE_URL='postgresql://ci:REDACTED@test-db.example/economic_taxonomy_release'
export STOCKSCANNER_TEST_ALLOW_POSTGRES=1
export TAXONOMY_ACTOR='release-artifact:operator'
export RELEASE_DIR='../release-artifacts/economic-taxonomy'
mkdir -p "$RELEASE_DIR"
read -rsp 'Taxonomy admin key: ' TAXONOMY_ADMIN_KEY && echo
export TAXONOMY_ADMIN_KEY
```

The deployment must already configure `ADMIN_API_KEY` and
`ADMIN_PRINCIPAL_ID`. The publication command compares the separately supplied
`TAXONOMY_ADMIN_KEY` credential to that configured key and records the configured
principal; it never accepts a caller-supplied audit identity. Do not put
credentials in the release artifact or command arguments.
`RELEASE_TEST_DATABASE_URL` must name an empty, disposable database: the pytest
fixture drops and recreates application tables for every test. Never point the
required PostgreSQL gate at `TARGET_DATABASE_URL`.

## Mandatory stop conditions

Stop without changing authority when any check reports one of these conditions:

- unresolved allocation or unresolved Social conflict;
- parent generation mismatch or semantic-invalidation mismatch;
- pending required prior-generation delivery;
- unstaged candidate projection;
- stale snapshot hash or any component artifact-integrity mismatch;
- missing reader capability or incompatible migration/reader contract;
- benchmark failure;
- unauthorized principal;
- skipped required PostgreSQL node, xfail, xpass, missing node, or non-PostgreSQL test database;
- `processor_unconfigured`, missing provider credentials, or an Economic Taxonomy worker that cannot complete a synthetic extraction/review request;
- a non-empty held structural-revision set; or
- rollback state `recovery_failed`.

Do not bypass a stop by editing an append-only row or pointer. Fix the input,
recapture outside the publication transaction, and prepare another generation.

## 1. Backup, migrate, and establish the release evidence

Take a database snapshot with the platform backup procedure, record its immutable
identifier, then run:

```bash
DATABASE_URL="$TARGET_DATABASE_URL" ./venv/bin/alembic upgrade head
DATABASE_URL="$TARGET_DATABASE_URL" ./venv/bin/alembic current
DATABASE_URL="$RELEASE_TEST_DATABASE_URL" STOCKSCANNER_TEST_ALLOW_POSTGRES=1 \
  ./venv/bin/python scripts/run_required_economic_taxonomy_postgres.py \
  --report "$RELEASE_DIR/postgres-gate.json"
```

The JSON must have `exit_code: 0`; all required nodes must be present and every
outcome must be `passed`. A missing or skipped required PostgreSQL node is a hard
stop. The runner independently connects to the configured database and records
its PostgreSQL identity.

All remaining commands target the migrated deployment database:

```bash
export DATABASE_URL="$TARGET_DATABASE_URL"
```

## 2. Seed the governed V1 snapshot

Run this once, while `taxonomy_authority.processing_taxonomy_version_id` is
null. It creates the governed draft dimension catalog and makes it the
processing taxonomy. Keep this snapshot in draft only while reviewed legacy
dispositions, destinations, and split allocations are being added.

The seed runs under the exclusive writer fence and handles each authority state:

| Authority row | Seed result |
|---|---|
| None | Creates one in `legacy` mode at epoch 1 |
| Implicit legacy row (no processing taxonomy) | Adopts it and keeps its epoch. The first fenced legacy write, such as a theme metrics run, creates this row. |
| Already has a processing taxonomy | Prints `STOP:` and writes nothing |
| Not `legacy` mode, or writes fenced | Prints `STOP:` and writes nothing |

```bash
./venv/bin/python scripts/seed_economic_taxonomy.py --actor "$TAXONOMY_ACTOR"
```

Record the draft UUID it prints. Never seed a second root while this draft
exists. On `STOP:`, inspect the authority row instead of seeding again.

## 3. Capture and review legacy migration inputs

Use the authenticated migration service to capture the legacy dataset against
the current draft migration target and build a sealed `TaxonomyMigrationRun`.
Store the captured JSON and its hash in the release artifact. Every legacy
identity needs exactly one reviewed disposition; `split_required` additionally
needs an allocation or reviewed exclusion for every current claim. Zero, one,
or multiple destinations are valid according to the disposition.

Review operations must call
`EconomicTaxonomyMigrationService.review_disposition()` or `review_split()` with
an `AdminPrincipal` holding `taxonomy:review`, the currently observed epoch, a
non-empty reason, explicit destination UUIDs, and explicit claim allocations.
Commit each reviewed batch. Do not edit mapping tables directly.

Then run `replay_manifest(run_id)` and record its semantic hash. The following
read-only check must return no rows before continuing:

```bash
./venv/bin/python - <<'PY'
from sqlalchemy import select
from app.database import SessionLocal
from app.models.economic_taxonomy_runtime import TaxonomyMigrationRun

with SessionLocal() as db:
    incomplete = db.scalars(select(TaxonomyMigrationRun).where(
        TaxonomyMigrationRun.coverage_complete_cache.is_(False)
    )).all()
    if incomplete:
        raise SystemExit(f"STOP: unresolved allocation/disposition runs: {[str(x.id) for x in incomplete]}")
    print("migration coverage complete")
PY
```

Resolve every Social association conflict through the reviewed Social adapter.
Conflicting accepted/rejected legacy decisions remain review-only and must not
be collapsed by processing order.

After coverage, replay, and conflict review are complete, seal the migration
target. `MIGRATION_RUN_ID` must be the reviewed run just replayed:

```bash
export MIGRATION_RUN_ID='00000000-0000-0000-0000-000000000000'
MIGRATION_RUN_ID="$MIGRATION_RUN_ID" ./venv/bin/python - <<'PY'
import os
from uuid import UUID
from app.database import SessionLocal
from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy_runtime import TaxonomyMigrationRun

with SessionLocal() as db:
    run = db.get(TaxonomyMigrationRun, UUID(os.environ["MIGRATION_RUN_ID"]))
    if run is None or not run.coverage_complete:
        raise SystemExit("STOP: migration coverage incomplete")
    sealed = EconomicTaxonomyRepository(db).seal_draft(run.taxonomy_version_id)
    db.commit()
    print(sealed.id, sealed.semantic_hash, sealed.artifact_integrity_hash)
PY
```

Record all three values. Never reopen a sealed snapshot.

## 4. Enter shadow, catch up, and benchmark

Admit the news, RSS, Substack, Reddit and legacy-X content ingested before this
release, so shadow comparisons include it. New items are admitted at ingest.

X posts are handled differently from other content:

- **Which X posts are backfilled:** only those with a recorded legacy observation
  and a tweet ID recoverable from their status URL. Posts that Social collected,
  or that have no status URL, are not backfilled.
- **Family:** admitted X posts join Social's `x:post:<tweet_id>` family.
- **Supersession:** a complete, non-archive Social capture of the same post
  supersedes the legacy packet. Archive and partial captures stay review-only.
  Either way, the legacy source's lens grants carry over.

Re-running is safe, and `--after-id` resumes from the `last_id` a previous run
printed:

```bash
./venv/bin/python scripts/backfill_content_economic_evidence.py
```

Then register the tested reader capability. This command reruns the PostgreSQL
gate, backend reader contract, focused frontend tests, and the production build
before inserting the immutable capability manifest:

```bash
DATABASE_URL="$RELEASE_TEST_DATABASE_URL" \
CAPABILITY_DATABASE_URL="$TARGET_DATABASE_URL" \
./venv/bin/python scripts/run_required_economic_taxonomy_postgres.py \
  --register-reader-capability --verified-by "$TAXONOMY_ACTOR" \
  --report "$RELEASE_DIR/reader-capability.json"
```

Read `reader_capability_manifest_id` from the JSON:

```bash
export READER_CAPABILITY_ID='00000000-0000-0000-0000-000000000000'
./venv/bin/python scripts/publish_economic_taxonomy.py \
  --capability-id "$READER_CAPABILITY_ID" --mode shadow
```

This publication verifies the reader capability and performs its final
compare-and-set barrier inside the coordinator. In shadow mode, legacy remains
the read and write authority; comparison projections are not live state.

Shadow publication needs no benchmark. Its generation records `benchmark: null`.
Its compatibility projections stay in shadow scope, so delivery never writes
them to the legacy tables. Every other generation needs a benchmark, rollback
included, because rollback publishes the current processing taxonomy.

The shadow generation is still a readable preview. The Themes page economic
catalog and the review routes serve it, unbenchmarked, and the benchmark export
below reads it through the shadow reader. Legacy tables and legacy readers are
untouched. Shadow publication is refused once authority is `dual` or
`economic` (`shadow_after_dual_unsupported`); to go back, roll back to legacy.

Now verify the processing pipeline. Processing is skipped in legacy mode, so
this check cannot run before shadow. The worker builds the default Economic
Taxonomy pipeline from a sanctioned extraction-provider credential and the
current processing taxonomy. The default binds extraction, claim review,
resolution, and Social's durable budget adapter. A deployment may replace it
through `configure_economic_taxonomy_pipeline(...)`. Dispatch one synthetic
request and require `processed: 1`. The worker fails closed as
`processor_unconfigured` when neither the default nor an override can be
built. Do not start catch-up until a synthetic request succeeds.

Run discovery and processing repeatedly through the deployed Celery workers:

```bash
./venv/bin/celery -A app.celery_app call \
  app.tasks.economic_taxonomy_tasks.discover_economic_taxonomy_work \
  --args='[500]'
./venv/bin/celery -A app.celery_app call \
  app.tasks.economic_taxonomy_tasks.process_economic_taxonomy_work \
  --args='[500]'
```

Repeat until eligible work is complete or durably held for review. This is
catch-up outside the publication transaction. Provider calls must not occur
inside the Social projection transaction or the publication fence.

Republish shadow after catch-up. Readers serve the published shadow
generation, so they still show the state before catch-up. The `refresh` task
skips every mode except `economic`, so it does not republish shadow. The
benchmark export below must read current output:

```bash
./venv/bin/python scripts/publish_economic_taxonomy.py \
  --capability-id "$READER_CAPABILITY_ID" --mode shadow
```

The benchmark is registered against the anchor. The anchor is the last
taxonomy version a seed or reviewed operation created. Versions created by
`system:economic-taxonomy-refresh` inherit its benchmark. That covers processor
catch-up and the automatic lifecycle job. The `system:` prefix is reserved, so
the seed refuses such an `--actor` and settings reject such an
`ADMIN_PRINCIPAL_ID`. After catch-up, the anchor is the sealed migration
target unless a reviewed operation ran since. Use the anchor's semantic hash as
`RECORDED_SEMANTIC_HASH` below.

The repository has no exporter for `benchmark-actual.json` yet. The
`benchmark_cases` in `contract_cases.json` hold only `required` and `forbidden`
outcome labels. They hold no inputs. An exporter needs an input scenario for
each case before it can produce this file.

Create `$RELEASE_DIR/benchmark-actual.json` by running every named
`benchmark_cases` fixture through the deployed shadow processor and exporting
the shadow reader's observed identities, relationships, assignments,
interpretation selections, and retractions. This is a reviewed release input,
not a file that may be synthesized from each fixture's `required` section (that
would make the benchmark tautological). The export must contain the anchor's
semantic hash, `economic-taxonomy-v1` policy bundle, and exactly one
complete outcome for every fixture name. Preserve the case-to-source evidence
map alongside it.

Validate that export with the fail-closed contrast benchmark:

```bash
./venv/bin/python scripts/run_economic_taxonomy_benchmark.py \
  --fixtures tests/fixtures/economic_taxonomy/contract_cases.json \
  --actual "$RELEASE_DIR/benchmark-actual.json" \
  --taxonomy-hash 'RECORDED_SEMANTIC_HASH' \
  --policy-bundle 'economic-taxonomy-v1' \
  --report "$RELEASE_DIR/benchmark-report.json"
```

An absent case, wrong taxonomy/policy hash, forbidden result, or benchmark
failure is a stop.

Register the passed report in the target database. Every publication except
shadow fails closed unless this append-only result matches the anchor's
semantic hash and the `economic-taxonomy-v1` policy bundle:

```bash
./venv/bin/python scripts/register_economic_taxonomy_benchmark.py \
  --report "$RELEASE_DIR/benchmark-report.json" \
  --verified-by "$TAXONOMY_ACTOR"
```

Record the printed benchmark-result UUID with the release evidence. A reviewed
operation creates a new anchor. It needs a new report and registration before
the next generation that is not shadow, a rollback included. A policy change
does too. Processor catch-up does not.

## 5. Enter dual and drain required compatibility work

After reviewed mappings, split allocations, Social conflicts, development
provenance, benchmark, and shadow comparisons pass:

```bash
./venv/bin/python scripts/publish_economic_taxonomy.py \
  --capability-id "$READER_CAPABILITY_ID" --mode dual
```

Dual still serves legacy readers while both producer routes use the shared
write fence. Drain published compatibility events by repeatedly dispatching:

```bash
./venv/bin/celery -A app.celery_app call \
  app.tasks.economic_taxonomy_tasks.deliver_taxonomy_outbox \
  --args='[500]'
```

Continue until a delivery invocation reports `claimed: 0`, `failures: 0`, and
`rollback_state: ready`. Checkpoints enforce target-side ordering, so delivery
of revision 1 after revision 2 is a stale no-op and mirror delivery cannot emit
a recursive event back to its origin.

## 6. Prepare and publish economic authority

1. Finish the last catch-up outside the publication transaction.
2. Capture a short cutoff by invoking the publisher. The coordinator briefly
   takes the exclusive writer fence to seal C, then releases it.
3. The command builds the interpretation, metrics, UI snapshot, benchmark
   result, and complete candidate projection set outside the final lock.
4. Its final compare-and-set barrier retakes the fence, verifies the parent,
   semantic-invalidation revision, generation-input manifest, hashes, reader
   capability, prior delivery acknowledgement, and staged projections, then
   switches all reader pointers and the epoch in one commit.

```bash
./venv/bin/python scripts/publish_economic_taxonomy.py \
  --capability-id "$READER_CAPABILITY_ID" --mode economic
```

Record the printed generation UUID. If the command reports a changed parent,
processing head, semantic invalidation, or manifest, the candidate is abandoned.
Release the lock, catch up again, and rerun the command. Do not reuse an
abandoned generation.

## 7. Verify publication and enable routine refresh

Verify authority, component hashes, reader pointers, rollback state, and
projection delivery:

```bash
./venv/bin/python - <<'PY'
from sqlalchemy import select
from app.database import SessionLocal
from app.models.economic_taxonomy_runtime import (
    GenerationInputManifest, ReaderSnapshotBundle, ReaderSnapshotPointer,
    ServingGeneration, TaxonomyAuthority,
)

with SessionLocal() as db:
    authority = db.get(TaxonomyAuthority, 1)
    generation = db.get(ServingGeneration, authority.serving_generation_id)
    manifest = db.get(GenerationInputManifest, generation.generation_input_manifest_id)
    snapshot = db.get(ReaderSnapshotBundle, generation.reader_snapshot_bundle_id)
    pointers = db.scalars(select(ReaderSnapshotPointer)).all()
    print({
        "mode": authority.mode, "epoch": authority.authority_epoch,
        "generation": str(generation.id), "generation_hash": generation.semantic_hash,
        "manifest": str(manifest.id), "manifest_hash": manifest.semantic_hash,
        "snapshot_hash": snapshot.semantic_hash,
        "rollback_state": authority.rollback_state,
        "pointers": {p.reader_key: str(p.reader_snapshot_bundle_id) for p in pointers},
    })
    assert authority.mode == "economic"
    assert all(p.reader_snapshot_bundle_id == snapshot.id for p in pointers)
PY
```

Exercise catalog, detail, ranking, stock-summary, digest, MCP, Social, and admin
review readers. They must resolve the same serving generation. After outbox
drain, rollback state must become `ready`.

The existing Celery beat entry runs automatic routine publication every minute.
It coalesces accepted routine revisions to no more than one generation per five
minutes and reuses the deployed reader capability. Structural revisions
(dimension proposals, ambiguous identities, structural operations, migration
and rollback-recovery revisions, and any unrecognized kind) stop automatic
publication with `status: held` until an operator publishes a reviewed
generation. Per-request outcomes (`provider_invalid_schema`,
`provider_terminal_failure`, `provider_outcome_uncertain`,
`claim_review_required`) do not stop publication: the failed request
contributes no interpretation, its lineage keeps its previous completed
attempt, and the result lists the revisions in `attention_revision_ids` with a
warning log. Find the affected work with
`SELECT id, source_lineage_id, completion_code FROM economic_processing_requests WHERE status = 'terminal_failure'`.
Monitor `refresh_economic_taxonomy_generation`; `backlog_preserved: true` is
retryable, not data loss.

## 8. Rollback and rollback_recovery

When `rollback_state` is `ready`, normal rollback is one coordinated generation
publication back to legacy authority:

```bash
TAXONOMY_ACTOR="$TAXONOMY_ACTOR" ./venv/bin/python - <<'PY'
import os
from app.config import settings
from app.database import SessionLocal
from app.services.economic_taxonomy_publication import EconomicTaxonomyPublicationCoordinator
from scripts.publish_economic_taxonomy import _publication_principal

principal = _publication_principal(
    settings, os.environ.get("TAXONOMY_ADMIN_KEY"))
result = EconomicTaxonomyPublicationCoordinator(SessionLocal).rollback(
    principal=principal, reason="operator requested rollback")
print(result.id, result.authority_epoch, result.mode, result.rollback_state)
PY
```

If publication reports `temporarily_unavailable`, first drain compatibility
events. Calling rollback while required compatibility delivery is unhealthy
starts `rollback_recovery`: writers are fenced, missing legacy projections are
rebuilt with ordered checkpoints, the recovery revision is recorded, and only
then is a legacy generation published. A recovery failure leaves
`writes_fenced=true`, `rollback_state=recovery_failed`; keep the service in
read-only incident mode, preserve the database, repair the target adapter, and
rerun rollback with the same reviewed reason. Never clear the fence manually.

After rollback, compare legacy reader output with the pre-cutover baseline and
the compatibility projection for the same source cutoff. Record reader
equivalence, new epoch, rollback generation, and final checkpoints.

## Rehearsal record requirements

Before production, repeat the full procedure on a disposable PostgreSQL clone
containing a Refining split, duplicate and conflicting Social associations, a
Social-native development, corrected-to-empty evidence, UI snapshots, and an
out-of-order compatibility event. Record:

- database and backup identifiers;
- migration run and taxonomy UUIDs plus semantic/artifact hashes;
- interpretation set, manifest, metrics, snapshot, and generation UUIDs/hashes;
- projection checkpoints and authority epoch after each mode;
- reader capability UUID and consumer-test hash; and
- post-rollback reader equivalence.

## Deferred Work Boundary

The items below are intentionally excluded from V1.
None of these capabilities is required for V1 cutover. They are not placeholders for opportunistic work,
and no implementation may begin until its listed decisions, evidence gate, and
separate design/ADR are approved.

### Directional economic driver pack

V1 ships `fundamental_attention`, an unsigned measure of the amount and recency
of selected fundamental evidence. It does not infer improving/deteriorating
direction. A later driver pack may add directional observations, transmission
paths, issuer materiality, and aggregation only after all decisions below are
settled:

1. **Driver vocabulary:** governed demand, pricing, volume, capacity,
   cost/input, margin, regulation, financing, and supply categories, including
   mixed, neutral, and unknown direction.
2. **Attribution grain:** whether direction belongs to a claim, company,
   constituent, theme, or development, and how a claim spanning grains avoids
   duplication.
3. **Materiality:** required revenue/profit exposure data, estimates, confidence
   model, and versioning.
4. **Corroboration/dependence:** publisher ownership, shared upstream sources,
   syndication, copied text, correlated analysts, and the boundary between
   deduplication and statistical independence.
5. **Point-in-time semantics:** roles of `published_at`, UTC `available_at`,
   effective period, revision, and restatement in live views and backtests.
6. **Aggregation:** polarity scale, weights, decay, conflict netting,
   missing-data behavior, and absolute versus benchmark-relative scores.
7. **Evaluation:** labeled fixtures/dataset, reviewer workflow, acceptance
   thresholds, calibration, and drift monitoring.
8. **Provider/budget:** extraction schema and model versions, fallback behavior,
   stable attempt keys, and incremental budget policy.
9. **Persistence/publication/API:** append-only driver observations and
   decisions, correction-to-empty behavior, generation-input pinning, API names,
   and compatibility with historical `fundamental_attention` generations.

**Entry gate:** an approved driver-pack design and ADR answering all nine items,
plus a representative labeled dataset and acceptance criteria. Until then, do
not reserve a `fundamental_momentum` field, alias, or UI label.

### Other deferred capabilities

| Deferred capability | Decisions and evidence required before work starts |
|---|---|
| Remove legacy Theme/Social storage or compatibility writes | Rollback-support end date; retention/export policy; machine-checked consumer inventory proving zero legacy reads; compatibility health history; explicit irreversible-cleanup approval. |
| Automatically approve structural changes | Accountable owner; confidence thresholds; reversible remediation; blast-radius limits; audited evaluation for false-positive cost covering new dimensions, merges, splits, mechanism changes, and retirement. |
| Recursive ontology propagation | Maximum depth; cycle behavior; attenuation and deduplication rules; provenance/evidence display; product evidence that V1 one-hop derivation is insufficient. |
| General asset master beyond `StockUniverse` | Supported asset classes; identifier authority; corporate-action lifecycle; market coverage; ownership and migration plan; amendment to the ADR boundary with StockUniverse. |
| Statistical source independence | Operational dependence definition; publisher/source graph data; syndication detection; refresh cadence; validation that the signal improves decisions. Distinct source-family counts remain deduplication only. |
