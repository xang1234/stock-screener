# Task 10 report — Celery scheduling, shared gates, and operations health

## Outcome

Implemented the dedicated `social_ingestion` task boundary for refresh, saved-analysis resumption, and explicit source diagnostics. The six-hour clock entry runs at minute 17 in the configured Celery timezone. Delivery reads the shared database runtime and performs no provider/model work while mode is off or the provider is disabled; keeping the clock entry installed allows an administrator's database runtime change to take effect without restarting Beat.

Refresh and source tests share the ownership-token Redis lease `social-signals:provider-read:lease`. Manual refresh has a separate one-hour cooldown and returns HTTP 429 with `Retry-After`. Provider rate-limit/reauthentication cooldown is persisted per provider in Redis and enforced across provider reconstruction, so another worker cannot immediately bypass an xui challenge or Official API reset. Scheduled refresh is not blocked by the manual cooldown.

Scheduled generation IDs use the active 00:17/06:17/12:17/18:17 cadence slot, while manual and replay identities retain their exact time. Duplicate deliveries therefore resume one running generation, finish a staged generation, or return an already terminal result without another provider read. Budget deferral schedules a new replay generation at the database budget reset and uses saved observations without X access.

Explicit Test List execution:

- is limited to pending or disabled sources;
- claims queued work as running under the source-registry lock;
- reads only the selected list with a hard limit of five;
- writes only redacted status/count/provider/time plus immutable audit events;
- writes no content, metrics, run, snapshot, or pointer rows;
- retries lease contention only, leaving adapter-owned transport retry policy intact.

The Operations projection reports DB-authoritative mode/provider/version, run and collection/processing/history state, social freshness, source counts, formula/extraction/model labels, budget balance/reset, backlog ages/counts, shared lease/cooldown TTLs, and allowlisted reason codes. It never exposes raw provider text, stderr, credential paths, cookies, or session state.

## Verification

- Task-focused suite: **23 passed**, 19 inherited dependency/deprecation warnings.
- Scheduling/source-registry/cache/options regression suite: **73 passed**, 19 inherited warnings.
- Full affected Social suite: **524 passed**, 21 inherited warnings in 88.57s.
- `git diff --check`: clean before report creation.
- No live X request, private xui session, paid model request, production database, image build, registry push, merge, or deployment mutation was used.

## Deferred ownership

- Task 11 owns the authenticated Social read/admin API, including the dedicated HTTP 202 source-test and admin refresh routes. The existing generic task endpoint now has the shared 429 behavior but is not the final Social admin surface.
- Tasks 12–13 own the user queue and Operations source-management UI.
- Task 15 adds the `social_ingestion` Docker worker process and private xui runtime image.
- Task 17 retains disposable PostgreSQL producer-contention and full fixture-only end-to-end proof.
