# Isolated extraction runtime

`generate_extractions` is an optional, explicitly enabled replay of the existing
`ThemeExtractionService`. It uses the ordinary extraction prompt, 10,000-character
truncation, JSON parser, ticker cleaning, and company-name resolution, but creates a
transient `ContentItem` for each admitted `ExtractionInput`. It does not add a
content item, assign a cluster, create a mention, or write telemetry.

Generation needs all of the following before it makes a provider request:

- `THEME_EVAL_DATABASE_URL` must be an explicit PostgreSQL URL for a dedicated,
  non-local evaluation database. Its parsed host, effective port, and database name
  must differ from the supplied application URL. It may share a PostgreSQL server
  hostname with the application when the database name is distinct. A loopback or
  `localhost` alias is rejected because it is ambiguous operationally.
- The evaluation database must contain the repository schema needed by its reference
  tables (`app_settings` and `stock_universe`), use a SELECT-only role for this
  command, and contain
  `app_settings.theme_evaluation_database = isolated-v1`. The runtime never creates,
  resets, seeds, migrates, or connects to the application database.
- The supplied reference manifest must contain the SHA-256 digest of the canonical
  active `stock_universe` rows (`symbol`, `name`, and `is_active`) and the extractor
  settings it reads (`llm_extraction_model`, `reprocessing_max_age_days`, and
  `theme_policy_overrides`). The database must contain at least one active row. An
  optional `active_stock_count` must match too.
- The invocation must set `allow_model_calls=True`, request exactly
  `minimax/MiniMax-M2.7`, use one or both of `technical` and `fundamental` exactly
  once, and supply a positive document limit and code revision.
- `MINIMAX_API_KEY` and `OPENCODE_GO_API_KEY` must both be present in the process
  environment. Values are checked only for presence and never enter an artifact,
  error, log, or manifest.

The database validation runs first, in a PostgreSQL `REPEATABLE READ READ ONLY`
transaction. A missing evaluation URL therefore fails before provider or production
service imports. Reference reads and the transient production extractor share the
same read-only transaction.

`LLMService` obtains its key managers through the application's `RuntimeServices`
context. The evaluation runtime binds a fresh context-local `RuntimeServices`
container to the evaluation-session factory only while it constructs and uses the
extractor, then restores the caller's context even if extraction raises. It never
initializes the process-wide production runtime and does not connect to the
application database or Redis. If the extractor still has no primary LLM client,
generation stops with `extraction_primary_client_unavailable` before any provider
route or record can be created; this prevents an empty failed attempt from appearing
as a successful baseline.

The approved model route is bounded: MiniMax M2.7 is called as the primary with
`allow_fallbacks=False`, so LLMService cannot route to Z.AI/GLM. If that call fails,
the runtime makes one direct OpenCode Go Kimi K2.6 request. The fallback uses the
existing evidence Kimi endpoint, bearer header, disabled-thinking setting, bounded
response size, and no temperature parameter. It does not set a JSON-object response
format because the unchanged production extractor deliberately parses a JSON array.
The honest evaluation user agent and a generated, stable `x-opencode-session` header
are reused for the whole evaluation batch; an explicit session ID is accepted only as
a bounded printable header value. No other provider is attempted.

Each actual provider attempt gets its own `RecordingLLM` proxy. A primary failure
and a Kimi success therefore appear as two sanitized call records; unknown returned
model/provider values remain null. Failures use stable codes and do not preserve
DSNs, provider exception text, request bodies, source text, or credentials.

The production extractor normally emits Redis telemetry in a `finally` block. The
evaluation runtime applies a short, locked no-op hook only around that extractor call.
This avoids a production telemetry write without changing production telemetry code.
Its security-master resolver is a new local deterministic resolver and its database
session is always the evaluation session.

Preflight checks only the isolated database and reference prerequisites. Generation
separately validates extractor startup before any provider route is created. A run
report records the frozen manifest and any provider-backed extraction results.
