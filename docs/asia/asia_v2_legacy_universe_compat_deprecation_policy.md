# ASIA v2 Legacy Universe Compatibility Adapter and Deprecation Policy

- Date: 2026-04-11
- Scope: `StockScreenClaude-asia.2.4`
- Applies to: `POST /api/v1/scans` requests that send legacy `universe` strings instead of typed `universe_def`.

## Goal

Keep existing clients functional while making migration to typed `universe_def` explicit, observable, and time-bounded.

## Compatibility Adapter Contract

1. If `universe_def` is present, it is authoritative.
2. If `universe_def` is absent, the legacy `universe` string is translated through the compatibility adapter into `UniverseDefinition`.
3. Legacy-path requests must emit deprecation telemetry in HTTP headers and server logs.
4. Unknown legacy values are only accepted when `symbols` are provided; they map to typed `custom` universes to avoid silent behavior loss.

## Legacy-to-Typed Mapping

| Legacy request | Typed `universe_def` equivalent |
|---|---|
| `all` | `{"type":"all"}` |
| `nyse` | `{"type":"exchange","exchange":"NYSE"}` |
| `nasdaq` | `{"type":"exchange","exchange":"NASDAQ"}` |
| `amex` | `{"type":"exchange","exchange":"AMEX"}` |
| `sp500` | `{"type":"index","index":"SP500"}` |
| `market:us` | `{"type":"market","market":"US"}` |
| `market:hk` | `{"type":"market","market":"HK"}` |
| `market:jp` | `{"type":"market","market":"JP"}` |
| `market:tw` | `{"type":"market","market":"TW"}` |
| `custom` + `symbols` | `{"type":"custom","symbols":[...]}` |
| `test` + `symbols` | `{"type":"test","symbols":[...]}` |
| unknown + `symbols` | coerced to `{"type":"custom","symbols":[...]}` with deprecation warning |

## Deprecation Timeline

- Deprecation announced: **2026-04-11**
- Sunset date for legacy `universe` strings: **2026-10-31**
- No earlier than 2026-10-31 may the API remove legacy parsing behavior.

## Legacy-Path Observability

When legacy path is used, the API returns:

- `Deprecation: true`
- `Sunset: Sat, 31 Oct 2026 00:00:00 GMT`
- `X-Universe-Compat-Mode: legacy`
- `X-Universe-Legacy-Value: <normalized_legacy_value>`
- `X-Universe-Migration-Hint: <typed_equivalent_hint>`

Server logs include a structured warning containing:

- legacy value
- typed key produced by adapter
- deprecation date
- sunset date

### Legacy Compatibility Telemetry Counters

Each legacy-path scan creation also increments Redis counters (DB 2) so
operators can watch remaining legacy callers trend toward zero before
sunset. The counter writes are best-effort and never fail the request
path when Redis is unavailable.

| Redis key | Purpose |
|---|---|
| `universe_compat:legacy_total` | Monotonic count of legacy-path scan creations across all buckets. |
| `universe_compat:legacy:<value>` | Per-value count (e.g. `universe_compat:legacy:nyse`, `universe_compat:legacy:sp500`). |
| `universe_compat:legacy_last_seen_ts` | Unix timestamp of the most recent legacy-path request. |

Unknown/oversized/whitespace legacy values are bucketed under
`universe_compat:legacy:unknown` to bound the keyspace.

Diagnostic snapshot in Python:

```python
from app.services.universe_compat_metrics import get_legacy_universe_counts
print(get_legacy_universe_counts())
# {"total": 42, "by_value": {"all": 30, "nyse": 10, "sp500": 2}, "last_seen_ts": 1793650000}
```

### Related Migration Guidance

For the full E8 (API and Frontend Multi-Market Productization) migration reference covering T1–T5 breakage tiers, before/after payloads, and the client adoption checklist, see the [E8 API / Client Migration Guide](./asia_v2_e8_api_migration_guide.md).

### Typed-first Response Contracts (T1, 2026-04-13)

`POST /api/v1/scans`, `GET /api/v1/scans`, and `GET /api/v1/scans/{id}/status`
now return a nested `universe_def` object. The legacy flat fields
(`universe`, `universe_type`, `universe_market`, `universe_exchange`,
`universe_index`, `universe_symbols_count`) have been removed from the
`ScanListItem` response shape — clients must read `universe_def.*` instead.
Request-side legacy input (`{"universe": "nyse"}`) continues to work until
the 2026-10-31 sunset.

## Non-Goals

- No breaking change to existing clients during transition window.
- No hidden behavior drift: all legacy translation remains deterministic and documented.

