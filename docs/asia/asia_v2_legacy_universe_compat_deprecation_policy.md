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

## Non-Goals

- No breaking change to existing clients during transition window.
- No hidden behavior drift: all legacy translation remains deterministic and documented.

