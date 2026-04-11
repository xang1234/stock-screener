# ADR ASIA-E1: SecurityMaster Identity Model (v1)

- Date: 2026-04-11
- Status: Accepted
- Issue: `StockScreenClaude-asia.1.1`

## Context

Current flows infer market identity from ticker suffix and US-biased assumptions. This causes inconsistent routing across ingestion, scanning, benchmark selection, and calendars.

## Decision

### Canonical Identity Contract

Security identity is explicit and centralized:

- `symbol` (canonical display/query symbol)
- `market` (`US`, `HK`, `JP`, `TW`)
- `exchange` (for example `NYSE`, `NASDAQ`, `HKEX`, `JPX`, `TWSE`, `TPEX`)
- `currency`
- `timezone`
- `local_code` (exchange-local identifier when different from canonical symbol)

### Resolution Rules

- SecurityMaster is the source of truth for market/exchange/currency resolution.
- Suffix heuristics are fallback-only and must not override explicit SecurityMaster records.
- Downstream services (benchmark, calendars, provider routing, validators) consume SecurityMaster outputs, not custom local inference.

### Compatibility Policy

- Legacy universe string paths remain temporarily supported via explicit adapters.
- Legacy requests map to typed market-aware payloads with observable translation behavior.

## Consequences

- Routing decisions become deterministic across subsystems.
- Hidden US assumptions are removed from core data/contract flows.
- Future market additions have one integration point for identity semantics.

## Rejected Alternatives

- "Keep per-service market heuristics": rejected due to drift and hard-to-debug disagreements.
- "Suffix-only identity forever": rejected because it cannot represent exchange/currency/timezone requirements reliably.
