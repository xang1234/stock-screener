# ADR ASIA-E3: MarketCalendarService Engine and Semantics (v1)

- Date: 2026-04-11
- Status: Accepted
- Issue: `StockScreenClaude-asia.1.1`

## Context

Current scheduling/freshness checks assume NYSE/US-Eastern semantics. ASIA rollout needs reliable session-day logic across US/HK/JP/TW, including lunch-break/session differences.

## Decision

### Calendar Engine

Use a unified `exchange_calendars` implementation for US/HK/JP/TW in MarketCalendarService.

### Canonical Calendar Mapping

MarketCalendarService must use the following canonical `exchange_calendars` IDs:

- US -> `XNYS`
- HK -> `XHKG`
- JP -> `XTKS`
- TW -> `XTAI`

No alias IDs or market-specific substitutes may be used in production without a superseding ADR revision.

### Service Contract

MarketCalendarService provides deterministic primitives:

- expected trading date for market at evaluation time
- session open/close boundaries
- market-specific freshness reference date
- trading-day determination for schedulers and health checks

### Semantics Policy

- Calendar logic is centralized; no subsystem-specific calendar math.
- US parity behavior must be preserved under migration.
- HK/JP lunch-break/session conventions and TW session-day correctness are first-class requirements.
- All schedule/freshness comparisons normalize timestamps to UTC internally.
- User-facing and audit/log outputs must include the market-local timezone label for rendered session boundaries:
  - US: `America/New_York`
  - HK: `Asia/Hong_Kong`
  - JP: `Asia/Tokyo`
  - TW: `Asia/Taipei`

## Consequences

- Freshness and scheduler behavior become market-correct.
- False stale/fresh outcomes caused by US assumptions are reduced.
- Launch gates can validate date/session behavior by market.

## Rejected Alternatives

- "Mixed calendar engines by market": rejected due to maintenance complexity and semantic drift.
- "Custom hand-rolled calendars": rejected due to error risk and ongoing maintenance burden.
