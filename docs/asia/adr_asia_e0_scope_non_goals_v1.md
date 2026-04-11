# ADR ASIA-E0: Scope and Non-Goals (v1)

- Date: 2026-04-11
- Status: Accepted
- Issue: `StockScreenClaude-asia.1.1`

## Context

ASIA expansion introduces HK/JP/TW market support into a US-first system. Without explicit scope boundaries, implementation tends to drift toward implicit promises (for example full parity on unavailable data fields).

## Decision

### In Scope (v1)

- Add market-aware screening, analytics, and theme workflows for US/HK/JP/TW.
- Preserve US behavior while adding non-US functionality behind explicit gates.
- Standardize market identity, benchmark routing, and calendar semantics.
- Support graceful degradation for fields unavailable in non-US markets.
- Expose data-availability/provenance semantics to API/UI and launch gates.

### Out of Scope / Non-Goals (v1)

- Delivering full parity for non-US institutional ownership, insider transactions, and short-interest fields when source systems do not provide them.
- Building or buying a paid premium market-data stack as a prerequisite for v1 launch.
- Redesigning scanner algorithms beyond what is needed for market normalization and transparent fallback behavior.
- Replacing the entire provider ecosystem in one release.

### Policy

Unsupported non-US fields must be explicit and non-blocking where possible. No hidden coercion to fabricated values is allowed.

## Consequences

- Product and API behavior remains honest about data gaps.
- Launch decisions can be based on evidence and known constraints, not implicit assumptions.
- Future parity improvements can be layered through new ADR revisions.

## Rejected Alternatives

- "Block launch until full non-US parity exists": rejected due to long lead time and dependency on unavailable free data.
- "Pretend parity by silent defaults": rejected because it introduces false confidence and undermines trust.
