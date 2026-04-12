# ASIA Benchmark Registry Table v1

- Version: `2026-04-11.v1`
- Owner: `StockScreenClaude-asia.3.6`
- Policy: index-primary with configured ETF fallback; non-US paths must not fall back to SPY.

| Market | Primary Symbol | Primary Kind | Fallback Symbol | Fallback Kind | Notes |
|---|---|---|---|---|---|
| US | `SPY` | etf | `IVV` | etf | US baseline parity |
| HK | `^HSI` | index | `2800.HK` | etf | index semantics first |
| JP | `^N225` | index | `1306.T` | etf | Nikkei primary |
| TW | `^TWII` | index | `0050.TW` | etf | TAIEX primary |

## Runtime Contract

1. Resolution order is deterministic: `primary -> fallback`.
2. Cache keys/locks are symbol-scoped (`benchmark:<symbol>:<period>`).
3. Health surfaces include both primary and fallback cache presence.
4. Non-US mappings are market-local only; no implicit SPY fallback.
