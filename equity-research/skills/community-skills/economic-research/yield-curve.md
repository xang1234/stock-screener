# Yield Curve & Recession Signals

description: Reads the US rate cycle and yield-curve dynamics for recession probability and cycle positioning. Pulls the current Fed funds rate, 2y / 10y / 30y treasury yields, the 10y-2y and 10y-3m spreads, inversion depth and duration, re-steepening pace, inflation breakevens, and high-yield credit spreads. Interprets the combined signal — inversion alone is not the trigger; inversion + re-steepening + credit-spread widening together is. Use for macro positioning, recession-probability framing, or as the macro backdrop for a sector thesis. Triggers on "yield curve", "rate cycle", "is the curve inverted", "re-steepening", "recession signals", "credit spreads", "inflation breakevens", "Fed funds", "what is the curve saying".

The 10y-2y spread is the most-watched recession indicator in fixed income. When it inverts (2-year yields exceed 10-year yields), it has preceded every US recession since 1955 — typically by 12 to 24 months. But the spread alone is not enough: the historically reliable recession signal is the *re-steepening* after inversion, confirmed by widening credit spreads.

## Workflow

### Step 1: Pull the current macro snapshot

- **Fed funds** — effective rate and target range
- **Treasury yields** — 3-month, 2-year, 5-year, 10-year, 30-year, current level + trailing 24-month series
- **Inflation expectations** — 5-year and 10-year breakevens vs. realized core CPI and core PCE
- **Credit spreads** — investment-grade and high-yield OAS, current and trailing 24-month series

### Step 2: Compute the key signals

| Signal | What to compute | Threshold |
|---|---|---|
| **10y-2y spread** | Current bps spread; depth and duration of any inversion in the last 24m | Inverted > 60 bps for > 6 months = historically robust pre-recession setup |
| **10y-3m spread** | Same. Often more reliable than 10y-2y for near-term recession | Inverted > 100 bps for > 6 months = strong signal |
| **Re-steepening pace** | Change in spread over the last 90 days after an inversion period | Sharp re-steepening (> 50 bps in 90 days) after sustained inversion = recession-onset signal |
| **Real yields** | 10-year nominal − 10-year breakeven | Real yields > 2% with the curve inverted = restrictive policy actively transmitting |
| **HY credit spread** | Current vs. 24m low; pace of widening | Widening through ~ 500 bps with rising vol = credit cycle turning |
| **IG-HY gap** | Difference between IG and HY OAS | Widening = quality flight; narrowing into a strong rally = late-cycle complacency |

### Step 3: Interpret the combined signal

The historically reliable recession signal requires three things together:
1. **Sustained inversion** for at least 6 months
2. **Re-steepening** off the inverted lows
3. **Credit spreads** widening, particularly HY OAS

Inversion alone is necessary but not sufficient. Re-steepening alone, in a never-inverted curve, is mid-cycle, not pre-recession. Credit confirming both is when the signal becomes actionable.

Also note the **inflation-expectations cross-check**: are 5-year breakevens stable / falling (soft-landing pricing) or rising despite restrictive policy (stagflation risk)?

### Step 4: Present

Open with a one-line read: "Late-cycle expansion / Pre-recession setup forming / Recession-onset signal active / Cycle reset complete." Then:
- Current snapshot table (Fed funds, 2y, 10y, 10y-2y, 10y-3m, breakevens, HY OAS) with 90-day and 12-month deltas
- Inversion timeline (when it inverted, depth, current state)
- Credit-confirm read (one paragraph)
- Implication for equity duration positioning, cyclical vs. defensive tilt, and any specific sector calls that follow

## Important Notes

- **Inversion is not a timing signal.** It says recession is more likely in the next 12–24 months; it does not say "sell now." The re-steepening is the more actionable trigger.
- **Each cycle has its own quirks.** The 2022–24 inversion was the longest and deepest in modern history yet did not produce an immediate recession — partly because the fiscal impulse offset the rate impulse. Apply judgment, do not pattern-match blindly.
- **Watch the term-premium decomposition.** When the 10y rises because of term premium (real or supply concerns) rather than rate expectations, the curve signal is muddier. Note when this is the case.
- **Credit and equities lead the curve at turns.** When HY OAS widens decisively, equities have usually already de-rated. The signal confirms, it does not predict.
- **Do not read a single day.** Curve readings are best assessed as 30-day rolling averages; daily noise misleads.
