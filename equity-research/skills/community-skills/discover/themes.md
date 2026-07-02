# Thematic Discovery

description: Reads what the market is currently rewarding by clustering top-performing individual stocks bottom-up, without relying on sector or industry labels. Use when the user wants to know what's working right now, when they want to map a macro view to specific names, or when they suspect a theme is forming that the news hasn't caught up to yet. Triggers on "what themes are working", "what's the market rewarding", "is there a theme around X", "read the tape", "thematic sweep", or "current themes".

The market is always voting on something. The question is whether you're reading the votes.

Most investors read about themes in the news — long after the market has already priced them in. This approach works backwards: instead of starting from narrative, we start from what the numbers are actually rewarding. Which stocks are leading, what do they have in common, and where is the analyst community still cautious? That divergence is where early-stage theme discovery lives.

We can also work forward from a macro view you already hold. If you believe in reshoring, energy transition, defense modernization, or AI capex — we can map that thesis to specific companies, trace the value chain, and find where the thesis is most purely expressed and least efficiently priced.

## Workflow

### Step 1: Frame the request

Two modes:
- **Read the tape** — user has no prior view, wants to know what's working *right now*. Go to Step 2.
- **Express a view** — user has a thesis (reshoring, AI capex, etc.). Go to Step 4.

If the user asks "what's working" without specifying, offer to read the tape and proceed.

### Step 2: Rank stocks bottom-up

- Pull the top ~300 US-listed names by recent total return.
- Use **multiple windows**: trailing 1 month, 3 months, 6 months. Compare across them — themes mutate, and acceleration into the recent window is a different story than a 6-month leader that has rolled over.
- Exclude noise categories that distort thematic reads:
  - Biotech / small-cap pharma (binary clinical-trial moves)
  - Recent IPOs with <6 months of trading history
  - Reverse-merger and shell-related tickers
  - Single-product clinical-stage names
- **Do not group by GICS sector or industry.** Sector labels hide cross-sector themes — "power for AI" spans utilities, industrials, semis, and REITs; "GLP-1 second-order effects" spans food, apparel, medtech, and restaurants.

### Step 3: Cluster naturally

Read the names. Group by what the companies actually do — product, end market, customer base, supply-chain position. Look for clusters of 3+ names with a shared driver. Then verify with fundamentals: are the cluster's revenue growth, margin trend, and earnings revisions consistent with the price action, or is this a re-rating with no fundamental confirm yet?

**Report every cluster with a clear signal, including small ones (3–5 names).** A tight 4-name theme is often more actionable than a sprawling 30-name one — fewer crowded-trade dynamics, often earlier in the cycle. Do not filter small clusters out for being "too narrow."

### Step 4: Express-a-view mode

If the user brought a thesis:
1. State the thesis crisply (one sentence).
2. Map the value chain — direct beneficiaries, picks-and-shovels, second-order names.
3. For each candidate, check: fundamentals improving? Valuation already pricing it in? Analyst coverage saturated or sparse?
4. Surface the most purely-exposed and least efficiently priced expressions.

### Step 5: Present the read

For each theme:

**[Theme name] — [1-line description]**

| Field | Detail |
|---|---|
| Names (3–10) | Tickers + 1-line each |
| Shared driver | What's actually pulling these together |
| Window of strength | 1m / 3m / 6m where it's most visible |
| Fundamental confirm | Revenue accel, margin trend, EPS revisions |
| Narrative status | Already in headlines / forming / under-the-radar |
| Best expression | The 1–2 names most purely exposed |
| What kills it | Specific catalyst that would break the theme |

Close with a one-line **"Where I'd dig next"** — the cluster most worth a deeper screen, forensic check, or initiation.

## Pairing with a quantitative screen

Themes get sharper when stacked with a quant filter. Once a theme is on the table, we can layer screens — value (FCF yield, EV/EBITDA discount to peers), quality (ROIC, margin stability), or accelerating growth (revenue inflection, positive revisions) — to narrow a 10-name theme to the 2–3 best-priced expressions. Ask for "themes + value screen" or "themes + quality screen" and we'll run both passes.

## Important Notes

- Sector labels lie. The most interesting themes cross GICS boundaries; rank stocks individually and let the cluster shape itself.
- Multi-window beats single-window. A name leading the 6-month list but rolling over in the 1-month is a different story than one accelerating into the recent window.
- Small themes are not noise. A 4-name cluster with a clear shared driver and unanimous fundamental confirm is a real signal — don't drop it for being "too small."
- Crowded trades age fast. If the cluster is already on every sell-side dashboard, ownership is heavy, and short interest is collapsing, you're late.
- Theme without catalyst is a slow bleed. Even a well-spotted early theme needs a path to broader recognition — quarterly print, contract announcement, regulatory tailwind.
- Re-rank when the tape changes. Themes mutate. A list from six weeks ago is a starting point, not a current map.
