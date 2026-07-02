# Event Radar

description: Scans the period between quarterly reports for material events that could affect the thesis — 8-K filings (acquisitions, divestitures, executive departures, restatements, debt actions, material contracts), structured deal-event data, executive changes, insider transactions, analyst rating shifts, and capital-structure events (debt, buybacks, dividends, offerings). Default scope: last 90 days, but configurable. Use to monitor a position list between earnings cycles or to investigate a recent unexplained move. Triggers on "what happened at [ticker] in the last 90 days", "any material events at [ticker]", "run event radar on watchlist", "anything notable since [date]", "exec departures or insider sales to flag".

Between quarterly reports, companies disclose material events that often matter more than the next earnings number. Most investors miss them because they are scattered across 8-Ks, press releases, executive-change filings, and 13D/G updates. This skill brings them into one view.

## Workflow

### Step 1: Frame the scope

- **Universe** — single ticker, named list, or the user's watchlist
- **Window** — default 90 days. Common alternatives: since the last earnings call; since a user-specified date; last 30 days for an active situation.

### Step 2: Pull the six event categories

| Category | What to pull | Materiality filter |
|---|---|---|
| **Material 8-Ks** | Items 1.01 (material agreement), 2.01 (acquisition), 2.05 (exit costs / impairments), 4.02 (non-reliance on prior financials), 5.02 (officer/director changes), 8.01 (Reg FD) | Drop routine items (annual-meeting results, regular dividend declaration); keep anything that changes the thesis. |
| **Deal events** | M&A, JV formation, partnership / strategic-investment announcements, definitive agreements signed or amended | All transactions ≥ 5% of market cap; all named partnerships with public peers. |
| **Executive changes** | CEO, CFO, COO, divisional-head, board arrivals and departures | Mid-year CFO departures; departures without a named successor; second resignation in the same function within 24 months; founder role changes. |
| **Insider transactions** | Open-market buys and sells by officers, directors, 10%+ owners | Cluster activity (3+ insiders in a window); single trades > 25% of insider's holdings; CFO net sells > 50% of position. |
| **Analyst rating shifts** | Consensus rating changes, price-target dispersion widening, downgrades from analysts with strong historical track records | Consensus moving > 1 notch; dispersion > 25% from low to high; net change in consensus PT vs. 90 days ago. |
| **Capital-structure events** | Debt issuance / refi, share-repurchase changes, dividend changes, secondary / convertible offerings, ATM activations | Anything that materially changes cost of capital or shareholder base. |

### Step 3: Score materiality

For each event, assign **High / Medium / Low** materiality, judged against:
- Size relative to the company (deal size as % of market cap; debt as % of total)
- Position in the thesis (a CFO departure at a turnaround thesis is High; at a stable compounder it may be Medium)
- Cluster pattern (a single insider sale is Low; clustered behavior is High)

### Step 4: Present

Group output by ticker, then within each ticker by materiality (High first):

| Date | Category | Description | Materiality | Implication |
|---|---|---|---|---|

Close with a one-line **Action items**: names where a `thesis-check` is now warranted, names where a `financial-forensics` recheck is warranted, names where positioning should be reviewed.

## Important Notes

- **Negative space is data.** A historically active filer going quiet for 60+ days is itself an event, especially heading into earnings.
- **Routine 8-Ks are noise.** Item 5.07 (annual meeting results), 7.01 (Reg FD where the content is benign), routine dividend declarations — these clutter the radar. Filter aggressively.
- **Single insider trades are not events.** Patterns are. Resist the urge to over-weight a one-off transaction at the High level.
- **Analyst rating changes follow more than they lead.** Use them as confirming evidence, not as the trigger.
- **A material agreement filed late is more interesting than a routine one filed on time.** Check filing date vs. event date when the 8-K item permits.
- **Cross-reference with the thesis.** Event radar surfaces items; whether they break the thesis is the `thesis-check` skill's job — flag, do not conclude.
