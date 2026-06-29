# Thesis Check

description: Structured review of whether an existing position's thesis is still intact. Recovers the original buy reasons, then tests them across fundamental tracking, qualitative shifts (competition, customers, regulation, management), and valuation drift. Produces an **Intact / Improved / Weakening / Broken** verdict with specific evidence. Use after each quarterly print, after a 10-Q, after a material 8-K, or any time something at the company or sector level looks like it might have changed. Triggers on "thesis check on [ticker]", "is my [ticker] thesis still intact", "check my positions", "has anything changed at [ticker]", "review my watchlist for thesis breaks".

Once you own something, the question changes from "should I buy?" to "is the thesis still intact?" This skill is the structured review that answers the second question.

Most thesis breaks happen quietly. The fundamental driver weakens before the price moves. The competitive position erodes one customer at a time. Management changes the metric definition before they change the business. A good thesis check looks for these specifically — and distinguishes them from noise.

## Workflow

### Step 1: Recover the original thesis

Pull the specific reasons the position was opened — from prior notes, journal entries, saved analysis, or `monitor/watchlist`. If none exists, ask the user for the 3–4 specific pillars that anchor the thesis. Concrete claims, not generic: "FCF should compound at >15%", "they will win > 30% share in segment X", "the regulatory tailwind expires FY27."

The check is against *these specific pillars*, not a generic "is the company doing well."

### Step 2: Fundamental tracking

For each pillar, check the corresponding number(s):

- **Growth pillar** → revenue and segment-revenue trajectory vs. the pillar's target
- **Margin pillar** → gross / operating / FCF margin trend
- **Returns pillar** → ROIC, FCF/share growth
- **Market-share pillar** → revenue growth vs. peer-set growth
- **Capital-return pillar** → buyback pace, share count, dividend

Score each pillar: **Tracking / Slightly ahead / Slightly behind / Diverging materially**.

### Step 3: Qualitative shifts

Cross-reference recent disclosures against the original qualitative assumptions:

- **Competitive landscape** — new entrants, customer-concentration changes, peer moves (read the company's risk factors plus peers' 10-Ks)
- **Customer base** — concentration shifts, churn signals, new large wins
- **Regulatory** — proceedings, rule changes, jurisdictional exposure
- **Management** — executive changes, comp-plan changes, insider patterns
- **Business model** — pivots, segment restructurings, pricing model changes

Score: **Stable / Mixed / Adverse shift**.

### Step 4: Valuation drift

- Has the multiple expanded or compressed materially since entry?
- Where is sell-side consensus relative to the original thesis (catching up, equalizing, moving against)?
- Insider direction (buying, selling, neutral) in the period since entry?

Score: **Cheaper / Fair / Stretched / Re-rated against thesis**.

### Step 5: Verdict

Combine the three scores into one of four verdicts, with specific evidence:

| Verdict | When to use |
|---|---|
| **Intact** | All pillars tracking; qualitative stable; valuation fair or cheaper |
| **Improved** | Pillars tracking ahead of plan; qualitative stable or improving; valuation still reasonable |
| **Weakening** | One pillar diverging materially OR adverse qualitative shift OR multiple stretched against original thesis |
| **Broken** | Two+ pillars diverging materially, or a single irreparable qualitative shift (competitive position destroyed, irreversible management issue, regulatory bar) |

Output: one-line verdict + 3–5 supporting bullets, each citing the specific data point or filing.

## Important Notes

- **Always check against the original specific thesis, not a refreshed one.** "I bought it for X" is the test. If the thesis has rotated post-purchase (commonly to justify holding), that itself is a finding — flag it as a separate item from the verdict.
- **A position can be Improved and still warrant a trim.** The original thesis being right too fast often coincides with valuation getting ahead.
- **Broken does not mean sell at any price.** It means the original reason is gone — the position is now a different bet. Make that bet consciously, or exit.
- **Noise vs. signal.** A single quarter of deceleration against a multi-year compounding pillar is noise. Three consecutive quarters is signal. Apply discipline; do not break a thesis on one print.
- **Insider direction is confirming evidence, not a verdict.** Use it to weight a verdict you have already formed from fundamentals + qualitative; do not let it overrule them.
- **Run after the right events.** Quarterly print, 10-Q, material 8-K, peer earnings, regulatory action. Do not run on price action alone — that is a different conversation.
