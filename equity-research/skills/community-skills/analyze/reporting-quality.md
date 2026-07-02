# Reporting Quality

description: Catches changes in how a company defines, classifies, or presents its **quantitative disclosures** — the changes most likely to flatter, hide, or break the comparability of the underlying business. Scans eight dimensions: KPI / metric-definition drift; above-vs-below-the-line reclassifications (operating ↔ non-operating, COGS ↔ OpEx, continuing ↔ discontinued); segment restructuring; capitalization policy; revenue-recognition policy; non-GAAP exclusion-category drift; restatements and prior-period adjustments; and selective omission of previously-disclosed metrics. For every flagged change, recomputes the current period under the prior definition where possible. Use before relying on any time-series, before using a non-GAAP metric for valuation, after a 10-K is filed, or whenever a beat looks too clean given prior trajectory. Triggers on "reporting quality", "did [ticker] change [metric] definition", "did [ticker] restate", "non-GAAP changed", "segment restructuring at [ticker]", "above the line", "below the line", "what does [ticker] no longer disclose", "is [ticker]'s comparability broken".

The most expensive disclosures to miss are the quiet ones — a definition narrowed by one footnote, a cost line moved below operating income, a segment cut differently, an item reclassified from one-time to recurring or back. These do not show up as accounting fraud. They show up as a beat that should not have been there, a margin that "expanded" without an operating reason, or a year-over-year comparison that no longer means what it used to.

This skill is the disciplined sweep that catches them, and — critically — restates the current-period numbers under the prior definitions so the analyst's working model stays honest.

## Workflow

### Step 1: Pull the disclosure base

For the named ticker:
- **Last 4–5 annual 10-Ks** — Note 1 (Significant Accounting Policies), Note on Segments, MD&A KPI definitions, non-GAAP reconciliation table, restatement and reclassification disclosures
- **Last 8 quarters of 10-Qs** — Note 1 updates, segment table, classification notes, any mid-year policy adjustment
- **Last 12 quarters of earnings press releases** — what got bullet-point treatment, what disappeared, the non-GAAP table line-by-line
- **Material 8-Ks since the last 10-K** — Item 4.02 (non-reliance on prior financials), 2.05 (impairments and exit charges)
- **Peer 10-Ks** for the same metric definitions and policy choices (calibrate against industry norms)

### Step 2: Score the eight quantitative-reporting dimensions

For each, score **Clean / Watch / Concern / Material concern** with verbatim before-and-after evidence.

| # | Dimension | What to check |
|---|---|---|
| 1 | **KPI / metric definition** | The exact definition of each headline KPI in MD&A — DAU, MAU, ARR, NRR, ASP, take rate, GMV, same-store sales, RPO, contract value — across 5 annual filings. Did the population, time window, inclusion criteria, or calculation change? |
| 2 | **Above ↔ below the line** | Has any cost or revenue line moved between operating and non-operating? Between COGS and OpEx? From recurring to "restructuring" or "transformation" (or vice versa)? From continuing operations to discontinued? These moves change every margin downstream. Track line by line vs. the prior 4 10-Ks. |
| 3 | **Segment restructuring** | When segments were last re-cut. What was in the old segments; where the components went in the new structure. Was prior-period data fully restated under the new structure, or only forward? Loss of historical comparability is the cost. |
| 4 | **Capitalization policy** | What is capitalized vs. expensed — internally developed software, cloud-computing arrangements, content production, customer-acquisition costs, R&D, leases. Has the policy or the capitalization *rate* changed? A disclosed rate change > 5pp YoY flatters current earnings. |
| 5 | **Revenue-recognition policy** | Principal vs. agent presentation (gross vs. net); point-in-time vs. over-time; treatment of variable consideration; volume rebates; bundled contracts. ASC 606 leaves judgment — and judgment shifts. |
| 6 | **Non-GAAP exclusion drift** | Line-by-line in the GAAP → adjusted reconciliation, quarter by quarter, over 8+ quarters. Are new exclusion categories appearing ("transformation costs", "AI investments", "integration costs")? Are recurring categories framed as one-time? |
| 7 | **Restatements & prior-period adjustments** | Any restatement, reclassification, or cumulative-effect adjustment disclosed in the most recent 10-K / 10-Q. Cause, magnitude, periods affected. A "voluntary reclassification" is still a finding. |
| 8 | **Selective omission of quantitative disclosures** | Metrics that got bullet-point press-release treatment in past good quarters and have since disappeared, moved to footnotes, or been replaced with a narrower metric. What was happening to the underlying metric when disclosure stopped? |

Each row gets: verdict, specific filing date and section, verbatim before-and-after language, and the directional impact (flatters current period / obscures comparability / hides a deterioration / neutral).

### Step 3: Recompute the prior-definition number

For every Concern or Material concern in Step 2, **restate the current-period number under the prior reporting standard** and report the gap. This is the most important output of the skill — a "definition change" finding without the restated number is half the work.

Examples:
- Old DAU definition included logged-out visits; new definition does not → recompute current DAU under the old definition; show the gap.
- Cost line moved from OpEx to "non-recurring" → show operating margin under both classifications, side by side.
- New non-GAAP exclusion adds back $300M of SBC-related items → show adjusted EPS without the new exclusion.
- Revenue recognition shifted from net to gross for a marketplace segment → show revenue and gross margin under both treatments.
- Segment recut moves a declining business into a growing one → show the new segment's growth under the old composition.

If the disclosure is insufficient to support a back-calculation, say so explicitly — and flag the company's failure to provide the bridge as itself a concern.

### Step 4: Present

Open with a one-line **comparability verdict**:
- **Comparable** — no material changes; the time series can be used as-is
- **Adjusted-comparable** — material changes, but the company restated prior periods consistently
- **Broken at [date]** — material changes without consistent restatement; pre/post comparison is not apples-to-apples
- **Repeatedly broken** — multiple changes within 3 years; treat all multi-year growth and margin claims as suspect

Then the 8-dimension scoring table with verbatim citations for every flagged item. For every Concern and Material concern, show the restated number from Step 3 next to the as-reported number, with the bps / $ / % gap.

Close with the single highest-impact finding and what to verify on the next 10-Q.

## Important Notes

- **Reclassifications are legal and almost always defensible.** That does not make them neutral. A cost moved below the line still has to be paid; an exclusion added to non-GAAP still hits cash flow. Score them by economic impact, not by whether they violate GAAP.
- **The press release leads the filing.** Favorable framing and selective omission appear in the press release first; the 10-Q usually still discloses (often deep in MD&A or footnotes). Cross-check the two — that is where most findings live.
- **Restated-prior-period filings are the cleanest tells.** When a company restates the prior-year segment table or KPI series in the current 10-K, read those restated columns against the prior 10-K's original columns. Differences = the change.
- **Definition drift rarely runs both ways.** A narrowed definition that makes the current number look bigger is a finding. A widened definition that makes the current number look bigger is also a finding. Either direction warrants the recompute in Step 3.
- **"One-time" that recurs is recurring.** "Transformation costs," "integration costs," "litigation costs" — any category that appears in three or more consecutive years is, by definition, not one-time. Re-add to adjusted earnings before using the metric for valuation.
- **Compare to peers on every dimension.** Industry norms set the baseline. A capitalization rate of 35% may be normal for media; alarming for SaaS. A non-GAAP gap of 25% may be normal for high-growth software; concerning for industrials.
- **Watch for changes that coincide with a guidance miss or analyst day.** A reclassification announced two weeks before earnings or alongside an analyst day is timed; treat the change as defensive until proven otherwise.
- **Cite verbatim, with filing + section + page.** The credibility of every finding rests on the citation. Quote the exact language; do not paraphrase.
- **The recompute is what makes this skill valuable.** Step 3 is the highest-leverage step. Without it, this is a list of findings; with it, it is a quantified impact on the analyst's working numbers.
