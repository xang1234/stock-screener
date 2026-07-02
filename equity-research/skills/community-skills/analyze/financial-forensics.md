# Financial Forensics

description: Looks for systematic divergences between reported earnings and economic reality — six specific patterns that show up in the financial statements before they show up in the price. Scores FCF-vs-net-income divergence, SBC dilution, channel-stuffing signals (DSO and inventory), non-GAAP gap widening, working-capital demand signal, and capitalization-policy shifts. Use before sizing a position, when a name looks too good on headline metrics, or when the multiple has expanded without an obvious fundamental driver. Triggers on "quality of earnings", "earnings quality", "FCF vs net income", "channel stuffing", "SBC dilution", "non-GAAP gap", "is [ticker]'s accounting clean", "forensic check on [ticker]".

The income statement tells you what management wants you to see. The cash flow statement tells you what actually happened.

Most analysts read the headline numbers. Financial forensics is about reading the gaps — the systematic divergences between reported earnings and economic reality that show up in the numbers before they show up in the stock price. The lens is six specific patterns, each with a measurable signal and a verdict.

## Workflow

### Step 1: Pull inputs

For the named ticker:
- **Income statement** — last 12 quarters: revenue, gross profit, operating income, SBC, net income, diluted EPS, diluted share count.
- **Cash flow statement** — last 12 quarters: operating cash flow, capex, free cash flow, SBC add-back.
- **Balance sheet** — last 12 quarters: A/R, inventory, deferred revenue, capitalized software/R&D balance.
- **Non-GAAP reconciliation** — most recent earnings release and 10-Q. Capture every line item in the reconciliation from GAAP → adjusted.
- **Capitalization policy disclosures** — 10-K notes on capitalized software development costs, R&D, internal-use software.

### Step 2: Score the six forensic patterns

For each pattern, score Clean / Watch / Concerning / Red Flag and cite specific numbers.

| Pattern | What to measure | Signal |
|---|---|---|
| **1. FCF vs. net income divergence** | Cumulative FCF / cumulative net income over trailing 3 years. Trend of the gap. | Clean: ratio > 0.85 and stable. Concerning: ratio < 0.70 and widening for 3+ years. |
| **2. SBC dilution** | SBC as % of revenue and as % of operating income. Diluted share count CAGR vs. EPS CAGR. | Clean: SBC < 5% of revenue, share count flat-to-shrinking. Concerning: SBC > 10% of revenue and share count growing faster than EPS. |
| **3. Channel-stuffing signals** | DSO trajectory vs. revenue growth. Inventory days vs. revenue growth. | Clean: DSO flat or down. Concerning: DSO expanding > 20% YoY without a known mix shift; inventory days expanding > 15% ahead of revenue. |
| **4. Non-GAAP gap widening** | (Adjusted EPS − GAAP EPS) / Adjusted EPS, trended over 3 years. Composition of the adjustments. | Clean: gap stable, dominated by truly one-time items. Concerning: gap widening; "restructuring" or "transformation" charges recurring quarterly. |
| **5. Working-capital demand signal** | Deferred revenue trajectory; A/R growth vs. revenue growth. | Clean: deferred revenue growing in line with or faster than revenue. Concerning: deferred revenue contracting; A/R growing materially faster than revenue. |
| **6. Capitalization choices** | Capitalized software/R&D as % of total R&D spend, trended. Year-over-year change. | Clean: capitalization rate stable and modest. Concerning: rising capitalization rate that coincides with an EPS-beat quarter. |

### Step 3: Cross-check against disclosure

For any pattern scored Concerning or Red Flag:
- Search the most recent 10-K and 10-Q for management's explanation. A legitimate divergence usually has one.
- Cross-check the most recent earnings call: did management address the pattern, or did they pivot away from it? (The Narrative Pivot dimension from `earnings-scorecard` is a useful companion check.)
- Compare the company against its 2–3 closest peers on the same pattern. A company whose FCF/NI gap is unique among peers is more interesting than one where the whole industry shares it.

### Step 4: Present

Open with a one-line **quality-of-earnings verdict**: Clean / Mixed / Concerning / Red Flag.

Then the six-pattern scoring table with specific numbers and one line of evidence each. For any pattern flagged Concerning or Red Flag, write 3–4 sentences:
- What the data shows, in numbers
- What management has and has not said about it
- The plausible benign explanation and why you believe it (or do not)
- What to watch in the next 1–2 quarters that would confirm or invalidate

Close with **what would move the verdict**: the specific next-quarter data point that would shift Concerning → Clean (or vice versa).

## Important Notes

- **One pattern is rarely the answer.** Quality-of-earnings concerns gain credibility when two or more patterns align — e.g., FCF divergence + rising capitalization + widening non-GAAP gap. A single divergent metric usually has a benign explanation.
- **Cyclical inventory builds are not channel stuffing.** A semiconductor company building inventory into an upcycle is doing its job. The forensic concern is inventory growing *without* a credible demand pull — verify against management's stated rationale and customer commentary.
- **SBC stripping is the most common adjusted-EPS abuse.** Treat SBC as a real cash cost. The adjusted EPS that excludes it is fictional.
- **Capitalization-rate shifts almost always show up at quarter-ends.** If the rate jumped in a beat quarter and the EPS would have missed without the change, that is the finding — disclose it clearly.
- **Forensics is about the gap, not the level.** A high but stable level of any of these is usually fine. A widening trend is what matters.
- **Distinguish accounting from economics.** Some divergences are economically real (a working-capital build for a growing company); some are accounting choices. Be specific about which you are flagging.
- **Don't confuse forensic with fraud.** Most of what this skill finds is aggressive accounting within GAAP, not fraud. Frame findings as quality-of-earnings risk, not allegations.
