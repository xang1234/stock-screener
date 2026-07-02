# Earnings Scorecard

description: Scores a company's most recent earnings call across 8 tone dimensions (Lofty↔Grounded, Honest↔Defensive, Clarity, Composed↔Nervous, Executive Candor, Analyst Friction, Narrative Pivot Index, Q&A Information Density) and 6 content-integrity checks (beat/miss framing, metric accuracy, guidance credibility, risk candor, SEC consistency, selective omission). Use to assess management quality, compare across portfolio names, or detect deterioration over consecutive calls. Triggers on "score [ticker]'s earnings call", "is [ticker] management trustworthy", "compare [A] and [B] on earnings tone", "did management pivot the narrative", "guidance credibility", "earnings tone scorecard".

An earnings call contains two parallel signals: the numbers, and the people presenting them. Reading both together is more informative than either alone.

Most investors focus on beat/miss vs. consensus. That's the least interesting part. The more durable signal is whether management's communication style is honest and grounded — because that predicts how they'll behave in the quarters where the news is bad. And the quantitative pattern — beat rate, guidance accuracy, estimate revision momentum — tells you whether the street is consistently underestimating or overestimating the business.

This skill scores both, side by side, against the company's own actuals and prior-period commitments.

## Workflow

### Step 1: Pull inputs

For the named ticker(s):
- **Most recent earnings call** — full transcript plus structured summaries (segment performance, management highlights, guidance, risks, full Q&A).
- **Prior quarter's call** — the guidance field specifically, to verify against this quarter's actuals.
- **Beat/miss history** — last 4 quarters of EPS and revenue estimated vs. actual.
- **Beat rate, last 12 quarters** — EPS and revenue, separately. Distinguishes deliberate expectation managers (11/12) from execution-issue companies (5/12).
- **Reported-quarter financials** — revenue, gross profit, operating income, net income, diluted EPS, FCF, debt — vs. prior quarter and YoY. Use these to identify the 2–3 most material financial events of the quarter.
- **Most recent SEC filings** — 10-Q and 8-Ks since the prior call, for cross-checking 1–2 specific verifiable claims.

### Step 2: Score the 8 tone dimensions

For each dimension, assign a 1–10 score with one specific quote or pattern as evidence.

| Dimension | What to measure | Scale |
|---|---|---|
| **Lofty ↔ Grounded** | Ratio of visionary/buzzword language ("transformational", "unprecedented", "AI revolution") vs. concrete specifics (unit economics, named customers, exact %s) | 1 = pure buzzwords … 10 = fully concrete |
| **Honest ↔ Defensive** | Does management front-foot bad news and own misses, or deflect to macro, bury negatives, use passive voice for failures? | 1 = highly defensive … 10 = fully transparent |
| **Clarity** | Are prepared remarks structured and direct? Are Q&A answers on-topic and specific, or rambling and circular? | 1 = incoherent/evasive … 10 = crisp and direct |
| **Composed ↔ Nervous** | Hedging density ("we believe", "we hope", "it's hard to say"), contradictions between prepared remarks and Q&A, backtracking under pressure | 1 = visibly uncertain … 10 = highly composed |
| **Executive Candor** | Do they volunteer weaknesses before being asked, or only address problems when analysts press? | 1 = never volunteers negatives … 10 = proactively transparent |
| **Analyst Friction** | How many Q&A questions got deflected, redirected, or answered with non-answers vs. direct engagement? | 1 = heavy deflection … 10 = fully engaged |
| **Narrative Pivot Index** | Identify the 2–3 most material financial events of the quarter from the actuals. Check whether each receives proportionate airtime on the call, or whether management displaces them with a positive story. High pivot = call is talking about something very different from what the numbers say. | **Inverted:** 1 = call mirrors financials … 10 = complete narrative substitution |
| **Q&A Information Density** | Did the analyst session produce genuinely *new* information not in the prepared remarks (specific customer names, exact timelines, new data points, precise competitive response)? Or did every answer restate the prepared narrative? | 1 = pure restatement … 10 = rich new disclosure |

### Step 3: Score the 6 content-integrity checks

| Check | How to verify | Verdict |
|---|---|---|
| **Beat/Miss Framing** | Cross-check actual vs. estimated. If EPS beat by < 2%, was it framed as "record"? If missed, acknowledged or buried? | Fair / Overcelebrated / Understated / Miss buried |
| **Metric Accuracy** | Spot-check 2–3 specific % claims from management highlights or segment commentary against actual financials. Flag inflation or rounding tricks. | Accurate / Inflated / Deflated |
| **Guidance Credibility** | Compare prior quarter's guidance against this quarter's actual. | Delivered / Partial / Missed / Vague (can't verify) |
| **Risk Candor** | Are the risks discussed specific to *this* company's situation, or boilerplate disclaimers anyone could copy-paste? | Specific / Generic boilerplate |
| **SEC Consistency** | Run SEC filings search on 1–2 verifiable claims (e.g., "supply constraints are resolved" → check the latest 10-Q risk factors). | Consistent / Divergent |
| **Selective Omission** | Are there known weak segments visible in the financials that were absent from management highlights? | None / Present |

### Step 4: Present

Open with the call header (ticker, period, call date) and the beat/miss table:

| Metric | Estimated | Actual | Delta |
|---|---|---|---|
| EPS | $X.XX | $X.XX | +X% / −X% |
| Revenue | $XB | $XB | +X% / −X% |

**Beat/miss framing verdict:** [Fair / Overcelebrated / Understated / Miss buried]

Then the 8-dimension tone scorecard table, the 6-dimension content-integrity table, a one-line **Overall Tone Profile**, a 3–4 sentence **Synthesis** of what stands out and what to watch, and a single **Most Revealing Quote** — the exact line from the transcript that best reveals management character.

For multi-ticker comparison, format as a cross-portfolio table with one column per ticker. Each cell = score + one-line remark.

## Important Notes

- **Use the full transcript for tone analysis.** AI-generated summaries are pre-filtered and lose the cues — hedging density, Q&A friction, and narrative pivots only show up in the raw text.
- **Narrative Pivot is inverted.** Lower is better (call mirrors financials). When color-coding output, reverse the scale: 1–3 green, 4–6 amber, 7–10 red. Every other dimension uses standard 8–10 green, 5–7 amber, 1–4 red.
- **Vague prior guidance gets a Vague verdict.** "Mid-single-digit growth" with no segment-level color is unverifiable. Rate it Vague and explain why — don't manufacture a credibility score from absence.
- **SEC cross-check is optional but valuable.** If no specific verifiable claim was made, skip the check and note it. If a claim was made and the filing says the opposite, that's the highest-signal finding the skill can produce.
- **Score in context.** A policy-recovery pivot in a structurally distressed sector is not the same as a narrative substitution in a miss quarter. Apply judgment — the framework is a discipline, not a checklist.
- **Beat rate over 12 quarters separates expectation managers from execution issues.** 11-of-12 says "we manage the street" — read forward guidance with that in mind. 5-of-12 says "we cannot forecast our own business" — discount forward commentary accordingly.
