# Labor Market

description: Reads the US labor market beyond the headline nonfarm-payrolls number using leading indicators — initial jobless claims, continuing claims, the quits rate, temporary-help employment, household-survey vs. establishment-survey divergence, payroll revision patterns, wage growth. Scores each against historical thresholds. Use as the macro backdrop for consumer, retail, financial, and homebuilder theses, or for cycle positioning. Triggers on "labor market", "is the labor market softening", "initial claims", "quits rate", "payrolls revisions", "leading labor indicators", "pre-recession labor signals".

The monthly nonfarm-payrolls number gets all the attention, but it is a lagging indicator — by the time it weakens decisively, the economy is often already in contraction. The leading indicators in the labor data are less followed and more informative.

## Workflow

### Step 1: Pull the leading indicators

- **Initial jobless claims** (weekly) — 4-week moving average and trend over the last 24 months
- **Continuing claims** — duration-of-unemployment signal
- **Quits rate** — from JOLTS, monthly
- **Temporary-help employment** — from the payrolls report, monthly
- **Household survey vs. establishment survey** — both unemployment-rate inputs and employment-level inputs
- **NFP revision pattern** — direction and magnitude of revisions to prior months, rolling 12 months
- **Wage growth** — average hourly earnings, ECI where available

### Step 2: Score against thresholds

| Indicator | Threshold for concern | Historical context |
|---|---|---|
| **Initial claims (4-wk avg)** | Sustained > 250k = deterioration; sustained > 300k = recession-adjacent | Crossed 300k 6 months before the 2007–08 recession |
| **Continuing claims** | Rising trend = workers staying unemployed longer | Leading indicator of the unemployment rate |
| **Quits rate** | Falling below 2.3% = workers losing confidence in mobility | Leads payrolls by 2–3 quarters historically |
| **Temp-help employment** | Peaked and rolling over | Hired first in expansions, cut first in contractions — leads total employment |
| **Household vs. establishment** | Establishment > household by > 1M jobs for 2+ quarters | Divergences typically resolve in the direction of the household survey |
| **NFP revisions** | Consistent downward revisions over rolling 6 months | Pattern coincides with late-cycle hiring overstatement followed by retroactive cuts |
| **Wage growth** | Falling toward 3% with unemployment rising = clear softening | Falling alone may be normalization |

### Step 3: Interpret the cycle stage

Combine the indicator scores into a cycle-stage read:

- **Mid-cycle / expansion** — Claims low and stable, quits rate stable or rising, temp employment stable or growing, divergences modest
- **Late-cycle / cooling** — Quits rate falling, temp employment rolling over, claims drifting up, payroll revisions turning negative
- **Pre-recession setup** — Claims trending decisively up, quits below threshold, temp employment in decline, large household-vs-establishment divergence with establishment higher
- **Recession active** — Claims sustained above the recession-adjacent threshold, household-survey job losses, broad-based payroll declines

### Step 4: Present

Open with the one-line cycle-stage read. Then:
- Indicator-scoring table with the current reading vs. threshold and historical context
- Cycle-stage assessment, naming the 2–3 indicators carrying the most weight in the current read
- Sector implications — consumer discretionary, financials (loan-loss exposure), homebuilders, staffing / temp companies — at a one-paragraph level
- One line on what to watch over the next 4 weeks of data that would change the read

## Important Notes

- **Payrolls is a lagging indicator, not a leading one.** Build the read around claims, quits, and temp employment first; treat payrolls as confirmation.
- **The household survey moves first at turns.** When the household survey is weakening while payrolls are still strong, lean toward the household read — historically, the establishment survey catches down over the following 6–12 months.
- **Wage growth is not the labor cycle.** It is a composition / Phillips-curve indicator. A cooling labor market typically shows up in claims and quits well before it shows up in wages.
- **Revisions tell you about the pattern of estimation, not the level.** Persistent downward revisions in late-cycle periods are diagnostic; the absolute revision size matters less than the consistency of direction.
- **One month is not a trend.** Claims spike on holidays, hurricanes, and one-state events. Use 4-week moving averages and look for trend changes that persist 6–8 weeks.
- **Each cycle has its own quirks.** The post-COVID labor cycle saw the quits rate collapse without payroll deterioration for two years. Do not pattern-match without context.
