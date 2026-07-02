# Government Contract Signals

description: Scans US federal contract awards as a leading revenue indicator — typically 1–3 quarters ahead of reported revenue for companies in defense, federal IT, healthcare services, and scientific research. Use to verify or anticipate a thesis on a federal-exposed name, find under-followed companies with accelerating federal exposure, or track a specific capability area. Triggers on "federal contracts", "DoD awards", "sole-source", "federal exposure", "NIH grants", "what has [ticker] won", "who's winning [agency] contracts".

Federal contracts are public information. Most investors don't read them.

## Workflow

### Step 1: Frame the search

One of three modes:
- **By company** — track all federal awards for a named ticker.
- **By capability / theme** — find companies winning in a space ("DoD generative AI", "VA EHR modernization", "HALEU enrichment").
- **By agency or program** — find dominant vendors for a specific buyer ("SDA", "NIH NIAID").

### Step 2: Pull awards

Fetch awards over the relevant window (12–24 months typical) with vendor, parent vendor, agency, contract type, competition type (full & open / sole-source / set-aside), obligated $, total ceiling, period of performance, NAICS/PSC code, and description.

### Step 3: Score the signal

| Signal | Why it matters |
|---|---|
| **Trend** | Trailing 6m vs. prior 6m — accelerating, flat, or rolling over |
| **Size shift** | Are recent awards larger? Indicates scope expansion |
| **Sole-source frequency** | Sole-source = customer-validated capability with pricing leverage |
| **Re-up vs. new program** | Re-up = sticky revenue; new program = optionality |
| **Coverage gap** | Sell-side coverage vs. federal $ exposure — under-followed = alpha setup |

### Step 4: Cross-check

Confirm material awards against the company's 8-K filings (a true win above the materiality threshold should be disclosed). Check the most recent earnings call for backlog or RPO commentary that aligns with the award trend.

### Step 5: Present

Open with a one-line read of the signal (accelerating / steady / decelerating / inflecting). Table of relevant awards (date, agency, $, competition type, description). Close with one line on how the signal modifies the fundamental view.

## Important Notes

- **Obligated $ ≠ contract ceiling.** IDIQ ceilings are upper bounds; only obligated $ is committed near-term spending. Don't quote ceiling as revenue.
- **Parent vendor matters.** Roll subsidiary awards up to the listed parent before drawing conclusions.
- **A drought is a signal.** A historically active vendor going quiet for 2–3 quarters often precedes guidance disappointment. Negative space is data.
- **Lag varies by contract type.** Firm-fixed-price awards convert to revenue fast (weeks–months); cost-plus and large IDIQs can lag 2–4 quarters.
