# Trade Flow Analysis

description: Measures supply-chain relocation and trade-pattern shifts at the HS-code (product) level using customs and trade data — rather than relying on narrative claims of reshoring or "China+1." Detects transshipment patterns (Chinese exports to Vietnam rising in the same categories Vietnam now exports to the US) and ties product-level shifts to specific public companies that benefit. Use to verify or quantify a supply-chain thesis, find under-followed beneficiaries of trade-flow shifts, or assess tariff exposure. Triggers on "trade flows", "reshoring", "China+1", "HS code [X]", "supply chain shift", "import sources for [product]", "Vietnam manufacturing rise", "Mexico nearshoring", "transshipment".

Supply-chain relocation has been asserted far more than it has been measured. Trade data lets you measure it directly — by product category and country pair, over multi-year windows. The narrative says "reshoring"; the data says where it is actually happening, where it is stuck, and where it is just being re-routed.

## Workflow

### Step 1: Frame the question

One of:
- **Product-first** — pick an HS code (or family of codes) and track origin-country shifts over time
- **Country-first** — pick a country and track which product categories are rising or falling
- **Thesis-first** — user has a specific claim ("Vietnam is gaining apparel share from China"); measure it directly

### Step 2: Pull the trade data

For the chosen scope:
- **US imports by partner country**, annual (and quarterly where available) for the last 5–10 years
- **Bilateral exports from third countries** in the same HS codes (e.g., China → Vietnam) — for transshipment cross-check
- **Global trade flows** if the question is about end-customer markets beyond the US

### Step 3: Compute the share shifts

| Signal | What to compute | Interpretation |
|---|---|---|
| **Source-country share shift** | Each partner's share of US imports in the category, year over year | The headline reshoring measure |
| **Concentration / HHI** | Herfindahl across source countries for the category | Lower HHI = diversified supply base; rising HHI in a new origin = single-country dependence forming |
| **Transshipment proxy** | Chinese exports to a third country in the category vs. that third country's exports to the US in the same category | Both rising together = transshipment; only third-country → US rising = genuine relocation |
| **Value vs. volume** | $ value share vs. unit volume share | Diverging = price effect (tariff pass-through) rather than real volume relocation |
| **Capacity vs. flow** | Compare against plant-opening / capex announcements in the destination country in the category | Real relocation comes with capacity build; without it, the flow shift is probably re-routing |

### Step 4: Tie to companies

Once the shift is established, identify public companies that benefit:
- US producers in the category whose import-competition has eased
- Foreign producers in the gaining country who are listed (often Taiwan, Korea, Mexico, India)
- US distributors / logistics / freight forwarders touching the new lane
- Equipment makers selling into the destination country's capacity build

Use the company-ontology layer to enumerate plausible names, then verify against SEC filings (segment commentary, revenue by geography).

### Step 5: Present

Open with a one-line read: "[Category] is genuinely relocating from [origin] to [destination]" / "Apparent shift is largely transshipment" / "Narrative-only, no measurable flow change." Then:
- Source-country share table over 5–10 years (top 8 partners + Rest of World)
- Transshipment cross-check table where relevant
- 3–5 named public companies with disclosed exposure to the shift, ranked by purity
- One paragraph on whether the shift is durable or tariff-cycle-dependent

## Important Notes

- **"China+1" is not reshoring.** A flow shift from China to Vietnam where Chinese intermediates still feed Vietnam is a re-routing, not a relocation. The transshipment cross-check separates them.
- **Tariff cycles distort short-window data.** A 2-year shift coinciding with a tariff hike may reverse if the tariff is rolled back. Look for multi-year capacity buildout to confirm durability.
- **Value vs. volume divergence reveals price effects.** A 30% jump in import value with flat volume is mostly higher unit prices passed through — typically a tariff signal, not a relocation one.
- **The smallest categories often move first.** Apparel and consumer goods relocated years before semiconductors did. A category that is "small" today can be a leading indicator for a related larger category.
- **Cite the HS code.** Trade-flow claims need the specific code (e.g., HS 8542 for integrated circuits, HS 6109 for cotton t-shirts) — vague "manufacturing" or "electronics" claims are unverifiable.
- **Customs data lags.** US monthly customs releases run with ~ 1–2 month lag. Treat the most recent month as preliminary.
