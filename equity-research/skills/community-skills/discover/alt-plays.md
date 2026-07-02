# Alternative Ticker Discovery

description: Given a seed ticker (or watchlist), surface look-alike companies, upstream suppliers, downstream enablers, and second-order beneficiaries — then rank candidates by which have more upside than the seed, using valuation gaps, analyst price targets, distance from 52-week highs, and thesis purity. Use when the user is considering buying a name and wants to verify it's the best expression of their thesis. Triggers on "alternatives to", "look-alike", "what else should I own if I own X", "second-order play", "downstream of", "better way to play", "upstream pure-play", or "compare X and Y as an expression of [theme]".

If you already like a company, there's a question worth asking before you buy: is this actually the best way to own this thesis?

A stock you admire might be the right idea but the wrong vehicle. It could be priced to perfection. It might have multiple business lines diluting the thesis. The pure-play upstream supplier might offer identical exposure at a 40% valuation discount. Or the downstream enabler might have faster earnings growth with less competition for analyst attention.

This skill starts from a ticker you already believe in and works outward — finding adjacent companies that share the same tailwind, then ranking them by which has more upside from here. We look at valuation gaps, analyst price targets, distance from 52-week highs, and thesis purity. The goal isn't to replace your conviction — it's to make sure you're expressing it in the most efficient way.

## Workflow

### Step 1: Resolve inputs

The seed is one or more tickers, or a saved watchlist. Resolve the watchlist to its tickers. For each seed, pull:

- Current price, market cap, valuation multiples (fwd P/E, EV/EBITDA, P/S), growth (fwd revenue & EPS, 3-year CAGRs)
- Price returns (1m / 3m / 6m / 1y) and key technical signals (200-day moving average, RSI)
- Earnings-beat history (last 4q surprise %, beat count out of last 12 quarters)
- Margins and leverage (GPM, EBIT, FCF; net debt / EBITDA)
- Analyst consensus price target (consensus level, target high/low, analyst count)
- 52-week high and low
- Company ontology / wiki profile: business description, products & services, supply chain, customers, competitors, business model

### Step 2: Build the theme map

Synthesize the investment thesis from the seed data. Lay out three things explicitly:

1. **Primary theme** — the macro tailwind ("AI compute demand", "data-center power buildout", "GLP-1", "reshoring capex", "defense modernization")
2. **Revenue driver** — the specific product or service the market is paying for ("H200 GPU", "PDU + liquid cooling for hyperscalers", "NAND / HBM for AI training")
3. **Value-chain position** — where the seed sits:
   - Raw material / feedstock supplier
   - Specialty component / IP licensor
   - Subsystem / module maker
   - Platform / prime contractor
   - Downstream enabler / beneficiary

Then sketch the chain around the seed, explicitly: upstream dependencies, downstream beneficiaries, adjacent infrastructure, second-order consumers. This is the surface area you will search in Step 3.

### Step 3: Decompose the theme into specific search terms

For each theme, enumerate **specific product / process / technology names** visible in public filings. Broad keywords ("AI", "data center", "power") return noise and miss the pure-plays.

| Theme | Specific search terms | Avoid |
|---|---|---|
| AI compute | "CoWoS", "HBM3e", "NVLink", "InfiniBand", "800G transceiver", "liquid cooling CDU", "busway PDU", "cold plate" | "artificial intelligence", "GPU", "data center" |
| Power infrastructure | "dry-type transformer", "paralleling switchgear", "UPS flywheel", "generator transfer switch" | "power", "energy", "electricity" |
| Semiconductor | "GaN HEMT", "SiC epitaxy", "PECVD", "CMP slurry", "ALD precursor", "photoresist" | "chip", "semiconductor", "wafer" |
| GLP-1 / obesity | "GLP-1 receptor agonist", "amylin analog", "oral semaglutide", "incretin mimetic" | "obesity", "weight loss", "diabetes" |
| Reshoring | "precision casting", "hot isostatic pressing", "titanium forging", "beryllium copper" | "manufacturing", "onshoring", "supply chain" |
| Defense | "GaN HEMT radar", "HALEU enrichment", "ammonium perchlorate", "rad-hard FPGA", "ablative TPS" | "defense", "military", "DoD" |

Generate 4–8 specific terms per seed theme.

### Step 4: Search for candidates

For each search term, run a company-ontology search to find tickers whose product, description, or supply-chain text matches. For the 2–3 most central terms, also run an SEC filings search — this catches pure-plays whose ontology entries are thin but whose 10-K / 10-Q filings reveal deep exposure.

Collect a raw candidate pool of 10–30 tickers. Deduplicate and exclude the seed(s).

### Step 5: Pull candidate data

For each candidate, pull the same fields as the seed in Step 1: valuation, growth, technicals, returns, margins, leverage, surprises, analyst targets, 52-week range.

### Step 6: Compute upside signals

For each candidate, compute and compare to the corresponding seed:

| Signal | What it tells you |
|---|---|
| **% off 52-week high** | How far from recent peak. Deeply negative may mean room to recover — or a broken story; Step 7 separates them. |
| **Analyst implied upside** | (Consensus PT / Current Price − 1). Sell-side view of the gap to close. |
| **Valuation discount vs. seed** | Candidate fwd P/E ÷ seed fwd P/E. Below 1.0 means cheaper. |
| **Growth-adjusted valuation (PEG proxy)** | Fwd P/E ÷ (Fwd EPS growth × 100). Lower is more attractive. |
| **Momentum vs. seed** | Candidate 3-month return vs. seed 3-month return. Has the candidate underperformed the seed recently? |

### Step 7: Classify each candidate

**BETTER UPSIDE** — all three hold:
1. Thesis is intact or improving (not a broken story)
2. Candidate is cheaper on at least one key multiple vs. seed, or has demonstrably faster EPS growth at a comparable multiple
3. Either: (a) analyst consensus implies > 15% upside, OR (b) candidate is > 20% off 52-week high while seed is within 15% of high, OR (c) pure-play exposure with no diluting business lines

**SIMILAR UPSIDE** — comparable thesis quality and price setup; worth holding alongside the seed.

**WATCH** — interesting thesis but currently expensive, momentum poor, or near-term catalysts unclear.

**SKIP** — thesis diluted (too many unrelated segments), fundamentals deteriorating, or already priced to perfection.

### Step 8: Present

Open with the seed snapshot:

| Ticker | Theme | Chain Position | Price | % Off 52w High | Fwd PE | Analyst Target | Implied Upside | Consensus |
|---|---|---|---|---|---|---|---|---|
| NVDA | AI compute (prime) | Platform | $XXX | −X% | XXx | $XXX | +XX% | Strong Buy |

Then the candidate map, grouped by theme:

| Ticker | Company | Relationship | Fwd PE | vs Seed PE | % Off 52w High | Analyst Upside | Verdict |
|---|---|---|---|---|---|---|---|
| VRT | Vertiv | Downstream enabler | 28x | −20% cheaper | −18% | +24% | **BETTER UPSIDE** |
| MU | Micron | Upstream supplier | 12x | −66% cheaper | −35% | +55% | **BETTER UPSIDE** |

For each **BETTER UPSIDE** pick, write 4–6 sentences:
- What the company makes and exactly how it connects to the seed's thesis (specific, not generic)
- Why the valuation / price setup is more attractive now vs. the seed
- The near-term catalyst or inflection (earnings beat, contract announcement, product cycle)
- The key risk (customer concentration, cycle timing, execution)
- What to watch to confirm or invalidate

Close with a **"What NOT to chase"** list — candidates that surfaced but were cut, one-line reason each. If no candidate clears the BETTER UPSIDE bar, say so explicitly: the seed is the right expression.

## Important Notes

- **Search discipline.** Always decompose themes into specific product / technology terms before searching. Broad queries on "AI" or "power" return noise and miss the pure-plays — Step 3 is not optional.
- **BETTER UPSIDE is a high bar.** Requires *both* an intact thesis *and* a concrete valuation or price-setup advantage. A stock that is merely down is not automatically better — deteriorating fundamentals make it worse, not cheaper.
- **Relationship taxonomy.** Use one of these labels for every candidate:
  - *Direct competitor* — same product, same customer, zero-sum share fight
  - *Upstream supplier* — provides inputs the seed's industry depends on
  - *Downstream enabler* — new demand created by the seed's product adoption
  - *Adjacent infrastructure* — necessary co-investment alongside the seed (e.g., power for GPUs)
  - *Second-order consumer* — economics improve because of what the seed's product enables
- **Valuation multiple priority.** Prefer fwd P/E first; fall back to EV/EBITDA fwd when EPS is not meaningful (early-cycle, restructuring); fall back to P/S fwd when EBITDA is not meaningful (capital-intensive, pre-profit).
- **No analyst coverage → no implied-upside column.** If consensus targets are unavailable, skip that column for that row and note it. Do not invent.
- **Mind the market-cap gap.** A $500M alternative to a $3T seed carries execution and liquidity risk that must be flagged regardless of how good the multiples look.
- **No better alternative is a valid answer.** When the seed is genuinely the best expression of the thesis, say so. The skill exists to verify conviction, not to manufacture trades.
