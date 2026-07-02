# Supply Chain Discovery

description: Traces upstream suppliers, downstream customers, and adjacent enablers of a company or theme using bidirectional SEC filing search, deal-event structured data, and company-ontology mapping. Use when the user wants exposure to a theme via a less-crowded name, wants to find the hidden picks-and-shovels around a household-name beneficiary, or wants a full industry map. Triggers on "supply chain", "hidden champions", "picks and shovels", "who supplies X", "who depends on X", "upstream exposure to [theme]", "map the value chain", or "what companies feed into X".

The household names get all the analyst coverage. The companies that supply them, enable them, or depend on them are often better businesses at better prices — and almost nobody is looking.

When a theme takes hold — AI compute, GLP-1, data center buildout, defense modernization — the prime beneficiary gets bid up fast. But the supply chain around it is slower to reprice. A specialty materials company supplying a critical input, a cooling systems maker whose entire addressable market just doubled, a contract manufacturer whose capacity is suddenly scarce — these are the hidden champions. Less covered, less crowded, often with more durable economics because they sit at structural chokepoints.

We trace the chain — around a single company, along a theme, or across a full industry — using a company-ontology scan to generate the hypothesis set, then SEC filing search and deal-event data to actually confirm each relationship.

## Workflow

### Step 1: Frame the node

Pin down what we're tracing:
- A single company (ticker named)
- A theme (AI compute, GLP-1, DC buildout)
- A full industry (HBM supply chain, LNG-liquefaction equipment, etc.)

If ambiguous, ask. The scope affects how broadly we cast in Step 2, not how we do the work.

### Step 2: Hypothesize counterparties — analyst sketch + ontology scan

Generate the candidate list from two complementary sources:

**(a) Analyst sketch from memory.** List the suppliers and customers you would *expect* to find — the obvious incumbents, the second-tier specialists, names that come up in recent earnings calls and news flow. This captures what an experienced reader of the sector already knows, including recency signals the ontology may not yet reflect.

**(b) Company-ontology scan.** Use the ontology layer to enumerate companies by what they *do* — by product, technology, customer base, supply-chain position. The ontology is a classifier, not a relationship engine; it answers "all companies that make X" or "all companies that serve Y," not "who supplies ticker Z" (that is Step 3's job). The value of the scan is that it surfaces foreign-listed, lesser-covered, and adjacent names that wouldn't come up from memory alone.

Useful ontology query patterns:
- For a company (NVDA as the node): "All companies making HBM memory"; "All advanced-packaging / CoWoS service providers"; "All AI-cloud / hyperscaler operators." Pick categories that are *plausibly* in the chain, then let Step 3 verify which actually are.
- For a theme: "All companies upstream of HBM memory production"; "All US electrical contractors with disclosed data-center exposure"
- For an industry: "All publicly listed Tier-1 LNG-liquefaction equipment suppliers"; "All foundry-related photomask, etchant gas, and CMP-slurry vendors"

Merge both lists into a single candidate set, deduping. Names that appear on both lists are higher-prior; names that appear on only one are worth flagging — the analyst-only ones may be too recent or too narrow for the ontology, and the ontology-only ones are exactly the kind of "hidden champion" this skill is designed to find.

The merged list is the seed for the bidirectional search in Step 3, and a record of expectations: if a candidate doesn't appear anywhere in filings, that itself is a signal (vertical integration, multi-sourcing, or the hypothesis was wrong).

### Step 3: Bidirectional SEC search — do not skip a side

For each candidate counterparty, run **two** searches and reconcile both:

**Direction A — Does the node disclose the counterparty?**
Search the node company's 10-K, 10-Q, S-1, and proxy filings for the counterparty's name. Capture context: customer concentration, key supplier, critical input source, joint-venture partner, milestone payor, related-party transaction.

**Direction B — Does the counterparty disclose the node?**
Search the counterparty's filings for the node company's name. Capture the same context.

Why both: a company may not name a supplier for negotiating reasons but the supplier *will* name them in customer-concentration disclosures. Conversely, a customer may name a supplier critical to their cost structure that the supplier doesn't disclose by name. **Single-direction search is the single largest cause of missed relationships.**

Tag each confirmed relationship with: direction (supplier / customer / partner), evidence type (concentration %, qualitative mention, contract value), and filing citation (filing type, date, section).

### Step 4: Cross-check with deal-event structured data

Run the company deal-events search on both the node and the candidate counterparties. Disclosed deal events surface what the periodic filings often lag:
- Multi-year supply agreements (term, $ value, exclusivity)
- Joint ventures and capacity-reservation contracts
- M&A involving suppliers or customers (a tell about vertical-integration intent)
- Strategic partnerships, capacity prepayments, off-take agreements

Deal events frequently catch the most recent — and most actionable — relationships before they appear in the next 10-K.

### Step 5: Second ontology pass (when the search uncovers a recurring third party)

If the bidirectional search in Step 3 keeps surfacing a name you didn't hypothesize in Step 2 — a vendor, a co-supplier, an integrator showing up in multiple counterparties' filings — re-run the ontology scan with that name as the seed. Recurring unhypothesized names are often the real chokepoints in the chain.

### Step 6: Score each name

For every confirmed supplier / customer / enabler, assign:

| Field | What it captures |
|---|---|
| Exposure purity | % of revenue/profit from the theme/node — pure-play vs diluted |
| Position in chain | Critical chokepoint, replaceable input, optional enabler |
| Listing & liquidity | Public/private; market cap; foreign listings noted |
| Pricing | Fwd P/E, EV/EBITDA vs. the node and vs. peers — is the supply-chain discount earned? |
| Coverage / crowding | Sell-side coverage, insider activity, short interest |
| Evidence | Both-direction SEC confirm / single-direction with flag / deal-event only |

### Step 7: Present the map

Output a value-chain table — node in the middle, upstream suppliers to the left, downstream customers to the right — annotated with the Step 6 score. For the 1–3 highest-conviction supply-chain picks, write a 3-bullet thesis: shared driver, what the market is missing, and the catalyst that re-prices the name.

## Important Notes

- **Bidirectional or it didn't happen.** A relationship found in only one direction needs a flag — either the asymmetry is informative (negotiating leverage, materiality threshold) or one of the filings is stale. Don't promote a single-direction find to "confirmed."
- **Customer concentration cuts both ways.** A supplier with 40% revenue from one customer is a chokepoint *or* a hostage — verify which by reading the contract terms (term length, exclusivity, take-or-pay).
- **Foreign suppliers matter.** Many of the most interesting hidden champions are Taiwan/Korea/Japan/Netherlands-listed. The ontology pass should not be US-only unless the user specifies.
- **Recency over the 10-K.** A 10-K is a year stale by definition. Deal events, 8-Ks, and recent transcripts are where the most actionable relationships first appear.
- **Pure-play wins on slope, scale wins on durability.** A 90%-exposed small supplier delivers more torque on the theme; a 20%-exposed compounder survives a theme rotation. Match the recommendation to the user's intent.
- **Vertical integration is a leading indicator.** When a node company starts naming a supplier in filings *and* the deal-event stream shows a JV or acquisition motion, the chokepoint is about to disappear. Flag immediately.
- **Don't double-count.** If a name appears in both "supplier to X" and "customer of Y" lists for the same chain, it's a node not a candidate — model it once.
