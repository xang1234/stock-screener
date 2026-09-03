# Stock Screener Context

Shared domain language for the stock screening platform. This file records project-specific terms so architecture reviews, implementation plans, and tests use the same names for the same concepts.

## Language

**Market**:
A supported trading region such as US, HK, IN, JP, KR, TW, or CN with one or more MICs, a primary calendar, a fallback default currency, and data refresh work. A Market may aggregate multiple MICs (e.g., Euronext groups XPAR, XAMS, XBRU, XLIS, XMIL, XOSL).
_Avoid_: region, locale

**MIC**:
ISO 10383 Market Identifier Code for an individual venue (XNYS, XNAS, XHKG, XPAR). The canonical exchange identifier inside SecurityMaster, calendar resolution, and snapshot keys.
_Avoid_: exchange code, venue code, bourse

**MIC Facts**:
Stable facts for one MIC, including calendar ID, timezone, default currency, and provider calendar ID when needed.
_Avoid_: market calendar facts

**MIC Alias**:
A backward-compatible informal venue name or legacy exchange label that resolves to one canonical **MIC** within a **Market** (e.g., US/NYSE → XNYS, IN/BSE → XBOM).
_Avoid_: exchange as canonical identifier

**Market Catalog**:
The backend-owned source of stable facts about supported Markets, including primary MIC, supported MICs, and capabilities.
_Avoid_: market constants, market config

**Market Capability**:
An app-visible feature available for a Market, such as breadth, group rankings, feature snapshot, or official universe.
_Avoid_: provider capability

**Field Capability**:
The availability state of a screening field for a Market based on provider coverage or computed data.
_Avoid_: market capability

**Provider Data Plan**:
The ordered provider execution strategy for one Market and dataset, with optional MIC-specific overrides.
_Avoid_: field capability matrix

**Default Currency**:
A fallback currency used only when creating identity before a row currency is known; MIC default currency is preferred over Market default currency when MIC context exists.
_Avoid_: market currency for FX

**Market Workload**:
The operational lifecycle state for work running against a Market, including refresh leases, activity progress, bootstrap readiness, and active holders.
_Avoid_: job status, task state

**Runtime Preferences**:
The mutable local choices for primary Market, enabled Markets, and Bootstrap state.
_Avoid_: market config

**SecurityMaster**:
The source of truth for a security's explicit identity fields: symbol, market, MIC, currency, timezone, and local code. The `exchange` field on existing models stores the MIC.
_Avoid_: ticker parser, suffix inference

**Universe**:
A named or structured set of tradable securities used by scans, refreshes, bootstrap, and static export.
_Avoid_: ticker list

**All Universe**:
The global Universe containing all active security rows across supported Markets, independent of Runtime Preferences.
_Avoid_: enabled markets universe

**Index Registry**:
The backend-owned source of index definitions used as Universe options, including owning Market, display label, and membership source.
_Avoid_: index enum

**Benchmark Registry**:
The backend-owned source of benchmark instruments used for relative-strength and scan comparisons.
_Avoid_: index registry

**Listing Tier**:
A board / segment classification within a Market and optionally a MIC (Euronext: Regulated/Growth/Access/Expand; HK: Main Board / GEM). Stored as an optional first-class Universe row field; used as a scan filter and breadth grouping axis.
_Avoid_: exchange group, board, segment

**Listing Tier Registry**:
The backend-owned source of canonical Listing Tier keys, display labels, and Market/MIC scoping.
_Avoid_: source tier labels

**Scan**:
A screening run over a Universe using one or more screening methodologies and filter criteria.
_Avoid_: search, query

**Scan Readiness**:
The backend admission decision that a Scan can run for a Universe with current enough data, required field coverage, and no blocking Market Workload conflict.
_Avoid_: market enabled state

**Bootstrap**:
The first-run hydration workflow that prepares selected Markets with universe, price, fundamentals, breadth, group ranking, feature snapshot, and initial autoscan data.
_Avoid_: setup, initial load

**Options Analytics Run**:
A bounded attempt to collect one selected option-chain observation for each member of a pinned US Candidate Cohort, calculate versioned metrics, and evaluate publication quality.
_Avoid_: options scan, flow scan

**Candidate Cohort**:
The deterministic union of Current Candidates and Continuity Candidates staged for one Options Analytics Run.
_Avoid_: options universe, watchlist

**Current Candidate**:
A security in the pinned published US feature run's independently capped Top Candidates or Leaders, with daily dollar volume strictly above USD 100 million.
_Avoid_: current leader

**Continuity Candidate**:
A recently absent former Current Candidate collected for no more than five US trading sessions so ticker history can survive cohort gaps; it is excluded from current rankings and publication coverage.
_Avoid_: stale candidate, reserve candidate

**Chain Observation**:
The single successful normalized payload for one security and selected expiration in an Options Analytics Run. Retries do not create additional observations.
_Avoid_: chain snapshot, raw Yahoo response

**Published Options Run**:
An Options Analytics Run that passed the Current Candidate coverage gate and was atomically selected for live and static reads.
_Avoid_: latest options run

**Model Estimate**:
An options metric whose value depends on documented assumptions rather than an observed market fact, including estimated GEX, gamma flip, and call/put walls.
_Avoid_: dealer position, market-maker fact

## Relationships

- A **Market Catalog** describes many **Markets**.
- A **Market** has one or more **MICs**; one MIC is primary.
- **MIC Facts** provide calendar and timezone data for each **MIC**; Market-level calendar calls use the primary MIC as fallback.
- Market-level display timezone is derived from the primary **MIC** and is not canonical for every MIC in the Market.
- A **MIC Alias** resolves to exactly one **MIC** within a **Market**; it is not a canonical venue fact.
- A **Market** may have one **Default Currency**, while each security row has its own currency for FX normalization.
- A **MIC** may have a narrower **Default Currency** than its **Market**.
- A **Market Workload** belongs to exactly one **Market**.
- **Runtime Preferences** choose which **Markets** are active locally; they do not define the **Market Catalog**.
- The backend exposes **Market Catalog** facts to the frontend through runtime capabilities.
- The **Market Catalog** records **Market Capabilities**; **Field Capabilities** are derived from provider coverage and computed-data coverage.
- A **Provider Data Plan** chooses provider execution order for a Market and dataset; a MIC-specific override may refine it for one venue.
- A **SecurityMaster** resolves securities within a **Market**, identified by **MIC**.
- Row timezone may be stored on a security for compatibility, but **MIC Facts** are the source of truth for calendar decisions.
- A **Universe** is scoped by **Market**, **MIC**, **Listing Tier**, index, custom symbols, or test symbols.
- The **Index Registry** defines index Universes; index membership rows are data, not Market Catalog facts.
- The **Benchmark Registry** defines comparison instruments; a benchmark is not necessarily an index Universe.
- The **All Universe** spans all active rows globally; enabled **Runtime Preferences** do not narrow it.
- A security in a **Universe** may have one **Listing Tier**, scoped by Market and optionally MIC.
- The **Listing Tier Registry** defines canonical tier keys; source labels are mapped during Universe ingestion.
- A security row is keyed by canonical symbol today, while `(Market, MIC, local code)` must remain unique for active rows.
- A **Scan** runs against exactly one **Universe**; that **Universe** may span multiple **Markets**.
- **Scan Readiness** is evaluated for a **Scan** and its **Universe**; it is not the same as a Market being enabled.
- **Bootstrap** creates **Market Workloads** for the primary and enabled **Markets**.
- One published US feature run supplies the **Current Candidates** for an **Options Analytics Run**.
- A **Candidate Cohort** contains Current Candidates plus bounded **Continuity Candidates**; only Current Candidates enter current ranking and publication coverage.
- Each cohort member has at most one successful **Chain Observation** per Options Analytics Run.
- Only a quality-approved **Published Options Run** is visible to readers; an unsuccessful attempt cannot replace it.
- Every **Model Estimate** carries its assumptions and must be presented as estimated rather than observed fact.

## Example Dialogue

> **Dev:** "Should the frontend import the **Market Catalog** directly?"
> **Domain expert:** "No. The backend owns the **Market Catalog** and exposes the current **Market** facts through runtime capabilities."
>
> **Dev:** "Should **Market Workload** decide whether a **Scan** can start?"
> **Domain expert:** "No. The **Scan** path owns scan-blocking policy; it asks **Market Workload** for current activity state."
>
> **Dev:** "Does the **Market Catalog** include enabled Markets?"
> **Domain expert:** "No. Enabled Markets are **Runtime Preferences**; the **Market Catalog** only describes stable supported-Market facts."

## Flagged Ambiguities

- "Market config" previously mixed stable **Market Catalog** facts with mutable **Market Workload** state. Resolution: stable facts belong to **Market Catalog**; operational state belongs to **Market Workload**.
- Queue names are **Market Workload** implementation details derived from **Market** codes, not **Market Catalog** facts.
- **Market Catalog** may include coarse provider capability facts, but provider routing policy remains behind provider-specific modules.
- "exchange" on existing models (`MarketProfile.exchanges`, `StockUniverse.exchange`) historically mixed MICs (XNYS) and informal venue names (NYSE). Resolution: **MIC** is the canonical identifier going forward; existing string aliases are kept as backward-compatible lookups but new code keys on MIC.
- **Market Catalog** owns canonical MIC facts only. **MIC Aliases** are Market-scoped compatibility lookup facts that belong to surfaces such as SecurityMaster or MarketRegistry and may be exposed to clients only under explicit alias fields.
- Legacy exchange alias resolution requires **Market** context unless the alias is explicitly known to be globally unambiguous.
- Calendar resolution is MIC-level. Market-level calendar APIs are compatibility defaults that resolve to the Market's primary MIC.
- "Market currency" previously meant both **Default Currency** and row currency. Resolution: **Default Currency** is fallback-only; FX normalization uses the security row's currency whenever row context exists.
- "Capabilities" previously mixed **Market Capabilities** with **Field Capabilities**. Resolution: **Market Catalog** owns app-visible Market availability; provider and screening-field coverage live outside the **Market Catalog**.
- **Provider Data Plan** is not a field matrix. Resolution: it owns provider execution order, fallback, batching, and provider keys; **Field Capability** owns field-level support.
- **Provider Data Plans** default to Market-level policy; MIC-specific overrides are used only when one MIC inside an aggregated **Market** has distinct provider behavior.
- Index definitions do not belong directly to the **Market Catalog**. Resolution: the **Index Registry** owns canonical index definitions; runtime capabilities may expose derived summaries.
- Benchmark definitions are separate from index definitions. Resolution: **Benchmark Registry** owns comparison instruments; **Index Registry** owns Universe index definitions.
- **Listing Tier** is not source-only JSON metadata. It is optional, queryable **Universe** row metadata; source-specific labels may remain in lineage metadata.
- **Listing Tier** labels are not global. Resolution: tier definitions/options are Market-scoped and may be MIC-scoped when a Market spans venues with distinct tier systems.
- **Listing Tier** definitions do not belong directly to the **Market Catalog**. Resolution: the **Listing Tier Registry** owns canonical tier definitions; runtime capabilities may expose derived summaries.
- **Listing Tier** changes are audit events, not active/inactive lifecycle statuses.
- "All" previously risked meaning "all enabled Markets." Resolution: **All Universe** means all active rows globally; runtime jobs use explicit Market-scoped Universes.
- Multi-Market scans are not multiple **Scans** by default. Resolution: one **Scan** has one **Universe**, and row-level Market/MIC/currency plus mixed-market policy govern result semantics.
- **Universe** row identity remains canonical symbol for now, but `(Market, MIC, local code)` is a required active-row invariant.
- "Scan-ready Market" previously risked meaning only enabled in Runtime Preferences. Resolution: **Scan Readiness** is a scan-admission decision using Universe data, freshness, field coverage, and Market Workload state.
