# Company exposure research is an issuer-centric assessment layer published through economic serving generations

Company exposure research ("why does this issuer participate in this theme, in which role, at what commercial stage, with what disclosed materiality, and on what evidence?") is a dedicated, issuer-centric layer of immutable evidence, claim and assessment revisions. It feeds the existing economic serving generation and membership projection. It is not a second theme catalog, a mutable company profile, or an independent publication authority.

Spec: `docs/superpowers/specs/2026-09-25-company-exposure-map-design.md` (R2).

## Context

ADR-0005 publishes economic taxonomy semantics as sealed snapshots and runtime interpretation as append-only facts selected by a serving generation. Source-derived `ThemeConstituentExposure` rows are attached to one source `ClaimAssignment`, and `exposure_strength` is populated from extraction confidence. Neither can represent a multi-document, primary-backed verification of an issuer's business, nor a disclosed materiality figure.

Research also crosses listings (one issuer, several securities), languages and markets, and it consumes a shared LLM subscription allowance and optional paid search.

## Considered Options

- **Extend source claims into a mutable company profile** — rejected: research refreshes would masquerade as source corrections, erase other origins' contributions, and make "latest" reads bypass generation pinning.
- **A separate exposure catalog with its own live pointer** — rejected: readers could combine an exposure map newer than the membership and grounding it claims coherence with.
- **A dedicated assessment layer published through the existing coordinator** — chosen.

## Decision

1. **Six separate objects.** Source evidence (immutable document revisions and passages), exposure claims (one proposition, immutable revisions), assessments (a dossier per issuer–theme with immutable revisions selecting claims), membership decisions (per security, origin-aware), grounding context (bounded, pinned), and source observations (never manufactured by research).
2. **Research is not source evidence.** Research never creates `ClaimAssignment`, `ThemeObservation`, `ThemeMention`, development or source-family rows, and never increases source-attention counts.
3. **Manifest order.** A generation's input manifest captures research revision references and a reserved, unsealed exposure selection ID plus its input fingerprint. The selection is filled and sealed outside the exclusive publisher lock; the generation then pins the sealed selection. The selection's semantic hash excludes its parent manifest ID, so no hash is defined circularly. Old generations have no exposure section and remain valid.
4. **Safety invalidation is blocking-only.** A current hold, expiry, issuer-link or role-policy disqualification can block or defer a new automatic action; it can never substitute newer unpinned evidence into an old result, and it never removes an existing member automatically.
5. **Issuer attestation facade.** Existing Social administrator attestations are imported with exact configuration version and audit provenance. After the facade switch there is one accepted issuer-link selection, exposed through the existing Social API shape; the old transaction-owning replacement is never called inside a fenced write.
6. **Compatibility owner keys.** Research membership delivery uses the existing projection outbox with logical owner `research-membership:<issuer_uuid>:<theme_uuid>`, projection kind `research_membership`, origin `exposure_research`. This is a transport key, not a source family; payloads replace only that owner's contribution.
7. **Resource honesty.** LLM research uses the OpenCode Go / Kimi subscription transport, accounted in requests and reported tokens with no invented dollar cost. Connect-phase failures are pre-dispatch; uncertain dispatches are never refunded and expire with their allocation period. Paid search requires explicit enablement plus a spending cap.
8. **Disabled by default.** `EXPOSURE_RESEARCH_MODE=disabled` and paid search off; each privilege (shadow research, generation reads, automatic additions, discovery, grounding) is enabled separately behind its own gates.

## Consequences

- Research tables live in dedicated `company_exposure_*` model modules, not in the economic runtime evidence module, but reuse its immutability conventions (ORM guards plus PostgreSQL append-only triggers).
- A US-only verify-only shadow preview may ship before full four-market launch; it is labelled `shadow_preview`, is not a serving-generation result, and cannot change membership or grounding.
- Automatic membership additions require a reviewed theme role policy and primary-backed current commercial participation; unknown materiality alone does not block them.
- Rollback to a binary that cannot read research contributions is blocked rather than hidden by synthesizing source rows.
