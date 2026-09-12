# Grounded theme extraction

The user approved grounding company identities/exposures and using related approved
image/article evidence with its parent post. Theme names remain fully open. This
increment does not change ranking, merge themes or implement full-article chunking.

## Design

1. Resolve explicit cashtags (and already resolved company-name symbols) against
   the active local stock universe before extraction. Supply canonical name and
   available attributed cached business information. Identity-only, missing,
   stale and conflicting information must remain explicit. Never look up remote
   profiles automatically, guess an issuer from a ticker, or invent an exposure list.
2. Build an immutable evaluation grounding packet from an existing approved input
   run and its bound bundle/preparation store. Only admitted inputs can be attached.
   Image transcription uses its parent post; articles require an explicit resolved
   followup relation. No URL resemblance, author similarity or topic matching.
   Source evidence keeps its IDs, URL, text, warnings and availability timestamp.
   Shared/reposted articles are not independent corroboration. Context availability
   is the current snapshot time, including the cached company context. Individual
   source availability times remain visible separately; this is not a historical replay.
3. Keep grounding separate from the 10,000-character primary-content budget. Bound
   related context and company fields explicitly and record omissions/truncation.
   The model may use business context to interpret the source, not as a fresh event
   or evidence of sector demand. Source instructions are untrusted data.
4. The shared extractor accepts an optional validated grounding context. Application
   calls obtain local company context from explicit cashtags only (heuristic name
   matching is not trusted for profile attachment); offline evaluation supplies a frozen packet
   so it cannot silently read a newer unpinned company profile. No related evidence
   is guessed for legacy ContentItems without stored acquisition relationships.
5. Evaluation generation binds the actual grounding packet into new records and
   exposes it in review. Legacy input IDs, records, run hashes, eligibility and
   exclusions remain unchanged. New context uses a separate treatment artifact.

## Verification

Tests cover explicit-symbol resolution, company-name preservation, unknown symbols,
stale/unattributed profiles, wrong-parent/excluded/context leakage, archive round
trips, changed packet hashes, source/availability provenance and prompt delivery.
Focused NBIS controls compare the same post with identity/business context and its
admitted image. Unrelated image controls must not appear in the prompt. A limited
live comparison uses only already approved providers and evidence after unit checks.

## Scope limits

Cached profiles are attributed reference data, not independently audited truths.
A missing profile is a visible limitation; company identity alone cannot prove a
specific industry. The pilot uses current snapshots and makes no historical-speed
claim. Prompt grounding improves the available evidence but does not prove that
all generated assertions are correct; inspect live outputs rather than declaring
hallucinations impossible.
