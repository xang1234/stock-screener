# Grounded theme extraction

Theme names remain open. Each extraction can receive canonical company identity,
attributed cached business information, and explicitly related reviewed evidence.
The theme stays separate from the development.

## Application behavior

The application resolves explicit cashtags against the active local universe. It
reads cached company profiles without a network call. Business fields need a
recognized provider and a provider timestamp no more than 180 days old; future,
stale, conflicting or unattributed data is withheld with a warning. Descriptions
are limited to 1,500 characters and company context to 20 companies. Missing
business data remains unknown. The optional `resolved_symbols` reader parameter
is for callers with already resolved identities; the application does not populate
it from heuristic name matches.

The extractor treats profiles as background, not evidence of a new event or
sector demand. Source, article and image text cannot supply instructions. It
receives related context outside its existing 10,000-character primary-text limit.

New ThemeMentions retain the context supplied to their extraction, exposed by the
mentions API. Migration `20260911_0039` adds this nullable JSON field. Apply it as
part of deployment; old mentions remain null. This worktree does not migrate the
running application automatically.

Legacy ContentItems do not have the offline acquisition bundle's article/image
relationships. The application therefore adds company context automatically but
does not guess attachment links. A caller with a validated `GroundingContext` can
supply it explicitly.

## Frozen evaluation

1. Export public cached company context into an input-ID keyed JSON file using a
   read-only session. Keep provider timestamps and the snapshot time.
2. Run `extract_theme_evidence.py prepare-grounding` with `--run`, `--bundle`,
   `--store`, `--company-context PATH`, `--as-of ISO_TIMESTAMP` and `--output PATH`.
   The output is created exclusively. Input admission is rederived from its
   original reviewed preparation and approval; excluded inputs stay excluded.
3. Run `generate` with the existing explicit model-call flag, isolated evaluation
   database/reference snapshot, plus `--grounding PATH` and
   `--grounding-bundle BUNDLE_PATH`. The latter proves exact article relationships
   against the sealed bundle. Without a packet, evaluation passes an explicitly
   disabled context and does not query newer company profiles.
4. Use `review` to produce Markdown, `grounding.csv`, and extraction details. Each
   new grounded success or failure stores its actual context. Old record
   serialization and run hashes are preserved.

Attached images require the same parent source ID. Articles require a persisted
followup edge to the exact admitted article or derivative reference. A partial
followup can contribute an article only when that exact article was admitted by
the evidence review. No link is inferred from a similar URL, author or topic.
Related evidence is capped at 6,000 characters, images first, with truncation and
omission warnings. It remains part of the same source family, not independent
confirmation. Full article chunking is a separate next step.

When an admitted image transcription has no direct company facts, a newly built
packet may copy cached company facts from its admitted parent post. The parent
must have the same source ID, be an admitted original or translation input, and
be available by the packet's `as_of` time. The original input is preferred;
translation is considered only when that parent has no usable company context.
The image keeps its own direct facts when it has them. This is a one-hop copy of
the parent's frozen facts, not a traversal through the parent's images, articles,
or other evidence. The packet records the parent input ID in a warning and
validation requires the image's copied companies to exactly equal that parent's
context. Company and related-evidence limits still apply unchanged.

Do not alter a saved packet or an extraction run to apply this rule. Rebuild a
new packet from the same approved input run, sealed bundle, preparation store,
read-only company-context export, and recorded `as_of` time, then write it to a
new exclusive output path. Verify its digest, admission coverage and parent
markers before any separately versioned replay. The previous packet and all of
its outputs remain reproducible evidence of the earlier treatment.

Context availability is the current snapshot time, including the company
snapshot; each source retains its own availability timestamp. This pilot cannot
measure historical detection speed. Profile attribution is not an independent
accuracy audit, and supplying grounding cannot guarantee correct model output.

A subsequent [claim support review](claim_support.md) now checks non-empty candidates
before they enter the theme pipeline. It retains rejected claims in a separate audit.
