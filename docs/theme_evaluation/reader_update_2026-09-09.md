# Evidence regeneration with updated xui-reader — September 9

Reader revision `1a3bfc4` was used to capture both required lists again, selecting 50 posts per list. The new snapshot contains 99 distinct posts and 33 images. Earlier evidence bundles remain unchanged. No extractions or benchmark labels are generated; evidence approval remains pending.

## Integration

The evidence importer preserves X translation status, original-text hash, post identity, provider, display mode, capture time and failures. It rejects mismatched identities/hashes and malformed captured text. A valid captured X translation is preferred during text preparation without calling Kimi; unavailable or failed X captures use the approved Kimi fallback. Numeric warnings remain active, and captures are not confused through a cache keyed only by the original text.

175 relevant tests pass. Changed Python files pass Ruff and whitespace checks. The original pilot and interim bundle identities are preserved.

## Results and limitations

- 13 X translations were captured; 18 were unavailable and three failed capture. The remaining 65 posts are English originals.
- Thirty of 33 image attachments have model outputs after one bounded retry. Earlier failed attempts are retained. Uncertainty and remaining failures require review.
- Readable text was recovered from three distinct article destinations, with duplicate short/expanded links retained as separate references. None is automatically certified complete. Some article bodies include page controls or have access/translation gaps.
- The sample contains English, Korean, Chinese and French text plus other/nonlinguistic tags. Japanese and Hebrew are absent from this window.
- Twenty posts have explicit complete-text metadata; 79 remain unknown. Unknown does not prove truncation.

## Reproducible reader capture issue

[Post 2097617015238500778](https://x.com/i/status/2097617015238500778) discusses a Samsung share purchase, including `718만주` and `1.9조원`. At `2026-09-09T09:31:40.380368+00:00`, the reader returned:

- `status`: `captured`
- `display_mode`: `automatic`
- `text`: `Samsung`
- `failure_reason`: null
- Original hash: `sha256:82390db58687dbb27615c8f67acd5580a90da628a7a90d9179cf8a652a27b241`

The English output omits the transaction and both quantities. The original hash matches, so identity binding alone cannot establish translation completeness. Preparation flags the numerical loss; this translation remains on hold for review/correction. No reader code was changed in this worktree task.

The exact public source record is retained in `data/xui-reader/theme-evaluation/reader-update-20260909/reader-gap.json` for investigation in the reader repo. The final packet begins at `data/xui-reader/theme-evaluation/reader-update-20260909/review/START_HERE.md`, with original/selected translations, X failures, article classification, image evidence, CSV detail and exact artifact IDs.
