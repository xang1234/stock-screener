# Interim real-evidence rebuild — September 8, 2026

This run prepares evidence while the separate xui-reader translation update is in progress. It does not generate extractions, theme labels, or benchmark scores. Evidence approval remains pending.

## Capture and implementation

Both required X lists were read with the existing authenticated reader and 50 posts selected per list. The resulting window contains 97 distinct posts, including three shared posts, and 49 image attachments. It has English, Korean, French, Chinese and Hebrew text, but no Japanese posts; the earlier Japanese model checks were synthetic and do not fill that coverage gap.

The importer now keeps explicit reader completeness, text provenance, expanded article destinations, aligned image captions and reply IDs. Complete long notes no longer depend on a length heuristic. Unknown completeness remains partial, and native Article previews still require their full bodies. Older sealed bundles retain their content and review identities.

A real chart reproduced the image adapter's 20-second timeout. The same image completed in 21.07 seconds with a longer allowance. The image read timeout is now 45 seconds, keeping the five-second connection timeout and 2,048-token output limit. The resumed run retained saved outputs, preserved unsuccessful attempts in history, and used up to three concurrent image calls. Kimi remains the only model service used.

## Evidence findings

- All 49 images were attempted; 48 have model outputs. Outputs with uncertainty remain reviewable, not approved.
- Of 97 posts, 16 have explicit complete-text metadata and 81 have unknown completeness. Unknown is not proof of truncation.
- No complete linked article body was recovered. The OWL article destination redirects to a subscription/feed introduction; that captured fragment must not count as the article. A native X Article remains unavailable through the public HTTP route.
- Most shortened links return to the referring X post. The review packet distinguishes those from article gaps and links to other posts needing context.
- Three manual image spot checks found accurate press-release and option-table transcription, plus a guessed chart level of 468.00 where the source label shows 450.25. The model had warned that the label was unclear.
- Manual language review flagged two Hebrew semantic errors, including rendering retweeting as deleting, and one Korean company-name preservation gap. A valid translation response is not a correctness verdict.

The complete sample remains unapproved. Correct the flagged derivatives and resolve or explicitly exclude missing article/context evidence before any extraction stage. After the reader translation work lands, capture a new immutable bundle to assess X-provided English translations alongside originals.

## Local artifacts

Source content and model outputs are intentionally ignored by Git:

- `data/xui-reader/theme-evaluation/pilot-rebuild-20260908/review/START_HERE.md`
- `data/xui-reader/theme-evaluation/pilot-rebuild-20260908/review/priority-review.md`
- `data/xui-reader/theme-evaluation/pilot-rebuild-20260908/review/article-review.csv`
- `data/xui-reader/theme-evaluation/pilot-rebuild-20260908/review/preparations.csv`
- `data/xui-reader/theme-evaluation/pilot-rebuild-20260908/review/review-assessment.json`

The content-addressed base/preparation IDs and final outcome counts are recorded in the review assessment. The original `pilot-20260908` bundle was verified unchanged.

## Validation

166 relevant tests passed. Changed Python files passed Ruff, and the diff passed whitespace checks. Tests cover updated reader metadata, explicit completeness, expanded article references, backward-compatible bundle identities, and the image timeout contract. Two existing dependency warnings remain.
