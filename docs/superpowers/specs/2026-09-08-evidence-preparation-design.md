# Article, language and image preparation

Scope accepted in chat: reader fixes belong to the reader repository. This worktree implements article recovery, multilingual preparation and image processing. Rebuild the sample only after both repositories are updated. Evidence approval remains required before theme extraction.

## Boundaries

Keep existing version-1 bundles byte-for-byte unchanged. Preparation is a separate immutable artifact bound to a base bundle ID, document/reference IDs and original content hashes. No changes to theme extraction, rankings, production ingestion, application database or the reader repository.

Reader handoff: a future reader adapter supplies expanded article destinations, actual image URLs and source language. Until its export schema is agreed, accept an explicit, validated input file mapping existing reference IDs to destination URLs and document IDs to language and media URLs. Reject unknown IDs and mismatched source hashes. Do not guess new reader fields or treat t.co links as image bytes.

## Article recovery

Use bounded public HTTP reads with redirects checked individually, no ambient credentials, timeouts and body-size limits. Parse JSON-LD articleBody or semantic article/main content; preserve title, canonical/final URL, retrieved time and response hash. A readable body is not proof of completeness: flag login/paywall/challenge pages, short snippets and bodies without clear boundaries. Do not follow links recursively or bypass access controls.

Unusable HTTP results create a browser/search follow-up queue. Existing agent tools can supply an exact-match body through a validated fallback import, with source URL, method and capture time. Native X Article recovery stays with xui-reader. One document version may be linked to multiple referring posts without independent-evidence credit.

## Multilingual preparation

Detect Korean Hangul, Japanese kana and Chinese Han; preserve explicit language metadata and flag mixed/ambiguous scripts. Do not classify every Latin-script document as English. Translation operates on bounded paragraph segments and preserves all source text. A failed/missing segment prevents a complete translation status. Keep paragraph alignment, provider/model, prompt version and actual generation time. Mark discrepancies in numbers, currencies and large-number units for review; such checks are warnings, not proof that a translation is accurate. No theme/ticker inference.

## Images

Read actual public image URLs or explicitly provided local image files. Validate bytes/type/dimensions and limits before processing. Preserve the image by content hash. A configured vision-capable adapter returns separate visible-text transcription, observed visual content, image type and uncertainty. OCR-like transcription must preserve original language, units, date labels and unreadable regions. Do not invent numerical chart values or investment conclusions. Translation of recognized text is a separate derivative. Repeated identical images reuse processing while retaining every parent document.

## Tool selections and approval status

The user permits configurable services but explicitly requires tools/models to be highlighted and approved before use. No new tools are installed or invoked by this design.

| Purpose | Proposal | Change and data flow |
| --- | --- | --- |
| Article HTTP and parsing | Installed HTTPX and BeautifulSoup | Public publisher requests and local parsing; no new dependency |
| Rendered article fallback | Existing Codex browser/web tools | Assisted recovery of accessible publisher pages; no new crawler/service |
| Translation | Existing MiniMax M2.7 provider route | Selected source text sent to MiniMax; API usage applies |
| Local OCR | PaddleOCR — deferred | Not part of the initial implementation; any later installation/model downloads require approval |
| Image transcription and chart interpretation | OpenCode Go, `kimi-k2.6` — approved | User explicitly selected Kimi K2.6 for both tasks. New preparation provider adapter; selected images/context sent through Go; subscription quota applies |

Kimi K2.6 is approved; the article tools and MiniMax translation proposal remain pending approval. Model selection is not a claim of measured accuracy on the corpus. Korean/Japanese text, small financial numerals, tables and chart axes require controlled-fixture validation. A provider/tool change requires fresh approval. Missing configuration/capability stays explicit.

### OpenCode Go selection and optional local OCR

The user explicitly replaced the direct Z.AI image proposal with OpenCode Go and, after comparing Kimi, Qwen, DeepSeek and GLM, approved Kimi K2.6 for transcription and chart interpretation on 2026-09-08. [Go documents its availability and chat-completions endpoint](https://opencode.ai/v2/docs/console/go), and [Kimi documents visual input support](https://platform.kimi.ai/docs/models). Use model ID `kimi-k2.6` at `https://opencode.ai/zen/go/v1/chat/completions`. Image transport and the configured parameters passed four controlled live fixtures on 2026-09-08 after adding Go client/session headers. Korean/Japanese transcription and the tested chart values passed; two commentary counting errors remain documented. See [live validation](../../theme_evaluation/kimi_validation_2026-09-08.md). Real-corpus accuracy remains unmeasured; follow the existing review-before-extraction sequence. Do not silently switch models or providers.

Use Go for both visible-text transcription and chart observations, retaining separate output fields and explicit unreadable regions. Local OCR installation is deferred. Consider PaddleOCR later if reviewed Korean/Japanese text or numerical labels show errors that a second OCR pass can resolve, with separate approval. No claim that vision transcription alone matches specialist OCR accuracy. Transcription preserves the source language; approval of image transcription does not select a provider for translation into English.

PaddleOCR footprint reference, checked 2026-09-08 (PP-OCRv5 mobile configuration, not PaddleOCR-VL):

- CPU-only inference is supported. The [published benchmark](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html) reports 2,220 MB peak RAM and 1.75 seconds/image for mobile models, versus 4,021 MB and 4.34 seconds/image for server models. The Xeon benchmark uses roughly ten CPU cores of utilization; these are not measurements on this Apple Silicon host. Plan 3–4 GB RAM for one mobile worker as headroom, not a guaranteed minimum; bound concurrency and image dimensions.
- [Model storage](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html): mobile detector 4.7 MB plus general recognizer 16 MB, approximately 21 MB before extra language models and optional preprocessing.
- Published compressed downloads: [PaddleOCR 3.7.0 wheel](https://pypi.org/project/paddleocr/) 146.8 kB; [PaddlePaddle 3.3.1 runtime](https://pypi.org/project/paddlepaddle/) 104.5 MB for macOS ARM64 or 194.8 MB for Linux x86-64. These exclude transitive dependencies, unpacking and model files, and are reference sizes rather than a compatibility-tested version lock.
- No full installation or Docker image has been built. Reserve a few GB of disk as a preliminary planning allowance; actual installed and compressed/uncompressed container sizes require measurement of a pinned CPU-only build. No CUDA image is needed for this proposal.

## Runtime and review

Expose explicit preparation CLI commands; ordinary verify/review remains offline. Model configuration is explicit and provider adapters are injected in tests. No live model calls, new list reads or real-sample rebuilding during implementation. Missing model capability is an unavailable result, not a silent text-only fallback for images.

Seal preparation files atomically with hashes. Preparation manifest v2 separates immutable attempt history from explicit current evidence. Unavailable retries cannot erase useful results for unchanged/uncaptured inputs; captured changes supersede old derived translations at the same locator. Only current article/image versions feed text preparation. Stage-specific result models own payload validation and derive status/warnings. Translation imports attach to stored segments without rerunning language detection or segmentation. Cache only successful exact-input results, keyed by source bytes/text, processing version and configured model; retain original processing timestamps on cache hits. Failures are retryable and are not cached as successes. An immutable request index makes lookups independent of unrelated results/assets; explicit verification audits stored assets/results and repairs missing index entries after validation. Render a separate Markdown/CSV report containing article bodies, original/translated segments, images/transcriptions, provenance and gaps. All acquired bodies/images/results stay in ignored local storage.

## Acceptance

- Version-1 bundles retain their IDs and existing tests pass.
- Korean/Japanese and mixed text can be prepared without changing originals.
- Long documents are segmented without omitted text; missing translation pieces remain partial.
- Paywalls, unusable articles, corrupt/oversized images and model failures remain visible.
- Browser recovery imports bind to the exact requested reference and source bundle.
- Images are displayed with their transcription and interpretation, never counted as independent corroboration of their source article.
- Identical preparation inputs reuse successful processing; changed model, prompt, text or image cannot hit the old cache.
- No extraction occurs and the existing pilot is not rebuilt.
