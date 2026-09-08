# Reader handoff for evidence preparation

The reader repository owns complete X post/Article text, expanded external links, source-language metadata, and actual attachment URLs. This worktree consumes those facts; it does not change reader behavior or guess fields in a future reader export.

Both required lists remain part of acquisition:

- `https://x.com/i/lists/1986290701492232693`
- `https://x.com/i/lists/1522014550211457024`

After both repositories are updated, create a new evidence bundle, then build this explicit mapping against its ID. Do not edit the existing frozen bundle to add corrected reader text. Source text changes require a new bundle and new mapping.

The importer now preserves the reader's `text_source`, `text_complete`,
`incomplete_text_reasons`, expanded `article_urls`, aligned `image_captions`, and
`reply_tweet_id` alongside language and image URLs. Explicit completeness takes
precedence over length: a complete long note can be full, while unknown or
incomplete text remains partial. A native Article preview still requires its body.
Expanded links join the reference queue even when absent from the visible text.
Legacy bundles retain their original content addresses and review identities.

The September 8 interim rebuild uses the currently installed reader while its
translation update is being developed separately. It uses Kimi translation; it
does not imply that X-provided translation ingestion has been implemented.

## Mapping file

```json
{
  "bundle_id": "<64-character bundle ID>",
  "documents": {
    "<existing document ID>": {
      "source_text_sha256": "<original document text SHA-256>",
      "language": "ko",
      "image_urls": ["https://pbs.twimg.com/media/<actual-image>.jpg"],
      "local_images": []
    }
  },
  "references": {
    "<existing reference ID>": "https://publisher.example/full-article"
  }
}
```

All IDs must already exist in the named bundle. Every document override requires its original text hash. References inherit the bundle's referring-post binding. Missing/unknown IDs or stale hashes fail before processing. `language` may be a BCP-47-style tag such as `ko`, `ja`, `zh-TW`, or omitted. Han-only text without trustworthy metadata remains ambiguous; Latin text is not automatically English.

`image_urls` contains actual public image destinations, never t.co landing pages or `/photo/1` web pages. For explicitly provided local images, use absolute paths in `local_images`. Image results preserve the bytes by content hash and keep every referring document/locator. Supported initial image formats are static JPEG, PNG, and WebP, up to 10 MiB and 20 million pixels. Animated, oversized, corrupt and unsupported images remain processing gaps.

Document URLs already in the bundle and mapped image URLs are combined, with identical locators deduplicated. The mapping supplies extra evidence metadata; it does not silently remove earlier recorded attachments. Review incorrect earlier metadata as a reader issue when building the new base bundle.

Kimi image requests allow 45 seconds for a response, with a 5-second connection
timeout and the existing 2,048-token output limit. A real chart timed out at the
earlier 20-second limit and completed in 21.07 seconds during diagnosis. Complete
responses still require visual review; unreadable values and unfinished model
responses are not accepted as facts.

## Article browser recovery import

The HTTP parser preserves readable JSON-LD or semantic article text but keeps completeness unverified. Inaccessible pages, ambiguous bodies, unsupported content types and missing destinations appear in `article_followups.csv`. Native X Article recovery remains the reader's responsibility.

Assisted browser recovery can supply accessible exact-match article text through:

```json
{
  "bundle_id": "<base bundle ID>",
  "articles": [{
    "reference_id": "<existing reference ID>",
    "destination_url": "<exact requested destination from mapping or reference>",
    "article": {
      "title": "Article title",
      "text": "Complete captured article text",
      "final_url": "https://publisher.example/story",
      "canonical_url": "https://publisher.example/story",
      "method": "browser_import",
      "capture_status": "full",
      "warnings": [],
      "response_sha256": "<SHA-256 of UTF-8 text field for browser imports>",
      "retrieved_at": "<actual capture time with UTC offset>",
      "match_basis": "How URL, title and publisher establish this is the referenced article"
    }
  }]
}
```

`retrieved_at` must be explicitly supplied. Use `partial` whenever the captured body is incomplete or uncertain. `full` is an attributed completeness assertion for review, not automated proof or evidence approval. Preserve access restrictions; do not bypass logins/paywalls.

## Translation import

Kimi K2.6 is selected for translation, using the configured OpenCode Go connection. Enable it with `--allow-translation-calls` on the text stage; model image calls remain separately controlled. Without this flag, the text stage preserves source segments and explicit unavailable translations. The library still supports an injected callable translator with provider/model/policy metadata. See [live translation validation](kimi_translation_validation_2026-09-08.md) for observed limitations.

The Kimi adapter protects digit-written CJK scale quantities during the request and restores their exact source notation afterward. Thus translated prose can contain `100억원` or `1兆2,500億円`; it does not automatically rescale these into billions/trillions. The review packet includes a unit legend. Missing or duplicated protected quantities fail explicitly. This does not detect every semantic error or preserve unrecognized/spelled-out quantities automatically.

To import attributed translations offline, take a text-stage result from a sealed preparation and provide one string or `null` for every original segment, in the same order:

```json
{
  "bundle_id": "<base bundle ID>",
  "preparation_id": "<preparation containing the text result>",
  "translations": [{
    "result_id": "<text-stage result ID>",
    "source_text_sha256": "<result request input_sha256>",
    "translations": ["First translated segment", null],
    "provider": "<attributed provider or human reviewer>",
    "model": "<model ID or human>",
    "policy_version": "<translation policy version>",
    "generated_at": "<actual translation time with UTC offset>"
  }]
}
```

Imports attach directly to the stored original segments, supplied/inferred language and language warnings. They do not repeat language detection or segmentation. Identity segments (including whitespace) must remain byte-for-byte equal to their stored originals. Both live translations and imports use the same finalization and numerical checks. Unknown result IDs, changed source hashes and missing segment slots are rejected. `null` retains an explicit missing translation. Number, sign, currency, large-unit changes and unchanged foreign-language output are review warnings. These checks do not establish semantic translation accuracy. Imported provenance is supplied by the importer, not independently verified with the named provider.
