# Reader handoff for evidence preparation

The reader repository owns complete X post/Article text, expanded external links, source-language metadata, and actual attachment URLs. This worktree consumes those facts; it does not change reader behavior or guess fields in a future reader export.

Both required lists remain part of acquisition:

- `https://x.com/i/lists/1986290701492232693`
- `https://x.com/i/lists/1522014550211457024`

After both repositories are updated, create a new evidence bundle, then build this explicit mapping against its ID. Do not edit the existing frozen bundle to add corrected reader text. Source text changes require a new bundle and new mapping.

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

Until a live translation service is selected, the text stage preserves paragraph segments, language metadata, and explicit unavailable translations. Translation is separate from the approved Kimi image transcription/interpretation. The library also supports an injected callable translator with provider/model/policy metadata, so an approved service can be connected without changing evidence records.

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
