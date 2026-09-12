# Live attachment evidence

The approved design connects ordinary ingestion to the existing article, translation and image preparation components. Preserve attached photo and expanded article references; persist their parent relation, provenance and preparation state; prepare asynchronously with bounded retries and cached successful results; supply prepared evidence to both extraction paths; re-extract on new evidence without duplicate mentions or changing historical social observations; expose preparation status.

Post text remains available immediately. Attachments stay within the parent's source family and do not count as independent corroboration. Kimi K2.6 on OpenCode Go is approved for transcription, chart observations and translation. Existing safe public fetching and HTML/PDF recovery are reused. Do not add providers, bypass the social extraction budget, or mutate benchmark artifacts.

Full article chunking is explicitly excluded. Keep bounded article and grounding limits and report truncation. Existing source evidence and model interpretations retain separate provenance. This work adds migrations but does not apply them to the running database or deploy changes.
