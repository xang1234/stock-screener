# Theme and development separation

The shared theme extractor now distinguishes a recurring investment exposure from
the source-specific development. Theme names remain fully open: prompt examples
are illustrations, not a catalog or a whitelist.

For a source saying Samsung may supply HBM4 amid rising AI memory demand:

```json
{
  "theme": "HBM",
  "development": "Samsung may supply HBM4 amid rising AI memory demand.",
  "excerpt": "Samsung may supply HBM4 as AI memory demand rises"
}
```

`theme` supplies the existing identity matching and clustering paths.
`development` is a model-written English summary on the mention, not a clustering
key or an alias. `excerpt` remains a separate supporting source quote. Sentiment,
confidence and ticker handling retain their existing behavior.

The shared prompt asks for evidence-supported specificity, preserves uncertainty,
and distinguishes different economic exposures such as crude and product tankers.
Both pipeline configurations use exposure names; technical observations and
fundamental catalysts belong in the development. This aligns their naming
instructions but does not merge the existing pipeline-scoped cluster stores.

## Data and compatibility

- New responses request `development` as text of at most 1,000 characters, or null
  when the source supports a theme without a distinct development.
- Missing, null or blank values become null. Older responses stay usable without
  inventing a development from their theme or excerpt. Invalid types or oversized
  values raise a parse error rather than silently discarding or truncating claims.
- `ThemeMention.development` is nullable text. Apply Alembic revision
  `20260910_0038` before running the updated application against an existing
  database. The migration leaves historical themes and excerpts unchanged and
  does not backfill developments. Downgrade removes only the new column.
- The theme mentions API exposes the field. Both theme source views show a labeled
  development above the existing excerpt when present.
- Evaluation records accept the optional field without rewriting legacy mention
  dictionaries or changing frozen run hashes. Markdown displays theme, development
  and evidence separately; `theme-mentions.csv` includes these fields alongside
  source text, URL, input kind, warnings, tickers, sentiment and confidence.
- Separate social-projection producers that do not use the shared extractor leave
  this optional field null; their claim schemas are unchanged in this increment.

## Validation and rollout

Regression coverage exercises extraction through real SQLite persistence and the
mentions API: two developments retain a single existing HBM identity. Additional
tests cover invalid values, legacy responses, migration upgrade/downgrade, and
byte-preserving old/new evaluation record round trips through review rendering.
These deterministic tests verify the data flow, not LLM classification accuracy.

The first-ten model comparison uses the same approved inputs and frozen reference
universe, with new outputs under
`data/xui-reader/theme-evaluation/gap-remediation-20260910/baseline-extraction/theme-development-v1/`.
The original baseline is preserved. The remaining 81 inputs are not part of this
comparison. No application database migration or deployment is performed by the
evaluation runner.

This increment does not introduce a curated taxonomy, company-business context
retrieval, cross-pipeline identity migration, historical renaming, or new novelty
ranking. Those remain separate improvements. Review new model outputs for category
relevance, unsupported specificity and lost or overstated qualifiers before using
the prompt's behavior as evidence of improved accuracy.
