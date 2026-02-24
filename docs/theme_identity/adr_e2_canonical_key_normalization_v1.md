# ADR E2: Canonical Key Specification and Normalization Policy (v1)

- Date: 2026-02-24
- Status: Accepted
- Issue: `StockScreenClaude-bv9.2.1`
- Machine-check corpus: `docs/theme_identity/adr_e2_canonical_key_normalization_v1.corpus.json`

## Context

Theme identity currently depends on free-form names (for example `ThemeCluster.name` and `ThemeCluster.aliases`),
which creates drift when providers or models produce lexical variants of the same concept.
To make identity deterministic across extraction, merge, and lifecycle workflows, we need a stable machine key
separated from any analyst-facing display string.

This ADR defines the normative canonical key contract used by downstream E2 tasks.

## Decision

### 1) Key contract and shape

Canonical keys are machine identifiers, not UX labels.

- Key charset: lowercase ASCII letters, digits, underscore.
- Key regex: `^[a-z0-9]+(?:_[a-z0-9]+)*$`
- Joiner: underscore only.
- No leading/trailing underscore.
- No repeated underscore.
- Max key length: 96 characters.

Canonical key generation is deterministic and idempotent:

- `canonical_key(normalize(x)) == canonical_key(x)`
- Same semantic input across providers yields the same key.

### 2) Normalization pipeline (ordered, deterministic)

Given raw theme text, apply all steps in this exact order:

1. Trim leading/trailing whitespace.
2. Unicode normalize using `NFKD`.
3. Remove combining diacritics.
4. Lowercase.
5. Replace punctuation/separators using these rules:
   - `&` -> ` and `
   - `glp-1` and `glp 1` pattern -> `glp1` (before generic hyphen/slash handling)
   - `a/s` pattern -> `as` (before generic slash handling)
   - `/`, `\`, `|`, `-`, `_`, `.`, `,`, `:`, `;`, `(`, `)`, `[`, `]`, `{`, `}`, `'`, `"` -> space
   - `+` between alnum tokens -> ` plus ` (for example `c++` -> `c plus plus`)
6. Collapse whitespace to a single space.
7. Tokenize on spaces.
8. Apply token-level mappings (section 3).
9. Drop stopwords (section 4) except for protected contexts (section 5).
10. Join remaining tokens with underscore.
11. Enforce max length (section 6).

### 3) Token-level mappings

Token mappings are applied before stopword removal:

- Acronym/term normalization:
  - `ai`, `a.i` -> `ai`
  - `ml`, `m.l` -> `ml`
  - `llm`, `l.l.m` -> `llm`
  - `glp`, `glp1` -> `glp1`
  - `gpu`, `gpus` -> `gpu`
  - `ev`, `evs` -> `ev`
  - `ipo`, `ipos` -> `ipo`
  - `etf`, `etfs` -> `etf`
- Domain plurals:
  - if token ends with `ies` and length > 4, replace `ies` with `y`
  - otherwise strip trailing `s` for tokens longer than 3 unless token is in exception set
  - exception set: `gas`, `as`, `us`, `esg`, `saas`
- Numeric normalization:
  - keep pure numerics as-is (`2`, `10`, `2025`)
  - keep alnum tokens as-is after punctuation cleanup (`3d`, `5g`, `b2b`)
  - remove thousands separators during punctuation pass (`1,000` -> `1000`)

### 4) Stopword policy

Default stopwords:

- `a`, `an`, `the`
- `of`, `for`, `to`
- `in`, `on`, `at`
- `by`, `from`

These are removed unless protected by section 5.

### 5) Protected-token policy

To avoid destructive collisions, never drop these tokens even if they would otherwise be treated as stopwords:

- `as` when tokenized from `a/s` in finance phrases
- `us` (country context)
- `non` when part of `non` prefixed constructions (for example `non gaap`)

### 6) Overflow and empties

- If all tokens are removed, return `unknown_theme`.
- If joined key exceeds 96 chars, truncate deterministically:
  - keep first 80 chars of the joined key
  - append `_` plus 8-char lowercase hex digest of the full pre-truncation key
  - final output remains regex-compliant

### 7) Display name policy

- Display name is independent from canonical key.
- `display_name` may preserve analyst-friendly casing and punctuation.
- `display_name` updates must not mutate `canonical_key` in-place.

### 8) Collision and compatibility policy

- Canonical key policy is a compatibility surface.
- Any behavior change requires:
  - new policy version (`v2`, etc.)
  - migration/backfill plan
  - before/after collision report
- Near-collision additions must be added to corpus fixtures first.

## Representative Corpus and Edge Cases

Normative examples and adversarial edge cases are defined in:

- `docs/theme_identity/adr_e2_canonical_key_normalization_v1.corpus.json`

This corpus includes acronym variants, numeric tokens, punctuation variants, Unicode/diacritic inputs,
and near-collision sets that must remain stable over time.

## Downstream Task References

This ADR is normative for all E2 implementation tasks:

- `StockScreenClaude-bv9.2.2` (`E2-T2`) normalization library and tests
- `StockScreenClaude-bv9.2.3` (`E2-T3`) schema migration to `canonical_key` plus `display_name`
- `StockScreenClaude-bv9.2.4` (`E2-T4`) alias table and repository operations
- `StockScreenClaude-bv9.2.5` (`E2-T5`) key/alias backfill
- `StockScreenClaude-bv9.2.6` (`E2-T6`) CI integrity checks

## Rollout Notes

- The v1 corpus is the source of truth for deterministic expected outcomes.
- E2-T2 must implement generator + display formatting as separate utilities.
- Existing ad hoc normalization (for example hand-curated theme maps) is transitional and should be routed
  through this contract in follow-up tasks.
