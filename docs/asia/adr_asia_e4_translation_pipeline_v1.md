# ADR ASIA-E4: TranslationPipeline and Multilingual Extraction Contract (v1)

- Date: 2026-04-11
- Status: Accepted
- Issue: `StockScreenClaude-asia.1.1`

## Context

Theme extraction currently assumes mostly English input. HK/JP/TW content requires deterministic language handling to avoid hallucinations and unstable outputs.

## Decision

### Pipeline Order (Deterministic)

1. Language detection
2. Translation (when non-English)
3. Deterministic alias/symbol resolution
4. LLM-assisted extraction fallback
5. Validation against SecurityMaster + active universe

### Data Contract

Persist and expose:

- source language
- translated text (if produced)
- translation metadata/confidence
- extraction provenance and validation outcomes

### Reliability Policy

- Rule/alias-first, LLM-second for ticker normalization.
- Retries are idempotent via detection/translation caching.
- Unsupported/ambiguous extraction outcomes are surfaced explicitly instead of silently coerced.

## Consequences

- Multilingual behavior is reproducible and auditable.
- Hallucination risk is reduced for CJK symbol/entity extraction.
- API/UI can render original + translated context with confidence semantics.

## Rejected Alternatives

- "Single-pass LLM-only normalization": rejected due to non-determinism and high false-positive risk.
- "Translate nothing": rejected because it degrades extraction quality and user trust for non-English sources.
