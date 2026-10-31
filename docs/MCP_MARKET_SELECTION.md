# Selecting a candidate market over MCP

`find_candidates` accepts an optional top-level `market` argument. For example:

```json
{"market":"US","filters":{"min_rs_rating":80,"stage":2},"limit":100}
```

```json
{"market":"DE","filters":{"min_rs_rating":80,"stage":2},"limit":100}
```

The server normalizes the requested code and validates it against the existing
supported-market registry. It resolves `latest_published_market:<MARKET>` using
the feature-run repository, verifies the referenced run is published and that
its stored market/universe metadata agrees with the request. A missing pointer,
unpublished or missing run, custom universe, or missing/conflicting market
evidence returns MCP `isError` with code `market_run_unavailable`. Invalid market
arguments return `invalid_arguments`. A stored run whose status or run type is not
a known value returns `invalid_stored_data`: the request was valid, the row is
not, and that is reported as damaged storage rather than a caller mistake.

There is no global-pointer or other-market fallback for an explicit market.
Run IDs, universe hashes, and symbol suffixes are not used to guess a market.
No pointer is created or changed by the lookup.

Explicit-market responses retain the existing candidate projection and add the
verified `market`, `universe_type: "market"`, and `pointer_key` to `run`.
`scope.market` also identifies the requested market. A valid empty result still
carries this provenance; it is different from an unavailable publication.

One exception: when `universe` names a watchlist that does not exist, the call
returns the existing not-found envelope with empty `symbols` and no `run` or
`scope` at all - the same shape an unknown watchlist produces without `market`.
That envelope reports a bad scope, not a missing publication, so it carries no
market provenance to verify.

Omitting `market` preserves the existing global latest-run behavior and response
shape. `universe` retains its existing meaning (`all`, `latest_published`, or a
watchlist name); it is not a market selector. A watchlist restriction may be
combined with an explicit market.

The limit remains 100. Clients collecting a complete universe must still detect
truncation and keep every query/partition for a market on the same run identity.
They must explicitly request each desired market and keep provenance separate.
Market-specific freshness must use the matching canonical market calendar;
equal dates on a weekend do not establish cross-market calendar equivalence.

This extension does not change candidate scoring, strategy rules, earnings
collection, calendar calculation, authentication, or write permissions. It does
not by itself migrate Hermes or enable any scanner job.
