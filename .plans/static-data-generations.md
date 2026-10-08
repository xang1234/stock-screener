# Coherent static data generations (#504 part b)

## Problem

Every publish overwrites `static-data/manifest.json` and `static-data/markets/…`
in place. Pages serves them with `cache-control: max-age=600`, and the static
app caches the manifest and every data query with `staleTime`/`gcTime:
Infinity`. An open tab therefore:

- keeps an old manifest forever, and lazily loads files (charts, scan chunks,
  group matrix) from whatever publish is live, mixing generations;
- can combine a browser-cached manifest with newer data files for up to ten
  minutes after a deploy.

## Decision: versioned paths, no retained previous generation

The deployed site is 1.17 GB uncompressed (Pages artifact of run
37733700679), already above GitHub's 1 GB documented site limit, and a daily
publish rewrites nearly every market file. Retaining the previous generation
(as #504 asks) would roughly double the deploy. Agreed with the owner
(2026-10-08): data moves under an immutable per-publish directory, and only
the current generation is deployed. A tab still on an older generation gets
an explicit "data updated, reload" state instead of mixed data.

## Publisher

- After the combine step, `relocate_into_generation(output_dir)` moves every
  entry except `manifest.json` into `g/<generation>/`. `<generation>` is the
  manifest's `generated_at` in compact form plus 8 hex of the manifest digest.
- The root `manifest.json` stays at the root (the pointer: other readers keep
  working) and gains `generation` and `data_root: "g/<generation>/"`. Paths
  inside the manifest and inside data files stay root-relative; they resolve
  against `data_root`.
- The move is verified: the file list under `g/<generation>/` must equal the
  list before the move. Advertised paths are already validated per market by
  the combiner before it writes.
- Switch: `export_static_site --data-generation`, passed by the publisher
  unless repo variable `STATIC_DATA_GENERATIONS=false`. The frontend reads a
  manifest without `data_root` from the flat layout, so rollback is the
  variable.

## Frontend

- `getStaticDataUrl(path, dataRoot)` and `fetchStaticJson(path, generation)`
  resolve data under `data_root`. A 404 under a generation raises
  `StaticGenerationExpiredError`.
- `useStaticManifest()` revalidates the root manifest with
  `cache: 'no-cache'` (ETag, usually a 304) on window focus and every
  5 minutes, instead of never.
- `useStaticGeneration()` derives `{ generation, dataRoot }` from the
  manifest; every static data query key ends with `gen:<generation>`.
- Nothing loads before the manifest names the generation. A flat-layout
  manifest (rollback) gets `flat-<generated_at>` so its publishes switch too.
- Switching: a new manifest switches every key at once. Each query keeps
  showing its previous data while the new generation loads, but only when
  the previous key differs from the new one by the generation alone (never
  another market's data). Not a cross-query transaction: each view swaps when
  its own data is ready. A query whose key or path comes from another query's
  data (options detail, COT history, scan chart index, themes variant) waits
  until that parent has its new-generation data, never using placeholder data.
- Query-cache bound: when a query with another generation becomes inactive,
  it is removed. Active queries are never evicted, so only the current
  generation plus the still-rendered previous one are held.
- Expired generation: a `StaticGenerationExpiredError` refetches the manifest;
  if that check leaves the tab on the same generation, a banner offers
  "Reload". A file genuinely missing from the current generation shows the
  same banner (worded for both cases); reload does not fix that one.
- Unavailable market: a `?market=` or saved selection listed in
  `unavailable_markets` stays selected and shows an explicit unavailable state
  with Retry, instead of silently switching to another market.

## Not built

- Retained previous generation (size, above).
- Frontend runtime archive reuse (#504 part a).
