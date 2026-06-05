# RS line: dynamic band in the price area's empty space

**Date:** 2026-06-05
**Status:** Approved design — ready for implementation plan
**Component:** `frontend/src/components/Charts/CandlestickChart.jsx`
**Builds on:** PR #215 (moved the RS line into a fixed strip below the price)

## Problem

PR #215 placed the relative-strength (RS) line in a reserved strip below the
candles (`rs` overlay scale at `scaleMargins {top: 0.64, bottom: 0.24}`). The
strip is ~12% of the pane, so the RS line is auto-scaled into a thin band and
looks **flat** — its movement is too small to read. Its peak always sits below
the lowest price bar, which guarantees no overlap but wastes the large empty
area inside the price chart (e.g. the lower-right region during an uptrend).

## Goal

Let the RS line use the **empty space inside the price area** so it becomes
dynamic and readable — its peak may rise *above* the lowest price bar — while
**never overlapping the candles**. When there isn't enough safe space
(price/RS divergence, choppy or zoomed-in windows), it degrades gracefully
toward a thin floor strip rather than overlapping.

This is "Option B" from brainstorming, with **capped-dynamic** behavior.

## Layout model

Three regions in a single chart pane (fractions of pane height, 0 = top):

| Region | RS shown | RS hidden |
| --- | --- | --- |
| Price + EMAs (`right` scale) | `[0.05, 0.66]` | `[0.05, 0.78]` |
| RS line (`rs` scale) | `[rTop, 0.78]` (dynamic) | not drawn |
| Volume (`volume` scale) | `[0.80, 1.00]` | `[0.80, 1.00]` |

Compressing the price band to a `0.66` floor (when RS is shown) guarantees
`[0.66, 0.78]` is **always empty** — no candle is ever below the price floor.
That empty band is the RS line's **floor strip** and its enforced minimum
(~12%, matching today's look). Above `0.66`, candles exist but with empty
pockets; the RS line expands up into those pockets when it is safe to do so.

The existing RS toggle and `rsStripShown` flag are retained. When RS is hidden
(toggled off or weekly timeframe), price expands to `[0.05, 0.78]` (the
reclaim behavior from PR #215 is kept, with updated numbers).

## Core algorithm: `computeRsBand()`

A **pure, testable** function (no chart dependency) that returns the RS scale's
top margin `rTop` for the current visible window.

Inputs (for the visible bars only):
- `lows[]` — price low per bar (used for the per-bar non-overlap constraint)
- `highs[]` — price high per bar (used only for `priceMax`, the scale's top)
- `rsValues[]` — RS value per bar (aligned to the same bars)
- Constants: `priceBandTop = 0.05`, `priceFloor = 0.66`, `rsBottom = 0.78`,
  `gap = 0.012`, `minBand = 0.12`, `maxBand = 0.38`

Steps:
1. `priceMin = min(lows)`, `priceMax = max(highs)` over the visible window —
   the same range lightweight-charts auto-scales to. (Use highs for the max to
   match the candlestick scale, which spans lows..highs.)
2. **Price screen position is logarithmic** — the `right` price scale is created
   with `mode: 1` (log). For each visible bar:
   `plf_i = priceBandTop + (ln(priceMax) - ln(low_i)) / (ln(priceMax) - ln(priceMin)) * (priceFloor - priceBandTop)`
3. The RS scale is **linear**. With band `[rTop, rsBottom]`:
   `rsf_i = rTop + a_i * (rsBottom - rTop)`, where
   `a_i = (rsMax - rs_i) / (rsMax - rsMin)` in `[0, 1]`
   (0 when `rs_i` is the max, 1 when it is the min).
4. Non-overlap constraint: `rsf_i >= plf_i + gap` for every visible bar.
   Solve each for the minimum `rTop` (taller band = smaller `rTop`):
   `rTop_i = (plf_i + gap - a_i * rsBottom) / (1 - a_i)` for `a_i < 1`
   (skip `a_i == 1`, the RS-min bar, which is unconstrained from above).
   `computedRTop = max_i(rTop_i)`.
5. **Clamp:** `rTop = clamp(computedRTop, rsBottom - maxBand, rsBottom - minBand)`
   → `rTop ∈ [0.40, 0.66]`, band height ∈ `[0.12, 0.38]`.
   If `computedRTop > rsBottom - minBand` (even the min strip would overlap),
   `rTop` pins to `rsBottom - minBand = 0.66` — the floor strip, which is empty
   by construction, so still no overlap.

Degenerate inputs return the floor strip (`rTop = 0.66`):
- `rsMax == rsMin` (flat RS) — `a_i` undefined.
- fewer than 2 visible bars.
- any non-positive price (log undefined) in the window — guard and fall back.

## Wiring in `CandlestickChart.jsx`

- **Remove** the fixed `rs` scaleMargins (`{top: 0.64, bottom: 0.24}`) and the
  PR #215 reclaim `useLayoutEffect` in its current form.
- **Price/volume margins** continue to be driven by `rsStripShown`:
  - price: `rsStripShown ? {top: 0.05, bottom: 0.34} : {top: 0.05, bottom: 0.22}`
    (price floor `0.66` vs `0.78`)
  - volume: `{top: 0.80, bottom: 0}` (fixed)
- **New effect** computes and applies the RS band:
  - reads the visible logical range from the chart's time scale, slices the
    `lows`/`rsValues` arrays to it, calls `computeRsBand()`, and applies
    `rsScale.applyOptions({ scaleMargins: { top: rTop, bottom: 0.22 } })`.
  - **Recompute triggers:** RS data / chart data change, and
    `subscribeVisibleTimeRangeChange` (debounced ~80ms — price re-auto-scales
    on pan/zoom, so the safe band changes). Static charts compute once on load.
  - Skipped when `rsStripShown` is false.
- The RS line series data and blue-dot markers are unchanged; only the `rs`
  scale's margins move.

## "RS" label

The label from PR #215 is pinned at a fixed `top: '64%'`. Since the RS band now
moves, the label should track `rTop` (e.g. `top: \`${rTop * 100}%\``) so it stays
at the top of the live RS band. Keep it gated on `!compact`.

## Edge cases

- Flat RS / <2 bars / non-positive price → floor strip (no overlap, ~12%).
- RS hidden or weekly timeframe → effect skipped, price uses full `[0.05, 0.78]`.
- `interactive = false` (static charts) → no pan/zoom; computes once from the
  initial visible range.
- Rapid pan/zoom → debounced recompute; band "breathes" (expected, bounded by
  the 12%–38% clamp).

## Testing

**Unit tests for `computeRsBand()`** (pure function, primary coverage):
- Correlated uptrend → tall band near the cap; assert `rsf_i >= plf_i` for all i
  (no overlap) using the same screen-mapping math.
- Divergence (price up, RS down) → clamps to `minBand`.
- Flat RS / single bar / non-positive price → floor strip.
- Max cap respected (band height never exceeds `maxBand`).
- Min height respected (band height never below `minBand`).

**Visual verification (Playwright, offline static-mode harness):**
- Correlated case: RS rises into the empty lower-right, no candle contact.
- Divergence case: RS degrades to the floor strip, no overlap.
- Separation stress test (extreme candles, RS at band edges, volume spike) at
  full and small heights.
- RS toggle off → price reclaims to full height; label disappears.

## Out of scope

- Changing what the RS line plots (still the raw stock/benchmark ratio).
- Volume layout changes beyond its fixed `[0.80, 1.00]` band.
- A separate pane (Option A) — explicitly not chosen.

## Tunable constants (defaults)

`priceFloor = 0.66`, `rsBottom = 0.78`, `volumeTop = 0.80`, `gap = 0.012`,
`minBand = 0.12`, `maxBand = 0.38`, `debounceMs = 80`. These are expected to be
adjusted during visual verification.
