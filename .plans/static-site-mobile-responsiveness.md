# Static site — mobile responsiveness fixes

> Plan doc lives in `.plans/static-site-mobile-responsiveness.md` (version-controlled & PR-reviewable).

## Context

"The static site" = the GitHub Pages build of the React frontend (`VITE_STATIC_SITE=true` → `StaticAppShell` + `StaticLayout` + `frontend/src/static/pages/*`, serving pre-baked JSON bundles). It was built desktop-first with fixed pixel dimensions and **no breakpoints**, so on phones:

1. **Header tabs unreachable** — `StaticLayout.jsx` toolbar has a `flexGrow:1` spacer pushing 4 raw `<Button>` nav items to the right of a fixed-width title + 140px market selector. No wrap/scroll → nav clipped off the right edge of a `position="static"` AppBar.
2. **Charts show "details but no prices"** — `StaticChartViewerModal.jsx` body is a flex *row* of a hardcoded **450px** metrics sidebar + chart. On a 375px phone the sidebar is wider than the viewport, so the chart container computes `clientWidth ≈ 0`; the `ResizeObserver` in `CandlestickChart.jsx:233` is guarded by `width > 0` so it never fires and the canvas stays blank while header/sidebar render.
3. **Tables don't scroll** — bespoke per-page tables (Groups/Breadth/Home) have no `minWidth`/`nowrap`, so columns crush/wrap instead of giving a horizontal scroll region. (`ResultsTable` on the Scan page already scrolls via `minWidth:2673 + overflow:auto`.)
4. **"No price data" in group charts** (separate, backend) — `_build_group_details` builds details for **every** group (all clickable), but Pass-3 chart export only covers constituents of groups ranked `≤ 50` (`STATIC_CHART_TOP_N_GROUPS`). Opening a lower-ranked group → every constituent card shows "No price data" on all devices.

Decisions (confirmed with user): chart → **stack metrics below chart** on phones; nav → **scrollable tab strip**; **include the backend coverage fix**.

Routed static pages only: `/` (Home), `/scan`, `/groups`, `/breadth`. `StaticThemesPage` is not routed → ignore it.

## Changes

### 1. Header nav → scrollable tabs — `frontend/src/static/StaticLayout.jsx`
- Add `Tabs`/`Tab` to the MUI import.
- Make `<Toolbar variant="dense">` wrap: `sx={{ minHeight: 48, flexWrap: 'wrap', rowGap: 0.5 }}`.
- Keep brand icon + title + "Read-only" chip + market selector + theme toggle as-is (row 1).
- Replace the `flexGrow:1` spacer + the `NAV_ITEMS.map(<Button>)` block with a scrollable Tabs strip:
  ```jsx
  <Box sx={{ flexGrow: 1 }} />
  <Tabs
    value={NAV_ITEMS.some((i) => i.path === location.pathname) ? location.pathname : false}
    variant="scrollable"
    scrollButtons="auto"
    allowScrollButtonsMobile
    textColor="inherit"
    TabIndicatorProps={{ sx: { bgcolor: 'white' } }}
    sx={{ minHeight: 40 }}
  >
    {NAV_ITEMS.map((item) => (
      <Tab
        key={item.path}
        component={RouterLink}
        to={item.path}
        value={item.path}
        label={item.label}
        sx={{ minHeight: 40, fontSize: '12px', textTransform: 'none' }}
      />
    ))}
  </Tabs>
  ```
- On wide screens the spacer keeps tabs right-aligned on one row; on narrow screens the toolbar wraps the tab strip to a second row that scrolls/swipes. No `useMediaQuery` needed (CSS handles it).

### 2. Chart modal responsive — `frontend/src/static/StaticChartViewerModal.jsx`
- Import `useMediaQuery`, `useTheme`. Compute `const isMobile = useMediaQuery(theme.breakpoints.down('md'))`.
- Responsive chart height: `const chartHeight = isMobile ? Math.round(viewportHeight * 0.6) : Math.max(viewportHeight - 60, 500);`
- Header bar (currently `height: 60`): allow it to grow on mobile — `minHeight: 60, height: 'auto', py: 0.5`; add `flexWrap: 'wrap', rowGap: 1` to the left metrics `Box` (line ~175) so the badges wrap instead of overflowing.
- Body row (line ~313): make it stack with chart on top on mobile —
  ```jsx
  sx={{ display: 'flex', flexDirection: { xs: 'column-reverse', md: 'row' }, flex: 1, overflowY: { xs: 'auto', md: 'hidden' }, overflowX: 'hidden' }}
  ```
  `column-reverse` puts the chart (2nd DOM child) on top and the sidebar below on mobile, while `row` keeps sidebar-left on desktop — no DOM reorder needed.
- Account for the fixed bottom keyboard-hints bar (`position:fixed`, ~48px): add `pb` on the body on mobile, or hide the hints bar on `xs` (`display: { xs: 'none', sm: 'flex' }`) since keyboard shortcuts are desktop-only anyway. Prefer hiding it on `xs`.

### 3. Metrics sidebar width responsive — `frontend/src/components/Scan/StockMetricsSidebar.jsx`
- Replace the hardcoded `width: 450` (3 occurrences: loading branch ~line 77, fundamentals-only branch ~line 92, and the main branch) with `width: { xs: '100%', md: 450 }`. Shared component — also improves the live chart viewer.

### 4. Tables scroll on mobile — bespoke static tables + shared `DailyScanRowsTable`
Pattern (apply per table): give the `TableContainer` an explicit horizontal scroll boundary and the inner `<Table>` a `minWidth` so content overflows (something to scroll) instead of crushing.
```jsx
<TableContainer sx={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
  <Table size="small" sx={{ minWidth: 640 }}>
```
Add a shared `const SCROLL_TABLE_SX = { overflowX: 'auto', WebkitOverflowScrolling: 'touch' }` (small const, not a component) to avoid repetition. Apply to:
- `frontend/src/components/shared/DailyScanRowsTable.jsx` (used by Home) — `TableContainer` ~line 85; pick `minWidth` ~720.
- `frontend/src/static/pages/StaticHomePage.jsx` — inline "Top 10 Groups" table (~line 323).
- `frontend/src/static/pages/StaticGroupsPage.jsx` — `MoversCard` (~line 38) and `GroupsTableView` rankings (~line 80); rankings `minWidth` ~720.
- `frontend/src/static/pages/StaticBreadthPage.jsx` — "Recent Sessions" table (~line 142).
- `ResultsTable.jsx` (Scan) already scrolls; just add `WebkitOverflowScrolling: 'touch'` + `maxWidth: '100%'` to its `TableContainer` sx (~line 600).

Skip a global `overflow-x: hidden` body guard (YAGNI) — per-table scroll + responsive nav/chart should resolve the page-level blowout. Add only if testing shows the page still scrolls instead of the table.

### 5. Backend chart coverage for all groups — `backend/app/services/static_site_export_service.py`
The frontend already caps group charts at `MAX_SYMBOLS = 50` per group (`StaticGroupChartsGrid.jsx:16`). Mirror that on the backend and cover **all** groups instead of only top-50:
- Add `STATIC_CHART_MAX_PER_GROUP = 50` near the other constants (~line 76).
- In `_collect_top_group_constituent_symbols` (~line 897): drop the `rank > top_n` skip (or pass `top_n=None` meaning "all groups") and take only the first `STATIC_CHART_MAX_PER_GROUP` of each group's already-sorted `detail["stocks"]` (they're pre-sorted by rs_rating→composite in `_build_group_details`). This bounds the extra symbols to ≈ groups × 50, deduped.
- Update the Pass-3 call (~line 867) and `log_label` accordingly.
- `_expand_extra_charts` already dedups against existing entries, batches lookups, and skips symbols without cached bars — no change needed; symbols lacking cached price still degrade to the frontend's "No price data" card gracefully.

**Cost note:** this enlarges the per-market chart bundle (more JSON files / larger gh-pages artifact + longer export). Bounded by cached-price availability and the 50/group cap. Keep the existing `logger.info` count so the export log reports how many extra charts were generated.

## Verification

- **Backend unit test** — extend `backend/tests/unit/test_static_site_export_service.py`: add a case for `_collect_top_group_constituent_symbols` proving (a) a group ranked > 50 now contributes symbols and (b) a group with > 50 constituents is capped at 50. Run: `cd backend && source venv/bin/activate && pytest tests/unit/test_static_site_export_service.py -q`.
- **Frontend tests** — `cd frontend && export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && npm run test:run` (covers existing `StaticChartViewerModal.test.jsx`; adjust if the layout assertions need updating). `npm run lint`.
- **Manual / Playwright (mobile viewport, ~390px)** — build the static site with a local export, then drive it with the Playwright MCP at width 390:
  1. Generate bundles: `cd backend && source venv/bin/activate && python -m app.scripts.export_static_site --output-dir frontend/public/static-data` (or the documented command), then `cd frontend && VITE_STATIC_SITE=true npm run dev`.
  2. Resize to 390×844 and verify: (a) nav tabs are visible and swipe-scrollable on every page; (b) Groups/Breadth/Home tables scroll horizontally; (c) open a stock chart → candlesticks render full-width with metrics stacked below; (d) open a low-ranked group's detail → constituent charts now show prices.

## Files touched
- `frontend/src/static/StaticLayout.jsx` (nav)
- `frontend/src/static/StaticChartViewerModal.jsx` (chart modal layout)
- `frontend/src/components/Scan/StockMetricsSidebar.jsx` (responsive width — shared)
- `frontend/src/components/shared/DailyScanRowsTable.jsx` + `StaticHomePage.jsx` / `StaticGroupsPage.jsx` / `StaticBreadthPage.jsx` + `ResultsTable.jsx` (table scroll)
- `backend/app/services/static_site_export_service.py` (chart coverage)
- `backend/tests/unit/test_static_site_export_service.py` (test)
