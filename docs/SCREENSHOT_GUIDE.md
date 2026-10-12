# Screenshot & GIF Capture Guide

This guide provides detailed instructions for capturing screenshots and GIFs for the README.

## Tools Recommended

### Screenshots
- **macOS**: Built-in Screenshot (Cmd+Shift+4) or CleanShot X
- **Windows**: Snipping Tool or ShareX
- **Cross-platform**: Browser DevTools (F12 → Device toolbar for consistent sizing)

### GIFs
- **macOS**: Kap (free), CleanShot X, or LICEcap
- **Windows**: ScreenToGif or LICEcap
- **Cross-platform**: Gifski (for converting video to GIF)

## Recommended Settings

- **Browser width**: 1400px (shows full UI without excessive whitespace)
- **Screenshot format**: PNG for lossless UI captures, or high-quality JPEG for full-page dashboard views
- **GIF settings**: 15-20 FPS, max 10 seconds, optimize for web
- **Theme**: Dark mode (default) - provides better contrast for README

## Screenshots to Capture

### 1. scan-results.png
**Location**: `/scan` page
**What to capture**: Full scan results table showing:
- Screener selection chips at top (Min, CAN, IPO, Cust, VolB, Setup)
- Results table with 10-15 rows of data
- Visible columns: Symbol (with market badge), RS Trend sparkline, Price sparkline, **Sector** (GICS), **IBD Industry**, **Themes** (chip cluster with `+N` overflow), **Grp** (IBD group rank), Composite, individual screener scores
- Sort by Composite Score descending

**Setup**:
1. Pick any market from the Market dropdown and run a scan
2. Wait for completion
3. Ensure at least the top rows have populated industry/theme/group data so the new classification columns are demonstrably useful
4. Capture the viewport (table area + control bar context)

---

### 1a. market-selector.jpg
**Location**: `/scan` page → Market dropdown opened
**What to capture**: The Market combobox in the scan control bar with the dropdown menu open, listing all five supported markets:
- United States
- Hong Kong
- India
- Japan
- Taiwan

**Setup**:
1. Land on `/scan`
2. Click the Market dropdown
3. Capture the viewport (the open listbox plus surrounding control bar)
4. Format: JPEG (matches existing asset) — quality 90 is sufficient for a UI dropdown

---

### 1b. market-badges.png
**Location**: `/scan` page after a multi-market scan, **or** any view that surfaces tickers from more than one market
**What to capture**: A few result rows showing the per-market badge in the Symbol cell — e.g., HK / IN / JP / TW / US. Each badge is a small color-coded chip next to the ticker.

**Setup**:
1. Either run a cross-market scan (requires all relevant universes hydrated), or pull badges from the Watchlists view if the dev stack only has a single market seeded
2. Capture a tight crop of 4-6 rows so the badges are the visual focus

---

### 2. scan-filters.png
**Location**: `/scan` page
**What to capture**: Expanded filter panel showing:
- Multiple filter categories visible (Fundamental, Technical, Rating)
- Some active filters applied (RS > 70, Stage = 2)
- Filter preset dropdown visible if possible

**Setup**:
1. Expand the filter panel
2. Set a few representative filters
3. Capture the filter panel area

---

### 3. watchlist-table.png
**Location**: `/` (home page) → Watchlists tab
**What to capture**: Watchlist table showing:
- 8-12 stocks with data
- RS sparklines (small line charts)
- Price sparklines
- Price change bars (green/red horizontal bars)
- Watchlist selector dropdown visible

**Setup**:
1. Navigate to home page, click Watchlists tab
2. Select a watchlist with diverse stocks
3. Ensure price change bars show both gains and losses
4. Capture the table area

---

### 4a. breadth-chart.jpg
**Location**: `/breadth` page
**What to capture**: Breadth overview showing:
- Advancing/declining and new-high/new-low health bars
- T2108, 10x ATR extension, and broad-universe context cards
- Main chart with green/red breadth areas and benchmark line overlay
- Time-range selector

**Setup**:
1. Navigate to Market Breadth page
2. Use a 1280px-or-wider desktop viewport
3. Ensure the health bars and chart have populated data
4. Capture the top viewport with both the context strip and chart visible

---

### 4b. breadth-history.jpg
**Location**: `/breadth` page → Recent History
**What to capture**: Full-width history table showing:
- Amber **Primary**, green **Secondary**, and blue **Context** group bands
- Multiline metric headings and visible group dividers
- Dark neutral, green, and red significance levels across several dates
- The complete table width without horizontal scrolling

**Setup**:
1. Scroll to Recent History
2. Keep the viewport at least 1280px wide
3. Reset the table to its newest rows and leftmost position
4. Capture the title, group bands, headings, and at least 12 populated rows

---

### 5. group-rankings.png
**Location**: `/groups` page
**What to capture**: Group rankings layout showing:
- Left panel: Top movers (gainers/losers cards)
- Right panel: Full rankings table
- Color-coded rank badges (green for top 20, red for bottom 20)
- Rank change arrows visible

**Setup**:
1. Navigate to Group Rankings page
2. Ensure movers panel shows meaningful changes
3. Scroll table to show top-ranked groups
4. Capture the full page layout

---

### 6. group-detail.png
**Location**: `/groups` page → Click any group row
**What to capture**: Group detail modal showing:
- Group name and current rank
- Rank history line chart
- Rank change cards (1W/1M/3M/6M)
- Constituent stocks table (partial)

**Setup**:
1. Click on an interesting group (one with rank changes)
2. Ensure chart shows rank movement over time
3. Capture the modal (may need to resize)

---

### 7. chatbot.png
**Location**: `/chatbot` page
**What to capture**: Chatbot interface showing:
- Left sidebar with conversation list/folders
- Main chat area with a few message exchanges
- Model selector or settings visible
- Research mode toggle if possible

**Setup**:
1. Navigate to Chatbot page
2. Have an existing conversation with stock research (e.g., "Analyze NVDA")
3. Show both user and assistant messages
4. Capture the full page

---

### 8. themes.png
**Location**: `/themes` page
**What to capture**: Themes page showing:
- Theme rankings table with momentum scores
- Emerging themes panel (right side)
- Alerts section if populated
- Source filter chips (Substack, Twitter, etc.)

**Setup**:
1. Navigate to Themes page
2. Ensure themes are populated with data
3. Select Technical or Fundamental pipeline
4. Capture the main content area

---

## GIFs to Capture

### 1. scan-workflow.gif (Hero GIF)
**What to capture** (automated by `capture-scan-workflow.sh`, see below):
1. Daily Snapshot
2. Scan results, then a stock's detail, walking to the next stocks with Space
3. Group rankings table
4. RRG (sector scope, labels on)
5. Stock Matrix grid, Technology sector, scrolled to the software rows

**Tips**:
- Optimize GIF size (target ~5MB)

---

### 2. chart-interaction.gif (Optional)
**Duration**: 5-7 seconds
**What to capture**: Chart modal interaction:
1. Click chart icon on a stock row
2. Chart modal opens with candlestick chart
3. Hover over chart showing tooltip
4. Close modal

---

## File Checklist

After capturing, verify you have:

```
docs/
├── screenshots/
│   ├── scan-results.png      ← Multi-screener results table with Sector/IBD Industry/Themes/Grp columns
│   ├── market-selector.jpg   ← Market dropdown showing US / HK / IN / JP / TW
│   ├── market-badges.png     ← Per-row market badges in mixed-universe scans
│   ├── scan-filters.png      ← Filter panel expanded
│   ├── watchlist-table.png   ← Watchlist with sparklines
│   ├── breadth-chart.jpg     ← Breadth health bars and benchmark overlay
│   ├── breadth-history.jpg   ← Grouped recent-history significance heatmap
│   ├── group-rankings.png    ← IBD group rankings page
│   ├── group-detail.png      ← Group detail modal
│   ├── chatbot.png           ← AI chatbot interface
│   ├── themes.png            ← Theme discovery page
│   ├── bootstrap-setup.jpg   ← First-run primary-market picker
│   └── bootstrap-progress.jpg ← First-run staged pipeline progress
└── gifs/
    └── scan-workflow.gif     ← Hero GIF of scan process
```

## Image Optimization

Before committing, optimize images for web:

```bash
# Install optipng and gifsicle (macOS)
brew install optipng gifsicle

# Optimize PNGs
optipng -o5 docs/screenshots/*.png

# Optimize GIFs
gifsicle -O3 --colors 256 docs/gifs/*.gif -o docs/gifs/*.gif
```

Or use online tools:
- PNGs: TinyPNG (https://tinypng.com)
- GIFs: ezgif.com optimizer

## Tips for Great Screenshots

1. **Consistent sizing**: Use same browser width for all captures
2. **Clean data**: Use realistic but not sensitive stock data
3. **Visual variety**: Show green and red values, different stages
4. **No personal info**: Avoid capturing any API keys or personal data
5. **Dark mode**: Looks better on GitHub's default view
6. **Crop tightly**: Remove browser chrome, focus on the UI

## Updating Screenshots

When the UI changes significantly:
1. Re-capture affected screenshots
2. Optimize the new images
3. Commit with message: `docs: update screenshots for [feature] changes`

## Automated Live-App Capture (screenshots + hero GIF)

The README screenshots and the hero `scan-workflow.gif` are regenerated from a **live full-app deployment** (not the static demo) via three Playwright scripts (the breadth images are the exception; see the note below). They authenticate by reading `SERVER_AUTH_PASSWORD` from the repo-root `.env` and POSTing `/api/v1/auth/login`, so the server session cookie rides in the browser context — the password is never printed or passed on the command line.

```bash
# one-time setup
brew install ffmpeg gifsicle

# point at your running instance (defaults to the maintainer's Tailscale host)
export SITE_URL=https://your-host.example.net

# refresh screenshots into .tmp/readme-shots/ for review
cd frontend && node scripts/capture-live-readme.mjs

# regenerate the hero tour GIF straight into docs/gifs/scan-workflow.gif
bash frontend/scripts/capture-scan-workflow.sh
```

> **GIF encoding note:** the tour GIF uses an ffmpeg flat-palette pass with `dither=none` followed by `gifsicle -O3 --lossy`, *not* gifski. The dark UI background is a near-black gradient, and error-diffusion dithering (gifski's default, with no opt-out) scatters it into patchy speckle. `dither=none` keeps the darks uniform; gifsicle recovers the file size. See `capture-scan-workflow.sh` for the exact pass.

`capture-live-readme.mjs` captures the Daily Snapshot, the Market Health & Exposure card (element crop, kept at 2x), Scan results, Market selector, Breadth, Group rankings, RRG (sector scope, labels on), Options Command Center, and Watchlist at `deviceScaleFactor: 2`. For a local Docker stack, set `SITE_URL=http://localhost`. `capture-group-readme.mjs` (with `GROUPS_SCREENSHOT_URL=http://localhost`) writes `group-rs-table`, `group-matrix-grid` and `group-matrix-clusters` to `.tmp/group-readme/` at their committed size (2400px wide); copy those as-is. Review the `capture-live-readme.mjs` output in `.tmp/readme-shots/`, then downscale the keepers to 1440px wide (matching the existing set) before copying into `docs/screenshots/`:

```bash
ffmpeg -y -i .tmp/readme-shots/breadth-chart.png -vf "scale=1440:-1:flags=lanczos" -q:v 2 docs/screenshots/breadth-chart.jpg
```

The table-focused `breadth-history.jpg` is currently captured separately using the steps in **4b** so its group headers and newest rows remain the focus.

**Breadth needs history.** Backfilled breadth only covers dates with a recorded point-in-time universe, so a freshly installed stack shows only a few sessions however wide the range. When that is the case, take `breadth-chart.jpg` and `breadth-history.jpg` from the static site (same page, long history) instead. The hero tour leaves Breadth out for the same reason.

**Capture `scan-results` on an Asian market when the stack has one.** The Themes chip-cluster column is only populated for the Asia universes in the reference deployment; a US scan leaves it blank. Set `SCAN_RESULTS_MARKET` to the market's label in the picker (e.g. `SCAN_RESULTS_MARKET=Japan`); the step switches back to United States afterwards. A stack with only US data keeps the default.

**Assets kept from prior captures** (regenerate them only when the live instance has the relevant data/state):

| Asset | Why it isn't auto-captured |
|-------|----------------------------|
| `chatbot.png` | Needs the Hermes assistant sidecar running |
| `themes.png` | Needs the theme-discovery pipeline populated |
| `market-badges.png` | Needs a single view spanning multiple markets (mixed-market badges) |
| `bootstrap-setup.jpg`, `bootstrap-progress.jpg` | First-run-only wizard screens |

## Automated Tour GIF (`static-site-tour.gif`)

The static-site page-tour GIF is regenerated by a Playwright + ffmpeg + gifsicle pipeline:

```bash
# one-time setup
brew install ffmpeg gifsicle

# regenerate the GIF
bash frontend/scripts/capture-static-site-tour.sh
```

What it does:
1. `frontend/scripts/capture-static-site-tour.mjs` drives a headless Chromium with Playwright's `recordVideo` API against `https://xang1234.github.io/stock-screener/`. The storyboard: Daily Snapshot → Scan (preset filter chips + space-bar stock detail walk) → Breadth → Groups (ranked table → RRG rotation, sector scope with labels).
2. `ffmpeg` trims the head spinner, applies a 1.5x speed-up, and extracts frames.
3. `ffmpeg` builds a flat global palette and re-encodes with `paletteuse=dither=none` (uniform darks — see the GIF encoding note above), then `gifsicle -O3 --lossy=30` writes `docs/gifs/static-site-tour.gif` at width 960, near the 5 MB budget.

To target a different host (e.g. a PR preview), pass `SITE_URL`:

```bash
SITE_URL=https://my-preview.example.com bash frontend/scripts/capture-static-site-tour.sh
```

Intermediate artifacts land in `.tmp/hero-capture/` (gitignored via `*.tmp`).
