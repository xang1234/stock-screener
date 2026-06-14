# Stock Screener 🇺🇸 🇨🇳 🇭🇰 🇯🇵 🇰🇷 🇹🇼 🇮🇳 🇩🇪 🇨🇦 🇸🇬 🇲🇾 🇦🇺

A stock screening platform with multi-methodology scans across **US, Hong Kong, India, Japan, Korea, Taiwan, mainland China A-share, Germany, Canada, Singapore, Malaysia, and Australia** markets, AI-assisted research, theme discovery from social and news feeds, and real-time market breadth analysis. The supported deployment path is a single-tenant server stack built around Docker, PostgreSQL, Redis, and nginx.

### Daily Market Dashboard

![Daily Snapshot — key index cards, trend sparklines, and top scan candidates](docs/screenshots/daily-snapshot.png)
*At-a-glance home dashboard: key index/ETF cards with trend sparklines plus the day's top composite-ranked scan candidates*

### Live App Tour

![Live app tour — Daily Snapshot, multi-screener scan with stock detail, market breadth, and industry-group RRG](docs/gifs/scan-workflow.gif)
*A walkthrough of the live app: Daily Snapshot → multi-screener Scan with a drill-in to an individual stock's chart and scores → Market Breadth → Group Rankings with the Relative Rotation Graph*

---

### Static Site Page Tour

![Static site page tour — Daily Snapshot, Scan with stock detail, Breadth, and Group Rankings with the RRG](docs/gifs/static-site-tour.gif)

Static demo: [https://xang1234.github.io/stock-screener/](https://xang1234.github.io/stock-screener/)

The static page is for demo purposes only. It is a read-only daily snapshot with reduced functionality compared with the full application, which includes live workflows such as chatbot, themes pipeline, watchlists, and the full interactive backend.

## Highlights

### Multi-Market Coverage

Scan and track twelve markets:

- 🇺🇸 **United States** — NYSE, NASDAQ, AMEX, S&P 500
- 🇭🇰 **Hong Kong** — HSI
- 🇮🇳 **India** — NSE, BSE
- 🇯🇵 **Japan** — Nikkei 225
- 🇰🇷 **Korea** — KOSPI, KOSDAQ
- 🇹🇼 **Taiwan** — TAIEX
- 🇨🇳 **Mainland China A-shares** — SSE, SZSE, BJSE
- 🇩🇪 **Germany** — XETRA, DAX
- 🇨🇦 **Canada** — TSX, TSXV
- 🇸🇬 **Singapore** — SGX
- 🇲🇾 **Malaysia** — Bursa Malaysia (Main + ACE), FBM KLCI
- 🇦🇺 **Australia** — ASX, S&P/ASX 200

Each market runs on its own exchange calendar (XNYS / XHKG / XNSE / XTKS / XKRX / XTAI / XSHG / XETR / XTSE / XSES / XKLS / XASX) with independent Celery refresh queues and locks, so US, Asia-Pacific, and Europe can hydrate in parallel without stepping on each other. Switch markets from the scan control bar; mixed-universe results are tagged with per-row colored badges.

![Market selector](docs/screenshots/market-selector.jpg)
*Market picker in the scan control bar — pick US, HK, IN, JP, KR, TW, CN, DE, CA, SG, MY, or AU and scope to an exchange or index*

![Market badges](docs/screenshots/market-badges.png)
*Color-coded per-market badges in the Symbol column — US (blue), HK (green), JP (yellow); Taiwan, India, Korea, China, Germany, Canada, Singapore, Malaysia, and Australia follow the same pattern*

Deep-dive: **[ASIA v2 ADRs & runbooks](docs/asia/README.md)**

### Multi-Strategy Screening

Run Minervini, CANSLIM, IPO, Volume Breakthrough, Setup Engine, and Custom scans simultaneously with composite scoring across 80+ configurable filters. Save filter presets and export results to CSV.

![Scan Results](docs/screenshots/scan-results.png)
*Results table with composite scores, RS sparklines, multi-screener ratings, and per-row classification columns — GICS Sector, IBD Industry, market themes, and group rank*

### Market Breadth Dashboard

StockBee-style advance/decline analysis with SPY overlay, daily movers (stocks up/down 4%+), and multi-period trend visualization across quarterly, monthly, and 34-day windows.

![Market Breadth](docs/screenshots/breadth-chart.png)
*Breadth chart with SPY price overlay and daily movers*

### IBD Industry Group Rankings

197 industry groups ranked by relative strength with top movers identification (1W/1M/3M/6M), historical rank charts, and constituent stock analysis.

![Group Rankings](docs/screenshots/group-rankings.png)
*Industry group rankings with movers panel*

#### Relative Rotation Graph (RRG)

A MarketSmith/Bloomberg-style quadrant view of the same 197-group dataset: every industry group (or GICS-sector roll-up) is plotted by **RS-Ratio** vs **RS-Momentum**, with a smooth spline tail and direction arrows tracing its weekly path through **Leading → Weakening → Lagging → Improving**. One screen answers *"what's rotating in, what's rolling over."* Drag a rectangle to zoom into crowded clusters, toggle collision-avoiding group labels, filter by quadrant, name, or current rank, and click any dot to drill into its constituents. Available for every enabled market, in group or sector scope.

![Relative Rotation Graph](docs/screenshots/rrg-rotation.png)
*RRG: GICS-sector rotation with direction-arrowed weekly tails — group scope (all 197 industry groups) available from the same view*

### Watchlists with Sparklines

Visual performance tracking with RS and price sparklines (30-day trends), price change bars across 7 time periods, drag-and-drop organization with folders, and full-screen chart navigation.

![Watchlist Table](docs/screenshots/watchlist-table.png)
*Watchlist with sparklines and price change visualization*

### AI Research Chatbot

Groq-powered research chat with optional Tavily/Serper web search, persistent conversation history, and tool-augmented investigation.

![Chatbot](docs/screenshots/chatbot.png)
*AI chatbot with conversation sidebar and research tools*

### Theme Discovery Pipeline

AI-powered market theme identification from RSS, Twitter/X, and news feeds. Tracks trending vs. emerging themes, monitors constituent stocks, and alerts on momentum shifts.

![Themes](docs/screenshots/themes.png)
*Theme discovery with rankings and emerging themes panel*

## Get Started

### Docker (Recommended for Servers)

```bash
cp .env.docker.example .env.docker
# Edit .env.docker:
#   BACKEND_IMAGE=ghcr.io/<owner>/stockscreenclaude-backend
#   FRONTEND_IMAGE=ghcr.io/<owner>/stockscreenclaude-frontend
#   APP_IMAGE_TAG=v1.3.0
#   SERVER_AUTH_PASSWORD=choose-a-long-random-password
#   GROQ_API_KEY=...
ENABLED_MARKETS=US,HK,CN scripts/docker-compose-enabled-markets.sh --env-file .env.docker -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml pull
ENABLED_MARKETS=US,HK,CN scripts/docker-compose-enabled-markets.sh --env-file .env.docker -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml up -d --no-build
# Open http://localhost
```

This path deploys the tagged `v1.3.0` GHCR images instead of building locally. After the stack is up, the UI opens to a first-run bootstrap screen — see [First-Run Bootstrap](#first-run-bootstrap) for the staged pipeline and market selection.

For local development or contributor laptops, use the default local compose stack instead:

```bash
cp .env.docker.example .env
# Edit .env: set SERVER_AUTH_PASSWORD and add at least one LLM API key (e.g., GROQ_API_KEY)
scripts/docker-compose-enabled-markets.sh up
```

Full guide with homelab, VPS, and GHCR deployment options: **[Docker Deployment](docs/INSTALL_DOCKER.md)**

### Starting Only Enabled Market Workers

The default Compose file defines worker services for every supported market, but market-specific workers are behind Compose profiles. Use the helper script to start only the workers for the markets in `ENABLED_MARKETS`:

```bash
ENABLED_MARKETS=US,HK,CN scripts/docker-compose-enabled-markets.sh up -d
```

For `ENABLED_MARKETS=US,HK,CN`, Docker starts US/HK/CN market job and user scan workers. IN/JP/KR/TW/DE/CA/SG/MY/AU worker containers are not created. The global data-fetch worker listens only to `data_fetch_shared,data_fetch_us,data_fetch_hk,data_fetch_cn`.

The first-run bootstrap wizard still persists runtime choices in Postgres. Keep the wizard's enabled markets within the deployment `ENABLED_MARKETS` set. To add a market later, update `ENABLED_MARKETS` and recreate the stack:

```bash
ENABLED_MARKETS=US,HK,CN,TW scripts/docker-compose-enabled-markets.sh up -d
```

### From Source (Contributors)

See the **[Development Guide](docs/DEVELOPMENT.md)** for full backend + frontend + Celery setup.

## First-Run Bootstrap

On first launch the app boots into a setup screen — no pre-seeded database required. Pick a **primary market** (the one that opens first) and any additional **enabled markets** to hydrate in the background, then click **Start bootstrap**.

> **Bootstrap performance note:** selecting multiple enabled markets starts separate universe, price, fundamentals, breadth, group-rank, and scan work for each market. This can noticeably slow first load and ongoing data updates on smaller hosts or when upstream market-data providers throttle requests. For the fastest first run, start with one primary market and enable additional markets after the workspace is ready.

<img src="docs/screenshots/bootstrap-setup.jpg" alt="Bootstrap setup" width="500" />

*Primary-market dropdown and enabled-markets checkboxes on first launch*

The orchestrator runs a staged Celery pipeline for the primary market:

1. **Universe refresh** — seeds the market's symbol list (S&P 500 / Russell / NDX for US via `refresh_stock_universe`; official exchange feeds for HK / IN / JP / KR / TW / CN / CA / DE / SG / MY / AU via `refresh_official_market_universe`).
2. **Benchmark + price refresh** — imports the GitHub daily price bundle first, accepts recent stale bundles during bootstrap, then live-fetches only missing/current-session gaps (`7d` top-up for stale symbols, `2y` only for no-history symbols).
3. **Fundamentals refresh** — quarterly and annual financials.
4. **Breadth calculation** — StockBee-style advance/decline with gap-fill.
5. **Group rankings** — IBD-style relative strength across 197 industry groups.
6. **Feature snapshot** (US only) — daily feature rollup used by the Setup Engine.
7. **Initial autoscan** — seeds a first scan with the default profile so you land on populated results.

<img src="docs/screenshots/bootstrap-progress.jpg" alt="Bootstrap progress" width="500" />

*Per-stage progress with per-market queue status while the pipeline is running*

The workspace opens as soon as the primary market reaches `ready`. Secondary markets keep hydrating in the background on their own queues (`data_fetch_{us,hk,in,jp,kr,tw,cn,de,ca,sg,my,au}`) so you can start scanning immediately. Scans against a market that's still refreshing return HTTP 409 `market_refresh_active`, and scans against a market whose data is not current yet return the usual stale-data response instead of triggering a first-run block.

State is persisted in `AppSetting` under `runtime.primary_market`, `runtime.enabled_markets`, and `runtime.bootstrap_state` (`not_started` → `running` → `ready`). To re-run the wizard, reset `runtime.bootstrap_state` to `not_started`.

## Configuration

The AI chatbot requires at least one LLM provider API key. Scanning and all other features work without any keys.

| Provider | Env Var | Free Tier | Notes |
|----------|---------|-----------|-------|
| Groq | `GROQ_API_KEY` | Yes | Supported default for chatbot and research |
| Gemini | `GEMINI_API_KEY` | Yes | Supported extraction fallback |
| Minimax | `MINIMAX_API_KEY` | No | Supported primary theme-extraction provider |
| Z.AI | `ZAI_API_KEY` | No | Optional alternate provider |

Optional web search keys (`TAVILY_API_KEY`, `SERPER_API_KEY`) enable the chatbot's research mode.

Full reference: **[Environment Variables](docs/ENVIRONMENT.md)**

## Application Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Routine | Market dashboard with Key Markets, Themes, Watchlists, Stockbee tabs |
| `/scan` | Bulk Scanner | Multi-market scanning (US / HK / IN / JP / KR / TW / CN / DE / CA / SG / MY / AU) with 80+ filters, per-market badges, and CSV export |
| `/breadth` | Market Breadth | StockBee-style breadth indicators and trends |
| `/groups` | Group Rankings | IBD industry group rankings with movers |
| `/themes` | Themes | AI-powered theme discovery with trending/emerging detection |
| `/chatbot` | Chatbot | Multi-provider AI research assistant with web search |
| `/stock/:symbol` | Stock Detail | Individual stock analysis with charts and fundamentals |

## Key Capabilities

- **12 supported markets** — US, Hong Kong, India, Japan, Korea, Taiwan, mainland China, Germany, Canada, Singapore, Malaysia, Australia — with per-market exchange calendars, independent refresh queues, and scan-time freshness guards
- **First-run bootstrap wizard** with live staged progress and background hydration of secondary markets
- **6 screening methodologies** with composite scoring (Minervini, CANSLIM, IPO, Volume Breakthrough, Setup Engine, Custom)
- **80+ configurable filters** with saved presets across fundamental, technical, and rating categories
- **AI chatbot** with Groq-first routing, web search research mode, and persistent conversations
- **Theme discovery** from RSS, Twitter/X, and news sources with AI clustering and lifecycle tracking
- **Market breadth** dashboard with StockBee-style indicators and historical trends
- **197 IBD industry groups** ranked by relative strength with movers, constituent analysis, and a zoomable Relative Rotation Graph (RRG)
- **Watchlists** with RS/price sparklines, multi-period change bars, and drag-and-drop organization
- **MCP integration** for AI copilot workflows with 12 tools via stdio and Streamable HTTP ([details](docs/MCP_INTEGRATION.md))
- **TradingView-style charts** with candlestick OHLC and technical overlays
- **CSV export** for scan results
- **Dark and light mode** UI
- **Docker deployment** with PostgreSQL, auto-HTTPS, and GHCR image releases

## Documentation

| Guide | Audience |
|-------|----------|
| [Docker Deployment](docs/INSTALL_DOCKER.md) | Server, homelab, VPS users |
| [Development Guide](docs/DEVELOPMENT.md) | Contributors, developers |
| [Architecture](docs/ARCHITECTURE.md) | Understanding the system design |
| [Environment Variables](docs/ENVIRONMENT.md) | Configuration reference |
| [MCP Integration](docs/MCP_INTEGRATION.md) | AI copilot workflows |
| [Backend API & Architecture](backend/README.md) | Backend developers |
| [Frontend Components](frontend/README.md) | Frontend developers |
| [Contributing](CONTRIBUTING.md) | Getting started as a contributor |
| [ASIA v2 (HK/JP/TW) ADRs & Runbooks](docs/asia/README.md) | Multi-market operators, auditors |

## Tech Stack

**Backend:** FastAPI, SQLAlchemy, Alembic, Celery, Redis, PostgreSQL
**Frontend:** React 18, Vite, Material-UI, TanStack Query / Table, Recharts
**Data:** yfinance, Finviz, Alpha Vantage, SEC EDGAR, official X API (optional)

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always do your own research and consult with a licensed financial advisor before making investment decisions.

## License

Released under the [Apache License 2.0](LICENSE).
