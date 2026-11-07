# stockscreen-mcp

<!-- mcp-name: io.github.xang1234/stockscreen-mcp -->

Daily stock screens, relative-strength ratings, market breadth and industry group rankings for **12 markets**, available to Claude, Cursor, VS Code or any other [MCP](https://modelcontextprotocol.io/) client.

- No API keys and no server to run.
- It reads the snapshot that the open-source [stock-screener](https://github.com/xang1234/stock-screener) project publishes after every market close.
- That's the same data as the [live demo](https://xang1234.github.io/stock-screener/).

```bash
uvx stockscreen-mcp
```

**Markets:** United States, Canada, Germany, Hong Kong, mainland China, Japan, Korea, Taiwan, India, Singapore, Malaysia and Australia.

## Use it from the terminal

You don't need an AI assistant to read the screens. With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
uvx stockscreen-mcp screen                                # list the screens
uvx stockscreen-mcp screen minervini --market US          # today's list as a table
uvx stockscreen-mcp screen minervini --tradingview        # save a TradingView watchlist
```

- `--tradingview` writes every match to a comma-separated file such as `minervini-US-2026-11-04.txt`, ignoring `--limit`. TradingView's watchlist import requires a comma-separated file. Pass a filename to choose the path, or `-` for stdout.
- TradingView watchlists hold at most 1,000 symbols. A longer list is still written in full, with a warning on stderr.
- Symbols are derived from the published listing data and not checked against TradingView, so a few may not import.
- Malaysia and Beijing listings are skipped and listed on stderr, because they have no TradingView symbol this tool can derive.
- `--limit` sets how many rows the table shows, and `--json` prints those rows as JSON.
- The data date and any warnings go to stderr, as does the table's row count, so stdout stays clean for pipes. `--json` carries the count in its `total` field.

## What you can ask

- "Which stocks pass the Minervini trend template in the US today, and which of them are in top-20 industry groups?"
- "Run the CANSLIM screen for Japan and show the five strongest by 3-month RS."
- "How healthy is the US market right now? Exposure, distribution days, breadth."
- "What are the leading industry groups this week, and which groups moved up the most?"
- "Tell me everything the scan knows about NVDA, and which screens it passes."
- "Find Hong Kong stocks with RS above 90, in stage 2, with quarterly EPS growth over 25%."
- "Check my watchlist of NVDA, AMD and TSLA against the screens. Which rules does each one fail, and which are one rule away from passing?"
- "How have stocks done 20 sessions after entering the US Minervini screen, and is the sample big enough to trust?"
- "Which stocks entered or left the US Minervini screen since yesterday, and how did NVDA's RS rating change?"

## Tools

All tools are read-only.

| Tool | What it returns |
|---|---|
| `list_markets` | Markets in the current publish, with each one's data date, status and features |
| `list_screens` | The 28 preset screens and their exact rules: Minervini, CANSLIM, VCP, cup with handle, high tight flag, momentum, IPOs and more |
| `describe_fields` | What each scan field means, its unit and any hard range, with exact counts from this market's rows: how many hold a value, how many are true on a flag, and how many hold each label. Filter with `group` or name `fields`, which also reveals internal bookkeeping fields |
| `run_screen` | A preset and/or your own filters over every stock in a market's scan, sorted, with the match count. A preset returns exactly the stocks the website shows for it |
| `get_stock` | One stock: RS ratings (overall, 1M, 3M, 12M), stage, screener scores, growth, group rank, setup and pattern fields, and the screens it passes |
| `audit_stocks` | A watchlist of up to 50 symbols against the preset screens: the screens each stock passes, and near misses that fail exactly one of several rules, as the field and how far short it is (the field alone when the rule is not numeric or the value is missing). `detail` also returns every failed rule per failing screen; responses over 80,000 characters are refused with a hint of how many symbols fit |
| `screen_track_record` | What happened after stocks newly entered each preset screen, for one holding period per call (5, 20 or 60 sessions): median return, median excess over the benchmark, positive rate and sample counts. Not every market publishes one |
| `what_changed` | Which stocks entered and left each preset screen between two archived sessions, by default the two newest. Moves where a field the rules read gained or lost its value are listed apart, as `entered_with_new_data` and `left_with_lost_data`. For up to 50 symbols it adds the screens each moved into and out of, every tracked field that changed, as from and to, and the tracked fields null in both sessions. A growth change carries each side's growth basis. Warnings flag moves that can come from the data rather than the stock, such as a rule change or a change of RS universe. A market needs two archived sessions |
| `market_overview` | Exposure score and stance, distribution days, follow-through day, StockBee-style breadth (4% movers, 5/10-day ratios, T2108, 52-week highs/lows), key index moves, top groups |
| `group_rankings` | IBD-style industry groups ranked by relative strength, with 1W/1M/3M/6M rank changes and weekly movers. Pass a group for its rank history and strongest stocks |

### Reading the data

- **RS ratings** (`rs_rating`, `rs_rating_1m/3m/12m`) are percentiles from 1 to 99 *within the stock's own market*, where 99 is the strongest.
- **Group rank 1** is the strongest group.
- **Track records** have one row per screen definition and origin. Compare the rows separately and don't rank `thin` ones. Each result's `how_to_read` note explains the numbers.
- **What changed** compares sessions as they were first published. Read its `warnings` before blaming a move on the stock, and remember that prices are not split-adjusted across sessions.
- **Every result has an `as_of_date`:** the last completed session the publish covers. This is end-of-day data with nothing intraday, and it is never fetched live from price providers.

### For agents

- **Skill:** the `stockscreen://skill` resource is an Agent Skills file teaching the tool sequence and the reading rules that prevent wrong answers.
- **Prompts:** `review_watchlist` and `find_candidates` start that sequence.
- **Field dictionary:** the `stockscreen://fields` resource holds every scan field's meaning, unit and hard range. `describe_fields` adds per-market counts. Without the server, read [fields.json](https://xang1234.github.io/stock-screener/fields.json) and [llms.txt](https://xang1234.github.io/stock-screener/llms.txt).

## Setup

The server needs [uv](https://docs.astral.sh/uv/getting-started/installation/), which provides `uvx`.

**Claude Code**

```bash
claude mcp add stockscreen -- uvx stockscreen-mcp
```

**Claude Desktop**: add to `claude_desktop_config.json`.

```json
{
  "mcpServers": {
    "stockscreen": { "command": "uvx", "args": ["stockscreen-mcp"] }
  }
}
```

**Cursor**: add the same `mcpServers` entry to `~/.cursor/mcp.json`.

**VS Code**: add to `.vscode/mcp.json`.

```json
{
  "servers": {
    "stockscreen": { "type": "stdio", "command": "uvx", "args": ["stockscreen-mcp"] }
  }
}
```

**Codex CLI**: add to `~/.codex/config.toml`.

```toml
[mcp_servers.stockscreen]
command = "uvx"
args = ["stockscreen-mcp"]
```

To run the newest version from `main` rather than the latest release, replace `stockscreen-mcp` in `args` with `--from`, `git+https://github.com/xang1234/stock-screener#subdirectory=stockscreen-mcp`, `stockscreen-mcp`.

## How it works

- **Data source:** the server reads the JSON that the static site publishes to GitHub Pages. The root `manifest.json` points at the current publish.
- **Download and caching:** each market's scan is downloaded once, on first use (US is about 7 MB compressed), and kept in memory. The two most recently used markets stay cached, and the cache refreshes when a new publish appears.
- **Data URL:** set `STOCKSCREEN_DATA_URL` to read from another publish, such as a fork's GitHub Pages site.
- **History:** `what_changed` reads the market history archive, a GitHub Release per market (`market-history-<m>-v1`) that holds one file per published session and an `index.json`. A US session is about 2.3 MB compressed. The index is cached for 10 minutes, and the four most recently used sessions stay in memory. Set `STOCKSCREEN_HISTORY_URL` to read another repository's Releases, such as `https://github.com/<owner>/stock-screener/releases/download/`.

## Development

```bash
cd stockscreen-mcp
uv run pytest                                      # unit tests, no network
STOCKSCREEN_LIVE=1 uv run pytest -m live           # against the real publish
```

### Releasing

1. Bump `__version__` in `src/stockscreen_mcp/__init__.py`. Bump the same version in `server.json` (both `version` fields) and in `mcpb/manifest.json` and `mcpb/pyproject.toml`. `tests/test_packaging.py` fails if they drift.
2. Push a `stockscreen-mcp-v<version>` tag. `.github/workflows/stockscreen-mcp-release.yml` publishes to PyPI and then to the [MCP registry](https://registry.modelcontextprotocol.io/), through `stockscreen-mcp-registry.yml` with GitHub OIDC.
3. **Bundles** (needs Node 20+, which the Smithery CLI requires; `smithery_bundle.py` checks it), built from `mcpb/`:
   - **Claude Desktop:** `npx @anthropic-ai/mcpb pack . stockscreen-mcp.mcpb` builds an MCPB `uv` bundle.
   - **Smithery:** its CLI can't read the `uv` type yet ([smithery-cli#801](https://github.com/arcadeai-labs/smithery-cli/issues/801)). `python smithery_bundle.py` packs a copy typed `python` as `stockscreen-mcp-smithery.mcpb`.
   - **Publish to Smithery:** `smithery mcp publish stockscreen-mcp-smithery.mcpb -n davidten7/stockscreen-mcp`, using the maintainer's Smithery namespace.

## License and disclaimer

Apache-2.0. This is screening data for education and research, **not investment advice**.
