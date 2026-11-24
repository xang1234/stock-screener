# Minervini X list: problems, trends and tools, August to November

Date: 2026-11-23. These are research recommendations, not an implementation specification. This report follows [the 2026-11-22 report](minervini_x_ideas_2026-11-22.md), covers a longer window, and focuses on three things: the problems traders say they face, how the market and methods changed, and the tools people shared.

Source: X list `1522014550211457024`, read with the local `xui-reader` skill.

## Pull summary

- **Window:** 2026-08-14 to 2026-11-23.
- **Size:** 8,559 unique posts from 120 authors. By month: August 1,511 (a partial month), September 2,186, October 2,536, November 2,325. 97 of the 102 days returned a full sample.
- **Method:** one search per day, `list:1522014550211457024 since:D until:D+1 min_faves:10 -filter:replies`, capped at 120 posts. A random pause of 25 to 90 seconds followed each day, and 5 to 10 minutes followed a throttled day. The posts are merged with last month's weekly pull. Most days hit the 120 cap, so each day is a sample of the most-liked posts, not every post.
- **Throttling:** after about four or five full days, X returns a day cut to 20 posts, then one or two empty days, with no error. The longer pause after a throttled day reduced each run of empty days to one. On retry, 18 of the 23 thin days came back full, including last month's 11/16–11/19 gap, so those gaps were throttling, not days without posts. Days still thin: 2026-09-04, 09-24, 09-29 and 11-07 (empty) and 10-12 (40 posts).
- **Reader bug still open:** `src/xui_reader/scheduler/read.py:439` passes the search tab as `user_tab`. This pull used the same shim as last time.

Most-mentioned tickers by month:

| Month | Tickers | What it shows |
| --- | --- | --- |
| August | NVDA 67, PLTR 54, MU 45, QQQ 34, SPY 33 | AI leaders, with biotech, crypto and energy beside them |
| September | MU 128, SNDK 80, QQQ 71, AMD 71, DELL 61 | Memory and storage take over |
| October | MU 103, QQQ 97, PLTR 59, SNDK 58, NVDA 58 | Optics (LITE 56) leads while the index becomes the main subject |
| November | QQQ 106, NVDA 104, SPY 75, SNDK 37, MU 35 | Talk is about the indexes; SPOT and U appear as new names |

## How the market and methods changed

| Month | Regime | Evidence |
| --- | --- | --- |
| August | Uptrend from the April FTD, getting narrower. Rotation out of tech hurt recent breakouts, and former leaders ran into overhead supply | [Minervini](https://x.com/markminervini/status/2090085196619993139); an NVDA rally with only 161 S&P stocks advancing ([post](https://x.com/markminervini/status/2092979126449635511)); strength in biotech, shipping, energy and gold ([Soreide](https://x.com/LeifSoreide/status/2090816853270561225)) |
| September | Rates and oil weighed on stocks and the market chopped. New lows beat new highs while the index held up. Leadership was optics, semis and storage | [Minervini](https://x.com/markminervini/status/2103112011563389210), [leadership](https://x.com/markminervini/status/2102405893451092003) |
| October | The Nasdaq made a new high with about 20% of stocks above their 50-day, then broke its 50-day on 10/17 | [ConnorJBates](https://x.com/ConnorJBates_/status/2108167454534803751); "34% of NASDAQ stocks above their 200-day" ([Minervini](https://x.com/markminervini/status/2110792793857593402)); [RossHaber](https://x.com/RossHaber/status/2111572541273985480) |
| November | The VIX reached Minervini's 28.5 bear-market warning on 11/6, and QQQ was down 20% by 11/21. The 11/11 FTD was graded "C" and led by stocks at their lows. RS moved to healthcare and biotech | [Minervini](https://x.com/markminervini/status/2118715580840644669), [Upticken](https://x.com/Upticken/status/2120617869985579203), [NickSchmidt](https://x.com/NickSchmidt/status/2122030102506614790), [Speculator_io](https://x.com/Speculator_io/status/2122129676319551534) |

Method trends, as posts per 1,000 each month from August to November. The counts come from keyword regexes over the first 7,400 posts, before the retried days were added, so treat them as rough.

- **Shorting and hedging rose (39, 33, 46, 51).** "Pullbacks into rising MAs are long interest, bounces into declining MAs are short interest" is the rule repeated most often ([ohiain](https://x.com/ohiain/status/2113326419388317946)). Minervini hedged with SPY ([post](https://x.com/markminervini/status/2120499249733423221)).
- **Cash and sitting out rose (18, 15, 31, 25)** ([Komar](https://x.com/BlogJulianKomar/status/2118454562436612241)).
- **Stage analysis rose (33, 22, 27, 39).** "Never buy under a declining 30-week" ([TraderLion](https://x.com/TraderLion/status/2120140120841990161)).
- **FTD and rally-day counting surged in November** (8 to 23). People counted the days by hand ([Venu](https://x.com/Venu_7_/status/2120536738137575470)).
- **Breakout and "tight" talk halved** (breakout 45 to 23, tight 44 to 18). 21 EMA pullbacks peaked in October. Komar: trust each repeat pullback to the 21 EMA less ([post](https://x.com/BlogJulianKomar/status/2105275239600914797)).
- **RS went from something to trade to something to track.** "Don't trade RS in a downtrend, track it" ([ohiain](https://x.com/ohiain/status/2106171315464114316)), and watch "the stocks that have corrected the least" ([Venu](https://x.com/Venu_7_/status/2118399368814608562)).
- **Entry and exit studies built from data grew.** Examples: entry extension measured in ADR units, and selling on day 5 to 8 ([Muninn](https://x.com/Muninn/status/2089746393183256879), [Muninn](https://x.com/Muninn/status/2111439836309585934)).

## Problems traders said they faced

| Problem | Peak | In their words | App today |
| --- | --- | --- | --- |
| Months of chop grinding accounts down | September | "Nearly 4 months of tough chop" ([ZaStocks](https://x.com/ZaStocks/status/2111108900359700531)); "death by a thousand cuts" ([LoneStockTrader](https://x.com/LoneStockTrader/status/2108240080439537958)) | The breadth page shows the inputs, but nothing says "breakouts aren't paying right now" outside the Backtest page |
| The index looked fine while most stocks didn't | September to October | "Indexes are bifurcated" ([Minervini](https://x.com/markminervini/status/2110792793857593402), [NickSchmidt](https://x.com/NickSchmidt/status/2110462774455480746)) | Breadth has up and down 4% counts, advance/decline, new highs and lows, and T2108. It has no % above the 50- or 200-day and no divergence flag |
| Breakouts failing | October to November | "Failed break outs/double tops everywhere" ([CFlanders7](https://x.com/CFlanders7/status/2110458870711611612)) | Covered on the Backtest page by the 20-entry-session window (#770). It isn't shown where scans are run |
| Giving back winners | September to October | "Constantly giving 4–5R winners back breakeven" ([ShakePryzby1](https://x.com/ShakePryzby1/status/2110094061340356997)); "the Trading Cardinal Sin" ([Minervini](https://x.com/markminervini/status/2102046227990765944)) | Backtest reports end-of-horizon returns and the 7% stop, not the share of the best move that was given back |
| Bottom fishing Stage 3 and 4 names | November | "Bottom Fishers Beware!" ([Minervini](https://x.com/markminervini/status/2121251695426449746)); buying declining 9 and 21 EMAs "is a pretty good way to collect paper cuts" ([ohiain](https://x.com/ohiain/status/2118005850673488342)) | The stage is stored. There is no MA-slope field, and no warning when a watchlist name drops into Stage 4 |
| Telling an FTD from a bear-market rally | November | "Another snapback rally within a broader downtrend" ([Minervini](https://x.com/markminervini/status/2119809854881620185)); [Soreide](https://x.com/LeifSoreide/status/2120523122826694700) | The state machine (#769) covers rally days and failures. It doesn't grade FTD quality |
| FOMO, overtrading, forcing trades | Steady, rising into November | "Forcing trades due to FOMO" ([CFlanders7](https://x.com/CFlanders7/status/2115951949002739950)); [Trader_Dante](https://x.com/Trader_Dante/status/2117984800136913002) | There is no trade data, so the app can't measure it |
| Never reviewing your own trades | September to October | "5 years repeating the same mistakes" ([ohiain](https://x.com/ohiain/status/2113982591099609182)); David Ryan's audit found "late buys" ([TraderLion](https://x.com/TraderLion/status/2102337290705977371)) | No journal |
| Leadership rotating every few weeks | October to November | "Money keeps jumping from group to group" ([Soreide](https://x.com/LeifSoreide/status/2111138879721922788)) | Group ranks exist, but there is no view of rank changes and no leader-health view |

## Tools people shared

Deepvue was the most active toolmaker in the window, with about 15 product posts. Shared AI workflows were the other growing category.

| Tool | Job it does | Evidence | App today |
| --- | --- | --- | --- |
| Trade journal with automatic broker import (Deepvue Journal) | Win rate, profit factor, equity curve, P&L tagged by emotion, AI questions about your own trades | [Deepvue](https://x.com/Deepvue/status/2111427011071582561), [Moglen](https://x.com/RichardMoglen/status/2104242809184440403) | Gap |
| An LLM reviewing a trade journal | A weekly review by Claude, Grok or Notion AI over thousands of entries. One trader learned to "cut size faster in slow periods" | [TedHZhang](https://x.com/TedHZhang/status/2091646111056339335), [TheOneLanceB](https://x.com/TheOneLanceB/status/2095536877361365042) | Gap. MCP tools cover screens, not the user's own trades |
| A "tight" screen using RMV (relative measured volatility) | RMV 0 marks a very tight area before a pivot. One trader called it a "cheat code" | [FranVezz](https://x.com/FranVezz/status/2116600863519035493), [Moglen](https://x.com/RichardMoglen/status/2102875749275390084) | Partial. VCP and three-weeks-tight detectors exist, but there is no tightness number to filter on |
| Komar's free TradingView screener | 70%+ above the 52-week low, ADR ≥ 4.5%, price above the 8 and 21 EMA | [Komar](https://x.com/BlogJulianKomar/status/2100929109975060609) (1.4k likes), [Komar](https://x.com/BlogJulianKomar/status/2093981874783731954) | Partial. `adr_percent` is in the feature store, but there's no EMA position preset |
| Model books | Save a chart to a tagged setup library in one click | [AmeetRai](https://x.com/AmeetRai/status/2093389872014991461) | Gap |
| Historical screener, plus indicators written in plain English | Run any screen as of a past date; describe a new field in words | [AmeetRai](https://x.com/AmeetRai/status/2113330001621754112), [Moglen](https://x.com/RichardMoglen/status/2105837002021331059) | Partial. Backtest replay and `describe_fields` exist; users can't define fields |
| Top movers tagged by catalyst | Daily movers with the reason, back-filled for 20 years | [AmeetRai](https://x.com/AmeetRai/status/2098464620063138199) | Gap |
| Sentiment panels (NAAIM, AAII, COT) | Weekly positioning as context for exposure | [Deepvue](https://x.com/Deepvue/status/2100656025049198885), [NickDrendel](https://x.com/NickDrendel/status/2100664727903072667) | Gap |
| An insider and political-trade connector for ChatGPT and Claude | An LLM can query insider and congressional trades | [QuiverQuant](https://x.com/QuiverQuant/status/2098405396360687752) (2.4k likes) | Partial. The app has MCP tools but no insider data |
| A stage-change screener | Weekly Stage 2A breakouts: stage change, close in the top quarter of the range, $100M+ dollar volume | [stageanalysis](https://x.com/stageanalysis/status/2113993543555428808) | Partial. Stage is stored, but there's no stage-change event |
| FTD grade by leader count | Grades the FTD A to D by how many new leaders take part | [Upticken](https://x.com/Upticken/status/2120617869985579203) | Gap (this is #32) |

Complaints about tools: TradingView's mobile app is "really bad and incomplete" ([TheShortBear](https://x.com/TheShortBear/status/2119158283638395102), 680 likes). Deepvue's stage-cycle chart doesn't go back far enough ([CFlanders7](https://x.com/CFlanders7/status/2108183260438188400)). One trader spends over $40k a year on data for his own market-overview app ([RealSimpleAriel](https://x.com/RealSimpleAriel/status/2091295615183434025)). Opinions on AI chart reads were mixed ([AsymTrading](https://x.com/AsymTrading/status/2095117397682196481)).

## What has shipped since the last two reports

- **#31, the rally-attempt state machine:** shipped as #769.
- **#33, setups working now:** shipped as #770 on the Backtest page. It isn't on the Daily page yet.
- **Everything else is still open:** #32 and #34–#39 from last month, and the breadth and stalk-trigger priorities from October.

## New candidates

Numbering continues from the 2026-11-22 list. Effort: S means days, M means one to two weeks.

| # | Idea | Effort | Note |
| --- | --- | --- | --- |
| 40 | **Tightness score**: today's range or ATR as a percentile of the last N sessions, as a scan column, a filter and a `what_changed` event | S | Deepvue's most-shared screen; complements VCP |
| 41 | **Entry extension in ADR units**: distance from the pivot or the 21 EMA divided by `adr_percent` | S | Muninn's and jfsrev's entry rule; uses a field that already exists |
| 42 | **Pullback count**: how many times a stock has pulled back to its 21 EMA since the trend began | S | Komar's "trust each repeat pullback less" |
| 43 | **Give-back stat on Backtest**: the best excess return inside the horizon against the final result, and how often a +10% move closed below +2% | S | Answers "giving back winners" with the validation data the app already has |
| 44 | **Breadth divergence flag**: the index within 3% of its high while the % above the 50-day is below 40% or new lows beat new highs | S | Needs #4's `pct_above_50dma` first |
| 45 | **Stage-change events and method badges**: "entered Stage 2" and "13-week-high breakout" events, plus compact badges (MM, ON, VCP) in results and TradingView exports | S | Lists are often shared with badges like these |
| 46 | **Trade journal import plus an `audit_my_trades` MCP tool**: a broker CSV in, each entry joined to the app's setup, RS, stage, extension and market cycle on that date | M | The biggest gap. Journal reviews were the AI workflow shared most often |
| 47 | Sentiment panel (NAAIM, AAII, COT) on the breadth page | S | Free weekly series |
| 48 | Sector relative drawdown over 21 sessions against the benchmark | S | Minervini's "worst since 2001" chart |

## Five priorities

These are ranked by how well each answers the problems above, and by cost. The ranking is qualitative.

### 1. Breadth participation and divergence (October's priority 2, plus #44)

**Why first.** It was the earliest warning before the October top ("34% above the 200-day" while the Nasdaq made a new high), and traders computed it by hand every week. It's been on the list since October, and the problem table shows why it matters. The `market_breadth` row has up and down counts, new highs and lows, and T2108, but no % above the 20-, 50- or 200-day and no % of stocks 20%+ off their highs.

**First slice.** Add `pct_above_20dma`, `pct_above_50dma`, `pct_above_200dma` and `pct_down_20_from_high` to the breadth calculation, using the existing eligible-universe counts. Then add the #44 flag, and pass the flag to market exposure as context, not as a score input.

**Status.** #771 ships the 200-day share (`above_sma200_pct`), the 20%-off-high share (`down_20pct_from_high_pct`) and the flag (`breadth_divergence` on the exposure payload). `feat/breadth-50-20-ftd-leaders` adds the 20- and 50-day shares (`above_sma20_pct`, `above_sma50_pct`), each over its own eligible universe, on the breadth row and in the breadth UI. Priority 1 is done.

**Mid-session.** The breadth calculation runs from end-of-day data. Count only the last completed session.

### 2. FTD quality grade (#32)

**Why.** Now that #769 marks FTDs, the next question in November was whether this one is any good: "led by stocks at lows" ([NickSchmidt](https://x.com/NickSchmidt/status/2122030102506614790)), "C" ([Upticken](https://x.com/Upticken/status/2120617869985579203)).

**First slice.** On an FTD date, store (a) the share of that day's 4%+ gainers within 15% of their 52-week high, and (b) the count of new Minervini-template entries over the next five sessions. Show both on the exposure card beside the FTD marker.

**Status.** #771 ships (a), as `up_4pct_near_high_count` on the breadth row and `ftd_quality` on the exposure payload. `feat/breadth-50-20-ftd-leaders` ships (b). The breadth row counts `trend_template_entry_count`, the stocks that pass the price criteria of the trend template today but not the session before. RS rank is left out, since breadth has no cross-section to rank against. `ftd_quality.new_template_entries` sums that count over the five sessions after the FTD, and the exposure card shows it beside the near-high share. Priority 2 is done.

### 3. Per-stock trend state: MA slope, pullback count, extension in ADR units (#34, #41, #42)

**Why.** "Rising MA long, declining MA short" was the rule repeated most often in all four months, and the bottom-fishing problem peaked in November. All three are fields derived from price, and ADR is already stored.

**First slice.** `ema21_slope` and `sma50_slope` (the sign and % per week), `ema21_pullbacks_in_trend`, and `extension_adr` as scan columns and filters. Then a "bounce into a declining MA" badge.

### 4. Tightness score and give-back stat (#40, #43)

**Why.** These are two cheap additions to features that exist. A tightness column covers the most-shared Deepvue screen without copying it. A give-back stat turns "giving back winners" into a number on the Backtest page, split by market cycle, which #770 already supports.

### 5. Trade journal import and `audit_my_trades` (#46)

**Why.** Three of the problem clusters (giving back winners, overtrading and never reviewing) need the user's own trades, and journaling plus LLM review was the tool trend most often shared. The app already has what a journal lacks: the setup, stage, RS and market cycle on any date, plus MCP tools. A CSV import and one MCP tool would let an agent answer "do I lose on extended entries, or when the cycle is in correction?"

**Risk.** It's a new domain with personal financial data. Start with an import that stays on the user's machine, store no broker credentials, and build no order paths.

## Close runners-up

- **Show "setups working now" on the Daily page**, from the #770 plan. The chop and breakout-failure problems are the reason to do it.
- **#45 stage-change events**: this is where the next leaders will come from once the correction ends.
- **#38 sessions since the earnings reaction**: tiny, and the 3-day rule was reposted again in November.

## Non-goals

These are unchanged: intraday entries, market calls, copying one trader's system, and placing orders. Also out of scope: a social trading layer (synced lists or community screens) and a model book with images.
