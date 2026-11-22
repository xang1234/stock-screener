# FTD state machine and "setups working now"

Goal: replace the v1 follow-through-day heuristic with a market-cycle state machine, and answer "are setups working right now?" from the backtests the live and static sites already publish. Source: priorities 1 and 2 of [the 2026-11-22 X list review](../docs/research/minervini_x_ideas_2026-11-22.md).

Status: design review. No code yet. Claims are labelled measured, read in code, or inferred.

## 1. Market-cycle state machine

### What v1 does

`detect_follow_through_day` (`backend/app/services/market_exposure_service.py:129`) flags the latest session in a 15-session window that closed at least 1.5% higher on rising volume, four or more sessions after the lowest close of the last 20 sessions. Its docstring calls itself a "v1 HEURISTIC" and defers the "full rally-attempt FSM" to v2. It feeds two things.

- `_score` raises the exposure score to a floor of 45 when the flag is set and the index is at or below its 50-DMA.
- `build_exposure_payload` marks a history point as an FTD when `follow_through_date` equals the row's date, and `MarketHealthExposure.jsx` draws those points on the exposure timeline and shows "Follow-through day: Yes/No".

Gaps, read in code:

1. No correction precondition. The anchor is the lowest close of any 20-session window, so routine pullbacks inside an uptrend produce FTDs.
2. The anchor is a close, not the rally low, and nothing counts rally days.
3. Nothing marks an FTD as failed. A new low only resets it by moving the 20-session minimum.
4. No state persists, so the app can't say "rally attempt, day 3".

### Prototype result

[`ftd-state-machine-proto.py`](ftd-state-machine-proto.py) (run `backend/venv/bin/python -I .plans/ftd-state-machine-proto.py SPY 0.08`) ran v1 and a state machine over yfinance SPY and QQQ daily bars, 2021-06-01 to 2026-11-20. Measured:

| | SPY v1 | SPY FSM | QQQ v1 | QQQ FSM |
|---|---|---|---|---|
| FTD event days since 2022-01-01 | 28 | 9 | 57 | 17 |
| Floor-eligible days (flag set, price at or below 50-DMA) | 45 | | 63 | |
| Floor days with no FSM-confirmed FTD in the last 15 sessions | 3 | | 4 | |

The state machine's QQQ events for November 2026 match how the list talked about it: correction from 10-30, attempt reset on 11-05, FTD on day 4 on 11-11, failed on 11-19 when QQQ broke the rally low.

What the numbers mean: v1 shows about three times too many FTD markers on the timeline. The score barely moves, because the `price <= ma50` gate filters out most false flags before the floor applies. The case for the state machine is the timeline, the rally-day count and the failure signal. The score correction is secondary.

### Data shape

A reducer over the index bars, one state per bar. `compute_exposure` runs it each day over the last 252 sessions of the benchmark frame, so no state is carried between rows. A daily run and a backfill then agree on a date whenever the frame holds 252 sessions before it.

Known limits, left for a follow-up that would persist a reducer checkpoint:

- A correction whose starting peak is more than 252 sessions old is forgotten when that peak leaves the window. On SPY, 2022 to 2026, the windowed state differs from one full-history pass on 38 of 1,227 sessions, and all 8 FTD dates match.
- The benchmark frame is two years back from the run date. A manual backfill of a session more than about a year old can have fewer than 252 earlier bars, so its cycle starts from an assumed uptrend. The automatic 220-day seed is not affected.

```python
class CycleState(StrEnum):
    UPTREND = "uptrend"
    CORRECTION = "correction"
    RALLY_ATTEMPT = "rally_attempt"

@dataclass(frozen=True)
class MarketCycle:
    state: CycleState
    rally_low: float | None        # intraday low being defended
    rally_low_date: date | None
    rally_day: int | None          # set only in RALLY_ATTEMPT
    ftd_date: date | None          # latest FTD whose rally low still holds
    failed_ftd_date: date | None   # set on the session an FTD's rally low broke

class CycleBar(NamedTuple):    # domain code may not import pandas
    date: date
    close: float
    low: float
    volume: float

def market_cycle(bars: Sequence[CycleBar], params: CycleParams) -> list[MarketCycle]: ...
```

Transitions, as prototyped:

| From | Event | To |
|---|---|---|
| UPTREND | close at least `correction_pct` below the highest close since the last FTD or start | CORRECTION, rally low is that bar's low |
| UPTREND | close below the confirming FTD's rally low | CORRECTION, record `failed_ftd_date` |
| CORRECTION | a bar's low undercuts the rally low | CORRECTION, new rally low |
| CORRECTION | close above the prior close | RALLY_ATTEMPT, day 1 |
| RALLY_ATTEMPT | a bar's low undercuts the rally low | CORRECTION, attempt reset |
| RALLY_ATTEMPT | day 4 or later, gain of at least `ftd_gain_pct`, volume above the prior session | UPTREND, `ftd_date` set |
| RALLY_ATTEMPT | day reaches `attempt_expiry_sessions` without an FTD | CORRECTION, same rally low. The next up close is day 1 |
| CORRECTION or RALLY_ATTEMPT | close above the highest close the correction started from | UPTREND, no FTD |

### Parameters, settled

[`ftd-state-machine-sweep.py`](ftd-state-machine-sweep.py) ran the variants over every market's benchmark from yfinance, 2022-01-01 to 2026-11-20. CN's `000300.SS` returned one row from yfinance, so CN is untested there. The app reads CN bars from its own database.

| Parameter | Value | Why |
|---|---|---|
| `correction_pct` | 8% | 7% added corrections without changing the FTD dates that matter |
| Entry on a 50-DMA break with heavy distribution | Rejected | SPY entered "correction" on 2026-09-10 and stayed there through October while it sat about 1% from a record |
| `ftd_gain_pct` | 1.5%, every market | A 0.9×ATR rule added FTDs on quiet indexes (AU 8 to 15, MY 3 to 10) without fixing a miss |
| `ftd_min_rally_day` | 4 | Unchanged from v1 |
| `attempt_expiry_sessions` | 25 | Without it, attempts on IN, AU and MY ran 127, 79 and 156 sessions |
| Recovery exit | On | Without it a slow grind back to a new high never leaves CORRECTION, because it never posts a 1.5% day |

The SPY result with these values: 8 corrections, 8 FTDs, 1 failure since 2022. The latest FTDs are 2024-08-13, 2025-04-22 and 2026-04-08, and SPY entered correction on 2026-11-19.

A benchmark whose volume is missing or zero can't confirm an FTD, as with v1. Every benchmark tested had volume on at least 99% of bars.

**Golden fixtures.** Synthetic frames for each transition, plus a recorded SPY frame pinned to the events above.

### Wiring

- `market_cycle` replaces `detect_follow_through_day` and its four `FTD_*` constants in one PR. No caller keeps the old function.
- `MarketExposure` gains `cycle_state`, `rally_day`, `rally_low_date` and `failed_ftd_date`. `follow_through_day` and `follow_through_date` keep their names with a stricter meaning: an FTD whose rally low still holds.
- `_score` applies the floor when `state == UPTREND` and `ftd_date` is within 15 sessions, and the index is at or below its 50-DMA. This changes 3 of 45 floor days on SPY since 2022 (measured).
- `build_exposure_payload` adds the state, the rally day and a failed-FTD marker to each history point. The timeline shows FTDs and failures. The detail row reads "Rally attempt, day 3" instead of "Follow-through day: No".
- Run `backfill_exposure` once per market after deploy, because every stored FTD marker changes. Say so in the PR.

**Mid-session.** `compute_exposure` checks only that the frame's last bar is dated `as_of_date`. It does not check that the session has closed. The guarantee comes from the callers, and this change does not alter it.

- `calculate_market_exposure` defaults to `last_completed_trading_day(market)`, and the daily pipeline runs it after the close. That path asks only for a completed session. It can still score a partial bar if a benchmark bar fetched during that session is still cached, because the cache matches bars by date, not by fetch time.
- An explicit `calculation_date`, or a `backfill_market_exposure` range that ends on the current session, can name a session still in progress. `resolve_benchmark_bundle` is not cache-only, so it can fetch that session's partial bar from Yahoo. The cycle then scores the partial bar, and a rally day or an FTD can appear and later vanish when the session completes.
- A run that straddles the close can see the session as open when it fetches and closed when it stores.

Run manual backfills when the market is closed, with `end_date` no later than `last_completed_trading_day(market)`. Rejecting an in-progress `as_of_date` inside `compute_exposure` is an open follow-up, not part of this change.

### FTD leadership

On an FTD date only, compute the share of the market's stocks that closed up 2% or more and sit within 15% of their 52-week high, against the share within 15% of their low. Store it on the FTD row as `ftd_leadership_pct`. This is NickSchmidt's April-versus-November comparison. It needs stock-level bars for one date, read from `stock_prices`, so it costs one query per FTD. It ships after the state machine.

### Delivery

1. Pure `market_cycle` plus golden tests, and a script that prints events per market for review.
2. Wire it into `compute_exposure`, add the migration, delete v1, backfill.
3. FTD leadership and the timeline changes.

## 2. "Setups working now" from the existing backtests

### What exists

| Surface | Events | History, as of 2026-11-22 | Metrics |
|---|---|---|---|
| Static scoreboard (`build_static_scoreboard.py`) | A symbol newly entering a preset, per `(preset, definition, origin)` cohort | Ledger seeded on 2026-11-05 from about 30 days of artifacts | 5/20/60 sessions, positive rate, beat-benchmark rate, medians, `effective_dates`, `thin` |
| Live Backtest, `scan_pick` | Top 10 composite picks per published run | Published runs since 2026-10-09 | 1/5/20/60 sessions, 7% stop, MFE/MAE, groups by stage, setup pattern and extension |
| Live Backtest, `replay_pick` | Point-in-time Minervini top 10 | US 2026-04-06 to 2026-10-29. HK, JP and KR not run | Same as `scan_pick` |

Read in code and memory: the static and live sources hold about six weeks. A "recent against 365-day base" comparison on them compares six weeks with six weeks. Only the replay source covers a full cycle: the April correction, the 04-08 FTD, the summer uptrend and the October top.

### Data shape

One new idea: slice by when the entry happened. Both surfaces already call `summarize_horizon`. Add a recent summary that runs the same function over events whose entry session is among the last 20 sessions with a closed 5-session window, and show it beside the full-window summary.

```python
@dataclass(frozen=True)
class RecentVsBase:
    horizon_sessions: int              # 5
    recent: ValidationHorizonSummary   # entries in the last 20 closed sessions
    base: ValidationHorizonSummary     # the existing full-window summary
```

- **Static.** Each cohort's block in `scoreboard.json` gains a `recent_5` horizon block, built by the existing `_horizon_block` over the filtered events. Cohorts stay unpooled, as the scoreboard's rules require. The regime read is the count of presets whose recent beat-benchmark rate sits below their base rate, not one pooled number.
- **Live.** `ValidationOverviewResponse` gains `recent` next to `horizons`. The window rule lives in `ValidationService` once, so the frontend doesn't rebuild "closed in the last 20 sessions" from `lookback_days`. The existing `lookback_days=30` doesn't give the same answer, because it filters by signal date and counts pending windows.
- **Cycle state at entry.** Add `market_state` as a group dimension in `_group_values`, read from the `MarketExposure` row on the event date. The Backtest page then answers whether picks taken during a rally attempt work, which ties priority 1 to priority 2. On replay, this covers April to October.

### Step 0: test the signal on replay before building UI

A 20-session window of 5-session outcomes lags by 5 to 25 sessions. It may only confirm what the exposure score already says. Test it before building UI.

1. Rerun `replay_scan_picks --market US` for 2026-10-30 to 2026-11-20, so the replay covers the November breakdown.
2. On a read-only session against the live database (see the memory note on branch code against the live DB), compute the rolling 20-session recent positive rate and beat-benchmark rate of replay picks for each date. Compare each with the next 20 sessions' pick outcomes and with the state machine's state.
3. Decision rule: put a strip on the Daily page only if the recent rate changes before the state machine's correction or FTD dates on at least one of April or October. Otherwise ship it as a Backtest-page table only.

This could not run in the review session, because the StockScreen stack wasn't running locally.

Caveats: replay picks use today's screener code and today's universe, so they're survivor-biased. Don't compare them with published picks from before #671 (2026-10-22).

**Mid-session.** Both builders read stored prices capped at the export or as-of date and never call a provider. The recent window counts closed windows only, so a session in progress adds nothing.

### Delivery

1. Step 0 measurement, written up in this file. Not run yet. The Daily-page strip waits on its decision rule.
2. Done: `market_state` group dimension on scan and replay picks.
3. Done: `recent` on the overview and on each group, and the Backtest page's recent section and group column. The window starts no earlier than the lookback cutoff, so the 30D view holds 16 to 19 entry sessions, not 20.
4. `recent_5` in the static scoreboard and `StaticScoreboardPage.jsx`.

## Non-goals

A pooled "market setup score" across presets, which the scoreboard's cohort rule forbids. Intraday rally-day counting. Using FTD state to block scans.
