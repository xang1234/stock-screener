---
name: stockscreen-mcp
description: Screen stocks and review watchlists with the stockscreen-mcp tools without misreading the data. Use when asked for stock candidates, market health, leading industry groups, or a check of holdings against Minervini, CANSLIM, VCP and other preset screens, in any of its 12 markets.
---

# Stock screening with stockscreen-mcp

This is end-of-day screening data, not investment advice. Report what the screens show. Do not give price targets, position sizes or buy and sell calls.

## Workflow

Work from the market down, and skip steps the question does not need.

1. **Pick the market.** Call `list_markets` when the user names a country or asks how fresh the data is. A status of late means the market is behind, so say so. Its features show which tools have data for that market.
2. **Read the regime.** Call `market_overview` and report the stance, exposure score, distribution days and breadth before naming any stock. If exposure or breadth comes back absent, say so and check the market's features in `list_markets`. Do not infer a regime from what is left.
3. **Choose screens.** Call `list_screens` for each screen's exact rules. Call `screen_track_record` to judge how much weight a screen deserves, and follow its how_to_read note. Do not rank a row marked thin. A market with no record returns an error, which says nothing about how its screens performed.
4. **Find candidates.** Call `run_screen` with a screen, your own filters, or both. total_matches counts every match, even when the rows stop at the limit.
5. **Confirm leadership.** Call `group_rankings` for strong and improving groups, then check each candidate's `ibd_group_rank`. Rank 1 is strongest. Some markets publish no groups.
6. **Review positions.** Call `audit_stocks` with a watchlist or holdings. It returns the screens each stock passes and its near_misses. Check short_by before calling a miss close.
7. **See what changed.** Call `what_changed` for "what changed since yesterday" or since a date. It compares two archived sessions and returns the stocks that entered and left each screen. For symbols you pass, it adds the screens they moved into and out of, the fields that changed, and under missing the fields null in both sessions, which are unknown rather than unchanged. A growth change carries each side's growth_metric_basis, and the symbol's warnings say when the bases differ, since the two values then compare different periods, or when one side's basis is unknown. Read its warnings before attributing a move to the stock, because a rule change, an RS universe or formula change, the stage fix or a session from a retained build can move a stock that did nothing. A symbol under new_to_publish or dropped_from_publish changed the publish, not the screen. One under entered_with_new_data or left_with_lost_data moved while a field the screen's rules read went from null to a value or back, such as joining the RS ranking, so treat it as a change of data coverage until the values show otherwise. Prices are not split-adjusted across sessions. A market with fewer than two archived sessions returns an error, which says nothing about its stocks.
8. **Look up one name.** Call `get_stock` for every field on one symbol.
9. **Define before quoting.** Call `describe_fields` before quoting or filtering on a field you have not used. Pass group or fields to keep the answer short. It also counts this market's rows: rows_with_value, true_rows on flags, and values on text and list labels. The stockscreen://fields resource holds the same definitions without the counts.

## Reading rules

**Units.**

- `volume` is average daily traded value in the listing currency, not shares, so it cannot be compared across currencies. `adv_usd` is the only USD liquidity figure, but it prices the shares at market cap over shares outstanding, so it can differ widely from `volume`, even for a US stock, and it can be 0. Look at both on a name before quoting liquidity. `market_cap_usd` compares size across markets.
- `pct_day`, `pct_week` and `pct_month` are percentile ranks. The percent changes are `price_change_1d`, `perf_week` and `perf_month`.
- `week_52_high_distance` is positive below the high, `week_52_low_distance` is positive above the low, and `se_distance_to_pivot_pct` is positive above the pivot.
- `current_price`, `se_pivot_price` and `vcp_pivot` are in the row's `currency`.
- `beta_adj_rs` is capped at 100 and is not a percentile.

**Growth.** `growth_metric_basis` on the row is the only authority for what `eps_growth_qq` and `sales_growth_qq` compare. quarterly_qoq compares with the prior quarter, and comparable_period_yoy with the same period a year earlier. With unavailable or null the comparison is unknown, so do not name one. `get_stock` returns the basis, and `run_screen` rows show it only if you filter or sort on it. For year over year, use `eps_growth_yy` and `sales_growth_yy`.

**Dates.** Every result has an as_of_date, the session the publish covers. It is end of day and never live. Quote it with the numbers, and do not call a close a live price. A row's last bar can be newer or older than the result's as_of_date. `run_screen` rows lack `opportunity_state`, so call `get_stock` and read its `opportunity_state.as_of_date` before dating one row's price. Markets can have different dates, so do not line them up as one day. `ipo_date` can be a provider floor date or the first stored bar in any market, so do not quote it as a listing date.

**Coverage.** A null is missing data, not a bad score. `rs_rating` needs a row in the market's RS universe with an adjusted close at every anchor back 252 sessions, so younger listings and anything outside that universe are null. The StockUniverse.is_common_stock flag on each listing, filtered by market, sets the universe, and some markets' universes include ETFs, so percentiles there are not stock-only. A rule that requires a value never matches a null, so a stock missing from a screen may only lack the data. When `describe_fields` reports rows_with_value 0, say the market has no values for that field. Do not reason from its absence.

**RS is per market.** `rs_rating`, `rs_rating_1m`, `rs_rating_3m` and `rs_rating_12m` are percentiles from 1 to 99 within the stock's own market, and `ibd_group_rank` ranks within one market. A 90 in one market does not compare with a 90 in another, so never rank across markets with them.

**Outliers.** Summarise with medians, not averages. `eps_growth_qq` and `eps_growth_yy` are uncapped. `se_up_down_volume_ratio_10d` gets huge when 10 sessions had no down day, because its denominator floors at 1e-9. Question an extreme value before repeating it.

**Labels.** `rating`, `minervini_rating`, `canslim_rating`, `volume_breakthrough_rating` and `action_state` are screening output. Strong Buy describes a score and is not a recommendation. `se_pattern_primary` names the best candidate even below the confidence floor, and `se_pattern_confidence` and `se_setup_score` do not read `stage`, so declining stocks can score high. Do not call a pattern name or a score alone a setup. A preset match means only that its rules passed. Before calling anything a setup, check `stage`, `se_in_early_zone`, `se_pivot_date` and `action_state` with `get_stock`, because a pivot can be years old. In `action_state`, exit_risk, event_risk and extended are cautions to report, and data_limited is usually a data gap and not a verdict on the stock. setup_ready depends on `se_setup_ready`, so read values in `describe_fields`: when setup_ready is absent, that says nothing about the stocks, so do not conclude there are no setups. `se_setup_ready` requires every detector's checks to pass, so a detector that found nothing makes it false. `correction_survivor` is false when a gate failed and also when evidence is missing. failed_checks in `opportunity_state` names which, so read it with `get_stock` before saying why a row is false.

**Empty screens.** Report zero matches as zero matches, not as proof that no such stocks exist, and do not name a cause. You can report `describe_fields` counts as facts, such as "no row has this label". They show what this publish holds, not why the screen is empty, because another rule may exclude the same rows. Do not call the screen broken or the stocks weak.

**Thresholds.** Use only numbers from the screen rules or tool output. If asked what counts as strong, read the rules from `list_screens` instead of supplying your own cut-off.

## Finishing

Name the market, the as_of_date, the screens you used and any data gaps. Close by saying this is screening data, not investment advice.
