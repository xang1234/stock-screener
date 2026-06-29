# Watchlist

description: The user's tracked list of tickers and themes. State, not methodology — read it when other monitor skills need a scope, or when the user asks to view, add, or remove items. Triggers on "my watchlist", "what am I tracking", "add [ticker] to watchlist", "remove [ticker]", "show my themes", "run [other skill] across my watchlist".

This file is your tracking list — tickers and themes you want to revisit. Maintain it however suits you. The other skills in `monitor/` use it as their default scope.

## Workflow

On invocation:
- If the user asked a question scoped to the watchlist (e.g., "anything reporting this week from my watchlist?"), answer directly using the list as the universe.
- If the user asked to add or remove, edit only the relevant line. Preserve every other line, including notes and comments.
- If the user is opening the watchlist with no specific request, show the current contents and ask what they would like to do.

## Important Notes

- **Preserve user structure.** Notes, ordering, custom headers — leave them alone. Never reformat.
- **Empty is fine.** A blank watchlist is a valid state; do not pre-populate with examples beyond the commented-out template.

---

## Tickers

<!-- Add one per line. Format suggestion: TICKER — short note on the thesis or why you're watching -->

<!-- Example:
- NVDA — AI compute platform leader, watching for hyperscaler capex digestion
- VRT — Power infrastructure beneficiary, more upside vs. NVDA at current valuations
- MU — HBM cycle exposure, contrarian on memory pricing
-->

## Themes

<!-- Macro or sector theses you're tracking. Format suggestion: theme — what would make you act -->

<!-- Example:
- AI infrastructure capex — watching for the first sign of slowdown in hyperscaler guidance
- Reshoring — tracking actual trade-flow shifts vs. narrative
- Defense modernization — sustained federal contract awards in specific capabilities
-->
