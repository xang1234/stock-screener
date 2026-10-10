# Economic Social baskets and association decisions (#515)

Part of #410. Owner decisions (2026-10-10):

1. **Theme keys → economic themes at publish:** the processing version's
   catalog. `canonical_theme_key` of `EconomicThemeRevision.display_name` and
   of `EconomicThemeAlias` aliases. A key with no match has no theme basket at
   publish; it can become a theme later through classification.
2. **Auto-acceptance:** in the processor. When classified Social assignments
   are projected, an economic association is accepted once it has 2+
   qualifying works (14 days, one per post, URL or copied thesis, verified
   company) for that (economic theme, company). Publish sends memberships as
   `proposed`, except pairs already decided economically, and reads no legacy
   baskets or associations.
3. **Decision API:** new `POST /social-signals/admin/economic-associations/{uuid}/decision`
   taking `target`, `reason`, `expected_revision`, for native and bridged
   associations alike. Under economic authority the legacy-id endpoint refuses,
   so neither the admin list nor the decision lookup reads legacy rows (the
   adapter still reads bridged rows to keep their legacy mirror in step).

Legacy, shadow and dual modes keep the legacy flow unchanged.

## PR A — decisions (decision 3)

- Endpoint + `SocialThemeProjectionService.decide_economic_association`
  (economic authority only; adapter `revise` with `expected_revision`).
- `decide` refuses legacy ids under economic authority
  (`economic_association_decision_required`); `_decide_economic` goes.
- Admin list under economic authority drops the legacy id and version (no
  `SocialThemeAssociation` read); every row is decidable by economic id.
- Frontend: the review panel decides economic rows through the new endpoint.
- Gate: `_SOCIAL_ASSOCIATIONS` entry removed.

## PR B — baskets and auto-acceptance (decisions 1, 2)

- `prepare` / `prepare_application` / `theme_keys` under economic authority
  read the economic catalog, never `ThemeCluster`/`ThemeAlias`/
  `ThemeMention`/`ThemeConstituent`/`SocialThemeAssociation`.
- Baskets come from `EconomicAcceptedBasketReader` per mapped economic theme.
- `admit_economic_evidence` sends `proposed` (or the existing economic
  decision) per membership.
- `project_native_assignments` applies the automatic rule.
- Gate: `_SOCIAL_PREPARATION` entries removed.
