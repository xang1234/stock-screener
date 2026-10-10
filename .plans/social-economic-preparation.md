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

As built:
- Authority checks use `theme_development_preparation.economic_authority`, a
  registered gate predicate. `LiveAcceptedBasketReader` refuses under economic
  authority whatever `authority_source` a caller passes.
- The 14-day window for the processor's rule ends at the newest Social packet
  the association has seen. The rule counts the association's Social works,
  only claims the catalog places under its theme, and needs 2+ independent
  authors for the verified company, like `_automatic_accepts`.
  `qualifying_social_evidence` is shared by both paths.
- The catalog drops a normalized key two themes claim: a shared display name
  drops both themes, and an alias shared across themes is dropped. A display
  name wins over another theme's alias. The publication fingerprint covers the
  whole mapping.
- Economic baskets count only `stock` securities as company stocks, like the
  legacy reader.
- A run prepared before a cutover and published after it is refused
  (`publication_basket_changed`), and the next run prepares under economic
  authority.
- Both Social tasks stay allowlisted under `_SOCIAL_BRIDGE` for
  `SocialThemeAssociation` only, because the adapter's `revise` keeps bridged
  legacy mirrors in step.
