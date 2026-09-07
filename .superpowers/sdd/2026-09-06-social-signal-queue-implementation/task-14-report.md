# Task 14 report: live-only static isolation

Status: complete inline. No provider, paid-model, production-database, publication, registry, or credential access occurred.

## Result

- Added a fail-closed static publication guard at the JSON writer and completed-bundle boundaries. Generated and copied JSON is recursively checked for live-only keys; copied artifact paths are checked before a combined bundle is returned.
- Preserved the existing static product boundary: the static manifest has no Themes feature/page and writes no `themes/` directory. The orphaned legacy `StaticThemesPage` remains un-routed and contains no Social Pulse or Social request.
- Added artifact-level assertions over a real synthetic static export and a real combined market bundle. Added negative tests proving direct JSON writes and copied market artifacts are rejected before publication when a forbidden key appears.
- Added static frontend graph and runtime-shell checks proving static production modules contain no Social import/route/request string, static navigation has no Social tab, and rendered static Home/Themes surfaces make no Social request or render Social Pulse.
- Existing Task 7B integration coverage remains authoritative for provenance semantics: independent legacy eligibility survives either canonical insertion order; Social-derived `ThemeMention` rows remain excluded from legacy extraction/sentiment; `ThemeConstituent` remains legacy membership even when a Social association is rejected. Because the current static publisher exports no Theme data at all, neither Social-only themes nor accepted/proposed/rejected Social associations can enter a static basket.

## Verification

- Backend RED: focused collection failed because `StaticSocialIsolationError` did not exist.
- Backend GREEN: `127 passed` for `test_static_site_export_service.py` and `test_export_static_site_script.py`.
- Added combined-artifact focus: `3 passed` for writer, normal combine, and malicious copied-artifact cases.
- Frontend GREEN: `18 passed` across `socialIsolation.test.jsx`, `StaticHomePage.test.jsx`, and `StaticThemesPage.test.jsx`.
- Frontend lint with `--quiet`: clean.
- `git diff --check`: clean.

## Design note

The guard inspects keys and publication paths, not arbitrary string values. This prevents feature/schema/route leakage without rejecting ordinary static editorial content merely because its prose mentions social media or a tweet.
