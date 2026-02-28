# XUI Reader Release Notes Template

Use this template for each tagged release.

## Release

- Version: `vX.Y.Z`
- Date: `YYYY-MM-DD`
- Owner: `<maintainer>`

## Summary

One-paragraph summary of the user-visible outcome for this release.

## Highlights

- `<highlight 1>`
- `<highlight 2>`
- `<highlight 3>`

## WS7 Gates

- Packaging/install smoke: `pass | fail`
- Unit/snapshot suite: `pass | fail`
- Diagnostics redaction checks: `pass | fail`

## Milestone Status

- `v2.0` gate: `pass | fail`
- `v2.1` gate: `pass | fail`
- `v2.2` gate: `pass | fail`

## Breaking Changes

- `None` or list of breaking changes with migration notes.

## Known Issues

- `<issue id>: <short description> (mitigation)`

## Rollback Plan

1. Revert to `<previous tag>`
2. Re-run auth/read/doctor smoke commands
3. Confirm diagnostics redaction behavior
