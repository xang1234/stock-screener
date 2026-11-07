# Market History

Each market has one Release, `market-history-<m>-v1`, that holds one immutable file per published session and an `index.json` rebuilt from the Release listing. The `record-history` job in `static-site-publish.yml` adds a session after each deploy, and the `Market History Backfill` workflow adds sessions from retained artifacts. The [Static Site Guide](../STATIC_SITE.md#market-history-archive) describes the files.

Run every command from `backend/` in a checkout of this repository. The commands need `gh` signed in, and the ones that change a Release need write access. The script is stdlib only.

## Backfill a market

Retained artifacts expire after 30 days, so a backfill cannot reach a session older than that. Dispatch the workflow once per market:

```bash
gh workflow run market-history-backfill.yml -f market=US
gh run list --workflow market-history-backfill.yml --limit 1
```

Each market has its own concurrency group, `market-history-backfill-<M>`, so you can dispatch every market at once. Within one market a newer pending run replaces an older pending one, so dispatch a market again only after its last run finishes.

A backfill reads every retained default-branch `static-market-<M>` artifact, oldest first. Each artifact is a separate download. These rules decide what it adds:

- The first usable artifact per `as_of_date` wins. Its `published_at` is the artifact's `created_at`. Its `origin` is `artifact`, or `artifact_pre_stage_fix` if the artifact was built before 2026-10-21T23:05:12Z.
- A date the Release already lists uploads nothing.
- A bad artifact is skipped with a warning, and the next artifact for that date takes its place.
- A failed artifact download stops the run, so a later artifact cannot take that date. Run the backfill again. The sessions it already added stay.

Market artifacts exist before the publisher deploys them, so a backfill can reach a session before `record-history` does. That session keeps its artifact `origin`, because no job replaces a session asset. A backfilled session holds what the build produced, and the site may never have shown that build.

The workflow does not share the publisher's concurrency group. Session assets are immutable and the index heals itself, so overlapping `record-history` is safe, and a shared group would let a pending backfill replace a pending publish.

To see the result without changing anything, pass a dry-run directory. The command reads the Release and the artifacts, then writes the files to `/tmp/history-check/us/`. `record` takes the same flag.

```bash
python3 -m app.scripts.market_history backfill --market US --dry-run /tmp/history-check
```

## Check a market

List the assets and read the index:

```bash
gh release view market-history-us-v1 --json assets --jq '.assets[] | "\(.name) \(.state)"'
gh release download market-history-us-v1 --pattern index.json --output -
```

Every line of the listing should end in `uploaded`, and `sessions` should hold the date of every session asset. The next run for that market rewrites an index that differs. An asset in any other state needs [a person to delete it](#delete-an-incomplete-asset).

## Delete a bad session

Deleting the asset is the only way to correct a session, because no job overwrites one. The delete cannot be undone, so keep a copy first:

```bash
mkdir /tmp/history-bad
gh release download market-history-us-v1 --pattern us-2026-11-06.json.gz --dir /tmp/history-bad
gh release delete-asset market-history-us-v1 us-2026-11-06.json.gz --yes
```

The index still names that session until the market's next `record-history` or backfill run. The session then stays absent until one of these adds it back:

- A publish whose site still shows that `as_of_date`. The new file has `origin: live` and that deploy's `published_at`.
- A backfill, while a retained artifact holds that date. The new file has an artifact `origin` and the artifact's `created_at`.

If neither can happen, that session is gone for good.

## Delete an incomplete asset

GitHub's [REST documentation](https://docs.github.com/en/rest/releases/assets) says a 502 during an upload "may leave an empty asset with a state of starter" that "can be safely deleted". An asset in any state but `uploaded` never counts as a session and never appears in the index, and no job deletes one. `gh release upload` refuses a name that exists, so uploads for that date stop with this error until you delete the asset:

```text
::error::US: US us-2026-11-06.json.gz is starter: an upload is running or died part way. If none is running, gh release delete-asset market-history-us-v1 us-2026-11-06.json.gz --yes; see docs/runbooks/market-history.md
```

`record-history` records one date per market, so the publishes of later dates still go green. Each of those runs prints the same message, without the leading `US: `, as a `::warning::` until you delete the asset. A backfill whose retained artifacts include that date stops there and adds no later date for that market until you delete the asset. Once no retained artifact holds the date, a backfill only warns about it.

1. Confirm that no publish or backfill is running, because a running upload may show the same state.
2. Delete the asset:

   ```bash
   gh release delete-asset market-history-us-v1 us-2026-11-06.json.gz --yes
   ```

3. Rerun the failed job. It keeps the run's `deployed_at`, so the session records as `live` with the time it first deployed.

   ```bash
   gh run rerun <run-id> --failed
   ```

   The rerun joins the publisher's concurrency group, and each new pending run replaces the one already pending, in both directions:

   - Rerun only when no publish is queued, because the rerun would replace it.
   - If the rerun did replace a publish, run `gh workflow run static-site-publish.yml` after the rerun finishes. The publisher has no schedule, so a replaced publish otherwise waits for the next market build to wake it.
   - A newer pending publish can also replace the pending rerun. Check that the rerun ran, and rerun it again if it was cancelled.

The `market-history-days` artifact lasts 7 days. After that the rerun fails at its download, and a backfill can still add the session as long as a market artifact holds it.

## What each state does

| State | `record-history` | Backfill |
|---|---|---|
| Release does not exist | Creates it as a prerelease with `--latest=false` | Same |
| Another run creates the Release first | Uses it | Same |
| Date already listed as `uploaded` | No session upload | Same |
| Date absent, including one older than the newest | Adds it | Same |
| Index missing, stale, garbled or incomplete | Rewrites it | Same |
| Index matches the listing | No upload | Same |
| Crash after the session upload, before the index upload | The next run rewrites the index | Same |
| Another run uploads the same session first | Keeps that file once a fresh listing shows it `uploaded` | Same |
| Session asset in any state but `uploaded` | Red for that market, no upload, other markets continue | Red, the run stops |
| Another date's asset in any state but `uploaded` | A warning names it and the delete command, then the run records as usual | Same |
| Another run's `--clobber` deletes the index between the listing and the download | Rewrites it | Same |
| Session upload fails any other way | Tries 3 times, then red for that market, index untouched | Tries 3 times, then red, the run stops |
| Zero rows, zero presets, wrong market, schema mismatch, or `as_of_date` past the publish UTC date plus one day | Red for that market, no upload, other markets continue | Skips the artifact with a warning |
| Extraction failed for a market (`<market>.failed`) | Red after finishing the other markets | Not applicable |
| The listing call fails | Red for that market, no upload | Red, no upload |
| Artifact download fails | Not applicable | Red, the run stops |
| No retained artifact is usable | Not applicable | Red |
| A Release reaches 1,000 assets, about four years of sessions | Uploads fail, red for that market. Move the tag to `-v2` in the script | Same |
| Branch dispatch or PR rehearsal | Job does not run, the build still extracts | Job does not run |

`release-asset-cleanup.yml` has no policy for these Releases, so it never prunes them.
