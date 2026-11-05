# Scoreboard Ledger

The `scoreboard-ledger-data` release holds one append-only ledger per market: `scoreboard-ledger-<m>-v1.jsonl.gz`, with `scoreboard-ledger-<m>-v1.previous.jsonl.gz` as the prior copy. The `record-ledger` job in `static-site-publish.yml` appends to it after each deploy, and the `Scoreboard Ledger Recover` workflow inserts a lost day. They share the publisher's concurrency group, so they cannot overlap. [Static Site Guide](../STATIC_SITE.md#forward-return-scoreboard) explains how the scoreboard uses it.

Run every command from `backend/` in a checkout of this repository. They need `gh` signed in with write access to the repository, and nothing else: the script is stdlib only.

## Seed the release (once)

Do this after the ledger code merges to `main`. Until the release exists, `record-ledger` warns and exits 0, and the site publishes no scoreboard. Retained artifacts expire after 30 days, so a day older than that cannot be seeded.

`seed` reads a local capture of the retained artifacts at `.audit/scoreboard-seed/<M>.jsonl.gz`, which is not in the repository. A complete capture covers all 12 markets, and its `capture.log` ends with `DONE`. `seed` exits 1 if any market has no input file, unless you pass `--allow-missing`, which starts that market's ledger empty and names it in a warning. It exits 1 with or without the flag if the directory holds no market files.

1. Build the ledger files. A market whose input file holds no records gets an empty ledger.

   ```bash
   python3 -m app.scripts.scoreboard_ledger seed --input ../.audit/scoreboard-seed --output /tmp/ledger
   ```

2. Parse a file back. Run this for each market you want to check. It exits 0 when the record job could append to the file.

   ```bash
   python3 -m app.scripts.scoreboard_ledger verify --market US --dir /tmp/ledger
   ```

3. Create the release and upload the files.

   ```bash
   gh release create scoreboard-ledger-data --title "Scoreboard Ledger" --notes "Append-only screen matches per market. See docs/runbooks/scoreboard-ledger.md."
   gh release upload scoreboard-ledger-data /tmp/ledger/*.jsonl.gz
   ```

The next `static-site.yml` market build then downloads the ledger and publishes the asset. A capture gap stays a gap, counted in `missing_sessions`, unless a retained artifact can still [recover it](#recover-a-lost-day).

### Re-seeding after live days exist

Do not upload a plain `seed` output once `record-ledger` has added a day. It would replace `live` lines with `artifact` lines or drop them, and `record-ledger` would then go red because canonical no longer holds what `.previous` holds. Merge instead, so every existing line stays byte for byte and the seed only fills gaps:

```bash
mkdir /tmp/ledger-current
gh release download scoreboard-ledger-data --pattern 'scoreboard-ledger-*-v1.jsonl.gz' --dir /tmp/ledger-current
python3 -m app.scripts.scoreboard_ledger seed --input ../.audit/scoreboard-seed --output /tmp/ledger-merged --merge-with /tmp/ledger-current
```

If the capture lacks a market, add `--allow-missing`, and that market's current ledger stays as it is. Check each market with `verify --dir`, confirm that no publish is running, then upload the files with `--clobber`. For one or two lost days, [recover](#recover-a-lost-day) is simpler. `--merge-with` exits 1 if any market's current ledger is absent or corrupt.

## Check a ledger

```bash
python3 -m app.scripts.scoreboard_ledger verify --market US
```

This downloads the market's assets and parses them strictly. Add `--dir <path>` to read local files instead. It exits 0 only when canonical is usable and holds every line of a valid `.previous`. The output has one line per asset and a last line naming what the record job would append to:

```text
US canonical: valid, 21 days, through 2026-11-03
US previous: absent, 0 days, through -
US record job appends to: canonical
```

## Recover a lost day

A failed `record-ledger` run or a capture gap leaves a date with no line. The workflow inserts it from the earliest retained default-branch artifact for that date, inside the publisher's concurrency group:

```bash
gh workflow run scoreboard-ledger-recover.yml -f market=US -f date=2026-11-04
gh run list --workflow scoreboard-ledger-recover.yml --limit 1
```

The recover and the publisher share one concurrency group, and each new pending run replaces the one already pending, in both directions:

- Dispatch a recover only when no publish is queued, because the recover would replace it.
- Wait for each recover to finish before dispatching the next, because a second recover replaces a first one that is still pending. This includes the "every later day, oldest first" step in [Repair a corrupt canonical](#repair-a-corrupt-canonical).
- If a recover did replace a publish, run `gh workflow run static-site-publish.yml` after the recover finishes. The publisher has no schedule, so a replaced publish otherwise waits for the next market build to wake it.

A newer pending publish can also replace a pending recover. Check that each recover ran, and dispatch it again if it was cancelled.

The day gets `origin: artifact` (`artifact_pre_stage_fix` if the artifact was built before 2026-10-21T23:05:12Z, the merge of #671) and `published_at` set to the artifact's `created_at`, never `live`. `recover` inserts it by date and uploads `.previous` first, as `record` does. A date the ledger already holds uploads nothing. Each candidate artifact is a separate download, about 33 MB for US.

After 30 days no artifact remains. The run fails and the date stays a gap, counted in `missing_sessions`.

To see the result without changing anything, run it locally with a dry-run directory. It downloads what it needs and writes the files it would upload there:

```bash
python3 -m app.scripts.scoreboard_ledger recover --market US --date 2026-11-04 --dry-run /tmp/recover-check
python3 -m app.scripts.scoreboard_ledger verify --market US --dir /tmp/recover-check
```

A local write needs `--i-paused-publishing` instead of `--dry-run`. Pass it only when no publish can start, because `record-ledger` overlapping a local `recover` can overwrite the line it just added.

## Repair a corrupt canonical

`record-ledger` goes red for that market and uploads nothing for it, and other markets continue. Market builds read `.previous` until canonical is repaired, so the scoreboard stays up.

**Repair it the same day.** A corrupt canonical blocks `record`, so the lag grows with every publish. Each day in the lag has to be recovered one by one once canonical is back, and it returns as `artifact`, not `live`, because the live publication time was never recorded.

1. Confirm that no publish is running, then confirm the state. The `appends to` line reads `nothing, it stops red` when canonical is corrupt.

   ```bash
   python3 -m app.scripts.scoreboard_ledger verify --market US
   ```

2. Download `.previous` and copy it over canonical locally.

   ```bash
   mkdir /tmp/ledger-repair
   gh release download scoreboard-ledger-data --pattern scoreboard-ledger-us-v1.previous.jsonl.gz --dir /tmp/ledger-repair
   cp /tmp/ledger-repair/scoreboard-ledger-us-v1.previous.jsonl.gz /tmp/ledger-repair/scoreboard-ledger-us-v1.jsonl.gz
   ```

3. Check the copy. It must exit 0 and name `canonical`.

   ```bash
   python3 -m app.scripts.scoreboard_ledger verify --market US --dir /tmp/ledger-repair
   ```

4. Upload it.

   ```bash
   gh release upload scoreboard-ledger-data /tmp/ledger-repair/scoreboard-ledger-us-v1.jsonl.gz --clobber
   ```

5. `.previous` is one recorded day behind the lost canonical. [Recover](#recover-a-lost-day) that day and every later day in the lag, oldest first, waiting for each run to finish before dispatching the next.

If `.previous` is also corrupt or absent, the market's scoreboard stays omitted and `record-ledger` stays red. It never starts a fresh ledger, because that would erase history that cannot be rebuilt. Rebuild from a capture with `seed`, or from your own copy, and upload that file.

## Canonical lost a line

`record-ledger` compares canonical with `.previous` before rotating. If a valid `.previous` holds a line canonical no longer holds byte for byte, something rewrote history. The job goes red for that market and uploads nothing, because rotating would overwrite the only copy that shows the loss. `verify` reports `canonical lost N line(s)` and exits 1.

Find out what changed before fixing it, because market builds keep reading canonical in this state. If `.previous` is right, follow the [repair steps](#repair-a-corrupt-canonical) from step 2. If the rewrite was deliberate, such as a plain re-seed before you read [the warning](#re-seeding-after-live-days-exist), upload `.previous` again from a copy of the file you now trust so the two agree.

## What each state does

| State | `record-ledger` | Market build |
|---|---|---|
| Release does not exist | `::warning`, exit 0 | Omits the asset |
| Canonical missing, `.previous` valid | Resumes from `.previous` | Uses `.previous` |
| Canonical corrupt | Red, no upload for that market, other markets continue | Uses `.previous`, else omits the asset |
| Canonical lost a line `.previous` holds | Red, no upload for that market, other markets continue | Uses canonical |
| `as_of_date` past the publish UTC date plus one day, or a day with zero presets | Red, no upload for that market, other markets continue | Unaffected |
| Extraction failed for a market (`<market>.failed`) | Red after finishing the other markets | Unaffected |
| Both assets absent in an existing release | Red, never starts fresh | Omits the asset |
| Upload fails | Retries 3 times. If `.previous` fails, skips canonical | `ledger_through` shows the lag |
| Same-date rerun or rewind | No upload | Recomputes, bars capped at the export's `as_of_date` |
| Branch dispatch or PR rehearsal | Job does not run | Downloads only |

A canonical that goes missing mid-upload (`gh release upload --clobber` deletes before it uploads) is the second row, and the next run heals it.
