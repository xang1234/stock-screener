# Scoreboard Ledger

The `scoreboard-ledger-data` release holds one append-only ledger per market: `scoreboard-ledger-<m>-v1.jsonl.gz`, with `scoreboard-ledger-<m>-v1.previous.jsonl.gz` as the prior copy. The `record-ledger` job in `static-site-publish.yml` appends to it after each deploy. [Static Site Guide](../STATIC_SITE.md#forward-return-scoreboard) explains how the scoreboard uses it.

Run every command from `backend/` in a checkout of this repository. They need `gh` signed in with write access to the repository, and nothing else: the script is stdlib only.

## Seed the release (once)

Do this after the ledger code merges to `main`. Until the release exists, `record-ledger` warns and exits 0, and the site publishes no scoreboard. Retained artifacts expire after 30 days, so a day older than that cannot be seeded.

`seed` reads a local capture of the retained artifacts at `.audit/scoreboard-seed/<M>.jsonl.gz`, which is not in the repository. The capture must have covered all 12 markets: its `capture.log` ends with `DONE`. `seed` exits 1 if the directory holds no market files.

1. Build the ledger files. A market with no captured records gets an empty ledger.

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

## Check a ledger

```bash
python3 -m app.scripts.scoreboard_ledger verify --market US
```

This downloads the market's assets and parses them strictly. Add `--dir <path>` to read local files instead. It exits 0 only when canonical is usable. The output has one line per asset and a last line naming what the record job would append to:

```text
US canonical: valid, 21 days, through 2026-11-03
US previous: absent, 0 days, through -
US record job appends to: canonical
```

## Recover a lost day

A failed `record-ledger` run or a capture gap leaves a date with no line. `record-ledger` is the ledger's only other writer, so a `recover` that overlaps it can overwrite the line it just added. First confirm that no publish is running:

```bash
gh run list --workflow static-site-publish.yml --status in_progress
```

Then insert the day from the earliest retained default-branch artifact for that date:

```bash
python3 -m app.scripts.scoreboard_ledger recover --market US --date 2026-11-04
```

The day gets `origin: artifact` (`artifact_pre_stage_fix` before 2026-10-22) and `published_at` set to the artifact's `created_at`, never `live`. `recover` inserts it by date and uploads `.previous` first, as `record` does. A date the ledger already holds uploads nothing. Each candidate artifact is a download of about 30 MB.

After 30 days no artifact remains. The command exits 1 and the date stays a gap, counted in `missing_sessions`.

## Repair a corrupt canonical

`record-ledger` goes red for that market and uploads nothing for it, and other markets continue. Market builds read `.previous` until canonical is repaired, so the scoreboard stays up, one day behind.

1. Confirm the state. The last line reads `nothing, it stops red` when canonical is corrupt.

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

5. `.previous` is one recorded day behind the lost canonical. Recover that day with `recover`.

If `.previous` is also corrupt or absent, the market's scoreboard stays omitted and `record-ledger` stays red. It never starts a fresh ledger, because that would erase history that cannot be rebuilt. Rebuild from a capture with `seed`, or from your own copy, and upload that file.

## What each state does

| State | `record-ledger` | Market build |
|---|---|---|
| Release does not exist | `::warning`, exit 0 | Omits the asset |
| Canonical missing, `.previous` valid | Resumes from `.previous` | Uses `.previous` |
| Canonical corrupt | Red, no upload for that market, other markets continue | Uses `.previous`, else omits the asset |
| Both assets absent in an existing release | Red, never starts fresh | Omits the asset |
| Upload fails | Retries 3 times. If `.previous` fails, skips canonical | `ledger_through` shows the lag |
| Same-date rerun or rewind | No upload | Recomputes, bars capped at the export's `as_of_date` |
| Branch dispatch or PR rehearsal | Job does not run | Downloads only |

A canonical that goes missing mid-upload (`gh release upload --clobber` deletes before it uploads) is the second row, and the next run heals it.
