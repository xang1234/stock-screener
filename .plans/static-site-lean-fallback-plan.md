# Static Site Lean Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static market never silently disappears, and a valid earlier artifact publishes when the current export fails.

**Architecture:** Fix the fallback policy in the validator, harden the combiner's shared advertised-asset validator, have the combiner record `unavailable_markets` and a per-market `publication` block (lag/state computed in the export service with the market calendar), show unavailable markets as disabled in the static selector, and replace the downloader's workflow-run scan with a by-name artifact lookup that only trusts this repository's default-branch runs.

**Tech Stack:** Python 3.11, pytest; React 18, MUI, Vitest + Testing Library; GitHub CLI (`gh api`).

**Spec:** `.plans/static-site-lean-fallback.md`

## Global Constraints

- Artifact retention is unchanged (repository default, 30 days). Do not add `retention-days`.
- No Release-backed snapshots, catalog ref, CAS, cleanup.
- `supported_markets` and `markets` in the root manifest keep meaning "served"; new data goes in `unavailable_markets` and `markets[M].publication`.
- US stays required (`REQUIRED_STATIC_MARKETS`).
- `app.scripts.download_static_market_fallbacks` must not import `app.database` (existing test).
- Worktree: `/Users/admin/StockScreenClaude/.claude/worktrees/issue-498-market-catalog`. Run backend tests from `<wt>/backend` with `/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest`. Call git as `/usr/bin/git -C <wt> …`, one command per Bash call.
- Frontend tests need Node 22: `export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && nvm use 22`.
- Conventional Commits; end commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. An artifact whose `workflow_run` lacks `repository_id`/`head_repository_id` (older API shapes): must be skipped, not trusted. Pinned in Task 5.
2. A chart index whose `symbols[*].path` uses `..` to escape the market root: must be rejected. Pinned in Task 2.
3. A market stored in `localStorage` / `?market=IN` that is now unavailable: must fall back to the default market, not render a broken page. Pinned in Task 4.
4. Market calendar raising for a market (coverage expired): `publication.state` must be `unknown`, combine must not fail. Pinned in Task 3.
5. Newest artifact by `created_at` carries an older `as_of_date` (rewound export): the older-created artifact with the newer session must still win. Pinned in Task 5.

---

### Task 1: Publish valid fallbacks after `export_failed`

**Files:**
- Modify: `backend/app/scripts/validate_static_market_artifacts.py:41`
- Test: `backend/tests/unit/test_static_market_artifact_validation.py:219-271`

**Interfaces:** Produces no new names; `validate_market_artifacts` accepts `export_failed` + fallback.

- [ ] **Step 1: Flip the two rejection tests to acceptance**

Replace `test_static_market_validator_rejects_failed_selected_market_fallback` and `test_static_market_validator_rejects_failed_selected_required_market_fallback` with:

```python
def test_static_market_validator_publishes_fallback_after_export_failed(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")
    _write_market_status(
        current_dir,
        "CN",
        has_current_artifact=False,
        status="failed",
        reason="export_failed",
    )

    result = validate_market_artifacts(
        current_dir=current_dir,
        fallback_dir=fallback_dir,
        selected_markets={"CN"},
        expected_markets={"US", "CN"},
    )

    assert result.selected_fallback_markets == {"CN"}
    assert result.selected_fallback_diagnostics == {"CN": "status failed/export_failed"}


def test_static_market_validator_publishes_required_fallback_after_export_failed(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(fallback_dir, "static-market-US", "US")
    _write_market_status(
        current_dir,
        "US",
        has_current_artifact=False,
        status="failed",
        reason="export_failed",
    )

    result = validate_market_artifacts(
        current_dir=current_dir,
        fallback_dir=fallback_dir,
        selected_markets={"US"},
        expected_markets={"US"},
    )

    assert result.selected_fallback_markets == {"US"}


def test_static_market_validator_still_rejects_fallback_when_current_artifact_exists(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")
    _write_market_status(
        current_dir,
        "CN",
        has_current_artifact=True,
        status="published",
        reason=None,
    )

    with pytest.raises(StaticMarketArtifactValidationError, match="CN"):
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"CN"},
            expected_markets={"US", "CN"},
        )
```

(Check `_write_market_status` accepts `reason=None`; if it requires a string, pass the default it uses for published statuses.)

- [ ] **Step 2: Run, expect the first two to FAIL**

Run: `…/python -m pytest tests/unit/test_static_market_artifact_validation.py -q`
Expected: the two `publishes_*` tests FAIL with `Refusing to publish fallback static market artifacts`.

- [ ] **Step 3: Implement**

```python
# A current-run export failure describes the new candidate, not the history:
# an independently valid earlier artifact stays publishable.
_ALLOWED_FALLBACK_REASONS = frozenset(
    {"not_trading_day", "no_current_artifact", "export_failed"}
)
```

- [ ] **Step 4: Run the file, expect PASS**

- [ ] **Step 5: Commit** — `fix(static): publish valid fallbacks after export_failed (#498)`

---

### Task 2: Validate every advertised path before combine

**Files:**
- Modify: `backend/app/services/static_artifact_combiner.py` (`_validate_advertised_assets`, new `_validate_advertised_paths`)
- Test: `backend/tests/unit/services/test_static_artifact_combiner.py`

**Interfaces:**
- Produces: `StaticArtifactCombiner._validate_advertised_paths(*, market: str, source_label: str, entry: dict, market_dir: Path) -> None`, raising `StaticArtifactFormulaError`. Called first inside `_validate_advertised_assets`, so `download_static_market_fallbacks.downloaded_market_has_advertised_assets` and `_discover` both get it.

- [ ] **Step 1: Write failing tests**

```python
def _rewrite_entry(root: Path, market: str, **changes) -> Path:
    path = root / f"static-market-{market}" / STATIC_MARKET_METADATA_FILENAME
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["entry"].update(changes)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path.parent


@pytest.mark.parametrize(
    "pages, files, message",
    [
        ({"home": {"path": "markets/us/home.json"}}, {}, "home.json"),
        ({"home": {"path": "markets/us/home.json"}}, {"home.json": "{not json"}, "home.json"),
        ({"home": {"path": "markets/us/../../escape.json"}}, {}, "escapes"),
    ],
)
def test_advertised_page_paths_must_exist_inside_root_and_parse(
    tmp_path: Path, pages, files, message
) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = _rewrite_entry(tmp_path, "US", pages=pages)
    for name, text in files.items():
        (market_dir / name).write_text(text, encoding="utf-8")
    entry = json.loads((market_dir / STATIC_MARKET_METADATA_FILENAME).read_text())["entry"]

    with pytest.raises(StaticArtifactFormulaError, match=message):
        StaticArtifactCombiner._validate_advertised_assets(
            market="US", source_label="fallback", entry=entry, market_dir=market_dir
        )


def test_chart_index_symbol_paths_must_stay_inside_root(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "index.json").write_text(
        json.dumps({"symbols": [{"symbol": "X", "path": "markets/us/../../x.json"}]}),
        encoding="utf-8",
    )
    _rewrite_entry(
        tmp_path, "US", assets={"charts": {"path": "markets/us/charts/index.json"}}
    )
    entry = json.loads((market_dir / STATIC_MARKET_METADATA_FILENAME).read_text())["entry"]

    with pytest.raises(StaticArtifactFormulaError, match="escapes"):
        StaticArtifactCombiner._validate_advertised_assets(
            market="US", source_label="fallback", entry=entry, market_dir=market_dir
        )


def test_chart_index_symbol_payload_must_exist(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "index.json").write_text(
        json.dumps({"symbols": [{"symbol": "X", "path": "markets/us/charts/X.json"}]}),
        encoding="utf-8",
    )
    _rewrite_entry(
        tmp_path, "US", assets={"charts": {"path": "markets/us/charts/index.json"}}
    )
    entry = json.loads((market_dir / STATIC_MARKET_METADATA_FILENAME).read_text())["entry"]

    with pytest.raises(StaticArtifactFormulaError, match="X.json"):
        StaticArtifactCombiner._validate_advertised_assets(
            market="US", source_label="fallback", entry=entry, market_dir=market_dir
        )
```

- [ ] **Step 2: Run, expect FAIL** (no exception raised).

- [ ] **Step 3: Implement**

```python
    @staticmethod
    def _resolve_advertised_path(
        *, market: str, source_label: str, market_dir: Path, advertised: object
    ) -> Path:
        text = str(advertised or "").strip()
        relative = Path(text)
        try:
            relative = relative.relative_to(Path("markets") / market.lower())
        except ValueError:
            pass  # older artifacts advertise paths relative to the market root
        root = market_dir.resolve()
        resolved = (root / relative).resolve()
        if not text or not resolved.is_relative_to(root):
            raise StaticArtifactFormulaError(
                f"{market} {source_label} advertised path escapes its artifact: {text!r}"
            )
        if not resolved.is_file():
            raise StaticArtifactFormulaError(
                f"{market} {source_label} advertised file is absent: {text!r}"
            )
        return resolved

    @classmethod
    def _validate_advertised_paths(
        cls, *, market: str, source_label: str, entry: dict, market_dir: Path
    ) -> None:
        descriptors = [
            *(entry.get("pages") or {}).values(),
            *(entry.get("assets") or {}).values(),
        ]
        for descriptor in descriptors:
            if not isinstance(descriptor, dict) or "path" not in descriptor:
                continue
            path = cls._resolve_advertised_path(
                market=market, source_label=source_label,
                market_dir=market_dir, advertised=descriptor["path"],
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise StaticArtifactFormulaError(
                    f"{market} {source_label} advertised file {descriptor['path']!r} "
                    f"does not parse: {exc}"
                ) from exc
            if descriptor is (entry.get("assets") or {}).get("charts"):
                # ponytail: chart payloads are checked for presence only; there
                # can be hundreds and the browser parses them lazily.
                for symbol in payload.get("symbols") or []:
                    cls._resolve_advertised_path(
                        market=market, source_label=source_label,
                        market_dir=market_dir, advertised=(symbol or {}).get("path"),
                    )
```

Call `cls._validate_advertised_paths(...)` as the first line of `_validate_advertised_assets` (change it from `@staticmethod` to `@classmethod`; callers use `StaticArtifactCombiner._validate_advertised_assets(...)`, unaffected).

- [ ] **Step 4: Run combiner, validator and workflow tests; fix fixtures that advertise files they never wrote**

Run: `…/python -m pytest tests/unit/services/test_static_artifact_combiner.py tests/unit/test_static_site_workflow.py tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py -q`
Expected: new tests PASS. Fixture failures with "advertised file is absent" mean the fixture advertises a path it never wrote: write the file in the fixture (do not loosen the validator).

- [ ] **Step 5: Commit** — `fix(static): reject artifacts whose advertised pages or charts are missing (#498)`

---

### Task 3: `unavailable_markets` and per-market `publication`

**Files:**
- Modify: `backend/app/services/static_artifact_combiner.py` (`combine`, `_build_manifest`, new module function `annotate_publication_lag`)
- Modify: `backend/app/services/static_site_export_service.py` (`combine_market_artifacts`)
- Test: `backend/tests/unit/services/test_static_artifact_combiner.py`

**Interfaces:**
- Produces: root manifest `unavailable_markets: list[str]`; `markets[M].publication = {"source": "current"|"fallback", "session_date": str, "session_lag": int|None, "state": "current"|"stale"|"unknown"}`.
- Produces: `annotate_publication_lag(manifest: dict, calendar) -> None` where `calendar` has `last_completed_trading_day(market) -> date` and `trading_days(market, start, end) -> list[date]`.
- Produces: `StaticSiteExportService.combine_market_artifacts(..., calendar=None)`; `None` → `MarketCalendarService()`.

- [ ] **Step 1: Write failing tests**

```python
from datetime import date

from app.services.static_artifact_combiner import annotate_publication_lag


def test_combined_manifest_lists_unavailable_markets_and_sources(tmp_path: Path) -> None:
    current = write_market_artifact(tmp_path / "current", market="US", formula=BALANCED_RS_FORMULA_VERSION)
    fallback = write_market_artifact(tmp_path / "fallback", market="HK", formula=BALANCED_RS_FORMULA_VERSION)
    output = tmp_path / "out"
    (output / "markets" / "in").mkdir(parents=True)  # stale tree from an older bundle

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=fallback,
        output_dir=output,
        required_formula_by_market={},
        optional_markets=[m for m in STATIC_SUPPORTED_MARKETS if m != "US"],
        clean=False,
    )

    manifest = result.manifest
    assert manifest["supported_markets"] == ["US", "HK"]
    assert manifest["unavailable_markets"] == [
        m for m in STATIC_SUPPORTED_MARKETS if m not in {"US", "HK"}
    ]
    assert manifest["markets"]["US"]["publication"] == {"source": "current", "session_date": "2026-04-10"}
    assert manifest["markets"]["HK"]["publication"]["source"] == "fallback"
    assert not (output / "markets" / "in").exists()


class _Calendar:
    def __init__(self, last, sessions, broken=()):
        self.last, self.sessions, self.broken = last, sessions, set(broken)

    def last_completed_trading_day(self, market):
        if market in self.broken:
            raise RuntimeError("calendar coverage expired")
        return self.last

    def trading_days(self, market, start, end):
        return [d for d in self.sessions if start <= d <= end]


def test_publication_lag_marks_current_stale_and_unknown() -> None:
    sessions = [date(2026, 4, 9), date(2026, 4, 10), date(2026, 4, 13)]
    manifest = {
        "markets": {
            "US": {"publication": {"source": "current", "session_date": "2026-04-13"}},
            "HK": {"publication": {"source": "fallback", "session_date": "2026-04-09"}},
            "JP": {"publication": {"source": "fallback", "session_date": "2026-04-10"}},
        }
    }

    annotate_publication_lag(manifest, _Calendar(date(2026, 4, 13), sessions, broken={"JP"}))

    assert manifest["markets"]["US"]["publication"] == {
        "source": "current", "session_date": "2026-04-13", "session_lag": 0, "state": "current"
    }
    assert manifest["markets"]["HK"]["publication"]["session_lag"] == 2
    assert manifest["markets"]["HK"]["publication"]["state"] == "stale"
    assert manifest["markets"]["JP"]["publication"]["session_lag"] is None
    assert manifest["markets"]["JP"]["publication"]["state"] == "unknown"
```

- [ ] **Step 2: Run, expect FAIL** (`KeyError: 'unavailable_markets'`, ImportError for `annotate_publication_lag`).

- [ ] **Step 3: Implement combiner side**

In `combine`, replace `entries[market] = artifact["entry"]` with:

```python
            entries[market] = artifact["entry"]
            entries[market]["publication"] = {
                "source": artifact["source_label"],
                "session_date": artifact["entry"].get("as_of_date"),
            }
```

In `_build_manifest`, add to the returned dict:

```python
            "unavailable_markets": [
                market for market in self._supported_markets
                if market not in market_entries
            ],
```

Module-level function (bottom of the combiner module):

```python
def annotate_publication_lag(manifest: dict[str, Any], calendar: Any) -> None:
    """Add session lag/state to each served market's publication block."""
    from app.services.market_session_lag import market_session_lag

    for market, entry in (manifest.get("markets") or {}).items():
        publication = entry.setdefault("publication", {})
        lag = None
        try:
            session = date.fromisoformat(str(publication.get("session_date"))[:10])
            lag = market_session_lag(
                calendar,
                market=market,
                start_date=session,
                end_date=calendar.last_completed_trading_day(market),
            )
        except Exception:  # noqa: BLE001 - unknown freshness must not block publication
            lag = None
        publication["session_lag"] = None if lag is None else max(0, lag)
        publication["state"] = (
            "unknown" if lag is None else "current" if lag <= 0 else "stale"
        )
```

(`max(0, lag)`: a session newer than the last completed day, e.g. a mid-session partial run, counts as current.)

- [ ] **Step 4: Implement service side**

`combine_market_artifacts` gains `calendar: Any | None = None`; right after `manifest = combined.manifest`:

```python
        if calendar is None:
            from app.services.market_calendar_service import MarketCalendarService

            calendar = MarketCalendarService()
        annotate_publication_lag(manifest, calendar)
```

The final `cls._write_json(... "manifest.json", manifest)` already rewrites the manifest after options/COT, so the annotated block lands on disk.

- [ ] **Step 5: Run combiner + export-service + script tests, expect PASS**

Run: `…/python -m pytest tests/unit/services/test_static_artifact_combiner.py tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py tests/unit/test_report_static_market_freshness.py -q`
Tests that assert the exact root-manifest dict need `unavailable_markets` added.

- [ ] **Step 6: Commit** — `feat(static): keep unavailable markets visible and record publication state (#498)`

---

### Task 4: Show unavailable markets as disabled in the static selector

**Files:**
- Modify: `frontend/src/static/dataClient.js` (new `getStaticUnavailableMarkets`)
- Modify: `frontend/src/static/StaticLayout.jsx:79-87`
- Create: `frontend/src/static/StaticLayout.test.jsx`

**Interfaces:** Consumes root manifest `unavailable_markets` from Task 3. Produces `getStaticUnavailableMarkets(manifest) -> string[]`.

- [ ] **Step 1: Write failing test**

```jsx
import { fireEvent, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { renderWithProviders } from '../test/renderWithProviders';
import { StaticMarketProvider } from './StaticMarketContext';
import StaticLayout from './StaticLayout';

const manifest = {
  default_market: 'US',
  supported_markets: ['US', 'HK'],
  unavailable_markets: ['IN'],
  markets: {
    US: { display_name: 'United States', features: {}, pages: {}, assets: {} },
    HK: { display_name: 'Hong Kong', features: {}, pages: {}, assets: {} },
  },
};

vi.mock('./dataClient', async (importOriginal) => ({
  ...(await importOriginal()),
  useStaticManifest: () => ({ data: manifest }),
}));

const renderLayout = (initialEntry = '/') => renderWithProviders(
  <MemoryRouter initialEntries={[initialEntry]}>
    <StaticMarketProvider supportedMarkets={manifest.supported_markets} defaultMarket="US">
      <StaticLayout><div /></StaticLayout>
    </StaticMarketProvider>
  </MemoryRouter>,
);

describe('StaticLayout market selector', () => {
  it('lists unavailable markets as disabled options', () => {
    renderLayout();
    fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Static market selector' }));
    const option = within(screen.getByRole('listbox')).getByRole('option', { name: /IN — unavailable/ });
    expect(option).toHaveAttribute('aria-disabled', 'true');
  });

  it('falls back to the default market when the URL names an unavailable market', () => {
    renderLayout('/?market=IN');
    expect(screen.getByRole('combobox', { name: 'Static market selector' })).toHaveTextContent('United States');
  });
});
```

(If `StaticLayout` needs `ColorModeContext` beyond its default value, wrap in `<ColorModeContext.Provider value={{ toggleColorMode() {} }}>`.)

- [ ] **Step 2: Run, expect the first test to FAIL**

Run: `cd frontend && npx vitest run src/static/StaticLayout.test.jsx`

- [ ] **Step 3: Implement**

`dataClient.js`:

```js
export const getStaticUnavailableMarkets = (manifest) => (
  Array.isArray(manifest?.unavailable_markets) ? manifest.unavailable_markets : []
);
```

`StaticLayout.jsx` — import it, then after the `supportedMarkets.map(...)` inside `<Select>`:

```jsx
                {getStaticUnavailableMarkets(manifestQuery.data).map((market) => {
                  const flag = marketFlag(market);
                  return (
                    <MenuItem key={market} value={market} disabled>
                      {flag ? `${flag}  ${market} — unavailable` : `${market} — unavailable`}
                    </MenuItem>
                  );
                })}
```

- [ ] **Step 4: Run `npm run test:run -- src/static` and `npm run lint`, expect PASS**

- [ ] **Step 5: Commit** — `feat(frontend): show unavailable static markets as disabled (#498)`

---

### Task 5: Find fallbacks by artifact name instead of scanning runs

**Files:**
- Modify: `backend/app/scripts/download_static_market_fallbacks.py` (remove `extract_runs`, `_workflow_run_upper_bound_date`, the runs API call and per-run artifact listing; add `_ArtifactRun`, `list_artifact_runs`)
- Modify: `backend/tests/unit/test_static_site_workflow.py` (fake `gh` serves `actions/artifacts?name=`), `backend/tests/unit/test_static_options_artifact_selector.py`, `backend/tests/unit/test_static_cot_pipeline.py` where they fake the runs API

**Interfaces:**
- Produces: `list_artifact_runs(*, repo: str, artifact_name: str, branch_name: str, current_run_id: int) -> list[_ArtifactRun]`, newest `created_on` first; `_ArtifactRun(run_id: int, created_on: date | None)`.
- `download_fallback_artifacts(...)` signature unchanged.

- [ ] **Step 1: Write failing unit tests for the lookup**

```python
def _artifact(run_id, created, *, branch="main", repo_id=1, head_repo_id=1, expired=False, name="static-market-US"):
    return {
        "name": name,
        "expired": expired,
        "created_at": created,
        "workflow_run": {
            "id": run_id, "head_branch": branch,
            "repository_id": repo_id, "head_repository_id": head_repo_id,
        },
    }


def test_list_artifact_runs_keeps_only_trusted_default_branch_runs(monkeypatch) -> None:
    calls = []

    def fake_gh_json(args):
        calls.append(args)
        return [{"artifacts": [
            _artifact(500, "2026-09-01T00:00:00Z"),
            _artifact(999, "2026-09-05T00:00:00Z"),                     # current run
            _artifact(501, "2026-09-04T00:00:00Z", expired=True),
            _artifact(502, "2026-09-04T00:00:00Z", branch="feature"),
            _artifact(503, "2026-09-04T00:00:00Z", head_repo_id=77),    # fork PR named main
            {**_artifact(504, "2026-09-04T00:00:00Z"), "workflow_run": {"id": 504, "head_branch": "main"}},
            _artifact(505, "2026-09-03T00:00:00Z", name="static-market-USX"),
            _artifact(506, "2026-09-03T00:00:00Z"),
        ]}]

    monkeypatch.setattr(fallback_script, "gh_json", fake_gh_json)

    runs = fallback_script.list_artifact_runs(
        repo="xang1234/stock-screener", artifact_name="static-market-US",
        branch_name="main", current_run_id=999,
    )

    assert [run.run_id for run in runs] == [506, 500]
    assert "actions/artifacts?name=static-market-US" in calls[0][-1]


def test_market_fallback_prefers_newer_session_over_newer_upload(tmp_path, monkeypatch) -> None:
    listings = {"static-market-US": [
        _artifact(600, "2026-09-06T00:00:00Z"),  # rewound export: older session
        _artifact(500, "2026-09-05T00:00:00Z"),
        _artifact(400, "2026-08-20T00:00:00Z"),  # cannot beat 2026-09-05
    ]}
    sessions = {600: date(2026, 9, 4), 500: date(2026, 9, 5), 400: date(2026, 8, 19)}
    downloaded = []

    def fake_gh_json(args):
        name = args[-1].split("name=", 1)[1].split("&", 1)[0]
        return [{"artifacts": listings.get(name, [])}]

    def fake_download(*, run_id, parent_dir, **_kwargs):
        downloaded.append(run_id)
        wrapper = parent_dir / f"c-{run_id}"
        wrapper.mkdir(parents=True)
        return fallback_script._DownloadedCandidate(wrapper, wrapper, sessions[run_id])

    installed = {}
    monkeypatch.setattr(fallback_script, "gh_json", fake_gh_json)
    monkeypatch.setattr(fallback_script, "_download_candidate", fake_download)
    monkeypatch.setattr(
        fallback_script, "_install_market_candidate",
        lambda *, target_dir, candidate_dir: installed.__setitem__(target_dir.name, candidate_dir.name),
    )

    markets = fallback_script.download_fallback_artifacts(
        repo="xang1234/stock-screener", current_run_id=999, branch_name="main",
        current_dir=tmp_path / "current", fallback_dir=tmp_path / "fallback",
    )

    assert markets == {"US"}
    assert downloaded == [600, 500]
    assert installed["static-market-US"] == "c-500"
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: list_artifact_runs`).

- [ ] **Step 3: Implement the lookup**

```python
@dataclass(frozen=True)
class _ArtifactRun:
    run_id: int
    created_on: date | None


def list_artifact_runs(
    *, repo: str, artifact_name: str, branch_name: str, current_run_id: int
) -> list[_ArtifactRun]:
    query = urlencode({"name": artifact_name, "per_page": "100"})
    try:
        artifacts = extract_artifacts(
            gh_json(["api", "--paginate", "--slurp", f"repos/{repo}/actions/artifacts?{query}"])
        )
    except subprocess.CalledProcessError as exc:
        warn(f"Artifact lookup for {artifact_name} failed with exit {exc.returncode}.{command_error_detail(exc)}")
        return []
    except json.JSONDecodeError as exc:
        warn(f"Artifact lookup for {artifact_name} was not valid JSON ({exc}).")
        return []
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        warn(f"Artifact lookup for {artifact_name} was invalid: {exc}")
        return []

    runs = []
    for artifact in artifacts:
        run = artifact.get("workflow_run")
        if artifact.get("name") != artifact_name or artifact.get("expired") or not isinstance(run, dict):
            continue
        run_id = run.get("id")
        repository_id = run.get("repository_id")
        # Trust boundary: only this repository's own default-branch runs. A fork
        # PR's head branch can also be called "main"; its head repository differs.
        if (
            not isinstance(run_id, int)
            or run_id == current_run_id
            or run.get("head_branch") != branch_name
            or repository_id is None
            or run.get("head_repository_id") != repository_id
        ):
            continue
        runs.append(_ArtifactRun(run_id, _coerce_manifest_date(artifact.get("created_at"))))
    runs.sort(key=lambda run: run.created_on or date.min, reverse=True)
    return runs
```

- [ ] **Step 4: Rewire `download_fallback_artifacts`**

Delete the runs query and the `for run in runs:` body. Keep the setup (formula requirements, current markets, `global_directories`, `global_fallback_dates`). Then:

```python
    for key, (_current_global_dir, fallback_global_dir) in global_directories.items():
        if fallback_global_dir is None:
            continue
        spec = GLOBAL_STATIC_ARTIFACTS[key]
        for ref in list_artifact_runs(
            repo=repo, artifact_name=spec.artifact_name,
            branch_name=branch_name, current_run_id=current_run_id,
        ):
            if _run_cannot_beat_incumbent(
                run_upper_bound=ref.created_on, incumbent_date=global_fallback_dates[key]
            ):
                break
            # existing candidate download / _candidate_is_newer / install block,
            # with run_id=ref.run_id

    for market in sorted(SUPPORTED_MARKET_CODES):
        artifact_name = f"static-market-{market}"
        for ref in list_artifact_runs(
            repo=repo, artifact_name=artifact_name,
            branch_name=branch_name, current_run_id=current_run_id,
        ):
            if market in fallback_markets and _run_cannot_beat_incumbent(
                run_upper_bound=ref.created_on,
                incumbent_date=fallback_dates_by_market.get(market),
            ):
                break
            # existing market candidate download / newer check / install block,
            # with run_id=ref.run_id
```

Import `SUPPORTED_MARKET_CODES` from `app.domain.markets` (does not import `app.database`; the existing import-isolation test guards this). Remove `extract_runs`, `_workflow_run_upper_bound_date`, and the `--branch` default stays `main`.

- [ ] **Step 5: Migrate the fake-`gh` tests**

In every fake `gh` script in `test_static_site_workflow.py`, replace the `actions/workflows/static-site.yml/runs` and `actions/runs/<id>/artifacts` branches with one branch:

```python
        if args[:3] == ["api", "--paginate", "--slurp"] and "actions/artifacts?name=" in args[3]:
            name = args[3].split("name=", 1)[1].split("&", 1)[0]
            print(json.dumps([{"artifacts": [
                {"name": name, "expired": False, "created_at": created,
                 "workflow_run": {"id": int(run_id), "head_branch": "main",
                                  "repository_id": 1, "head_repository_id": 1}}
                for run_id, created, names in RUNS if name in names
            ]}]))
```

with a per-test `RUNS = [("333", "2026-08-05T00:00:00Z", {"static-market-US", ...}), ...]` table carrying the same run → artifact mapping the old fake encoded. Keep each test's assertions; update expected download order only where it now follows market order (`sorted(SUPPORTED_MARKET_CODES)`) instead of run order, and say so in the test. Same treatment for `test_options_fallback_*` (monkeypatched `gh_json`), `test_static_options_artifact_selector.py` and `test_static_cot_pipeline.py`.

- [ ] **Step 6: Run the downloader-related suites, expect PASS**

Run: `…/python -m pytest tests/unit/test_static_site_workflow.py tests/unit/test_static_options_artifact_selector.py tests/unit/test_static_cot_pipeline.py -q`

- [ ] **Step 7: Commit** — `perf(static): look up fallback artifacts by name, trusting only this repo's main runs (#498)`

---

### Task 6: Full verification

- [ ] Run: `…/python -m pytest tests/unit/test_static_market_artifact_validation.py tests/unit/services/test_static_artifact_combiner.py tests/unit/test_static_site_workflow.py tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py tests/unit/test_static_options_artifact_selector.py tests/unit/test_static_cot_pipeline.py tests/unit/test_report_static_market_freshness.py -q`
- [ ] Run: `cd frontend && npm run test:run -- src/static && npm run lint`
- [ ] Mutation check: revert Task 1's line and confirm the `publishes_*` tests fail; drop the `head_repository_id` clause and confirm the fork test fails; restore both.
- [ ] Strict report-only reviewer agent on `git diff origin/main` (per project practice), fix confirmed findings test-first, then push and open the PR referencing #498.
