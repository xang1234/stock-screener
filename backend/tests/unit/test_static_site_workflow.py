from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _build_market_job() -> str:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    return content.split("  build-market:\n", 1)[1].split(
        "\n  combine-and-build:",
        1,
    )[0]


def _combine_and_build_job() -> str:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    return content.split("  combine-and-build:\n", 1)[1].split(
        "\n  deploy:",
        1,
    )[0]


def test_static_site_market_build_failures_are_not_marked_continue_on_error() -> None:
    build_market_job = _build_market_job()

    assert "continue-on-error: true" not in build_market_job


def test_static_site_daily_price_seed_allows_stale_bootstrap() -> None:
    build_market_job = _build_market_job()
    seed_step = build_market_job.split("      - name: Seed daily price bundle from GitHub\n", 1)[1].split(
        "\n      - name: Export market static data bundle",
        1,
    )[0]

    assert "--allow-stale" in seed_step


def test_static_site_combine_downloads_current_and_per_market_fallback_artifacts() -> None:
    combine_job = _combine_and_build_job()

    assert "Download per-market fallback artifacts" in combine_job
    assert "Download current market artifacts" in combine_job
    assert "/tmp/static-market-artifacts-current" in combine_job
    assert "/tmp/static-market-artifacts-fallback" in combine_job
    assert "--fallback-artifacts-dir /tmp/static-market-artifacts-fallback" in combine_job
    assert "FALLBACK_MARKETS: US HK IN JP KR TW" in combine_job
    assert "github.ref_name" in combine_job
    assert "--paginate" in combine_job
    assert "runs = extract_runs(pages)" in combine_job
    assert "static-market-{market}" in combine_job
    assert '"gh",' in combine_job
    assert '"run",' in combine_job
    assert '"download",' in combine_job
    assert "actions/runs/{run_id}/artifacts" in combine_job
    assert "artifact.get(\"expired\")" in combine_job
    assert "run.get(\"conclusion\") == \"success\"" not in combine_job
    assert "subprocess.CalledProcessError" in combine_job
    assert "::warning::Unable to download fallback market artifact" in combine_job
    assert "isinstance(payload, dict)" in combine_job
    assert "isinstance(page, dict)" in combine_job
    assert "Unexpected GitHub API response shape" in combine_job
