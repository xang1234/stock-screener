from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.stock import StockPrice
from app.services.daily_price_bundle_service import DailyPriceBundleService
from app.services.github_release_sync_service import GitHubReleaseSyncService
from app.services.price_history_coverage import classify_price_history
from app.services.static_price_checkpoint import (
    checkpoint_asset_names,
    checkpoint_fingerprint,
    resume_price_checkpoint,
    write_price_checkpoint,
)

from .daily_price_bundle_test_helpers import (
    make_session,
    price_row,
    stock_row,
)

SEEDED = date(2026, 4, 16)
AS_OF = date(2026, 4, 17)
BUNDLE, MANIFEST = checkpoint_asset_names("US")


class _ReleaseDir:
    """A requests-like session serving one GitHub release from a directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, url, **_kwargs):
        if "/releases/tags/" in url:
            assets = [
                {"name": path.name, "browser_download_url": f"file://{path}"}
                for path in sorted(self.root.iterdir())
            ]
            return SimpleNamespace(status_code=200, json=lambda: {"assets": assets})
        return SimpleNamespace(
            status_code=200,
            content=Path(url.removeprefix("file://")).read_bytes(),
        )


def _runner_db(*, baseline: str = "daily_prices_us:seed"):
    """A fresh runner database after the daily-bundle seed."""
    db = make_session()()
    db.add_all(
        [
            stock_row("AAPL", "US", "NASDAQ", 1000.0),
            stock_row("MSFT", "US", "NASDAQ", 900.0),
            price_row("AAPL", SEEDED, 100.0),
            price_row("MSFT", SEEDED, 200.0),
        ]
    )
    service = DailyPriceBundleService()
    service._upsert_import_state(
        db,
        market="US",
        source_revision=baseline,
        as_of_date=SEEDED.isoformat(),
        symbol_count=2,
        bar_period=service.DAILY_PRICE_BAR_PERIOD,
    )
    return db


def _checkpointed_release(tmp_path: Path) -> Path:
    """First attempt: AAPL reached the as-of session before the deadline."""
    db = _runner_db()
    db.add(price_row("AAPL", AS_OF, 101.0))
    db.commit()
    release = tmp_path / "release"
    write_price_checkpoint(db, market="US", as_of_date=AS_OF, output_dir=release)
    db.close()
    return release


def _resume(db, release: Path, *, as_of_date: date = AS_OF):
    return resume_price_checkpoint(
        db,
        market="US",
        as_of_date=as_of_date,
        sync_service=GitHubReleaseSyncService(session=_ReleaseDir(release)),
    )


def _dates(db, symbol: str) -> set[date]:
    return {
        row_date
        for (row_date,) in db.query(StockPrice.date).filter(StockPrice.symbol == symbol)
    }


def test_checkpoint_writes_only_its_own_assets(tmp_path):
    release = _checkpointed_release(tmp_path)

    assert {path.name for path in release.iterdir()} == {BUNDLE, MANIFEST}
    manifest = json.loads((release / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["kind"] == "price_checkpoint"
    assert manifest["bundle_asset_name"] == BUNDLE
    assert manifest["fingerprint"]["as_of_date"] == AS_OF.isoformat()
    assert manifest["fingerprint"]["baseline_daily_price_revision"] == "daily_prices_us:seed"


def test_resume_imports_finished_symbols_so_only_the_rest_is_refetched(tmp_path):
    release = _checkpointed_release(tmp_path)
    db = _runner_db()

    first = _resume(db, release)
    second = _resume(db, release)

    assert first["status"] == "imported"
    assert second["status"] == "imported"
    assert _dates(db, "AAPL") == {SEEDED, AS_OF}
    assert _dates(db, "MSFT") == {SEEDED}
    coverage = classify_price_history(db, symbols=["AAPL", "MSFT"], as_of_date=AS_OF)
    assert list(coverage.fresh) == ["AAPL"]
    assert list(coverage.stale) == ["MSFT"]
    # The completed daily-import state still names the seeded bundle.
    state = DailyPriceBundleService().get_import_state(db, "US")
    assert state["source_revision"] == "daily_prices_us:seed"


@pytest.mark.parametrize(
    "field",
    sorted(
        checkpoint_fingerprint(
            _runner_db(), market="US", as_of_date=AS_OF
        )
    ),
)
def test_resume_rejects_every_fingerprint_mismatch(tmp_path, field):
    release = _checkpointed_release(tmp_path)
    manifest_path = release / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint"][field] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    db = _runner_db()

    result = _resume(db, release)

    assert result["status"] == "incompatible"
    assert field in result["reason"]
    assert _dates(db, "AAPL") == {SEEDED}


def test_resume_rejects_a_later_session(tmp_path):
    release = _checkpointed_release(tmp_path)
    db = _runner_db()

    result = _resume(db, release, as_of_date=date(2026, 4, 20))

    assert result["status"] == "incompatible"
    assert "as_of_date" in result["reason"]
    assert _dates(db, "AAPL") == {SEEDED}


def test_resume_rejects_a_newer_baseline_bundle(tmp_path):
    release = _checkpointed_release(tmp_path)
    db = _runner_db(baseline="daily_prices_us:newer")

    result = _resume(db, release)

    assert result["status"] == "incompatible"
    assert "baseline_daily_price_revision" in result["reason"]


def test_resume_without_a_checkpoint_ignores_the_latest_daily_manifest(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    db = _runner_db()
    DailyPriceBundleService().export_daily_price_bundle(
        db,
        market="US",
        output_path=release / "daily-price-us-20260416.json.gz",
        bundle_asset_name="daily-price-us-20260416.json.gz",
        latest_manifest_path=release / "daily-price-latest-us.json",
        as_of_date=SEEDED,
    )

    assert _resume(db, release)["status"] == "missing"


def test_resume_rejects_a_daily_manifest_under_the_checkpoint_name(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    db = _runner_db()
    DailyPriceBundleService().export_daily_price_bundle(
        db,
        market="US",
        output_path=release / BUNDLE,
        bundle_asset_name=BUNDLE,
        latest_manifest_path=release / MANIFEST,
        as_of_date=AS_OF,
    )

    result = _resume(db, release)

    assert result["status"] == "incompatible"
    assert "not a price checkpoint" in result["reason"]


def test_resume_ignores_data_replaced_without_its_manifest(tmp_path):
    # An interrupted upload: new data landed, its manifest did not.
    release = _checkpointed_release(tmp_path)
    (release / BUNDLE).write_bytes((release / BUNDLE).read_bytes() + b"\n")
    db = _runner_db()

    result = _resume(db, release)

    assert result["status"] == "invalid"
    assert _dates(db, "AAPL") == {SEEDED}
