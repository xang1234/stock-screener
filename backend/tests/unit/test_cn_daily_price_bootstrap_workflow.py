from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _workflow() -> str:
    return (ROOT / ".github" / "workflows" / "cn-daily-price-bootstrap.yml").read_text(
        encoding="utf-8"
    )


def test_cn_daily_price_bootstrap_workflow_is_separate_and_dispatchable() -> None:
    workflow = _workflow()

    assert "name: CN Daily Price Bootstrap" in workflow
    assert "workflow_dispatch:" in workflow
    assert "static-site.yml" not in workflow
    assert "pages" not in workflow.lower()


def test_cn_daily_price_bootstrap_workflow_uses_durable_shards_and_final_bundle() -> None:
    workflow = _workflow()

    assert "cn-daily-price-shards" in workflow
    assert "daily-price-data" in workflow
    assert "github.ref_name == github.event.repository.default_branch" in workflow
    assert "app.scripts.bootstrap_cn_daily_price_shard" in workflow
    assert "app.scripts.import_daily_price_bundle" in workflow
    assert "app.scripts.build_daily_price_bundle" in workflow
    assert "daily-price-cn-${AS_OF_COMPACT}-shard-${SHARD_INDEX}-of-${SHARD_COUNT}.json.gz" in workflow
    assert "--checkpoint-release-tag cn-daily-price-shards" in workflow
    assert "gh release upload daily-price-data" in workflow
    assert "needs: [plan, build-cn-price-shard]" in workflow


def test_cn_daily_price_bootstrap_workflow_validates_inputs_and_checkpoint_manifest() -> None:
    workflow = _workflow()

    assert "calendar.is_trading_day(\"CN\", as_of)" in workflow
    assert "latest_completed = calendar.last_completed_trading_day(\"CN\")" in workflow
    assert "SHARD_MANIFEST=" in workflow
    assert "--manifest \"/tmp/cn-prior-price-shard/${SHARD_MANIFEST}\"" in workflow
    assert "--expected-market CN" in workflow
    assert "--expected-as-of-date \"${AS_OF_DATE}\"" in workflow
    assert "--expected-bundle-asset-name \"${SHARD_ASSET}\"" in workflow
