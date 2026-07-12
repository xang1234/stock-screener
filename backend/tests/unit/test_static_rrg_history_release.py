from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.scripts.restore_static_rrg_history import main
from app.services.github_release_sync_service import (
    NamedAssetFetchResult,
    NamedAssetFetchStatus,
)
from app.services.static_rrg_history_release import (
    StaticRRGHistoryReleaseRestorer,
    StaticRRGHistoryRestoreResult,
    StaticRRGHistoryRestoreStatus,
)


def test_rrg_release_restorer_retries_transient_failure(tmp_path):
    output_path = tmp_path / "rrg-history-hk.json.gz"
    results = iter(
        [
            NamedAssetFetchResult(
                status=NamedAssetFetchStatus.NETWORK_ERROR,
                asset_name=output_path.name,
                error="temporary",
            ),
            NamedAssetFetchResult(
                status=NamedAssetFetchStatus.SUCCESS,
                asset_name=output_path.name,
                output_path=output_path,
            ),
        ]
    )
    sleeps: list[float] = []
    restorer = StaticRRGHistoryReleaseRestorer(
        sync_service=SimpleNamespace(fetch_named_asset=lambda **_kwargs: next(results)),
        sleep=sleeps.append,
    )

    restored = restorer.restore(
        repository_full_name="xang1234/stock-screener",
        release_tag="rrg-history-data",
        asset_name=output_path.name,
        output_path=output_path,
        github_token="token",
        request_timeout_seconds=60,
        attempts=3,
        retry_delay_seconds=2,
    )

    assert restored.status is StaticRRGHistoryRestoreStatus.RESTORED
    assert restored.safe_to_publish is True
    assert sleeps == [2]


@pytest.mark.parametrize(
    ("status", "expected_exit", "safe_to_publish"),
    [
        (StaticRRGHistoryRestoreStatus.RESTORED, 0, True),
        (StaticRRGHistoryRestoreStatus.MISSING, 0, True),
        (StaticRRGHistoryRestoreStatus.FAILED, 1, False),
    ],
)
def test_restore_static_rrg_history_cli_contract(
    tmp_path,
    capsys,
    status,
    expected_exit,
    safe_to_publish,
):
    output_path = tmp_path / "rrg-history-hk.json.gz"
    calls: list[dict[str, object]] = []

    def restore(**kwargs):
        calls.append(kwargs)
        return StaticRRGHistoryRestoreResult(
            status=status,
            asset_name=output_path.name,
            output_path=output_path,
            detail="fixture",
        )

    exit_code = main(
        [
            "--repository",
            "xang1234/stock-screener",
            "--asset-name",
            output_path.name,
            "--output-path",
            str(output_path),
            "--attempts",
            "2",
            "--retry-delay-seconds",
            "0",
        ],
        restorer=SimpleNamespace(restore=restore),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == expected_exit
    assert payload == {
        "asset_name": output_path.name,
        "detail": "fixture",
        "output_path": str(output_path),
        "safe_to_publish": safe_to_publish,
        "status": status.value,
    }
    assert calls[0]["attempts"] == 2
    assert calls[0]["output_path"] == output_path
