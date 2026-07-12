from app.scripts.restore_github_release_asset import restore_result_is_usable


def test_restore_result_is_usable_fails_closed():
    assert restore_result_is_usable(
        {"status": "success"},
        allow_missing=True,
    )
    assert restore_result_is_usable(
        {"status": "missing_asset"},
        allow_missing=True,
    )
    assert not restore_result_is_usable(
        {"status": "missing_asset"},
        allow_missing=False,
    )
    assert not restore_result_is_usable(
        {"status": "network_error"},
        allow_missing=True,
    )
    assert not restore_result_is_usable({}, allow_missing=True)
