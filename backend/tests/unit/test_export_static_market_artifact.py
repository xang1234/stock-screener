import json


def test_price_checkpoint_exit_is_reported_as_checkpointed(tmp_path):
    from app.scripts import export_static_market_artifact as wrapper

    path = wrapper.write_market_status(
        output_dir=tmp_path,
        market="US",
        exit_code=80,
    )

    status = json.loads(path.read_text())
    assert status["status"] == "failed"
    assert status["reason"] == "price_checkpointed"
    assert status["has_current_artifact"] is False
    assert status["has_price_bundle"] is False
