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


def test_the_artifact_validator_accepts_a_checkpointed_market_status():
    from app.scripts.validate_static_market_artifacts import MarketArtifactStatus

    status = MarketArtifactStatus.from_payload(
        {
            "market": "US",
            "has_current_artifact": False,
            "has_price_bundle": False,
            "status": "failed",
            "reason": "price_checkpointed",
        }
    )

    assert status.reason == "price_checkpointed"
