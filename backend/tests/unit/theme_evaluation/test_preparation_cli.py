import json

from app.services.theme_evaluation.bundle import seal_bundle
from app.services.theme_evaluation.records import Bundle


def test_offline_cli_seals_and_reviews_without_keys(
    bundle, tmp_path, capsys, monkeypatch
):
    from app.services.theme_evaluation import preparation_cli

    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    args = ["--bundle", str(base), "--output-root", str(tmp_path / "out")]
    assert preparation_cli.main(["prepare", *args, "--stages", "text"]) == 0
    pid = json.loads(capsys.readouterr().out)["preparation_id"]
    assert (
        preparation_cli.main(
            [
                "review",
                *args,
                "--preparation",
                pid,
                "--output",
                str(tmp_path / "review"),
            ]
        )
        == 0
    )
    assert (tmp_path / "review" / "evidence.md").exists()


def test_live_model_flag_requires_explicit_key_and_safe_error(
    bundle, tmp_path, capsys, monkeypatch
):
    from app.services.theme_evaluation import preparation_cli

    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    assert (
        preparation_cli.main(
            [
                "prepare",
                "--bundle",
                str(base),
                "--output-root",
                str(tmp_path / "out"),
                "--stages",
                "image",
                "--allow-model-calls",
            ]
        )
        == 2
    )
    assert "error" in json.loads(capsys.readouterr().err)
    assert not (tmp_path / "out").exists()
