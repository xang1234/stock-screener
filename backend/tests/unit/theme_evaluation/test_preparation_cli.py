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


def test_translation_requires_its_own_flag_and_records_kimi_provenance(
    bundle, document, tmp_path, capsys, monkeypatch
):
    import httpx
    from app.services.theme_evaluation import preparation_cli
    from app.services.theme_evaluation.preparation_store import PreparationStore

    calls = []

    def factory(key):
        from app.services.theme_evaluation.kimi_translation import OpenCodeGoTranslator

        def handler(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {"translation": "Revenue increased."}
                                )
                            },
                        }
                    ]
                },
            )

        return OpenCodeGoTranslator(key, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(preparation_cli, "OpenCodeGoTranslator", factory, raising=False)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(text="매출 증가", original_language="ko")])
        ),
    )
    args = [
        "--bundle",
        str(base),
        "--output-root",
        str(tmp_path / "out"),
        "--stages",
        "text",
    ]
    assert preparation_cli.main(["prepare", *args]) == 0
    capsys.readouterr()
    assert not calls
    assert preparation_cli.main(["prepare", *args, "--allow-translation-calls"]) == 0
    pid = json.loads(capsys.readouterr().out)["preparation_id"]
    store = PreparationStore(tmp_path / "out")
    result = store.load_result(store.load(base, pid).current_bindings[0].result_id)
    assert result.request.provider == "opencode-go"
    assert result.request.model == "kimi-k2.6"
    assert result.payload.segments[0].translated == "Revenue increased."
    assert len(calls) == 1


def test_translation_switch_requires_key_before_output(
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
                "text",
                "--allow-translation-calls",
            ]
        )
        == 2
    )
    assert "error" in json.loads(capsys.readouterr().err)
    assert not (tmp_path / "out").exists()
