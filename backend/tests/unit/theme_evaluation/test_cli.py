import json
import os
import subprocess
import sys
from pathlib import Path

from app.services.theme_evaluation.cli import main


def test_import_review_roundtrip_needs_no_database_or_provider(tmp_path, xui_payloads, capsys):
    inputs = []
    for key, value in xui_payloads.items():
        path = tmp_path / (key + '.json')
        path.write_text(json.dumps(value))
        inputs.append(str(path))
    root = tmp_path / 'corpus'
    assert main(['import-x', '--first', inputs[0], '--second', inputs[1],
                 '--mode', 'controlled', '--output-root', str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    path = result['bundle_path']
    assert main(['references', '--bundle', path, '--output', str(tmp_path / 'refs.json')]) == 0
    assert main(['review', '--bundle', path, '--output', str(tmp_path / 'report')]) == 0
    capsys.readouterr()
    assert (tmp_path / 'report' / 'evidence.md').exists()
    assert main(['verify', '--bundle', path]) == 0
    capsys.readouterr()
    (Path(path) / 'bundle.json').write_text('{}')
    assert main(['verify', '--bundle', path]) == 5


def test_standalone_cli_help_does_not_load_application_settings():
    script = Path(__file__).resolve().parents[3] / 'scripts' / 'theme_evaluation.py'
    env = dict(os.environ, DATABASE_URL='invalid://must-not-be-used')
    result = subprocess.run([sys.executable, str(script), '--help'], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'import-x' in result.stdout
    assert 'generate-extractions' not in result.stdout
