"""Move a combined static-data tree under an immutable generation directory (#504).

Every publish overwrites the same data paths, so an open tab could pair its
manifest with files from a later publish. After relocation the root
``manifest.json`` is the pointer: it names ``generation`` and ``data_root``
(``g/<generation>/``), and all data lives under that directory. Paths inside
the manifest and the data files stay root-relative; the frontend resolves
them under ``data_root``. Only the current generation is deployed (the site
is already near the Pages size limit), so a tab left on an older generation
must reload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENERATIONS_DIRNAME = "g"
MANIFEST_FILENAME = "manifest.json"


class StaticDataGenerationError(RuntimeError):
    """The static-data tree cannot be relocated into a generation."""


def _files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def relocate_into_generation(output_dir: Path) -> str:
    """Move every entry except the root manifest under ``g/<generation>/``.

    Returns the generation id: the manifest's ``generated_at`` in compact form
    plus 8 hex of the manifest digest, so two publishes in one second differ.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise StaticDataGenerationError(f"No {MANIFEST_FILENAME} in {output_dir}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("data_root") or (output_dir / GENERATIONS_DIRNAME).exists():
        raise StaticDataGenerationError(f"{output_dir} is already relocated into a generation")

    stamp = "".join(ch for ch in str(manifest.get("generated_at") or "") if ch.isalnum())
    generation = f"{stamp or 'unstamped'}-{hashlib.sha256(raw).hexdigest()[:8]}"
    data_root = output_dir / GENERATIONS_DIRNAME / generation

    entries = [entry for entry in output_dir.iterdir() if entry.name != MANIFEST_FILENAME]
    before = {
        path.relative_to(output_dir).as_posix()
        for entry in entries
        for path in ([entry] if entry.is_file() else entry.rglob("*"))
        if path.is_file()
    }
    data_root.mkdir(parents=True)
    for entry in entries:
        entry.rename(data_root / entry.name)
    if _files(data_root) != before:
        raise StaticDataGenerationError(f"Relocation into {data_root} lost or added files")

    manifest["generation"] = generation
    manifest["data_root"] = f"{GENERATIONS_DIRNAME}/{generation}/"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return generation
