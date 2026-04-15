"""Shared signed-artifact primitives for governance reports.

Both the weekly telemetry audit (bead asia.10.4) and the launch-gate runner
(bead asia.11.1) emit the same trio: indented JSON, Markdown, and a
sha256sum-compatible sidecar. The two-hash contract (semantic content_hash
inside the JSON, file-bytes hash in the .sha256 sidecar) is identical for
both — extracted here so a future hash-format change touches one module.

Contract:
- ``compute_content_hash(data)``: SHA-256 over canonical compact JSON
  (sorted keys, tight separators, ``content_hash`` set to None). Stable
  across runs given identical inputs; embedded into the data dict's
  ``content_hash`` field for programmatic re-verification.
- ``write_signed_artifact_trio(...)``: writes
  ``<stamp>.{json,md,sha256}`` and returns absolute paths. The .sha256
  sidecar holds SHA-256 of the actual .json file bytes (not content_hash)
  so ``sha256sum -c <stamp>.sha256`` works out of the box.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_content_hash(data: Dict[str, Any]) -> str:
    """Hash the data with ``content_hash`` nulled.

    Semantic integrity check: detects edits to any field other than
    ``content_hash`` itself, since the field is excluded from the input
    via temporary mutation. The caller is expected to set the resulting
    hash back onto ``data["content_hash"]``.
    """
    saved = data.get("content_hash")
    data["content_hash"] = None
    try:
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    finally:
        # Restore prior value (typically None on the first call) so callers
        # can compute, then assign the result onto the field — without this
        # restore the second call on the same dict would still see None.
        data["content_hash"] = saved


def write_signed_artifact_trio(
    *,
    out_dir: Path,
    stamp: str,
    json_blob: str,
    md_blob: str,
) -> Dict[str, str]:
    """Write the JSON + Markdown + sha256 trio. Return absolute paths.

    The .sha256 file uses sha256sum format (``<hash>  <filename>``) and
    hashes the raw bytes of the .json file, NOT the content_hash inside —
    they are different artifacts with different verification paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}.json"
    md_path = out_dir / f"{stamp}.md"
    hash_path = out_dir / f"{stamp}.sha256"

    json_path.write_text(json_blob, encoding="utf-8")
    md_path.write_text(md_blob, encoding="utf-8")
    # Hash the actual on-disk bytes, not the pre-write string. On Windows,
    # write_text() translates \n → \r\n by default, so encoding the string
    # directly would produce a hash that never matches sha256sum output.
    file_hash = hashlib.sha256(json_path.read_bytes()).hexdigest()
    hash_path.write_text(f"{file_hash}  {json_path.name}\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "sha256": str(hash_path),
    }
