"""Portable Markdown/CSV review of original evidence and preparation derivatives."""

import json
from pathlib import Path

from .image_preparation import validate_image
from .review import _text, _write_csv, render_review


def active_binding_indices(manifest, store):
    latest = {}
    for index, binding in enumerate(manifest.bindings):
        result = store.load_result(binding.result_id)
        key = (
            binding.source_kind,
            binding.source_id,
            result.request.stage,
            binding.parent_result_id,
            binding.input_locator,
        )
        latest[key] = index
    roots = {
        (
            manifest.bindings[i].source_kind,
            manifest.bindings[i].source_id,
            manifest.bindings[i].result_id,
        )
        for i in latest.values()
        if manifest.bindings[i].parent_result_id is None
    }
    return {
        i
        for i in latest.values()
        if not manifest.bindings[i].parent_result_id
        or (
            manifest.bindings[i].source_kind,
            manifest.bindings[i].source_id,
            manifest.bindings[i].parent_result_id,
        )
        in roots
    }


def render_preparation(base, store, preparation_id: str, output: Path):
    manifest = store.load(base, preparation_id)
    bundle = store.validate(base, manifest)
    output.mkdir(parents=True, exist_ok=False)
    render_review(bundle, output / "original")
    rows, followups = [], []
    active = active_binding_indices(manifest, store)
    lines = [
        "# Prepared evidence — review before extraction",
        "",
        "Evidence review: **pending**. Extraction: **awaiting_evidence_approval**.",
        "",
        f"Base bundle: `{manifest.bundle_id}`",
        f"Preparation: `{preparation_id}`",
        "",
        "[Original posts, articles and source coverage](original/evidence.md)",
        "",
        (
            "Transcriptions, translations and visual observations are derivatives of the original evidence. "
            "They do not count as independent corroboration."
        ),
        "",
    ]
    images = output / "images"
    for number, binding in enumerate(manifest.bindings, 1):
        result = store.load_result(binding.result_id)
        row = {
            "version": "current" if number - 1 in active else "superseded",
            "input_locator": binding.input_locator or "",
            "source_kind": binding.source_kind,
            "source_id": binding.source_id,
            "result_id": binding.result_id,
            "parent_result_id": binding.parent_result_id or "",
            "stage": result.request.stage,
            "status": result.status,
            "created_at": result.created_at.isoformat(),
            "provider": result.request.provider,
            "model": result.request.model or "",
            "policy_version": result.request.policy_version,
            "source_url": result.source_url or "",
            "warnings": json.dumps(result.warnings, ensure_ascii=False),
            "payload": json.dumps(result.payload, ensure_ascii=False),
        }
        rows.append(row)
        lines.extend(
            [
                f"## {number}. {_text(binding.source_id)} — {result.request.stage}",
                "",
                f"Version: **{row['version']}** · Status: **{result.status}** · Provider/model: {_text(result.request.provider)} / {_text(result.request.model or 'none')}",
                f"Captured/generated: {_text(result.created_at.isoformat())}",
                "Warnings: " + _text(", ".join(result.warnings) or "none"),
                "",
            ]
        )
        if result.source_url:
            lines.extend(["Source URL: " + _text(result.source_url), ""])
        if result.request.stage == "image":
            for asset in result.assets:
                raw = store.load_asset(asset)
                metadata = validate_image(raw)
                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }[metadata["mime_type"]]
                images.mkdir(exist_ok=True)
                target = images / (asset + suffix)
                if not target.exists():
                    target.write_bytes(raw)
                lines.extend([f"![Original image](images/{target.name})", ""])
            for label, value in (
                ("Transcription", result.payload.get("transcription", "")),
                ("Observations", "\n".join(result.payload.get("observations", []))),
                ("Uncertainty", "\n".join(result.payload.get("uncertainties", []))),
            ):
                lines.extend([f"**{label}**", ""])
                lines.extend("> " + _text(line) for line in value.splitlines())
                lines.append("")
        elif result.request.stage == "text":
            lines.extend(
                ["Language: " + _text(result.payload.get("source_language", "und")), ""]
            )
            for segment in result.payload.get("segments", []):
                lines.extend(["**Original segment**", ""])
                lines.extend(
                    "> " + _text(line) for line in segment["original"].splitlines()
                )
                lines.extend(["", "**Translation: " + segment["status"] + "**", ""])
                lines.extend(
                    "> " + _text(line)
                    for line in (segment["translated"] or "[unavailable]").splitlines()
                )
                lines.append("")
        else:
            lines.extend(
                "> " + _text(line)
                for line in result.payload.get("text", "").splitlines()
            )
            lines.append("")
            if result.status != "success" and number - 1 in active:
                followups.append(row)
    (output / "evidence.md").write_text("\n".join(lines), encoding="utf-8")
    fields = [
        "version",
        "input_locator",
        "source_kind",
        "source_id",
        "result_id",
        "parent_result_id",
        "stage",
        "status",
        "created_at",
        "provider",
        "model",
        "policy_version",
        "source_url",
        "warnings",
        "payload",
    ]
    _write_csv(output / "preparations.csv", fields, rows)
    _write_csv(output / "article_followups.csv", fields, followups)
    summary = {
        "preparation_id": preparation_id,
        "bundle_id": manifest.bundle_id,
        "evidence_review": "pending",
        "extraction": "awaiting_evidence_approval",
        "results": len({b.result_id for b in manifest.bindings}),
        "bindings": len(manifest.bindings),
        "gaps": sum(
            row["status"] != "success" and row["version"] == "current" for row in rows
        ),
        "documents_without_image_metadata": sum(
            not d.source_metadata.image_urls
            and not (
                manifest.handoff.documents.get(d.document_id)
                and (
                    manifest.handoff.documents[d.document_id].image_urls
                    or manifest.handoff.documents[d.document_id].local_images
                )
            )
            for d in bundle.documents
        ),
    }
    (output / "coverage.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
