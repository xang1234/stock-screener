"""Portable Markdown/CSV review of original evidence and preparation derivatives."""

import json
from pathlib import Path

from .image_preparation import validate_image
from .preparation_results import ArticleResult, ImageResult, TextResult
from .review import _text, _write_csv, render_review


def render_preparation(base, store, preparation_id: str, output: Path):
    manifest = store.load(base, preparation_id)
    bundle = store.validate(base, manifest)
    output.mkdir(parents=True, exist_ok=False)
    render_review(bundle, output / "original")
    rows, followups = [], []
    active = set(manifest.current.values())
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
            "They do not count as independent corroboration. "
            "Translations may retain original quantity notation: 억/億/亿 = 100 million; "
            "만/万/萬 = 10 thousand; 조/兆 = one trillion. Check the source currency and units."
        ),
        "",
    ]
    images = output / "images"
    for number, binding in enumerate(manifest.bindings, 1):
        result = store.load_result(binding.result_id)
        row = {
            "version": "current" if binding.binding_id in active else "superseded",
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
            "payload": json.dumps(
                result.payload.model_dump(mode="json") if result.payload else None,
                ensure_ascii=False,
            ),
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
        if isinstance(result, ImageResult):
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
                (
                    "Transcription",
                    result.payload.transcription if result.payload else "",
                ),
                (
                    "Observations",
                    "\n".join(result.payload.observations) if result.payload else "",
                ),
                (
                    "Uncertainty",
                    "\n".join(result.payload.uncertainties) if result.payload else "",
                ),
            ):
                lines.extend([f"**{label}**", ""])
                lines.extend("> " + _text(line) for line in value.splitlines())
                lines.append("")
        elif isinstance(result, TextResult):
            lines.extend(["Language: " + _text(result.payload.source_language), ""])
            for segment in result.payload.segments:
                lines.extend(["**Original segment**", ""])
                lines.extend(
                    "> " + _text(line) for line in segment.original.splitlines()
                )
                lines.extend(["", "**Translation: " + segment.status + "**", ""])
                lines.extend(
                    "> " + _text(line)
                    for line in (segment.translated or "[unavailable]").splitlines()
                )
                lines.append("")
        elif isinstance(result, ArticleResult):
            lines.extend("> " + _text(line) for line in result.source_text.splitlines())
            lines.append("")
            if result.status != "success" and binding.binding_id in active:
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
