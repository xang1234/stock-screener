"""Prioritized Markdown and CSV views of a verified assessment."""

import json
from collections import Counter

from .bundle import canonical_bytes
from .review import _text, _write_csv


def write_assessment_packet(output, assessment, assessment_id):
    (output / "assessment.json").write_bytes(canonical_bytes(assessment))
    entries = assessment["entries"]
    fields = [
        "entry_id",
        "source_id",
        "issue_code",
        "severity",
        "legacy_status",
        "eligible",
        "result_id",
        "selected_result_id",
        "candidate_result_ids",
        "input_sha256",
        "source_text_sha256",
        "reviewer_disposition",
        "claim",
        "explanation",
        "next_action",
        "manual_decision",
    ]
    for stage, filename in [
        ("text", "translation"),
        ("article", "article"),
        ("image", "image"),
    ]:
        rows = []
        lines = [
            f"# {filename.title()} review",
            "",
            "[Review overview](START_HERE.md) · [Detailed evidence and history](evidence.md)",
            "",
        ]
        for key, entry in entries.items():
            if entry["stage"] != stage:
                continue
            row = {field: entry.get(field, "") for field in fields}
            row["entry_id"] = key
            for field in ("candidate_result_ids", "manual_decision"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            rows.append(row)
            lines.extend(
                [
                    f"## {_text(entry['source_id'])} — {entry['severity']}",
                    "",
                    f"Issue: `{entry['issue_code']}` · Legacy status: `{entry['legacy_status']}` · Eligible: `{entry['eligible']}`",
                    "",
                    _text(entry["explanation"]),
                    "",
                    _text(entry["next_action"]),
                    "",
                    f"Entry: `{key}` · Result: `{entry['result_id'] or 'original'}`",
                    "",
                ]
            )
        _write_csv(output / (filename + "-review.csv"), fields, rows)
        (output / (filename + "-review.md")).write_text(
            "\n".join(lines), encoding="utf-8"
        )
    counts = Counter(e["severity"] for e in entries.values())
    lines = [
        "# Evidence review starts here",
        "",
        "**Evidence review pending. Extraction awaiting_evidence_approval.**",
        "",
        f"Bundle: `{assessment['bundle_id']}`",
        f"Preparation: `{assessment['preparation_id']}`",
        f"Assessment: `{assessment_id}` ({assessment['policy_version']})",
        "",
        "[Translations](translation-review.md) · [Articles](article-review.md) · [Images](image-review.md) · [Original evidence and all attempts](evidence.md)",
        "",
        "[Linked-post follow-ups](linked-post-review.md) · [Reference relationships](reference-manifest.csv) · [Run counts](preparation-runs.json)",
        "",
        "Legacy preparation status describes processing only. Assessment severity and deterministic eligibility are separate; neither is human approval.",
        "",
        "## Coverage",
        "",
        "| Measure | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {key.replace('_', ' ')} | {value} |"
        for key, value in assessment["coverage"].items()
    )
    lines.extend(
        [
            "",
            "Reference records preserve every relationship; distinct destinations and image inputs use separate denominators. Excluded evidence remains in coverage.",
            "",
        ]
    )
    groups = [
        ("Material holds", lambda e: e["severity"] == "hold"),
        ("Recoverable gaps and review", lambda e: e["severity"] == "review"),
        ("Informational notes", lambda e: e["severity"] == "info"),
        ("Exclusions", lambda e: e["reviewer_disposition"] == "exclude"),
    ]
    for heading, predicate in groups:
        lines.extend(["## " + heading, ""])
        selected = [
            (key, e)
            for key, e in entries.items()
            if predicate(e)
            and (heading == "Exclusions" or e["reviewer_disposition"] != "exclude")
        ]
        for key, entry in selected:
            reason = (
                (
                    entry["manual_decision"]["reason"]
                    if entry["manual_decision"]
                    else "Excluded by whole-evidence annotation."
                )
                if heading == "Exclusions"
                else entry["next_action"]
            )
            lines.extend(
                [
                    f"- **{_text(entry['source_id'])}** — `{entry['issue_code']}`. {_text(reason)} Entry `{key}`."
                ]
            )
        if not selected:
            lines.append("None.")
        lines.append("")
    (output / "START_HERE.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(
        assessment_id=assessment_id,
        assessment_severity_counts=dict(counts),
        selection_id=assessment["selection_id"],
        **assessment["coverage"],
    )
