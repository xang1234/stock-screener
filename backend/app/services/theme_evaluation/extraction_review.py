"""Review frozen baseline inputs/results; never invoke a provider."""

import json
from collections import Counter

from .exclusion_review import EXCLUSION_FIELDS, exclusion_rows
from .extraction_capture import load_extraction_run
from .review import _text, _write_csv

_HOLD_ACTIONS = {"held_theme", "held_development"}


def _claim_decisions(audit):
    """Return only readable decision objects from an optional, versioned audit."""
    if not isinstance(audit, dict):
        return []
    return [decision for decision in audit.get("decisions", []) if isinstance(decision, dict)]


def _claim_counts(audit):
    decisions = _claim_decisions(audit)
    actions = Counter(decision.get("action") for decision in decisions)
    return {
        "candidate_count": len(audit.get("candidates", []))
        if isinstance(audit, dict) and isinstance(audit.get("candidates"), list)
        else 0,
        "accepted_candidates": actions["accepted"],
        "held_theme_candidates": actions["held_theme"],
        "held_development_candidates": actions["held_development"],
        "unavailable_candidates": actions["review_unavailable"],
        "decision_count": len(decisions),
    }


def _result_status(result):
    if result is None:
        return "Pending"
    if result.status == "failed":
        return "Failed"
    audit = result.claim_review
    counts = _claim_counts(audit)
    unavailable = counts["unavailable_candidates"]
    if result.mentions:
        suffix = (
            f" · {unavailable} candidate review unavailable"
            if unavailable
            else ""
        )
        return f"{len(result.mentions)} theme mentions{suffix}"
    if counts["candidate_count"]:
        if unavailable:
            return (
                "Candidate review partially unavailable "
                f"({unavailable} candidate unavailable)"
            )
        if (
            counts["decision_count"] == counts["candidate_count"]
            and all(
                decision.get("action") in _HOLD_ACTIONS
                for decision in _claim_decisions(audit)
            )
        ):
            return "All candidates held after evidence review"
        return "Candidate review outcome incomplete"
    return "No themes"


def _decision_description(decision):
    action = _text(decision.get("action", "unknown"))
    if decision.get("action") == "review_unavailable":
        return action + ": " + _text(
            decision.get("error_code") or "review result unavailable"
        )
    theme = decision.get("theme")
    development = decision.get("development")
    theme_reason = theme.get("reason") if isinstance(theme, dict) else None
    development_reason = (
        development.get("reason") if isinstance(development, dict) else None
    )
    return (
        action
        + ": "
        + _text(theme_reason or "theme verdict unavailable")
        + " / "
        + _text(development_reason or "development verdict unavailable")
    )


def render_extractions(run_path, output, *, base=None, store=None):
    run = load_extraction_run(run_path)
    excluded = exclusion_rows(run, base=base, store=store)
    output.mkdir(parents=True, exist_ok=False)
    manifest, inputs, records = run["manifest"], run["inputs"], run["records"]
    by_input = {(r.input_id, r.pipeline): r for r in records}
    counts = Counter(r.status for r in records)
    summary = {
        "run_id": run["run_id"],
        "admitted_inputs": len(inputs),
        "successful_extractions": counts["success"],
        "failed_extractions": counts["failed"],
        "empty_successes": sum(
            r.status == "success" and not r.mentions for r in records
        ),
        "extraction_status": manifest.get("extraction_status", "unknown"),
        "ranking_evaluation": "unavailable",
        "labels": "not_reviewed",
        "held_theme_candidates": sum(
            d.get("action") == "held_theme"
            for r in records
            for d in (r.claim_review or {}).get("decisions", [])
        ),
        "held_developments": sum(
            d.get("action") == "held_development"
            for r in records
            for d in (r.claim_review or {}).get("decisions", [])
        ),
        "claim_review_unavailable": sum(
            (r.claim_review or {}).get("status") == "unavailable" for r in records
        ),
        "claim_review_partial_outcomes": sum(
            (r.claim_review or {}).get("status") == "partial" for r in records
        ),
        "claim_review_unavailable_candidates": sum(
            _claim_counts(r.claim_review)["unavailable_candidates"] for r in records
        ),
    }
    lines = [
        "# Frozen baseline extraction review",
        "",
        f"Run: `{run['run_id']}`",
        "",
        f"{len(inputs)} admitted inputs; {len(records)} recorded extraction outcomes.",
        "",
        (
            "Pending means no model output. No themes means a successful empty extraction. "
            "Failed means unavailable output. These are not interchangeable."
        ),
        "",
        "Theme labels remain unreviewed. This pilot does not report ranking accuracy or historical speed.",
        "",
        "[Input details](inputs.csv) · [Theme and development details](theme-mentions.csv) · [Extraction details](extractions.csv) · [Excluded evidence](exclusions.csv) · [Grounding context](grounding.csv) · [Claim review and held candidates](claim-review.csv)",
        "",
    ]
    mention_rows = []
    grounding_rows = []
    claim_rows = []
    for item in inputs:
        lines.extend(
            [
                "## " + _text(item.source_id) + " — " + item.input_kind,
                "",
                "**" + _text(item.title) + "**",
                "",
                "Source: " + _text(item.source_url),
                "",
                f"Input: `{item.input_id}` · Available: {item.available_at.isoformat()}",
                "",
                "Limitations: " + _text(", ".join(item.warnings) or "none recorded"),
                "",
            ]
        )
        lines.extend("> " + _text(line) for line in item.text.splitlines())
        lines.append("")
        for pipeline in ("technical", "fundamental"):
            result = by_input.get((item.input_id, pipeline))
            status = _result_status(result)
            lines.extend(["**" + pipeline.title() + ": " + status + "**", ""])
            if result:
                if result.claim_review is not None:
                    audit = result.claim_review
                    claim_counts = _claim_counts(audit)
                    claim_rows.append(
                        {
                            "input_id": item.input_id,
                            "pipeline": pipeline,
                            "source_url": item.source_url,
                            "source_text": item.text,
                            "status": audit.get("status"),
                            "error_code": audit.get("error_code"),
                            **claim_counts,
                            "candidates": json.dumps(
                                audit.get("candidates", []), ensure_ascii=False
                            ),
                            "decisions": json.dumps(
                                audit.get("decisions", []), ensure_ascii=False
                            ),
                        }
                    )
                    lines.extend(
                        ["Claim review: " + _text(audit.get("status", "unknown")), ""]
                    )
                    for decision in _claim_decisions(audit):
                        lines.extend(["- " + _decision_description(decision), ""])
                context = result.grounding_context
                if context is not None:
                    grounding_rows.append(
                        {
                            "input_id": item.input_id,
                            "pipeline": pipeline,
                            "source_url": item.source_url,
                            "context_available_at": context.get("context_available_at"),
                            "companies": json.dumps(
                                context.get("companies", []), ensure_ascii=False
                            ),
                            "evidence": json.dumps(
                                context.get("evidence", []), ensure_ascii=False
                            ),
                            "warnings": json.dumps(
                                context.get("warnings", []), ensure_ascii=False
                            ),
                        }
                    )
                    lines.extend(["**Grounding supplied:**", ""])
                    for company in context.get("companies", []):
                        lines.extend(
                            [
                                "- "
                                + _text(company.get("symbol", ""))
                                + ": "
                                + _text(company.get("name") or "Unknown name")
                                + " — "
                                + _text(
                                    company.get("business_description")
                                    or "Business description unavailable"
                                )
                            ]
                        )
                    for evidence in context.get("evidence", []):
                        lines.extend(
                            [
                                "",
                                "Related evidence ("
                                + _text(evidence["relation"])
                                + "): "
                                + _text(evidence["source_url"]),
                                "",
                                _text(evidence["text"]),
                            ]
                        )
                    lines.extend(
                        [
                            "",
                            "Context limitations: "
                            + _text(
                                ", ".join(context.get("warnings", []))
                                or "none recorded"
                            ),
                            "",
                        ]
                    )
                if result.error_code:
                    lines.extend([_text(result.error_code), ""])
                for mention in result.mentions:
                    lines.extend(
                        [
                            "**Theme:** " + _text(mention["theme"]),
                            "",
                            "**Development:** "
                            + _text(mention.get("development") or "Not recorded"),
                            "",
                            "**Evidence:** " + _text(mention["excerpt"]),
                            "",
                            "Tickers: "
                            + _text(", ".join(mention["tickers"]) or "none")
                            + " · Sentiment: "
                            + _text(mention["sentiment"])
                            + " · Confidence: "
                            + str(mention["confidence"]),
                            "",
                        ]
                    )
                    mention_rows.append(
                        {
                            "input_id": item.input_id,
                            "pipeline": pipeline,
                            "theme": mention["theme"],
                            "theme_support": (mention.get("claim_support") or {}).get(
                                "theme"
                            ),
                            "development_support": (
                                mention.get("claim_support") or {}
                            ).get("development"),
                            "development": mention.get("development"),
                            "excerpt": mention["excerpt"],
                            "tickers": json.dumps(
                                mention["tickers"], ensure_ascii=False
                            ),
                            "sentiment": mention["sentiment"],
                            "confidence": mention["confidence"],
                            "source_id": item.source_id,
                            "source_url": item.source_url,
                            "source_text": item.text,
                            "input_kind": item.input_kind,
                            "warnings": json.dumps(item.warnings, ensure_ascii=False),
                        }
                    )
    (output / "extractions.md").write_text("\n".join(lines), encoding="utf-8")
    input_rows = [i.model_dump(mode="json") for i in inputs]
    for row in input_rows:
        for key in ("result_ids", "warnings", "normalization"):
            row[key] = json.dumps(row[key], ensure_ascii=False)
    _write_csv(
        output / "inputs.csv",
        list(inputs[0].model_fields) if inputs else ["input_id"],
        input_rows,
    )
    fields = [
        "input_id",
        "pipeline",
        "status",
        "mentions",
        "error_code",
        "generated_at",
        "requested_model",
        "reference_sha256",
        "code_revision",
        "calls",
        "grounding_context",
        "claim_review",
    ]
    rows = [r.model_dump(mode="json") for r in records]
    for row in rows:
        for key in ("mentions", "calls", "grounding_context", "claim_review"):
            row[key] = json.dumps(row.get(key), ensure_ascii=False)
    _write_csv(output / "extractions.csv", fields, rows)
    _write_csv(
        output / "claim-review.csv",
        [
            "input_id",
            "pipeline",
            "source_url",
            "source_text",
            "status",
            "error_code",
            "candidate_count",
            "accepted_candidates",
            "held_theme_candidates",
            "held_development_candidates",
            "unavailable_candidates",
            "decision_count",
            "candidates",
            "decisions",
        ],
        claim_rows,
    )
    _write_csv(
        output / "grounding.csv",
        [
            "input_id",
            "pipeline",
            "source_url",
            "context_available_at",
            "companies",
            "evidence",
            "warnings",
        ],
        grounding_rows,
    )
    _write_csv(
        output / "theme-mentions.csv",
        [
            "input_id",
            "pipeline",
            "theme",
            "theme_support",
            "development_support",
            "development",
            "excerpt",
            "tickers",
            "sentiment",
            "confidence",
            "source_id",
            "source_url",
            "source_text",
            "input_kind",
            "warnings",
        ],
        mention_rows,
    )
    _write_csv(
        output / "exclusions.csv",
        EXCLUSION_FIELDS,
        excluded,
    )
    (output / "coverage.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
