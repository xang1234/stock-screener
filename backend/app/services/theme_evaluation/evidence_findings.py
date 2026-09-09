"""Stage-specific findings without changing legacy preparation statuses."""

from .preparation_results import ImageResult
from .translation_quality import assess_translation


def _text_findings(result, decision, root):
    if (
        result.payload.source_language == "zxx"
        and "nonlinguistic_content" in result.warnings
    ):
        return [
            (
                "translation_not_needed",
                "info",
                False,
                "Original nonlinguistic content retained; no English translation is needed.",
                "Inspect original if relevant.",
            )
        ]
    if root and decision is None:
        return [
            (
                "translation_selection_missing",
                "review",
                False,
                "No verified selection sidecar exists for this preparation.",
                "Prepare text again or review exact candidates.",
            )
        ]
    if root:
        if decision.eligible:
            return [
                (
                    "translation_adequate",
                    "info",
                    True,
                    "Selected candidate passes translation-quality-v1; earlier attempts remain in detailed evidence.",
                    "Review source and selected translation.",
                )
            ]
        return [
            (
                issue.code,
                "hold" if issue.severity == "blocker" else issue.severity,
                False,
                issue.source_excerpt + "\nTranslation: " + issue.translated_excerpt,
                "Recover or review the full source translation; exclude unsupported claims with a reason.",
            )
            for issue in decision.issues
        ] or [
            (
                "translation_pending",
                "review",
                False,
                "Translation selection remains unresolved.",
                "Review candidates.",
            )
        ]
    translations = [segment.translated for segment in result.payload.segments]
    if not translations or any(value is None for value in translations):
        return [
            (
                "translation_segment_missing",
                "hold",
                False,
                "At least one derivative translation segment is unavailable.",
                "Recover the missing segment or exclude the unsupported claims.",
            )
        ]
    assessment = assess_translation(
        result.source_text,
        "".join(s.translated or "" for s in result.payload.segments),
        language=result.payload.source_language,
    )
    return [
        (
            issue.code,
            "hold" if issue.severity == "blocker" else issue.severity,
            False,
            issue.source_excerpt + "\nTranslation: " + issue.translated_excerpt,
            "Review this derivative against its parent evidence.",
        )
        for issue in assessment.issues
    ] or [
        (
            "translation_adequate",
            "info",
            True,
            "Derivative translation passes the quality policy.",
            "Review parent evidence.",
        )
    ]


def findings(result, decision, root):
    if result.stage == "text":
        return _text_findings(result, decision, root)
    if isinstance(result, ImageResult):
        if not result.payload:
            return [
                (
                    code,
                    "review",
                    False,
                    "Image processing unavailable: " + code,
                    "Inspect retained image or retry only a diagnosed transient failure.",
                )
                for code in result.failure_reasons
            ]
        if result.payload.uncertainties:
            return [
                (
                    "image_uncertainty",
                    "review",
                    False,
                    claim,
                    "Inspect this field or claim; annotate significance and exclude unreadable values.",
                )
                for claim in result.payload.uncertainties
            ]
        return [
            (
                "image_output_unreviewed",
                "info",
                True,
                "Model output has no reported uncertainty; correctness has not been approved.",
                "Compare transcription and observations with the image.",
            )
        ]
    if not result.payload:
        return [
            (
                code,
                "info" if code in {"not_article", "same_x_post"} else "review",
                False,
                code,
                "Read linked posts through xui."
                if code == "linked_x_post"
                else "Inspect reference disposition; recover an accessible body if relevant.",
            )
            for code in result.failure_reasons
            if code != "browser_followup_required"
        ]
    if (
        result.payload.capture_status == "full"
        and not (result.payload.completeness_basis or "").strip()
    ):
        return [
            (
                "article_completeness_unverified",
                "review",
                False,
                "Legacy full status has no explicit completeness basis.",
                "Inspect the captured body and record completeness evidence.",
            )
        ]
    codes = result.payload.warnings or (
        [] if result.payload.capture_status == "full" else ["captured_partial"]
    )
    return [
        (
            code,
            "review",
            False,
            "Article evidence: " + code,
            "Supply reviewed completeness evidence for partial text; use browser import only when rendering is required.",
        )
        for code in codes
    ] or [
        (
            "article_captured",
            "info",
            True,
            "Article body has reviewer-supplied completeness provenance.",
            "Review article evidence.",
        )
    ]
