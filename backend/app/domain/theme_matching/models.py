"""Value objects for theme-cluster matching decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MatchThresholdConfig:
    """Threshold policy with pipeline and source-type override support."""

    version: str = "v1"
    default_threshold: float = 1.0
    pipeline_overrides: dict[str, float] = field(default_factory=dict)
    source_type_overrides: dict[str, float] = field(default_factory=dict)
    pipeline_source_type_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    def resolve_threshold(self, *, pipeline: str, source_type: str | None = None) -> float:
        """Resolve the effective threshold for a pipeline and optional source type."""
        normalized_pipeline = (pipeline or "").strip().lower()
        normalized_source_type = (source_type or "").strip().lower()

        pipeline_source = self.pipeline_source_type_overrides.get(normalized_pipeline, {})
        if normalized_source_type and normalized_source_type in pipeline_source:
            return float(pipeline_source[normalized_source_type])
        if normalized_pipeline in self.pipeline_overrides:
            return float(self.pipeline_overrides[normalized_pipeline])
        if normalized_source_type and normalized_source_type in self.source_type_overrides:
            return float(self.source_type_overrides[normalized_source_type])
        return float(self.default_threshold)


@dataclass(frozen=True)
class MatchDecision:
    """Typed decision envelope used for cluster assignment telemetry."""

    selected_cluster_id: int | None
    method: str
    score: float
    threshold: float
    threshold_version: str
    fallback_reason: str | None = None
    best_alternative_cluster_id: int | None = None
    best_alternative_score: float | None = None
    score_margin: float | None = None
    score_model: str | None = None
    score_model_version: str | None = None
