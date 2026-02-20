"""Compatibility shim for legacy VCP detector import path.

Canonical implementation now lives in
`app.analysis.patterns.legacy_vcp_detection` so analysis-layer detectors can
reuse the same logic without importing scanner modules.
"""

from app.analysis.patterns.legacy_vcp_detection import VCPDetector, quick_vcp_score

__all__ = ("VCPDetector", "quick_vcp_score")
