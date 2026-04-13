"""Multilingual QA harness (T7.6).

Runs the deterministic stages of the multilingual extraction pipeline
against :mod:`multilingual_qa_corpus` and asserts precision / recall /
accuracy meet the gate thresholds defined below. The gate numbers
come from the Asia v2 launch-gate charter (G5: multilingual precision
≥ 0.85, recall ≥ 0.75).

The harness is pure pytest — no DB, no network, no LLM — so it runs
in the standard unit-test suite and fails fast on regressions.

Diagnostic output
-----------------
On any gate failure pytest shows the aggregate numerator/denominator
plus the per-item parametrized failure, so reviewers see both the
"did we drift below the gate" signal AND the exact misclassified
golden items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from app.services.cjk_alias_resolver_service import METHOD_NONE, resolve_alias
from app.services.language_detection_service import detect_language
from app.services.multi_market_ticker_validator import normalize_extracted_ticker

from .multilingual_qa_corpus import (
    ALIAS_CORPUS,
    CORPUS_VERSION,
    LANGUAGE_CORPUS,
    NORMALIZER_CORPUS,
    AliasCase,
    LanguageCase,
    NormalizerCase,
)

# Launch-gate thresholds. Asymmetric by design — raising the precision
# floor above recall is what makes the "fail-closed into LLM fallback"
# policy actually safe (see module docstring on T7.4).
GATE_MIN_PRECISION: float = 0.85
GATE_MIN_RECALL: float = 0.75
GATE_MIN_ACCURACY: float = 0.85


# ---------------------------------------------------------------------------
# Metric primitives (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryMetrics:
    """Precision / recall / support for a binary classification task.

    Treats "resolver emitted a canonical symbol" as the positive class.
    - TP: resolver emitted canonical AND it matches the expected one
    - FP: resolver emitted canonical when expected was None, OR emitted
          the wrong canonical
    - FN: resolver emitted None when a canonical was expected
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 1.0

    @property
    def support(self) -> int:
        return (
            self.true_positive + self.false_positive
            + self.false_negative + self.true_negative
        )


def _score_prediction(
    predicted: Optional[str], expected: Optional[str],
) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) contribution for a single prediction."""
    predicted_positive = predicted is not None
    expected_positive = expected is not None
    if predicted_positive and expected_positive:
        return (1, 0, 0, 0) if predicted == expected else (0, 1, 0, 0)
    if predicted_positive and not expected_positive:
        return (0, 1, 0, 0)
    if not predicted_positive and expected_positive:
        return (0, 0, 1, 0)
    return (0, 0, 0, 1)


def _aggregate(
    predictions: list[tuple[Optional[str], Optional[str]]],
) -> BinaryMetrics:
    tp = fp = fn = tn = 0
    for predicted, expected in predictions:
        dtp, dfp, dfn, dtn = _score_prediction(predicted, expected)
        tp += dtp
        fp += dfp
        fn += dfn
        tn += dtn
    return BinaryMetrics(tp, fp, fn, tn)


# ---------------------------------------------------------------------------
# Corpus version pin
# ---------------------------------------------------------------------------


class TestCorpusContract:
    def test_corpus_version_is_pinned(self):
        # Any edit to a corpus entry requires bumping CORPUS_VERSION in
        # lockstep. Asserting a specific version here forces a reviewer
        # to touch this test when the corpus changes.
        assert CORPUS_VERSION == "2026.04.13.1"

    def test_corpora_are_non_trivial(self):
        # Defend against a corpus accidentally getting emptied.
        assert len(ALIAS_CORPUS) >= 20
        assert len(LANGUAGE_CORPUS) >= 15
        assert len(NORMALIZER_CORPUS) >= 10

    def test_corpora_include_adversarial_cases(self):
        # The bead explicitly calls for false-positive and ambiguous
        # cases — enforce their presence in every corpus.
        assert any(c.tag == "adversarial" for c in ALIAS_CORPUS)
        assert any(c.tag == "adversarial" for c in LANGUAGE_CORPUS)
        assert any(c.tag == "adversarial" for c in NORMALIZER_CORPUS)


# ---------------------------------------------------------------------------
# Alias resolver golden set
# ---------------------------------------------------------------------------


def _resolve_alias_for_corpus(case: AliasCase) -> Optional[str]:
    result = resolve_alias(case.query, hint_market=case.hint_market)
    if result.method == METHOD_NONE:
        return None
    return result.canonical_symbol


class TestAliasResolverGoldenSet:
    @pytest.mark.parametrize("case", ALIAS_CORPUS, ids=lambda c: f"{c.tag}:{c.query!r}")
    def test_individual_item(self, case: AliasCase):
        predicted = _resolve_alias_for_corpus(case)
        assert predicted == case.expected_canonical, (
            f"Golden item drift: {case!r} → predicted {predicted!r}, "
            f"expected {case.expected_canonical!r}"
        )

    def test_precision_meets_gate(self):
        predictions = [
            (_resolve_alias_for_corpus(c), c.expected_canonical)
            for c in ALIAS_CORPUS
        ]
        metrics = _aggregate(predictions)
        assert metrics.precision >= GATE_MIN_PRECISION, (
            f"Alias resolver precision {metrics.precision:.3f} below "
            f"gate {GATE_MIN_PRECISION} "
            f"(TP={metrics.true_positive}, FP={metrics.false_positive})"
        )

    def test_recall_meets_gate(self):
        predictions = [
            (_resolve_alias_for_corpus(c), c.expected_canonical)
            for c in ALIAS_CORPUS
        ]
        metrics = _aggregate(predictions)
        assert metrics.recall >= GATE_MIN_RECALL, (
            f"Alias resolver recall {metrics.recall:.3f} below "
            f"gate {GATE_MIN_RECALL} "
            f"(TP={metrics.true_positive}, FN={metrics.false_negative})"
        )


# ---------------------------------------------------------------------------
# Language detection golden set
# ---------------------------------------------------------------------------


class TestLanguageDetectionGoldenSet:
    @pytest.mark.parametrize("case", LANGUAGE_CORPUS, ids=lambda c: f"{c.tag}:{c.expected_language}")
    def test_individual_item(self, case: LanguageCase):
        predicted = detect_language(case.text)
        assert predicted == case.expected_language, (
            f"Language detection drift on {case.text!r}: "
            f"predicted {predicted!r}, expected {case.expected_language!r}"
        )

    def test_accuracy_meets_gate(self):
        correct = sum(
            1 for c in LANGUAGE_CORPUS
            if detect_language(c.text) == c.expected_language
        )
        total = len(LANGUAGE_CORPUS)
        accuracy = correct / total if total else 1.0
        assert accuracy >= GATE_MIN_ACCURACY, (
            f"Language detection accuracy {accuracy:.3f} ({correct}/{total}) "
            f"below gate {GATE_MIN_ACCURACY}"
        )


# ---------------------------------------------------------------------------
# Ticker normalizer golden set
# ---------------------------------------------------------------------------


def _normalize_for_corpus(case: NormalizerCase) -> Optional[str]:
    return normalize_extracted_ticker(case.raw).canonical


class TestTickerNormalizerGoldenSet:
    @pytest.mark.parametrize(
        "case", NORMALIZER_CORPUS, ids=lambda c: f"{c.tag}:{c.raw!r}",
    )
    def test_individual_item(self, case: NormalizerCase):
        predicted = _normalize_for_corpus(case)
        assert predicted == case.expected_canonical, (
            f"Normalizer drift on {case.raw!r}: "
            f"predicted {predicted!r}, expected {case.expected_canonical!r}"
        )

    def test_precision_meets_gate(self):
        predictions = [
            (_normalize_for_corpus(c), c.expected_canonical)
            for c in NORMALIZER_CORPUS
        ]
        metrics = _aggregate(predictions)
        assert metrics.precision >= GATE_MIN_PRECISION, (
            f"Normalizer precision {metrics.precision:.3f} below "
            f"gate {GATE_MIN_PRECISION} "
            f"(TP={metrics.true_positive}, FP={metrics.false_positive})"
        )

    def test_recall_meets_gate(self):
        predictions = [
            (_normalize_for_corpus(c), c.expected_canonical)
            for c in NORMALIZER_CORPUS
        ]
        metrics = _aggregate(predictions)
        assert metrics.recall >= GATE_MIN_RECALL, (
            f"Normalizer recall {metrics.recall:.3f} below "
            f"gate {GATE_MIN_RECALL} "
            f"(TP={metrics.true_positive}, FN={metrics.false_negative})"
        )


# ---------------------------------------------------------------------------
# Metric-primitive sanity tests (keeps the harness' own plumbing honest)
# ---------------------------------------------------------------------------


class TestMetricPrimitives:
    def test_all_correct_positives_give_perfect_precision(self):
        metrics = _aggregate([("A", "A"), ("B", "B")])
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_wrong_positive_counts_as_false_positive(self):
        # Returning the wrong canonical is as bad as guessing on a
        # None-expected case — both count against precision.
        metrics = _aggregate([("WRONG.HK", "0700.HK")])
        assert metrics.true_positive == 0
        assert metrics.false_positive == 1

    def test_missed_positive_counts_as_false_negative(self):
        metrics = _aggregate([(None, "0700.HK")])
        assert metrics.false_negative == 1
        assert metrics.recall == 0.0

    def test_correctly_rejected_is_true_negative(self):
        metrics = _aggregate([(None, None)])
        assert metrics.true_negative == 1
        # Precision is undefined on zero positives — we define it as 1.0
        # so a corpus of all-negatives doesn't spuriously fail the gate.
        assert metrics.precision == 1.0

    def test_hallucinated_positive_breaks_precision(self):
        metrics = _aggregate([("0700.HK", None)])
        assert metrics.false_positive == 1
        assert metrics.precision == 0.0
