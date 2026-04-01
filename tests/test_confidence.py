"""Tests for Stage 5: Confidence estimation and abstention.

FIX H4: Added tests verifying the REJECT decision is now reachable.
"""
import pytest

from src.confidence.abstention import AbstentionRouter
from src.confidence.abstention import ReviewDecision
from src.confidence.estimator import ConfidenceEstimator
from src.confidence.estimator import ConfidenceScore
from src.confidence.estimator import ConfidenceSignals
from src.schemas.extraction import ConfidenceLevel


class TestConfidenceEstimator:
    def test_high_confidence(self):
        est = ConfidenceEstimator()
        score = est.estimate(
            ConfidenceSignals(evidence_support_score=0.95, self_consistency_ratio=1.0, token_entropy=0.3)
        )
        assert score.level == ConfidenceLevel.HIGH
        assert score.composite_score > 0.8

    def test_low_confidence(self):
        est = ConfidenceEstimator()
        score = est.estimate(
            ConfidenceSignals(evidence_support_score=0.2, self_consistency_ratio=0.3, token_entropy=4.0)
        )
        assert score.level in (ConfidenceLevel.LOW, ConfidenceLevel.UNABLE_TO_DETERMINE)

    def test_self_consistency(self):
        # Normalized: headache×4, nausea×1 → majority share 4/5
        assert ConfidenceEstimator.compute_self_consistency(
            ["Headache", "headache", "Headache", "Nausea", "headache"]
        ) == pytest.approx(0.8)


class TestAbstentionRouter:
    def test_auto_accept(self):
        router = AbstentionRouter()
        score = ConfidenceScore(composite_score=0.95, level=ConfidenceLevel.HIGH, signals=ConfidenceSignals())
        assert router.route(score) == ReviewDecision.AUTO_ACCEPT

    def test_human_review_above_threshold(self):
        router = AbstentionRouter()
        score = ConfidenceScore(composite_score=0.75, level=ConfidenceLevel.MEDIUM, signals=ConfidenceSignals())
        assert router.route(score, is_safety_critical=False) == ReviewDecision.HUMAN_REVIEW

    def test_reject_below_threshold(self):
        """FIX H4: Scores below threshold but above 0.30 now get REJECT."""
        router = AbstentionRouter()
        score = ConfidenceScore(composite_score=0.50, level=ConfidenceLevel.MEDIUM, signals=ConfidenceSignals())
        assert router.route(score, is_safety_critical=True) == ReviewDecision.REJECT

    def test_abstain_very_low(self):
        router = AbstentionRouter()
        score = ConfidenceScore(
            composite_score=0.15, level=ConfidenceLevel.UNABLE_TO_DETERMINE, signals=ConfidenceSignals()
        )
        assert router.route(score) == ReviewDecision.ABSTAIN

    def test_safety_critical_stricter(self):
        """FIX H4: Safety-critical threshold (0.80) is stricter than standard (0.60)."""
        router = AbstentionRouter()
        score = ConfidenceScore(composite_score=0.65, level=ConfidenceLevel.MEDIUM, signals=ConfidenceSignals())
        assert router.route(score, is_safety_critical=False) == ReviewDecision.HUMAN_REVIEW
        assert router.route(score, is_safety_critical=True) == ReviewDecision.REJECT
