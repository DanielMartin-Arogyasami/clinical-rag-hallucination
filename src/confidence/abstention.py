"""Abstention router — risk-proportionate thresholds for human review routing.

FIX H4: Fixed dead threshold logic. Now:
  - AUTO_ACCEPT: >= 0.90
  - HUMAN_REVIEW: >= threshold (task-dependent)
  - REJECT: >= 0.30 but below threshold (below threshold = probably wrong)
  - ABSTAIN: < 0.30 (too uncertain to even attempt)
"""
from __future__ import annotations

from enum import Enum

from src.config import AbstentionConfig
from src.confidence.estimator import ConfidenceScore


class ReviewDecision(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class AbstentionRouter:
    """Routes extractions to auto-accept, human review, rejection, or abstention.

    FIX H4: The threshold now actually controls the boundary between
    HUMAN_REVIEW and REJECT, making risk-proportionate routing functional.
    """

    def __init__(self, config: AbstentionConfig | None = None):
        self.config = config or AbstentionConfig()

    def route(self, score: ConfidenceScore, is_safety_critical: bool = False) -> ReviewDecision:
        threshold = (
            self.config.safety_critical_threshold
            if is_safety_critical
            else self.config.standard_threshold
        )

        composite = score.composite_score

        if composite >= 0.90:
            return ReviewDecision.AUTO_ACCEPT
        elif composite >= threshold:
            return ReviewDecision.HUMAN_REVIEW
        elif composite >= 0.30:
            # FIX H4: Below threshold but above noise floor → REJECT
            return ReviewDecision.REJECT
        else:
            return ReviewDecision.ABSTAIN

    def should_abstain(self, score: ConfidenceScore, is_safety_critical: bool = False) -> bool:
        decision = self.route(score, is_safety_critical)
        return decision in (ReviewDecision.ABSTAIN, ReviewDecision.REJECT)
